import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


PREDICTION_HORIZONS_SECONDS: Tuple[int, ...] = (3, 5, 8)
PREDICTION_TARGETS: Tuple[str, ...] = (
    "vehicle_count",
    "queue_length",
    "avg_speed",
    "braking_probability",
)


def _trend_symbol(current: float, predicted: float, tolerance: float = 0.05) -> str:
    if predicted > current + tolerance:
        return "↑"
    if predicted < current - tolerance:
        return "↓"
    return "→"


class GRUPredictor(nn.Module):
    """Small GRU regressor for lane metrics forecasting."""

    def __init__(self, input_size: int = 4, hidden_size: int = 32, num_layers: int = 1, output_size: int = 4):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


class PredictiveMetricsLayer:
    """
    Forecasts lane metrics for future horizons.

    Uses a pretrained GRU if checkpoint exists; otherwise uses a robust extrapolation fallback.
    """

    def __init__(self):
        self.model = GRUPredictor()
        self.model_loaded = False
        self.checkpoint_path = os.getenv("TRAFFIC_PREDICTOR_CHECKPOINT", "SmarTSignalAI/models/gru_predictor.pt")
        self._try_load_checkpoint()

    def _try_load_checkpoint(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            state = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state)
            self.model.eval()
            self.model_loaded = True
        except Exception as exc:
            print(f"[WARNING] Could not load predictor checkpoint: {exc}")

    def _lane_series(self, frame_metrics: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
        lane_series: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in frame_metrics:
            lane_id = str(row.get("lane_id"))
            lane_series[lane_id].append(row)
        for lane_id in lane_series:
            lane_series[lane_id].sort(key=lambda x: float(x.get("timestamp", 0.0)))
        return lane_series

    def _heuristic_forecast(self, history: List[Dict[str, object]], horizon: int) -> Dict[str, float]:
        latest = history[-1]
        prev = history[-2] if len(history) > 1 else latest
        dt = max(float(latest.get("timestamp", 0.0)) - float(prev.get("timestamp", 0.0)), 1.0)
        step = horizon / dt

        def project(key: str, low: float = 0.0) -> float:
            cur = float(latest.get(key, 0.0))
            old = float(prev.get(key, cur))
            return max(low, cur + (cur - old) * step)

        projected = {
            "vehicle_count": round(project("vehicle_count", 0.0), 3),
            "queue_length": round(project("queue_length", 0.0), 3),
            "avg_speed": round(project("avg_speed", 0.0), 3),
            "braking_probability": round(min(max(project("braking_probability", 0.0), 0.0), 1.0), 4),
        }
        return projected

    def _gru_forecast(self, history: List[Dict[str, object]]) -> Dict[str, float]:
        features = []
        for row in history[-20:]:
            features.append(
                [
                    float(row.get("vehicle_count", 0.0)),
                    float(row.get("queue_length", 0.0)),
                    float(row.get("avg_speed", 0.0)),
                    float(row.get("braking_probability", 0.0)),
                ]
            )
        if not features:
            return {k: 0.0 for k in PREDICTION_TARGETS}
        x = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            pred = self.model(x).cpu().numpy()[0]
        return {
            "vehicle_count": round(max(0.0, float(pred[0])), 3),
            "queue_length": round(max(0.0, float(pred[1])), 3),
            "avg_speed": round(max(0.0, float(pred[2])), 3),
            "braking_probability": round(min(max(float(pred[3]), 0.0), 1.0), 4),
        }

    def predict(self, frame_metrics: List[Dict[str, object]]) -> Dict[str, object]:
        lanes = self._lane_series(frame_metrics)
        predicted_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        trend_direction: Dict[str, Dict[str, str]] = {}
        confidences = []

        for lane_id, history in lanes.items():
            if not history:
                continue
            current = history[-1]
            predicted_metrics[lane_id] = {}
            trend_direction[lane_id] = {}

            for horizon in PREDICTION_HORIZONS_SECONDS:
                if self.model_loaded and len(history) >= 5:
                    forecast = self._gru_forecast(history)
                    model_source = "gru"
                else:
                    forecast = self._heuristic_forecast(history, horizon)
                    model_source = "heuristic"

                key = f"t_plus_{horizon}s"
                predicted_metrics[lane_id][key] = forecast
                trend_direction[lane_id][key] = {
                    metric: _trend_symbol(float(current.get(metric, 0.0)), value)
                    for metric, value in forecast.items()
                }

                confidence = 0.75 if model_source == "gru" else 0.62
                if len(history) >= 10:
                    confidence += 0.08
                confidences.append(min(confidence, 0.95))

        return {
            "predicted_metrics": predicted_metrics,
            "prediction_confidence": round(sum(confidences) / max(1, len(confidences)), 4),
            "trend_direction": trend_direction,
            "prediction_model": "gru" if self.model_loaded else "heuristic",
            "prediction_horizons_seconds": list(PREDICTION_HORIZONS_SECONDS),
        }

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ContextMemory:
    previous_signal_phase: str = "UNKNOWN"
    time_since_last_green: float = 0.0
    previous_congestion_state: str = "UNKNOWN"


class CauseReasoningLayer:
    def analyze(
        self,
        current_metrics: Dict[str, object],
        predictive_metrics: Dict[str, object],
        context_memory: ContextMemory,
    ) -> Dict[str, object]:
        lane_summary = current_metrics.get("frameMetricsLaneSummary", {}) or {}
        predicted = current_metrics.get("predictiveMetrics", {}) or {}
        avg_speed = float(current_metrics.get("avgSpeed", 0.0) or 0.0)
        lane_density = current_metrics.get("laneDensity", {}) or {}

        peak_density = max(lane_density.values()) if lane_density else 0.0
        avg_exit = 0.0
        avg_turn = 0.0
        avg_queue = 0.0
        near_spacing = 999.0
        if lane_summary:
            values = list(lane_summary.values())
            count = max(1, len(values))
            avg_exit = sum(v.get("avgExitRate", 0.0) for v in values) / count
            avg_turn = sum(v.get("avgTurnRatio", 0.0) for v in values) / count
            avg_queue = sum(v.get("avgQueueLength", 0.0) for v in values) / count
            near_spacing = sum(v.get("avgNearVehicleDistance", 0.0) for v in values) / count

        predicted_queue_growth = False
        for lane_data in predicted.values():
            now_queue = avg_queue
            t3 = lane_data.get("t_plus_3s", {})
            if float(t3.get("queue_length", now_queue)) > now_queue + 0.5:
                predicted_queue_growth = True
                break

        detected: List[str] = []
        supporting = {
            "avgTurnRatio": round(avg_turn, 3),
            "avgExitRate": round(avg_exit, 3),
            "peakDensity": round(peak_density, 3),
            "avgSpeed": round(avg_speed, 3),
            "avgQueueLength": round(avg_queue, 3),
            "nearVehicleDistance": round(near_spacing, 3),
            "predictedQueueGrowth": predicted_queue_growth,
            "timeSinceLastGreen": round(context_memory.time_since_last_green, 3),
        }

        if avg_turn > 0.35 and avg_exit < 0.8:
            detected.append("TURN_BOTTLENECK")
        if peak_density > 0.45 and avg_speed < 15 and avg_exit < 0.8:
            detected.append("BLOCKAGE")
        if avg_speed < 12 and near_spacing < 2.5:
            detected.append("NEAR_MISS_RISK")
        if context_memory.time_since_last_green > 45 and predicted_queue_growth:
            detected.append("SIGNAL_STARVATION")

        primary = detected[0] if detected else "NO_CLEAR_BOTTLENECK"
        secondary = detected[1:] if len(detected) > 1 else []

        return {
            "primary_cause": primary,
            "secondary_causes": secondary,
            "supporting_metrics": supporting,
        }

    def update_context(self, context_memory: ContextMemory, duration_seconds: float, congestion_state: str) -> ContextMemory:
        context_memory.time_since_last_green += max(0.0, duration_seconds)
        context_memory.previous_congestion_state = congestion_state
        return context_memory

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional


@dataclass
class DecisionResult:
    traffic_state: str
    confidence: float
    recommendation: str
    reasoning: str


TRAFFIC_LABELS = [
    "free_flow",
    "moderate_flow",
    "heavy_congestion",
    "incident_risk",
]


def _build_traffic_prompt(stats: Dict[str, object]) -> str:
    lane_density = stats.get("laneDensity", {}) or {}
    lane_queue = stats.get("laneQueue", {}) or {}
    direction_dist = stats.get("directionDistribution", {}) or {}

    density_text = ", ".join(f"{k}:{v:.3f}" for k, v in lane_density.items()) if lane_density else "none"
    queue_text = ", ".join(f"{k}:{v:.3f}" for k, v in lane_queue.items()) if lane_queue else "none"
    direction_text = ", ".join(f"{k}:{v}" for k, v in direction_dist.items()) if direction_dist else "none"

    return (
        "Traffic analytics summary: "
        f"avg speed is {stats.get('avgSpeed', 0)} km/h, "
        f"flow rate is {stats.get('flowRatePerMin', 0)} vehicles/min, "
        f"stopped event ratio is {stats.get('stoppedEventRatio', 0)}, "
        f"unique vehicles {stats.get('uniqueVehicles', 0)}. "
        f"Lane density values: {density_text}. "
        f"Lane queue values: {queue_text}. "
        f"Direction distribution: {direction_text}."
    )


def _heuristic_decision(stats: Dict[str, object]) -> DecisionResult:
    avg_speed = float(stats.get("avgSpeed", 0) or 0)
    flow_rate = float(stats.get("flowRatePerMin", 0) or 0)
    stopped_ratio = float(stats.get("stoppedEventRatio", 0) or 0)
    lane_density = stats.get("laneDensity", {}) or {}
    lane_queue = stats.get("laneQueue", {}) or {}

    peak_density = max(lane_density.values()) if lane_density else 0.0
    peak_queue = max(lane_queue.values()) if lane_queue else 0.0

    if stopped_ratio > 0.45 or (peak_density > 0.6 and avg_speed < 10):
        return DecisionResult(
            traffic_state="incident_risk",
            confidence=0.68,
            recommendation="Trigger caution mode, prioritize emergency corridor checks, and dispatch operator review.",
            reasoning="High stop ratio or dense slow-moving traffic pattern indicates potential incident risk.",
        )

    if peak_density > 0.45 or peak_queue > 3.0 or avg_speed < 18:
        return DecisionResult(
            traffic_state="heavy_congestion",
            confidence=0.74,
            recommendation="Extend green time for overloaded lanes and apply adaptive phasing immediately.",
            reasoning="Lane occupancy/queue and speed suggest sustained congestion pressure.",
        )

    if flow_rate > 10 and avg_speed >= 18:
        return DecisionResult(
            traffic_state="moderate_flow",
            confidence=0.71,
            recommendation="Maintain adaptive timing and monitor queue growth for early intervention.",
            reasoning="Traffic is moving with moderate demand and manageable lane pressure.",
        )

    return DecisionResult(
        traffic_state="free_flow",
        confidence=0.7,
        recommendation="Keep baseline cycle plan with periodic monitoring.",
        reasoning="Low density and healthy speed indicate stable traffic conditions.",
    )


@lru_cache(maxsize=1)
def _load_classifier():
    model_name = os.getenv("TRAFFIC_DECISION_MODEL", "typeform/distilbert-base-uncased-mnli")
    try:
        from transformers import pipeline

        return pipeline("zero-shot-classification", model=model_name)
    except Exception as exc:
        print(f"[WARNING] Transformer decision model unavailable; using heuristic fallback: {exc}")
        return None


def _map_label_to_recommendation(label: str) -> str:
    recommendations = {
        "free_flow": "Keep default phase timing and monitor for sudden inflow spikes.",
        "moderate_flow": "Use adaptive split optimization to prevent queue buildup.",
        "heavy_congestion": "Increase green split for dense approaches and reduce phase dead-time.",
        "incident_risk": "Activate incident-aware control and request immediate operator verification.",
    }
    return recommendations.get(label, "Maintain adaptive monitoring.")


def classify_traffic_state(stats: Dict[str, object]) -> Dict[str, object]:
    classifier = _load_classifier()
    prompt = _build_traffic_prompt(stats)

    if classifier is None:
        heuristic = _heuristic_decision(stats)
        return {
            "trafficState": heuristic.traffic_state,
            "confidence": heuristic.confidence,
            "recommendation": heuristic.recommendation,
            "reasoning": heuristic.reasoning,
            "source": "heuristic_fallback",
        }

    try:
        output = classifier(prompt, TRAFFIC_LABELS, multi_label=False)
        label = output["labels"][0]
        score = float(output["scores"][0])

        reasoning = (
            f"Model classified the scene as '{label}' after analyzing speed, flow, queue and lane density signals."
        )
        return {
            "trafficState": label,
            "confidence": round(score, 4),
            "recommendation": _map_label_to_recommendation(label),
            "reasoning": reasoning,
            "source": "transformer_zero_shot",
        }
    except Exception as exc:
        print(f"[WARNING] Transformer inference failed; using heuristic fallback: {exc}")
        heuristic = _heuristic_decision(stats)
        return {
            "trafficState": heuristic.traffic_state,
            "confidence": heuristic.confidence,
            "recommendation": heuristic.recommendation,
            "reasoning": heuristic.reasoning,
            "source": "heuristic_fallback",
        }

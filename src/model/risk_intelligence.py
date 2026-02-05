from statistics import pstdev
from typing import Dict, List


class RiskSafetyLayer:
    def assess(self, current_metrics: Dict[str, object]) -> Dict[str, object]:
        sample = current_metrics.get("frameMetricsSample", []) or []
        lane_summary = current_metrics.get("frameMetricsLaneSummary", {}) or {}

        speeds: List[float] = [float(item.get("avg_speed", 0.0) or 0.0) for item in sample]
        speed_variance = pstdev(speeds) if len(speeds) > 1 else 0.0

        near_dists = [
            float(v.get("avgNearVehicleDistance", 0.0) or 0.0)
            for v in lane_summary.values()
            if v.get("avgNearVehicleDistance") is not None
        ]
        mean_near_dist = sum(near_dists) / max(1, len(near_dists))

        avg_turn_ratio = 0.0
        if lane_summary:
            avg_turn_ratio = sum(float(v.get("avgTurnRatio", 0.0) or 0.0) for v in lane_summary.values()) / len(lane_summary)

        avg_speed = float(current_metrics.get("avgSpeed", 0.0) or 0.0)
        relative_speed_mps = max(avg_speed / 3.6, 0.1)
        ttc = mean_near_dist / relative_speed_mps if relative_speed_mps > 0 else 99.0

        lateral_conflict_probability = min(max(avg_turn_ratio * 1.2, 0.0), 1.0)

        # Risk normalization to [0, 1]
        ttc_risk = min(max((3.0 - ttc) / 3.0, 0.0), 1.0)
        speed_var_risk = min(speed_variance / 15.0, 1.0)
        risk_score = round(min(max(0.45 * ttc_risk + 0.3 * speed_var_risk + 0.25 * lateral_conflict_probability, 0.0), 1.0), 4)

        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "ttc": round(ttc, 4),
            "speed_variance": round(speed_variance, 4),
            "lateral_conflict_probability": round(lateral_conflict_probability, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

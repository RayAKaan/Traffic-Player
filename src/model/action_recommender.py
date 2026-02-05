from typing import Dict


ALLOWED_ACTIONS = {
    "extend_green",
    "shorten_green",
    "activate_protected_turn",
    "hold_all_red",
    "prioritize_emergency_lane",
}


class ActionRecommendationLayer:
    def recommend(
        self,
        predictive_output: Dict[str, object],
        cause_output: Dict[str, object],
        risk_output: Dict[str, object],
    ) -> Dict[str, object]:
        predicted = predictive_output.get("predicted_metrics", {}) or {}
        primary_cause = cause_output.get("primary_cause", "NO_CLEAR_BOTTLENECK")
        risk_level = risk_output.get("risk_level", "LOW")

        target_lane = "lane_1"
        max_queue = -1.0
        queue_growth = False
        for lane_id, lane_pred in predicted.items():
            t3 = lane_pred.get("t_plus_3s", {})
            t8 = lane_pred.get("t_plus_8s", {})
            q3 = float(t3.get("queue_length", 0.0) or 0.0)
            q8 = float(t8.get("queue_length", q3) or q3)
            if q3 > max_queue:
                max_queue = q3
                target_lane = lane_id
            if q8 > q3 + 0.5:
                queue_growth = True

        if risk_level == "HIGH":
            action = "hold_all_red"
            reason = "High risk score requires immediate safety-first stabilization."
            expected = "Reduce conflict risk and near-miss probability in next cycle."
            confidence = 0.82
        elif primary_cause == "TURN_BOTTLENECK" and queue_growth:
            action = "activate_protected_turn"
            reason = "Turn bottleneck with projected queue growth detected."
            expected = "Increase turn clearance and reduce turn-lane queue buildup."
            confidence = 0.78
        elif primary_cause in {"BLOCKAGE", "SIGNAL_STARVATION"}:
            action = "extend_green"
            reason = "Blockage/starvation indicates insufficient throughput on overloaded approach."
            expected = f"Reduce queue on {target_lane} by increasing discharge time."
            confidence = 0.74
        elif risk_level == "MEDIUM":
            action = "prioritize_emergency_lane"
            reason = "Elevated risk and conflict probability warrant guarded priority handling."
            expected = "Lower conflict exposure while keeping flow partially adaptive."
            confidence = 0.68
        else:
            action = "shorten_green"
            reason = "Low risk and stable flow suggest reclaiming cycle efficiency."
            expected = "Improve cycle balance while maintaining current service levels."
            confidence = 0.63

        if action not in ALLOWED_ACTIONS:
            action = "extend_green"

        return {
            "recommended_action": action,
            "reason": reason,
            "expected_outcome": expected,
            "confidence": confidence,
            "target_lane": target_lane,
        }

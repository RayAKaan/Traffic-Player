from typing import Dict


class ExplanationEngine:
    def render(
        self,
        current_state: Dict[str, object],
        predicted_state: Dict[str, object],
        cause: Dict[str, object],
        risk: Dict[str, object],
        action: Dict[str, object],
        confidence: float,
    ) -> Dict[str, str]:
        explanation = (
            "CURRENT STATE:\n"
            f"avgSpeed={current_state.get('avgSpeed')}, flowRatePerMin={current_state.get('flowRatePerMin')}, "
            f"laneQueue={current_state.get('laneQueue')}\n"
            "PREDICTED STATE:\n"
            f"{predicted_state}\n"
            "CAUSE:\n"
            f"primary={cause.get('primary_cause')}, secondary={cause.get('secondary_causes')}\n"
            "RISK:\n"
            f"score={risk.get('risk_score')}, level={risk.get('risk_level')}, ttc={risk.get('ttc')}\n"
            "ACTION:\n"
            f"{action.get('recommended_action')} on {action.get('target_lane')} because {action.get('reason')}\n"
            "EXPECTED EFFECT:\n"
            f"{action.get('expected_outcome')}\n"
            "CONFIDENCE:\n"
            f"{round(confidence, 4)}"
        )
        return {"explanation": explanation}

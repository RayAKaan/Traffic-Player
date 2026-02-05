import json
import os
from typing import Dict, List


class ClosedLoopLearningLayer:
    def __init__(self, memory_path: str = "SmarTSignalAI/data/learning_memory.json"):
        self.memory_path = memory_path

    def _load(self) -> List[Dict[str, object]]:
        if not os.path.exists(self.memory_path):
            return []
        try:
            with open(self.memory_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return []

    def _save(self, rows: List[Dict[str, object]]) -> None:
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as fh:
            json.dump(rows[-500:], fh)

    def evaluate_and_store(
        self,
        action: Dict[str, object],
        expected_outcome: str,
        current_metrics: Dict[str, object],
    ) -> Dict[str, object]:
        # Placeholder online evaluation: action executed externally; we mark pending and log baseline context.
        row = {
            "action": action.get("recommended_action"),
            "target_lane": action.get("target_lane"),
            "context": {
                "avgSpeed": current_metrics.get("avgSpeed"),
                "laneQueue": current_metrics.get("laneQueue"),
                "riskScore": current_metrics.get("risk", {}).get("risk_score"),
            },
            "expected_outcome": expected_outcome,
            "actual_queue_change": None,
            "actual_speed_change": None,
            "actual_risk_change": None,
            "effectiveness": None,
            "outcome_quality": "PENDING",
        }
        rows = self._load()
        rows.append(row)
        self._save(rows)
        return row

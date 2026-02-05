from pathlib import Path

from src.model.closed_loop_learning import ClosedLoopLearningLayer


def test_closed_loop_stores_pending_outcome(tmp_path: Path):
    memory_file = tmp_path / "memory.json"
    layer = ClosedLoopLearningLayer(memory_path=str(memory_file))

    row = layer.evaluate_and_store(
        action={"recommended_action": "extend_green", "target_lane": "lane_1"},
        expected_outcome="Reduce queue",
        current_metrics={"avgSpeed": 20, "laneQueue": {"lane_1": 2}, "risk": {"risk_score": 0.2}},
    )

    assert row["outcome_quality"] == "PENDING"
    assert memory_file.exists()

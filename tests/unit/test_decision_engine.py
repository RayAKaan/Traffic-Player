from src.model import decision_engine


def test_heuristic_congestion_decision():
    stats = {
        "avgSpeed": 12,
        "flowRatePerMin": 35,
        "stoppedEventRatio": 0.25,
        "uniqueVehicles": 60,
        "laneDensity": {"lane_1": 0.7, "lane_2": 0.5},
        "laneQueue": {"lane_1": 5.0, "lane_2": 3.0},
        "directionDistribution": {"N": 3, "S": 2},
    }

    decision_engine._load_classifier.cache_clear()
    original_loader = decision_engine._load_classifier
    decision_engine._load_classifier = lambda: None
    try:
        result = decision_engine.classify_traffic_state(stats)
    finally:
        decision_engine._load_classifier = original_loader

    assert result["source"] == "heuristic_fallback"
    assert result["trafficState"] in {"heavy_congestion", "incident_risk"}
    assert "recommendation" in result


def test_prompt_contains_key_metrics():
    stats = {
        "avgSpeed": 20,
        "flowRatePerMin": 12,
        "stoppedEventRatio": 0.1,
        "uniqueVehicles": 15,
        "laneDensity": {"lane_1": 0.2},
        "laneQueue": {"lane_1": 1.0},
        "directionDistribution": {"E": 8},
    }
    prompt = decision_engine._build_traffic_prompt(stats)
    assert "avg speed" in prompt
    assert "Lane density values" in prompt
    assert "Direction distribution" in prompt

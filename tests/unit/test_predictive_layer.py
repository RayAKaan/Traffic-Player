from src.model.predictive_layer import PREDICTION_HORIZONS_SECONDS, PredictiveMetricsLayer


def _frame(timestamp: float, lane_id: str, vehicle_count: float, queue_length: float, avg_speed: float, braking: float):
    return {
        "timestamp": timestamp,
        "lane_id": lane_id,
        "vehicle_count": vehicle_count,
        "vehicle_classes": {"car": int(vehicle_count)},
        "avg_speed": avg_speed,
        "avg_acceleration": 0.0,
        "queue_length": queue_length,
        "entry_rate": 1.0,
        "exit_rate": 0.5,
        "turn_ratio": 0.2,
        "near_vehicle_distance": 3.0,
        "braking_probability": braking,
    }


def test_predictive_contract_with_heuristic_fallback():
    layer = PredictiveMetricsLayer()
    layer.model_loaded = False

    frames = [
        _frame(1.0, "lane_1", 5, 2, 30, 0.1),
        _frame(2.0, "lane_1", 6, 2, 28, 0.15),
        _frame(3.0, "lane_1", 7, 3, 26, 0.2),
        _frame(4.0, "lane_1", 8, 4, 24, 0.25),
    ]

    result = layer.predict(frames)

    assert "predicted_metrics" in result
    assert "prediction_confidence" in result
    assert "trend_direction" in result
    assert result["prediction_model"] == "heuristic"

    lane_preds = result["predicted_metrics"]["lane_1"]
    for horizon in PREDICTION_HORIZONS_SECONDS:
        key = f"t_plus_{horizon}s"
        assert key in lane_preds
        assert set(lane_preds[key].keys()) == {
            "vehicle_count",
            "queue_length",
            "avg_speed",
            "braking_probability",
        }

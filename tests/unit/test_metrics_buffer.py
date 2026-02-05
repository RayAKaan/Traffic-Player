from src.model.metrics_buffer import FrameLaneMetrics, SlidingMetricsBuffer


def _record(t: float, lane: str, count: int, speed: float, queue: int):
    return FrameLaneMetrics(
        timestamp=t,
        lane_id=lane,
        vehicle_count=count,
        vehicle_classes={"car": count},
        avg_speed=speed,
        avg_acceleration=0.5,
        queue_length=queue,
        entry_rate=1.0,
        exit_rate=0.5,
        turn_ratio=0.2,
        near_vehicle_distance=3.0,
        braking_probability=0.2,
    )


def test_sliding_metrics_buffer_trims_by_timestamp_window():
    buffer = SlidingMetricsBuffer(window_seconds=10)
    for t in [0.0, 4.0, 9.0, 12.0]:
        buffer.add(_record(t, "lane_1", 2, 15.0, 1))

    assert len(buffer) == 3
    timestamps = [item["timestamp"] for item in buffer.to_list()]
    assert timestamps == [4.0, 9.0, 12.0]


def test_lane_summary_aggregates_metrics():
    buffer = SlidingMetricsBuffer(window_seconds=10)
    buffer.add(_record(1.0, "lane_1", 2, 10.0, 1))
    buffer.add(_record(2.0, "lane_1", 4, 20.0, 3))

    summary = buffer.lane_summary()
    lane = summary["lane_1"]
    assert lane["avgVehicleCount"] == 3.0
    assert lane["avgSpeed"] == 15.0
    assert lane["avgQueueLength"] == 2.0
    assert lane["avgBrakingProbability"] == 0.2

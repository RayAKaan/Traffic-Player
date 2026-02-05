from collections import deque

from src.model.traffic_metrics import (
    bbox_area,
    bbox_center,
    build_lane_rois,
    compute_track_speed_kmh,
    infer_direction,
    intersection_area,
    smooth_recent,
)


def test_build_lane_rois_splits_width_evenly():
    rois = build_lane_rois(width=100, height=40, lane_count=4)
    assert rois["lane_1"] == (0, 0, 25, 40)
    assert rois["lane_4"] == (75, 0, 100, 40)


def test_geometry_helpers():
    box = (10, 10, 30, 20)
    assert bbox_center(box) == (20.0, 15.0)
    assert bbox_area(box) == 200.0
    assert intersection_area(box, (20, 0, 40, 15)) == 50.0


def test_direction_and_speed_helpers():
    assert infer_direction((0.0, 0.0), (10.0, 2.0)) == "E"
    assert infer_direction((5.0, 10.0), (4.0, 0.0)) == "N"

    speed = compute_track_speed_kmh((0.0, 0.0), (10.0, 0.0), fps=10.0, pixels_to_meters=0.1)
    assert round(speed, 2) == 36.0


def test_smooth_recent_average():
    values = deque([2.0, 4.0, 6.0], maxlen=3)
    assert smooth_recent(values) == 4.0

from statistics import mean
from typing import Deque, Dict, Tuple


def build_lane_rois(width: int, height: int, lane_count: int) -> Dict[str, Tuple[int, int, int, int]]:
    lane_count = max(1, lane_count)
    lane_width = width // lane_count
    rois: Dict[str, Tuple[int, int, int, int]] = {}
    for idx in range(lane_count):
        x1 = idx * lane_width
        x2 = width if idx == lane_count - 1 else (idx + 1) * lane_width
        rois[f"lane_{idx + 1}"] = (x1, 0, x2, height)
    return rois


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((x2 - x1) * (y2 - y1))


def infer_direction(prev: Tuple[float, float], curr: Tuple[float, float]) -> str:
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    if abs(dx) > abs(dy):
        return "E" if dx > 0 else "W"
    return "S" if dy > 0 else "N"


def compute_track_speed_kmh(
    prev: Tuple[float, float], curr: Tuple[float, float], fps: float, pixels_to_meters: float
) -> float:
    pixel_dist = ((curr[0] - prev[0]) ** 2 + (curr[1] - prev[1]) ** 2) ** 0.5
    meters_per_sec = pixel_dist * pixels_to_meters * fps
    return meters_per_sec * 3.6


def smooth_recent(values: Deque[float]) -> float:
    return round(mean(values), 3) if values else 0.0

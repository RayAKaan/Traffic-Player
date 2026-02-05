from dataclasses import asdict, dataclass
from typing import Deque, Dict, List, Optional
from collections import deque


@dataclass
class FrameLaneMetrics:
    timestamp: float
    lane_id: str
    vehicle_count: int
    vehicle_classes: Dict[str, int]
    avg_speed: float
    avg_acceleration: float
    queue_length: int
    entry_rate: float
    exit_rate: float
    turn_ratio: float
    near_vehicle_distance: Optional[float]
    braking_probability: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SlidingMetricsBuffer:
    """Timestamp-based rolling buffer for frame-level lane metrics."""

    def __init__(self, window_seconds: float = 20.0):
        self.window_seconds = max(1.0, float(window_seconds))
        self._records: Deque[FrameLaneMetrics] = deque()

    def add(self, record: FrameLaneMetrics) -> None:
        self._records.append(record)
        self._trim(record.timestamp)

    def _trim(self, current_timestamp: float) -> None:
        cutoff = current_timestamp - self.window_seconds
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    def __len__(self) -> int:
        return len(self._records)

    def to_list(self) -> List[Dict[str, object]]:
        return [record.to_dict() for record in self._records]

    def lane_summary(self) -> Dict[str, Dict[str, float]]:
        grouped: Dict[str, List[FrameLaneMetrics]] = {}
        for record in self._records:
            grouped.setdefault(record.lane_id, []).append(record)

        summary: Dict[str, Dict[str, float]] = {}
        for lane_id, records in grouped.items():
            count = max(1, len(records))
            summary[lane_id] = {
                "avgVehicleCount": round(sum(r.vehicle_count for r in records) / count, 3),
                "avgSpeed": round(sum(r.avg_speed for r in records) / count, 3),
                "avgAcceleration": round(sum(r.avg_acceleration for r in records) / count, 3),
                "avgQueueLength": round(sum(r.queue_length for r in records) / count, 3),
                "avgEntryRate": round(sum(r.entry_rate for r in records) / count, 3),
                "avgExitRate": round(sum(r.exit_rate for r in records) / count, 3),
                "avgTurnRatio": round(sum(r.turn_ratio for r in records) / count, 3),
                "avgNearVehicleDistance": round(
                    sum((r.near_vehicle_distance or 0.0) for r in records) / count,
                    3,
                ),
                "avgBrakingProbability": round(sum(r.braking_probability for r in records) / count, 4),
            }
        return summary

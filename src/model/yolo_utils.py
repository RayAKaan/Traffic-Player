# src/model/yolo_utils.py

from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

# Load YOLOv8 model
try:
    model = YOLO("yolov8x.pt")  # Swap to yolov8n.pt or yolov8s.pt for speed on CPU
except Exception as e:
    raise RuntimeError(f"Failed to load YOLO model: {e}")

# Define target vehicle classes
TRACKED_CLASSES = {"car", "bus", "truck", "motorbike", "bicycle"}


def get_color_by_class(label: str) -> Tuple[int, int, int]:
    color_map = {
        "car": (0, 255, 0),
        "bus": (255, 0, 0),
        "truck": (0, 0, 255),
        "motorbike": (255, 255, 0),
        "bicycle": (0, 255, 255),
    }
    return color_map.get(label, (255, 255, 255))


def detect_and_track_yolo(
    frame: cv2.Mat,
    enhanced: bool = False,
    allowed_classes: Optional[set] = TRACKED_CLASSES,
) -> Tuple[cv2.Mat, List[Dict[str, object]], Dict[str, int]]:
    """
    Run YOLO detection + ByteTrack tracking for a single frame.

    Returns:
    - annotated frame
    - list of detection dicts: {track_id, label, conf, bbox}
    - per-frame stats by class
    """
    results = model.track(frame, verbose=False, persist=True, tracker="bytetrack.yaml")
    detections: List[Dict[str, object]] = []
    stats = {cls: 0 for cls in allowed_classes}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            if label not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            track_id = int(box.id[0]) if box.id is not None else None

            detections.append(
                {
                    "track_id": track_id,
                    "label": label,
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                }
            )
            stats[label] += 1

            if enhanced:
                color = get_color_by_class(label)
                display_id = f" ID:{track_id}" if track_id is not None else ""
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label}{display_id} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

    return frame, detections, stats


def detect_objects_yolo(
    frame: cv2.Mat,
    enhanced: bool = False,
    allowed_classes: Optional[set] = TRACKED_CLASSES,
) -> Tuple[cv2.Mat, List[str], Dict[str, int]]:
    """Backward-compatible detection-only wrapper."""
    tracked_frame, detections, stats = detect_and_track_yolo(
        frame, enhanced=enhanced, allowed_classes=allowed_classes
    )
    labels = [d["label"] for d in detections]
    return tracked_frame, labels, stats

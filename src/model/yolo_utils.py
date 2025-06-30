# src/model/yolo_utils.py

from ultralytics import YOLO
import cv2
from typing import Tuple, List, Dict, Optional

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

def detect_objects_yolo(
    frame: cv2.Mat,
    enhanced: bool = False,
    allowed_classes: Optional[set] = TRACKED_CLASSES
) -> Tuple[cv2.Mat, List[str], Dict[str, int]]:
    results = model(frame, verbose=False)
    detections = []
    stats = {cls: 0 for cls in allowed_classes}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            detections.append(label)
            if label in allowed_classes:
                stats[label] += 1

            if enhanced:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), get_color_by_class(label), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_color_by_class(label), 1)

    return frame, detections, stats

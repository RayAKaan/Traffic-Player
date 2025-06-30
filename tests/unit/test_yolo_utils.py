# tests/unit/test_yolo_utils.py

import cv2
from src.model.yolo_utils import detect_objects_yolo

def test_detect_objects_on_sample():
    img = cv2.imread("tests/samples/sample_frame.jpg")
    assert img is not None, "Test image not found."

    frame, detections, stats = detect_objects_yolo(img, enhanced=False)

    assert isinstance(detections, list)
    assert isinstance(stats, dict)
    assert all(isinstance(v, int) for v in stats.values())

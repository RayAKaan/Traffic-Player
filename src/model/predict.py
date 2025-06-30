# src/model/predict.py

import cv2
import uuid
import os
import subprocess
from typing import Dict, Tuple
from src.model.yolo_utils import detect_objects_yolo

def process_video_with_model(
    input_path: str,
    output_dir: str = "SmarTSignalAI/data/processed",
    progress_store: Dict[str, dict] = {},
    task_id: str = "",
    enhanced: bool = False
) -> Tuple[str, dict]:
    """
    Processes a video using YOLOv8 detection, tracks progress, and saves annotated output.

    Args:
        input_path (str): Path to the input video.
        output_dir (str): Directory to save the processed video.
        progress_store (Dict): Dictionary to update processing progress.
        task_id (str): Task identifier to track progress.
        enhanced (bool): If True, draw bounding boxes and labels on video frames.

    Returns:
        Tuple[str, dict]: Final processed video path and aggregated detection stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    raw_filename = f"{uuid.uuid4().hex}_raw.avi"
    raw_path = os.path.join(output_dir, raw_filename)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_idx = 0
    stats = {"car": 0, "bus": 0, "truck": 0, "motorbike": 0, "bicycle": 0, "avgSpeed": 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            detected_frame, _, frame_stats = detect_objects_yolo(frame, enhanced=enhanced)
        except Exception as e:
            print(f"[ERROR] YOLO inference failed on frame {frame_idx}: {e}")
            continue

        out.write(detected_frame)

        for key in frame_stats:
            if key in stats:
                stats[key] += frame_stats[key]

        frame_idx += 1
        if total_frames:
            progress = int((frame_idx / total_frames) * 100)
            progress_store[task_id]["progress"] = min(progress, 99)

    cap.release()
    out.release()

    final_filename = raw_filename.replace("_raw.avi", "_processed.mp4")
    final_path = os.path.join(output_dir, final_filename)

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264", "-preset", "ultrafast", final_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        os.remove(raw_path)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Video conversion failed: {e}")

    stats["avgSpeed"] = round(20 + (5 if enhanced else 0), 1)  # Simulated logic for now

    return final_path, stats

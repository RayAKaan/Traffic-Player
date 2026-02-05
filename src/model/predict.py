# src/model/predict.py
import os
import subprocess
import uuid
from collections import defaultdict, deque
from math import dist
from typing import Deque, Dict, Optional, Tuple

import cv2
from sqlmodel import select

from app.database import VideoTask, get_session
from src.model.decision_engine import classify_traffic_state
from src.model.risk_intelligence import RiskSafetyLayer
from src.model.explanation_engine import ExplanationEngine
from src.model.closed_loop_learning import ClosedLoopLearningLayer
from src.model.cause_reasoning import CauseReasoningLayer, ContextMemory
from src.model.action_recommender import ActionRecommendationLayer
from src.model.metrics_buffer import FrameLaneMetrics, SlidingMetricsBuffer
from src.model.predictive_layer import PredictiveMetricsLayer
from src.model.traffic_metrics import (
    bbox_area,
    bbox_center,
    build_lane_rois,
    compute_track_speed_kmh,
    infer_direction,
    intersection_area,
    smooth_recent,
)
from src.model.yolo_utils import TRACKED_CLASSES, detect_and_track_yolo

PIXELS_TO_METERS = float(os.getenv("PIXELS_TO_METERS", "0.05"))
LANE_COUNT = int(os.getenv("LANE_COUNT", "4"))
SMOOTHING_WINDOW = int(os.getenv("SMOOTHING_WINDOW", "30"))
METRICS_WINDOW_SECONDS = float(os.getenv("METRICS_WINDOW_SECONDS", "20"))


def update_task_progress(task_id: str, progress: int, status: Optional[str] = None):
    """Safely update task progress in the database."""
    try:
        with get_session() as session:
            task = session.exec(select(VideoTask).where(VideoTask.id == task_id)).first()
            if task:
                task.progress = progress
                if status:
                    task.status = status
                session.add(task)
                session.commit()
    except Exception as e:
        print(f"[WARNING] Could not update progress for {task_id}: {e}")



def process_video_with_model(
    input_path: str,
    output_dir: str = "SmarTSignalAI/data/processed",
    task_id: str = "",
    enhanced: bool = False,
) -> Tuple[str, dict]:
    """
    Processes a video using YOLOv8 + ByteTrack and saves annotated output.
    Computes richer traffic metrics and updates task progress in DB.
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Starting video processing: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"[ERROR] Cannot open video file: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width == 0 or height == 0:
        cap.release()
        raise ValueError(f"[ERROR] Invalid video dimensions (width={width}, height={height})")

    lane_rois = build_lane_rois(width, height, LANE_COUNT)
    lane_areas = {lane: bbox_area(roi) for lane, roi in lane_rois.items()}

    raw_filename = f"{uuid.uuid4().hex}_raw.avi"
    raw_path = os.path.join(output_dir, raw_filename)
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

    frame_idx = 0
    class_totals = {cls: 0 for cls in TRACKED_CLASSES}
    track_history: Dict[int, Tuple[float, float]] = {}
    unique_track_ids = set()
    direction_counter = defaultdict(int)

    rolling_speed_kmh: Deque[float] = deque(maxlen=SMOOTHING_WINDOW)
    rolling_density: Dict[str, Deque[float]] = {lane: deque(maxlen=SMOOTHING_WINDOW) for lane in lane_rois}
    rolling_queue: Dict[str, Deque[float]] = {lane: deque(maxlen=SMOOTHING_WINDOW) for lane in lane_rois}

    flow_events = 0
    stopped_events = 0
    metrics_buffer = SlidingMetricsBuffer(window_seconds=METRICS_WINDOW_SECONDS)
    predictive_layer = PredictiveMetricsLayer()
    cause_layer = CauseReasoningLayer()
    risk_layer = RiskSafetyLayer()
    action_layer = ActionRecommendationLayer()
    explanation_engine = ExplanationEngine()
    learning_layer = ClosedLoopLearningLayer()
    context_memory = ContextMemory()
    track_last_speed: Dict[int, float] = {}
    track_last_lane: Dict[int, str] = {}
    previous_active_tracks = set()

    update_task_progress(task_id, 0, "processing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            detected_frame, tracked_detections, frame_stats = detect_and_track_yolo(frame, enhanced=enhanced)
        except Exception as e:
            print(f"[WARNING] YOLO tracking failed on frame {frame_idx}: {e}")
            detected_frame = frame
            tracked_detections = []
            frame_stats = {cls: 0 for cls in TRACKED_CLASSES}

        for cls in class_totals:
            class_totals[cls] += frame_stats.get(cls, 0)

        lane_density_area = {lane: 0.0 for lane in lane_rois}
        lane_queue_count = {lane: 0 for lane in lane_rois}
        lane_class_counts = {lane: {cls: 0 for cls in TRACKED_CLASSES} for lane in lane_rois}
        lane_speed_values = {lane: [] for lane in lane_rois}
        lane_acc_values = {lane: [] for lane in lane_rois}
        lane_direction_values = {lane: [] for lane in lane_rois}
        lane_centers = {lane: [] for lane in lane_rois}
        lane_entry_events = {lane: 0 for lane in lane_rois}
        lane_exit_events = {lane: 0 for lane in lane_rois}
        lane_brake_events = {lane: 0 for lane in lane_rois}
        lane_motion_events = {lane: 0 for lane in lane_rois}
        active_tracks = set()

        for detection in tracked_detections:
            track_id = detection.get("track_id")
            bbox = detection["bbox"]
            label = detection["label"]
            center = bbox_center(bbox)
            prev_center = track_history.get(track_id) if track_id is not None else None
            current_speed = 0.0
            current_acceleration = 0.0
            current_direction = None
            dominant_lane = None
            best_overlap = 0.0

            if track_id is not None:
                unique_track_ids.add(track_id)
                if prev_center is None:
                    flow_events += 1
                else:
                    current_speed = compute_track_speed_kmh(prev_center, center, fps, PIXELS_TO_METERS)
                    rolling_speed_kmh.append(current_speed)
                    current_direction = infer_direction(prev_center, center)
                    direction_counter[current_direction] += 1
                    if current_speed < 3.0:
                        stopped_events += 1
                    prev_speed = track_last_speed.get(track_id, current_speed)
                    current_acceleration = (current_speed - prev_speed) / max(1.0 / fps, 1e-6)
                    track_last_speed[track_id] = current_speed
                if prev_center is None:
                    track_last_speed[track_id] = 0.0

                track_history[track_id] = center
                active_tracks.add(track_id)

            for lane, roi in lane_rois.items():
                overlap = intersection_area(bbox, roi)
                if overlap <= 0:
                    continue
                if overlap > best_overlap:
                    best_overlap = overlap
                    dominant_lane = lane
                lane_density_area[lane] += overlap
                if prev_center is not None:
                    speed_for_queue = compute_track_speed_kmh(prev_center, center, fps, PIXELS_TO_METERS)
                    if speed_for_queue < 3.0:
                        lane_queue_count[lane] += 1

            if dominant_lane is not None:
                lane_class_counts[dominant_lane][label] += 1
                lane_centers[dominant_lane].append(center)
                if prev_center is not None:
                    lane_speed_values[dominant_lane].append(current_speed)
                    lane_acc_values[dominant_lane].append(current_acceleration)
                    lane_motion_events[dominant_lane] += 1
                    if current_acceleration < -2.0:
                        lane_brake_events[dominant_lane] += 1
                    if current_direction is not None:
                        lane_direction_values[dominant_lane].append(current_direction)
                if track_id is not None and prev_center is None:
                    lane_entry_events[dominant_lane] += 1
                if track_id is not None:
                    track_last_lane[track_id] = dominant_lane

        exited_tracks = previous_active_tracks - active_tracks
        for track_id in exited_tracks:
            lane = track_last_lane.get(track_id)
            if lane in lane_exit_events:
                lane_exit_events[lane] += 1
        previous_active_tracks = active_tracks

        timestamp = frame_idx / fps if fps else float(frame_idx)
        for lane in lane_rois:
            centers = lane_centers[lane]
            nearest_distances = []
            for idx, center in enumerate(centers):
                others = [dist(center, candidate) for j, candidate in enumerate(centers) if j != idx]
                if others:
                    nearest_distances.append(min(others) * PIXELS_TO_METERS)

            directions = lane_direction_values[lane]
            turn_count = sum(1 for d in directions if d in {"E", "W"})
            turn_ratio = (turn_count / len(directions)) if directions else 0.0
            frame_metric = FrameLaneMetrics(
                timestamp=round(timestamp, 3),
                lane_id=lane,
                vehicle_count=sum(lane_class_counts[lane].values()),
                vehicle_classes=lane_class_counts[lane],
                avg_speed=round(sum(lane_speed_values[lane]) / max(1, len(lane_speed_values[lane])), 3),
                avg_acceleration=round(sum(lane_acc_values[lane]) / max(1, len(lane_acc_values[lane])), 3),
                queue_length=lane_queue_count[lane],
                entry_rate=round(lane_entry_events[lane], 3),
                exit_rate=round(lane_exit_events[lane], 3),
                turn_ratio=round(turn_ratio, 3),
                near_vehicle_distance=(
                    round(sum(nearest_distances) / len(nearest_distances), 3) if nearest_distances else None
                ),
                braking_probability=round(
                    lane_brake_events[lane] / max(1, lane_motion_events[lane]),
                    4,
                ),
            )
            metrics_buffer.add(frame_metric)

        for lane in lane_rois:
            density = min(1.0, lane_density_area[lane] / lane_areas[lane]) if lane_areas[lane] else 0.0
            rolling_density[lane].append(density)
            rolling_queue[lane].append(float(lane_queue_count[lane]))

        if enhanced:
            for lane_name, (x1, y1, x2, y2) in lane_rois.items():
                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(
                    detected_frame,
                    lane_name,
                    (x1 + 10, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

        out.write(detected_frame)

        frame_idx += 1
        if total_frames and frame_idx % 5 == 0:
            progress = min(int((frame_idx / total_frames) * 100), 99)
            update_task_progress(task_id, progress)

    cap.release()
    out.release()

    # Convert to MP4
    final_filename = raw_filename.replace("_raw.avi", "_processed.mp4")
    final_path = os.path.join(output_dir, final_filename)

    try:
        print(f"[INFO] Converting video to MP4: {final_path}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-c:v", "libx264", "-preset", "ultrafast", final_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(raw_path)
    except Exception as e:
        print(f"[ERROR] FFmpeg conversion failed, keeping AVI: {e}")
        final_path = raw_path

    duration_seconds = frame_idx / fps if fps else 0.0
    avg_speed_kmh = smooth_recent(rolling_speed_kmh)
    flow_per_minute = (flow_events / duration_seconds * 60.0) if duration_seconds > 0 else 0.0

    lane_density_smoothed = {lane: smooth_recent(values) for lane, values in rolling_density.items()}
    lane_queue_smoothed = {lane: smooth_recent(values) for lane, values in rolling_queue.items()}

    predictive_output = predictive_layer.predict(metrics_buffer.to_list())

    phase_current = {
        "avgSpeed": round(avg_speed_kmh, 2),
        "flowRatePerMin": round(flow_per_minute, 2),
        "laneQueue": lane_queue_smoothed,
        "laneDensity": lane_density_smoothed,
        "frameMetricsLaneSummary": metrics_buffer.lane_summary(),
        "predictiveMetrics": predictive_output["predicted_metrics"],
    }
    cause_output = cause_layer.analyze(phase_current, predictive_output, context_memory)
    risk_output = risk_layer.assess({**phase_current, "frameMetricsSample": metrics_buffer.to_list()[-min(20, len(metrics_buffer)): ]})
    action_output = action_layer.recommend(predictive_output, cause_output, risk_output)
    explanation_output = explanation_engine.render(
        current_state=phase_current,
        predicted_state=predictive_output["predicted_metrics"],
        cause=cause_output,
        risk=risk_output,
        action=action_output,
        confidence=(predictive_output["prediction_confidence"] + action_output["confidence"]) / 2.0,
    )
    context_memory = cause_layer.update_context(
        context_memory=context_memory,
        duration_seconds=duration_seconds,
        congestion_state=classify_traffic_state({"avgSpeed": avg_speed_kmh, "flowRatePerMin": flow_per_minute}).get("trafficState", "UNKNOWN"),
    )
    learning_record = learning_layer.evaluate_and_store(
        action=action_output,
        expected_outcome=action_output["expected_outcome"],
        current_metrics={"avgSpeed": avg_speed_kmh, "laneQueue": lane_queue_smoothed, "risk": risk_output},
    )

    stats = {
        **class_totals,
        "uniqueVehicles": len(unique_track_ids),
        "avgSpeed": round(avg_speed_kmh, 2),
        "flowRatePerMin": round(flow_per_minute, 2),
        "stoppedEventRatio": round(stopped_events / max(1, flow_events), 3),
        "directionDistribution": dict(direction_counter),
        "laneDensity": lane_density_smoothed,
        "laneQueue": lane_queue_smoothed,
        "durationSeconds": round(duration_seconds, 2),
        "metricsBufferWindowSeconds": METRICS_WINDOW_SECONDS,
        "frameMetricsBufferSize": len(metrics_buffer),
        "frameMetricsLaneSummary": metrics_buffer.lane_summary(),
        "frameMetricsSample": metrics_buffer.to_list()[-min(20, len(metrics_buffer)) :],
        "predictiveMetrics": predictive_output["predicted_metrics"],
        "predictionConfidence": predictive_output["prediction_confidence"],
        "trendDirection": predictive_output["trend_direction"],
        "predictionHorizonsSeconds": predictive_output["prediction_horizons_seconds"],
        "predictionModel": predictive_output["prediction_model"],
        "causeAnalysis": cause_output,
        "risk": risk_output,
        "recommendedAction": action_output,
        "explanation": explanation_output["explanation"],
        "closedLoop": learning_record,
        "contextMemory": {
            "previous_signal_phase": context_memory.previous_signal_phase,
            "time_since_last_green": round(context_memory.time_since_last_green, 3),
            "previous_congestion_state": context_memory.previous_congestion_state,
        },
    }
    stats["aiDecision"] = classify_traffic_state(stats)

    update_task_progress(task_id, 100, "completed")

    print(f"[INFO] Video processing completed for {task_id}")
    return final_path, stats

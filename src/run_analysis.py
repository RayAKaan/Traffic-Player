import argparse
from pathlib import Path
from src.analysis.traffic_insights import TrafficInsights
from src.model.predict import process_video_with_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traffic video analysis with SmarTSignalAI.")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="SmarTSignalAI/data/processed",
                        help="Directory to save processed video and outputs")
    parser.add_argument("--enhanced", action="store_true",
                        help="Draw bounding boxes and speeds on output video")
    parser.add_argument("--model_type", type=str, default="yolov8s",
                        help="YOLO model type (yolov8n, yolov8s, etc.)")
    parser.add_argument("--confidence", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--task_id", type=str, default=None,
                        help="Optional task ID for progress tracking")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        return

    print(f"[INFO] Starting analysis for: {video_path}")

    # Process video with YOLO + tracker
    output_path, stats = process_video_with_model(
        input_path=str(video_path),
        output_dir=args.output_dir,
        enhanced=args.enhanced,
        task_id=args.task_id,
        model_type=args.model_type,
        confidence=args.confidence
    )

    # Initialize TrafficInsights with vehicle history from processed stats
    traffic_insights = TrafficInsights(frame_rate=30, meter_per_pixel=0.05)
    if 'track_positions' in stats:
        for track_id, positions in stats['track_positions'].items():
            for idx, pos in enumerate(positions):
                traffic_insights.vehicle_history[track_id].append({
                    "frame": idx,
                    "center": pos,
                    "class": stats['track_labels'].get(track_id, "unknown")
                })

    # Generate intelligent scene description
    scene_summary = traffic_insights.generate_scene_description()
    stats['traffic_insights'] = {
        "scene_description": scene_summary,
        "density": traffic_insights.compute_density(),
        "congestion_level": traffic_insights.compute_congestion_level()
    }

    # Print full report
    print("\n========== TRAFFIC REPORT ==========")
    print(f"Processed Video: {output_path}")
    print(f"Total Frames: {stats.get('total_frames')}")
    print(f"Vehicle Counts: {stats.get('per_class_counts')}")
    print(f"Average Speed (km/h): {stats.get('avg_speed_kmph')}")
    print(f"Average Density: {stats.get('avg_density')}")
    print(f"Moving Vehicles: {stats.get('moving_vehicles')}")
    print(f"Idle Vehicles: {stats.get('idle_vehicles')}")
    print(f"Congestion Level: {stats.get('congestion_level')}")
    print(f"Scene Description: {stats.get('scene_description')}")
    print(f"Accident Risk Level: {stats.get('accident_risk_level')}")
    if stats.get('alerts'):
        print(f"Alerts: {stats.get('alerts')}")
    print("\n[TrafficInsights Summary]")
    traffic = stats.get('traffic_insights', {})
    print(f"Traffic Scene: {traffic.get('scene_description')}")
    print(f"Vehicle Density per Type: {traffic.get('density')}")
    print(f"Congestion Level: {traffic.get('congestion_level')}")
    print("===================================")


if __name__ == "__main__":
    main()

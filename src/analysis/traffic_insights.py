# traffic_insights.py
import math
from collections import defaultdict

class TrafficInsights:
    """
    Generates intelligent traffic insights from vehicle detection and tracking.
    """

    def __init__(self, frame_rate=30, meter_per_pixel=0.05):
        """
        :param frame_rate: FPS of video feed
        :param meter_per_pixel: scale to convert pixel distances to real-world meters
        """
        self.frame_rate = frame_rate
        self.meter_per_pixel = meter_per_pixel
        self.vehicle_history = defaultdict(list)  # track positions per vehicle ID

    def update(self, detections, frame_idx):
        """
        Update traffic data with new frame detections.

        :param detections: list of dicts with keys:
                           'id' - unique vehicle ID
                           'bbox' - [x1, y1, x2, y2]
                           'class' - vehicle type (car, truck, bus, bike)
        :param frame_idx: current frame index
        """
        for det in detections:
            vid = det['id']
            bbox = det['bbox']
            center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
            self.vehicle_history[vid].append({'frame': frame_idx, 'center': center, 'class': det['class']})

    def compute_speed(self, vid):
        """
        Compute average speed (km/h) of a vehicle based on tracked positions.
        """
        positions = self.vehicle_history[vid]
        if len(positions) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(1, len(positions)):
            x0, y0 = positions[i-1]['center']
            x1, y1 = positions[i]['center']
            dist_pixels = math.hypot(x1-x0, y1-y0)
            total_dist += dist_pixels * self.meter_per_pixel
        total_time = (positions[-1]['frame'] - positions[0]['frame']) / self.frame_rate
        if total_time <= 0:
            return 0.0
        speed_m_s = total_dist / total_time
        speed_kmh = speed_m_s * 3.6
        return round(speed_kmh, 2)

    def compute_density(self):
        """
        Compute vehicle density per type.
        """
        density = defaultdict(int)
        for vid, history in self.vehicle_history.items():
            if history:
                v_type = history[-1]['class']
                density[v_type] += 1
        return dict(density)

    def compute_congestion_level(self):
        """
        Estimate congestion level (light, moderate, heavy)
        based on total vehicle count and speed trends.
        """
        total_vehicles = len(self.vehicle_history)
        if total_vehicles == 0:
            return "No traffic"

        avg_speed = sum(self.compute_speed(vid) for vid in self.vehicle_history) / total_vehicles

        if avg_speed > 40 and total_vehicles < 10:
            return "Light"
        elif avg_speed > 20:
            return "Moderate"
        else:
            return "Heavy"

    def generate_scene_description(self):
        """
        Generate a textual summary of the current traffic scene.
        """
        density = self.compute_density()
        congestion = self.compute_congestion_level()
        speeds = {vid: self.compute_speed(vid) for vid in self.vehicle_history}

        description = f"Traffic Scene Summary:\n"
        description += f"Congestion Level: {congestion}\n"
        description += f"Vehicle Density: {density}\n"
        description += f"Individual Speeds (km/h): {speeds}\n"

        # Optional reasoning: detect potential issues
        slow_vehicles = [vid for vid, spd in speeds.items() if spd < 5]
        if slow_vehicles:
            description += f"Alert: Slow-moving vehicles detected - IDs {slow_vehicles}\n"

        return description

# Example usage
if __name__ == "__main__":
    insights = TrafficInsights(frame_rate=30, meter_per_pixel=0.05)

    # Mock detection input per frame
    frame_0 = [
        {'id': 1, 'bbox': [100, 50, 150, 100], 'class': 'car'},
        {'id': 2, 'bbox': [200, 80, 260, 140], 'class': 'bus'}
    ]
    frame_1 = [
        {'id': 1, 'bbox': [110, 50, 160, 100], 'class': 'car'},
        {'id': 2, 'bbox': [205, 85, 265, 145], 'class': 'bus'}
    ]

    insights.update(frame_0, frame_idx=0)
    insights.update(frame_1, frame_idx=1)

    print(insights.generate_scene_description())

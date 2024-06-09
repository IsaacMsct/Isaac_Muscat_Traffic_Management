import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Define paths to your traffic report files
file_paths = {
    'msida_traffic_path': 'msida_traffic_report.txt',
    'marsahamrun_traffic_path': 'marsahamrun_traffic_report.txt',
    'qormicanonroad_traffic_path': 'qormicanonroad_traffic_report.txt'
}

class TrafficDataProcessor:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.data = self.load_and_process_data()

    def round_to_nearest_fifteen_minutes(self, dt):
        discard = timedelta(minutes=dt.minute % 15, seconds=dt.second, microseconds=dt.microsecond)
        dt -= discard
        if discard >= timedelta(minutes=7.5):
            dt += timedelta(minutes=15)
        return dt.replace(second=0, microsecond=0)

    def load_and_process_data(self):
        combined_data = {}
        for path in self.file_paths.values():
            with open(path, 'r') as file:
                for line in file:
                    parts = line.strip().split(', ')
                    timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
                    rounded_timestamp = self.round_to_nearest_fifteen_minutes(timestamp)
                    conditions = {part.split(': ')[0].strip(): part.split(': ')[1].strip() for part in parts[1:]}
                    if rounded_timestamp in combined_data:
                        combined_data[rounded_timestamp].update(conditions)
                    else:
                        combined_data[rounded_timestamp] = conditions.copy()
        df = pd.DataFrame.from_dict(combined_data, orient='index')
        df = df.fillna("unknown")  # Fill NaN values with 'unknown' before mapping
        return df.applymap(self.map_traffic)


    @staticmethod
    def map_traffic(traffic):
        if isinstance(traffic, str):
            if 'no traffic' in traffic:
                return 0
            elif 'moderate' in traffic:
                return 10
            elif 'heavy' in traffic:
                return 15
        return None  # Default return for non-string or unexpected inputs


# Process Traffic Data
processor = TrafficDataProcessor(file_paths)
hourly_avg = processor.data.resample('H').mean().fillna(0)  # Resampling to hourly and filling NaNs

# Visualization
background = cv2.imread("malta.jpg")
window_name = "Traffic Conditions"
cv2.namedWindow(window_name)

# Define locations on the map image
locations = {
    "Msida Direction Mater Dei": [(477, 294), (492, 297)],
    "Msida Direction St Julians": [(492, 297), (498, 283)],
    "Msida Direction Marsa": [(492, 297), (487, 313)],
    "Msida Direction Hamrun": [(496, 299), (494, 312)],
    "Marsa Hamrun Direction Marsa": [(484, 347), (511, 355)],
    "Marsa Hamrun Direction Santa Venera": [(487, 319), (484, 347)],
    "Marsa Hamrun Direction Qormi": [(466, 341), (482, 345)],
    "Qormi Direction Attard": [(440, 336), (456, 343)],
    "Attard Direction Hamrun": [(448, 329), (462, 337)]
}

def traffic_condition_to_color(condition):
    if condition <= 2:
        return (0, 255, 0)  # No traffic
    elif condition <= 7:
        return (0, 165, 255)  # Moderate traffic
    else:
        return (0, 0, 255)  # Heavy traffic

def update_display(hour):
    img_copy = background.copy()
    current_conditions = hourly_avg.iloc[hour]
    for location, points in locations.items():
        condition = current_conditions.get(location, 0)
        color = traffic_condition_to_color(condition)
        cv2.line(img_copy, points[0], points[1], color, thickness=10, lineType=cv2.LINE_AA)
    text = f"Hour: {hour}:00"
    cv2.putText(img_copy, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow(window_name, img_copy)

cv2.createTrackbar("Hour", window_name, 0, 23, update_display)
update_display(0)  # Display the initial condition

while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()

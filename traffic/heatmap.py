import cv2
import numpy as np
from datetime import datetime, timedelta

# Define paths to your traffic report files
msida_traffic_path = 'msida_traffic_report.txt'
marsahamrun_traffic_path = 'marsahamrun_traffic_report.txt'
qormicanonroad_traffic_path = 'qormicanonroad_traffic_report.txt'

def round_to_nearest_fifteen_minutes(dt):
    discard = timedelta(minutes=dt.minute % 15, seconds=dt.second, microseconds=dt.microsecond)
    dt -= discard
    if discard >= timedelta(minutes=7.5):
        dt += timedelta(minutes=15)
    return dt.replace(second=0, microsecond=0)

def parse_traffic_report(file_path):
    traffic_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            timestamp = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            rounded_timestamp = round_to_nearest_fifteen_minutes(timestamp)
            conditions = {part.split(': ')[0].strip(): part.split(': ')[1].strip() for part in parts[1:]}
            traffic_data[rounded_timestamp] = conditions
    return traffic_data

def combine_traffic_data(*data_dicts):
    combined_data = {}
    for data_dict in data_dicts:
        for timestamp, conditions in data_dict.items():
            if timestamp in combined_data:
                combined_data[timestamp].update(conditions)
            else:
                combined_data[timestamp] = conditions.copy()  # Copy to avoid mutable issues
    return combined_data


msida_traffic_data = parse_traffic_report(msida_traffic_path)
marsahamrun_traffic_data = parse_traffic_report(marsahamrun_traffic_path)
qormicanonroad_traffic_data = parse_traffic_report(qormicanonroad_traffic_path)
combined_traffic_data = combine_traffic_data(msida_traffic_data, marsahamrun_traffic_data, qormicanonroad_traffic_data)
sorted_timestamps = sorted(combined_traffic_data.keys())

background = cv2.imread("malta.jpg")  # Update with the correct path to your image
window_name = "Traffic Conditions"
cv2.namedWindow(window_name)

global img_copy
img_copy = background.copy()  # Initialize img_copy with the background

# Updated locations with start and end points for lines
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
    return {
        "no traffic": (0, 255, 0),
        "moderate traffic": (0, 165, 255),
        "heavy traffic": (0, 0, 255)
    }.get(condition, (255, 255, 255))  # Default for unknown conditions

def update_display(val):
    global img_copy
    img_copy = background.copy()  # Reset img_copy with a fresh background for each update
    target_timestamp_index = max(0, min(val, len(sorted_timestamps) - 1))
    target_timestamp = sorted_timestamps[target_timestamp_index]
    conditions = combined_traffic_data[target_timestamp]
    
    for location, points in locations.items():
        condition = conditions.get(location, "no data")
        color = traffic_condition_to_color(condition)
        cv2.line(img_copy, points[0], points[1], color, thickness=10, lineType=cv2.LINE_AA)

    display_timestamp(target_timestamp)

def display_timestamp(timestamp):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 1
    text = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = img_copy.shape[1] - text_size[0] - 10  # Position on the right
    text_y = 30
    cv2.putText(img_copy, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    cv2.imshow(window_name, img_copy)

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        display_img_copy = img_copy.copy()  # Use the current state of img_copy
        # Display the current x, y coordinates on the frame
        text_position = (10, display_img_copy.shape[0] - 10)  # Position at the bottom
        cv2.putText(display_img_copy, f'X: {x}, Y: {y}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(window_name, display_img_copy)

cv2.setMouseCallback(window_name, mouse_callback)

cv2.createTrackbar("Time", window_name, 0, len(sorted_timestamps) - 1, update_display)
update_display(0)  # Initial call to display the first set of traffic conditions

while True:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
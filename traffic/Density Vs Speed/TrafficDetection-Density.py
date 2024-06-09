from selenium import webdriver
from browsermobproxy import Server
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
sys.path.append("..")
import time
import cv2
import pandas as pd
import numpy as np
import schedule
from ultralytics import YOLO
from tracker import *
from matplotlib.path import Path


# Specify paths to BrowserMob Proxy and ChromeDriver
bmp_path = r"D:\school\TrafficDetection\venv\browsermob-proxy\browsermob-proxy-2.1.4\bin\browsermob-proxy.bat"
chrome_driver_path = "D:\\school\\TrafficDetection\\venv\\ChromeDriver\\chromedriver.exe"

# Start BrowserMob Proxy
server = Server(bmp_path)
server.start()
proxy = server.create_proxy()

# ChromeDriver setup with proxy
chrome_options = Options()
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument(f"--proxy-server={proxy.proxy}")

service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

# Open the target webpage and handle consent
driver.get("https://www.skylinewebcams.com/en/webcam/malta/malta/traffic/traffic-cam2.html")
driver.maximize_window()
WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, "//p[contains(text(), 'Consent')]"))).click()

# Attempt to click the play button
try:
    play_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "svg.poster-icon")))
    play_button.click()
    print("Play button clicked.")
except Exception as e:
    print("Play button not found or other error:", e)

# Start capturing network traffic
proxy.new_har("skylinewebcams", options={'captureHeaders': True, 'captureContent': True})

# Add a delay for the stream to start and for any dynamic content to load
time.sleep(10)

# Retrieve the HAR data and search for the live stream URL
har_data = proxy.har
stream_url = None
for entry in har_data['log']['entries']:
    _url = entry['request']['url']
    if "live.m3u8" in _url:
        stream_url = _url
        break

if stream_url:
    print(f"Stream URL found: {stream_url}")
else:
    print("Stream URL not found.")

# Cleanup WebDriver and BrowserMob Proxy
driver.quit()
server.stop()

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(stream_url)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0
tracker = Tracker()

cy1 = 390  # line 1
cy2 = 440  # line 2
offset = 6
vh_down = {}
vh_down_speed = {}
counter = []
vh_up = {}
vh_up_speed = {}
counter1 = []
vh_down_start_time = {}
vh_up_start_time = {}
vehicle_states = {}
calibration_value = 0.55

# Frame processing parameters
frame_processing_interval = 3  # Process every 3rd frame to reduce load
frame_counter = 0  # Counter to keep track of processed frames
target_width = 1020  # Resize target dimensions, adjust as needed for optimal performance
target_height = 500

# Initialize lists for storing traffic situations for the last 100 frames for each zone
traffic_history_zone_1 = []
traffic_history_zone_2 = []
traffic_history_zone_3 = []

# Define monitoring zones
monitoring_zones = [
    [(400, 330), (280, 330), (700, 500), (980, 500)],
    [(300, 350), (190, 350), (420, 500), (700, 500)],
    [(140, 350), (200, 350), (370, 460), (250, 470),]
]

def check_vehicle_proximity(vehicles, zones, vehicle_states, current_time):
    zone_densities = {zone_idx: {'traffic_density': 0, 'stationary_count': 0} for zone_idx, _ in enumerate(zones)}
    
    for zone_idx, zone in enumerate(zones):
        path = Path(zone)
        
        for vehicle in vehicles:
            # Unpack the bounding box and ID from each vehicle's data
            *bbox, vehicle_id = vehicle  # This matches your data structure
            bbox = tuple(bbox)  # Convert the bbox list to a tuple if necessary
            
            vehicle_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            if path.contains_point(vehicle_center):
                if vehicle_id not in vehicle_states:
                    vehicle_states[vehicle_id] = {'last_position': vehicle_center, 'last_move': current_time, 'is_stationary': False}
                else:
                    state = vehicle_states[vehicle_id]
                    if vehicle_center != state['last_position']:
                        state['last_position'] = vehicle_center
                        state['last_move'] = current_time
                        state['is_stationary'] = False
                    elif current_time - state['last_move'] > 10:  # Stationary threshold
                        state['is_stationary'] = True
                        zone_densities[zone_idx]['stationary_count'] += 1
                
                zone_densities[zone_idx]['traffic_density'] += 1  # Increment for every vehicle in the zone

    return zone_densities, vehicle_states

def update_traffic_history(zone_id, situation, history_lists):
    history = history_lists[zone_id - 1]  # Adjust for zero-based indexing
    if len(history) >= 100:
        history.pop(0)  # Remove the oldest entry if list exceeds 100 items
    history.append(situation)
    # Calculate the most frequent (mode) situation
    most_frequent = max(set(history), key=history.count)
    return most_frequent

Traffic0="Traffic_Situation"
Traffic1="Traffic_Situation"
Traffic2="Traffic_Situation"

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_counter += 1
    # Skip frames based on the specified interval to reduce load
    if frame_counter % frame_processing_interval != 0:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (target_width, target_height))

    current_frame_time = time.time()

    results = model.predict(frame)
    detected_vehicles  = []
    if results:  
        for detection in results[0].boxes:
            xyxy = detection.xyxy[0].tolist()  # Bounding box coordinates
            class_id = int(detection.cls[0].item())  # Class ID
            c = class_list[class_id]
            if 'car' in c:
                detected_vehicles.append(xyxy[:4])  # Append bounding box coordinates
        bbox_id = tracker.update(detected_vehicles, current_frame_time)

        for bbox in bbox_id:
            if len(bbox) >= 5:
                x3, y3, x4, y4, id = bbox
                x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
                cx = (x3 + x4) // 2
                cy = (y3 + y4) // 2

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.putText(frame, str(id), (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

                # Draw monitoring zones on the frame
                for zone in monitoring_zones:
                    points = np.array([zone], np.int32)  # Convert zone to a NumPy array of int32 type
                    points = points.reshape((-1, 1, 2))  # Reshape for polylines
                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    print(bbox_id)

    zone_criteria = [
        {'density_heavy': 10, 'density_moderate': 5, 'stationary_heavy': 5, 'stationary_moderate': 1},  # Criteria for Zone 1
        {'density_heavy': 8, 'density_moderate': 4, 'stationary_heavy': 4, 'stationary_moderate': 1},   # Criteria for Zone 2
        {'density_heavy': 4, 'density_moderate': 2, 'stationary_heavy': 2, 'stationary_moderate': 1}    # Criteria for Zone 3
        # Add more criteria for additional zones if necessary
    ]

    # Assuming bbox_id contains tuples of (id, bbox)
    zone_densities, vehicle_states = check_vehicle_proximity(bbox_id, monitoring_zones, vehicle_states, time.time())


     # Output traffic situation for each zone using zone-specific criteria
    for zone_idx, info in zone_densities.items():
        traffic_density = info['traffic_density']
        stationary_count = info['stationary_count']
        criteria = zone_criteria[zone_idx]  # Get the criteria for the current zone

        # Determine the traffic situation based on zone-specific criteria
        if traffic_density > criteria['density_heavy'] or stationary_count > criteria['stationary_heavy']:
            traffic_situation = "heavy traffic"
        elif criteria['density_moderate'] < traffic_density <= criteria['density_heavy'] or criteria['stationary_moderate'] < stationary_count <= criteria['stationary_heavy']:
            traffic_situation = "moderate traffic"
        else:
            traffic_situation = "no traffic"

        # Here's where you call update_traffic_history for each zone, passing the current situation
        # and the corresponding history list for the zone
        history_lists = [traffic_history_zone_1, traffic_history_zone_2, traffic_history_zone_3]
        most_frequent_situation = update_traffic_history(zone_idx + 1, traffic_situation, history_lists)

        # Example of displaying the most frequent situation for Zone 1. Adjust coordinates and repeat for other zones.
        if zone_idx == 0:
            cv2.putText(frame, f"Direction Marsa: {most_frequent_situation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            Traffic0 = most_frequent_situation
        elif zone_idx == 1:
            cv2.putText(frame, f"Direction Santa Venera: {most_frequent_situation}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            Traffic1 = most_frequent_situation
        elif zone_idx == 2:
            cv2.putText(frame, f"Direction Qormi: {most_frequent_situation}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            Traffic2 = most_frequent_situation
            
    cv2.imshow("RGB", frame)
    # Check for scheduled jobs
    schedule.run_pending()
    
    # Check for the "T" key press to manually trigger a post
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        twitter_post_job()
        
    if cv2.waitKey(1) & 0xFF == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
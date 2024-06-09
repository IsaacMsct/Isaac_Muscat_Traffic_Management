from selenium import webdriver
from browsermobproxy import Server
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import cv2
import pandas as pd
import numpy as np
import schedule
import datetime
import gc
import torch
from datetime import datetime, timedelta
from Scrape import getUrl
from Twitter import PostQormiTweet
from ultralytics import YOLO
from tracker import *
from matplotlib.path import Path
from collections import deque

print(torch.__version__)
print(torch.cuda.is_available())

global cap

def start_stream():
    global cap

    while True:
        try:
            stream_url = getUrl("https://www.skylinewebcams.com/en/webcam/malta/malta/traffic/traffic-cam9.html")
            if stream_url:
                cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    print("Stream started successfully.")
                    break
                else:
                    print("Failed to open stream. Retrying...")
            else:
                print("Failed to get stream URL. Retrying in 5 seconds...")
        except Exception as e:
            print(f"Error: {e}. Retrying in 5 seconds...")
        time.sleep(5)  # Wait before retrying

def monitor_and_restart_stream():
    global cap
    if not cap.isOpened():  # Checks if the stream is closed
        print("Stream closed or failed. Restarting...")
        cap.release()
        start_stream()

start_stream()  # Initial stream start
model = YOLO('yolov5su.pt')
tracker = Tracker()

def RGB(frame):
    # Check if 'frame' is not None
    if frame is not None:
        cv2.imshow('RGB', frame)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 

count = 0
vehicle_states = {}

# Frame processing parameters
frame_processing_interval = 2  # Process every n frame to reduce load
frame_counter = 0  # Counter to keep track of processed frames
target_width = 1020  # Resize target dimensions, adjust as needed for optimal performance
target_height = 500

# Initialize lists for storing traffic situations for the last 100 frames for each zone
history_length = 100  # Maximum length of the history
traffic_history_zone_1 = deque(maxlen=history_length)

# Define monitoring zones
monitoring_zones = [
    [(365, 280), (365, 380), (835, 180), (655, 180)],
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
    history.append(situation)
    # Calculate the most frequent (mode) situation
    most_frequent = max(set(history), key=history.count)
    return most_frequent

Traffic0="Traffic_Situation"

def twitter_post_job():
    # Assuming 'frame' is globally accessible or you pass it some other way
    cv2.imwrite("QormiMdinaRoad.jpg", frame)  # Save the frame
    # Now, call your function to post to Twitter, e.g.,
    PostQormiTweet(Traffic0)

# Global variable to store the last logged traffic situation
last_logged_info = None

 
def log_traffic_situation():
    global last_logged_info
    # Define the filename
    filename = "qormimdinaroad_traffic_report.txt"
    # Get the current time and date
    now = datetime.now()
    formatted_date_time = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare the data to be written based on current traffic situations for each direction
    data_to_write = f"{formatted_date_time},Qormi Mdina Road Roundabout: {Traffic0}"
    
    # Check if the new data is the same as the last logged data to prevent duplicate entries
    if data_to_write == last_logged_info:
        return  # Skip logging if the data is identical
    
    # Update the last logged info with current data
    last_logged_info = data_to_write
    
    # Append a newline for formatting if writing to the file
    data_to_write += "\n"
    
    # Open the file in append mode and write the data
    with open(filename, "a") as file:
        file.write(data_to_write)

def cleanup_vehicle_states(vehicle_states, max_entries=20):
    """Keep only the last `max_entries` items in the vehicle_states."""
    if len(vehicle_states) > max_entries:
        # Identify keys of the entries to be kept
        keys_to_keep = list(vehicle_states.keys())[-max_entries:]
        # Create a new dictionary with only the last `max_entries` items
        vehicle_states = {key: vehicle_states[key] for key in keys_to_keep}
    return vehicle_states

# Schedule the job every hour
schedule.every().hour.do(twitter_post_job)
# Schedule the function to run every 2 minutes
schedule.every(2).minutes.do(log_traffic_situation)

detected_vehicles  = []
frame = None  # Placeholder initialization

while True:    
    # Monitor and restart the stream if necessary
    monitor_and_restart_stream()

    ret, frame = cap.read()
    if not ret:
        time.sleep(1)  # Wait a bit before retrying to fetch the frame
        continue
    
    detected_vehicles.clear()

    frame_counter += 1

    if frame_counter % 200 == 0:
        vehicle_states = cleanup_vehicle_states(vehicle_states, max_entries=20)
        gc.collect

    # Skip frames based on the specified interval to reduce load
    if frame_counter % frame_processing_interval != 0:
        continue

    # Resize frame for faster processing
    frame = cv2.resize(frame, (target_width, target_height))

    current_frame_time = time.time()

    results = model.predict(frame)
    if results:  
        for detection in results[0].boxes:
            xyxy = detection.xyxy[0].tolist()  # Bounding box coordinates
            class_id = int(detection.cls[0].item())  # Class ID
            c = class_list[class_id]
            if 'car' in c:
                detected_vehicles.append(xyxy[:4])  # Append bounding box coordinates
        current_time = time.time()
        bbox_id = tracker.update(detected_vehicles, current_time)

        for bbox in bbox_id:
            if len(bbox) >= 5:
                x3, y3, x4, y4, id = bbox
                x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
                cx = (x3 + x4) // 2
                cy = (y3 + y4) // 2

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                #cv2.putText(frame, str(id), (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

                # Draw monitoring zones on the frame
                for zone in monitoring_zones:
                    points = np.array([zone], np.int32)  # Convert zone to a NumPy array of int32 type
                    points = points.reshape((-1, 1, 2))  # Reshape for polylines
                    cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    
    print(bbox_id)

    zone_criteria = [
        {'density_heavy': 4, 'density_moderate': 3, 'stationary_heavy': 2, 'stationary_moderate': 1},  # Criteria for Zone 1
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
        history_lists = [traffic_history_zone_1]
        most_frequent_situation = update_traffic_history(zone_idx + 1, traffic_situation, history_lists)

        # Example of displaying the most frequent situation for Zone 1. Adjust coordinates and repeat for other zones.
        if zone_idx == 0:
            cv2.putText(frame, f"Mdina Road Roundabout: {most_frequent_situation}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            Traffic0 = most_frequent_situation

    RGB(frame)
    # Check for scheduled jobs
    schedule.run_pending()
    
    del frame
    frame = None  # Reset for safety
    gc.collect()

    # Check for the "T" key press to manually trigger a post
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        twitter_post_job()
        
    if cv2.waitKey(1) & 0xFF == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
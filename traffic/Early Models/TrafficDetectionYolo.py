import sys
sys.path.append("..")
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import time

model = YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('marsa-hamrun.mp4')
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
calibration_value = 0.55

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    list = []
    if results:  
        for detection in results[0].boxes:
            xyxy = detection.xyxy[0].tolist()  # Bounding box coordinates
            class_id = int(detection.cls[0].item())  # Class ID
            c = class_list[class_id]
            if 'car' in c:
                list.append(xyxy[:4])  # Append bounding box coordinates
        bbox_id = tracker.update(list)

        for bbox in bbox_id:
            if len(bbox) >= 5:
                x3, y3, x4, y4, id = bbox
                x3, y3, x4, y4 = int(x3), int(y3), int(x4), int(y4)
                cx = (x3 + x4) // 2
                cy = (y3 + y4) // 2

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                cv2.putText(frame, str(id), (x3, y3 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)

                #####going DOWN#####
                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    vh_down[id] = time.time()
                if id in vh_down:
                    if cy2 < (cy + offset) and cy2 > (cy - offset):
                        elapsed_time = time.time() - vh_down[id]
                        if elapsed_time > 0 and id not in counter:  # Check if elapsed time is greater than 0
                            counter.append(id)
                            distance = 20 * calibration_value  # meters
                            a_speed_ms = distance / elapsed_time
                            a_speed_kh = a_speed_ms * 3.6
                            vh_down_speed[id] = a_speed_kh
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, f"{int(a_speed_kh)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

                #####going UP#####
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    vh_up[id] = time.time()
                if id in vh_up:
                    if cy1 < (cy + offset) and cy1 > (cy - offset):
                        elapsed1_time = time.time() - vh_up[id]
                        if elapsed1_time > 0 and id not in counter1:  # Check if elapsed time is greater than 0
                            counter1.append(id)      
                            distance1 = 20 * calibration_value  # meters
                            a_speed_ms1 = distance1 / elapsed1_time
                            a_speed_kh1 = a_speed_ms1 * 3.6
                            vh_up_speed[id] = a_speed_kh1
                            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.putText(frame, f"{int(a_speed_kh1)} Km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # Calculate traffic situation
    avg_speed_down = 0
    avg_speed_up = 0
    traffic_situation_down = ""
    traffic_situation_up = ""

    if len(vh_down_speed) > 2:
        avg_speed_down = sum(vh_down_speed.values()) / len(vh_down_speed)
        traffic_situation_down = "no traffic" if avg_speed_down > 30 else "slow moving traffic" if avg_speed_down > 20 else "heavy traffic"

    if len(vh_up_speed) > 2:
        avg_speed_up = sum(vh_up_speed.values()) / len(vh_up_speed)
        traffic_situation_up = "no traffic" if avg_speed_up > 30 else "slow moving traffic" if avg_speed_up > 20 else "heavy traffic"

    # Display traffic situation
    cv2.line(frame, (240, cy1), (1000, cy1), (255, 255, 255), 1)
    cv2.putText(frame, ('L1'), (277, 320), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.line(frame, (300, cy2), (1100, cy2), (255, 255, 255), 1)
    cv2.putText(frame, ('L2'), (300, 367), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'going down: {avg_speed_down:.2f} km/h, traffic situation: {traffic_situation_down}', (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'going up: {avg_speed_up:.2f} km/h, traffic situation: {traffic_situation_up}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'count: {len(vh_down_speed) + len(vh_up_speed)}', (60, 170), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

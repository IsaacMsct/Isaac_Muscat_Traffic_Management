import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import os
import xml.etree.ElementTree as ET
import psutil

# Load YOLO model
model = YOLO('yolov8s.pt')

# Set video path
video_path = 'D:\\school\\TrafficDetection\\venv\\Yolo_Vs_Dlib\\marsa-hamrun.mp4'

# Check if the file exists
if not os.path.isfile(video_path):
    print(f"Error: File {video_path} does not exist.")
    exit()

cap = cv2.VideoCapture(video_path)

# Check if the video capture was successful
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Function to parse Pascal VOC XML file
def parse_voc_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        class_label = obj.find('name').text
        if class_label.lower() == 'car':
            bbox = obj.find('bndbox')
            x1 = int(float(bbox.find('xmin').text))
            y1 = int(float(bbox.find('ymin').text))
            x2 = int(float(bbox.find('xmax').text))
            y2 = int(float(bbox.find('ymax').text))
            boxes.append([class_label, x1, y1, x2, y2])
    return boxes

# Load ground truth annotations
annotation_folder = 'D:\\school\\TrafficDetection\\venv\\Yolo_Vs_Dlib\\Annotations'  # Folder containing Pascal VOC XML files

# Initialize metrics
all_precisions = []
all_recalls = []
all_f1_scores = []
total_loss = 0
iou_losses = []
fps_list = []
cpu_usages = []
memory_usages = []

# Initialize timing for FPS calculation
start_time = time.time()
frame_count = 0

# Mapping COCO class ID to label
class_list = ['car']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original_height, original_width = frame.shape[:2]
    frame_resized = cv2.resize(frame, (1020, 500))
    resized_height, resized_width = frame_resized.shape[:2]

    # Perform inference
    results = model.predict(frame_resized)
    
    if not results:
        continue

    # Extract predictions
    preds = results[0].boxes.data
    preds = pd.DataFrame(preds).astype("float")
    
    # Placeholder for storing predicted bounding boxes and class labels
    predictions = []
    
    for index, row in preds.iterrows():
        x1 = int(row[0] * (original_width / resized_width))
        y1 = int(row[1] * (original_height / resized_height))
        x2 = int(row[2] * (original_width / resized_width))
        y2 = int(row[3] * (original_height / resized_height))
        conf = row[4]
        cls = int(row[5])
        
        # Filter to only include 'car' class (COCO class id for car is 2)
        if cls == 2:
            predictions.append(['car', x1, y1, x2, y2, conf])

    # Debug: Print predictions
    print(f"Frame {frame_count} Predictions: {predictions}")

    # Load ground truth annotations for the current frame
    annotation_file = os.path.join(annotation_folder, f'frame_{frame_count:06d}.xml')  # Adjust file naming convention if needed
    if not os.path.isfile(annotation_file):
        continue

    current_gt = parse_voc_annotation(annotation_file)

    # Debug: Print ground truth annotations
    print(f"Frame {frame_count} Ground Truth: {current_gt}")

    # Draw bounding boxes on the frame
    for pred in predictions:
        _, x1, y1, x2, y2, conf = pred
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Car {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for gt in current_gt:
        _, x1, y1, x2, y2 = gt
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'GT Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Compute precision, recall, and F1 score
    tp = 0
    fp = 0
    fn = len(current_gt)

    for pred in predictions:
        pred_cls, pred_x1, pred_y1, pred_x2, pred_y2, pred_conf = pred
        matched = False
        
        for gt in current_gt:
            gt_cls, gt_x1, gt_y1, gt_x2, gt_y2 = gt

            # Calculate IoU
            inter_x1 = max(pred_x1, gt_x1)
            inter_y1 = max(pred_y1, gt_y1)
            inter_x2 = min(pred_x2, gt_x2)
            inter_y2 = min(pred_y2, gt_y2)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            union_area = float(pred_area + gt_area - inter_area)

            if union_area > 0:
                iou = inter_area / union_area

                if iou > 0.5 and pred_cls == gt_cls:
                    tp += 1
                    fn -= 1
                    matched = True
                    iou_losses.append(1 - iou)
                    break

        if not matched:
            fp += 1

    if tp + fp > 0:
        precision = tp / (tp + fp)
        all_precisions.append(precision)
    else:
        precision = 0

    if tp + fn > 0:
        recall = tp / (tp + fn)
        all_recalls.append(recall)
    else:
        recall = 0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
        all_f1_scores.append(f1_score)
    else:
        f1_score = 0

    # Compute loss (for demonstration purposes, using sum of IoU loss)
    frame_loss = sum(iou_losses) / len(iou_losses) if iou_losses else 0
    total_loss += frame_loss

    # Calculate FPS
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    fps_list.append(fps)

    # Monitor hardware usage
    cpu_usages.append(psutil.cpu_percent())
    memory_usages.append(psutil.virtual_memory().percent)

# Calculate mean metrics
mean_precision = np.mean(all_precisions) if all_precisions else 0
mean_recall = np.mean(all_recalls) if all_recalls else 0
mean_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
mean_fps = np.mean(fps_list) if fps_list else 0
mean_iou_loss = np.mean(iou_losses) if iou_losses else 0
mean_cpu_usage = np.mean(cpu_usages) if cpu_usages else 0
mean_memory_usage = np.mean(memory_usages) if memory_usages else 0

# Print metrics
print("\n\nThese are the results for yolov8s")
print(f"Mean Average Precision (mAP): {mean_precision}")
print(f"Precision: {mean_precision}")
print(f"Recall: {mean_recall}")
print(f"Frame Per Second (FPS): {mean_fps}")
print(f"F1 Score: {mean_f1_score}")
print(f"Total Loss: {total_loss}")
print(f"IoU Loss: {mean_iou_loss}")
print(f"Average CPU Usage: {mean_cpu_usage}%")
print(f"Average Memory Usage: {mean_memory_usage}%")

cap.release()
cv2.destroyAllWindows()

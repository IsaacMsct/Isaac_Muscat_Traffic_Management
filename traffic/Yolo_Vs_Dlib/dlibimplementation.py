import cv2
import dlib
import time
import os
import xml.etree.ElementTree as ET
import numpy as np
import psutil

# Constants
WIDTH = 1280
HEIGHT = 720

# Load the car cascade and video
carCascade = cv2.CascadeClassifier("D:\\school\\TrafficDetection\\venv\\Yolo_Vs_Dlib\\vech.xml")
video = cv2.VideoCapture("D:\\school\\TrafficDetection\\venv\\Yolo_Vs_Dlib\\marsa-hamrun.mp4")

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

# IoU Calculation Function
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Track multiple objects and calculate metrics
def trackMultipleObjects():
    frameCounter = 0
    currentCarID = 0
    rectangleColor = (0, 255, 255)
    groundTruthColor = (255, 0, 0)

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}

    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    total_loss = 0
    iou_losses = []
    fps_list = []
    cpu_usages = []
    memory_usages = []

    annotation_folder = 'D:\\school\\TrafficDetection\\venv\\Yolo_Vs_Dlib\\Annotations'

    start_time = time.time()

    while True:
        start_loop_time = time.time()
        ret, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter += 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, x + w, y + h]
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_x + t_w, t_y + t_h]

        # Evaluate the predictions against ground truth
        annotation_file = os.path.join(annotation_folder, f'frame_{frameCounter:06d}.xml')
        if os.path.isfile(annotation_file):
            ground_truth = parse_voc_annotation(annotation_file)

            tp = 0
            fp = 0
            fn = len(ground_truth)

            # Draw ground truth bounding boxes
            for gt in ground_truth:
                _, gt_x1, gt_y1, gt_x2, gt_y2 = gt
                cv2.rectangle(resultImage, (gt_x1, gt_y1), (gt_x2, gt_y2), groundTruthColor, 2)

            for carID in carTracker.keys():
                [x1, y1, x2, y2] = carLocation2[carID]
                iou_max = 0
                for gt in ground_truth:
                    _, gt_x1, gt_y1, gt_x2, gt_y2 = gt

                    iou = calculate_iou([x1, y1, x2, y2], [gt_x1, gt_y1, gt_x2, gt_y2])
                    if iou > iou_max:
                        iou_max = iou

                if iou_max > 0.5:
                    tp += 1
                    fn -= 1
                    iou_losses.append(1 - iou_max)
                else:
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

            frame_loss = sum(iou_losses) / len(iou_losses) if iou_losses else 0
            total_loss += frame_loss

        end_loop_time = time.time()
        fps = 1.0 / (end_loop_time - start_loop_time)
        fps_list.append(fps)

        cpu_usages.append(psutil.cpu_percent())
        memory_usages.append(psutil.virtual_memory().percent)

        cv2.imshow('result', resultImage)
        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()

    mean_precision = np.mean(all_precisions) if all_precisions else 0
    mean_recall = np.mean(all_recalls) if all_recalls else 0
    mean_f1_score = np.mean(all_f1_scores) if all_f1_scores else 0
    mean_fps = np.mean(fps_list) if fps_list else 0
    mean_iou_loss = np.mean(iou_losses) if iou_losses else 0
    mean_cpu_usage = np.mean(cpu_usages) if cpu_usages else 0
    mean_memory_usage = np.mean(memory_usages) if memory_usages else 0

    print("\n\nThese are the results for dlib")
    print(f"Mean Average Precision (mAP): {mean_precision}")
    print(f"Precision: {mean_precision}")
    print(f"Recall: {mean_recall}")
    print(f"Frame Per Second (FPS): {mean_fps}")
    print(f"F1 Score: {mean_f1_score}")
    print(f"Total Loss: {total_loss}")
    print(f"IoU Loss: {mean_iou_loss}")
    print(f"Average CPU Usage: {mean_cpu_usage}%")
    print(f"Average Memory Usage: {mean_memory_usage}%")

if __name__ == '__main__':
    trackMultipleObjects()

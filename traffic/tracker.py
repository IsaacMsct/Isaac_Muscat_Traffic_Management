import math
import time

class Tracker:
    def __init__(self):
        self.center_points = {}  # Store the center positions of the objects
        self.id_count = 0        # Keep the count of the IDs
        self.last_seen = {}      # Keep track of the last frame an ID was seen

    def update(self, objects_rect, current_time, max_age=15):
        objects_bbs_ids = []
        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    self.last_seen[id] = current_time
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            # New object is detected
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                self.last_seen[self.id_count] = current_time
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Cleanup old tracks
        new_center_points = {}
        new_last_seen = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            # Only keep if seen in the last max_age seconds
            if current_time - self.last_seen[object_id] < max_age:
                new_center_points[object_id] = self.center_points[object_id]
                new_last_seen[object_id] = self.last_seen[object_id]

        self.center_points = new_center_points
        self.last_seen = new_last_seen
        return objects_bbs_ids

import cv2
import math
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_path='yolov8n.pt', config=None):
        self.config = config or {}
        # Load the YOLOv8 model
        self.model = YOLO(model_path)
        # Classes we care about (COCO indices)
        # 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = [2, 3, 5, 7]
        self.safe_distance_px = self.config.get('collision', {}).get('safe_distance_threshold_px', 150)
        self.warning_ttc = self.config.get('collision', {}).get('warning_ttc_threshold', 2.5)

    def detect_and_track(self, frame):
        """
        Runs YOLOv8 tracking on the frame.
        Returns the annotated frame and a list of detected vehicle metadata.
        """
        # Run inference with tracking
        results = self.model.track(frame, persist=True, verbose=False)
        
        result_frame = results[0].plot()
        detections = []
        
        # Parse results
        if results[0].boxes:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.target_classes:
                    continue
                
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                
                # Distance estimation (Simple heuristic)
                # Assuming standard car width ~1.8m. 
                # distance ~ (focal_length * real_width) / pixel_width
                # We'll use a simplified inverse relationship for demo purposes.
                # Adjust 'constant' to calibrate.
                CONSTANT = 3000 # Example constant
                distance = CONSTANT / w if w > 0 else 999
                
                # Track ID
                track_id = int(box.id[0]) if box.id is not None else -1
                
                detections.append({
                    'id': track_id,
                    'class': cls_id,
                    'box': (x1, y1, x2, y2),
                    'distance': distance,
                    'is_unsafe': distance < self.safe_distance_px # Using raw pixel/heuristic threshold
                })
        
        return result_frame, detections

    def check_risk(self, detections):
        """
        Evaluates collision risk based on detections.
        Returns a risk level (0=Safe, 1=Warning, 2=Critical) and message.
        """
        risk_level = 0
        message = "Safe"
        
        min_dist = float('inf')
        
        for d in detections:
            # Check for very close objects
            # In a real system, we'd convert 'distance' to meters. 
            # Here 'distance' is a heuristic value (higher is further, actually wait...)
            # My heuristic above: distance = CONSTANT / w. So higher w = lower distance.
            # So SMALLER 'distance' value means CLOSER vehicle.
            
            if d['distance'] < min_dist:
                min_dist = d['distance']
        
        # Threshold checking
        # Assuming our heuristic 'distance' is roughly in meters * 10 or something. 
        # Let's say safe_distance_px (from config) isn't used directly if we used w-based formula.
        # Let's stick to the heuristic: < 20 means very close, < 40 means close.
        if min_dist < 15:
             risk_level = 2
             message = "CRITICAL COLLISION RISK!"
        elif min_dist < 30:
             risk_level = 1
             message = "Warning: Vehicle Ahead"
             
        return risk_level, message

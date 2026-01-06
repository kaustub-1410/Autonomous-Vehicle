from .lane_detection import LaneDetector
from .vehicle_detection import VehicleDetector
from .drowsiness_detection import DriverMonitor
from .alert_system import AlertSystem
import cv2

class SafetySystem:
    def __init__(self, config):
        self.config = config
        
        # Initialize modules
        self.lane_detector = LaneDetector(config['lane_detection'])
        self.vehicle_detector = VehicleDetector(config['models']['yolo_model'], config)
        self.driver_monitor = DriverMonitor(config=config, predictor_path=config['models']['dlib_landmarks'])
        self.alert_system = AlertSystem(config)
        
        self.alert_history = []

    def process(self, road_frame, driver_frame=None):
        """
        Main processing loop.
        road_frame: Image from forward-facing camera
        driver_frame: Image from driver-facing camera (optional)
        """
        status = {
            'lane_offset': 0,
            'risk_level': 0, # 0: safe, 1: warning, 2: critical
            'driver_state': "Not Monitored",
            'ear': 0.0,
            'nearest_vehicle_dist': 999
        }
        
        # 1. Lane Detection
        processed_road, lane_offset = self.lane_detector.process(road_frame)
        status['lane_offset'] = lane_offset
        
        # 2. Vehicle Detection
        # YOLO works on the original frame or processed frame? Original is better.
        processed_road, detections = self.vehicle_detector.detect_and_track(processed_road)
        
        # Get nearest vehicle distance for status
        if detections:
            min_dist = min(d['distance'] for d in detections)
            status['nearest_vehicle_dist'] = min_dist
            
        risk_level, risk_msg = self.vehicle_detector.check_risk(detections)
        status['risk_level'] = max(status['risk_level'], risk_level)
        
        # 3. Drowsiness Detection (if driver frame is provided)
        processed_driver = driver_frame
        if driver_frame is not None:
            processed_driver, driver_state, ear = self.driver_monitor.process(driver_frame)
            status['driver_state'] = driver_state
            status['ear'] = ear
            
            if driver_state == "CRITICAL FATIGUE":
                status['risk_level'] = max(status['risk_level'], 2)
                risk_msg = "DRIVER FATIGUE CRITICAL"
            elif driver_state == "Drowsy":
                status['risk_level'] = max(status['risk_level'], 1)
                if risk_level < 2: # Don't overwrite collision critical
                    risk_msg = "Driver Drowsy"

        # 4. Global Alerting
        if status['risk_level'] > 0:
            severity = "critical" if status['risk_level'] == 2 else "warning"
            self.alert_system.trigger(severity, risk_msg)
            # Log alert
            self.alert_history.append((severity, risk_msg))
            
            # Overlay alert on road frame
            color = (0, 0, 255) if severity == "critical" else (0, 255, 255)
            cv2.putText(processed_road, f"ALERT: {risk_msg}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        return processed_road, processed_driver, status

import cv2
try:
    from mediapipe import solutions as mp_solutions
    _MEDIAPIPE_AVAILABLE = True
except Exception:
    mp_solutions = None
    _MEDIAPIPE_AVAILABLE = False

import numpy as np
import time
from scipy.spatial import distance as dist

class DriverMonitor:
    def __init__(self, config=None, predictor_path=None):
        """
        predictor_path is ignored for MediaPipe but kept for signature compatibility.
        """
        self.config = config or {}
        
        # Initialize MediaPipe Face Mesh if available; otherwise run a no-op monitor
        self.enabled = False
        if _MEDIAPIPE_AVAILABLE and hasattr(mp_solutions, 'face_mesh'):
            try:
                self.mp_face_mesh = mp_solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.enabled = True
            except Exception:
                self.enabled = False

        # EAR thresholds
        self.ear_threshold = self.config.get('drowsiness', {}).get('ear_threshold', 0.25)
        self.drowsy_time_thresh = self.config.get('drowsiness', {}).get('drowsy_time_threshold', 1.5)
        self.critical_time_thresh = self.config.get('drowsiness', {}).get('critical_time_threshold', 2.5)
        
        # State variables
        self.eye_closed_start_time = None
        self.status = "Normal"
        
        # Landmark indices for MediaPipe (Fixed indices for eyes)
        # Left eye: 362, 385, 387, 263, 373, 380
        # Right eye: 33, 160, 158, 133, 153, 144
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def calculate_ear(self, landmarks, frame_w, frame_h):
        """
        landmarks: list of normalized landmark objects
        indices: list of indices for a specific eye
        """
        def get_coords(idx):
            return np.array([landmarks[idx].x * frame_w, landmarks[idx].y * frame_h])

        # Left Eye
        l_p = [get_coords(i) for i in self.LEFT_EYE]
        # Vertical
        l_A = dist.euclidean(l_p[1], l_p[5])
        l_B = dist.euclidean(l_p[2], l_p[4])
        # Horizontal
        l_C = dist.euclidean(l_p[0], l_p[3])
        left_ear = (l_A + l_B) / (2.0 * l_C)

        # Right Eye
        r_p = [get_coords(i) for i in self.RIGHT_EYE]
        # Vertical
        r_A = dist.euclidean(r_p[1], r_p[5])
        r_B = dist.euclidean(r_p[2], r_p[4])
        # Horizontal
        r_C = dist.euclidean(r_p[0], r_p[3])
        right_ear = (r_A + r_B) / (2.0 * r_C)

        return (left_ear + right_ear) / 2.0

    def process(self, frame):
        """
        Detects face, computes EAR, and updates driver status.
        """
        # If mediapipe face mesh not available, return a passthrough with status
        if not self.enabled:
            return frame, "Not Monitored", 0.0
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        current_ear = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate EAR
                current_ear = self.calculate_ear(face_landmarks.landmark, w, h)
                
                # Visualize Eyes (Simple dots)
                for idx in self.LEFT_EYE + self.RIGHT_EYE:
                    lm = face_landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Logic for drowsiness
                if current_ear < self.ear_threshold:
                    if self.eye_closed_start_time is None:
                        self.eye_closed_start_time = time.time()
                    
                    duration = time.time() - self.eye_closed_start_time
                    
                    if duration >= self.critical_time_thresh:
                        self.status = "CRITICAL FATIGUE"
                    elif duration >= self.drowsy_time_thresh:
                        self.status = "Drowsy"
                    else:
                        self.status = "Normal" # Blink
                else:
                    self.eye_closed_start_time = None
                    self.status = "Normal"
                
                # We only process the first face
                break
        else:
            self.eye_closed_start_time = None
            self.status = "No Face Detected"
            
        return frame, self.status, current_ear

import cv2
import numpy as np

class LaneDetector:
    def __init__(self, config=None):
        self.config = config or {}
        self.roi_vertices = None
        # Default parameters from config or hardcoded fallback
        self.canny_low = self.config.get('canny_threshold1', 50)
        self.canny_high = self.config.get('canny_threshold2', 150)
        self.hough_rho = self.config.get('hough_rho', 2)
        self.hough_theta = self.config.get('hough_theta', np.pi/180)
        self.hough_threshold = self.config.get('hough_threshold', 100)
        self.min_line_len = self.config.get('hough_min_line_len', 40)
        self.max_line_gap = self.config.get('hough_max_line_gap', 50)

    def preprocess(self, frame):
        """Applies grayscale, blur, and edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        return edges

    def region_of_interest(self, edges):
        """Masks the region of interest for lane detection."""
        height, width = edges.shape
        mask = np.zeros_like(edges)

        # Define a triangular polygon for ROI (bottom center of image)
        # Adjust these ratios based on camera mounting
        polygon = np.array([[
            (0, height),
            (width // 2, int(height * 0.6)),
            (width, height),
        ]], np.int32)

        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        return masked_edges

    def detect_lines(self, masked_edges):
        """Detects line segments using Hough Transform."""
        lines = cv2.HoughLinesP(
            masked_edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            np.array([]),
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap
        )
        return lines

    def separate_lines(self, lines, img_shape):
        """Separates lines into left and right lanes."""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return None, None

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # Ignore vertical lines
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter noise based on slope
                if slope < -0.5: # Left lane
                    left_lines.append((slope, y1 - slope * x1))
                elif slope > 0.5: # Right lane
                    right_lines.append((slope, y1 - slope * x1))

        left_lane = self._average_lane(left_lines, img_shape)
        right_lane = self._average_lane(right_lines, img_shape)
        
        return left_lane, right_lane

    def _average_lane(self, lines, img_shape):
        """Averages line parameters and computes a single line."""
        if not lines:
            return None
        
        avg_slope, avg_intercept = np.mean(lines, axis=0)
        y1 = img_shape[0] # Bottom of image
        y2 = int(y1 * 0.6) # Horizon line
        
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        
        return ((x1, y1), (x2, y2))

    def draw_lanes(self, frame, left_lane, right_lane):
        """Draws detected lanes and drivable area overlay."""
        overlay = frame.copy()
        
        # Draw lines
        if left_lane:
            cv2.line(overlay, left_lane[0], left_lane[1], (0, 0, 255), 10)
        if right_lane:
            cv2.line(overlay, right_lane[0], right_lane[1], (0, 0, 255), 10)
            
        # Draw drivable area if both lanes exist
        if left_lane and right_lane:
            pts = np.array([
                left_lane[0], left_lane[1], right_lane[1], right_lane[0]
            ], np.int32)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
    def process(self, frame):
        """Full pipeline processing for a single frame."""
        edges = self.preprocess(frame)
        masked_edges = self.region_of_interest(edges)
        lines = self.detect_lines(masked_edges)
        left_lane, right_lane = self.separate_lines(lines, frame.shape)
        result_frame = self.draw_lanes(frame, left_lane, right_lane)
        
        # Calculate lane status (simple offset)
        # Assuming camera center is image center
        center_offset = 0
        if left_lane and right_lane:
            lane_center_x = (left_lane[0][0] + right_lane[0][0]) / 2
            image_center_x = frame.shape[1] / 2
            center_offset = (lane_center_x - image_center_x)
            
        return result_frame, center_offset

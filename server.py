from flask import Flask, render_template, Response, jsonify
import cv2
import yaml
import time
import threading
import sqlite3
import os
from datetime import datetime
from src.safety_controller import SafetySystem
import download_data  # Helper script

app = Flask(__name__)

# Global state
output_frame = None
lock = threading.Lock()
system_status = {}
config = None
DB_FILE = "safety_logs.db"

def init_db():
    """Initialize SQLite database for alerts."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  level TEXT, 
                  message TEXT)''')
    conn.commit()
    conn.close()
    print("Database initialized.")

def log_alert(level, message):
    """Log an alert to the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO alerts (timestamp, level, message) VALUES (?, ?, ?)",
                  (timestamp, level, message))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

# Load config
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# --- AUTO-SETUP ON STARTUP ---
print("Running automated setup...")
try:
    download_data.setup_models()
    download_data.setup_sample_data()
    print("Setup complete.")
except Exception as e:
    print(f"Setup failed: {e}")

config = load_config()
system = SafetySystem(config)
init_db()

def process_feed():
    global output_frame, system_status
    
    # Video Sources
    video_source_path = config['input']['road_video_path']
    cap_road = cv2.VideoCapture(video_source_path)
    
    # Try different indices for webcam
    cap_driver = None
    for index in [0, 1]:
        try:
            temp_cap = cv2.VideoCapture(index)
            if temp_cap.isOpened():
                print(f"Driver webcam found at index {index}")
                cap_driver = temp_cap
                break
        except:
             continue
    
    if cap_driver is None:
        print("Warning: No driver webcam found.")

    last_alert_time = 0

    while True:
        ret, road_frame = cap_road.read()
        if not ret:
            cap_road.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
            continue
            
        driver_frame = None
        driver_cam_active = False
        
        if cap_driver:
            ret_d, d_frame = cap_driver.read()
            if ret_d:
                driver_frame = d_frame
                driver_cam_active = True
            else:
                pass
        
        # Fallback if road frame is missing/failed
        if not ret or road_frame is None:
            # Create a "Static / Error" frame
            import numpy as np
            road_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(road_frame, "VIDEO SOURCE ERROR", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(road_frame, f"Checking: {video_source_path}", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Process
        try:
            p_road, p_driver, status = system.process(road_frame, driver_frame)
        except Exception as e:
            print(f"Processing Error: {e}")
            p_road = road_frame # Fallback to raw frame

        # Log Critical Alerts to DB (Rate limited)
        risk_level = status.get('risk_level', 0)
        if risk_level == 2:
            current_time = time.time()
            if current_time - last_alert_time > 5.0: # Log every 5 seconds max
                log_alert("CRITICAL", "High Collision Risk or Driver Fatigue Detected")
                last_alert_time = current_time

        # Update global status for API
        with lock:
            system_status = status
            
            # Picture-in-Picture Overlay
            r_h, r_w = p_road.shape[:2]
            target_w = int(r_w * 0.25)
            target_h = int(target_w * 0.75) # Aspect ratio if cam missing
            
            if p_driver is not None:
                d_h, d_w = p_driver.shape[:2]
                ratio = target_w / float(d_w)
                target_h = int(d_h * ratio)
                small_driver = cv2.resize(p_driver, (target_w, target_h))
                p_road[20:20+target_h, r_w-target_w-20:r_w-20] = small_driver
                
                # Draw border
                cv2.rectangle(p_road, (r_w-target_w-20, 20), (r_w-20, 20+target_h), (0, 255, 0), 2)
                cv2.putText(p_road, "DRIVER CAM", (r_w-target_w-20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                # Draw "NO SIGNAL" placeholder
                start_x = r_w - target_w - 20
                start_y = 20
                cv2.rectangle(p_road, (start_x, start_y), (r_w-20, start_y+target_h), (0, 0, 0), -1)
                cv2.rectangle(p_road, (start_x, start_y), (r_w-20, start_y+target_h), (0, 0, 255), 2)
                
                text = "NO DRIVER CAM"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = start_x + (target_w - tw) // 2
                text_y = start_y + (target_h + th) // 2
                cv2.putText(p_road, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            output_frame = p_road.copy()
            
        # Limit FPS
        time.sleep(0.01)

        # DEBUG: Print status every 100 frames
        if app.debug_counter % 100 == 0:
             pass
             # cam_status = "ACTIVE" if driver_cam_active else "FAIL/NONE"
             # print(f"DEBUG: ...")
        app.debug_counter += 1

# Add global counter
app.debug_counter = 0

def generate_mjpeg():
    global output_frame, lock
    while True:
        if output_frame is None:
            time.sleep(0.1)
            continue
            
        with lock:
            if output_frame is None: 
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        
        if not flag:
            continue
            
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + b'\r\n')
        time.sleep(0.05) # Limit stream to ~20 FPS

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def get_status():
    global system_status
    with lock:
        return jsonify(system_status)

@app.route("/api/alerts")
def get_alerts():
    """Return recent alerts from DB."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)

@app.route("/api/test_alert")
def test_alert():
    """Manually trigger an alert for testing."""
    system.alert_system.trigger("critical", "MANUAL SYSTEM TEST")
    return jsonify({"status": "triggered"})

def play_startup_sound():
    try:
        import winsound
        winsound.Beep(1000, 200)
        time.sleep(0.1)
        winsound.Beep(1500, 400)
    except:
        pass

if __name__ == "__main__":
    play_startup_sound()
    t = threading.Thread(target=process_feed, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

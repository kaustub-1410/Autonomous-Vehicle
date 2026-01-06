import streamlit as st
import cv2
import yaml
import time
import os
import numpy as np
from src.safety_controller import SafetySystem

# Set page config
st.set_page_config(page_title="AV Safety System", layout="wide")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    st.title("🚗 Autonomous Vehicle Safety System Simulation")
    st.markdown("### Software-Based ADAS: Lane Keeping, Collision Avoidance, Driver Monitoring")

    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load Config
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("config.yaml not found!")
        return

    # Controls
    run_app = st.sidebar.checkbox("Start Simulation", value=False)
    enable_webcam = st.sidebar.checkbox("Enable Driver Monitoring (Webcam)", value=False)
    
    # Video Source
    video_source_path = config['input']['road_video_path']
    # Allow override in case file is missing
    if not os.path.exists(video_source_path):
        st.sidebar.warning(f"Default video {video_source_path} not found.")
        uploaded_video = st.sidebar.file_uploader("Upload a road video (mp4)", type=["mp4", "avi"])
        if uploaded_video is not None:
            # Save to temp file
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_video.read())
            video_source_path = "temp_video.mp4"
    
    # Init System (Cache to avoid reloading models every frame loop if streamlit reruns, 
    # but here we use a dedicated loop inside 'if run_app')
    if 'system' not in st.session_state:
        status_text = st.sidebar.empty()
        status_text.text("Initializing AI Models... Please wait.")
        st.session_state.system = SafetySystem(config)
        status_text.text("System Ready")

    system = st.session_state.system

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Road View (Lane & Vehicles)")
        road_display = st.empty()
    with col2:
        st.subheader("Driver Monitor (Drowsiness)")
        driver_display = st.empty()

    # Metrics
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    metric_lane = m_col1.empty()
    metric_dist = m_col2.empty()
    metric_driver = m_col3.empty()
    metric_alert = m_col4.empty()

    # Alert Log
    st.subheader("System Logs")
    log_display = st.empty()

    if run_app:
        # Open video capture
        if not video_source_path or not os.path.exists(video_source_path):
             st.error("No valid video source enabled.")
             return
             
        cap_road = cv2.VideoCapture(video_source_path)
        
        cap_driver = None
        if enable_webcam:
            cap_driver = cv2.VideoCapture(0)

        prev_time = time.time()
        
        while cap_road.isOpened() and run_app:
            ret, road_frame = cap_road.read()
            if not ret:
                # Loop video
                cap_road.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Driver Frame
            driver_frame = None
            if cap_driver and cap_driver.isOpened():
                ret_d, driver_frame_raw = cap_driver.read()
                if ret_d:
                    driver_frame = driver_frame_raw

            # Update System
            processed_road, processed_driver, status = system.process(road_frame, driver_frame)

            # Display Road
            # Convert BGR to RGB
            road_rgb = cv2.cvtColor(processed_road, cv2.COLOR_BGR2RGB)
            road_display.image(road_rgb, channels="RGB", use_column_width=True)

            # Display Driver
            if processed_driver is not None:
                driver_rgb = cv2.cvtColor(processed_driver, cv2.COLOR_BGR2RGB)
                driver_display.image(driver_rgb, channels="RGB", use_column_width=True)
            else:
                driver_display.info("Driver Camera Disabled")

            # Update Metrics
            metric_lane.metric("Lane Offset", f"{status['lane_offset']:.1f} px")
            
            dist_val = status['nearest_vehicle_dist']
            dist_str = f"{dist_val:.1f} m" if dist_val != 999 else "Clear" # Heuristic value
            metric_dist.metric("Nearest Vehicle", dist_str)
            
            state = status['driver_state']
            state_color = "normal"
            if state == "Drowsy": state_color = "off" # Streamlit doesn't strictly have color args for metric, but we pass string
            metric_driver.metric("Driver Status", state, delta=f"EAR: {status['ear']:.2f}")

            # Alert status
            risk = "SAFE"
            if status['risk_level'] == 1: risk = "WARNING"
            if status['risk_level'] == 2: risk = "CRITICAL"
            metric_alert.metric("System Status", risk)
            
            # FPS Control (approx)
            # time.sleep(0.01) 
            
            # Check Stop
            # Just relying on 'run_app' checkbox state which updates on interaction (forcing rerun), 
            # but inside this loop we are blocking. Streamlit handles this via 'st.empty' updates usually, 
            # but standard 'while' loop blocks UI updates unless we use unique keys or rerun.
            # Actually, Streamlit 'magic' loop works fine usually, but 'Stop' button won't register 
            # until loop yields. We will just run.
            pass

        cap_road.release()
        if cap_driver:
            cap_driver.release()
    else:
        st.info("Check 'Start Simulation' in the sidebar to run.")

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import yaml
import os
from src.safety_controller import SafetySystem

def verify():
    print("Loading config...")
    if not os.path.exists("config.yaml"):
        print("FAIL: config.yaml not found")
        return

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Initializing SafetySystem...")
    try:
        system = SafetySystem(config)
        print("PASS: SafetySystem initialized.")
    except Exception as e:
        print(f"FAIL: SafetySystem init failed: {e}")
        return

    print("Testing processing loop with dummy data...")
    try:
        # Create dummy frames (black images)
        dummy_road = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_driver = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Run process
        p_road, p_driver, status = system.process(dummy_road, dummy_driver)
        
        print("PASS: Processing loop completed.")
        print(f"Status Output: {status}")
        
    except Exception as e:
        print(f"FAIL: Processing loop error: {e}")

if __name__ == "__main__":
    verify()

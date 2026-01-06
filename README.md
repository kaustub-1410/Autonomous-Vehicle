# Software-Only Autonomous Vehicle Safety System

A purely software-based simulation of an Autonomous Vehicle Safety System using Python.
This project demonstrates advanced Driver Assistance Systems (ADAS) concepts including Lane Detection, Vehicle Proximity Detection, and Driver Drowsiness Monitoring using simulated video inputs.

## Features
- **Lane Detection**: Real-time road lane detection and curvature estimation using OpenCV.
- **Vehicle Detection**: Object detection using YOLOv8 to identify vehicles and estimate collision risks.
- **Drowsiness Detection**: Facial landmark analysis (dlib) to monitor driver fatigue via webcam or video.
- **Safety Controller**: Central logic fusing data to trigger audio and visual alerts.
- **Dashboard**: Interactive Streamlit UI for real-time visualization.

## Setup

### Prerequisites
- Python 3.8+
- Webcam (for drowsiness detection demo)

### Installation
1.  Clone the repository or extract the project.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download required models and sample data:
    ```bash
    python download_data.py
    ```
    *This will download `yolov8n.pt` (automatically on first run or you can let the script do it), `shape_predictor_68_face_landmarks.dat` for dlib, and a sample road video.*

## Usage

### Running the Dashboard
To launch the full system with the UI:
```bash
streamlit run app.py
```

### Configuration
Edit `config.yaml` to change:
- Video source paths
- Alert thresholds (TTC, EAR)
- Email alert settings

## Project Structure
- `src/`: Core logic modules (lane, vehicle, drowsiness detection).
- `data/`: Sample videos and input files.
- `models/`: ML models (YOLO, dlib landmarks).
- `app.py`: Main Streamlit dashboard application.

import os
import requests
import sys
import bz2
import shutil

def download_file(url, save_path):
    """Downloads a file from a URL to a specific path."""
    print(f"Downloading {save_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def setup_models():
    """Downloads required models."""
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Dlib Shape Predictor
    dlib_model_name = "shape_predictor_68_face_landmarks.dat"
    dlib_bz2_name = f"{dlib_model_name}.bz2"
    dlib_path = os.path.join(models_dir, dlib_model_name)
    dlib_bz2_path = os.path.join(models_dir, dlib_bz2_name)
    
    dlib_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    
    if not os.path.exists(dlib_path):
        if not os.path.exists(dlib_bz2_path):
            success = download_file(dlib_url, dlib_bz2_path)
            if not success:
                print("Failed to download dlib model. Please download manually.")
                return

        print("Extracting dlib model...")
        try:
            with bz2.BZ2File(dlib_bz2_path) as fr, open(dlib_path, 'wb') as fw:
                shutil.copyfileobj(fr, fw)
            os.remove(dlib_bz2_path)
            print("Dlib model extracted successfully.")
        except Exception as e:
            print(f"Error extracting dlib model: {e}")
    else:
        print(f"Dlib model already exists at {dlib_path}")

def setup_sample_data():
    """Downloads sample video data for testing."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Sample Road Video (using a standard test video from a public source or placeholder)
    # Using the classic 'solidWhiteRight.mp4' or similar if available, or a reliable generic one.
    # For now, we will use a placeholder or a known available sample. 
    # Let's try to fetch a sample from a robust source or instruct user.
    # Sintel trailer is often used as a dummy, but we want a road video.
    # We'll use a specific focused sample if possible. 
    # For now, let's create a placeholder README in data if we can't find a direct link guaranteed to work forever.
    # Actually, let's try to download a small sample from a repo that hosts lane detection data.
    
    video_name = "road_sample.mp4"
    video_path = os.path.join(data_dir, video_name)
    
    # This is a sample video often used in lane detection tutorials
    sample_video_url = "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4" 
    
    if not os.path.exists(video_path):
        success = download_file(sample_video_url, video_path)
        if success:
            print(f"Sample video downloaded to {video_path}")
    else:
        print(f"Sample video already exists at {video_path}")

if __name__ == "__main__":
    print("Starting setup...")
    setup_models()
    setup_sample_data()
    print("Setup complete.")

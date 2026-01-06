import sys
import mediapipe
print(f"Python: {sys.version}")
print(f"MediaPipe File: {mediapipe.__file__}")
print(f"Dir: {dir(mediapipe)}")
try:
    print(f"Solutions: {mediapipe.solutions}")
except AttributeError as e:
    print(f"Error: {e}")

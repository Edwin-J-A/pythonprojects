#clicking using eye blink

import cv2
import dlib
import pyautogui
from scipy.spatial import distance
import time

# Initialize webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # Ensure this file is available

# EAR threshold for blink detection
EAR_THRESHOLD = 0.2
BLINK_FRAMES = 3
blink_counter = 0
click_triggered = False

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    ret, frame = cam.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        continue
    
    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        print(f"EAR: {avg_ear:.2f}")  # Debugging EAR values
        
        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= BLINK_FRAMES and not click_triggered:
                pyautogui.click()
                print("Blink detected - Click performed")
                click_triggered = True
                time.sleep(0.2)  # Small delay to prevent rapid clicking
        else:
            blink_counter = 0
            click_triggered = False
    
    cv2.imshow('Blink-Controlled Clicking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit key pressed. Closing program.")
        break

cam.release()
cv2.destroyAllWindows()

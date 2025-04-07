import cv2
import mediapipe as mp
import numpy as np
import dlib
import pyautogui
import time
import threading
import speech_recognition as sr
import pyttsx3
from scipy.spatial import distance

# Initialize webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# PyAutoGUI settings
pyautogui.PAUSE = 0  # Remove built-in delays between actions
pyautogui.FAILSAFE = False  # Disable fail-safe

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize dlib for blink detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
listening = False

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Blink detection variables
EAR_THRESHOLD = 0.2
BLINK_FRAMES = 3
blink_counter = 0
click_triggered = False

# Scrolling variables
scroll_speed = 20
previous_mouth_top = None
previous_mouth_bottom = None

# Function to check lighting conditions
def is_bad_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 50

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to detect if mouth is open for scrolling
def is_mouth_open(shape):
    return shape[66][1] - shape[62][1] > 10

# Function to handle voice commands
def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

def listen_for_command():
    global listening
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                if "hey bro" in command:
                    speak("Yes, I'm listening.")
                    listening = True
                    listen_for_voice_commands()
                    break
            except:
                pass

def listen_for_voice_commands():
    global listening
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while listening:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                if "zoom in" in command:
                    speak("Zooming in...")
                    pyautogui.hotkey("ctrl", "+")
                elif "zoom out" in command:
                    speak("Zooming out...")
                    pyautogui.hotkey("ctrl", "-")
                elif "pause" in command:
                    speak("Pausing...")
                elif "resume" in command:
                    speak("Resuming...")
                elif "bye bro" in command:
                    speak("Goodbye!")
                    listening = False
                    break
                elif "stop" in command:
                    speak("Stopping the program. Goodbye!")
                    exit()
            except:
                pass

# Run voice command listener in a separate thread
threading.Thread(target=listen_for_command, daemon=True).start()

while True:
    ret, frame = cam.read()
    if not ret:
        continue
    
    if is_bad_lighting(frame):
        print("Bad lighting detected! Stopping program.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face Mesh detection for nose tracking (optional)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            x, y = int(nose_tip.x * w), int(nose_tip.y * h)

            # Convert to screen coordinates and constrain within screen bounds.
            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])
            screen_x = max(0, min(screen_x, screen_w - 1))
            screen_y = max(0, min(screen_y, screen_h - 1))

            pyautogui.moveTo(screen_x, screen_y)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Blink detection for clicking functionality.
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Click detection logic.
        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
            
            if blink_counter >= BLINK_FRAMES and not click_triggered:
                pyautogui.mouseDown()  # Simulate mouse press.
                time.sleep(0.05)      # Short delay to simulate click.
                pyautogui.mouseUp()   # Simulate mouse release.
                click_triggered = True
                
        else:
            blink_counter = 0
            click_triggered = False

        # Scroll detection using mouth movement.
        shape_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        
        mouth_open = is_mouth_open(shape_points)
        
        if mouth_open and previous_mouth_bottom is not None: 
            scroll_direction = 1 if shape_points[66][1] > previous_mouth_bottom else -1 
            pyautogui.scroll(scroll_direction * scroll_speed)

        previous_mouth_top = shape_points[62][1]
        previous_mouth_bottom = shape_points[66][1]

    cv2.imshow("Integrated Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

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
import platform
import os
import pygetwindow as gw
import ctypes
import winsound
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()


cap = None

def stop_program():
    global running, video_feed_paused, cap
    print("Stopping the program. Goodbye!")
    speak("Stopping the program. Goodbye!")

    running = False
    video_feed_paused = False

    try:
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error releasing camera or closing windows:", e)

    try:
        engine.stop()
    except Exception as e:
        print("Error stopping speech engine:", e)

    # Force immediate shutdown to prevent hang
    os._exit(0)


def click(x=None, y=None):
    if x is not None and y is not None:
        ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(0x02, 0, 0, 0, 0)
    ctypes.windll.user32.mouse_event(0x04, 0, 0, 0, 0)
    time.sleep(0.05)
    ctypes.windll.user32.mouse_event(0x02, 0, 0, 0, 0)
    ctypes.windll.user32.mouse_event(0x04, 0, 0, 0, 0)

    # 🎵 Sound feedback
    winsound.Beep(800, 100)  # 800 Hz tone for 100 ms

#def click(x, y):
#    ctypes.windll.user32.SetCursorPos(x, y)
#    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # Left button down
 #   ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # Left button up


# Initialize webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Camera warm-up: discard first few frames
print("Warming up the camera...")
for _ in range(10):
    cam.read()
time.sleep(2)
speak("System ready.")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize dlib for blink and mouth detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()
listening = False
paused = False  # For video feed pause/resume

# Screen dimensions
screen_w, screen_h = pyautogui.size()

# Blink detection variables
EAR_THRESHOLD = 0.2
BLINK_FRAMES = 3
blink_counter = 0
click_triggered = False
last_click_time = 0

# Scrolling variables
scroll_speed = 10
scroll_sensitivity = 0.8
previous_mouth_top = None
previous_mouth_bottom = None

# Accesibility mode
current_profile = "default"


cursor_sensitivity = 1.0
scroll_multiplier = 1.0
ear_threshold = 0.2
blink_frames_required = 3
tts_rate = 150

# Profile Switching
current_profile = "default"

profile_settings = {
    "default": {
        "speak_enabled": True,
        "scroll_speed": 10,
        "blink_frames": 3,
    },
    "presentation": {
        "speak_enabled": True,
        "scroll_speed": 0,     # disable scrolling
        "blink_frames": 999,   # disable clicking
    },
    "silent": {
        "speak_enabled": False,
        "scroll_speed": 10,
        "blink_frames": 3,
    },
    "accessibility": {
        "speak_enabled": True,
        "scroll_speed": 20,
        "blink_frames": 5,
    }
}

# Applying profiles
def apply_profile(profile_name):
    global scroll_speed, BLINK_FRAMES, current_profile
    global cursor_sensitivity, scroll_multiplier, ear_threshold, blink_frames_required, tts_rate

    if profile_name in profile_settings:
        settings = profile_settings[profile_name]
        scroll_speed = settings["scroll_speed"]
        BLINK_FRAMES = settings["blink_frames"]
        current_profile = profile_name

        # Apply custom control settings
        if profile_name == "accessibility":
            cursor_sensitivity = 0.5
            scroll_multiplier = 1.5
            ear_threshold = 0.18
            blink_frames_required = 4
            tts_rate = 120
        else:
            cursor_sensitivity = 1.0
            scroll_multiplier = 1.0
            ear_threshold = 0.2
            blink_frames_required = 3
            tts_rate = 150

        # Apply TTS speed
        engine.setProperty('rate', tts_rate)

        # Feedback
        if settings["speak_enabled"]:
            speak(f"{profile_name.capitalize()} mode activated.")
        else:
            print(f"{profile_name.capitalize()} mode activated (silent).")
    else:
        speak("Unknown profile.")


# Function to check lighting conditions
def is_bad_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 50

# EAR (eye aspect ratio) calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Mouth open detection
def is_mouth_open(shape):
    return shape[66][1] - shape[62][1] > 10

# TTS function
def speak(text):
    if profile_settings[current_profile]["speak_enabled"]:
        print(text)
        engine.say(text)
        engine.runAndWait()


# Minimize current window
def minimize_window():
    try:
        window = gw.getActiveWindow()
        if window:
            window.minimize()
            speak("Window minimized.")
    except Exception as e:
        print(f"Error minimizing: {e}")
        speak("Sorry, couldn't minimize the window.")

# Maximize current window
def maximize_window():
    try:
        window = gw.getActiveWindow()
        if window:
            window.maximize()
            speak("Window maximized.")
    except Exception as e:
        print(f"Error maximizing: {e}")
        speak("Sorry, couldn't maximize the window.")

# Voice command trigger listener
def listen_for_command():
    global listening
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                print("Listening for hotword...")
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                if "hello" in command:
                    speak("Yes, I'm listening.")
                    listening = True
                    threading.Thread(target=listen_for_voice_commands, daemon=True).start()
                    break
            except sr.UnknownValueError:
                continue
            except Exception as e:
                print(f"Error in listen_for_command: {e}")

# Voice command processor
dragging = False
def listen_for_voice_commands():
    global listening, paused, dragging
    while listening:
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Voice command thread active.")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                command = recognizer.recognize_google(audio).lower()
                print("Recognized command:", command)
                

                if "zoom in" in command:
                    speak("Zooming in...")
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('+')  # Works in most apps where '+' triggers zoom
                    pyautogui.keyUp('ctrl')
                elif "zoom out" in command:
                    speak("Zooming out...")
                    pyautogui.keyDown('ctrl')
                    pyautogui.press('-')
                    pyautogui.keyUp('ctrl')
                   # pyautogui.hotkey("ctrl", "-")
                elif "minimise" in command:
                    minimize_window()
                elif "maximize" in command:
                    maximize_window()
                elif "hold" in command:
                    speak("Pausing video feed.")
                    paused = True
                elif "resume" in command:
                    speak("Resuming video feed.")
                    paused = False
                elif "new tab" in command:
                    speak("Opening new tab.")
                    pyautogui.hotkey("ctrl", "t")
                elif "next tab" in command:
                    speak("Switching to next tab.")
                    pyautogui.hotkey("ctrl", "tab")
                elif "previous tab" in command:
                    speak("Switching to previous tab.")
                    pyautogui.hotkey("ctrl", "shift", "tab")
                elif "stop" in command:
                    speak("Stopping the program. Goodbye!")
                    cam.release()
                    cv2.destroyAllWindows()
                    os._exit(0)
                #elif "default mode" in command:
                #    apply_profile("default")
                elif "presentation mode" in command:
                    apply_profile("presentation")
                elif "silent mode" in command:
                    apply_profile("silent")
                elif "accessibility mode" in command:
                    apply_profile("accessibility")
                elif "default mode" in command:
                    apply_profile("default")
                elif "drag mode" in command:
                    pyautogui.mouseDown()
                    dragging = True
                    speak("Dragging started.")
                elif "drop" in command:
                    #if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                        speak("Dropped.")
                elif "close" in command:
                    try:
                        window = gw.getActiveWindow()
                        if window:
                            window.close()
                            speak("Window closed.")
                        else:
                            speak("No active window to close.")
                    except Exception as e:
                        print(f"Error closing window: {e}")
                        speak("Sorry, I couldn't close the window.")

                elif "bye" in command or "bye bro" in command:
                    speak("Goodbye!")
                    listening = False
                    break
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected.")
            continue
        except sr.UnknownValueError:
            print("Didn't catch that.")
            continue
        except Exception as e:
            print(f"Error in voice command thread: {e}")
            continue
# Start background listener
threading.Thread(target=listen_for_command, daemon=True).start()

# Main video loop
while True:
    if paused:
        key = cv2.waitKey(100)  # Allows other processes to run while waiting
        if key == ord('q'):
            break
        continue  # Go back to loop start

    ret, frame = cam.read()
    if not ret:
        continue
    if is_bad_lighting(frame):
        print("Bad lighting detected! Stopping program.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face mesh processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if current_profile != "presentation":
                nose_tip = face_landmarks.landmark[1]
                x, y = int(nose_tip.x * w), int(nose_tip.y * h)
                screen_x = np.interp(x, [0, w], [0, screen_w])
                screen_y = np.interp(y, [0, h], [0, screen_h])
                # Apply cursor sensitivity
                current_x, current_y = pyautogui.position()
                new_x = current_x + (screen_x - current_x) * cursor_sensitivity
                new_y = current_y + (screen_y - current_y) * cursor_sensitivity
                pyautogui.moveTo(new_x, new_y)

        
            cv2.circle(frame, (int(face_landmarks.landmark[1].x * w), int(face_landmarks.landmark[1].y * h)), 5, (0, 255, 0), -1)

    # Blink detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
            if blink_counter >= BLINK_FRAMES and not click_triggered:
                current_time = time.time()
                if current_time - last_click_time > 0.5:
                    screen_x, screen_y = pyautogui.position()  # Or wherever the current mouse is
                    click(screen_x, screen_y)

                    click_triggered = True
                    last_click_time = current_time
        else:
            blink_counter = 0
            click_triggered = False

    # Scroll with mouth
    for face in faces:
        landmarks = predictor(gray, face)
        shape = np.array([(p.x, p.y) for p in landmarks.parts()])
        mouth_open = is_mouth_open(shape)
    
        if current_profile != "presentation":
            if previous_mouth_top is not None and previous_mouth_bottom is not None:
                mouth_movement = (shape[66][1] - previous_mouth_bottom) * scroll_sensitivity
                if mouth_open:
                    pyautogui.scroll(int(mouth_movement * scroll_speed * scroll_multiplier))
        previous_mouth_top = shape[62][1]
        previous_mouth_bottom = shape[66][1]
    # Visual feedback for dragging
    if dragging:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(frame, "Dragging...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)


    cv2.imshow("Integrated Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
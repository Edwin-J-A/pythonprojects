import speech_recognition as sr
import pyautogui  # For simulating keyboard actions
import pyttsx3  # For text-to-speech
import cv2  # For video feed
import threading  # For handling video feed in a separate thread
import platform  # For OS detection

# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()
listening = False  # Flag to check if command mode is active
video_paused = False  # Flag to check if video is paused
stop_program = False  # Flag to stop the program

# Detect OS
ios_system = platform.system()

# Function to speak responses
def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()

# Function to start listening for commands after detecting "Hey Bro"
def listen_for_command():
    global listening
    with sr.Microphone() as source:
        print("Say 'Hey Bro' to activate voice commands...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Audio detected: {command}")

                # Activate voice commands if "Hey Bro" is detected
                if "hey bro" in command:
                    speak("Voice command mode activated. I'm listening.")
                    listening = True
                    listen_for_voice_commands()  # Start listening for commands
                    break
            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

# Function to listen for commands after "Hey Bro"
def listen_for_voice_commands():
    global listening, video_paused, stop_program
    with sr.Microphone() as source:
        print("Listening for commands...")
        recognizer.adjust_for_ambient_noise(source)

        while listening:
            try:
                audio = recognizer.listen(source)
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")

                # Perform actions based on command
                if "zoom in" in command:
                    speak("Zooming in...")
                    if ios_system == "Windows":
                        pyautogui.hotkey("ctrl", "+")  # Zoom in for Windows (Ctrl + +)
                    elif ios_system == "Darwin":  # macOS
                        pyautogui.hotkey("command", "+")  # Zoom in for macOS (Command + +)
                    else: 
                        speak("Zoom in command is not supported on this OS.")
                elif "zoom out" in command:
                    speak("Zooming out...")
                    if ios_system == "Windows":
                        pyautogui.hotkey("ctrl", "-")  # Zoom out for Windows (Ctrl + -)
                    elif ios_system == "Darwin":  # macOS
                        pyautogui.hotkey("command", "-")  # Zoom out for macOS (Command + -)
                    else: 
                        speak("Zoom out command is not supported on this OS.")
                elif "hold" in command:
                    speak("Holding video feed...")
                    video_paused = True
                elif "resume" in command:
                    speak("Resuming video feed...")
                    video_paused = False
                elif "stop" in command:
                    speak("Stopping the program. Goodbye!")
                    stop_program = True
                    listening = False  # Stop listening
                    break
                else:
                    speak("Command not recognized. Please try again.")

            except sr.UnknownValueError:
                print("Sorry, I didn't understand that.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

# Video feed function
def video_feed():
    global video_paused, stop_program
    cap = cv2.VideoCapture(0)
    while not stop_program:
        if not video_paused:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Video Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_program = True
            break
    cap.release()
    cv2.destroyAllWindows()

# Start video feed and listen for "Hey Bro"
video_thread = threading.Thread(target=video_feed)
video_thread.start()
listen_for_command()

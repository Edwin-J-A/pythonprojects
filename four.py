#scrolling using mouth movement

import cv2
import dlib
import time
import pyautogui
import numpy as np

# Initialize variables
scrolling = False
scroll_locked = False
mouth_open_duration = 0
MAX_MOUTH_OPEN_DURATION = 5  # seconds before locking the scroll if mouth is open
scroll_speed = 20  # Scroll speed when mouth is open
scroll_direction = 0  # 1 for scrolling down, -1 for scrolling up, 0 for no scroll

# Load dlib's pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib website

# Function to detect if mouth is open and scroll direction
def is_mouth_open(shape):
    # Mouth points: 48-67 (using dlib's 68-point model)
    top_lip = shape[62][1]
    bottom_lip = shape[66][1]
    # Check vertical distance between top and bottom lip
    return bottom_lip - top_lip > 10  # You can adjust the threshold

def determine_scroll_direction(shape, previous_mouth_top, previous_mouth_bottom):
    # Determine the scroll direction based on mouth movement.
    # Compare the current and previous mouth positions.
    mouth_top = shape[62][1]
    mouth_bottom = shape[66][1]

    if previous_mouth_top is not None and previous_mouth_bottom is not None:
        # Check if the mouth opened downward or upward based on the difference
        if mouth_bottom > previous_mouth_bottom:
            return 1  # Scroll down
        elif mouth_bottom < previous_mouth_bottom:
            return -1  # Scroll up
    return 0  # No scroll if the mouth is not significantly moving

# Start webcam to capture face
cap = cv2.VideoCapture(0)



# Time-based mouth open check
last_mouth_open_time = None
previous_mouth_top = None
previous_mouth_bottom = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for better visualization
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for better performance with dlib
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using dlib detector
    faces = detector(gray)
    
    # Assume only one face in the frame for simplicity
    for face in faces:
        # Get landmarks for the face
        landmarks = predictor(gray, face)
        # Convert landmarks to numpy array
        shape = np.array([(p.x, p.y) for p in landmarks.parts()])
        
        # Detect if mouth is open
        mouth_open = is_mouth_open(shape)

        if mouth_open:
            if not scrolling and not scroll_locked:
                scrolling = True
                print("Mouth opened, scrolling started.")
            
            # Track how long the mouth has been open
            if last_mouth_open_time is None:
                last_mouth_open_time = time.time()

            # Check if mouth has been open too long (locking scroll)
            if time.time() - last_mouth_open_time > MAX_MOUTH_OPEN_DURATION:
                scroll_locked = True
                print("Mouth has been open for too long. Scroll locked.")
            
            # Determine scroll direction based on mouth movement using NumPy for efficiency
            scroll_direction = determine_scroll_direction(shape, previous_mouth_top, previous_mouth_bottom)
        else:
            if scrolling:
                scrolling = False
                print("Mouth closed, scrolling stopped.")
            if scroll_locked:
                print("Scroll locked. Mouth closed.")
            last_mouth_open_time = None  # Reset if mouth is closed
            scroll_direction = 0  # Reset scroll direction if mouth is closed

        # Update previous mouth positions for next iteration
        previous_mouth_top = shape[62][1]
        previous_mouth_bottom = shape[66][1]

        # Handle scrolling using NumPy for more efficient operations
        if scrolling and not scroll_locked and scroll_direction != 0:
            # Use NumPy to calculate scrolling direction with a more efficient method
            pyautogui.scroll(scroll_direction * scroll_speed)

    # Show webcam feed
    cv2.imshow("Face Detection", frame)

    # Exit condition: Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

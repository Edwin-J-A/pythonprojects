#recognises only one face at a time and stops program if bad lighting is encountered

import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

def is_bad_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness < 50  # Adjust threshold as needed

while True:
    ret, frame = cam.read()
    
    # Flip the frame horizontally for better reflection (optional)
    frame = cv2.flip(frame, 1)

    if not ret or frame is None:
        print("Error: Could not read frame.")
        continue
    
    if is_bad_lighting(frame):
        print("Bad lighting detected! Stopping program.")
        break
    
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        # Process only the first detected face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape
        x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                              int(bboxC.width * w), int(bboxC.height * h))
        
        # Draw bounding box around the first detected face
        cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Single Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

import cv2
import torch
import pyautogui
import time
import mediapipe as mp

# Initialize webcam
cam = cv2.VideoCapture(0)

# Load YOLOv5 model for UI element detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='local')

# Initialize MediaPipe Face Mesh for nose tracking
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

NOSE_SNAP_THRESHOLD = 20  # Pixel distance to trigger snap-to-element

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face mesh for nose tracking
    results = face_mesh.process(rgb_frame)
    nose_x, nose_y = None, None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]  # Nose tip index
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)  # Draw nose tip indicator

    # Detect UI elements using YOLO
    yolo_results = model(frame)

    for detection in yolo_results.xyxy[0]:  # Iterate over detected objects
        x1, y1, x2, y2, conf, cls = detection.cpu().numpy()
        if conf > 0.6:  # Confidence threshold
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"UI Element", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # If nose is near the element, snap the cursor
            if nose_x is not None and abs(nose_x - center_x) < NOSE_SNAP_THRESHOLD and abs(nose_y - center_y) < NOSE_SNAP_THRESHOLD:
                pyautogui.moveTo(center_x, center_y)
                print("Cursor snapped to UI element")

    cv2.imshow("Snap-to-Elements", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

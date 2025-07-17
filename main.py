from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np

# Load YOLOv8m model
model = YOLO("yolov8m.pt")  # Automatically downloads if not present

# Open webcam
cap = cv2.VideoCapture(0)

# Buffer to smooth count
recent_counts = deque(maxlen=5)

# Parameters
CONF_THRESHOLD = 0.5
MIN_AREA = 5000  # Filter out small boxes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # Run tracking
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    frame = results.orig_img
    current_count = 0

    if results.boxes is not None and results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i, box in enumerate(boxes):
            if int(classes[i]) != 0:  # Only count person class
                continue
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_AREA:
                continue
            current_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Smooth the count
    recent_counts.append(current_count)
    smooth_count = round(np.mean(recent_counts))

    # Display count
    cv2.putText(frame, f'People in Frame: {smooth_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("YOLOv8m + ByteTrack Headcount", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

from flask import Flask, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2
from threading import Thread, Lock
import numpy as np
from collections import deque

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture(0)

recent_counts = deque(maxlen=5)
lock = Lock()
current_count = 0

def detect_people():
    global current_count
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))
        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=0.5, verbose=False)[0]
        count = 0

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()

            for i, box in enumerate(boxes):
                if int(classes[i]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) * (y2 - y1) < 5000:
                    continue
                count += 1

        recent_counts.append(count)
        smooth_count = round(np.mean(recent_counts))

        with lock:
            current_count = smooth_count

@app.route("/headcount")
def get_headcount():
    with lock:
        return jsonify({"count": current_count})

if __name__ == "__main__":
    Thread(target=detect_people, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)

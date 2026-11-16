"""Run the shipped model on a video and write detections.csv (frame,class,confidence,x1,y1,x2,y2).

Run from the repo folder:  python tools/make_detections.py [video] [out.csv]
Same format as the app's live detections (app/core/detect.py): frames start at 1, goals are left out, confidence >= 0.25.
"""
import csv
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "best.pt"
IMGSZ = 960            # the champion was trained and validated at 960 (E2)
CONF = 0.25
DEVICE = 0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

video = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "cycle-ball.mp4"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "models" / "detections.csv"

model = YOLO(WEIGHTS)
keep = [i for i, n in model.names.items() if n != "goal"]
cap = cv2.VideoCapture(str(video))
frame = rows = 0
with out.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frame += 1
        boxes = model.predict(image, imgsz=IMGSZ, conf=CONF, classes=keep, device=DEVICE, verbose=False)[0].boxes.cpu()
        for cls, confidence, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            writer.writerow([frame, model.names[int(cls)], f"{confidence:.3f}", *(f"{v:.1f}" for v in xyxy.tolist())])
            rows += 1
cap.release()
print(f"{frame} frames, {rows} boxes: {out}")

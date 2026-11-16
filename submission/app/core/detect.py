"""Run the fine-tuned model on a clip: one table of boxes, with a team (blue / white) on every player box.

Same columns as models/detections.csv, plus `team`. One pass over the video does both jobs:
the model finds the boxes, and for every player box we also describe the shirt colour.
When the pass is done, K-Means splits the player boxes into the two teams.
(This is notebook 06's method, moved here so the app can run it on any clip.)
"""
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from core.config import ROOT

WEIGHTS = ROOT / "models" / "best.pt"
IMGSZ = 960      # the champion was trained and validated at 960 (experiment E2)
CONF = 0.25      # lowest confidence kept; the sliders in the app filter upwards from here
SEED = 0         # K-Means starts from random centres; a fixed seed gives the same teams every run

# Shirt = 10-50% of the box height (below the head, above the shorts), without 20% on each side.
SHIRT_TOP, SHIRT_BOTTOM, SHIRT_SIDE = 0.10, 0.50, 0.20
DARK_V = 40                  # darker than this: shadow, no usable colour
WHITE_S, WHITE_V = 60, 130   # white = low saturation and bright (loose: arena light makes white look yellowish)
HUE_BINS = 12                # hue histogram with 12 bins of 15 OpenCV hue units
BLUE_BINS = slice(6, 9)      # bins 6-8 are blue; used only to decide which cluster is "blue"


def pick_device():
    if torch.cuda.is_available():
        return 0
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_model():
    return YOLO(WEIGHTS)


def shirt_histogram(hsv, box):
    """Share of shirt pixels per hue bin, then white, then dark (HUE_BINS + 2 numbers that sum to 1)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pixels = hsv[int(y1 + SHIRT_TOP * h):int(y1 + SHIRT_BOTTOM * h), int(x1 + SHIRT_SIDE * w):int(x2 - SHIRT_SIDE * w)]
    pixels = pixels.reshape(-1, 3).astype(int)
    if len(pixels) == 0:
        return np.zeros(HUE_BINS + 2)
    hue, sat, val = pixels.T
    dark = val < DARK_V
    white = ~dark & (sat < WHITE_S) & (val > WHITE_V)
    hues = np.bincount(hue[~dark & ~white] * HUE_BINS // 180, minlength=HUE_BINS)
    return np.r_[hues, white.sum(), dark.sum()] / len(pixels)


def split_teams(histograms):
    """Two K-Means clusters on the square-rooted histograms. The cluster with more blue is 'blue', the other 'white'."""
    features = np.sqrt(np.float32(histograms))
    cv2.setRNGSeed(SEED)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    cluster = cv2.kmeans(features, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)[1].ravel()
    blue_share = np.asarray(histograms)[:, BLUE_BINS].sum(axis=1)
    blue_cluster = 0 if blue_share[cluster == 0].mean() >= blue_share[cluster == 1].mean() else 1
    return np.where(cluster == blue_cluster, "blue", "white")


def detect_video(path, progress=None):
    """Boxes for every frame of the clip. `progress(done, total)` is called after each frame.

    Returns columns frame (starts at 1), class, confidence, x1, y1, x2, y2, team (player boxes only).
    All four classes are kept (player, bike, ball, goal). df.attrs["n_frames"] holds the clip's frame count, because the last frames may have no boxes.
    """
    model = load_model()
    device = pick_device()
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, histograms = [], []
    frame = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frame += 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        boxes = model.predict(image, imgsz=IMGSZ, conf=CONF, device=device, verbose=False)[0].boxes.cpu()
        for cls, confidence, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            name, box = model.names[int(cls)], xyxy.tolist()
            rows.append((frame, name, float(confidence), *box))
            if name == "player":
                histograms.append(shirt_histogram(hsv, box))
        if progress:
            progress(frame, total)
    capture.release()

    det = pd.DataFrame(rows, columns=["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
    det["team"] = None
    is_player = (det["class"] == "player").to_numpy()
    if is_player.sum() >= 2:                      # K-Means needs at least two boxes
        det.loc[is_player, "team"] = split_teams(histograms)
    det.attrs["n_frames"] = frame
    return det

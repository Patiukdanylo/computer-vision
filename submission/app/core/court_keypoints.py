"""Automated keypoints: find the 14 court landmarks with our pose model and fit the pixel -> metre homography.

This replaces the hand-clicked points (data/court_keypoints.json), which only fit the sample clip's camera.

    keypoints_in_frame(model, frame)   -> pixel position + confidence of each landmark in one frame
    homography_from_keypoints(xy, conf) -> H for that frame (RANSAC over the confident landmarks)
    homography_from_video(path)        -> one H for a static camera: per landmark, the median over sampled frames

The model was trained and evaluated in notebook 08. Its keypoints (court layout) or classes (landmarks layout) are in
the order of the landmarks in data/court_reference.json, which also holds each landmark's court position in metres.

GenAI: written with Claude Code (Opus 5.5) on 2026-11-13.
"""
import json
from functools import lru_cache

import cv2
import numpy as np

from core.config import ROOT

MODEL_PATH = ROOT / "models" / "court_keypoints.pt"
REFERENCE_FILE = ROOT / "data" / "court_reference.json"
MIN_CONFIDENCE = 0.5      # landmarks below this are not used for the homography
RANSAC_METRES = 0.3       # a landmark further than this from where the fit puts it is an outlier
MIN_KEYPOINTS = 4         # a homography has 8 unknowns, each landmark gives 2 equations


@lru_cache(maxsize=1)
def court_template():
    """Landmark names and their court positions in metres, (14, 2), in the model's keypoint order."""
    ref = json.loads(REFERENCE_FILE.read_text(encoding="utf-8"))
    names = list(ref["keypoints"])
    return names, np.float32([ref["keypoints"][k]["m"] for k in names])


@lru_cache(maxsize=1)
def load_model(path=MODEL_PATH):
    from ultralytics import YOLO          # imported here so the app starts without loading torch
    return YOLO(str(path))


def keypoints_in_frame(model, frame, imgsz=640):
    """Pixel positions (14, 2) and confidences (14,) of the landmarks in one BGR frame. Works for both layouts:
    - court (K1): one `court` object with 14 keypoints; we keep the most confident court. Confidence = the keypoint's.
    - landmarks (K2): one object per landmark (class = landmark index, 1 keypoint); per class we keep the most
      confident box. Confidence = the box's.
    A landmark that isn't found gets confidence 0."""
    result = model.predict(frame, imgsz=imgsz, verbose=False)[0]
    n = len(court_template()[0])
    xy, conf = np.zeros((n, 2), np.float32), np.zeros(n, np.float32)
    if result.keypoints is None or len(result.boxes) == 0:
        return xy, conf
    kpts = result.keypoints.xy.cpu().numpy()                      # (objects, keypoints per object, 2)
    box_conf = result.boxes.conf.cpu().numpy()
    if kpts.shape[1] == n:                                        # court layout
        best = int(box_conf.argmax())
        kpt_conf = result.keypoints.conf
        return kpts[best].astype(np.float32), (kpt_conf[best].cpu().numpy() if kpt_conf is not None
                                                else np.ones(n, np.float32)).astype(np.float32)
    classes = result.boxes.cls.cpu().numpy().astype(int)          # landmarks layout
    for i in np.argsort(box_conf):                                # most confident last, so it wins
        xy[classes[i]], conf[classes[i]] = kpts[i, 0], box_conf[i]
    return xy, conf


def homography_from_keypoints(xy, conf, min_confidence=MIN_CONFIDENCE, ransac_metres=RANSAC_METRES):
    """H (pixels -> metres) from the confident landmarks, and a mask of the landmarks RANSAC kept.
    Returns (None, mask) when fewer than 4 landmarks are usable."""
    _, metres = court_template()
    use = conf >= min_confidence
    inliers = np.zeros(len(conf), dtype=bool)
    if use.sum() < MIN_KEYPOINTS:
        return None, inliers
    H, mask = cv2.findHomography(xy[use], metres[use], cv2.RANSAC, ransac_metres)
    if H is None:
        return None, inliers
    inliers[np.flatnonzero(use)[mask.ravel() > 0]] = True
    return H, inliers


def sample_frames(video_path, n=30, frame_range=None):
    """n frames spread evenly over the clip, or over frame_range = (first, last), 1-based and inclusive.
    Returns a list of (frame number, BGR frame)."""
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first, last = frame_range or (1, total)
    wanted = set(np.linspace(first - 1, last - 1, min(n, last - first + 1)).round().astype(int))
    frames, i = [], 0
    while wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if i in wanted:
            frames.append((i + 1, frame))
            wanted.discard(i)
        i += 1
    capture.release()
    return frames


def homography_from_video(video_path, n_frames=30, model=None, frame_range=None):
    """One H for a static camera. A rider covers a different landmark in every frame, so per landmark we take the
    median position over the frames where the model is confident, then fit H once. Returns (H, details)."""
    model = model or load_model()
    per_frame = [keypoints_in_frame(model, frame) for _, frame in sample_frames(video_path, n_frames, frame_range)]
    xy = np.stack([p[0] for p in per_frame])            # (frames, 14, 2)
    conf = np.stack([p[1] for p in per_frame])          # (frames, 14)
    names, _ = court_template()
    median_xy = np.zeros((len(names), 2), np.float32)
    support = (conf >= MIN_CONFIDENCE).sum(0)           # frames in which each landmark was found
    for k in range(len(names)):
        good = conf[:, k] >= MIN_CONFIDENCE
        if good.any():
            median_xy[k] = np.median(xy[good, k], axis=0)
    # a landmark counts if it was found in at least a third of the frames
    found = (support >= max(1, len(per_frame) // 3)).astype(np.float32)
    H, inliers = homography_from_keypoints(median_xy, found)
    details = {"names": names, "xy": median_xy, "frames_found": support, "n_frames": len(per_frame),
               "inliers": inliers, "spread_px": _spread(xy, conf, median_xy)}
    return H, details


def _spread(xy, conf, median_xy):
    """Median distance (px) of each landmark's per-frame positions from its median: how much the model jitters."""
    out = np.full(len(median_xy), np.nan, np.float32)
    for k in range(len(median_xy)):
        good = conf[:, k] >= MIN_CONFIDENCE
        if good.any():
            out[k] = np.median(np.linalg.norm(xy[good, k] - median_xy[k], axis=1))
    return out

"""Pixel -> court-metre mapping for the fixed sample camera.

The camera never moves, so one homography H is computed once and reused for every frame. Two ways to get it
(HOMOGRAPHY_SOURCE in core/config.py):
- "keypoints": found automatically by our court keypoint model. tools/auto_homography.py
  runs the model on the clip and saves H to models/court_homography.json.
- "manual": the MVP version, from the points clicked by hand on frame 1 (data/court_keypoints.json).
If the keypoint file is missing, the manual points are used.
"""
import json
from functools import lru_cache

import cv2
import numpy as np

from core.config import COURT_LENGTH, COURT_WIDTH, HOMOGRAPHY_SOURCE, ROOT

KEYPOINTS_FILE = ROOT / "data" / "court_keypoints.json"
AUTO_HOMOGRAPHY_FILE = ROOT / "models" / "court_homography.json"


def _intersect(p1, p2, q1, q2):
    """Where the line through p1,p2 crosses the line through q1,q2 (pixels)."""
    a = np.cross([*p1, 1], [*p2, 1])
    b = np.cross([*q1, 1], [*q2, 1])
    x = np.cross(a, b)
    return x[:2] / x[2]


@lru_cache(maxsize=1)
def court_correspondences():
    """Hand-clicked pixel points and their court positions in metres, as two (N, 2) float32 arrays.

    Both near corners are off-frame, so each is found where an end line (far corner ->
    low end-board point) crosses the near sideline (the two clicked sideline points).
    """
    k = json.loads(KEYPOINTS_FILE.read_text())
    side = (k["near_sideline_left"], k["near_sideline_right"])
    near_left = _intersect(k["far_left_corner"], k["left_end_low"], *side)
    near_right = _intersect(k["far_right_corner"], k["right_end_low"], *side)
    pixels = np.float32([k["far_left_corner"], k["far_right_corner"], near_right, near_left, k["centre_spot"]])
    metres = np.float32([[0, 0], [COURT_LENGTH, 0], [COURT_LENGTH, COURT_WIDTH], [0, COURT_WIDTH],
                         [COURT_LENGTH / 2, COURT_WIDTH / 2]])
    return pixels, metres


@lru_cache(maxsize=1)
def manual_homography():
    """3x3 matrix H from the hand-clicked points, least-squares fit over all 5 point pairs (method 0 = plain)."""
    pixels, metres = court_correspondences()
    H, _ = cv2.findHomography(pixels, metres, 0)
    return H


@lru_cache(maxsize=1)
def keypoint_homography():
    """3x3 matrix H found by the court keypoint model, or None if tools/auto_homography.py hasn't been run."""
    if not AUTO_HOMOGRAPHY_FILE.exists():
        return None
    return np.array(json.loads(AUTO_HOMOGRAPHY_FILE.read_text())["H_px_to_m"], dtype=np.float64)


def get_homography(source=None):
    """H for the given source ("keypoints" or "manual"; default from config). Keypoints fall back to manual."""
    if (source or HOMOGRAPHY_SOURCE) == "keypoints":
        H = keypoint_homography()
        if H is not None:
            return H
    return manual_homography()


def to_court(points_px, source=None):
    """Project pixel points, shape (N, 2), onto the court. Returns (N, 2) metres (x along length, y along width)."""
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:                      # OpenCV returns None for an empty input
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(pts, get_homography(source)).reshape(-1, 2)


def reprojection_error():
    """Distance in metres between each of the 5 hand-clicked points and where the manual H sends it."""
    pixels, metres = court_correspondences()
    return np.linalg.norm(to_court(pixels, source="manual") - metres, axis=1)

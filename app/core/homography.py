"""Pixel -> court-metre mapping for the fixed sample camera.

The camera never moves, so one homography H is computed once from the hand-picked
points in data/court_keypoints.json and reused for every frame.
"""
import json
from functools import lru_cache

import cv2
import numpy as np

from core.config import COURT_LENGTH, COURT_WIDTH, ROOT

KEYPOINTS_FILE = ROOT / "data" / "court_keypoints.json"


def _intersect(p1, p2, q1, q2):
    """Where the line through p1,p2 crosses the line through q1,q2 (pixels)."""
    a = np.cross([*p1, 1], [*p2, 1])
    b = np.cross([*q1, 1], [*q2, 1])
    x = np.cross(a, b)
    return x[:2] / x[2]


@lru_cache(maxsize=1)
def court_correspondences():
    """Pixel points and their court positions in metres, as two (N, 2) float32 arrays.

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
def get_homography():
    """3x3 matrix H, least-squares fit over all 5 point pairs (method 0 = plain)."""
    pixels, metres = court_correspondences()
    H, _ = cv2.findHomography(pixels, metres, 0)
    return H


def to_court(points_px):
    """Project pixel points, shape (N, 2), onto the court. Returns (N, 2) metres (x along length, y along width)."""
    pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) == 0:                      # OpenCV returns None for an empty input
        return np.empty((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(pts, get_homography()).reshape(-1, 2)


def reprojection_error():
    """Distance in metres between each of our 5 points and where H sends it."""
    pixels, metres = court_correspondences()
    return np.linalg.norm(to_court(pixels) - metres, axis=1)

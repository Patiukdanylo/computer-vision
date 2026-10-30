"""Paths and settings shared across the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLIP = ROOT / "data" / "cycle-ball.mp4"
UPLOADS_DIR = ROOT / "outputs" / "uploads"
STATIC_DIR = ROOT / "app" / "static"

# International cycle ball court, in metres
COURT_LENGTH = 14.0
COURT_WIDTH = 11.0

# Classes in our labelled dataset, and the starting point for each confidence slider (Ultralytics' default)
DETECTION_CLASSES = ("player", "ball", "bike", "goal")
DEFAULT_CONFIDENCE = 0.25

"""Paths and settings shared across the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CLIP = ROOT / "data" / "cycle-ball.mp4"
UPLOADS_DIR = ROOT / "outputs" / "uploads"

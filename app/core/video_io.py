"""Saving and removing clips on disk."""
import shutil
import uuid
from pathlib import Path

import cv2

from core.config import STATIC_DIR, UPLOADS_DIR


def save_upload(data: bytes) -> Path:
    """Write uploaded bytes to disk, because OpenCV needs a real file path."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOADS_DIR / f"{uuid.uuid4().hex}.mp4"
    path.write_bytes(data)
    return path


def delete_upload(path: Path) -> None:
    """Remove a saved upload. Anything outside the uploads folder, like the sample clip, is left alone."""
    if path.parent == UPLOADS_DIR:
        path.unlink(missing_ok=True)


def get_duration(path: Path) -> float:
    """Clip length in seconds. Raises ValueError if the file is not a readable video."""
    capture = cv2.VideoCapture(str(path))
    frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if frames <= 0 or fps <= 0:
        raise ValueError(f"Not a readable video: {path.name}")
    return frames / fps


def publish_clip(path: Path) -> str:
    """Copy a clip into the static folder so the browser player can load it, and return its file name."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.mp4"
    shutil.copy(path, STATIC_DIR / name)
    return name


def unpublish_clip(name: str) -> None:
    (STATIC_DIR / name).unlink(missing_ok=True)

"""Saving and removing clips on disk."""
import shutil
import subprocess
import uuid
from pathlib import Path

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
    """Clip length in seconds. Raises if the file is not a readable video."""
    command = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    return float(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def publish_clip(path: Path) -> str:
    """Copy a clip into the static folder so the browser player can load it, and return its file name."""
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.mp4"
    shutil.copy(path, STATIC_DIR / name)
    return name


def unpublish_clip(name: str) -> None:
    (STATIC_DIR / name).unlink(missing_ok=True)

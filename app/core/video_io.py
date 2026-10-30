"""Saving and removing clips on disk."""
import uuid
from pathlib import Path

from core.config import UPLOADS_DIR


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

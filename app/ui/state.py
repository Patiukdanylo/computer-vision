"""All session_state access in one place. Streamlit reruns the script on every interaction, and this survives it."""
from pathlib import Path

import streamlit as st

from core.video_io import delete_upload


def get_clip() -> tuple[Path, str] | None:
    return st.session_state.get("clip")


def set_clip(path: Path, name: str) -> None:
    st.session_state["clip"] = (path, name)


def clear_clip() -> None:
    clip = st.session_state.pop("clip", None)
    if clip:
        delete_upload(clip[0])

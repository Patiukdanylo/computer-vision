"""All session_state access in one place. Streamlit reruns the script on every interaction, and this survives it."""
from pathlib import Path

import streamlit as st

from core.video_io import delete_upload, unpublish_clip


def get_clip() -> tuple[Path, str] | None:
    return st.session_state.get("clip")


def set_clip(path: Path, name: str) -> None:
    st.session_state["clip"] = (path, name)


def get_run() -> dict | None:
    return st.session_state.get("run")


def set_run(run: dict) -> None:
    st.session_state["run"] = run


def clear_clip() -> None:
    run = st.session_state.pop("run", None)
    if run:
        unpublish_clip(run["clip_name"])
    clip = st.session_state.pop("clip", None)
    if clip:
        delete_upload(clip[0])

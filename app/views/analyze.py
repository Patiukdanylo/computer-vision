"""Upload a clip, process it, and show the video beside the 2D court."""
import subprocess

import streamlit as st

from core.config import SAMPLE_CLIP
from core.demo_data import HZ, make_demo_tracks
from core.video_io import get_duration, publish_clip, save_upload
from ui import state
from ui.match_player import render_match_player


def on_upload():
    uploaded = st.session_state["uploader"]
    if uploaded:
        state.set_clip(save_upload(uploaded.getvalue()), uploaded.name)


def upload_screen():
    with st.container(key="dropzone"):
        st.file_uploader("Upload a clip", type="mp4", key="uploader", on_change=on_upload, label_visibility="collapsed")
    with st.container(horizontal_alignment="center"):
        if st.button("Or try the sample clip", type="tertiary", disabled=not SAMPLE_CLIP.exists()):
            state.set_clip(SAMPLE_CLIP, SAMPLE_CLIP.name)
            st.rerun()


def clip_screen(path, name):
    st.video(path)
    st.caption(f"{name} · {path.stat().st_size / 1e6:.1f} MB")
    with st.container(horizontal=True):
        process = st.button("Process clip", type="primary", help="Shows demo data until the backend exists")
        st.button("Choose another clip", on_click=state.clear_clip)
    if process:
        try:
            tracks = make_demo_tracks(get_duration(path))
        except (subprocess.CalledProcessError, ValueError):
            st.error("This file could not be read as a video.")
        else:
            state.set_run({"clip_name": publish_clip(path), "tracks": tracks})
            st.rerun()


def results_screen(run):
    st.caption(":orange[Demo data: the map dots are fake, and the video has no detections yet.]")
    render_match_player(f"/app/static/{run['clip_name']}", run["tracks"], HZ)
    st.button("Process another clip", on_click=state.clear_clip)


run = state.get_run()
clip = state.get_clip()

st.title("Cycle Ball Tactical Map")
if not run:
    st.caption("From a static-camera clip to a synchronized 2D court.")

if run:
    results_screen(run)
elif clip:
    clip_screen(*clip)
else:
    upload_screen()

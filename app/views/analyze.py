"""Upload a clip, process it, and show the video beside the 2D court."""
import filecmp

import streamlit as st

from core.config import DETECTION_CLASSES, SAMPLE_CLIP
from core.tracks import build_boxes, build_tracks
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


def confidence_sliders() -> dict[str, float]:
    saved = state.get_confidences()
    st.markdown("Confidence thresholds")
    columns = st.columns(len(DETECTION_CLASSES))
    chosen = {
        name: column.slider(name.capitalize(), 0.05, 0.95, saved[name], 0.05, key=f"confidence_{name}")
        for name, column in zip(DETECTION_CLASSES, columns)
    }
    state.set_confidences(chosen)
    st.caption("A detection below its class threshold is dropped: lower finds more, but adds false detections. "
               "Bike, player and ball sliders apply to the sample clip; goals are not detected yet, and nothing "
               "below 0.25 exists because the saved detections start there.")
    return chosen


def clip_screen(path, name):
    st.video(path)
    st.caption(f"{name} · {path.stat().st_size / 1e6:.1f} MB")
    confidences = confidence_sliders()
    with st.container(horizontal=True):
        process = st.button("Process clip", type="primary", help="Maps the saved detections of the sample clip onto the court")
        st.button("Choose another clip", on_click=state.clear_clip)
    if process:
        if path != SAMPLE_CLIP and not filecmp.cmp(path, SAMPLE_CLIP, shallow=False):   # an upload of the sample file counts too
            st.error("Only the sample clip (data/cycle-ball.mp4) can be mapped for now: the court points were picked for its "
                     "camera angle, and its detections are precomputed.")
        else:
            tracks, hz = build_tracks(get_duration(path), confidences)
            if len(tracks) < 5:    # the map needs 2 riders per team and the ball
                st.error("Too few detections at these thresholds: the map needs both teams and the ball. Lower the sliders.")
                return
            state.set_run({"clip_name": publish_clip(path), "tracks": tracks, "hz": hz, "confidences": confidences,
                           "boxes": build_boxes(confidences)})
            st.rerun()


def results_screen(run):
    st.caption("Boxes on the video are the raw detections. Dots on the map are the two teams (colour clustering) and the "
               "ball, with gaps filled by interpolation.")
    render_match_player(f"/app/static/{run['clip_name']}", run["tracks"], run["hz"], run["boxes"])
    thresholds = " · ".join(f"{name.capitalize()} {value:.2f}" for name, value in run["confidences"].items())
    st.caption(f"Confidence thresholds: {thresholds}")
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

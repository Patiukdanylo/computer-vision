"""Upload a clip, process it, and show the video beside the 2D court."""
import numpy as np
import streamlit as st

from core.config import COURT_LENGTH, COURT_WIDTH, DETECTION_CLASSES, SAMPLE_CLIP
from core.detect import detect_video
from core.court_keypoints import homography_from_video
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
               "Nothing below 0.25 exists because the model only keeps boxes from 0.25 up.")
    return chosen


OFF_COURT_MAX = 0.2      # more than this share of rider points outside the court means the court corners are wrong
OFF_COURT_MARGIN_M = 3   # a point may sit this far outside the lines (riders leave the court, the boards are close)


def get_detections(path):
    """Run the model once per clip (the 25 s sample: ~20 s on an Apple GPU, ~1 min on a CPU). Moving a slider then reuses the result."""
    cached = st.session_state.get("detections")
    if cached and cached[0] == path:
        return cached[1]
    bar = st.progress(0.0, text="Finding riders and the ball...")
    det = detect_video(path, lambda done, total: bar.progress(min(done / total, 1.0), text=f"Frame {done} of {total}"))
    bar.empty()
    st.session_state["detections"] = (path, det)
    return det


def get_homography(path):
    """The clip's own pixel -> court matrix, found once with the court keypoint model. None for the sample (its saved one is used)."""
    if path == SAMPLE_CLIP:
        return None
    cached = st.session_state.get("clip_homography")
    if cached and cached[0] == path:
        return cached[1]
    with st.spinner("Finding the court lines..."):
        H, _ = homography_from_video(path)
    st.session_state["clip_homography"] = (path, H)
    return H


def share_off_court(tracks):
    """Share of rider-track points more than OFF_COURT_MARGIN_M outside the 14 x 11 m court."""
    points = np.array([p for t in tracks if t["kind"] != "ball" for p in t["points"]])
    if len(points) == 0:
        return 1.0
    m = OFF_COURT_MARGIN_M
    outside = (points[:, 0] < -m) | (points[:, 0] > COURT_LENGTH + m) | (points[:, 1] < -m) | (points[:, 1] > COURT_WIDTH + m)
    return float(outside.mean())


def clip_screen(path, name):
    st.video(path)
    st.caption(f"{name} · {path.stat().st_size / 1e6:.1f} MB")
    confidences = confidence_sliders()
    with st.container(horizontal=True):
        process = st.button("Process clip", type="primary", help="Runs the model on every frame and maps the riders onto the court")
        st.button("Choose another clip", on_click=state.clear_clip)
    if process:
        try:
            duration = get_duration(path)
        except ValueError:
            st.error("This file is not a readable video. Choose another mp4 clip.")
            return
        H = get_homography(path)
        if path != SAMPLE_CLIP and H is None:
            st.error("Could not find the court in this clip: too few court landmarks were visible. The camera must show "
                     "the whole court from a fixed position.")
            return
        det = get_detections(path)
        tracks, hz = build_tracks(duration, confidences, det, H)
        if len(tracks) < 5:    # the map needs 2 riders per team and the ball
            st.error("Too few detections at these thresholds: the map needs both teams and the ball. Lower the sliders.")
            return
        if share_off_court(tracks) > OFF_COURT_MAX:
            st.error("Most riders land outside the court, so the court lines were probably not found correctly in this clip. "
                     "Try a clip where the whole floor is visible.")
            return
        state.set_run({"clip_name": publish_clip(path), "tracks": tracks, "hz": hz, "confidences": confidences,
                       "boxes": build_boxes(confidences, det)})
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

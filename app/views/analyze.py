"""Upload a clip, process it, and show the video beside the 2D court."""
import streamlit as st

from core.config import SAMPLE_CLIP
from core.video_io import save_upload
from ui import state


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
        st.button("Process clip", type="primary", disabled=True, help="Available once the pipeline is built")
        st.button("Choose another clip", on_click=state.clear_clip)


st.title("Cycle Ball Tactical Map")
st.caption("From a static-camera clip to a synchronized 2D court.")

clip = state.get_clip()
if clip:
    clip_screen(*clip)
else:
    upload_screen()

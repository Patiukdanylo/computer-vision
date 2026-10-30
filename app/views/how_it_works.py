"""The pipeline explained, from raw video to minimap. Numbers here are measured by the team, not demo data."""
import streamlit as st

from ui.homography_diagram import render_homography_diagram

PIPELINE_STEPS = [
    ("Video", "One static camera. Every frame is kept, so video and map stay in sync.", "Working"),
    ("Detection", "Fine-tuned YOLO finds players, bikes, the ball and goals.", "Not connected"),
    ("Floor point", "The bottom-centre of the bike box, where the wheels touch the floor.", "Not connected"),
    ("Homography", "A 3x3 matrix turns that floor pixel into court metres.", "Not connected"),
    ("2D map", "The video's own time decides which positions to show.", "Demo data"),
]

# Baseline E0a and fine-tune E1 from modeling/README.md (branch feat/base-model), both at 640 px.
# Measured on an older split: update these once they are re-run on the split in configs/dataset.yaml.
MODEL_RESULTS = [("Player", 0.176, 0.935), ("Ball", 0.727, 0.995), ("Bike", 0.673, 0.818)]

LIMITS = [
    ("Flat floor only", "An airborne ball lands in the wrong map spot, because it isn't on the floor."),
    ("One camera view", "A clip from another angle needs new court points."),
    ("Bikes get missed", "Bike recall is 0.67, so the rider's box is the fallback, and it ends at the pedals."),
    ("Nearest isn't possession", "Being closest to the ball doesn't mean having it."),
]

st.title("How it works")
st.caption(":orange[The app shows demo data for now: detection and homography aren't connected yet.]")

st.subheader("From video to map")
for column, (number, (name, description, status)) in zip(st.columns(5), enumerate(PIPELINE_STEPS, start=1)):
    column.markdown(f"**{number}. {name}**")
    column.caption(description)
    column.markdown(f"<small>{status}</small>", unsafe_allow_html=True)

st.subheader("The data")
columns = st.columns(4)
columns[0].metric("Labelled frames", "73")
columns[1].metric("Train / validation", "48 / 25")
columns[2].metric("Classes", "4")
columns[3].metric("Frames in the clip", "1,170")
st.caption("The split is by time, not random: neighbouring frames are near-duplicates, so a random split "
           "would put the same moment in both sets and inflate the score.")

st.subheader("The model")
st.markdown("AP50 on validation, after fine-tuning YOLO11s, compared with the same model before it.")
for column, (name, before, after) in zip(st.columns(3), MODEL_RESULTS):
    column.metric(name, f"{after:.3f}", f"{after - before:+.3f} vs pretrained")
st.caption(":orange[Early results from an older split with 13 validation balls, not the 48 / 25 split above "
           "(25 balls). They need re-running on the current split.] The pretrained ball score flatters it: "
           "it found balls only at confidence 0.06 or lower, which is useless at a normal threshold.")

st.subheader("Homography: from camera view to top-down")
st.markdown("The camera sees the court at an angle, so its rectangle looks like a trapezoid. If we know where at "
            "least 4 court points are in both views, OpenCV can calculate the matrix that undoes the angle. "
            "The camera doesn't move, so this is done once.")
render_homography_diagram()
st.caption("Illustration only: these aren't the real court points, which haven't been picked yet.")

st.subheader("Known limits")
for column, (name, description) in zip(st.columns(len(LIMITS)), LIMITS):
    column.markdown(f"**{name}**")
    column.caption(description)

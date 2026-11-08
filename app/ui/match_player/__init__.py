"""The video and the 2D court together, because they must share one clock to stay in sync."""
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent
_component = st.components.v2.component(
    "match_player",
    html=(_HERE / "player.html").read_text(),
    css=(_HERE / "player.css").read_text(),
    js=(_HERE / "player.js").read_text(),
)


def render_match_player(video_url: str, tracks: list[dict], hz: int) -> None:
    _component(data={"videoUrl": video_url, "tracks": tracks, "hz": hz}, key="match_player")

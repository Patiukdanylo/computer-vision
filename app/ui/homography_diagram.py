"""Illustration of a homography: the same four points and one rider in both views.

The rider's top-down position is calculated, not eyeballed: cv2.perspectiveTransform maps (232, 150) to (765, 182).
"""
import streamlit as st

_HTML = """
<svg viewBox="0 0 900 270" role="img"
     aria-label="A trapezoid camera view mapped by a matrix H to a top-down rectangle, with matching numbered points">
  <text class="caption" x="190" y="24">Camera view (pixels)</text>
  <polygon class="court-shape" points="110,60 270,60 350,230 30,230" />
  <circle class="pair" cx="110" cy="60" r="11" /><text class="pair-label" x="110" y="60">1</text>
  <circle class="pair" cx="270" cy="60" r="11" /><text class="pair-label" x="270" y="60">2</text>
  <circle class="pair" cx="350" cy="230" r="11" /><text class="pair-label" x="350" y="230">3</text>
  <circle class="pair" cx="30" cy="230" r="11" /><text class="pair-label" x="30" y="230">4</text>
  <circle class="rider" cx="232" cy="150" r="7" />

  <path class="arrow" d="M 385 145 L 505 145 M 497 138 L 505 145 L 497 152" />
  <text class="matrix" x="445" y="130">H</text>
  <text class="caption" x="445" y="170">cv2.findHomography</text>

  <text class="caption" x="710" y="24">Top-down court (metres)</text>
  <rect class="court-shape" x="550" y="50" width="320" height="190" />
  <circle class="pair" cx="550" cy="50" r="11" /><text class="pair-label" x="550" y="50">1</text>
  <circle class="pair" cx="870" cy="50" r="11" /><text class="pair-label" x="870" y="50">2</text>
  <circle class="pair" cx="870" cy="240" r="11" /><text class="pair-label" x="870" y="240">3</text>
  <circle class="pair" cx="550" cy="240" r="11" /><text class="pair-label" x="550" y="240">4</text>
  <circle class="rider" cx="765" cy="182" r="7" />
</svg>
"""

_CSS = """
svg { width: 100%; max-width: 900px; display: block; margin: 0.5rem auto; }
.court-shape { fill: #171B22; stroke: rgba(232, 234, 237, 0.35); stroke-width: 1.5; }
.pair { fill: #F5B83D; }
.pair-label { fill: #0E1116; font-size: 12px; font-weight: 600; text-anchor: middle; dominant-baseline: central; }
.rider { fill: #4DA3FF; }
.caption { fill: rgba(232, 234, 237, 0.55); font-size: 13px; text-anchor: middle; }
.matrix { fill: #E8EAED; font-size: 15px; text-anchor: middle; }
.arrow { stroke: rgba(232, 234, 237, 0.55); stroke-width: 1.5; fill: none; }
"""

_component = st.components.v2.component("homography_diagram", html=_HTML, css=_CSS)


def render_homography_diagram() -> None:
    _component(key="homography_diagram")

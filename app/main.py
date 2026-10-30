"""Entry point: page setup and navigation. Run from the repo root: streamlit run app/main.py"""
from pathlib import Path

import streamlit as st

from ui.navbar import render_navbar

st.set_page_config(page_title="Cycle Ball Tactical Map", page_icon=":material/sports_score:", layout="wide")
st.html(Path(__file__).parent / "ui" / "styles.css")

pages = [
    st.Page("views/analyze.py", title="Analyze", default=True),
    st.Page("views/analytics.py", title="Analytics"),
    st.Page("views/how_it_works.py", title="How it works"),
]
current_page = st.navigation(pages, position="hidden")
render_navbar(pages, current_page)
current_page.run()

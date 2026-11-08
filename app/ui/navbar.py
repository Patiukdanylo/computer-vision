"""Top navigation bar, shared by every page."""
import streamlit as st


def render_navbar(pages, current_page):
    with st.container(key="navbar", horizontal=True):
        for i, page in enumerate(pages):
            key = "nav-link-active" if page.title == current_page.title else f"nav-link-{i}"
            with st.container(key=key):
                st.page_link(page, label=page.title, width="stretch")

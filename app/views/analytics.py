"""Charts for a finished run."""
import numpy as np
import streamlit as st

from core.analytics import occupancy, share_in_left_half, track_stats
from core.demo_data import HZ
from ui import charts, state

CHART_OPTIONS = {"theme": None, "width": "stretch", "config": {"displayModeBar": False}}


def empty_screen():
    st.caption("Process a clip first; its charts will appear here.")
    st.page_link("views/analyze.py", label="Go to Analyze")


def analytics_screen(tracks):
    st.caption(":orange[Demo data: every number on this page comes from fake positions.]")
    stats = [track_stats(track, HZ) for track in tracks]
    by_kind = lambda kind: [track for track in tracks if track["kind"] == kind]
    team_distance = lambda kind: sum(row["distance_m"] for row in stats if row["kind"] == kind)
    ball = by_kind("ball")[0]

    columns = st.columns(4)
    columns[0].metric("Clip length", f"{(len(ball['points']) - 1) / HZ:.0f} s")
    columns[1].metric("Team A distance", f"{team_distance('team-a'):.0f} m")
    columns[2].metric("Team B distance", f"{team_distance('team-b'):.0f} m")
    columns[3].metric("Ball in left half", f"{share_in_left_half(ball):.0%}")

    st.subheader("Where the action happens")
    cells = {kind: occupancy(by_kind(kind)) for kind in charts.COLORS}
    max_percent = max(float(grid.max()) for grid in cells.values())
    names = {"team-a": "Team A", "team-b": "Team B", "ball": "Ball"}
    for column, kind in zip(st.columns(3), cells):
        column.markdown(names[kind])
        column.plotly_chart(charts.occupancy_heatmap(cells[kind], kind, max_percent), **CHART_OPTIONS)
    st.caption(f"Share of time per 1 m cell. Brighter means more time; the brightest cell is {max_percent:.1f}%.")

    left, right = st.columns(2)
    with left:
        st.subheader("Distance covered")
        st.plotly_chart(charts.rider_bars(stats, "distance_m", "m"), **CHART_OPTIONS)
    with right:
        st.subheader("Top speed")
        st.plotly_chart(charts.rider_bars(stats, "top_kmh", "km/h"), **CHART_OPTIONS)

    st.subheader("Ball position over time")
    x_positions = np.array(ball["points"])[:, 0]
    st.plotly_chart(charts.ball_position_line(np.arange(len(x_positions)) / HZ, x_positions), **CHART_OPTIONS)

    with st.expander("See the numbers as a table"):
        rows = [
            {
                "Tracked": row["name"],
                "Distance (m)": round(row["distance_m"], 1),
                "Average speed (km/h)": round(row["average_kmh"], 1),
                "Top speed (km/h)": round(row["top_kmh"], 1),
            }
            for row in stats
        ]
        st.dataframe(rows, hide_index=True)


st.title("Analytics")
run = state.get_run()
if run:
    analytics_screen(run["tracks"])
else:
    empty_screen()

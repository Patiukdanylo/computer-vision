"""Charts for a finished run."""
import numpy as np
import streamlit as st

from core.analytics import ball_in_play, distance_to_ball, occupancy, team_arrays, team_summary, track_stats
from ui import charts, state

CHART_OPTIONS = {"theme": None, "width": "stretch", "config": {"displayModeBar": False}}


def empty_screen():
    st.caption("Process a clip first; its charts will appear here.")
    st.page_link("views/analyze.py", label="Go to Analyze")


def analytics_screen(tracks, hz):
    ball = np.array(next(t for t in tracks if t["kind"] == "ball")["points"])
    window = ball_in_play(ball, hz)
    seconds = (window.stop - window.start) / hz
    arrays = team_arrays(tracks, window)
    summary = team_summary(arrays)
    stats = [track_stats(track, hz, window) for track in tracks if track["kind"] != "ball"]
    team_distance = lambda kind: sum(row["distance_m"] for row in stats if row["kind"] == kind)

    st.caption(f"Only the {seconds:.0f} s in which the ball is moving are counted. Positions come from detections mapped to "
               "metres; two riders of one team can swap identity, so team numbers are safer than rider numbers.")

    columns = st.columns(4)
    columns[0].metric("Ball in play", f"{seconds:.0f} s")
    columns[1].metric("Distance A / B", f"{team_distance('team-a'):.0f} / {team_distance('team-b'):.0f} m")
    columns[2].metric("Closest to ball A / B", f"{summary['a_closest']:.0%} / {1 - summary['a_closest']:.0%}")
    columns[3].metric("Contested", f"{summary['contested']:.0%}")
    st.caption("Distance: total path of both riders, measured over half-second steps. Closest to ball: share of frames in which "
               "a team's nearest rider is nearer to the ball. Contested: both teams have a rider within 3 m of the ball.")

    st.subheader("Who is near the ball")
    st.plotly_chart(charts.distance_to_ball_line(np.arange(window.stop - window.start) / hz, distance_to_ball(arrays)), **CHART_OPTIONS)
    spacing = summary["spacing_m"]
    st.caption(f"Average gap between a team's two riders: A {spacing['team-a']:.1f} m, B {spacing['team-b']:.1f} m. "
               "Where the ball was not detected, its position is interpolated.")

    st.subheader("Where the action happens")
    cells = {kind: occupancy(arrays[kind].reshape(-1, 2)) for kind in charts.COLORS}
    max_percent = max(float(grid.max()) for grid in cells.values())
    names = {"team-a": "Team A", "team-b": "Team B", "ball": "Ball"}
    for column, kind in zip(st.columns(3), cells):
        column.markdown(names[kind])
        column.plotly_chart(charts.occupancy_heatmap(cells[kind], kind, max_percent), **CHART_OPTIONS)
    st.caption(f"Share of time per 1 m cell while the ball is in play. The brightest cell is {max_percent:.1f}%. "
               "A single clip does not show tactics; it shows where this passage of play happened.")

    left, right = st.columns(2)
    with left:
        st.subheader("Distance covered")
        st.plotly_chart(charts.rider_bars(stats, "distance_m", "m"), **CHART_OPTIONS)
    with right:
        st.subheader("Fast speed")
        st.plotly_chart(charts.rider_bars(stats, "fast_kmh", "km/h"), **CHART_OPTIONS)
    st.caption("Fast speed: the speed a rider is below 95% of the time, over half-second steps. A single-frame maximum would "
               "mostly show detector jitter.")

    with st.expander("See the numbers as a table"):
        st.dataframe([{"Tracked": row["name"], "Distance (m)": round(row["distance_m"], 1),
                       "Average speed (km/h)": round(row["average_kmh"], 1), "Fast speed (km/h)": round(row["fast_kmh"], 1)}
                      for row in stats], hide_index=True)


st.title("Analytics")
run = state.get_run()
if run:
    analytics_screen(run["tracks"], run["hz"])
else:
    empty_screen()

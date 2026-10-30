"""Plotly figures, styled to match the app. Takes prepared numbers and only draws them."""
import numpy as np
import plotly.graph_objects as go

from core.config import COURT_LENGTH, COURT_WIDTH

SURFACE = "#171B22"
TEXT = "#E8EAED"
MUTED = "rgba(232, 234, 237, 0.55)"
LINE = "rgba(232, 234, 237, 0.35)"
GRID = "rgba(232, 234, 237, 0.08)"
COLORS = {"team-a": "#4DA3FF", "team-b": "#F2F2F2", "ball": "#FF4D4D"}
TEAM_NAMES = {"team-a": "Team A", "team-b": "Team B"}


def _base_layout(**overrides) -> dict:
    layout = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Nata Sans, sans-serif", "color": TEXT, "size": 13},
        "hoverlabel": {"bgcolor": SURFACE, "bordercolor": LINE, "font": {"color": TEXT}},
        "margin": {"l": 0, "r": 0, "t": 8, "b": 0},
    }
    return layout | overrides


def _court_shapes() -> list[dict]:
    """Same markings as the map. Only the outer size is official."""
    line = {"color": LINE, "width": 1}
    middle_x, middle_y = COURT_LENGTH / 2, COURT_WIDTH / 2
    circle = lambda x, y, r: {"type": "circle", "x0": x - r, "x1": x + r, "y0": y - r, "y1": y + r, "line": line}
    return [
        {"type": "rect", "x0": 0, "x1": COURT_LENGTH, "y0": 0, "y1": COURT_WIDTH, "line": line},
        {"type": "line", "x0": middle_x, "x1": middle_x, "y0": 0, "y1": COURT_WIDTH, "line": line},
        circle(middle_x, middle_y, 3.5),
        circle(middle_x, middle_y, 1.5),
        circle(0, middle_y, 2.5),
        circle(COURT_LENGTH, middle_y, 2.5),
    ]


def occupancy_heatmap(cells: np.ndarray, kind: str, max_percent: float) -> go.Figure:
    """One hue from the court floor up to the entity's colour. Pass the same maximum to compare heatmaps fairly."""
    rows, columns = cells.shape
    figure = go.Figure(go.Heatmap(
        z=cells,
        x=np.arange(columns) + 0.5,
        y=np.arange(rows) + 0.5,
        zmin=0,
        zmax=max_percent,
        colorscale=[[0, SURFACE], [1, COLORS[kind]]],
        showscale=False,
        xgap=1,
        ygap=1,
        hovertemplate="x %{x:.1f} m, y %{y:.1f} m<br>%{z:.1f}% of time<extra></extra>",
    ))
    figure.update_layout(_base_layout(height=300, shapes=_court_shapes()))
    figure.update_xaxes(range=[0, COURT_LENGTH], visible=False, constrain="domain")
    figure.update_yaxes(range=[COURT_WIDTH, 0], visible=False, scaleanchor="x")
    return figure


def rider_bars(stats: list[dict], key: str, unit: str) -> go.Figure:
    """Riders are always in the same order and keep their team colour. Values are labelled directly."""
    riders = [row for row in stats if row["kind"] in TEAM_NAMES]
    figure = go.Figure()
    for kind, team_name in TEAM_NAMES.items():
        team = [row for row in riders if row["kind"] == kind]
        figure.add_bar(
            name=team_name,
            y=[row["name"] for row in team],
            x=[row[key] for row in team],
            orientation="h",
            marker={"color": COLORS[kind], "cornerradius": 4},
            text=[f"{row[key]:.1f} {unit}" for row in team],
            textposition="outside",
            textfont={"color": TEXT},
            cliponaxis=False,
            hovertemplate=f"%{{y}}: %{{x:.1f}} {unit}<extra></extra>",
        )
    figure.update_layout(_base_layout(
        height=240,
        barmode="overlay",
        bargap=0.45,
        margin={"l": 36, "r": 64, "t": 32, "b": 0},
        legend={"orientation": "h", "x": 0, "y": 1.18, "font": {"color": MUTED}},
    ))
    figure.update_xaxes(visible=False, rangemode="tozero")
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=[row["name"] for row in reversed(riders)],
        showgrid=False,
        ticks="",
        tickfont={"color": TEXT},
        automargin=True,
    )
    return figure


def ball_position_line(seconds: np.ndarray, x_positions: np.ndarray) -> go.Figure:
    figure = go.Figure(go.Scatter(
        x=seconds,
        y=x_positions,
        mode="lines",
        line={"color": COLORS["ball"], "width": 2},
        hovertemplate="%{x:.1f} s: %{y:.1f} m along the court<extra></extra>",
    ))
    middle = COURT_LENGTH / 2
    figure.add_hline(y=middle, line={"color": LINE, "width": 1, "dash": "dot"})
    figure.add_annotation(x=1, xref="paper", y=middle, text="Halfway", showarrow=False, xanchor="right",
                          yanchor="bottom", font={"color": MUTED, "size": 12})
    figure.update_layout(_base_layout(height=240, margin={"l": 56, "r": 8, "t": 8, "b": 44}, hovermode="closest"))
    figure.update_xaxes(title={"text": "Time (s)", "font": {"color": MUTED, "size": 12}}, showgrid=False,
                        tickfont={"color": MUTED}, zeroline=False, automargin=True)
    figure.update_yaxes(title={"text": "Along the court (m)", "font": {"color": MUTED, "size": 12}},
                        range=[0, COURT_LENGTH], gridcolor=GRID, tickfont={"color": MUTED}, zeroline=False, automargin=True)
    return figure

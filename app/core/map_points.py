"""Turn the model's pixel boxes (models/detections.csv) into court positions in metres.

One row per thing drawn on the minimap: a rider or the ball, per frame.
"""
import numpy as np
import pandas as pd

from core.config import ROOT
from core.homography import to_court

DETECTIONS_CSV = ROOT / "models" / "detections.csv"
TEAMS_CSV = ROOT / "models" / "teams.csv"   # blue/white per player box, from the team-clustering notebook
BIKE_CONF = 0.5      # lower values add extra bike boxes (error analysis)
PLAYER_CONF = 0.5
BALL_CONF = 0.25     # the ball is hard for the model, so scores are low even when it is right
MIN_OVERLAP = 0.3    # a rider box must cover this share of a bike box to count as the same rider


def load_detections(path=DETECTIONS_CSV, teams_path=TEAMS_CSV):
    """All boxes, plus a `team` column (blue / white) on player boxes."""
    det = pd.read_csv(path)
    teams = pd.read_csv(teams_path)
    return det.merge(teams, how="left", on=["frame", "x1", "y1", "x2", "y2"])


def _overlap_share(rider, bike):
    """Share of the BIKE box covered by the rider box (0..1). Rider boxes sit on top of their bike."""
    w = min(rider.x2, bike.x2) - max(rider.x1, bike.x1)
    h = min(rider.y2, bike.y2) - max(rider.y1, bike.y1)
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / ((bike.x2 - bike.x1) * (bike.y2 - bike.y1))


def _riders_in_frame(players, bikes):
    """Pair each rider with a bike (greedy, most-confident rider first).

    Map point = bottom-centre of the bike box (the wheels touch the floor, so the homography is
    exact there). A rider with no bike falls back to the bottom-centre of the rider box, which is
    slightly off because it is about 0.3 m above the floor.
    """
    free = set(bikes.index)
    rows = []
    for _, rider in players.sort_values("confidence", ascending=False).iterrows():
        best, best_share = None, MIN_OVERLAP
        for i in free:
            share = _overlap_share(rider, bikes.loc[i])
            if share > best_share:
                best, best_share = i, share
        if best is None:
            rows.append((rider.frame, (rider.x1 + rider.x2) / 2, rider.y2, "player", rider.team))
        else:
            free.discard(best)
            b = bikes.loc[best]
            rows.append((rider.frame, (b.x1 + b.x2) / 2, b.y2, "bike", rider.team))
    return rows


def build_map_points(det, player_conf=PLAYER_CONF, bike_conf=BIKE_CONF, ball_conf=BALL_CONF):
    """Return a DataFrame: frame, kind (rider/ball), source (bike/player/ball), team, px, py, x_m, y_m."""
    rows = []
    for frame, g in det.groupby("frame"):
        players = g[(g["class"] == "player") & (g.confidence >= player_conf)]
        bikes = g[(g["class"] == "bike") & (g.confidence >= bike_conf)]
        for f, px, py, source, team in _riders_in_frame(players, bikes):
            rows.append((f, "rider", source, team, px, py))
        balls = g[(g["class"] == "ball") & (g.confidence >= ball_conf)]
        if len(balls):
            b = balls.sort_values("confidence").iloc[-1]          # one ball: keep the most confident
            rows.append((frame, "ball", "ball", None, (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2))
    out = pd.DataFrame(rows, columns=["frame", "kind", "source", "team", "px", "py"])
    out[["x_m", "y_m"]] = to_court(out[["px", "py"]].to_numpy())
    return out

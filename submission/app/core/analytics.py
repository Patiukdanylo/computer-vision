"""Numbers behind the Analytics page, calculated from tracks in metres. No plotting here.

Every number is one short formula, and the page prints its definition under it. Two choices keep the numbers honest:
- Only the part of the clip where the ball is in play counts (the ball moves; a ball lying still at the end says nothing).
- Speeds are measured over half a second, not frame to frame: the video has duplicate frames and the detector jitters a few
  centimetres, and at 25-47 frames per second both turn into fake 20+ km/h spikes.
"""
import numpy as np

from core.config import COURT_LENGTH, COURT_WIDTH

MS_TO_KMH = 3.6
STEP_SECONDS = 0.5          # speeds and distances use points this far apart in time
REST_WINDOW_SECONDS = 1.0   # the ball is "in play" if it moved more than REST_MOVE_M within +-1 s
REST_MOVE_M = 0.5
CONTESTED_M = 3.0           # the ball is "contested" when both teams have a rider this close


def ball_in_play(ball: np.ndarray, hz: float) -> slice:
    """First to last frame in which the ball moves (more than 0.5 m between 1 s before and 1 s after)."""
    n, k = len(ball), max(1, round(REST_WINDOW_SECONDS * hz))
    moved = np.linalg.norm(ball[np.minimum(np.arange(n) + k, n - 1)] - ball[np.maximum(np.arange(n) - k, 0)], axis=1) > REST_MOVE_M
    if not moved.any():
        return slice(0, n)
    where = np.flatnonzero(moved)
    return slice(int(where[0]), int(where[-1]) + 1)


def _steps(points: np.ndarray, hz: float) -> np.ndarray:
    """Distances (m) between points half a second apart."""
    w = max(1, round(STEP_SECONDS * hz))
    return np.linalg.norm(np.diff(points[::w], axis=0), axis=1)


def track_stats(track: dict, hz: float, window: slice) -> dict:
    points = np.array(track["points"])[window]
    steps = _steps(points, hz)
    speeds = steps / STEP_SECONDS * MS_TO_KMH
    return {
        "name": track["name"],
        "kind": track["kind"],
        "distance_m": float(steps.sum()),
        "average_kmh": float(speeds.mean()) if len(speeds) else 0.0,
        "fast_kmh": float(np.percentile(speeds, 95)) if len(speeds) else 0.0,   # 95th percentile: "its fast", not one spike
    }


def team_arrays(tracks: list[dict], window: slice) -> dict:
    """{kind: (riders, frames, 2)} inside the window, for team-a and team-b, plus the ball as (frames, 2)."""
    out = {kind: np.array([t["points"] for t in tracks if t["kind"] == kind])[:, window] for kind in ("team-a", "team-b")}
    out["ball"] = np.array([t["points"] for t in tracks if t["kind"] == "ball"][0])[window]
    return out


def distance_to_ball(arrays: dict) -> dict:
    """Per frame: distance (m) from the ball to the nearest rider of each team."""
    return {kind: np.linalg.norm(arrays[kind] - arrays["ball"][None], axis=2).min(axis=0) for kind in ("team-a", "team-b")}


def team_summary(arrays: dict) -> dict:
    near = distance_to_ball(arrays)
    a_closer = near["team-a"] < near["team-b"]
    contested = (near["team-a"] < CONTESTED_M) & (near["team-b"] < CONTESTED_M)
    spacing = {}
    for kind in ("team-a", "team-b"):
        riders = arrays[kind]
        spacing[kind] = float(np.linalg.norm(riders[0] - riders[1], axis=1).mean()) if len(riders) >= 2 else float("nan")
    return {"a_closest": float(a_closer.mean()), "contested": float(contested.mean()), "spacing_m": spacing,
            "near_ball_m": {kind: float(near[kind].mean()) for kind in near}}


def occupancy(points: np.ndarray) -> np.ndarray:
    """Share of time, in percent, spent in each 1 m x 1 m court cell. points: (n, 2). Rows are y, columns are x."""
    bins = [int(COURT_LENGTH), int(COURT_WIDTH)]
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=[[0, COURT_LENGTH], [0, COURT_WIDTH]])
    return counts.T * 100 / len(points)

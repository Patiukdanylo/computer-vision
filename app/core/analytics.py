"""Numbers behind the Analytics page, calculated from tracks in metres. No plotting here."""
import numpy as np

from core.config import COURT_LENGTH, COURT_WIDTH

MS_TO_KMH = 3.6


def _step_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def track_stats(track: dict, hz: int) -> dict:
    points = np.array(track["points"])
    steps = _step_lengths(points)
    speeds = steps * hz * MS_TO_KMH
    return {
        "name": track["name"],
        "kind": track["kind"],
        "distance_m": float(steps.sum()),
        "average_kmh": float(speeds.mean()),
        "top_kmh": float(speeds.max()),
    }


def occupancy(tracks: list[dict]) -> np.ndarray:
    """Share of time, in percent, spent in each 1 m x 1 m court cell. Rows are y, columns are x."""
    points = np.concatenate([np.array(track["points"]) for track in tracks])
    bins = [int(COURT_LENGTH), int(COURT_WIDTH)]
    counts, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins, range=[[0, COURT_LENGTH], [0, COURT_WIDTH]])
    return counts.T * 100 / len(points)


def share_in_left_half(track: dict) -> float:
    x_positions = np.array(track["points"])[:, 0]
    return float((x_positions < COURT_LENGTH / 2).mean())

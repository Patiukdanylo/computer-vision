"""Fake court positions, so the map can be built before the backend exists. Delete this once real data arrives."""
import math
import random

from core.config import COURT_LENGTH, COURT_WIDTH

HZ = 10  # positions per second
MARGIN = 0.5  # metres kept from the court edge
TEAM_A_X, TEAM_B_X = 0.35, 0.65  # where each team drifts around, as a share of the court length


def make_path(rng: random.Random, duration: float, center_x: float, speed: float) -> list[list[float]]:
    """Smooth wandering path in metres: slow sine waves added together."""
    waves = [(rng.uniform(0.2, 0.5) * speed, rng.uniform(0, math.tau), rng.uniform(0.2, 0.5) * speed,
              rng.uniform(0, math.tau)) for _ in range(3)]
    points = []
    for step in range(math.ceil(duration * HZ) + 1):
        t = step / HZ
        x = center_x + sum(math.sin(wx * t + px) for wx, px, _, _ in waves) * COURT_LENGTH * 0.15
        y = COURT_WIDTH / 2 + sum(math.sin(wy * t + py) for _, _, wy, py in waves) * COURT_WIDTH * 0.18
        points.append([round(min(max(x, MARGIN), COURT_LENGTH - MARGIN), 2),
                       round(min(max(y, MARGIN), COURT_WIDTH - MARGIN), 2)])
    return points


def make_demo_tracks(duration: float) -> list[dict]:
    """Two riders per team and a ball, in the shape the map expects."""
    rng = random.Random(7)
    return [
        {"name": "A1", "kind": "team-a", "points": make_path(rng, duration, COURT_LENGTH * TEAM_A_X, 1.0)},
        {"name": "A2", "kind": "team-a", "points": make_path(rng, duration, COURT_LENGTH * TEAM_A_X, 1.0)},
        {"name": "B1", "kind": "team-b", "points": make_path(rng, duration, COURT_LENGTH * TEAM_B_X, 1.0)},
        {"name": "B2", "kind": "team-b", "points": make_path(rng, duration, COURT_LENGTH * TEAM_B_X, 1.0)},
        {"name": "Ball", "kind": "ball", "points": make_path(rng, duration, COURT_LENGTH / 2, 2.0)},
    ]

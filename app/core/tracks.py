"""Map points -> the tracks the minimap player expects: two riders per team plus the ball, one point per video frame.

Identity is kept with a tiny tracker: each team has two slots, and every frame the team's detected
riders are matched to the slots by distance on the court. Gaps (a missed rider or ball) are filled
by straight-line interpolation, because the player cannot draw a missing point.
"""
from itertools import product

import numpy as np

from core.map_points import build_map_points, load_detections

TEAM_KIND = {"blue": "team-a", "white": "team-b"}      # colours match the dot colours in the app
RIDERS_PER_TEAM = 2
MAX_JUMP_M = 0.5                  # most a rider may move between two frames and still be the same rider
JUMP_PER_MISSED_FRAME_M = 0.1     # extra allowance for every frame the rider was not seen
UNMATCHED_COST = 100.0            # any allowed match beats leaving a slot empty (the jump limit already rejects bad ones)
BALL_WINDOW = 15                  # frames each side used to judge whether a ball box is an outlier
BALL_OUTLIER_M = 3.0              # a ball box this far from its neighbours' median is a false detection
SMOOTH_FRAMES = 11                # about 0.25 s at 46.8 fps


def _fill_gaps(points):
    """points: (n, 2) array with NaN where unknown. Linear interpolation inside, hold the nearest value at the ends."""
    out = points.copy()
    known = ~np.isnan(out[:, 0])
    if not known.any():
        return None
    index = np.arange(len(out))
    for axis in (0, 1):
        out[:, axis] = np.interp(index, index[known], out[known, axis])
    return out


def _track_team(frames, n_frames):
    """frames: {frame_index: [(x, y), ...]} for one team. Returns two (n_frames, 2) arrays with NaN gaps.

    A detection may only join a slot if it is close enough to where that slot was last seen. The allowed
    distance grows with the number of frames the slot has been missing, so a lost rider can be found again,
    while a wrong-team or duplicate box that jumps across the court is ignored.
    """
    slots = np.full((RIDERS_PER_TEAM, n_frames, 2), np.nan)
    last = [None] * RIDERS_PER_TEAM                        # last known position of each slot
    last_seen = [0] * RIDERS_PER_TEAM
    for i in range(n_frames):
        found = frames.get(i, [])
        if not found:
            continue
        for point in found:                                # slots not started yet take the first detections
            free = [s for s in range(RIDERS_PER_TEAM) if last[s] is None]
            if free:
                last[free[0]], last_seen[free[0]] = point, i
                slots[free[0], i] = point
        if any(p is None for p in last):
            continue
        allowed = [MAX_JUMP_M + JUMP_PER_MISSED_FRAME_M * (i - last_seen[s]) for s in range(RIDERS_PER_TEAM)]
        best = None                                        # every way to give detections (or nothing) to the slots
        for choice in product([None, *range(len(found))], repeat=RIDERS_PER_TEAM):
            used = [c for c in choice if c is not None]
            if len(used) != len(set(used)):
                continue
            cost, ok = 0.0, True
            for s, c in enumerate(choice):
                if c is None:
                    cost += UNMATCHED_COST
                    continue
                d = np.hypot(found[c][0] - last[s][0], found[c][1] - last[s][1])
                if d > allowed[s]:
                    ok = False
                    break
                cost += d
            if ok and (best is None or cost < best[0]):
                best = (cost, choice)
        if best is None:
            continue
        for s, c in enumerate(best[1]):
            if c is not None:
                last[s], last_seen[s] = found[c], i
                slots[s, i] = found[c]
    return slots


def _drop_ball_outliers(ball):
    """ball: (n, 2) with NaN gaps. Remove boxes that sit far from the median of nearby ball boxes (wheels, shoes...)."""
    out = ball.copy()
    for i in np.flatnonzero(~np.isnan(ball[:, 0])):
        lo, hi = max(0, i - BALL_WINDOW), i + BALL_WINDOW + 1
        near = np.delete(ball[lo:hi], i - lo, axis=0)
        near = near[~np.isnan(near[:, 0])]
        if len(near) >= 2 and np.hypot(*(ball[i] - np.median(near, axis=0))) > BALL_OUTLIER_M:
            out[i] = np.nan
    return out


def _smooth(points):
    """Centred moving average over SMOOTH_FRAMES frames, to remove box jitter (speeds are computed from steps)."""
    pad = SMOOTH_FRAMES // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in (0, 1)], axis=1)


BOX_CODES = {"blue": 0, "white": 1}   # 0 = team A, 1 = team B, 2 = ball, 3 = player of unknown team


def build_boxes(confidences=None):
    """Detection boxes for drawing on the video: a list with one entry per frame, each [code, x1, y1, x2, y2] in pixels."""
    confidences = confidences or {}
    det = load_detections()
    n_frames = int(det.frame.max())
    boxes = [[] for _ in range(n_frames)]
    players = det[(det["class"] == "player") & (det.confidence >= confidences.get("player", 0.5))]
    for row in players.itertuples():
        boxes[row.frame - 1].append([BOX_CODES.get(row.team, 3), *map(round, (row.x1, row.y1, row.x2, row.y2))])
    balls = det[(det["class"] == "ball") & (det.confidence >= confidences.get("ball", 0.25))]
    for frame, group in balls.groupby("frame"):
        best = group.sort_values("confidence").iloc[-1]                    # one ball per frame, like the map
        boxes[frame - 1].append([2, *map(round, (best.x1, best.y1, best.x2, best.y2))])
    return boxes


def build_tracks(duration, confidences=None):
    """Real tracks for the sample clip. Returns (tracks, hz) as lists of [x, y] metres per video frame."""
    confidences = confidences or {}
    det = load_detections()
    n_frames = int(det.frame.max())
    mp = build_map_points(det, player_conf=confidences.get("player", 0.5), bike_conf=confidences.get("bike", 0.5),
                          ball_conf=confidences.get("ball", 0.25))
    tracks = []
    for color, kind in TEAM_KIND.items():
        riders = mp[(mp.kind == "rider") & (mp.team == color)]
        by_frame = {}
        for row in riders.itertuples():
            by_frame.setdefault(row.frame - 1, []).append((row.x_m, row.y_m))
        slots = _track_team(by_frame, n_frames)
        for number, slot in enumerate(slots, 1):
            filled = _fill_gaps(slot)
            if filled is not None:
                tracks.append({"name": f"{kind[-1].upper()}{number}", "kind": kind, "points": _smooth(filled).round(2).tolist()})
    ball = np.full((n_frames, 2), np.nan)
    for row in mp[mp.kind == "ball"].itertuples():
        ball[row.frame - 1] = (row.x_m, row.y_m)
    filled = _fill_gaps(_drop_ball_outliers(ball))
    if filled is not None:
        tracks.append({"name": "Ball", "kind": "ball", "points": _smooth(filled).round(2).tolist()})
    return tracks, round(n_frames / duration, 3)

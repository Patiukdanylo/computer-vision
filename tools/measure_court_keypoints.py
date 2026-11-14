"""Measure the 14 court keypoints of our camera view to sub-pixel precision: the reference labels for the keypoint model.

Run from the repo folder:  python tools/measure_court_keypoints.py
Writes data/court_reference.json and outputs/keypoints/reference_check.jpg (zoomed crops to check every point by eye).

Why measure instead of click: the camera never moves, so every landmark sits on the same pixel in all 1,170 frames.
We label it once, on a "background plate": the median of 90 frames spread over the clip. Riders never stay in one
place, so the median removes them and every court marking is visible. A painted line is ~6 px wide; fitting a line to
it is more precise than one click (the clicked points in data/court_keypoints.json are off by up to 8 px, see notebook 08).

The keypoints and their court position in metres (x along the 14 m length from the left end line, y across the 11 m
width from the far board towards the camera):
  far_left_corner / far_right_corner   (0, 0) / (14, 0)  black end line meets the far board
  *_goal_far_post / *_goal_near_post   (0 or 14, y)      base of the goal's front post, on the end line
  centre_spot                          (7, 5.5)          dark dot in the centre of the circles
  inner_circle_* / outer_circle_*      r = 2 m / 4 m     where the circle crosses the court's two centre axes (no outer near point: hidden)

The goals are movable, so their y is measured, not assumed: least squares fits one homography to the corners, the
centre spot, ~330 points on the two painted circles (must lie 2 m and 4 m from the centre) and the four posts (must lie
on the end lines, x = 0 or x = 14). The posts' y is then read from that homography.

GenAI: written with Claude Code. every point was checked by eye on outputs/keypoints/reference_check.jpg.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "cycle-ball.mp4"
OUT_JSON = ROOT / "data" / "court_reference.json"
OUT_DIR = ROOT / "outputs" / "keypoints"
INNER_R, OUTER_R = 2.0, 4.0           # radii of the painted circles (measured: 2.005 and 4.018 m)
CENTRE = np.array([7.0, 5.5])

# Rough pixel positions to start each fit from (read off frame 1). The fits below move them onto the markings.
END_LINES = {"left": ((690, 707), (380, 828)), "right": ((2350, 718), (2640, 862))}
FAR_BOARD = ((760, 702), (2300, 712))
POSTS = {  # name: (rough x of the post just above the floor, rows to fit the post axis on, end line, x in metres)
    "left_goal_far_post": (497, (715, 774), "left", 0.0),
    "left_goal_near_post": (364, (760, 825), "left", 0.0),
    "right_goal_far_post": (2513, (735, 795), "right", 14.0),
    "right_goal_near_post": (2627, (790, 850), "right", 14.0),
}
CENTRE_DOT_WINDOW = (1468, 1502, 821, 830)    # x0, x1, y0, y1 around the dark centre dot
# Where each circle crosses the two centre axes. No near point on the outer circle: at (7, 9.5) the floor is hidden
# behind the near board and the officials' table, so nothing can be labelled there.
CIRCLE_POINTS = {"inner": ("far", "near", "left", "right"), "outer": ("far", "left", "right")}
DIRECTIONS = {"far": (0, -1), "near": (0, 1), "left": (-1, 0), "right": (1, 0)}
NAMES = ["far_left_corner", "far_right_corner", "left_goal_far_post", "left_goal_near_post", "right_goal_far_post",
         "right_goal_near_post", "centre_spot", "inner_circle_far", "inner_circle_near", "inner_circle_left",
         "inner_circle_right", "outer_circle_far", "outer_circle_left", "outer_circle_right"]


def background_plate(step=13):
    """Median of every 13th frame: the riders move, so they disappear and the court markings stay."""
    capture, frames, i = cv2.VideoCapture(str(VIDEO)), [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(frame)
        i += 1
    capture.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


class Plate:
    """Sub-pixel measurements on the grey background plate."""

    def __init__(self, image):
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.g = cv2.GaussianBlur(grey, (3, 3), 0)

    def value(self, x, y):
        return cv2.getRectSubPix(self.g, (1, 1), (float(x), float(y)))[0, 0]

    def dark_centre(self, p, normal, half=8, step=0.25, min_contrast=12):
        """Walk across a dark painted line along `normal` and return the middle of the line (darkness-weighted)."""
        ts = np.arange(-half, half + step, step)
        vals = np.array([self.value(*(p + t * normal)) for t in ts])
        darkness = np.clip(np.percentile(vals, 90) - vals, 0, None)
        i = int(np.argmin(vals))
        lo, hi = max(i - 12, 0), min(i + 13, len(ts))          # +-3 px around the darkest sample
        if darkness[i] < min_contrast:
            return None                                         # no clear line here (worn paint, shadow)
        return p + (ts[lo:hi] * darkness[lo:hi]).sum() / darkness[lo:hi].sum() * normal

    def edge(self, p, normal, half=6, step=0.25):
        """Strongest dark-to-light step along `normal` (the bottom edge of the far board)."""
        ts = np.arange(-half, half + step, step)
        grad = np.gradient(np.array([self.value(*(p + t * normal)) for t in ts]))
        i = int(np.argmax(grad))
        if grad[i] < 2:
            return None
        lo, hi = max(i - 4, 0), min(i + 5, len(ts))
        w = np.clip(grad[lo:hi], 0, None)
        return p + (ts[lo:hi] * w).sum() / w.sum() * normal


def fit_line(points):
    """Straight line through points (total least squares, one pass of outlier removal). Returns (homogeneous line, rms px)."""
    for _ in range(2):
        centre = points.mean(0)
        direction = np.linalg.svd(points - centre)[2][0]
        normal = np.array([-direction[1], direction[0]])
        residual = (points - centre) @ normal
        points = points[np.abs(residual) < 2.5 * max(residual.std(), 0.3)]
    return np.cross([*centre, 1], [*(centre + direction), 1]), float(np.sqrt((residual ** 2).mean()))


def trace(plate, a, b, finder, n=60):
    """Measure points along the rough segment a-b with `finder` (dark_centre or edge)."""
    a, b = np.float64(a), np.float64(b)
    d = (b - a) / np.linalg.norm(b - a)
    normal = np.array([-d[1], d[0]]) * (1 if d[0] >= 0 else -1)      # points down (+y) for left-to-right segments
    found = [finder(a + s * (b - a), normal) for s in np.linspace(0, 1, n)]
    return np.array([p for p in found if p is not None])


def intersect(l1, l2):
    x = np.cross(l1, l2)
    return x[:2] / x[2]


def post_axis(plate, x_guess, rows, half=10):
    """Axis of a white goal post: the brightness-weighted middle of the post in each row, then a straight line x = a*y + b."""
    pts = []
    for y in np.arange(*rows, 1.0):
        xs = np.arange(x_guess - half, x_guess + half, 0.25)
        vals = np.array([plate.value(x, y) for x in xs])
        w = np.clip(vals - np.percentile(vals, 30), 0, None)
        i = int(np.argmax(vals))
        lo, hi = max(i - 12, 0), min(i + 13, len(xs))
        if w[lo:hi].sum() > 0:
            pts.append(((xs[lo:hi] * w[lo:hi]).sum() / w[lo:hi].sum(), y))
    pts = np.array(pts)
    a, b = np.polyfit(pts[:, 1], pts[:, 0], 1)
    y0, y1 = rows
    return np.cross([a * y0 + b, y0, 1], [a * y1 + b, y1, 1])


def to_m(H, px):
    return cv2.perspectiveTransform(np.float64(px).reshape(-1, 1, 2), H).reshape(-1, 2)


def to_px(H, m):
    return cv2.perspectiveTransform(np.float64(m).reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)


def circle_points(plate, H, radius, n=180, max_y=975):
    """Points on a painted circle: start where H puts the circle, walk across the line to its darkest middle.
    Rows below max_y are skipped: the officials' table hides the bottom of the outer circle in some frames."""
    found = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        u = np.array([np.cos(a), np.sin(a)])
        p0, p1 = to_px(H, [CENTRE + radius * u, CENTRE + (radius + 0.05) * u])
        q = plate.dark_centre(p0, (p1 - p0) / np.linalg.norm(p1 - p0), half=14)
        if q is not None and q[1] < max_y:
            found.append(q)
    return np.array(found)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image = background_plate()
    cv2.imwrite(str(OUT_DIR / "background_plate.png"), image)
    plate = Plate(image)
    px, report = {}, {}

    # 1. Lines: the two black end lines and the bottom edge of the far board.
    lines = {}
    for side, (a, b) in END_LINES.items():
        lines[side], report[f"{side}_end_line_rms_px"] = fit_line(trace(plate, a, b, plate.dark_centre))
    lines["far"], report["far_board_edge_rms_px"] = fit_line(trace(plate, *FAR_BOARD, plate.edge, n=80))
    px["far_left_corner"] = intersect(lines["left"], lines["far"])
    px["far_right_corner"] = intersect(lines["right"], lines["far"])

    # 2. Goal posts: post axis meets the end line.
    for name, (x_guess, rows, side, _) in POSTS.items():
        px[name] = intersect(post_axis(plate, x_guess, rows), lines[side])

    # 3. Centre spot: centroid of the darkness of the small dot.
    x0, x1, y0, y1 = CENTRE_DOT_WINDOW
    xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    w = np.clip(np.percentile(plate.g[ys, xs], 70) - plate.g[ys, xs], 0, None)
    px["centre_spot"] = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])

    # 4. A first homography from rule positions (goals centred, 2 m wide), only to find the circles.
    rule = {"far_left_corner": (0, 0), "far_right_corner": (14, 0), "centre_spot": (7, 5.5),
            "left_goal_far_post": (0, 4.5), "left_goal_near_post": (0, 6.5),
            "right_goal_far_post": (14, 4.5), "right_goal_near_post": (14, 6.5)}
    H0, _ = cv2.findHomography(np.float64([px[k] for k in rule]), np.float64(list(rule.values())), 0)
    inner, outer = circle_points(plate, H0, INNER_R), circle_points(plate, H0, OUTER_R)

    # 5. Reference homography: corners and centre fixed, circles round with their radius, posts on the end lines.
    fixed = ["far_left_corner", "far_right_corner", "centre_spot"]
    fixed_px, fixed_m = np.float64([px[k] for k in fixed]), np.float64([rule[k] for k in fixed])

    def residuals(h):
        H = np.append(h, 1).reshape(3, 3)
        r = [(to_m(H, fixed_px) - fixed_m).ravel()]
        for pts, radius in [(inner, INNER_R), (outer, OUTER_R)]:
            r.append((np.linalg.norm(to_m(H, pts) - CENTRE, axis=1) - radius) / np.sqrt(len(pts) / 8))
        r.append([to_m(H, [px[name]])[0, 0] - x_m for name, (*_, x_m) in POSTS.items()])
        return np.concatenate(r)

    fit = least_squares(residuals, (H0 / H0[2, 2]).ravel()[:8], loss="soft_l1", f_scale=0.05)
    H_ref = np.append(fit.x, 1).reshape(3, 3)

    # 6. Court positions in metres: posts measured (x snapped to the end line), the rest by definition.
    metres = {k: np.float64(rule[k]) for k in fixed}
    for name, (*_, x_m) in POSTS.items():
        metres[name] = np.array([x_m, round(float(to_m(H_ref, [px[name]])[0, 1]), 2)])
    radii = {"inner": INNER_R, "outer": OUTER_R}
    for ring, places in CIRCLE_POINTS.items():
        for where in places:
            metres[f"{ring}_circle_{where}"] = CENTRE + radii[ring] * np.float64(DIRECTIONS[where])

    # 7. Circle keypoints in pixels: where H_ref puts them, then moved across the line onto the painted circle.
    for ring, places in CIRCLE_POINTS.items():
        radius = radii[ring]
        for where in places:
            name = f"{ring}_circle_{where}"
            u = (metres[name] - CENTRE) / radius
            p0, p1 = to_px(H_ref, [metres[name], metres[name] + 0.05 * u])
            q = plate.dark_centre(p0, (p1 - p0) / np.linalg.norm(p1 - p0), half=10)
            px[name] = q if q is not None else p0       # worn paint: keep the homography's position
            report[f"{name}_snapped"] = q is not None

    # 8. Check: the homography from the 14 final labels, and how far each label is from where it should be.
    label_px = np.float64([px[k] for k in NAMES])
    label_m = np.float64([metres[k] for k in NAMES])
    H_label, _ = cv2.findHomography(label_px, label_m, 0)
    err = np.linalg.norm(to_m(H_label, label_px) - label_m, axis=1)
    fitted = [np.linalg.norm(to_m(H_label, pts) - CENTRE, axis=1) for pts in (inner, outer)]
    report.update({
        "label_reprojection_error_m": {k: round(float(e), 3) for k, e in zip(NAMES, err)},
        "inner_circle_radius_m": [round(float(fitted[0].mean()), 3), round(float(fitted[0].std()), 3)],
        "outer_circle_radius_m": [round(float(fitted[1].mean()), 3), round(float(fitted[1].std()), 3)],
        "circle_points_used": [len(inner), len(outer)],
    })

    OUT_JSON.write_text(json.dumps({
        "about": "Court keypoints of data/cycle-ball.mp4, measured by tools/measure_court_keypoints.py on the median "
                 "background plate. px = pixel on the full 3024 x 1078 frame, m = court position in metres "
                 "(x along the 14 m length from the left end line, y from the far board towards the camera).",
        "frame_size": [int(image.shape[1]), int(image.shape[0])],
        "keypoints": {k: {"px": [round(float(v), 2) for v in px[k]], "m": [round(float(v), 3) for v in metres[k]]}
                      for k in NAMES},
        "H_px_to_m": H_label.tolist(),
        "checks": report,
    }, indent=2), encoding="utf-8")
    save_check_sheet(image, px, H_label)
    print(json.dumps(report, indent=1))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")


def save_check_sheet(image, px, H, zoom=6, half=22):
    """One zoomed crop per keypoint (red dot = label) and an overview with the court template drawn through H."""
    bright = np.clip(255 * (image / 255.0) ** 0.6, 0, 255).astype(np.uint8)
    tiles = []
    for name in NAMES:
        x, y = px[name]
        x0, y0 = int(round(x)) - half, int(round(y)) - half
        crop = cv2.resize(bright[max(y0, 0):y0 + 2 * half, max(x0, 0):x0 + 2 * half], (2 * half * zoom, 2 * half * zoom),
                          interpolation=cv2.INTER_CUBIC)
        c = (int((x - max(x0, 0)) * zoom), int((y - max(y0, 0)) * zoom))
        cv2.drawMarker(crop, c, (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
        cv2.circle(crop, c, 3, (0, 0, 255), -1)
        cv2.putText(crop, name, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        tiles.append(crop)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 5)          # fill the last row
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    cv2.imwrite(str(OUT_DIR / "reference_check.jpg"), np.vstack(rows))

    overview = bright.copy()
    t = np.linspace(0, 2 * np.pi, 200)
    shapes = [([[0, 0], [14, 0], [14, 11], [0, 11]], True), ([[7, 0], [7, 11]], False), ([[0, 5.5], [14, 5.5]], False)]
    shapes += [(np.c_[7 + r * np.cos(t), 5.5 + r * np.sin(t)], True) for r in (INNER_R, OUTER_R)]
    for pts, closed in shapes:
        cv2.polylines(overview, [to_px(H, pts).round().astype(np.int32)], closed, (0, 255, 0), 2, cv2.LINE_AA)
    for name in NAMES:
        cv2.circle(overview, tuple(np.round(px[name]).astype(int)), 7, (0, 0, 255), -1)
    cv2.imwrite(str(OUT_DIR / "reference_overview.jpg"), overview[550:])


if __name__ == "__main__":
    main()

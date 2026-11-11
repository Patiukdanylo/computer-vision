"""Click court landmarks on frame 1 and save their pixel coordinates (full resolution, 3024 x 1078).

Run from the repo folder:  python tools/pick_points.py

One window, two views:
  1. Overview: frame 1 shrunk to fit the screen (and brightened: the hall is dark). Click roughly on the next landmark.
  2. Zoom: the same window now shows that spot 10 times bigger. Click the exact spot. The point is saved.
Keys: u = undo last point, Esc = back from zoom to overview, s = save, q = save and quit.
The points go to data/court_keypoints.json as {"name": [x, y], ...} and are also printed in the terminal.
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "cycle-ball.mp4"
OUT = ROOT / "data" / "court_keypoints.json"
WINDOW = "pick points"

OVERVIEW_W = 1600      # width of the overview, in screen pixels
ZOOM_HALF = 40         # the zoom crop is 80 x 80 px of the original frame...
ZOOM_SCALE = 10        # ...shown 10 times bigger
GAMMA = 0.6            # < 1 brightens a dark image

# (name, plain-language instruction). Click them in this order.
STEPS = [
    ("far_left_corner", "FAR-LEFT corner of the court floor (top-left, where the left end board meets the back board)"),
    ("far_right_corner", "FAR-RIGHT corner of the court floor (top-right, where the right end board meets the back board)"),
    ("centre_spot", "the small dot in the exact CENTRE of the centre circle"),
    ("left_end_low", "LEFT end board, where it touches the floor, LOW in the picture (bottom-left, near the camera)"),
    ("right_end_low", "RIGHT end board, where it touches the floor, LOW in the picture (bottom-right, near the camera)"),
    ("near_sideline_left", "NEAR sideline (floor line along the bottom edge), on the LEFT part of the court"),
    ("near_sideline_right", "NEAR sideline (floor line along the bottom edge), on the RIGHT part of the court"),
]
NAMES = [name for name, _ in STEPS]
HELP = dict(STEPS)


def load_frame():
    """Frame 1 of the video, brightened."""
    capture = cv2.VideoCapture(str(VIDEO))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Could not read {VIDEO}")
    return np.clip(255 * (frame / 255.0) ** GAMMA, 0, 255).astype(np.uint8)


frame = load_frame()
height, width = frame.shape[:2]
scale = OVERVIEW_W / width                       # screen pixels per original pixel in the overview
overview = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

points = []                                      # list of (name, x, y) in original pixels
zoom_origin = None                               # (x0, y0) of the zoom crop while in the zoom view, else None


def banner(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def show_overview():
    global zoom_origin
    zoom_origin = None
    image = overview.copy()
    for number, (name, x, y) in enumerate(points, 1):
        cx, cy = round(x * scale), round(y * scale)
        cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(image, str(number), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if len(points) < len(NAMES):
        banner(image, f"{len(points)+1}/{len(NAMES)}  {HELP[NAMES[len(points)]]}   [u=undo q=quit]")
    else:
        banner(image, "all points done: press q to save and quit")
    cv2.imshow(WINDOW, image)


def show_zoom(x, y):
    """Magnify the area around the rough click (x, y, original pixels) in the same window."""
    global zoom_origin
    x0 = int(np.clip(x - ZOOM_HALF, 0, width - 2 * ZOOM_HALF))
    y0 = int(np.clip(y - ZOOM_HALF, 0, height - 2 * ZOOM_HALF))
    zoom_origin = (x0, y0)
    crop = frame[y0:y0 + 2 * ZOOM_HALF, x0:x0 + 2 * ZOOM_HALF]
    crop = cv2.resize(crop, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_CUBIC)
    banner(crop, f"{NAMES[len(points)]}: click the EXACT spot   (Esc = back)")
    cv2.imshow(WINDOW, crop)


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if zoom_origin is None:                       # overview click -> open the zoom on that spot
        if len(points) < len(NAMES):
            show_zoom(x / scale, y / scale)
    else:                                         # zoom click -> save the exact point
        x_orig = zoom_origin[0] + x / ZOOM_SCALE
        y_orig = zoom_origin[1] + y / ZOOM_SCALE
        points.append((NAMES[len(points)], round(x_orig, 1), round(y_orig, 1)))
        print(f"{len(points):>2}. {points[-1][0]:<20} x={x_orig:7.1f}  y={y_orig:7.1f}", flush=True)
        show_overview()


def save():
    OUT.write_text(json.dumps({name: [x, y] for name, x, y in points}, indent=2))
    print(f"saved {len(points)} points to {OUT.relative_to(ROOT)}", flush=True)


cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_click)
show_overview()
print("Window open. Click once on it to give it focus, then click the first landmark.", flush=True)
while True:
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        save()
    if key == ord("u") and points and zoom_origin is None:
        print("undo:", points.pop()[0], flush=True)
        show_overview()
    if key == 27 and zoom_origin is not None:     # Esc: back to the overview without saving a point
        show_overview()
save()
cv2.destroyAllWindows()

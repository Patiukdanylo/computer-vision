"""Build the court keypoint datasets (YOLO pose format) from the video and the 14 reference keypoints.

Two label layouts for the same images (notebook 08 compares them):
- court (K0, K1), output_dir: one object per image, class `court`. Its box is the part of the court floor inside the
  picture, its 14 keypoints are the landmarks in data/court_reference.json.
- landmarks (K2), landmarks_output_dir: one object per landmark, 14 classes. Each has a small box around the landmark
  (landmark_box_px wide on the full frame) and 1 keypoint, the landmark itself. Same images (hard links, no copies).
The camera never moves, so the reference labels are valid for every frame: frames only differ in where the riders
are, i.e. which landmarks are covered.

The problem with a static camera: every label would sit on the same pixel in every image, so the model could learn
the coordinates by heart and ignore the image. It would then fail the moment the camera moves (or for another clip).
So each frame is saved several times: once as filmed, and as seen by a virtual camera that pans, tilts, rolls and
zooms (settings in configs/keypoints.yaml). The keypoints move with the view, so the model has to find them.

Virtual camera: x_new = K' R K^-1 x. K holds the focal length (px) and the image centre, R is the rotation, K' the
zoomed focal length. A camera that only rotates around its own centre sees every point (floor, riders, goals) through
this one homography, so the warped views are what a pan-tilt-zoom camera at the same spot would really film.

Input:  configs/keypoints.yaml -> video, reference labels, detections (to flag keypoints behind riders)
Output: output_dir/images/{train,val}/frame_XXXXX_vN.jpg   (v0 = as filmed, v1.. = virtual camera)
        output_dir/labels/{train,val}/frame_XXXXX_vN.txt   (class, box, 14 x (x, y, visibility), all 0-1)
        output_dir/views.csv   one row per image: the warp, the image-to-court homography G, visible/hidden keypoints
        output_dir/data.yaml, data_as_filmed.yaml (K0: train on the v0 images only)
        landmarks_output_dir/images, labels (one line per landmark in the picture: class, box, x, y, visibility)
        landmarks_output_dir/data.yaml

Visibility: 2 = in the picture, 1 = in the picture but inside a rider or bike box, 0 = outside the picture.
YOLO trains on 1 and 2 alike (the model learns where a hidden landmark is from the rest of the court), 0 is ignored.

Run from anywhere:  python data/prepare_keypoint_dataset.py

GenAI: written with Claude Code (Opus 5.5) on 2026-11-13, modelled on data/prepare_dataset.py.
"""
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "keypoints.yaml"
COURT_CORNERS_M = np.float64([[0, 0], [14, 0], [14, 11], [0, 11]])


def load_reference(path):
    """Names, pixel positions (full frame), the frame-to-court homography and the frame size."""
    ref = json.loads(path.read_text(encoding="utf-8"))
    names = list(ref["keypoints"])
    px = np.float64([ref["keypoints"][k]["px"] for k in names])
    return names, px, np.float64(ref["H_px_to_m"]), tuple(ref["frame_size"])


def flip_index(names):
    """For a mirrored picture, keypoint i becomes keypoint flip[i] (left <-> right, the rest stays)."""
    swap = {"left": "right", "right": "left"}
    mirrored = ["_".join(swap.get(part, part) for part in name.split("_")) for name in names]
    return [names.index(m) for m in mirrored]


def rotation(pan, tilt, roll):
    """Camera rotation from pan (around the vertical axis), tilt (around the horizontal axis) and roll, in degrees."""
    p, t, r = np.radians([pan, tilt, roll])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return rz @ rx @ ry


def view_homography(frame_size, out_size, focal, pan=0.0, tilt=0.0, roll=0.0, zoom=1.0):
    """3x3 matrix from full-frame pixels to pixels of the saved (smaller) view. All zeros = the frame as filmed."""
    (w, h), (ow, oh) = frame_size, out_size
    scale = ow / w
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    K_out = np.array([[focal * zoom * scale, 0, ow / 2], [0, focal * zoom * scale, oh / 2], [0, 0, 1]])
    return K_out @ rotation(pan, tilt, roll) @ np.linalg.inv(K)


def apply(M, pts):
    """Project (N, 2) points with a 3x3 matrix. Returns points and the depth sign (w > 0 = in front of the camera)."""
    p = np.c_[pts, np.ones(len(pts))] @ M.T
    return p[:, :2] / p[:, 2:], p[:, 2]


def court_box(court_px, out_size):
    """Box (x1, y1, x2, y2) around the part of the court floor inside the picture, or None."""
    ow, oh = out_size
    rect = np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]])
    area, inside = cv2.intersectConvexConvex(np.float32(court_px), rect)
    if area <= 0 or inside is None:
        return None
    inside = inside.reshape(-1, 2)
    return [*inside.min(0), *inside.max(0)]


def occluded_keypoints(px, boxes):
    """True for every keypoint (full-frame pixels) inside one of the frame's rider or bike boxes."""
    if len(boxes) == 0:
        return np.zeros(len(px), dtype=bool)
    x, y = px[:, :1], px[:, 1:]
    return ((x >= boxes.x1.values) & (x <= boxes.x2.values) & (y >= boxes.y1.values) & (y <= boxes.y2.values)).any(1)


def make_view(M, px, hidden, court_px_full, out_size):
    """Label line for one view, plus the number of keypoints inside it. None if the court is not in the picture."""
    ow, oh = out_size
    pts, depth = apply(M, px)
    inside = (depth > 0) & (pts[:, 0] >= 0) & (pts[:, 0] < ow) & (pts[:, 1] >= 0) & (pts[:, 1] < oh)
    box = court_box(apply(M, court_px_full)[0], out_size)
    if box is None:
        return None, 0, inside
    x1, y1, x2, y2 = box
    values = [0, (x1 + x2) / 2 / ow, (y1 + y2) / 2 / oh, (x2 - x1) / ow, (y2 - y1) / oh]
    for (x, y), ok, hid in zip(pts, inside, hidden):
        values += [x / ow, y / oh, 1 if hid else 2] if ok else [0, 0, 0]
    line = " ".join(str(v) if isinstance(v, int) else f"{float(v):.6f}" for v in values)
    return line, int(inside.sum()), inside


def landmark_lines(M, px, hidden, inside, out_size, box_px):
    """Labels for the landmarks layout: one line per landmark in the picture, its class = its index.
    The box is a square of box_px full-frame pixels around the landmark, warped with the view, clipped to the picture."""
    ow, oh = out_size
    half = box_px / 2
    lines = []
    for k in np.flatnonzero(inside):
        square = px[k] + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
        corners, _ = apply(M, square)
        x1, y1 = np.clip(corners.min(0), 0, [ow, oh])
        x2, y2 = np.clip(corners.max(0), 0, [ow, oh])
        x, y = apply(M, px[k:k + 1])[0][0]
        values = [int(k), (x1 + x2) / 2 / ow, (y1 + y2) / 2 / oh, (x2 - x1) / ow, (y2 - y1) / oh,
                  x / ow, y / oh, 1 if hidden[k] else 2]
        lines.append(" ".join(str(v) if isinstance(v, int) else f"{float(v):.6f}" for v in values))
    return "\n".join(lines)


def link_or_copy(src, dst):
    """Hard link (same file on disk, no extra space); copy if the file system can't."""
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def plan_views(frames, split, config, rng):
    """For each frame: view 0 as filmed, then random virtual-camera views (parameters only, no images yet)."""
    plan = []
    for frame in frames:
        plan.append((frame, 0, dict(pan=0.0, tilt=0.0, roll=0.0, zoom=1.0)))
        for v in range(1, config["extra_views"][split] + 1):
            params = dict(pan=rng.uniform(-1, 1) * config["pan_deg"], tilt=rng.uniform(-1, 1) * config["tilt_deg"],
                          roll=rng.uniform(-1, 1) * config["roll_deg"], zoom=rng.uniform(*config["zoom"]))
            plan.append((frame, v, params))
    return plan


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    out_dir = ROOT / config["output_dir"]
    names, ref_px, H_ref, frame_size = load_reference(ROOT / config["reference"])
    w, h = frame_size
    out_size = (config["image_width"], round(h * config["image_width"] / w))
    court_px_full = apply(np.linalg.inv(H_ref), COURT_CORNERS_M)[0]

    det = pd.read_csv(ROOT / config["detections"])
    occluders = det[det["class"].isin(["player", "bike"]) & (det.confidence >= config["occluder_confidence"])]
    occluders_by_frame = dict(tuple(occluders.groupby("frame")))

    n_frames = int(cv2.VideoCapture(str(ROOT / config["video"])).get(cv2.CAP_PROP_FRAME_COUNT))
    all_frames = range(1, n_frames + 1, config["frame_step"])
    is_val = lambda f: config["val_first_frame"] <= f <= config["val_last_frame"]
    plans = {}
    for split in ("train", "val"):
        frames = [f for f in all_frames if is_val(f) == (split == "val")]
        plans[split] = plan_views(frames, split, config, np.random.default_rng(config["seed"][split]))

    lm_dir = ROOT / config["landmarks_output_dir"]
    for base in (out_dir, lm_dir):
        for folder in ("images", "labels"):
            shutil.rmtree(base / folder, ignore_errors=True)
            for split in plans:
                (base / folder / split).mkdir(parents=True, exist_ok=True)

    by_frame = {}
    for split, plan in plans.items():
        for frame, v, params in plan:
            by_frame.setdefault(frame, []).append((split, v, params))

    rows, rng_retry = [], np.random.default_rng(123)
    video, frame = cv2.VideoCapture(str(ROOT / config["video"])), 0
    while frame < max(by_frame):
        ok, picture = video.read()
        if not ok:
            break
        frame += 1
        if frame not in by_frame:
            continue
        small = cv2.resize(picture, out_size, interpolation=cv2.INTER_AREA)
        to_small = np.diag([w / out_size[0], h / out_size[1], 1.0])        # small-image pixels -> full-frame pixels
        hidden = occluded_keypoints(ref_px, occluders_by_frame.get(frame, occluders.iloc[:0]))
        for split, v, params in by_frame[frame]:
            for _ in range(20):                                             # resample views that miss the court
                M = view_homography(frame_size, out_size, config["focal_px"], **params)
                line, n_visible, inside = make_view(M, ref_px, hidden, court_px_full, out_size)
                if v == 0 or (line and n_visible >= config["min_visible"]):
                    break
                params = dict(pan=rng_retry.uniform(-1, 1) * config["pan_deg"], tilt=rng_retry.uniform(-1, 1) * config["tilt_deg"],
                              roll=rng_retry.uniform(-1, 1) * config["roll_deg"], zoom=rng_retry.uniform(*config["zoom"]))
            if line is None:
                continue                                                    # no usable view found: skip it
            stem = f"frame_{frame:05d}_v{v}"
            image = small if v == 0 else cv2.warpPerspective(small, M @ to_small, out_size, flags=cv2.INTER_LINEAR,
                                                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            cv2.imwrite(str(out_dir / "images" / split / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, config["jpeg_quality"]])
            (out_dir / "labels" / split / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")
            link_or_copy(out_dir / "images" / split / f"{stem}.jpg", lm_dir / "images" / split / f"{stem}.jpg")
            (lm_dir / "labels" / split / f"{stem}.txt").write_text(
                landmark_lines(M, ref_px, hidden, inside, out_size, config["landmark_box_px"]) + "\n", encoding="utf-8")
            G = H_ref @ np.linalg.inv(M)                                    # saved-image pixels -> court metres
            G = G / G[2, 2]
            rows.append({"image": f"{stem}.jpg", "split": split, "frame": frame, "view": v,
                         **{k: round(float(val), 3) for k, val in params.items()}, "n_visible": n_visible,
                         "hidden": ";".join(n for n, hid, ins in zip(names, hidden, inside) if hid and ins),
                         **{f"g{i}": float(val) for i, val in enumerate(G.ravel())}})
    video.release()

    with open(out_dir / "views.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    data_yaml = {"train": "images/train", "val": "images/val", "kpt_shape": [len(names), 3],
                 "flip_idx": flip_index(names), "names": {0: "court"}, "kpt_names": {0: names}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    # Baseline experiment K0: train on the frames as filmed only (same val set), to show what the virtual camera adds.
    originals = [f"./images/train/{r['image']}" for r in rows if r["split"] == "train" and r["view"] == 0]
    (out_dir / "train_as_filmed.txt").write_text("\n".join(originals) + "\n", encoding="utf-8")
    (out_dir / "data_as_filmed.yaml").write_text(
        yaml.safe_dump({**data_yaml, "train": "train_as_filmed.txt"}, sort_keys=False), encoding="utf-8")
    # Landmarks layout (K2): one class per landmark, 1 keypoint each. No flip_idx: a mirrored left corner would still
    # be labelled "left", so K2 trains without left-right flips.
    lm_yaml = {"train": "images/train", "val": "images/val", "kpt_shape": [1, 3], "flip_idx": [0],
               "names": dict(enumerate(names))}
    (lm_dir / "data.yaml").write_text(yaml.safe_dump(lm_yaml, sort_keys=False), encoding="utf-8")

    table = pd.DataFrame(rows)
    print(f"Wrote {out_dir} and {lm_dir}")
    for split, g in table.groupby("split", sort=False):
        print(f"  {split}: {g.frame.nunique()} frames, {len(g)} images ({(g.view == 0).sum()} as filmed), "
              f"visible keypoints per image: median {g.n_visible.median():.0f}, min {g.n_visible.min()}")


if __name__ == "__main__":
    main()

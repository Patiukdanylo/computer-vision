"""Find the court homography of a clip automatically with the keypoint model, and save it for the app.

Run from the repo folder:  python tools/auto_homography.py                    (the sample clip)
                           python tools/auto_homography.py path/to/clip.mp4 --out models/other_homography.json

Runs models/court_keypoints.pt on 30 frames spread over the clip, takes each landmark's median position over the
frames where it was found (a rider covers a different landmark in each frame), and fits one H with RANSAC.
Writes models/court_homography.json, which app/core/homography.py uses instead of the hand-clicked points.

GenAI: written with Claude Code (Opus 5.5) on 2026-11-13.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from core.config import SAMPLE_CLIP                                  # noqa: E402
from core.court_keypoints import MODEL_PATH, court_template, homography_from_video, load_model   # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", default=str(SAMPLE_CLIP))
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", default=str(ROOT / "models" / "court_homography.json"))
    parser.add_argument("--model", default=str(MODEL_PATH), help="pose weights, e.g. a run's weights/last.pt")
    args = parser.parse_args()

    H, d = homography_from_video(args.video, args.frames, model=load_model(args.model))
    if H is None:
        raise SystemExit("Fewer than 4 landmarks found: no homography. Is the court in the picture?")
    H = H / H[2, 2]
    _, metres = court_template()
    projected = (np.c_[d["xy"], np.ones(len(metres))] @ H.T)
    error_m = np.linalg.norm(projected[:, :2] / projected[:, 2:] - metres, axis=1)
    landmarks = {name: {"px": [round(float(v), 1) for v in d["xy"][i]], "frames_found": int(d["frames_found"][i]),
                        "used": bool(d["inliers"][i]), "error_m": round(float(error_m[i]), 3),
                        "spread_px": None if np.isnan(d["spread_px"][i]) else round(float(d["spread_px"][i]), 1)}
                 for i, name in enumerate(d["names"])}
    Path(args.out).write_text(json.dumps({
        "about": "Pixel -> court metres for this clip, from the court keypoint model (tools/auto_homography.py).",
        "video": Path(args.video).name, "model": Path(args.model).name, "date": date.today().isoformat(),
        "frames_sampled": d["n_frames"], "H_px_to_m": H.tolist(), "landmarks": landmarks,
    }, indent=2), encoding="utf-8")
    used = sum(v["used"] for v in landmarks.values())
    print(f"{used} of {len(landmarks)} landmarks used; error at the used ones: "
          f"mean {np.mean([v['error_m'] for v in landmarks.values() if v['used']]):.3f} m")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

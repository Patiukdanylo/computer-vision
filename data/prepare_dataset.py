"""Build our YOLO dataset from the repo alone: the video and the label export, with a time-based train/val split.

Why we don't use the Platform's own converter:
- it renames every image to a hash (e.g. 0b094932e8cdfb47c0b310dec545aac0_1.jpg), so we lose the frame number
  and can't see which part of the video a frame comes from;
- it uses the train/val split stored on the Platform. We want the split to be a rule in our own config
  (a range of frame numbers), so neighbouring, near-identical frames can never end up in both sets.

Why the frames are cut from the video: the frames are not in git (the repo is public), but the video and the
label export are. Frame N in the labels is the Nth frame of the video (OpenCV reads it as index N - 1); we checked
this for all 73 labelled frames. So anyone who clones the repo can rebuild exactly the same dataset.

Input:  the video and the label export (configs/dataset.yaml -> video, labels_ndjson).
Output: output_dir/images/{train,val}/frame_XXXXX.jpg
        output_dir/labels/{train,val}/frame_XXXXX.txt   (YOLO format)
        output_dir/data.yaml                           (tells YOLO where the images are and the class names)

Run from anywhere:  python data/prepare_dataset.py

GenAI: written with Claude Code (Opus 5.5) on 2026-10-29, changed on 2026-10-30 to read the frames from the video.
Prompt: "Code: run data/prepare_dataset.py [...] It outputs data/cycle-ball-dataset/ with images/{train,val},
labels/{train,val} and data.yaml." What we checked or changed ourselves: _to fill in_.
"""
import json
import shutil
from pathlib import Path

import cv2
import yaml

# This file lives in data/, so the repo root is one folder up. Paths then work no matter where you run it from.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "dataset.yaml"


def read_export(ndjson_path):
    """Read the NDJSON export. Returns the class names ({0: "player", ...}) and the list of image records."""
    lines = ndjson_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]

    # The "dataset" line holds the class names. Its keys are text ("0"), YOLO wants numbers (0).
    dataset_info = next(record for record in records if record["type"] == "dataset")
    class_names = {int(class_id): name for class_id, name in dataset_info["class_names"].items()}

    images = [record for record in records if record["type"] == "image"]
    return class_names, images


def frame_number(file_name):
    """'frame_00667.jpg' -> 667"""
    return int(Path(file_name).stem.split("_")[1])


def split_of(frame, config):
    """'val' if the frame number is inside the val range from the config, otherwise 'train'."""
    return "val" if config["val_first_frame"] <= frame <= config["val_last_frame"] else "train"


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output_dir = ROOT / config["output_dir"]

    class_names, images = read_export(ROOT / config["labels_ndjson"])

    # Start from empty images/ and labels/ folders, so frames removed from the export don't stay behind.
    for folder in ("images", "labels"):
        shutil.rmtree(output_dir / folder, ignore_errors=True)

    # 1. Label files: one per labelled frame.
    counts = {"train": 0, "val": 0}
    for image in images:
        split = split_of(frame_number(image["file"]), config)
        label_dir = output_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)

        # YOLO label file: one line per box, "class x_center y_center width height",
        # all as fractions (0-1) of the image size. The export already stores boxes in exactly this order.
        boxes = image["annotations"].get("boxes", [])
        label_text = "".join(" ".join(str(value) for value in box) + "\n" for box in boxes)
        (label_dir / f"{Path(image['file']).stem}.txt").write_text(label_text, encoding="utf-8")
        counts[split] += 1

    # 2. Images: read the video frame by frame and save the frames that have labels, under their own names.
    file_names = {frame_number(image["file"]): image["file"] for image in images}
    still_needed = set(file_names)
    video = cv2.VideoCapture(str(ROOT / config["video"]))
    frame = 0
    while still_needed:
        ok, picture = video.read()
        if not ok:
            break  # end of the video
        frame += 1  # our frame numbers start at 1, so the first frame read is frame 1
        if frame in still_needed:
            image_dir = output_dir / "images" / split_of(frame, config)
            image_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_dir / file_names[frame]), picture, [cv2.IMWRITE_JPEG_QUALITY, config["jpeg_quality"]])
            still_needed.remove(frame)
    video.release()

    if still_needed:
        raise SystemExit(f"These labelled frames are not in the video: {sorted(still_needed)}")

    # No "path" key on purpose: YOLO then uses the folder data.yaml is in as the dataset root,
    # so the whole folder can be moved (e.g. uploaded to Colab) without editing this file.
    data_yaml = {"train": "images/train", "val": "images/val", "names": class_names}
    (output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print(f"Wrote {output_dir}")
    print(f"  train: {counts['train']} images, val: {counts['val']} images "
          f"(val = frames {config['val_first_frame']}-{config['val_last_frame']})")


if __name__ == "__main__":
    main()

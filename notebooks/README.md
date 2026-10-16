# Notebooks

Run in this order:

| Notebook | What it does |
|---|---|
| `01-frames-baseline` | cut the clip, extract frames, run standard YOLO on them, show what it misses (the data gap) |
| `02-dataset` | load the Roboflow export, check labels and split, write `data.yaml` |
| `03-training` | the fine-tuning runs and their mAP50 / mAP50-95 |
| `04-error-analysis` | confusion matrix, worst frames, false positives and negatives, what we changed and whether it helped |
| `05-homography` | court template, keypoints, `cv2.findHomography`, projection check |

Start each notebook with its title and who worked on it, and put a short markdown cell above each code cell saying what it does. Keep the outputs.

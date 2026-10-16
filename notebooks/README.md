# Notebooks

Run in this order:

| Notebook | What it does |
|---|---|
| `01-frames-baseline` | cut the clip, extract frames, run standard YOLO on them, show what it misses (the data gap) |
| `02-dataset` | load the Roboflow export, check labels and split, write `data.yaml` |
| `03-training` | the fine-tuning runs, each appended to `results/runs.csv` |
| `04-error-analysis` | confusion matrix, worst frames, false positives and negatives, what we changed and whether it helped |
| `05-homography` | court template, keypoints, `cv2.findHomography`, projection check |

Every notebook starts with:

```markdown
# Cycle ball: <title>

**Author:** <name>: <what you did in this notebook>
**GenAI:** <what AI helped with here, and what you decided yourself>
```

Then a markdown cell above every code cell, saying what the next cell does. Keep the outputs: the report and the oral exam refer to them.

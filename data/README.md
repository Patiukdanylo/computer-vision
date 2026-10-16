# Data

Only this file and `labelling-rules.md` are committed. The frames come from a YouTube stream we don't own and the repo is public, so clips, frames and the labelled export stay on our machines and in Roboflow.

## Source footage

Fixed elevated camera on the whole court, same framing for hours (checked 16 Oct 2026):

- "1. Bundesliga Radball 2026", Radball-Netzer Brüheim, YouTube: https://www.youtube.com/watch?v=l92Cfs0_fA8
- "Radball Bundesliga Final Five Großkoschen", same channel: https://www.youtube.com/watch?v=oUxWVCUMQnE (backup, lower angle)

Known issues in this footage: the scoreboard overlay covers the top-right corner of the court, and the camera operator and the officials' table are in the bottom of the frame.

Cite the channel and the video in the report.

## Getting a clip

Cut a section instead of downloading hours (needs ffmpeg):

```bash
yt-dlp --download-sections "*1:00:00-1:05:00" -f "bv*[height<=1080]+ba/b[height<=1080]" \
  -o "data/raw/%(id)s_%(section_start)s.%(ext)s" "https://www.youtube.com/watch?v=l92Cfs0_fA8"
```

## Local layout

```
data/raw/       clips cut from the stream
data/frames/    all extracted frames
data/dataset/   the labelled set exported from Roboflow (YOLO format, with data.yaml)
```

## Labelled dataset

Roboflow project: link to add once created. Export in YOLO (Ultralytics) format and download it into `data/dataset/`.

The train/validation split is by match segment, never random frames: consecutive frames are near-duplicates, so a random split would put the same moment in both sets. Write down which segments went where in `labelling-rules.md`.

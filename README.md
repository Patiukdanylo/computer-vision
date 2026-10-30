# computer-vision
## Sport: Cycle ball

Participants:
- Ivayla Pernikova r1028557
- Danylo-Makarii Patiuk r1028011
- Ryan Omondi r1019767
- Artem Kucheruk r1026280

Deep Learning (Computer Vision) challenge, Thomas More 2026: from a static-camera cycle ball video to a synchronized 2D tactical minimap. We fine-tune YOLO on 73 labelled frames (players, bikes, goals and the ball), project the positions onto a top-down court with a homography, and show both side by side in a Streamlit app.

## Structure

```
notebooks/   numbered notebooks, run in order
app/         Streamlit dashboard and the inference code it needs
models/      the trained weights the app uses (best.pt)
configs/     settings files (dataset paths and train/val split)
data/        in git: the video, label exports (.ndjson), labelling rules, dataset script
             local only: frames, the prepared dataset
docs/        report and slides
```

## Data

### Footage

- `data/cycle-ball.mp4`: 25 s of one cycle-ball match, filmed by one static, wide camera. Added by Danylo on 2026-10-16.
- Source: _to fill in: YouTube link, which match, and the start and end time of the cut._
- Checked with ffprobe and OpenCV:
  - 3024 × 1078, H.264, 25.0 s, 1,170 frames.
  - Variable frame rate, on average 46.784 fps (140400/3001). ffprobe's `r_frame_rate` says 120 fps; that is wrong, don't use it. OpenCV's `CAP_PROP_FPS` gives the right value.
  - About half the frames are a near-exact copy of the frame before. Keep all of them: every video we write must have the same 1,170 frames, or it goes out of sync with the others.

### Frames

`data/frames/frame_00001.jpg` … `frame_01170.jpg` (local only). Frame N is the Nth frame of the video, which is OpenCV index N − 1. Full resolution, no resizing. Only needed to upload new frames to the Platform for labelling; the training dataset is cut straight from the video (see *Get the dataset*).

```bash
ffmpeg -i data/cycle-ball.mp4 -q:v 2 -fps_mode passthrough data/frames/frame_%05d.jpg
```

- `-q:v 2`: high JPEG quality. With it, `frame_00001.jpg` comes out byte for byte the same as ours; the other 1,169 frames haven't been compared yet.
- `-fps_mode passthrough`: write every frame of the video exactly once. Without it, ffmpeg believes the 120 fps header and adds copies of frames to fill it.

### Labels

- Labelled on the Ultralytics Platform (`ivayla-pernikova/datasets/cycle-ball`), following `data/labelling-rules.md`. We don't use Roboflow.
- Every dataset version is exported as an NDJSON file and committed as `data/labelled/cycle-ball-vN.ndjson`: text only (about 50 KB), with every box of every labelled frame. The images are not in it.
- `python data/prepare_dataset.py` turns the export into a YOLO dataset in `data/cycle-ball-dataset/`: it writes the label files, cuts the labelled frames out of the video, and applies the time-based split from `configs/dataset.yaml`. `notebooks/01_dataset_eda.ipynb` runs the script and explores the result.
- The NDJSON also holds links to the images on the Platform. They expire after about a week and we don't use them: the images always come from the video.

### Get the dataset (after cloning)

Everything needed is in the repo: the video and the label export. `data/cycle-ball-dataset/` is not in git; everyone builds it locally in a few seconds.

1. Install the requirements (see *Setup*).
2. From the repo folder, run:

   ```bash
   python data/prepare_dataset.py
   ```

   It reads `data/cycle-ball.mp4` and `data/labelled/cycle-ball-v1.ndjson`, and prints `train: 48 images, val: 25 images (val = frames 1-234)`.
3. Train with `data=data/cycle-ball-dataset/data.yaml`.

On Colab, do the same in a notebook cell:

```bash
!git clone https://github.com/Patiukdanylo/computer-vision.git
%cd computer-vision
!pip install -r requirements.txt
!python data/prepare_dataset.py
```

### Dataset versions

Every model we train is logged with the dataset version it was trained on. Otherwise we can't tell whether a better score comes from the model settings or from changed labels.

| Version | Date | Frames (train / val) | Boxes | Changes |
|---|---|---|---|---|
| v1 | 2026-10-29 | 73 (48 / 25) | 785: 287 player, 279 bike, 146 goal, 73 ball | First full set. Frame 749: removed a `player` box on a person outside the court. Checked by eye on 2026-10-30, nothing to change: the ball boxes in frames 595 and 53, and the bikes in val frames 22, 32, 34, 39, 53, 80, 102. |

### Export a new dataset version (data lead)

1. On the Platform, save a version (Versions tab) and download its NDJSON.
2. Save it as `data/labelled/cycle-ball-vN.ndjson` next to the older ones (never overwrite a version), and point `labels_ndjson` in `configs/dataset.yaml` to it.
3. Run `python data/prepare_dataset.py` and the EDA notebook, and check the counts.
4. Add a row to the table above: what changed and why.
5. Commit the new NDJSON, the config and this README together, e.g. `feat(data): add dataset v2`. After pulling, teammates only rerun `python data/prepare_dataset.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training runs in Google Colab with a GPU runtime; save `best.pt` to Drive after every run.

## Workflow

- One branch per task (`feat/…`, `fix/…`, `docs/…`), a pull request into `main`, and one teammate reviews before merging.
- Notebooks: numbered in run order, a short markdown cell above each code cell, outputs kept.
- The report must end with a GenAI Disclosure (course policy): which parts AI wrote, 3–5 real prompts, and how we checked its output.

## Upload (Canvas, Sun 15 Nov 2026, 18:00)

One zip per team: the PDF report, the notebooks, the code and weights to run the app, and the slides.

```bash
zip -r cv-challenge-upload.zip docs/report.pdf docs/slides.pptx notebooks app models configs \
  data/prepare_dataset.py data/labelling-rules.md requirements.txt README.md \
  -x "*.ipynb_checkpoints*" "*/__pycache__/*"
```

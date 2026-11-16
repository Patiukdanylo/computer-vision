# computer-vision
## Sport: Cycle ball

Participants:
- Ivayla Pernikova r1028557 (role A: data and labelling; court keypoint model)
- Danylo-Makarii Patiuk r1028011 (role D: Streamlit app, homography, tracking)
- Ryan Omondi r1019767 (role C: error analysis, team clustering)
- Artem Kucheruk r1026280 (role B: modeling)

Deep Learning (Computer Vision) challenge, Thomas More 2026: from a static-camera cycle ball video to a synchronized 2D tactical minimap. We fine-tune YOLO on 73 labelled frames (players, bikes, goals and the ball), project the positions onto a top-down court with a homography, and show both side by side in a Streamlit app.

## Structure

```
notebooks/   numbered notebooks, in the order of the story (see below)
app/         Streamlit dashboard and the inference code it needs
tools/       scripts: build the datasets, detections for a clip, automatic homography, point picker
models/      best.pt (champion detector), court_keypoints.pt (court keypoints), the homography,
             the sample clip's detections and teams, archive/ (the first model, for notebook 05)
configs/     settings files (dataset paths and train/val split, keypoint dataset)
data/        the video, label exports (.ndjson), labelling rules, the prepared dataset
docs/        report and slides
```

## Notebooks, in order

1. `01_dataset_eda.ipynb`: build the YOLO dataset from our labels and explore it (the ball is tiny, the split is by time).
2. `02_baseline_finetune.ipynb`: E0 (pretrained, no training) and E1 (first fine-tune at 640).
3. `03_experiments_e1_e3_e6.ipynb`: one change at a time: seeds (the noise yardstick), freezing, augmentation, epochs, model size.
4. `04_image_size_e2_colab.ipynb`: image size 640 / 960 / 1280 on a Colab GPU. 960 wins; this is the champion `models/best.pt`.
5. `05_error_analysis.ipynb`: what the model still gets wrong, before (640) and after (960), and why.
6. `06_team_clustering.ipynb`: team colours from the shirts (HSV + K-Means), Extra Mile.
7. `07_homography_minimap.ipynb`: pixel boxes to court metres, hand-clicked homography.
8. `08_automated_keypoints.ipynb`: a keypoint model finds the court lines itself, Extra Mile.

Notebooks 03 and 04 were run on a Mac and on Colab; their saved outputs are the record (their run folders are not part of the upload). 01, 05, 06 and 07 re-run from this folder; 08 trains models, so only its saved outputs are the record.

## Data

### Footage

- `data/cycle-ball.mp4`: 25 s of one cycle-ball match, filmed by one static, wide camera. Added by Danylo on 2026-10-16.
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
- `python tools/prepare_dataset.py` turns the export into a YOLO dataset in `data/cycle-ball-dataset/`: it writes the label files, cuts the labelled frames out of the video, and applies the time-based split from `configs/dataset.yaml`. `notebooks/01_dataset_eda.ipynb` runs the script and explores the result.
- The NDJSON also holds links to the images on the Platform. They expire after about a week and we don't use them: the images always come from the video.

### Get the dataset (after cloning)

Everything needed is in the repo: the video and the label export. `data/cycle-ball-dataset/` is not in git; everyone builds it locally in a few seconds. In this upload it is already built.

1. Install the requirements (see *Setup*).
2. From the repo folder, run:

   ```bash
   python tools/prepare_dataset.py
   ```

   It reads `data/cycle-ball.mp4` and `data/labelled/cycle-ball-v1.ndjson`, and prints `train: 48 images, val: 25 images (val = frames 1-234)`.
3. Train with `data=data/cycle-ball-dataset/data.yaml`.

On Colab, do the same in a notebook cell:

```bash
!git clone https://github.com/Patiukdanylo/computer-vision.git
%cd computer-vision
!pip install -r requirements.txt
!python tools/prepare_dataset.py
```

### Dataset versions

Every model we train is logged with the dataset version it was trained on. Otherwise we can't tell whether a better score comes from the model settings or from changed labels.

| Version | Date | Frames (train / val) | Boxes | Changes |
|---|---|---|---|---|
| v1 (called `v1b` in run names and the report) | 2026-10-29 | 73 (48 / 25) | 785: 287 player, 279 bike, 146 goal, 73 ball | First full set. Frame 749: removed a `player` box on a person outside the court. Checked by eye on 2026-10-30, nothing to change: the ball boxes in frames 595 and 53, and the bikes in val frames 22, 32, 34, 39, 53, 80, 102. |

### Export a new dataset version (data lead)

1. On the Platform, save a version (Versions tab) and download its NDJSON.
2. Save it as `data/labelled/cycle-ball-vN.ndjson` next to the older ones (never overwrite a version), and point `labels_ndjson` in `configs/dataset.yaml` to it.
3. Run `python tools/prepare_dataset.py` and the EDA notebook, and check the counts.
4. Add a row to the table above: what changed and why.
5. Commit the new NDJSON, the config and this README together, e.g. `feat(data): add dataset v2`. After pulling, teammates only rerun `python tools/prepare_dataset.py`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Training runs in Google Colab with a GPU runtime; save `best.pt` to Drive after every run.

## Run the app

Start it from the repo folder, so it finds `.streamlit/config.toml`:

```bash
source .venv/bin/activate
streamlit run app/main.py
```

It opens at http://localhost:8501. Upload a cycle-ball clip (mp4, H.264, one static camera that shows the whole court) or press *Or try the sample clip*, then *Process clip*. The app runs the champion detector on every frame, finds the court lines with the keypoint model, maps riders and ball onto a 14 x 11 m court and plays video and map together. Processing the 25 s sample takes about 20 s on an Apple GPU or CUDA and about 1 minute on a CPU.

`models/detections.csv` and `models/teams.csv` are the sample clip's results; only the notebooks read them, the app computes everything itself. To recreate them: `python tools/make_detections.py`, then run notebook 06.

## Workflow

- One branch per task (`feat/…`, `fix/…`, `docs/…`), a pull request into `main`, and one teammate reviews before merging.
- Notebooks: numbered in run order, a short markdown cell above each code cell, outputs kept.
- The report must end with a GenAI Disclosure (course policy): which parts AI wrote, 3–5 real prompts, and how we checked its output.

## Upload (Canvas, Sun 15 Nov 2026, 18:00)

This folder is the upload: zip its content as it is (one zip per team). `docs/` holds `report.pdf` and `presentation.pptx`.

```bash
cd <this folder> && zip -r ../cv-challenge-upload.zip . -x "*.DS_Store" "*/__pycache__/*" "app/static/*" "outputs/*"
```

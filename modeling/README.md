# Modeling (Role B)

**Owner:** Artem Kucheruk — Modeling lead
**Rubric weight:** Modeling & Fine-Tuning Core — **35%**
**Status (29 Oct):** E0 baseline and E1 first fine-tune done on dataset v1, in [`01_baseline_finetune.ipynb`](01_baseline_finetune.ipynb). Results are in the [experiment log](#9-experiment-log-map-log), the dated history is in [PROGRESS.md](PROGRESS.md). Next: E2 (image size 1280) on Colab, and a re-run on dataset v2 once data cleaning is done.

> Sport: **cycle ball** (2 vs 2 on bicycles, indoor court 14 × 11 m, goals 2 × 2 m; ball played with the bike and the head, except when defending the goal). Static camera. Framework: **Ultralytics YOLO11**.
>
> ⚠️ Cycle ball is **not** in the brief's list of allowed sports (soccer, basketball, tennis, ice/field hockey). It fits the stated condition ("action is predominantly planar"), but get the instructor's OK in writing.

---

## 1. Scope

From the team split:

> Baseline test of pretrained YOLO, fine-tuning, experiments (image size, augmentations, frozen layers, epochs), mAP log.

From the brief (MVP §2): fine-tune a YOLO detector on the annotated dataset and evaluate with **mAP@50** and **mAP@50:95**.

**Not mine:** labeling, split, dataset versions (A); FP/FN gallery, confusion-matrix interpretation, failure hypotheses (C); homography, Streamlit, tracking (D). I produce the raw material C and D need.

### What the grader checks → where it is covered

| Rubric item | Covered in |
|---|---|
| Structured experimentation beyond defaults | §5 Experiment plan |
| Justified augmentations | §6 Augmentations |
| Rigorous validation metric tracking (mAP) | §9 Experiment log |
| *Oral:* vision architecture | §8.1 |
| *Oral:* training mechanics | §8.3 |
| *Oral:* loss dynamics | §7, §8.2 |
| *Oral:* rationale behind tuning choices | Hypothesis column in §5 and §9 |

---

## 2. Interfaces with the team

### Inputs from A (Data lead) — requirements

| Need | Why it matters for modeling |
|---|---|
| A **Version** (immutable snapshot) of the Ultralytics Platform dataset for every dataset I train on — v1, v2… | Every log row must name the dataset version, or numbers are not comparable. |
| Split **by clip / time segment**, not random frames | Adjacent frames are near-duplicates. A random split leaks them into val and inflates mAP. |
| Frames kept at **native resolution** (no resizing when extracting or exporting) | Resizing — especially stretching to a square — distorts aspect ratio and shrinks a 1080p ball to a few pixels before training. It also makes the image-size experiment (E2) pointless. |
| **No pre-augmented copies** in the dataset | Augmentation is done in Ultralytics (§6). Doing it twice confounds the augmentation experiments. |
| A small **held-out test clip** (10–15 frames from a different segment), not used for any decision | Configs are selected on val, so val mAP is optimistic. The test set is evaluated once, at the end. |
| Class set agreed before labeling (§3) | Determines whether one model is enough. |

### Outputs to C (Error analysis)

Per experiment, the Ultralytics run folder:
`results.csv`, `confusion_matrix.png` / `confusion_matrix_normalized.png`, PR/F1 curves, `val_batch*_pred.jpg`, and `predictions.json` (`save_json=True`), plus `best.pt` so C can run it on new frames.

C sends back failure hypotheses → they become the next experiments (column *Triggered by* in §9).

### Outputs to D (Pipeline)

- `best.pt` of the chosen model, class names. Current model: `runs/modeling/E1_640_v1/weights/best.pt`, `imgsz=640`, about 8 ms per frame on an M4 Mac (`mps`).
- Recommended `imgsz`, `iou`, and **per-class `conf`**.
- **`detections.csv`**, one row per box. Format agreed with D on 29 Oct:

  ```
  frame,class,confidence,x1,y1,x2,y2
  ```

  - `frame`: starts at 1, runs 1–1170. Frame N is the Nth frame of `cycle-ball.mp4` (OpenCV index N−1) and matches the dataset file names (`frame_00960.jpg` = frame 960). Checked against the decoded video.
  - `class`: `player` (rider only), `bike` or `ball`. Goals are left out.
  - `confidence`: boxes with confidence ≥ 0.25.
  - `x1, y1, x2, y2`: box corners in pixels of the original 3024 × 1078 frame.
  - Frames with no detections have no rows.
- **Map point:** the bottom-centre of the **bike** box, where the wheels touch the floor. The `player` box is the rider only and ends at the pedals; its bottom-centre is the fallback when no bike is detected (bike recall is lower: 0.67 vs 0.89 on val). For the ball, use the box centre, which is only correct when the ball is on the floor.
- **Annotated video:** all 1,170 frames in order, 3024 × 1078, written at **46.784 fps** (140400/3001), so it lasts exactly 25.0 s. The file header says 120 fps, which is wrong: the video has a variable frame rate. OpenCV's `CAP_PROP_FPS` reports the correct 46.784. Keep duplicate frames (e.g. frames 12 and 13), or the sync with the minimap breaks.

---

## 3. Key design decision: class set and catastrophic forgetting

The brief says: use a pretrained detector for players/referees, and annotate the data gap (the ball).

**Trap:** fine-tuning replaces the COCO head with a new one for our `nc` classes. If the 40–80 frames only have ball boxes, every player in those frames is unlabeled, so the model learns "player = background" and **loses player detection**.

**Cycle ball specifics:**

- COCO has `person` and `bicycle` as **separate** classes. A cycle-ball player is one rider on one bike. The pretrained model gives two boxes per player, and its `person` box ends at the pedals, not at the floor.
- Only 4 players + 1 ball per frame (plus maybe a referee), so labeling every object is cheap.
- The goalkeeper is a role, not a kit: both teammates wear the same jersey and swap roles. So there's no `goalkeeper` class. Team identity is not a detection class either: C clusters jersey colours (k = 2).

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. One model, all classes labeled** *(recommended)* | Classes `ball`, `player` (**rider + bike in one box**), plus `referee` only if referees stand on the court in our footage. Players are pre-labeled with the Ultralytics Platform's smart annotation (SAM) and corrected by hand; the ball is labeled manually. | One box per player whose bottom edge = wheels on the floor (what D needs); one inference pass. | Model must learn the new `player` shape (cheap: ~4 per frame). |
| B. Pretrained `person` + fine-tuned ball-only model | COCO `person` (`classes=[0]`) for players + a **ball-only** model | No forgetting risk | The `person` box excludes the bike, so its bottom edge is the pedals, not the floor → court positions are off; 2× inference cost. Not recommended for cycle ball. |

**Decision (dataset v1):** A labelled **`player` (rider only), `ball`, `bike` and `goal` as separate classes**, which is neither option above. Everything is labelled, so there is no forgetting risk. Consequences:
- `player` and `bike` map 1:1 to COCO `person` and `bicycle`, so the E0 baseline is like-for-like on every class except `goal`.
- The court position comes from the bottom of the `bike` box (see §2, outputs to D).
- `goal` is static in a fixed camera, so it is near-perfect and pushes up the all-class mAP. Compare per class.

---

## 4. Baseline (E0): pretrained YOLO, no training

Purpose: the **"before"** of the incremental delta. Same val split as every later experiment.

1. Model: COCO-pretrained `yolo11s.pt` — same size as the fine-tune reference, so the delta measures the data, not the model size.
2. Mapping to COCO ids: player → `person` (0), bike → `bicycle` (1), ball → `sports ball` (32). `goal` has no COCO class and is dropped from the baseline.
3. `yolo val` matches by class id, so the notebook makes a remapped copy of val (`runs/modeling/E0_data_v1/`) with a yaml holding COCO's 80 names. Implementation: [`01_baseline_finetune.ipynb`](01_baseline_finetune.ipynb) §2. AP is computed only for classes present in the ground truth, so the other 77 COCO classes don't affect the score.

**Result (v1):** see the log in §9. `person` was the *worst* class (AP50 0.18): COCO boxes the officials' table, spectators and the referee. `sports ball` gets AP50 0.73 even though recall is 0: it finds 11/13 balls but only with confidence 0.002–0.06, and AP depends only on ranking. In practice the baseline is useless for the ball.

**Comparability:** all three classes compare like-for-like with E1. The E1 all-class mAP also includes `goal`, so compare the 3-class mean.

---

## 5. Experiment plan

### Rules

- **One factor at a time** against the current best config (the "champion").
- Write the **hypothesis before the run**, and the verdict after.
- Same dataset version and val split across compared rows. If A releases a new version, re-run E0 and the champion on it.
- **Val is tiny → mAP is noisy.** With ~15 val balls, one ball ≈ 6–7 AP points. Differences under ~3 points are not evidence. Re-run the top two configs with 3 seeds and report mean ± std.
- **Selection metric:** mean **AP@50:95 over ball, player and bike**. *(Changed 29 Oct: the plan was ball AP@50, but E1 already reaches 0.995 on it, so it can no longer tell configs apart. `goal` is excluded because it is near-perfect.)* Always report ball AP@50:95, AP@50 and the all-class mAP too.
- **Run names carry the dataset version** (`E1_640_v1`). The notebook skips training when `best.pt` already exists, so a new dataset version needs a new name.

### Reference fine-tune (E1, pure defaults)

In [`01_baseline_finetune.ipynb`](01_baseline_finetune.ipynb) §3. Defaults: `imgsz=640`, `epochs=100`, `patience=100`, `batch=16`, `freeze=None`, default augmentation, `optimizer=auto`. Device: `device=mps` on Mac, `device=0` on Colab/Kaggle GPU. Record the device and Ultralytics version in the log.

**Hardware limit:** the Mac (M4, 16 GB) handles 640 (100 epochs ≈ 24 min). At `imgsz=1280` it ran out of memory and swapped (~12 min per batch, i.e. days per run), so E2 and every later 1280 run go on Colab.

### Sweep

| ID | Factor | Values | Hypothesis | Look at |
|---|---|---|---|---|
| E0 | Baseline, no training | `yolo11s.pt` @640, @1280 | Finds people, misses the ball | per-class AP |
| E1 | Defaults fine-tune | as above | Large ball gain over E0 | Δ vs E0 |
| E2 | Image size | 640 → 960 → 1280 (Colab) | The 3024 px frame shrinks 4.7× at 640, so the median val ball (28 px) becomes ~6 px, below the stride-8 grid (§8.1). *Updated after E1:* ball AP50 is already 0.995 at 640, so the expected gain is in **AP50-95** (box precision) for ball and bike. Evidence: the untrained E0 improves from 640 to 1280 on ball (AP50 0.73 → 0.93) and bike (AP50-95 0.39 → 0.58). | ball and bike AP50-95, train time, ms/frame |
| E3 | Frozen layers | `freeze=0` / `11` (backbone) / `23` (all but head) | ~50 images → freezing reduces overfitting. Counter-hypothesis: COCO features are weak for a blurred tiny ball, so full fine-tuning wins. | train vs val loss gap, ball AP |
| E4 | Augmentations | see §6 | Each change has its own hypothesis in §6 | ball AP, FN types from C |
| E5 | Epochs | 50 / 100 / 200 with `patience=50` | Small data converges fast; long runs overfit | epoch of best fitness, val loss curve |
| E6 | Model size | n / s / m | Bigger → higher AP, slower app | AP vs ms/frame (D's constraint) |
| F | Final | champion × 3 seeds | — | mean ± std on val, then **test once** |

Order E2 → E6 because image size has the largest expected effect on the ball and interacts with mosaic (E4).

**Freeze indices (YOLO11):** layers 0–10 are the backbone (ends with SPPF at 9, C2PSA at 10), 11–22 the neck, 23 the Detect head. `freeze=N` freezes `model.0` … `model.N-1`. Check `ultralytics/cfg/models/11/yolo11.yaml` if using another version.

---

## 6. Augmentations (with justification)

Static camera, flat indoor court, small fast ball that is often pressed against a wheel, two teams in different jerseys.

| Param (default) | Plan | Reason |
|---|---|---|
| `fliplr` (0.5) | keep | Court is left-right symmetric. |
| `flipud` (0.0) | keep 0 | Camera is never upside down; vertical perspective cue is real. |
| `degrees` (0) | keep 0 (≤5 max) | Static camera, fixed horizon. |
| `hsv_h` (0.015) | keep small | Ball colour is a key cue for telling the ball from wheel hubs and floor marks; a large hue shift hides it. |
| `hsv_s` (0.7), `hsv_v` (0.4) | keep; test `hsv_v` higher if C finds lighting/reflection errors | Hall lighting differs between venues; polished floors reflect lights. |
| `scale` (0.5) | test 0.3 vs 0.5 vs 0.7 | Models ball size change with depth; strong zoom-out can push the ball below a few pixels. |
| `mosaic` (1.0), `close_mosaic` (10) | test `mosaic` 1.0 vs 0.5, `close_mosaic` 10 vs 30 | Mosaic multiplies context variety (good for small data) but downsizes each tile ~½ → ball even smaller. Interacts with E2. |
| `mixup` (0.0) | keep 0 | Blended ghost players give confusing targets. |
| `translate` (0.1) | keep | Harmless. |
| Motion blur | only if C's FN gallery shows blurred-ball misses dominate | The built-in Albumentations blur runs at p=0.01 — negligible. A stronger motion blur needs a custom Albumentations transform via the Python API. |

---

## 7. Reading the curves (loss dynamics)

`results.csv` columns: `train/{box,cls,dfl}_loss`, `val/{box,cls,dfl}_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `lr/pg0..2`.

| Pattern | Meaning | Action |
|---|---|---|
| train ↓, val ↓ then ↑ | Overfitting | `best.pt` already saved at best fitness; try freeze / more aug / fewer epochs |
| train and val both flat and high | Underfitting, frozen too much, or LR too low | Unfreeze, longer run |
| Step change in the last `close_mosaic` epochs | Mosaic switched off → training images now look like val | Normal; loss usually drops |
| High `cls_loss` early | New classification head starts from random init | Normal during warmup |
| Noisy val curves | Tiny val set | Compare seeds, not single runs |
| mAP plateaus while val loss still falls | Localization still improving below the IoU thresholds | Fine |

---

## 8. Oral defense notes

### 8.1 Architecture (YOLO11 detect)

- **Backbone:** conv + C3k2 (CSP-style) blocks, **SPPF** (pooling at several receptive fields), **C2PSA** (attention block). Produces feature maps at **strides 8, 16, 32** (P3, P4, P5).
- **Neck:** PAN-FPN. Top-down then bottom-up fusion, so the fine P3 map (small objects) gets semantic information from deep layers.
- **Head:** **anchor-free, decoupled**. Separate branches for class scores and box. Each grid point predicts the distances to the four box sides, each as a discrete distribution (`reg_max=16` bins).
- **Output:** at 640 input, 80×80 + 40×40 + 20×20 = **8400 candidates**, then NMS.
- **Why the ball is hard:** at stride 8 one P3 cell covers 8×8 px. A 4 px ball is half a cell. After three stride-2 downsamplings its features are mostly mixed with the floor and the wheel it's touching.

### 8.2 Loss

| Term | What | Default weight |
|---|---|---|
| `box_loss` | **CIoU** — overlap + centre distance + aspect ratio | 7.5 |
| `dfl_loss` | **Distribution Focal Loss** — sharpens the per-side distance distributions → better localization | 1.5 |
| `cls_loss` | **BCE** per class (sigmoid, not softmax) | 0.5 |

**Assignment — Task-Aligned Assigner (TAL):** for each ground-truth box, candidate anchor points must lie **inside the box**; among them the top-k (k=10) by `score^0.5 · IoU^6` become positives.
→ Consequence for the ball: anchor points sit every 8 px on P3. **A ball smaller than the stride may contain no anchor point and gets no positive sample at all** — the model never gets trained on it. A 4×4 px ball contains a P3 anchor point only about (4/8)² ≈ 25% of the time. At `imgsz=1280` the same ball is ~8 px and always contains one. This is the mechanism behind the E2 hypothesis.

### 8.3 Training mechanics

- **Transfer learning:** backbone and neck start from COCO; the Detect head's class output is re-initialized for our `nc`.
- **`optimizer=auto`:** for short runs (<10k iterations) Ultralytics picks **AdamW** with an auto LR `≈ 0.002·5/(4+nc)` (≈0.0013 for 4 classes) and **ignores `lr0`**. To experiment with LR, set `optimizer=SGD` or `AdamW` explicitly.
- **Gradient accumulation:** nominal batch `nbs=64`. With `batch=16`, gradients accumulate over 4 batches. ~50 train images = 4 batches/epoch → **about 1 optimizer step per epoch**. 100 epochs ≈ 100 steps — reason to test longer runs (E5).
- **Warmup:** `max(3 epochs, 100 iterations)`. With 4 batches/epoch that is **~25 epochs** of warmup, which explains slow early progress on small data.
- **LR schedule:** linear decay to `lrf=0.01 × lr`.
- **EMA** of weights is used for validation and saving.
- **`close_mosaic=10`:** mosaic off for the last 10 epochs.
- **`best.pt` selection:** fitness = **mAP50-95 over all classes** (weights `[0, 0, 0, 1]` in Ultralytics 8.4.192, `utils/metrics.py`; older versions used `0.1·mAP50 + 0.9·mAP50-95`). It is not ball-specific, and it includes `goal`.

*(Accumulation, warmup, and auto-optimizer details are from `ultralytics/engine/trainer.py`. Re-check them in the installed version before the defense.)*

### 8.4 Metrics

- **Precision / recall** at a confidence threshold; a prediction is a TP if IoU ≥ threshold with an unmatched GT box of the same class.
- **AP** = area under the precision-recall curve (101-point interpolation). **mAP@50** = mean over classes at IoU 0.5. **mAP@50:95** = also averaged over IoU 0.50, 0.55 … 0.95 → rewards precise boxes.
- Val runs at `conf=0.001` to trace the full PR curve, so mAP is **threshold-independent**. The app's `conf` is a separate operating-point choice (handed to D).
- **Tiny boxes and IoU:** a 10×10 px ball box off by 2 px in x and y gives IoU = 64/136 ≈ **0.47** — a miss even at IoU 0.5. So ball AP@50:95 is dominated by **pixel-level label precision**. That is a direct link from annotation quality (A) to the metric, and a likely FP+FN pair source for C.

### 8.5 Questions to be ready for

- Why did fine-tuning beat the pretrained model on the ball but maybe not on people?
- Why this image size / freeze setting / number of epochs? → point at the log row and its hypothesis.
- What does each loss term measure, and what did its curve tell you?
- Why is mAP@50:95 so much lower than mAP@50 for the ball?
- How do you know the improvement is not noise? → seeds, val size, held-out test.
- Why does the E0 ball have recall 0 but AP50 0.73? → P/R are measured at one confidence threshold, while AP uses only the ranking. The ball boxes are correct but have confidence ≤ 0.06 (notebook §2).
- Why was the pretrained `person` class so bad? → it boxes the officials' table, spectators and the referee (precision 0.18).

---

## 9. Experiment log (mAP log)

Pull per-class numbers:

```python
from ultralytics import YOLO

m = YOLO("runs/modeling/E2_1280/weights/best.pt")
r = m.val(data="data.yaml", imgsz=1280, split="val")
print(f"all  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}")
for i, c in enumerate(r.box.ap_class_index):
    print(f"{m.names[c]:<10} AP50={r.box.ap50[i]:.3f}  AP50-95={r.box.ap[i]:.3f}")
```

| ID | Date | Data ver | Model | imgsz | Epochs (best) | Freeze | Aug Δ | Seed | mAP50 all | mAP50-95 all | Ball AP50 | Ball AP50-95 | Player AP50 | Bike AP50 | Triggered by | Hypothesis | Verdict / next |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0a | 29 Oct | v1 | yolo11s (COCO) | 640 | — | — | — | — | 0.525 | 0.315 | 0.727 | 0.468 | 0.176 (person) | 0.673 | brief | Pretrained misses ball | Partly wrong: ball found but confidence ≤ 0.06 (R = 0 at a usable threshold). Player is worst: officials, spectators, referee. |
| E0b | 29 Oct | v1 | yolo11s (COCO) | 1280 | — | — | — | — | 0.629 | 0.447 | 0.925 | 0.663 | 0.160 (person) | 0.803 | brief | Upscaling alone helps ball a bit | Confirmed, strongly: ball +0.20 AP50, bike AP50-95 +0.19. Motivates E2. |
| E1 | 29 Oct | v1 | yolo11s | 640 | 100 (65) | — | default | 0 | 0.936 | 0.687 | 0.995 | 0.697 | 0.935 | 0.818 | — | Defaults beat E0; ball stays weakest at ~6 px | Beats E0 on every class (3-class mean AP50 0.525 → 0.916). Ball *not* the weakest: refuted. Bike weakest (R 0.67). Next: E2 @1280 on Colab. |

*mAP "all" = the classes present in that run (E0: ball, player, bike; E1 adds goal). Mac M4 `mps`, Ultralytics 8.4.192, torch 2.15. E1 trained in 23.7 min. E1 @1280 on the Mac was aborted after 3 batches (out of memory, see §5).*

---

## 10. Checklist

- [x] Class set: A chose `player` / `ball` / `bike` / `goal` (§3).
- [x] Environment: `.venv`, Ultralytics 8.4.192, torch 2.15, `mps`.
- [x] E0 baseline @640 and @1280 → log.
- [x] E1 defaults → log.
- [ ] Hand the run folders (`runs/modeling/E0_*_v1`, `E1_640_v1`) to C. Feedback for A: wrong `player` box in `frame_00749`; 11 train frames with fewer than 4 bikes labelled.
- [ ] D: generate `detections.csv` and the annotated video (format agreed in §2).
- [ ] Dataset v2 (after cleaning) → re-run E0 and E1 with `VERSION = "v2"`.
- [ ] Ask A for a held-out test clip; v1 has none.
- [ ] E2 @1280 on Colab, then E3–E6, each row with a hypothesis; feed in C's hypotheses.
- [ ] Final config × 3 seeds; evaluate the held-out test set once.
- [ ] Hand-off to D: final `best.pt`, `imgsz`, per-class `conf`, ms/frame.
- [ ] Defense: rehearse §8 and be able to justify every log row without notes.

## 11. Artifacts

Dataset, `runs/` and `*.pt` weights are not committed (size).

- **Dataset source of truth:** Ultralytics Platform. For each version used in the log: Versions tab → download NDJSON → convert → train locally.

  ```python
  import asyncio
  from ultralytics.data.converter import convert_ndjson_to_yolo
  asyncio.run(convert_ndjson_to_yolo("dataset_v1.ndjson"))  # in a notebook: await convert_ndjson_to_yolo(...)
  ```

  Training straight from `data=ul://<user>/datasets/<name>` (needs `ULTRALYTICS_API_KEY`) is fine for quick tries. It may not be pinned to a version, so don't log those runs.
- **Dataset backup:** zip the converted folder as `dataset_vN.zip` on S3, never overwritten. The NDJSON alone is not a backup: its image URLs are signed and expire.
- **Weights + run folders:** S3, named by log ID (`E2_1280/`), so every log row points to its files.

## 12. GenAI disclosure

| Date | Tool | Used for | My review |
|---|---|---|---|
| 2026-10-27 | Claude (Claude Code) | First draft of this modeling plan from the brief | |
| 2026-10-29 | Claude (Claude Code) | Dataset v1 check; notebook `01_baseline_finetune.ipynb` (code, explanations, conclusions); running E0/E1; checking the video's fps and frame numbering; the CSV format for D; updating this plan and PROGRESS.md | |

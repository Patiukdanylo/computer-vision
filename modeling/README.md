# Modeling (Role B)

**Owner:** Artem Kucheruk — Modeling lead
**Rubric weight:** Modeling & Fine-Tuning Core — **35%**
**Status (13 Nov):** E0 and E1 on dataset `v1b` in [`notebooks/02_baseline_finetune.ipynb`](../notebooks/02_baseline_finetune.ipynb). E1 noise (3 seeds), NMS IoU, E3 (freeze), E4 (augmentation), E5 (epochs) and E6 (model size) at 640 on the Mac in [`notebooks/07_experiments_e3_e6.ipynb`](../notebooks/07_experiments_e3_e6.ipynb). E2 (image size) and `yolo26m` at batch 16 on a Colab T4 in [`notebooks/06_e2_image_size_colab.ipynb`](../notebooks/06_e2_image_size_colab.ipynb). **Champion: `yolo26s` at `imgsz=960`**, selection metric 0.528 ± 0.014 over 3 seeds vs 0.468 ± 0.012 at 640, the gain mostly on the ball (AP50-95 0.39–0.42 vs 0.23–0.29). 1280 is no better than 960, and the Mac's `yolo26m` gain did not reproduce at batch 16. Results are in the [experiment log](#9-experiment-log-map-log). Next: error analysis (notebook 03) on the champion, hand-off to D at `imgsz=960`, held-out test clip, dataset v2.

> Sport: **cycle ball** (2 vs 2 on bicycles, indoor court 14 × 11 m, goals 2 × 2 m; ball played with the bike and the head, except when defending the goal). Static camera. Framework: **Ultralytics YOLO26** (differences from YOLO11 in §8). The first E0/E1 runs used YOLO11.
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

- `best.pt` of the chosen model, class names. Current model: `models/best.pt` (copy of `runs/modeling/E1_yolo26s_640_v1b/weights/best.pt`), `imgsz=640`, about 7–11 ms per frame on an M4 Mac (`mps`).
- Recommended `imgsz`, `iou`, and **per-class `conf`**.
- **`detections.csv`** (`models/detections.csv`, written by notebook §7), one row per box. Format agreed with D on 29 Oct:

  ```
  frame,class,confidence,x1,y1,x2,y2
  ```

  - `frame`: starts at 1, runs 1–1170. Frame N is the Nth frame of `cycle-ball.mp4` (OpenCV index N−1) and matches the dataset file names (`frame_00960.jpg` = frame 960). Checked against the decoded video.
  - `class`: `player` (rider only), `bike` or `ball`. Goals are left out.
  - `confidence`: boxes with confidence ≥ 0.25.
  - `x1, y1, x2, y2`: box corners in pixels of the original 3024 × 1078 frame.
  - Frames with no detections have no rows.
- **Map point:** the bottom-centre of the **bike** box, where the wheels touch the floor. The `player` box is the rider only and ends at the pedals; its bottom-centre is the fallback when no bike is detected (bike recall is lower: 0.79 vs 0.90 on val). For the ball, use the box centre, which is only correct when the ball is on the floor.
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

1. Model: COCO-pretrained `yolo26s.pt` (`yolo11s.pt` for the first runs) — same size as the fine-tune reference, so the delta measures the data, not the model size.
2. Mapping to COCO ids: player → `person` (0), bike → `bicycle` (1), ball → `sports ball` (32). `goal` has no COCO class and is dropped from the baseline.
3. `yolo val` matches by class id, so the notebook makes a remapped copy of val (`runs/modeling/E0_data_v1/`) with a yaml holding COCO's 80 names. Implementation: [`notebooks/02_baseline_finetune.ipynb`](../notebooks/02_baseline_finetune.ipynb) §2. AP is computed only for classes present in the ground truth, so the other 77 COCO classes don't affect the score.

**Result (`v1b`):** see the log in §9. `sports ball` is the *worst* class (AP50 0.147, recall 0.05): it finds 8 of the 25 balls with well-placed boxes, only 3 of them with confidence above 0.1. In practice the baseline is useless for the ball. `person` is next (AP50 0.33, precision 0.23): in the first runs' prediction images COCO boxes the officials' table, spectators and the referee. `bicycle` is fine (AP50 0.85). On the old split the ball looked much better (AP50 0.73 with YOLO11, 0.84 with YOLO26) because val was one resting ball.

**Comparability:** all three classes compare like-for-like with E1. The E1 all-class mAP also includes `goal`, so compare the 3-class mean.

---

## 5. Experiment plan

### Rules

- **One factor at a time** against the current best config (the "champion").
- Write the **hypothesis before the run**, and the verdict after.
- Same dataset version and val split across compared rows. If A releases a new version, re-run E0 and the champion on it.
- **Val is tiny → mAP is noisy.** With 25 val balls, one ball ≈ 4 AP points. Differences under ~3 points are not evidence. Re-run the top two configs with 3 seeds and report mean ± std.
- **Selection metric:** mean **AP@50:95 over ball, player and bike**. *(Changed 29 Oct, when the planned ball AP@50 looked saturated at 0.995. That value came from the old val set, one resting ball; on `v1b` it is 0.521. We keep the metric because it still covers the ball and rewards precise boxes. `goal` is excluded because it is near-perfect.)* Always report ball AP@50:95, AP@50 and the all-class mAP too.
- **Run names carry the model and the dataset version** (`E1_yolo11s_640_v1`). The notebook skips training when `best.pt` already exists, so a new model or dataset version needs a new name.

### Reference fine-tune (E1, pure defaults)

In [`notebooks/02_baseline_finetune.ipynb`](../notebooks/02_baseline_finetune.ipynb) §3. Defaults: `imgsz=640`, `epochs=100`, `patience=100`, `batch=16`, `freeze=None`, default augmentation, `optimizer=auto`. Device: `device=mps` on Mac, `device=0` on Colab/Kaggle GPU. Record the device and Ultralytics version in the log.

**Hardware limit:** the Mac (M4, 16 GB) handles 640 (100 epochs ≈ 12 min with YOLO26, 24 min with YOLO11). At `imgsz=1280` it ran out of memory and swapped (~12 min per batch, i.e. days per run), so E2 and every later 1280 run go on Colab.

### Sweep

| ID | Factor | Values | Hypothesis | Look at |
|---|---|---|---|---|
| E0 | Baseline, no training | `yolo26s.pt` @640, @1280 | Finds people, misses the ball | per-class AP |
| E1 | Defaults fine-tune | as above | Large ball gain over E0 | Δ vs E0 |
| E2 | Image size | 640 → 960 → 1280 (Colab) | The 3024 px frame shrinks 4.7× at 640, so the ball's median short side becomes ~7 px (EDA §2.4), below the stride-8 grid (§8.1). *Updated after E1 on `v1b`:* expected gain in ball AP50 and AP50-95 and in bike AP50-95, but the missed balls are not smaller than the found ones (§9), so blurred and hidden balls may stay missed. Evidence: the untrained E0 improves from 640 to 1280 on ball (AP50 0.147 → 0.305) and bike (AP50-95 0.510 → 0.624). | ball and bike AP50-95, train time, ms/frame |
| E3 | Frozen layers | `freeze=0` / `11` (backbone) / `23` (all but head) | ~50 images → freezing reduces overfitting. Counter-hypothesis: COCO features are weak for a blurred tiny ball, so full fine-tuning wins. | train vs val loss gap, ball AP |
| E4 | Augmentations | see §6 | Each change has its own hypothesis in §6 | ball AP, FN types from C |
| E5 | Epochs | 50 / 100 / 200 with `patience=50` | Small data converges fast; long runs overfit | epoch of best fitness, val loss curve |
| E6 | Model size | n / s / m | Bigger → higher AP, slower app | AP vs ms/frame (D's constraint) |
| F | Final | champion × 3 seeds | — | mean ± std on val, then **test once** |

Order E2 → E6 because image size has the largest expected effect on the ball and interacts with mosaic (E4).

**Freeze indices (YOLO11 and YOLO26, same layout):** layers 0–10 are the backbone (ends with SPPF at 9, C2PSA at 10), 11–22 the neck, 23 the Detect head. `freeze=N` freezes `model.0` … `model.N-1`. Check `ultralytics/cfg/models/11/yolo11.yaml` or `26/yolo26.yaml` if using another version.

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
| `mosaic` (1.0), `close_mosaic` (10) | test `mosaic` 1.0 vs 0.5, `close_mosaic` 10 vs 30 | Mosaic multiplies context variety (good for small data). It does **not** shrink the tiles (checked in the 8.4.192 source, 11 Nov: each frame is resized to `imgsz` as usual, four go into a 2×`imgsz` canvas, and a centre `imgsz` window is cut out; zoom comes only from `scale`). With our 2.8 : 1 frames a non-mosaic input is 64% grey padding, a mosaic input about 70% picture, so mosaic shows more balls per step. Run in `notebooks/07` §4. |
| `mixup` (0.0) | keep 0 | Blended ghost players give confusing targets. |
| `translate` (0.1) | keep | Harmless. |
| Motion blur | only if C's FN gallery shows blurred-ball misses dominate | The built-in Albumentations blur runs at p=0.01 — negligible. A stronger motion blur needs a custom Albumentations transform via the Python API. |

---

## 7. Reading the curves (loss dynamics)

`results.csv` columns: `train/{box,cls,dfl}_loss`, `val/{box,cls,dfl}_loss`, `metrics/precision(B)`, `metrics/recall(B)`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `lr/pg0..2`. YOLO26 logs `l1_loss` instead of `dfl_loss` (§8.2).

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
- **YOLO26** (same backbone and neck, `cfg/models/26/yolo26.yaml`): `reg_max=1`, so each side is one regressed distance instead of a 16-bin distribution (no DFL). `end2end=True`: the head has a second, **one-to-one** branch trained to give one box per object, which keeps the top `max_det=300` scores and **skips NMS**. In Ultralytics 8.4.192 that branch runs only with `nms=False` (val, predict, export). The default `nms=None` uses the one-to-many branch plus NMS, like YOLO11, and so does the validation that picks `best.pt` during training.
- **Why the ball is hard:** at stride 8 one P3 cell covers 8×8 px. A 4 px ball is half a cell. After three stride-2 downsamplings its features are mostly mixed with the floor and the wheel it's touching.

### 8.2 Loss

| Term | What | Default weight |
|---|---|---|
| `box_loss` | **CIoU** — overlap + centre distance + aspect ratio | 7.5 |
| `dfl_loss` | **Distribution Focal Loss** — sharpens the per-side distance distributions → better localization | 1.5 |
| `cls_loss` | **BCE** per class (sigmoid, not softmax) | 0.5 |

**YOLO26:** `l1_loss` replaces `dfl_loss`: an L1 loss on the four side distances (scaled by image size), same 1.5 weight. Both head branches are trained: loss = `o2m` × one-to-many loss (TAL top-10) + `o2o` × one-to-one loss (TAL top-7, then 1 per box). `o2m` decays linearly from 0.8 to 0.1 over the epochs, so training shifts to the NMS-free branch. The logged losses are the one-to-one branch's (`E2ELoss` in `utils/loss.py`).

**Assignment — Task-Aligned Assigner (TAL):** for each ground-truth box, candidate anchor points must lie **inside the box**; among them the top-k (k=10) by `score^0.5 · IoU^6` become positives.
→ Consequence for the ball: anchor points sit every 8 px on P3, so a 4×4 px ball would contain one only about (4/8)² ≈ 25% of the time and often get **no positive sample at all**. Ultralytics 8.4.192 guards against this for YOLO11 and YOLO26 alike: for the candidate check only, any box side under 16 px (the P4 stride) is widened to 16 px around its centre (`utils/tal.py`). So the tiny ball does get trained on. What stays hard at 640 is the features (§8.1) and box precision (AP50-95, §8.4): on `v1b`, E1 reaches ball AP50 0.521 and AP50-95 0.226. E2 at `imgsz=960` (ball ~11 px) raises them to 0.63–0.75 and 0.39–0.42 over 3 seeds (§9).

### 8.3 Training mechanics

- **Transfer learning:** backbone and neck start from COCO; the Detect head's class output is re-initialized for our `nc`.
- **`optimizer=auto`:** for short runs (<10k iterations) Ultralytics picks **AdamW** with an auto LR `≈ 0.002·5/(4+nc)` (≈0.0013 for 4 classes) and **ignores `lr0`**. To experiment with LR, set `optimizer=SGD` or `AdamW` explicitly. Same rule for YOLO26; above 10k iterations it picks **MuSGD** (SGD + Muon), which our runs never reach.
- **Gradient accumulation:** nominal batch `nbs=64`. With `batch=16`, gradients accumulate over 4 batches. 48 train images = 3 batches/epoch → **about 3 optimizer steps every 4 epochs**. 100 epochs = 300 batches ≈ 75 steps — reason to test longer runs (E5).
- **Warmup:** `warmup_epochs=3` × batches per epoch = **9 iterations** with 3 batches/epoch (`_get_warmup_iterations` in Ultralytics 8.4.192, no 100-iteration minimum). Accumulation ramps from 1 to 4 batches during warmup.
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
- Why can the E0 ball have precision 1.00 and recall 0.05 but AP50 0.147 (`v1b`)? → P/R are measured at one confidence threshold, while AP uses only the ranking. The pretrained model finds only 8 of 25 balls, mostly with very low confidence (notebook §2).
- Why was the pretrained `person` class so bad? → it boxes the officials' table, spectators and the referee (precision 0.23 on `v1b`).
- Why 960 and not 1280? → 1280 (0.518, one seed) sits at the bottom of the 960 seed range (0.518–0.544), so it is no better, and it costs 1.3× the training time and 1.8× the pixels. Its ball AP50 (0.578) is even below 640.
- Why not `yolo26m`, when it won on the Mac? → at batch 16 on the T4 it scored 0.438, below the 640 control (0.462). One run each, so the Mac's 0.503 was batch 8 or luck. `yolo26s` at 960 gains more and runs faster.

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
| E1 | 29 Oct | v1 | yolo11s | 640 | 100 (65) | — | default | 0 | 0.936 | 0.687 | 0.995 | 0.697 | 0.935 | 0.818 | — | Defaults beat E0; ball stays weakest at ~6 px | Beats E0 on every class (3-class mean AP50 0.525 → 0.916). Ball *not* the weakest: refuted (old val only, reversed on `v1b`). Bike weakest (R 0.67). Next: E2 @1280 on Colab. |
| E0a-26v1 | 30 Oct | v1 | yolo26s (COCO) | 640 | — | — | — | — | 0.544 | 0.326 | 0.835 | 0.549 | 0.206 (person) | 0.592 | switch to YOLO26 | `player`, `bike` fine, ball poor | Old val, not comparable with `v1b`. |
| E0b-26v1 | 30 Oct | v1 | yolo26s (COCO) | 1280 | — | — | — | — | 0.600 | 0.415 | 0.802 | 0.584 | 0.228 (person) | 0.769 | switch to YOLO26 | Upscaling alone helps ball a bit | Ball AP50 did not improve, bike did. Old val, not comparable with `v1b`. |
| E1-26v1 | 30 Oct | v1 | yolo26s | 640 | 100 (70) | — | default | 0 | 0.968 | 0.691 | 0.986 | 0.674 | 0.975 | 0.916 | switch to YOLO26 | About the same as YOLO11 E1; bike stays weakest | Tie with YOLO11 on the selection metric (0.594 vs 0.592). Bike weakest. Old val, superseded by E1-26v1b. |
| E1-26v1 no NMS | 30 Oct | v1 | yolo26s, `nms=False` | 640 | same weights | — | default | 0 | 0.945 | 0.652 | 0.942 | 0.537 | 0.952 | 0.890 | YOLO26 one-to-one head | Can the app drop NMS? | Lower on ball, player and bike. Old val. |
| E0a-26v1b | 8 Nov | v1b | yolo26s (COCO) | 640 | — | — | — | — | 0.444 | 0.266 | 0.147 | 0.098 | 0.333 (person) | 0.853 | EDA split | `player`, `bike` fine, ball poor | Partly right: ball worst (R 0.05), bike fine, but player poor too (P 0.23). |
| E0b-26v1b | 8 Nov | v1b | yolo26s (COCO) | 1280 | — | — | — | — | 0.556 | 0.368 | 0.305 | 0.214 | 0.393 (person) | 0.970 | EDA split | Upscaling alone helps ball a bit | Confirmed: ball AP50 +0.16, bike AP50-95 +0.11. Motivates E2. |
| E1-26v1b | 8 Nov | v1b | yolo26s | 640 | 100 (84) | — | default | 0 | 0.840 | 0.590 | 0.521 | 0.226 | 0.934 | 0.910 | EDA split | Large gain over E0; ball drops once val has a moving ball and has the lowest AP (EDA §3) | Selection metric **0.455** (E0a-26v1b: 0.266). Ball drop confirmed (0.986 → 0.521), ball is the weakest class, bike no longer is. Next: moving-ball frames (v2), E2. |
| E1-26v1b no NMS | 8 Nov | v1b | yolo26s, `nms=False` | 640 | same weights | — | default | 0 | 0.822 | 0.572 | 0.509 | 0.219 | 0.930 | 0.856 | YOLO26 one-to-one head | Can the app drop NMS? | No: lower on ball, player and bike (selection metric 0.431 vs 0.455), saves under 1 ms of post-processing. Keep NMS. |
| E1-26v1b s1 | 11 Nov | v1b | yolo26s | 640 | 100 (84) | — | default | 1 | 0.863 | 0.608 | 0.573 | 0.285 | 0.937 | 0.948 | 1 val ball ≈ 4 AP points | Seeds differ by 0.01–0.02 in the selection metric | Selection 0.479. |
| E1-26v1b s2 | 11 Nov | v1b | yolo26s | 640 | 100 (92) | — | default | 2 | 0.859 | 0.602 | 0.568 | 0.275 | 0.934 | 0.940 | same | same | Selection 0.471. With seed 0: **0.468 ± 0.012, range 0.024**, the noise yardstick for every row below. Seed 0 (0.455) was the unluckiest. Roughly confirmed. |
| E1-26v1b NMS 0.5 | 11 Nov | v1b | yolo26s, `iou=0.5` | 640 | E1 seed 0 weights | — | default | 0 | 0.839 | 0.586 | 0.547 | 0.237 | 0.897 | 0.919 | C #6 (duplicate riders) | Player precision up; risk: real riders overlapping in the scrum get removed | Refuted: player P 0.946 → 0.936 and AP50 0.934 → 0.897, the risk happened. Selection 0.450 vs 0.455. Keep `iou=0.7`; duplicates need data (C #1). |
| E3-f11 | 11 Nov | v1b | yolo26s | 640 | 100 (64) | 11 | default | 0 | 0.833 | 0.604 | 0.456 | 0.260 | 0.940 | 0.940 | plan §5 | No overfitting in E1 → tie or small loss, on the ball; faster | Confirmed: selection 0.474 (tie), ball AP50 below every E1 seed, 37% faster (7.3 min). |
| E3-f23 | 11 Nov | v1b | yolo26s | 640 | 100 (84) | 23 | default | 0 | 0.781 | 0.565 | 0.297 | 0.191 | 0.929 | 0.904 | plan §5 | Clear loss, most on the ball | Confirmed: selection 0.423 (−0.045, beyond noise), ball AP50 0.297. |
| E4-scale0.3 | 11 Nov | v1b | yolo26s | 640 | 100 (57) | — | `scale=0.3` | 0 | 0.857 | 0.599 | 0.536 | 0.228 | 0.958 | 0.938 | plan §6 | Less zoom-out keeps the ball ≥ 5.6 px → ball up, AP50-95 more | Refuted: ball AP50-95 at the low end of the seed range. Selection 0.468 (tie). |
| E4-mosaic0.5 | 11 Nov | v1b | yolo26s | 640 | 100 (84) | — | `mosaic=0.5` | 0 | 0.849 | 0.590 | 0.515 | 0.241 | 0.950 | 0.937 | plan §6 + source check | Fewer balls per step → ball a little lower | Not confirmed: ball AP50 just under the seed range, selection 0.457, within noise. |
| E5-ep50 | 11 Nov | v1b | yolo26s | 640 | 50 (43) | — | default | 0 | 0.844 | 0.586 | 0.435 | 0.160 | 0.974 | 0.972 | plan §5 | ~37 optimizer steps → a little lower, mostly the ball | Confirmed: selection 0.450 (−0.019, borderline). Ball worst of the `s` runs; player and bike AP50 best of all runs. |
| E5-ep200 | 11 Nov | v1b | yolo26s | 640 | 161 (111), `patience=50` | — | default | 0 | 0.866 | 0.606 | 0.579 | 0.297 | 0.937 | 0.955 | plan §5 | Val loss still falling at 100 → small gain within noise, best epoch > 100 | Confirmed: selection 0.476 (+0.008, tie), early stop at 161, 19.6 min. |
| E6-n | 11 Nov | v1b | yolo26n | 640 | 100 (68) | — | default | 0 | 0.772 | 0.559 | 0.235 | 0.122 | 0.951 | 0.908 | plan §5 | Less capacity → lower on ball and bike | Confirmed: selection 0.414 (−0.055, beyond noise), ball AP50 0.235. |
| E6-m | 11 Nov | v1b | yolo26m, `batch=8` | 640 | 100 (90) | — | default | 0 | 0.890 | 0.626 | 0.651 | 0.323 | 0.948 | 0.964 | plan §5 | At most slightly better: 48 images can't use the capacity | **Refuted:** selection **0.503 (+0.034, beyond noise)**, best ball of all runs. 29.8 min, 21 ms/img (3× `s`). Batch 8 because 16 swapped on the Mac (same optimizer steps via `nbs`). Batch 16 on Colab did not reproduce it (E6-m b16 below). |
| E2-640 | 13 Nov | v1b | yolo26s | 640 | 100 (97) | — | default | 0 | 0.886 | 0.595 | 0.617 | 0.241 | 0.948† | 0.968† | GPU control | Lands inside the Mac E1 seed range | Confirmed: selection 0.462 (Mac seeds 0.455–0.479). The T4 adds no shift of its own. Reference for the T4 rows below. 4.7 min. |
| E2-960 | 13 Nov | v1b | yolo26s | 960 | 100 (88) | — | default | 0 | 0.915 | 0.657 | 0.750 | 0.422 | 0.959† | 0.948† | plan §5, C #2 | Ball ~11 px: ball AP50-95 gains most, ball AP50 less, bike AP50-95 up, player same; 2–4× cost | **Confirmed:** selection **0.544 (+0.082 vs 640, beyond noise)**. Ball AP50-95 0.241 → 0.422, ball AP50 +0.13, player AP50-95 +0.03, bike +0.04. 7.3 min (1.5×). → 3 seeds. |
| E2-1280 | 13 Nov | v1b | yolo26s | 1280 | 100 (90) | — | default | 0 | 0.862 | 0.638 | 0.578 | 0.410 | 0.940† | 0.911† | plan §5 | Ball ~15 px: the largest gain | Refuted: selection 0.518, at the bottom of the 960 seed range. Ball AP50-95 like 960, ball AP50 below 640, bike AP50-95 down. 9.7 min (2.0×). Not worth it. |
| E2-960 s1 | 13 Nov | v1b | yolo26s | 960 | 100 (92) | — | default | 1 | 0.903 | 0.637 | 0.718 | 0.421 | — | — | champion × 3 seeds | 960 stays beyond noise over seeds | Selection 0.518. |
| E2-960 s2 | 13 Nov | v1b | yolo26s | 960 | 100 (93) | — | default | 2 | 0.884 | 0.640 | 0.630 | 0.385 | — | — | same | same | Selection 0.522. With seed 0: **0.528 ± 0.014** (0.518–0.544) vs E1 @640 0.468 ± 0.012: **+0.060**, worst 960 seed above best 640 seed. Confirmed. **Champion: `E2_yolo26s_960_v1b`.** |
| E6-m b16 | 13 Nov | v1b | yolo26m | 640 | 100 (91) | — | default | 0 | 0.802 | 0.577 | 0.342 | 0.186 | 0.940† | 0.926† | E6-m (Mac, batch 8) | Gain holds at batch 16, mostly on the ball | **Refuted:** selection 0.438 (−0.024 vs the 640 control), ball AP50 0.342. One run each, so batch 8 vs seed luck can't be told apart. Dropped: 960 `s` gains more, at lower cost. 9.2 min. |

*mAP "all" = the classes present in that run (E0: ball, player, bike; E1 adds goal). Mac M4 `mps`, Ultralytics 8.4.192, torch 2.15. E1 trained in 23.7 min (YOLO11), 13.8 min (YOLO26, `v1`) and 11.7 min (YOLO26, `v1b`). E1 @1280 on the Mac was aborted after 3 batches (out of memory, see §5). Rows from 11 Nov: `notebooks/07_experiments_e3_e6.ipynb`, one seed each except E1; "selection" = mean AP50-95 over ball, player and bike; "beyond noise" = more than the E1 seed range (0.024) from the seed mean 0.468. Rows from 13 Nov: `notebooks/06_e2_image_size_colab.ipynb`, Colab Tesla T4, Ultralytics 8.4.192, torch 2.14.1+cu130; Δ is against E2-640, the control on the same GPU. † Player and bike AP50 are not in notebook 06's table; they come from re-validating the same `best.pt` on the Mac (13 Nov). On the Mac the other columns differ from the T4 values by up to 0.05 in ball AP50 (about one ball) and 0.006 in the selection metric. The 960 seed runs are only on Drive, so they have no Mac values (—).*

**Data versions in this log.** `v1` = the first local build (60 train / 13 val, val frames 960–1167). The EDA found the ball at rest in the same spot in all 13 val frames, and frame 749 had a wrong `player` box, so `v1` rows are kept for history only. `v1b` = the dataset the main README calls v1 (48 train / 25 val, val frames 13–226, frame 749 fixed). Compare rows only within one version. YOLO11 was not re-run on `v1b`.

**Where the ball fails (E1-26v1b, one-off check, not a notebook cell).** `best.pt` at `imgsz=640`; hit = confidence ≥ 0.25 and IoU ≥ 0.5. 14 of 25 balls hit, 10 missed, 1 just under IoU 0.5. 2 false ball boxes, both in frame 47, one of them that near-hit. Frames 13–102 hit 2 of 12, frames 118–226 hit 12 of 13. Missed balls are not smaller (median short side at 640 about 8.0 px, hits 7.5 px). Of 6 missed frames viewed, 4 show a blurred ball or one overlapping a wheel or foot (13, 22, 60, 80), and 2 show no clear ball inside the box (53, 74). Frame 74's box sits on a shoe: for A to re-check.

---

## 10. Checklist

- [x] Class set: A chose `player` / `ball` / `bike` / `goal` (§3).
- [x] Environment: `.venv`, Ultralytics 8.4.192, torch 2.15, `mps`.
- [x] E0 baseline @640 and @1280 → log.
- [x] E1 defaults → log.
- [ ] Hand the run folders (`runs/modeling/E0_yolo26s_*_v1b`, `E1_yolo26s_640_v1b`) to C. Feedback for A: re-check the ball box in `frame_00074` (it sits on a shoe).
- [x] D: `models/best.pt` and `models/detections.csv` from E1-26v1b (notebook §7).
- [ ] D: annotated video (format agreed in §2).
- [ ] Dataset v2: label more frames with a moving, blurred or partly hidden ball (after frame 234, so val stays clean) → re-run E0 and E1 with `VERSION = "v2"`.
- [x] Switch to YOLO26: `MODEL = "yolo26s.pt"` in the notebook, E0 and E1 re-run on the old split and on the EDA split (`v1b`), logged as new rows next to the YOLO11 ones. YOLO11 was not re-run on `v1b`.
- [ ] Ask A for a held-out test clip; v1 has none.
- [x] E3–E6 at 640 on the Mac, each row with a hypothesis (11 Nov, notebook 07); C's #6 tested (refuted).
- [x] E1 noise: 3 seeds, 0.468 ± 0.012.
- [x] E2 (640 control / 960 / 1280) and `yolo26m` at batch 16 on a Colab T4 (13 Nov, notebook 06): 960 wins, `yolo26m` gain not reproduced; C's #2 confirmed.
- [x] Champion × 3 seeds on val: `yolo26s` @960, 0.528 ± 0.014.
- [ ] Download the 960 seed runs from Drive into `runs/modeling/` (the other 13 Nov run folders are already on the Mac).
- [ ] Tell C: #6 refuted (stricter NMS removes real riders), #2 confirmed (960 lifts ball AP50-95 0.24 → 0.42); re-run notebook 03 on the champion (`E2_yolo26s_960_v1b`, `imgsz=960`).
- [ ] Evaluate the held-out test set once (needs A's test clip).
- [ ] Hand-off to D: champion `best.pt` → `models/best.pt`, `imgsz=960`, regenerate `detections.csv`, per-class `conf`, ms/frame measured on the Mac.
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
| 2026-10-29 | Claude (Claude Code) | Dataset v1 check; notebook `notebooks/02_baseline_finetune.ipynb` (code, explanations, conclusions); running E0/E1; checking the video's fps and frame numbering; the CSV format for D; updating this plan and PROGRESS.md | |
| 2026-10-30 | Claude (Claude Code) | Moving the notebook to `notebooks/`; preparing the switch to YOLO26 (model in run names, YOLO26 notes in §5, §7, §8, checked against Ultralytics 8.4.192 source) | |
| 2026-11-08 | Claude (Claude Code) | Checking the `v1b` runs (dataset split, training args, frame alignment, per-ball hits and misses); updating the experiment log, these notes and the notebook text; resolving the `.gitignore` merge conflict with `main`; the `detections.csv` cell (notebook §7) and `save_json`; new numbers on the app's How it works page | |
| 2026-11-11 | Claude (Claude Code) | Notebooks `06_e2_image_size_colab.ipynb` (E2) and `07_experiments_e3_e6.ipynb` (E1 seeds, NMS IoU, E3–E6): code and hypothesis texts; source check of Ultralytics mosaic, scale and checkpoint format (correction in §6); running notebook 07 on the Mac (restarted `yolo26m` with `batch=8` after it swapped); notebook 07 conclusions and the 11 Nov rows in §9 | |
| 2026-11-13 | Claude (Claude Code) | Reading the E2 and `yolo26m` batch-16 results in notebook 06; re-validating the four local `best.pt` on the Mac for player and bike AP50; notebook 06 seed cell (§1c) and conclusions; the 13 Nov rows in §9, status, §8 notes, checklist and PROGRESS.md | |

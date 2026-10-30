# Labelling rules

Agree on these before anyone labels. Four of us label, so the model only learns one ball if we all draw the same box.

## Classes

Every frame gets all four classes. The ids are the ones in the Ultralytics Platform export.

| id | Class | What to box |
|---|---|---|
| 0 | `player` | The rider only: from the head down to the feet on the pedals. The bike is a separate box. |
| 1 | `ball` | The ball in play. |
| 2 | `bike` | The whole bicycle, both wheels, down to where the tyres touch the floor. |
| 3 | `goal` | Each goal: frame and net. Both goals in every frame. |

- Rider and bike are separate boxes, the same way COCO has `person` and `bicycle` as separate classes. The minimap uses the bottom of the bike box, because that is where the wheels touch the floor.
- Not labelled: referees, officials, substitutes and spectators. They are not riders, so they are not `player`.

Why we label the riders ourselves, although the brief suggests the pretrained detector for players:
- Pretrained YOLO often detected people in the crowd as `person`. We box every rider and never a spectator or the referee, so the fine-tuned model learns that a `player` is a rider on a bike on the court, and that everyone else is background.
- Fine-tuning replaces the model's output layer with our four classes. A rider we didn't box would count as background, so the model would learn to ignore riders. That's why every rider in every labelled frame must have a box.

## Box

- Tight box around the object, edge to edge.
- Only the ball in play; spare balls outside the court are not labelled.
- Riders close together: every rider and every bike still gets its own box, even when the boxes overlap.

## Occlusion

The same rule for every class:

- At least half of the object visible: box the whole object, up to where you expect its edges.
- Less than half visible: don't label it.

## Motion blur

- A blurred ball is still labelled: a box the size of the ball, centred on the middle of the streak.

## Ball in the air

- A ball in the air (a header, a shot) is labelled where it is in the image, like a ball on the floor.

## Scoreboard overlay

- A ball under the overlay is not labelled.

## Which frames

Pick hard ones: ball near the goal, in the air, behind a wheel or a rider, on the far side, fast shots, crowded moments. Skip the ball lying still and near-identical consecutive frames.

What we did for v1: from all 1,170 frames we picked 73 with confusing shots, mainly riders and bikes overlapping and a blurred ball, so the model learns the hard cases.

- About half the frames of our video are a near-exact copy of the frame before. Before you label a frame, check that it really differs from the labelled frame next to it.
- Still missing in our set: the ball on the right half of the court, and frames where the ball is completely hidden. Label everything else in those frames and draw no ball box: the model has to learn that a frame can have no visible ball.

## Split

By match segment, never random frames: consecutive frames are near-duplicates, so a random split puts the same moment in both sets.

- Validation: frames 1–234 (the first 20% of the video).
- Train: frames 235–1170.
- Test: none yet. The modeling plan asks for 10–15 held-out frames from a segment that is in neither set.

The split is set in `configs/dataset.yaml` and applied by `data/prepare_dataset.py`; the split stored on the Platform is not used. Validation is the start of the video, not the end: from frame ~960 the referee puts the ball down and it lies still until the end of the clip, so a validation set from the end would only test a ball that doesn't move.

## How we labelled as a team

- **Tool:** Ultralytics Platform. Assisted labelling: _to fill in: did we use the Platform's smart annotation (SAM) for riders and bikes? Was the ball drawn by hand?_
- **Who labelled what:** _to fill in: how the frames were divided between us._
- **Before labelling:** we agreed on the rules in this file.
- **Review:** _to fill in: did someone check another person's frames, and how?_
- **Checks on the export:** `notebooks/01_dataset_eda.ipynb` counts the boxes in every frame (we expect 4 players, 4 bikes, 2 goals, 1 ball) and flags the frames that don't fit. On the first export it found:
  - frame 749: a person outside the court boxed as a `player` (fixed in v1);
  - 7 val frames (22, 32, 34, 39, 53, 80, 102) with fewer bikes than riders. Checked by eye: in each, the missing bike is more than half hidden behind another rider, so by the occlusion rule above it gets no box. Nothing to change.

> **GenAI:** rules for the four classes, occlusion, ball in the air, missing frames, the split and the team-process section drafted with Claude Code (Opus 5.5) on 2026-10-29/30, from our labelled data, the EDA notebook and our explanations. What we checked or changed ourselves: _to fill in_.

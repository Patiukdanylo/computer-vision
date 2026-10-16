# Labelling rules

**Status: draft.** Agree on these as a team before anyone labels, then only change them through the changelog at the bottom. Four people label, so the model only learns one ball if we all draw the same box.

## Classes

- `ball`: the cycle ball.
- Players and referees are not labelled: pretrained YOLO already finds them as `person`. Add a `player` class only if the baseline run in notebook 01 shows it failing on riders.

## Box

- Tight box around the visible ball, edge to edge, no margin.
- One ball in play per frame. Spare balls lying outside the court are not labelled.

## Occlusion

- At least half of the ball visible: box the full ball, as if it were not hidden.
- Less than half visible: don't label it, and tag the frame `occluded` in Roboflow.

## Motion blur

- A blurred ball is still labelled: a box the size of the ball, centred on the middle of the streak.

## Scoreboard overlay

- A ball under the overlay is not labelled; tag the frame `overlay`.

## Which frames ("challenging")

Pick: ball near or in the goal, ball in the air, ball behind a wheel or a rider, far side of the court, fast shots, crowded moments at the goal.
Avoid: the ball lying still in the centre circle, near-identical consecutive frames.

## Split

- Train: frames from segments … (fill in: stream, start–end time)
- Validation: frames from segment … (a different match or half, different kits if possible)

## Cross-review

Each person checks 10 frames labelled by someone else. Record how many boxes you'd have drawn differently.

| Reviewer | Frames checked | Disagreements | What was different |
|---|---|---|---|
| | | | |

## Changelog

| Date | Rule changed | Why (which frames showed the problem) |
|---|---|---|
| | | |

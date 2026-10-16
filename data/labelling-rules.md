# Labelling rules

Agree on these before anyone labels. Four of us label, so the model only learns one ball if we all draw the same box.

## Classes

- `ball`: the cycle ball.
- Players and referees are not labelled: pretrained YOLO already finds them as `person`. We add a `player` class only if the baseline run shows it failing on riders.

## Box

- Tight box around the ball, edge to edge.
- Only the ball in play; spare balls outside the court are not labelled.

## Occlusion

- At least half of the ball visible: box the whole ball.
- Less than half visible: don't label it.

## Motion blur

- A blurred ball is still labelled: a box the size of the ball, centred on the middle of the streak.

## Scoreboard overlay

- A ball under the overlay is not labelled.

## Which frames

Pick hard ones: ball near the goal, in the air, behind a wheel or a rider, on the far side, fast shots, crowded moments. Skip the ball lying still and near-identical consecutive frames.

## Split

By match segment, never random frames: consecutive frames are near-duplicates, so a random split puts the same moment in both sets.

- Train: …
- Validation: …

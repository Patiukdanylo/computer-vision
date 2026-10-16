# computer-vision
## Sport: Cycle ball

Participants:
- Ivayla Pernikova r1028557
- Danylo-Makarii Patiuk r1028011
- Ryan Omondi r1019767
- Artem Kucheruk r1026280

Deep Learning (Computer Vision) challenge, Thomas More 2026: from a static-camera cycle ball video to a synchronized 2D tactical minimap. Pretrained YOLO finds the players; we fine-tune it on 40–80 labelled frames so it also finds the ball, project the positions onto a top-down court with a homography, and show both side by side in a Streamlit app.

## Structure

```
notebooks/   numbered notebooks, run in order (see notebooks/README.md)
app/         Streamlit dashboard and the inference code it needs
models/      the trained weights the app uses (best.pt)
data/        local only: clips, frames, labelled export (see data/README.md and data/labelling-rules.md)
docs/        report and slides
```

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
zip -r cv-challenge-upload.zip docs/report.pdf docs/slides.pptx notebooks app models requirements.txt README.md \
  -x "*.ipynb_checkpoints*" "*/__pycache__/*"
```

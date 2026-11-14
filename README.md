# DIGIT Tactile Collector

Record, review and classify touches from a [Meta DIGIT](https://digit.ml) tactile sensor on macOS.

**Goal:** collect labelled touch episodes of everyday objects, then test whether a simple model can tell objects apart by feel. It does well on hard vs soft: about 97% of frames.

## Quick start
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python collect.py 0        # 0 = DIGIT camera index
# open http://localhost:5050
```

## Collecting data (web UI)
- **Classes:** add one with **N**; take a quick webcam photo for its card. Select a class with **1–9**.
- **Record:** in manual mode press **space** to start and stop. In auto mode (**A**), recording starts on steady pressure and stops after *padding* seconds with no contact. *Padding* is also how much buffered time is kept before the press.
- **Tuning:** min press, padding and sensitivity can be set in the panel.
- **Review and delete:** **E** opens a class's episodes to play or delete them. **⌫** deletes the last episode.
- **Datasets:** archive the current set and start fresh, or restore an old one.
- **View and sensor:** **V** toggles raw / baseline-diff view. **↻ reconnect** (or **R**) reconnects the sensor.

Episodes are saved as `episodes/<id>.npz` (frames + timestamps) and `<id>.json` (label). The dataset is on Hugging Face: [Team-Mulberry/digit-tactile-nov13](https://huggingface.co/datasets/Team-Mulberry/digit-tactile-nov13).

## Training
```bash
.venv/bin/python train.py hard soft          # any labels present in episodes/
CLASSIFY=1 .venv/bin/python collect.py 0     # live prediction using model.joblib
```
Features per contact frame:
- **ResNet-18 latent:** a frozen network's encoding of the baseline-diff image, with flip/rotate/shift augmentation.
- **Intensity stats:** how strongly and widely the gel is pressed.
- **Force-field stats:** a 12×16 grid of a normal-pressure proxy plus optical-flow shear.

The classifier is logistic regression (one-vs-one), evaluated with leave-episodes-out cross-validation.

## Files
| file | what |
|---|---|
| `collect.py` | Flask server: capture, auto-record, classes, datasets, playback |
| `templates/index.html` | collector UI |
| `tactile_model.py` | diff, contact score, ResNet features, force field |
| `train.py` | train and compare classifiers, save `model.joblib` |
| `digit_view.py` | minimal OpenCV live viewer |

## What we built today
- Connected the DIGIT through OpenCV and fixed the image's vertical roll/wrap.
- Built a minimal collection UI with keyboard control, auto-record and class photo cards.
- Trained frozen ResNet-18 + logistic regression: about 80% on 3 foods, about 97% on hard vs soft.
- Derived a force-vector field from the video; its physical stats alone reach about 91% on hard vs soft.
- Collected about 1,200 episodes across 12 objects.

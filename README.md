# Football Tracker — AGH AiPO

Player detection, tracking, and tactical analytics for football match footage.

`YOLO26` detector → `ByteTrack` tracker → `YOLO26-pose` pitch keypoints →
per-frame homography → top-down minimap + per-player distance / speed.

---

## Four scripts, one repo

| Script        | Subcommands                                                  | Writes |
|---------------|--------------------------------------------------------------|--------|
| `dataset.py`  | `soccernet · roboflow · pitch · soccernet-pitch · preprocess-pitch · youtube · merge · all` | datasets under `data/` |
| `train.py`    | `detector · pitch · all · export`                            | weights + `outputs/reports/train/*.json` |
| `eval.py`     | `detector · pitch · homography · all`                        | `outputs/reports/eval/*.json` |
| `demo.py`     | (flags only)                                                 | annotated MP4 + (with `--dump`) `outputs/reports/demo/<slug>/…` |

Every script prints `--help` with examples and accepts `--device` (auto / cpu / mps / cuda / cuda:N / 0,1,2).

---

## Models

| Component       | What                                  | Trained? | Artifact                       |
|-----------------|---------------------------------------|----------|--------------------------------|
| Player detector | YOLO26-s, 4 classes                   | yes      | `models/checkpoints/best.pt`   |
| Pitch model     | YOLO26s-pose, 32 keypoints            | yes      | `models/checkpoints/pitch.pt`  |
| Tracker         | ByteTrack (Kalman + Hungarian)        | no       | —                              |
| Homography      | RANSAC fit + EMA smooth over visible keypoints | no | —                          |

Player classes (`configs/classes.yaml`): `0 player · 1 goalkeeper · 2 referee · 3 ball`.
Pitch landmarks (`configs/pitch_keypoints.yaml`): 32 points in pitch metres.

Both `.pt` files are the only artifacts that cross machines.

---

## Train (RTX box)

### Quick path — Roboflow only (no SoccerNet, no NDA)

```bash
git clone <repo> && cd AiPO-proj
make install-cuda
export ROBOFLOW_API_KEY=...

python dataset.py all              # Roboflow players + Roboflow pitch + merge
python train.py all                # → best.pt + pitch.pt + 2× training JSON
python eval.py all                 # → eval JSON for both models
```

### Full path — SoccerNet broadcast + Roboflow + green-suppression

The pitch model lives or dies on having broadcast frames in the training mix.
SoccerNet calibration-2023 ships ~13 k of them; combined with Roboflow tactical
(317 frames) and after green-suppression, the keypoint model sees
**~13.3 k train / ~2.5 k val** images, every one with at least 4 visible
landmarks (the geometric minimum for a homography fit).

```bash
export ROBOFLOW_API_KEY=...
export SOCCERNET_PASSWORD=...

python dataset.py roboflow                                  # players, ~600 frames
python dataset.py pitch                                     # Roboflow tactical pitch, 317 frames
python dataset.py soccernet-pitch --zip /path/to/train.zip  # SoccerNet broadcast, ~13 k frames
python dataset.py preprocess-pitch                          # green-suppression + merge → 13.3 k frames
python dataset.py merge                                     # players combined

python train.py all                # detector + pitch (~1.5 h on RTX 3090)
python eval.py all                 # mAP for both
```

Then ship back **two files**: `models/checkpoints/best.pt` and
`models/checkpoints/pitch.pt`. Optional but useful: also the training-report
JSONs from `outputs/reports/train/` — they have per-epoch loss curves + best-epoch
metrics, helpful for the course report.

---

## Run (Mac)

```bash
make install-mac
# drop best.pt + pitch.pt into models/checkpoints/
python demo.py --source clip.mp4               # or --source 0 for webcam
```

Auto-picks MPS on Apple Silicon (~14 fps end-to-end). Override with
`--device cpu` or `--device cuda` on a box with both.

With `pitch.pt` the homography is recomputed every frame and EMA-smoothed across
successive fits — the HUD shows `H: dyn (12vis/8inl)` when a fit lands, or the
specific rejection reason on fallback (`dyn-fallback (3: keypoints cluster (16×54 m))`).

Without `pitch.pt`: first frame prompts you to click the four pitch corners
(TL → TR → BR → BL); the matrix is saved to `configs/homography.json` and reused.

### Useful demo flags

```bash
python demo.py --source clip.mp4 --output annotated.mp4 --no-show     # headless render
python demo.py --source clip.mp4 --dump                                # per-frame JSONL + summary JSON
python demo.py --source clip.mp4 --no-pitch                            # force static click-homography
python demo.py --source clip.mp4 --weights /dev/null --no-pitch        # stock yolo26n COCO fallback (no fine-tuning needed)
```

---

## Reports — what JSON goes where

Every train / eval / dumped demo run writes a structured JSON next to nothing
else in `outputs/reports/`. Useful for **(a)** debugging regressions after the
fact and **(b)** dropping numbers straight into the AiPO course report.

```
outputs/reports/
├── train/
│   ├── detector-yolo26s_v2-<timestamp>.json    # per-epoch metrics, best epoch, elapsed seconds
│   └── pitch-yolo26s_pitch_v4-<timestamp>.json
├── eval/
│   ├── detector-<timestamp>.json               # mAP50-95, mAP50, P, R, per-class
│   ├── pitch-<timestamp>.json                  # box + keypoint mAP
│   └── homography-<timestamp>.json             # fit rate, rejection-reason histogram, per-keypoint visibility
└── demo/
    └── <video_slug>-<timestamp>/
        ├── summary.json                        # run-level: frames, fit rate, per-track distance/speed
        └── frames.jsonl                        # one record per frame: bboxes, IDs, pitch_xy, H status
```

Override the root with `--report-dir <path>` on any script.

`eval.py pitch` defaults `--data` to the combined+green-suppressed val set
(SoccerNet + Roboflow). Pass `--data configs/pitch.yaml` to score against the
317-frame Roboflow-only val instead.

### Worked example — the broadcast clip we tested on

```bash
python eval.py homography --source data/raw/youtube/demo.mp4 --stride 5 --device mps
# → outputs/reports/eval/homography-2026-06-14T15-30-00.json
```

This JSON contains:
- summary stats (frames processed, fit rate, fallback rate, mean kpts visible)
- a `rejection_reasons` histogram (no detection / cluster / RANSAC failed / …)
- a `per_keypoint` array showing which of the 32 landmarks the model actually sees on this footage

That's the data you cite in the report when explaining why the minimap was sparse
on a given clip.

---

## All make targets

```
make install-mac      Mac dev install (CPU)
make install-cuda     RTX install (CUDA 12.4 torch + train extras)

make data             Roboflow (players + pitch) + merge
make data-full        SoccerNet + Roboflow (players + pitch) + merge
make data-soccernet   SoccerNet alone (needs SOCCERNET_PASSWORD)
make data-roboflow    Roboflow players alone
make data-pitch       Roboflow pitch keypoints alone
make data-merge       Re-merge whatever player datasets are in data/raw/

make train            Fine-tune detector → best.pt + train report JSON
make train-pitch      Fine-tune pitch model → pitch.pt + train report JSON
make train-all        Both, sequentially
make train-export     ONNX export of detector

make eval-detector    mAP / PR on detector val (writes JSON)
make eval-pitch       Box + keypoint mAP for pitch model (writes JSON)
make eval-homography  Per-frame H stability on a video (writes JSON)
make eval-all         Detector + pitch eval

make demo SOURCE=...  Full pipeline (SOURCE=0 for webcam)
make demo-pretrained  Stock yolo26n.pt + COCO classes — no fine-tuning needed
make demo-record SOURCE=... OUT=...   Headless render to file

make smoke            Run scripts/smoke_test.py
```

---

## Layout

```
.
├── dataset.py          ← single entry: data download + preprocess + merge
├── train.py            ← single entry: train both models, dumps train JSON
├── eval.py             ← single entry: detector / pitch / homography eval, dumps JSON
├── demo.py             ← single entry: combined live pipeline, optional --dump JSON
├── Makefile            ← all flows as `make <target>`
├── configs/
│   ├── classes.yaml             # player class scheme
│   ├── pitch_keypoints.yaml     # 32-landmark scheme in pitch metres
│   ├── training.yaml            # detector hyperparams (yolo26s, v2)
│   ├── training_pitch.yaml      # pitch hyperparams (yolo26s-pose, v4)
│   ├── combined.yaml            # player dataset YAML (generated by `merge`)
│   ├── pitch.yaml               # Roboflow-only pitch YAML (generated by `pitch`)
│   ├── combined_pitch.yaml      # SoccerNet + Roboflow source spec for preprocess-pitch
│   └── homography.json          # generated by click-calibrate fallback
├── outputs/
│   └── reports/                 # JSON dumps from every run
├── src/football_tracker/
│   ├── datasets/       # SoccerNet, Roboflow (players + pitch), YouTube, merge, preprocess
│   ├── training/       # detector / pitch trainers, eval, ONNX export, report_utils
│   ├── tracking/       # ByteTrack wrapper (returns tracked + untracked)
│   ├── pitch/          # keypoint scheme, dynamic homography (with EMA), minimap, preprocess
│   ├── analytics/      # distance / speed / heatmap accumulators
│   ├── pipeline/live.py   # detect → track → project → analyse → render (+ dump)
│   ├── reporting.py    # dump_json / timestamp_slug / JsonlWriter
│   └── device.py       # pick_device() — CUDA > MPS > CPU
└── scripts/smoke_test.py
```

---

## Goals (AiPO spec)

Required:
- Player + ball + referee detection.
- Persistent ID tracking across frames (trajectories).
- Mapping bounding boxes onto a top-down pitch schematic.

Implemented:
- Distance covered + speed per player.
- Per-frame homography from learned pitch keypoints (RANSAC + EMA smoothing).
- Structured JSON dumps of every run for offline analysis.
- Pre-flight dataset checks and clear errors in train / eval scripts.

Stretch:
- Heatmaps per player / team (module ready; not yet wired to live HUD).
- Pass / ball-contact detection.
- Formation analysis.

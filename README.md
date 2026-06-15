# Football Tracker — AGH AiPO

Player detection, tracking, and tactical analytics for football match footage.

`YOLO26` detector → `ByteTrack` tracker → `YOLO26-pose` pitch keypoints →
per-frame homography → top-down minimap + per-player distance / speed.

---

## Five scripts, one repo

| Script        | Subcommands / flags                                                                                                | Writes |
|---------------|--------------------------------------------------------------------------------------------------------------------|--------|
| `dataset.py`  | `soccernet · roboflow · pitch · soccernet-pitch · preprocess-pitch · augment-pitch · youtube · merge · all`        | datasets under `data/` |
| `train.py`    | `detector · pitch · all · export`                                                                                  | weights + `outputs/reports/train/*.json` |
| `eval.py`     | `detector · pitch · homography · all`                                                                              | `outputs/reports/eval/*.json` |
| `demo.py`     | live preview window (interactive)                                                                                  | annotated MP4 + (with `--dump`) `outputs/reports/demo/<slug>/…` |
| `process.py`  | one-shot: video in → annotated MP4 out, stage logging, summary table, `--minimal` for boxes/trails only            | same dump tree as `demo.py` |

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

### Optional — offline augmentation for the pitch retrain

YOLO's on-the-fly augmentation under-samples both photometric variance **and**
partial-pitch framing. The combined dataset's source frames all show the
whole pitch, so the model learns "centre line ≈ pixel 50 % of image" and then
mislabels centre keypoints as left side-line keypoints when broadcast TV
zooms into one half.

`augment-pitch` writes offline copies that fix both blind spots:

```bash
pip install -e '.[training]'                                  # adds albumentations
python dataset.py augment-pitch --copies 2 --partials         # ~170 k images
# point training_pitch.yaml at data/processed/combined_pitch_gs_aug/data.yaml
python train.py pitch
```

- `--copies N` writes N photometric variants per source (HSV / gamma / blur /
  dropout / mild affine — albumentations, keypoint-aware).
- `--partials` writes **11 strategic crops per source** that teach the model
  partial-pitch views directly: top / bottom / left / right halves, four
  quarters, centre zoom, left-goal close-up, right-goal close-up. Each crop
  is then run through the photometric pipeline too.

Default totals (13 k sources, `--copies 2 --partials`):
~ 1 original + 2 photometric + 11 partials ≈ 14 × per source = **~180 k
train images**. Drop `--partials` for the smaller ~40 k photometric-only
variant.

Then ship back **two files**: `models/checkpoints/best.pt` and
`models/checkpoints/pitch.pt`. Optional but useful: also the training-report
JSONs from `outputs/reports/train/` — they have per-epoch loss curves + best-epoch
metrics, helpful for the course report.

---

## Run (Mac)

```bash
make install-mac
# drop best.pt + pitch.pt into models/checkpoints/

# Live preview (interactive, Esc to quit):
python demo.py --source clip.mp4               # or --source 0 for webcam

# Batch / headless: video in → annotated MP4 + JSON summary:
python process.py clip.mov                     # → clip_processed.mp4
```

Auto-picks MPS on Apple Silicon (~20 fps end-to-end on a 60 s 720p clip).
Override with `--device cpu` or `--device cuda`.

With `pitch.pt` the homography is recomputed every frame and EMA-smoothed across
successive fits — the HUD shows `H: dyn (12vis/8inl)` when a fit lands, or the
specific rejection reason on fallback (`dyn-fallback (3: keypoints cluster (16×54 m))`).

Without `pitch.pt`: first frame prompts you to click the four pitch corners
(TL → TR → BR → BL); the matrix is saved to `configs/homography.json` and reused.

### Useful flags

```bash
# demo.py — interactive
python demo.py --source clip.mp4 --output annotated.mp4 --no-show       # headless render
python demo.py --source clip.mp4 --dump                                 # per-frame JSONL + summary JSON
python demo.py --source clip.mp4 --no-pitch                             # force static click-homography
python demo.py --source clip.mp4 --weights /dev/null --no-pitch         # stock yolo26n COCO fallback

# process.py — batch, with stage timing + summary table
python process.py clip.mov                                              # default: full overlays
python process.py clip.mp4 --minimal                                    # boxes + IDs + trails only
python process.py clip.mp4 --no-minimap --no-analytics                  # keep keypoint dots + HUD, drop minimap + km/h
python process.py clip.mp4 -o annotated.mp4 --device cuda
```

`process.py` accepts any container FFmpeg can decode (mp4, mov, mkv, avi, webm,
m4v, ts, flv, wmv, mpg). 720p source → MPS render at ~20 fps, real-time factor
~0.8×.

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

## Pipeline guards (why the minimap doesn't lie)

The live pipeline isn't just "predict keypoints → solve homography → project
players". Several guards stop the obvious failure modes from polluting the
output.

**Homography validation** (`src/football_tracker/pitch/dynamic_homography.py`):

- Kp conf gate (≥ 0.4) — kills the model's tendency to hallucinate off-screen
  keypoints when the pitch isn't fully framed.
- Minimum 6 visible keypoints + 5 RANSAC inliers — RANSAC has a real outlier
  budget rather than fitting from 4 noisy correspondences.
- World-span and image-span checks — reject H's fitted from a cluster of
  keypoints that span < 15 % of the pitch (or < 15 % of the image).
- **Orientation sanity** — when TL/TR (or BL/BR) are both above conf, their
  pixel ordering must match the world ordering. Catches the swap-bug that
  mirrors the minimap on partial-pitch frames.
- **Wild-fit rejection** — a candidate H that projects the image centre more
  than 40 m away from the smoothed H's projection is dropped as outlier.
- **Unlimited extrapolation with stale tint** — the last good H is held
  indefinitely; past 12 s without a fresh fit the minimap is tinted amber
  so the viewer knows it's stale.

**Speed math** (`src/football_tracker/analytics/distance.py`) is gated to
**fresh-fit frames only**. Camera pans during extrapolation would otherwise
project a stationary player as moving with the camera. Per-sample delta is
divided by `dt_frames / fps` so gaps between fresh fits are measured correctly;
sliding-window median rejects spikes; constant noise floor (5 cm) pins truly
stationary players to 0 km/h; hard cap at 40 km/h.

**Tracking** uses `supervision.ByteTrack` with `lost_track_buffer = 90` so an
ID survives ~3.6 s of occlusion (replays, ad-board cuts, close-ups). Trails
fade comet-tail style — newest segment full brightness + 3 px, oldest 25 %
intensity + 1 px, anti-aliased throughout.

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
├── dataset.py          ← data download + preprocess + augment + merge
├── train.py            ← train both models, dumps train JSON
├── eval.py             ← detector / pitch / homography eval, dumps JSON
├── demo.py             ← interactive live preview (cv2 window)
├── process.py          ← batch: video in → annotated MP4 out + summary table
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
├── samples/clips/      # 5 × 2-min Liverpool–PSG tactical-cam fragments (1080p, git-tracked)
├── outputs/
│   └── reports/                 # JSON dumps from every run (gitignored)
├── src/football_tracker/
│   ├── datasets/       # SoccerNet, Roboflow (players + pitch), YouTube, merge,
│   │                   # preprocess (green-suppression), augment_pitch_dataset
│   ├── training/       # detector / pitch trainers, eval, ONNX export, report_utils
│   ├── tracking/       # ByteTrack wrapper (lost_buffer=90, returns tracked + untracked)
│   ├── pitch/          # keypoint scheme, dynamic homography (orientation + jump guards
│   │                   # + EMA + unlimited extrap), minimap, preprocess
│   ├── analytics/      # distance / speed (fresh-fit gated, median window) + heatmap
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
- Distance covered + speed per player — fresh-fit gated, median-windowed,
  realistic 40 km/h ceiling.
- Per-frame homography from learned pitch keypoints (RANSAC + EMA smoothing,
  orientation + jump guards, unlimited extrapolation with stale tint).
- Structured JSON dumps of every run for offline analysis.
- Pre-flight dataset checks and clear errors in train / eval scripts.
- One-shot batch CLI (`process.py`) with stage timing + summary table +
  `--minimal` flag.
- Offline augmentation pipeline (`dataset.py augment-pitch`) for the
  keypoint retrain.

Stretch:
- Heatmaps per player / team (module ready; not yet wired to live HUD).
- Pass / ball-contact detection.
- Formation analysis.

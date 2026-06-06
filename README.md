# Football Tracker — AGH AiPO

Player detection, tracking, and tactical analytics for football match footage.

`YOLO26` detector → `ByteTrack` tracker → `YOLO26-pose` pitch keypoints →
per-frame homography → top-down minimap + per-player distance / speed.

## Entry points

| Script        | Subcommands                                              |
|---------------|----------------------------------------------------------|
| `dataset.py`  | `soccernet`, `roboflow`, `pitch`, `youtube`, `merge`, `all` |
| `train.py`    | `detector`, `pitch`, `all`, `export`                     |
| `eval.py`     | `detector`, `pitch`, `homography`, `all`                 |
| `demo.py`     | flags only — the unified live pipeline                    |

Run `python <script>.py --help` for flags. `Makefile` wraps the common invocations.

## Models

| Component       | What                                  | Trained? | Artifact                       |
|-----------------|---------------------------------------|----------|--------------------------------|
| Player detector | YOLO26, 4 classes                     | yes      | `models/checkpoints/best.pt`   |
| Pitch model     | YOLO26n-pose, 32 keypoints            | yes      | `models/checkpoints/pitch.pt`  |
| Tracker         | ByteTrack (Kalman + Hungarian)        | no       | —                              |
| Homography      | RANSAC fit over visible keypoints     | no       | —                              |

Classes: `0 player · 1 goalkeeper · 2 referee · 3 ball` (`configs/classes.yaml`).
Pitch landmarks: 32 points in pitch metres (`configs/pitch_keypoints.yaml`).

Both `.pt` files are the only artifacts that cross machines.

## Train (RTX box)

```bash
git clone <repo> && cd AiPO-proj
make install-cuda

export ROBOFLOW_API_KEY=...
# optional, for SoccerNet:
export SOCCERNET_PASSWORD=...

python dataset.py all     # Roboflow players + pitch + merge
python train.py all       # → best.pt + pitch.pt
python eval.py all        # mAP for both
```

Ship `models/checkpoints/best.pt` and `models/checkpoints/pitch.pt` back.

## Run (Mac)

```bash
make install-mac
# drop best.pt + pitch.pt into models/checkpoints/
python demo.py --source clip.mp4        # or --source 0 for webcam
```

With `pitch.pt` the homography is recomputed every frame — the HUD shows
`H: dyn (24/32 kpts)`. Without it, the first frame asks you to click the four
pitch corners (TL → TR → BR → BL); the matrix is saved and reused.

**No trained model?** `python demo.py --source clip.mp4 --weights /dev/null --no-pitch`
falls back to stock `yolo26n.pt` + COCO `person`/`sports ball`.

**Post-install check:** `python scripts/smoke_test.py` (≈5 s, no GPU).

## Layout

```
.
├── dataset.py train.py eval.py demo.py     # entry points
├── Makefile                                # one-line wrappers
├── configs/                                # class schemes, keypoints, hyperparams
├── scripts/smoke_test.py
└── src/football_tracker/
    ├── datasets/      # SoccerNet, Roboflow (players + pitch), YouTube, merge
    ├── training/      # trainers, eval (detector + pitch + homography), ONNX export
    ├── tracking/      # ByteTrack wrapper, per-ID trail buffer
    ├── pitch/         # keypoint scheme, static + dynamic homography, minimap
    ├── analytics/     # distance / speed / heatmap
    └── pipeline/      # live.py — combined inference loop
```

The `football_tracker` package is importable on its own; the top-level scripts
are thin argparse wrappers around it.

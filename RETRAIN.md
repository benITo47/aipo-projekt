# Pitch model retrain — RTX box playbook

Hand-off doc for the teammate's AI agent on the CUDA box. Follow steps in
order; each command is self-contained. Stage 1 produces `pitch.pt`; Stage 2
produces `pitch_finetuned.pt`. Both should be pushed back when done.

If anything fails, the **Failure modes** section at the bottom covers the
known ones. Don't improvise — the recipes are tuned.

---

## TL;DR — the full sequence

```bash
git pull origin master                                         # get latest

pip install -e '.[training]'                                   # adds albumentations
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# must print: True NVIDIA GeForce RTX 3090  (or similar)

# Stage 1 — from-scratch on partials + originals
python dataset.py augment-pitch                                # ~110 k images, ~10-13 GB
python train_pitch_partials.py                                 # ~6-12 h on RTX 3090

# Stage 2 — fine-tune on partials only
python dataset.py augment-pitch --partials-only \
       --out data/processed/combined_pitch_gs_partials_only    # ~80 k images
python train_pitch_finetune.py                                 # ~1-3 h

# Evaluate both
python eval.py pitch --weights models/checkpoints/pitch.pt
python eval.py pitch --weights models/checkpoints/pitch_finetuned.pt

# Ship back
git add models/checkpoints/pitch.pt models/checkpoints/pitch_finetuned.pt \
        outputs/reports/train/pitch-* outputs/reports/eval/pitch-*
git commit -m "feat(models): pitch v5 — partials + finetune"
git push origin master
```

---

## Step 0 — verify environment

```bash
# Branch + last commit
git log --oneline -3
# expect (or newer): ff4e65a feat(training): two-stage partials retrain

# Python + CUDA
python --version                  # 3.10 or 3.11
nvidia-smi                        # confirm GPU present
```

If `nvidia-smi` doesn't see the GPU, **stop**. Don't try CPU training.

---

## Step 1 — base dataset must exist

```bash
ls data/processed/combined_pitch_gs/
# expect: train/  val/  data.yaml

cat data/processed/combined_pitch_gs/data.yaml | head
# data: must have kpt_shape: [32, 3] and a 32-entry flip_idx
```

If missing, regenerate (needs env vars):

```bash
export ROBOFLOW_API_KEY=...
export SOCCERNET_PASSWORD=...

python dataset.py pitch                                        # Roboflow tactical, ~298 train
python dataset.py soccernet-pitch --zip /path/to/train.zip     # ~13 k SoccerNet broadcast
python dataset.py preprocess-pitch                             # green-suppression merge
```

Expected combined+gs size: ~13.3 k train, ~2.5 k val, every label with
≥ 4 visible keypoints.

---

## Step 2 — augment for Stage 1

```bash
python dataset.py augment-pitch
# Defaults: --copies 2  +  --partials (now ON by default)
# Output:   data/processed/combined_pitch_gs_aug/data.yaml
```

Per source image: 1 symlinked original + 2 photometric copies + up to 11
partial-pitch crops (halves / quarters / centre / left-goal / right-goal,
each photometric-augmented). Crops with < 4 visible keypoints after
translation are silently dropped.

**Expected output**: 100-130 k train images, ~10-13 GB on disk. Val
symlinked unchanged (~2.5 k).

**Sanity check after the run**:
```bash
ls data/processed/combined_pitch_gs_aug/train/images/ | wc -l    # ~100-130 k
ls data/processed/combined_pitch_gs_aug/train/labels/ | wc -l    # same
df -h data/processed/                                            # confirm not full
```

---

## Step 3 — Stage 1 training

```bash
python train_pitch_partials.py
```

Reads `configs/training_pitch_partials.yaml`. Key params (don't change
without reading the rationale comments in the config):

- `model: yolo26s-pose.pt`  — fresh start, keypoint head re-init for 32 kpts
- `lr0: 0.001`              — head trains from scratch, needs the higher LR
- `epochs: 60`              — augmented dataset is 8× larger than baseline; converges faster
- `batch: 16, imgsz: 960`   — ~6 GB VRAM on yolo26s-pose
- `cos_lr: true`, `close_mosaic: 10`, `crop_fraction: 0.85`

**Wall-clock**: 6-12 h on RTX 3090.

**Output**:
- `models/checkpoints/pitch.pt` (best checkpoint, copied automatically)
- `runs/pose/yolo26s_pitch_partials/`  — Ultralytics run dir
- `outputs/reports/train/pitch-yolo26s_pitch_partials-<ts>.json`

**Validation**: the report JSON's last-epoch metrics. Look for:
- `metrics/mAP50(P)` ≥ 0.88  (keypoint mAP @ IoU 0.5)
- `metrics/mAP50-95(P)` ≥ 0.72
- `metrics/mAP50(B)` ≥ 0.93  (bbox mAP)

If keypoint mAP < 0.7, something went wrong — see Failure modes.

---

## Step 4 — augment for Stage 2 (partials only)

```bash
python dataset.py augment-pitch --partials-only \
       --out data/processed/combined_pitch_gs_partials_only
```

Writes only the 11 partial crops per source — no original, no full-frame
photometric copies. Val stays symlinked unchanged.

**Expected output**: ~70-90 k train images. Val unchanged.

**Sanity check**:
```bash
ls data/processed/combined_pitch_gs_partials_only/train/images/ | wc -l   # 70-90 k
ls data/processed/combined_pitch_gs_partials_only/train/images/ | head -5
# expect names like SoccerNetFrame_xyz_top_half.jpg, _bottom_left.jpg, etc.
```

---

## Step 5 — Stage 2 fine-tuning

```bash
python train_pitch_finetune.py
```

Reads `configs/training_pitch_partials_finetune.yaml`. Key params:

- `model: models/checkpoints/pitch.pt`  — the Stage 1 output
- `lr0: 0.00005`              — **20× lower** than Stage 1; protects existing weights
- `epochs: 30`                — short adaptation run
- light on-the-fly aug (dataset already carries variance)

**Wall-clock**: 1-3 h on RTX 3090.

**Output**:
- `models/checkpoints/pitch_finetuned.pt` — does **NOT** overwrite `pitch.pt`
- `runs/pose/yolo26s_pitch_partials_finetune/`
- `outputs/reports/train/pitch-yolo26s_pitch_partials_finetune-<ts>.json`

**Validation**: keypoint mAP50 should be **at least 95 % of Stage 1's** on
the val set. Lower than Stage 1 by a small margin is fine — the val set is
mostly full-pitch, so a partials-specialist will read slightly lower there.
The fine-tune wins on partial-pitch broadcast clips, which the standard
eval doesn't directly measure.

---

## Step 6 — evaluate both checkpoints

```bash
python eval.py pitch --weights models/checkpoints/pitch.pt
python eval.py pitch --weights models/checkpoints/pitch_finetuned.pt
```

Each writes a JSON to `outputs/reports/eval/pitch-<ts>.json` with mAP50,
mAP50-95, precision, recall, per-class breakdown. Both files needed for
the report.

Optional but useful — run the homography eval on a stored broadcast clip:

```bash
python eval.py homography --source samples/clips/liv_psg_clip1.mp4 \
       --weights models/checkpoints/pitch.pt --stride 5 --device cuda
python eval.py homography --source samples/clips/liv_psg_clip1.mp4 \
       --weights models/checkpoints/pitch_finetuned.pt --stride 5 --device cuda
```

Compare fit-rate + rejection-reason histograms. The fine-tune should beat
`pitch.pt` on partial-pitch frames.

---

## Step 7 — ship back

Push everything in one commit:

```bash
git add models/checkpoints/pitch.pt models/checkpoints/pitch_finetuned.pt \
        outputs/reports/train/pitch-yolo26s_pitch_partials-*.json \
        outputs/reports/train/pitch-yolo26s_pitch_partials_finetune-*.json \
        outputs/reports/eval/pitch-*.json
git commit -m "feat(models): pitch v5 — partials + finetune

Stage 1 (from scratch on aug dataset): keypoint mAP50 = X.XXX, mAP50-95 = X.XXX
Stage 2 (finetune on partials-only):   keypoint mAP50 = X.XXX, mAP50-95 = X.XXX

Both .pt files committed; pitch.pt is the from-scratch version, pitch_finetuned.pt
is the partials-specialised follow-up. Swap by hand if eval favours the latter."
git push origin master
```

Fill in the actual mAP numbers from the JSONs.

---

## Cleanup (optional, after pushing)

Augmented datasets are large (~10-13 GB each). Safe to delete once the
checkpoints are pushed and the eval JSONs are saved:

```bash
rm -rf data/processed/combined_pitch_gs_aug
rm -rf data/processed/combined_pitch_gs_partials_only
```

Source `combined_pitch_gs/` should be kept — it's the regeneratable input.

---

## Failure modes

### CUDA OOM during training

Drop `batch: 16` → `batch: 8` in both configs:
- `configs/training_pitch_partials.yaml`
- `configs/training_pitch_partials_finetune.yaml`

Don't touch `imgsz` — 960 is required for the 32-keypoint scheme to fit.

### `cuSOLVER Cholesky failed` during RLE loss

Already handled — `train_pitch.py` switches to `magma` when CUDA. If
something still fails here, your PyTorch wheel doesn't have magma:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### `augment-pitch` exhausts disk

Each augmented dataset is 10-13 GB. With both stages you need ~25 GB free.
If tight: drop `--copies 2` → `--copies 1` for Stage 1 only, which halves
the photometric duplicates.

### albumentations import / API errors

We pin `>=1.4,<2.0` in `pyproject.toml`. albumentations 2.0 broke
`CoarseDropout` and `KeypointParams` signatures. If the install pulled 2.x:
```bash
pip install 'albumentations>=1.4,<2.0' --force-reinstall
```

### Partials dataset has very few samples

If `ls data/processed/combined_pitch_gs_partials_only/train/images/ | wc -l`
returns < 30 k, the SoccerNet labels are sparse or the source frames are
already cropped tight. Check:
```bash
head -1 data/processed/combined_pitch_gs/train/labels/*.txt | awk '{print NF}'
# every label must have 101 fields (1 cls + 4 bbox + 32×3 kpts)
```
If field count differs, the conversion is broken — re-run `dataset.py preprocess-pitch`.

### Stage 1 keypoint mAP < 0.7

Almost always one of:
1. `lr0` was changed below 0.001 — the keypoint head can't escape init.
   Confirm `configs/training_pitch_partials.yaml:23` reads `lr0: 0.001`.
2. Training was killed before `close_mosaic` kicked in (epoch 50+). Resume
   from the last checkpoint:
   ```bash
   yolo train resume=True model=runs/pose/yolo26s_pitch_partials/weights/last.pt
   ```
3. The dataset YAML lost its `flip_idx`. Verify
   `data/processed/combined_pitch_gs_aug/data.yaml` has a 32-entry
   `flip_idx` line.

### Resuming a killed training run

```bash
# For Stage 1:
yolo train resume=True model=runs/pose/yolo26s_pitch_partials/weights/last.pt

# For Stage 2:
yolo train resume=True model=runs/pose/yolo26s_pitch_partials_finetune/weights/last.pt
```

---

## Pre-push validation checklist

Before `git push`, confirm:

- [ ] `models/checkpoints/pitch.pt` exists and is ~25-30 MB (yolo26s-pose size)
- [ ] `models/checkpoints/pitch_finetuned.pt` exists and is similarly sized
- [ ] `python -c "from ultralytics import YOLO; m = YOLO('models/checkpoints/pitch.pt'); print(m.model)"` loads without error
- [ ] Both eval JSONs exist in `outputs/reports/eval/`
- [ ] Both train JSONs exist in `outputs/reports/train/`
- [ ] Eval JSON for `pitch.pt`: `metrics/mAP50(P)` ≥ 0.88
- [ ] Train JSON for Stage 1: `epoch` reached the configured `epochs` (60) OR the patience-based stop was hit (look at `best_fitness`)
- [ ] No huge augmented datasets accidentally committed: `git status` should show only the `.pt` files + JSON reports

When all boxes are ticked, push. The Mac dev will pull and evaluate which
checkpoint becomes the production `pitch.pt`.

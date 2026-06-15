"""Programmatic offline augmentation for YOLO-pose pitch keypoint datasets.

YOLO's on-the-fly augmentation menu (mosaic, scale, perspective, fliplr,
randaugment, erasing) is strong on geometry and weak on photometric variance —
each training run sees the same 13 k frames in roughly the same colour/lighting
distribution, scrambled differently per epoch.

This module grows the dataset by writing N augmented copies of every image to
disk with photometric jitter (HSV / gamma / CLAHE / blur / noise / compression
/ shadow-style dropout / occasional grayscale) plus mild geometric perturbation
(small rotate / translate / scale). Keypoint coordinates and visibilities are
transformed alongside the image via `albumentations`, so the labels stay
correct. The bbox is recomputed from the post-transform visible keypoints.

Usage::

    python dataset.py augment-pitch                        # defaults
    python dataset.py augment-pitch --copies 3 --seed 42

Or as a standalone script::

    python src/football_tracker/datasets/augment_pitch_dataset.py \\
        --src data/processed/combined_pitch_gs/data.yaml \\
        --out data/processed/combined_pitch_gs_aug \\
        --copies 2
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.progress import track

console = Console()

# Skip samples with fewer than this many visible keypoints after transform —
# YOLO can train on partial-pitch frames but anything below the homography
# minimum is useless as a training signal.
MIN_VISIBLE_KPTS = 4
NUM_KEYPOINTS = 32


def build_pipeline():
    """Augmentation pipeline.  Each call randomises the per-transform probability
    independently, so two augmented copies of the same source image look
    visibly different."""
    try:
        import albumentations as A
    except ImportError as e:
        raise SystemExit(
            "albumentations is required. Install with:\n"
            "  pip install 'albumentations>=1.4,<2.0'\n"
            "(or `pip install -e '.[training]'` to pull the training extras)"
        ) from e

    return A.Compose(
        [
            # ---- Photometric (most weight here — YOLO under-samples this) ----
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                    A.MedianBlur(blur_limit=3, p=1.0),
                ],
                p=0.25,
            ),
            A.GaussNoise(p=0.25),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
            A.ToGray(p=0.04),
            # ---- Occlusion (crowd / players / graphics blocking lines) ----
            A.CoarseDropout(
                max_holes=5, max_height=40, max_width=40,
                min_holes=1, min_height=8, min_width=8, p=0.25,
            ),
            # ---- Mild geometric (YOLO covers big shifts; we add small ones) ----
            A.Affine(
                rotate=(-5, 5),
                translate_percent=(-0.04, 0.04),
                scale=(0.95, 1.05),
                interpolation=cv2.INTER_LINEAR,
                p=0.5,
            ),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
            label_fields=["kpt_labels"],
        ),
    )


def _parse_label(label_path: Path, img_w: int, img_h: int):
    """Read YOLO-pose label (32 keypoints, format `cls cx cy w h x0 y0 v0 ...`).
    Returns (cls, kpts_px [(x,y) pixels], visibilities [int]) or None."""
    if not label_path.exists():
        return None
    parts = label_path.read_text().strip().split()
    expected = 1 + 4 + NUM_KEYPOINTS * 3
    if len(parts) != expected:
        return None
    cls = int(parts[0])
    kp_chunks = parts[5:]
    kpts_px = []
    vis = []
    for i in range(NUM_KEYPOINTS):
        x = float(kp_chunks[3 * i]) * img_w
        y = float(kp_chunks[3 * i + 1]) * img_h
        v = int(kp_chunks[3 * i + 2])
        kpts_px.append((x, y))
        vis.append(v)
    return cls, kpts_px, vis


def _write_label(label_path: Path, cls: int, bbox_norm, kpts_norm, vis):
    parts = [str(cls)]
    parts.extend(f"{v:.6f}" for v in bbox_norm)
    for (x, y), v in zip(kpts_norm, vis):
        parts.extend([f"{x:.6f}", f"{y:.6f}", str(v)])
    label_path.write_text(" ".join(parts) + "\n")


def _augment_one(
    img_path: Path, label_path: Path,
    out_img: Path, out_label: Path,
    pipeline,
    jpeg_quality: int,
) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    parsed = _parse_label(label_path, w, h)
    if parsed is None:
        return False
    cls, kpts_px, vis = parsed

    out = pipeline(image=img, keypoints=kpts_px, kpt_labels=list(range(NUM_KEYPOINTS)))
    new_img = out["image"]
    new_kpts = out["keypoints"]
    nh, nw = new_img.shape[:2]

    # Update visibilities: anything mapped outside the frame after transform
    # becomes invisible (=0). Already-invisible stays invisible.
    new_vis = []
    visible_xs, visible_ys = [], []
    for (x, y), v in zip(new_kpts, vis):
        if v == 0 or not (0 <= x < nw and 0 <= y < nh):
            new_vis.append(0)
        else:
            new_vis.append(v)
            visible_xs.append(x)
            visible_ys.append(y)

    if sum(1 for v in new_vis if v > 0) < MIN_VISIBLE_KPTS:
        return False

    # Recompute bbox from visible keypoints, with a 5% padding
    bx_min, bx_max = min(visible_xs), max(visible_xs)
    by_min, by_max = min(visible_ys), max(visible_ys)
    bw = min(((bx_max - bx_min) * 1.05) / nw, 1.0)
    bh = min(((by_max - by_min) * 1.05) / nh, 1.0)
    bx = (bx_min + bx_max) / 2 / nw
    by = (by_min + by_max) / 2 / nh

    kpts_norm = []
    for (x, y), v in zip(new_kpts, new_vis):
        kpts_norm.append((0.0, 0.0) if v == 0 else (x / nw, y / nh))

    cv2.imwrite(str(out_img), new_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    _write_label(out_label, cls, [bx, by, bw, bh], kpts_norm, new_vis)
    return True


def _resolve_img_dirs(cfg: dict) -> tuple[list[Path], list[Path]]:
    root = Path(cfg.get("path", ".")).resolve()
    def _dirs(key: str) -> list[Path]:
        v = cfg.get(key, [])
        items = [v] if isinstance(v, str) else v
        return [Path(p) if Path(p).is_absolute() else root / p for p in items]
    return _dirs("train"), _dirs("val")


def augment_dataset(
    src_yaml: Path,
    out_dir: Path,
    copies: int = 2,
    seed: int = 0,
    jpeg_quality: int = 90,
    augment_val: bool = False,
) -> Path:
    """Read a YOLO-pose data.yaml, write each train image plus `copies`
    augmented copies into `out_dir`. The val split is symlinked over (no
    augmentation) by default — keep eval comparable to past runs.

    Returns the path to the new data.yaml.
    """
    random.seed(seed)
    np.random.seed(seed)
    cfg = yaml.safe_load(src_yaml.read_text())
    train_dirs, val_dirs = _resolve_img_dirs(cfg)
    pipeline = build_pipeline()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _process(src_dirs: list[Path], kind: str, do_aug: bool) -> list[str]:
        out_image_dirs: list[str] = []
        for i, src_img_dir in enumerate(src_dirs):
            suffix = "" if len(src_dirs) == 1 else str(i)
            split_dir = out_dir / f"{kind}{suffix}"
            out_imgs = split_dir / "images"
            out_lbls = split_dir / "labels"
            out_imgs.mkdir(parents=True, exist_ok=True)
            out_lbls.mkdir(parents=True, exist_ok=True)
            src_lbl_dir = src_img_dir.parent / "labels"
            imgs = sorted(src_img_dir.glob("*.jpg")) + sorted(src_img_dir.glob("*.png"))
            multiplier = (copies + 1) if do_aug else 1
            console.log(
                f"[cyan]{kind}{suffix}[/] {src_img_dir} — "
                f"{len(imgs)} originals × {multiplier} = {len(imgs) * multiplier}"
            )

            aug_written = 0
            for img_path in track(imgs, description=f"[cyan]{kind}{suffix}[/]"):
                src_lbl = src_lbl_dir / img_path.with_suffix(".txt").name

                # Symlink (fast, near-zero disk) the original image + label
                dst_orig = out_imgs / img_path.name
                dst_orig_lbl = out_lbls / src_lbl.name
                if not dst_orig.exists():
                    try:
                        dst_orig.symlink_to(img_path.resolve())
                    except OSError:
                        shutil.copy2(img_path, dst_orig)
                if src_lbl.exists() and not dst_orig_lbl.exists():
                    try:
                        dst_orig_lbl.symlink_to(src_lbl.resolve())
                    except OSError:
                        shutil.copy2(src_lbl, dst_orig_lbl)

                # Augmented copies (only when do_aug)
                if do_aug and src_lbl.exists():
                    for j in range(copies):
                        aug_stem = f"{img_path.stem}_aug{j}"
                        aug_img = out_imgs / f"{aug_stem}.jpg"
                        aug_lbl = out_lbls / f"{aug_stem}.txt"
                        if aug_img.exists() and aug_lbl.exists():
                            continue
                        if _augment_one(
                            img_path, src_lbl, aug_img, aug_lbl,
                            pipeline, jpeg_quality,
                        ):
                            aug_written += 1
            if do_aug:
                console.log(f"  [green]→ {aug_written} augmented samples written[/]")
            out_image_dirs.append(str(out_imgs.resolve()))
        return out_image_dirs

    out_train = _process(train_dirs, "train", do_aug=True)
    out_val = _process(val_dirs, "val", do_aug=augment_val)

    new_cfg = {k: v for k, v in cfg.items() if k not in ("path", "train", "val")}
    new_cfg["path"] = str(out_dir.resolve())
    new_cfg["train"] = out_train if len(out_train) > 1 else out_train[0]
    new_cfg["val"] = out_val if len(out_val) > 1 else out_val[0]
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.dump(new_cfg, default_flow_style=False, sort_keys=False))
    console.log(f"[green]Done.[/] Dataset at {out_dir}\n         YAML: {out_yaml}")
    return out_yaml


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src", type=Path,
        default=Path("data/processed/combined_pitch_gs/data.yaml"),
        help="Source data.yaml. Default: the green-suppressed combined dataset.",
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("data/processed/combined_pitch_gs_aug"),
    )
    p.add_argument("--copies", type=int, default=2,
                   help="Augmented copies per source image (final size = (copies+1)×).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--augment-val", action="store_true",
                   help="Also augment the val split. Off by default so eval mAP "
                        "stays comparable to non-augmented runs.")
    args = p.parse_args()
    augment_dataset(
        args.src, args.out, args.copies, args.seed, args.jpeg_quality, args.augment_val,
    )

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


# Strategic partial-pitch crops — each entry is (name, (x_min, y_min, x_max, y_max))
# in normalised image coords. The model otherwise mislabels centre-line keypoints
# as left side-line keypoints when only the right half is visible (and vice
# versa) because every training frame showed the full pitch; these crops force
# it to learn "this is what a half / quarter / corner view actually looks like".
PARTIAL_CROPS: list[tuple[str, tuple[float, float, float, float]]] = [
    # Halves — overlap by 5 % so keypoints near the cut line aren't always lost.
    ("top_half",     (0.00, 0.00, 1.00, 0.55)),
    ("bottom_half",  (0.00, 0.45, 1.00, 1.00)),
    ("left_half",    (0.00, 0.00, 0.55, 1.00)),
    ("right_half",   (0.45, 0.00, 1.00, 1.00)),
    # Quarters
    ("top_left",     (0.00, 0.00, 0.60, 0.60)),
    ("top_right",    (0.40, 0.00, 1.00, 0.60)),
    ("bottom_left",  (0.00, 0.40, 0.60, 1.00)),
    ("bottom_right", (0.40, 0.40, 1.00, 1.00)),
    # Targeted zooms
    ("centre",       (0.25, 0.20, 0.75, 0.80)),
    ("left_goal",    (0.00, 0.20, 0.40, 0.80)),
    ("right_goal",   (0.60, 0.20, 1.00, 0.80)),
]


def _apply_partial_crop(
    img: np.ndarray,
    kpts_px: list[tuple[float, float]],
    vis: list[int],
    crop_norm: tuple[float, float, float, float],
) -> tuple[np.ndarray, list[tuple[float, float]], list[int]]:
    """Crop the image to a normalised rect and translate keypoints. Keypoints
    that fall outside the crop become invisible (vis=0)."""
    h, w = img.shape[:2]
    x0_n, y0_n, x1_n, y1_n = crop_norm
    px0 = max(0, int(x0_n * w))
    py0 = max(0, int(y0_n * h))
    px1 = min(w, int(x1_n * w))
    py1 = min(h, int(y1_n * h))
    if px1 - px0 < 32 or py1 - py0 < 32:
        return img, kpts_px, vis   # crop degenerate — fall through
    cropped = img[py0:py1, px0:px1].copy()
    nh, nw = cropped.shape[:2]
    new_kpts: list[tuple[float, float]] = []
    new_vis: list[int] = []
    for (x, y), v in zip(kpts_px, vis):
        if v == 0:
            new_kpts.append((0.0, 0.0))
            new_vis.append(0)
            continue
        nx = x - px0
        ny = y - py0
        if 0 <= nx < nw and 0 <= ny < nh:
            new_kpts.append((nx, ny))
            new_vis.append(v)
        else:
            new_kpts.append((0.0, 0.0))
            new_vis.append(0)
    return cropped, new_kpts, new_vis


def _augment_one(
    img_path: Path, label_path: Path,
    out_img: Path, out_label: Path,
    pipeline,
    jpeg_quality: int,
    crop_norm: tuple[float, float, float, float] | None = None,
) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    parsed = _parse_label(label_path, w, h)
    if parsed is None:
        return False
    cls, kpts_px, vis = parsed

    # Optional pre-crop to teach the model partial-pitch views directly.
    if crop_norm is not None:
        img, kpts_px, vis = _apply_partial_crop(img, kpts_px, vis, crop_norm)
        # Skip if the crop already left too few visible keypoints.
        if sum(1 for v in vis if v > 0) < MIN_VISIBLE_KPTS:
            return False

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
    partials: bool = False,
) -> Path:
    """Read a YOLO-pose data.yaml, write each train image plus `copies`
    photometric augmented copies into `out_dir`. With ``partials=True``,
    each source also gets one variant per entry in :data:`PARTIAL_CROPS`
    (top/bottom/left/right halves, four quarters, centre, two goal areas).
    The val split is symlinked over (no augmentation) by default so eval
    mAP stays comparable to past runs.

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
            per_source = 1
            if do_aug:
                per_source = 1 + copies + (len(PARTIAL_CROPS) if partials else 0)
            console.log(
                f"[cyan]{kind}{suffix}[/] {src_img_dir} — "
                f"{len(imgs)} originals × ≤{per_source} = up to "
                f"{len(imgs) * per_source} samples"
                + (f"  (incl. {len(PARTIAL_CROPS)} partial crops/src)" if partials else "")
            )

            aug_written = 0
            partial_written = 0
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

                if not (do_aug and src_lbl.exists()):
                    continue

                # Photometric copies of the full frame
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

                # Strategic partial-pitch crops (each then photometric-augmented)
                if partials:
                    for crop_name, crop_norm in PARTIAL_CROPS:
                        pc_stem = f"{img_path.stem}_{crop_name}"
                        pc_img = out_imgs / f"{pc_stem}.jpg"
                        pc_lbl = out_lbls / f"{pc_stem}.txt"
                        if pc_img.exists() and pc_lbl.exists():
                            continue
                        if _augment_one(
                            img_path, src_lbl, pc_img, pc_lbl,
                            pipeline, jpeg_quality, crop_norm=crop_norm,
                        ):
                            partial_written += 1
            if do_aug:
                console.log(
                    f"  [green]→ {aug_written} photometric + "
                    f"{partial_written} partial-crop samples written[/]"
                )
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
    p.add_argument("--partials", action="store_true",
                   help=("Also generate strategic partial-pitch crops per source "
                         "(halves, quarters, centre, goal close-ups). Teaches the "
                         "model to recognise partial-pitch views directly instead "
                         "of mislabeling centre keypoints as left-edge keypoints."))
    args = p.parse_args()
    augment_dataset(
        args.src, args.out, args.copies, args.seed, args.jpeg_quality,
        args.augment_val, args.partials,
    )

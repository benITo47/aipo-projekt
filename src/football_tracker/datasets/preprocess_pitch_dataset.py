"""Pre-process all images in a pitch dataset using enhance_pitch_lines.

Creates a new dataset with green-suppressed images and symlinked labels.
The resulting YAML is drop-in for any training config.

Usage::

    PYTHONPATH=src python src/football_tracker/datasets/preprocess_pitch_dataset.py \\
        --src configs/combined_pitch.yaml \\
        --out data/processed/combined_pitch_gs \\
        --green-factor 0.75
"""
from __future__ import annotations

from pathlib import Path

import cv2
import yaml
from rich.console import Console
from rich.progress import track

from football_tracker.pitch.preprocess import enhance_pitch_lines

console = Console()


def _resolve_img_dirs(cfg: dict) -> tuple[list[Path], list[Path]]:
    root = Path(cfg.get("path", "."))
    def _dirs(key: str) -> list[Path]:
        v = cfg.get(key, [])
        items = [v] if isinstance(v, str) else v
        return [Path(p) if Path(p).is_absolute() else root / p for p in items]
    return _dirs("train"), _dirs("val")


def _process_img_dir(src_img_dir: Path, out_split_dir: Path, green_factor: float) -> None:
    out_img = out_split_dir / "images"
    out_lbl = out_split_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # labels sit next to images/ at the same level
    src_lbl = src_img_dir.parent / "labels"

    imgs = sorted(src_img_dir.glob("*.jpg")) + sorted(src_img_dir.glob("*.png"))
    for img_path in track(imgs, description=f"[cyan]{src_img_dir.parent.parent.name}/{src_img_dir.parent.name}[/]"):
        dst = out_img / img_path.name
        if not dst.exists():
            frame = cv2.imread(str(img_path))
            if frame is not None:
                cv2.imwrite(str(dst), enhance_pitch_lines(frame, green_factor))

        lbl = src_lbl / img_path.with_suffix(".txt").name
        dst_lbl = out_lbl / lbl.name
        if lbl.exists() and not dst_lbl.exists():
            dst_lbl.symlink_to(lbl.resolve())


def preprocess_dataset(src_yaml: Path, out_dir: Path, green_factor: float = 0.75) -> Path:
    cfg = yaml.safe_load(src_yaml.read_text())
    train_dirs, val_dirs = _resolve_img_dirs(cfg)

    out_train_dirs, out_val_dirs = [], []

    for i, d in enumerate(train_dirs):
        split_dir = out_dir / f"train{i if len(train_dirs) > 1 else ''}"
        _process_img_dir(d, split_dir, green_factor)
        out_train_dirs.append(str((split_dir / "images").resolve()))

    for i, d in enumerate(val_dirs):
        split_dir = out_dir / f"val{i if len(val_dirs) > 1 else ''}"
        _process_img_dir(d, split_dir, green_factor)
        out_val_dirs.append(str((split_dir / "images").resolve()))

    new_cfg = {k: v for k, v in cfg.items() if k not in ("path", "train", "val")}
    new_cfg["path"] = str(out_dir.resolve())
    new_cfg["train"] = out_train_dirs if len(out_train_dirs) > 1 else out_train_dirs[0]
    new_cfg["val"] = out_val_dirs if len(out_val_dirs) > 1 else out_val_dirs[0]

    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.dump(new_cfg, default_flow_style=False, sort_keys=False))
    console.log(f"[green]Done.[/] Dataset at {out_dir}, YAML: {out_yaml}")
    return out_yaml


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("configs/combined_pitch.yaml"))
    p.add_argument("--out", type=Path, default=Path("data/processed/combined_pitch_gs"))
    p.add_argument("--green-factor", type=float, default=0.75)
    args = p.parse_args()
    preprocess_dataset(args.src, args.out, args.green_factor)

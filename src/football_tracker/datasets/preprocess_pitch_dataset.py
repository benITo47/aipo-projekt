"""Pre-process all images in a pitch dataset using enhance_pitch_lines.

Creates a new dataset directory with preprocessed images and symlinked labels.
The resulting YAML is drop-in for any training config.

Usage::

    python -m football_tracker.datasets.preprocess_pitch_dataset \\
        --src configs/combined_pitch.yaml \\
        --out data/processed/combined_pitch_gs \\
        --green-factor 0.75
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import yaml
from rich.console import Console
from rich.progress import track

from football_tracker.pitch.preprocess import enhance_pitch_lines

console = Console()


def preprocess_dataset(src_yaml: Path, out_dir: Path, green_factor: float = 0.75) -> Path:
    cfg = yaml.safe_load(src_yaml.read_text())
    root = Path(cfg.get("path", "."))

    out_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(p: str | list) -> list[Path]:
        paths = [p] if isinstance(p, str) else p
        return [root / x for x in paths]

    def _process_split(name: str, src_img_dirs: list[Path]) -> list[str]:
        out_rel_paths = []
        for src_img_dir in src_img_dirs:
            if not src_img_dir.exists():
                console.log(f"[yellow]skip missing[/] {src_img_dir}")
                continue
            src_lbl_dir = src_img_dir.parent.parent / "labels" / src_img_dir.name.replace("images", "labels").lstrip("/")
            # simple sibling: images/ → labels/
            src_lbl_dir = src_img_dir.parent / "labels"

            suffix = src_img_dir.relative_to(root) if src_img_dir.is_relative_to(root) else Path(name)
            out_img_dir = out_dir / suffix / "images"
            out_lbl_dir = out_dir / suffix / "labels"
            out_img_dir.mkdir(parents=True, exist_ok=True)
            out_lbl_dir.mkdir(parents=True, exist_ok=True)

            imgs = sorted(src_img_dir.glob("*.jpg")) + sorted(src_img_dir.glob("*.png"))
            for img_path in track(imgs, description=f"[cyan]{name}[/] {src_img_dir.name}"):
                dst = out_img_dir / img_path.name
                if not dst.exists():
                    frame = cv2.imread(str(img_path))
                    if frame is not None:
                        processed = enhance_pitch_lines(frame, green_factor)
                        cv2.imwrite(str(dst), processed)

                lbl_path = src_lbl_dir / img_path.with_suffix(".txt").name
                dst_lbl = out_lbl_dir / lbl_path.name
                if lbl_path.exists() and not dst_lbl.exists():
                    dst_lbl.symlink_to(lbl_path.resolve())

            out_rel_paths.append(str(out_img_dir.relative_to(out_dir)))
        return out_rel_paths

    train_paths = _process_split("train", _resolve(cfg.get("train", "train/images")))
    val_paths = _process_split("val", _resolve(cfg.get("val", "valid/images")))

    new_cfg = {k: v for k, v in cfg.items() if k not in ("path", "train", "val")}
    new_cfg["path"] = str(out_dir.resolve())
    new_cfg["train"] = train_paths if len(train_paths) > 1 else (train_paths[0] if train_paths else "train/images")
    new_cfg["val"] = val_paths if len(val_paths) > 1 else (val_paths[0] if val_paths else "valid/images")

    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.dump(new_cfg, default_flow_style=False, sort_keys=False))
    console.log(f"[green]Dataset written to[/] {out_dir}, YAML: {out_yaml}")
    return out_yaml


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, default=Path("configs/combined_pitch.yaml"))
    p.add_argument("--out", type=Path, default=Path("data/processed/combined_pitch_gs"))
    p.add_argument("--green-factor", type=float, default=0.75)
    args = p.parse_args()
    preprocess_dataset(args.src, args.out, args.green_factor)

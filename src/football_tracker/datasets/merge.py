"""Merge SoccerNet + Roboflow YOLO datasets into a single training YAML.

Both downloaders already remap labels to canonical class IDs (see
`configs/classes.yaml`), so we only need to symlink-or-copy their image/label
trees under one root and emit a combined `data.yaml`.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run(
    soccernet: Path | None,
    roboflow: Path | None,
    out_dir: Path,
    use_symlinks: bool = True,
) -> Path:
    if not soccernet and not roboflow:
        raise SystemExit("Pass at least one of --soccernet / --roboflow.")

    out_dir = out_dir.resolve()
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]] = []
    if soccernet:
        sn_root = soccernet.resolve() / "yolo"
        if not sn_root.exists():
            raise SystemExit(
                f"SoccerNet YOLO root missing: {sn_root}. "
                "Run `aipo download soccernet` (with --convert) first."
            )
        sources.append(("soccernet", sn_root))
    if roboflow:
        rf_root = roboflow.resolve()
        if not (rf_root / "data.yaml").exists():
            raise SystemExit(
                f"Roboflow data.yaml missing under: {rf_root}. "
                "Run `aipo download roboflow` first."
            )
        sources.append(("roboflow", rf_root))

    for tag, root in sources:
        _link_split(root, out_dir, "train", tag, use_symlinks)
        _link_split(root, out_dir, "val", tag, use_symlinks)

    combined_yaml = Path("configs/combined.yaml")
    combined_yaml.parent.mkdir(parents=True, exist_ok=True)
    combined_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"},
            },
            sort_keys=False,
        )
    )
    console.log(f"[green]Combined dataset ready[/] → {combined_yaml}")
    console.log(f"[green]Image root[/] → {out_dir}")
    return combined_yaml


def _link_split(src_root: Path, dst_root: Path, split: str, tag: str, use_symlinks: bool) -> None:
    """Source uses either `{split}` (SoccerNet) or `{split}` / `valid` (Roboflow) layouts."""
    # Roboflow YOLO export uses 'valid' instead of 'val'
    src_split = split if (src_root / "images" / split).exists() else (
        "valid" if split == "val" else split
    )
    src_img_dir = src_root / "images" / src_split
    src_lbl_dir = src_root / "labels" / src_split
    # Roboflow puts images and labels under e.g. train/images, train/labels (no top-level images/)
    if not src_img_dir.exists():
        src_img_dir = src_root / src_split / "images"
        src_lbl_dir = src_root / src_split / "labels"
    if not src_img_dir.exists():
        console.log(f"[yellow]Skipping {tag}/{split} — no images at {src_img_dir}[/]")
        return

    n = 0
    for img in src_img_dir.iterdir():
        if not img.is_file():
            continue
        target_img = dst_root / "images" / split / f"{tag}__{img.name}"
        target_lbl = dst_root / "labels" / split / f"{tag}__{img.stem}.txt"
        lbl_src = src_lbl_dir / f"{img.stem}.txt"
        _link_or_copy(img, target_img, use_symlinks)
        if lbl_src.exists():
            _link_or_copy(lbl_src, target_lbl, use_symlinks)
        n += 1
    console.log(f"[cyan]Linked[/] {tag}/{src_split} → {split}  ({n} images)")


def _link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if use_symlinks:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    # fallback (Windows without admin, or cross-filesystem)
    import shutil
    shutil.copy(src, dst)

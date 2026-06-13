"""Evaluate a trained YOLO26 detector — mAP / PR curves on the val split."""
from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from football_tracker.device import pick_device

console = Console()


def _check_dataset_present(data_yaml: Path) -> None:
    """Resolve val path from the YAML and fail fast with a friendly message if missing."""
    body = yaml.safe_load(data_yaml.read_text())
    root = Path(body.get("path", ".")).resolve()
    val_rel = body.get("val", "")
    val_dir = root / val_rel
    if not val_dir.exists() or not any(val_dir.iterdir()):
        raise SystemExit(
            f"Dataset val split is missing or empty: {val_dir}\n"
            "Download / merge the datasets first:\n"
            "  python dataset.py all     # Roboflow players + pitch\n"
            "  python dataset.py merge   # if you already downloaded SoccerNet"
        )


def run(weights: Path, data: Path, imgsz: int = 1280, device: str | None = None) -> None:
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not data.exists():
        raise SystemExit(
            f"Dataset YAML not found: {data}\n"
            "Generate it with `python dataset.py merge` (after `python dataset.py roboflow`)."
        )
    _check_dataset_present(data)

    from ultralytics import YOLO

    device = pick_device(device)
    console.log(f"[cyan]Device[/] {device}")
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), imgsz=imgsz, plots=True, device=device)
    console.log(f"[green]mAP50-95[/] = {metrics.box.map:.4f}")
    console.log(f"[green]mAP50[/]    = {metrics.box.map50:.4f}")
    console.log(f"[green]Precision[/] = {metrics.box.mp:.4f}")
    console.log(f"[green]Recall[/]    = {metrics.box.mr:.4f}")

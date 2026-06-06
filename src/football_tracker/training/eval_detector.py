"""Evaluate a trained YOLO26 detector — mAP / PR curves on the val split."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def run(weights: Path, data: Path, imgsz: int = 1280) -> None:
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not data.exists():
        raise SystemExit(f"Dataset YAML not found: {data}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    metrics = model.val(data=str(data), imgsz=imgsz, plots=True)
    console.log(f"[green]mAP50-95[/] = {metrics.box.map:.4f}")
    console.log(f"[green]mAP50[/]    = {metrics.box.map50:.4f}")
    console.log(f"[green]Precision[/] = {metrics.box.mp:.4f}")
    console.log(f"[green]Recall[/]    = {metrics.box.mr:.4f}")

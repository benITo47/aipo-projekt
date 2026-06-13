"""Evaluate a trained YOLO26-pose pitch keypoint model.

Reports bbox mAP (treating the whole pitch as one box) and keypoint mAP (object
keypoint similarity) on the val split.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

from football_tracker.device import pick_device

console = Console()


def run(
    weights: Path, data: Path, imgsz: int = 960, device: str | None = None
) -> dict[str, float]:
    """imgsz defaults to 960 to match the pitch model's training resolution."""
    if not weights.exists():
        raise SystemExit(f"Pitch weights not found: {weights}")
    if not data.exists():
        raise SystemExit(f"Pitch dataset YAML not found: {data}")

    from ultralytics import YOLO

    device = pick_device(device)
    console.log(f"[cyan]Device[/] {device}")
    model = YOLO(str(weights))
    metrics = model.val(data=str(data), imgsz=imgsz, plots=True, device=device)

    out = {
        "box_map50_95": float(metrics.box.map),
        "box_map50": float(metrics.box.map50),
    }
    # Keypoint metrics are only present on pose models
    if hasattr(metrics, "pose") and metrics.pose is not None:
        out["pose_map50_95"] = float(metrics.pose.map)
        out["pose_map50"] = float(metrics.pose.map50)
        console.log(f"[green]Keypoint mAP50-95[/] = {out['pose_map50_95']:.4f}")
        console.log(f"[green]Keypoint mAP50[/]    = {out['pose_map50']:.4f}")
    console.log(f"[green]Box mAP50-95[/]       = {out['box_map50_95']:.4f}")
    console.log(f"[green]Box mAP50[/]          = {out['box_map50']:.4f}")
    return out

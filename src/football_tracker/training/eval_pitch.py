"""Evaluate a trained YOLO26-pose pitch keypoint model.

Reports bbox mAP (treating the whole pitch as one box) and keypoint mAP (object
keypoint similarity) on the val split.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from football_tracker.device import pick_device
from football_tracker.reporting import dump_json, report_path

console = Console()


def _check_dataset_present(data_yaml: Path) -> None:
    body = yaml.safe_load(data_yaml.read_text())
    root = Path(body.get("path", ".")).resolve()
    val_rel = body.get("val", "")
    # val can be a string or a list of dirs (combined datasets use a list)
    rels = [val_rel] if isinstance(val_rel, str) else (val_rel or [])
    for rel in rels:
        val_dir = Path(rel) if Path(rel).is_absolute() else root / rel
        if not val_dir.exists() or not any(val_dir.iterdir()):
            raise SystemExit(
                f"Pitch dataset val split is missing or empty: {val_dir}\n"
                "Download it first: python dataset.py pitch"
            )


def run(
    weights: Path, data: Path, imgsz: int = 960, device: str | None = None,
    report_dir: Path | None = None,
) -> dict[str, float]:
    """imgsz defaults to 960 to match the pitch model's training resolution."""
    if not weights.exists():
        raise SystemExit(f"Pitch weights not found: {weights}")
    if not data.exists():
        raise SystemExit(
            f"Pitch dataset YAML not found: {data}\n"
            "Generate it with `python dataset.py pitch`."
        )
    _check_dataset_present(data)

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

    report = {
        "task": "pitch_eval",
        "weights": str(weights),
        "data_yaml": str(data),
        "imgsz": imgsz,
        "device": device,
        "metrics": out,
    }
    rp = report_path("eval", "pitch", root=report_dir)
    dump_json(report, rp)
    console.log(f"[cyan]Report[/] → {rp}")
    return out

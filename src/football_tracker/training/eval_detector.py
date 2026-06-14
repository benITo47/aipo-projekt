"""Evaluate a trained YOLO26 detector — mAP / PR curves on the val split."""
from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from football_tracker.device import pick_device
from football_tracker.reporting import dump_json, report_path

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


def run(
    weights: Path, data: Path, imgsz: int = 1280, device: str | None = None,
    report_dir: Path | None = None,
) -> Path:
    """Returns the path to the JSON report dumped in outputs/reports/eval/."""
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

    # ---------- dump ----------
    names = model.names if hasattr(model, "names") else {}
    per_class = {}
    try:
        ap50 = metrics.box.ap50    # per-class mAP@0.5
        for cls_idx, ap in enumerate(ap50):
            per_class[names.get(cls_idx, str(cls_idx))] = {"mAP50": float(ap)}
    except Exception:
        per_class = {}

    report = {
        "task": "detector_eval",
        "weights": str(weights),
        "data_yaml": str(data),
        "imgsz": imgsz,
        "device": device,
        "metrics": {
            "mAP50_95": float(metrics.box.map),
            "mAP50": float(metrics.box.map50),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        },
        "per_class": per_class,
    }
    out = report_path("eval", "detector", root=report_dir)
    dump_json(report, out)
    console.log(f"[cyan]Report[/] → {out}")
    return out

"""Export a trained YOLO26 .pt to ONNX for snappy Mac CPU inference."""
from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()


def run(weights: Path, imgsz: int = 1280, half: bool = False, dynamic: bool = True) -> Path:
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    from ultralytics import YOLO

    console.log(f"[cyan]Exporting[/] {weights} → ONNX (imgsz={imgsz})")
    model = YOLO(str(weights))
    out = model.export(format="onnx", imgsz=imgsz, half=half, dynamic=dynamic, opset=17)
    console.log(f"[green]ONNX exported[/] → {out}")
    return Path(out)

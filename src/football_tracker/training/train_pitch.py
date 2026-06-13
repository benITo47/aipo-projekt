"""Fine-tune YOLO26-pose on the pitch-keypoints dataset.

Mirrors `train_detector.py` but expects a pose-format dataset YAML (`kpt_shape`,
`flip_idx`) and writes the best weights to `models/checkpoints/pitch.pt`.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def _check_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise SystemExit(
            f"Pitch dataset YAML missing: {data_yaml}\n"
            "Run `python dataset.py pitch` first."
        )
    body = yaml.safe_load(data_yaml.read_text())
    root = Path(body.get("path", ".")).resolve()
    for split in ("train", "val"):
        rel = body.get(split, "")
        d = root / rel
        if not d.exists() or not any(d.iterdir()):
            raise SystemExit(
                f"Pitch dataset {split} split missing or empty: {d}\n"
                "Download the data: `python dataset.py pitch`"
            )


def run(config_path: Path) -> Path:
    if not config_path.exists():
        raise SystemExit(f"Pitch training config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    model_name = cfg.pop("model", "yolo26n-pose.pt")
    data_path = Path(cfg.pop("data"))
    cfg.pop("task", None)   # passed via the model itself
    _check_dataset(data_path)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from e

    # YOLO26 RLE loss uses Cholesky decomp — cusolver can fail on some drivers.
    # Fall back to magma which is more stable on consumer GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.preferred_linalg_library("magma")

    console.log(f"[cyan]Loading[/] {model_name}")
    model = YOLO(model_name)

    console.log(f"[cyan]Training pitch keypoint model[/] data={data_path}")
    results = model.train(data=str(data_path), **cfg)

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    target = Path("models/checkpoints/pitch.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, target)
        console.log(f"[green]Copied best pitch weights[/] → {target}")
    return target

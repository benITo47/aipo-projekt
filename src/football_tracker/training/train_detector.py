"""Fine-tune a YOLO26 detector on a YOLO-format dataset.

Entry point used by `aipo train detector`. The YAML at `config_path` is loaded
verbatim and forwarded to `ultralytics.YOLO.train()`, so any new ultralytics
flag works without code changes.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from rich.console import Console

console = Console()


def run(config_path: Path) -> Path:
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    model_name = cfg.pop("model", "yolo26m.pt")
    data_path = Path(cfg.pop("data"))
    if not data_path.exists():
        raise SystemExit(
            f"Dataset YAML missing: {data_path}. "
            "Run `aipo download soccernet` or `aipo download roboflow` first."
        )

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from e

    console.log(f"[cyan]Loading[/] {model_name}")
    model = YOLO(model_name)

    console.log(f"[cyan]Training[/] data={data_path} cfg={config_path}")
    results = model.train(data=str(data_path), **cfg)

    # Copy `best.pt` to a stable path the demo pipeline reads by default.
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    target = Path("models/checkpoints/best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, target)
        console.log(f"[green]Copied best weights[/] → {target}")
    return target

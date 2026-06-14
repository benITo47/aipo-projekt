"""Fine-tune a YOLO26 detector on a YOLO-format dataset.

Entry point used by `python train.py detector`. The YAML at `config_path` is
loaded verbatim and forwarded to `ultralytics.YOLO.train()`, so any new
ultralytics flag works without code changes.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import yaml
from rich.console import Console

from football_tracker.reporting import dump_json, report_path
from football_tracker.training.report_utils import build_training_summary

console = Console()


def _check_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise SystemExit(
            f"Dataset YAML missing: {data_yaml}\n"
            "Run `python dataset.py merge` first (or `python dataset.py all` to "
            "download Roboflow and merge in one shot)."
        )
    body = yaml.safe_load(data_yaml.read_text())
    root = Path(body.get("path", ".")).resolve()
    for split in ("train", "val"):
        rel = body.get(split, "")
        d = root / rel
        if not d.exists() or not any(d.iterdir()):
            raise SystemExit(
                f"Dataset {split} split missing or empty: {d}\n"
                "Download the data: `python dataset.py all`"
            )


def run(config_path: Path, report_dir: Path | None = None) -> Path:
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text())

    model_name = cfg.pop("model", "yolo26m.pt")
    data_path = Path(cfg.pop("data"))
    _check_dataset(data_path)

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from e

    console.log(f"[cyan]Loading[/] {model_name}")
    model = YOLO(model_name)

    console.log(f"[cyan]Training detector[/] data={data_path} cfg={config_path}")
    t0 = time.perf_counter()
    results = model.train(data=str(data_path), **cfg)
    elapsed = time.perf_counter() - t0

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    target = Path("models/checkpoints/best.pt")
    target.parent.mkdir(parents=True, exist_ok=True)
    if best.exists():
        shutil.copy(best, target)
        console.log(f"[green]Copied best detector weights[/] → {target}")
    else:
        console.log(f"[yellow]No best.pt found in {save_dir}/weights — keeping previous checkpoint[/]")

    summary = build_training_summary(
        task="detector", save_dir=save_dir,
        config_path=config_path, config=dict(cfg),
        model_base=model_name, data_yaml=str(data_path),
        elapsed_sec=elapsed, output_checkpoint=target,
    )
    run_name = cfg.get("name") or save_dir.name
    rp = report_path("train", f"detector-{run_name}", root=report_dir)
    dump_json(summary, rp)
    console.log(f"[cyan]Training report[/] → {rp}")
    return target

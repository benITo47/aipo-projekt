"""Shared training-report builder. Ultralytics writes per-epoch metrics to
results.csv inside the run save_dir; we parse that into a structured JSON
for offline analysis and the AiPO course report.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


def build_training_summary(
    task: str,                  # "detector" | "pitch"
    save_dir: Path,             # ultralytics run dir
    config_path: Path,
    config: dict[str, Any],     # the parsed training YAML (model + data already popped)
    model_base: str,            # e.g. "yolo26m.pt"
    data_yaml: str,             # e.g. "configs/combined.yaml"
    elapsed_sec: float,
    output_checkpoint: Path,    # where we copied best.pt to
) -> dict[str, Any]:
    """Parse results.csv → produce a JSON-friendly training summary."""
    summary: dict[str, Any] = {
        "task": f"{task}_train",
        "config_path": str(config_path),
        "config": config,
        "model_base": model_base,
        "data_yaml": data_yaml,
        "save_dir": str(save_dir),
        "elapsed_sec": float(elapsed_sec),
        "output_checkpoint": str(output_checkpoint),
        "epochs": [],
        "best_epoch": None,
        "best_metrics": {},
    }

    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return summary

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return summary

    # Normalize keys (ultralytics writes them with spaces sometimes).
    def _f(row: dict[str, str], key: str) -> float | None:
        for k in (key, key.replace("/", " "), key.replace(" ", "/")):
            if k in row and row[k] not in ("", None):
                try:
                    return float(row[k])
                except ValueError:
                    return None
        return None

    for row in rows:
        ep = {k.strip(): v for k, v in row.items()}
        # Coerce floats where we can
        ep = {k: (float(v) if _can_float(v) else v) for k, v in ep.items()}
        summary["epochs"].append(ep)

    # Best epoch — prefer pose mAP50-95 if present, else box mAP50-95
    def _best_metric_key() -> str | None:
        sample_keys = set(summary["epochs"][-1].keys())
        for candidate in (
            "metrics/mAP50-95(P)", "metrics/mAP50-95(B)",
            "metrics/mAP50-95", "metrics/mAP_0.5:0.95",
        ):
            if candidate in sample_keys:
                return candidate
        return None

    metric_key = _best_metric_key()
    if metric_key:
        best = max(summary["epochs"], key=lambda e: e.get(metric_key, -1) or -1)
        summary["best_epoch"] = int(best.get("epoch", -1) or -1)
        summary["best_metrics"] = {
            k: v for k, v in best.items()
            if k.startswith("metrics/") and isinstance(v, float)
        }
    return summary


def _can_float(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def stopwatch() -> tuple[callable, callable]:
    """Returns (start, elapsed). `elapsed()` gives seconds since `start()`."""
    state = {"t0": 0.0}

    def start() -> None:
        state["t0"] = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - state["t0"]

    return start, elapsed

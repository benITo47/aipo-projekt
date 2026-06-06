"""Measure how reliable the dynamic homography is on real footage.

For each frame: run the pitch model, count visible keypoints, attempt to fit a
homography, record whether it succeeded vs fell back. Report aggregate stats.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from football_tracker.pitch.dynamic_homography import DynamicHomographyEstimator

console = Console()


def run(
    source: str,
    pitch_weights: Path,
    max_frames: int | None = None,
    stride: int = 1,
) -> dict[str, float]:
    if not Path(pitch_weights).exists():
        raise SystemExit(f"Pitch weights not found: {pitch_weights}")

    from ultralytics import YOLO

    model = YOLO(str(pitch_weights))
    estimator = DynamicHomographyEstimator()

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    n_processed = 0
    n_fit = 0
    n_fallback = 0
    kpt_counts: list[int] = []
    h_diffs: list[float] = []
    last_H = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        pres = model.predict(frame, verbose=False, imgsz=1280)[0]
        result = estimator.update(pres)
        kpt_counts.append(result.n_visible)

        if result.homography is not None and not result.used_fallback:
            n_fit += 1
            H = result.homography.H
            if last_H is not None:
                h_diffs.append(float(np.linalg.norm(H - last_H)))
            last_H = H
        else:
            n_fallback += 1

        n_processed += 1
        frame_idx += 1
        if max_frames and n_processed >= max_frames:
            break

    cap.release()

    if not kpt_counts:
        raise SystemExit("No frames processed.")

    kpts = np.asarray(kpt_counts)
    fit_rate = n_fit / n_processed
    out = {
        "frames_processed": float(n_processed),
        "fit_rate": fit_rate,
        "fallback_rate": n_fallback / n_processed,
        "kpts_mean": float(kpts.mean()),
        "kpts_std": float(kpts.std()),
        "kpts_min": float(kpts.min()),
        "kpts_max": float(kpts.max()),
        "h_drift_mean": float(np.mean(h_diffs)) if h_diffs else 0.0,
    }

    _report(out)
    return out


def _report(stats: dict[str, float]) -> None:
    table = Table(title="Homography stability", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("frames processed", f"{int(stats['frames_processed'])}")
    table.add_row("frames with fitted H", f"{stats['fit_rate']*100:.1f}%")
    table.add_row("frames using fallback H", f"{stats['fallback_rate']*100:.1f}%")
    table.add_row("keypoints visible (mean ± std)",
                 f"{stats['kpts_mean']:.1f} ± {stats['kpts_std']:.1f}")
    table.add_row("keypoints visible (min / max)",
                 f"{int(stats['kpts_min'])} / {int(stats['kpts_max'])}")
    table.add_row("frame-to-frame H drift (L2)", f"{stats['h_drift_mean']:.4f}")
    console.print(table)

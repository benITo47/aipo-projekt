"""Measure how reliable the dynamic homography is on real footage.

For each frame: run the pitch model, count visible keypoints, attempt to fit a
homography, record whether it succeeded vs fell back. Report aggregate stats
plus a per-keypoint visibility breakdown (useful for deciding whether some
landmarks should be dropped from the canonical scheme).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from football_tracker.device import pick_device
from football_tracker.pitch.dynamic_homography import (
    KP_CONF_THRESHOLD,
    DynamicHomographyEstimator,
)
from football_tracker.pitch.keypoints import load as load_scheme
from football_tracker.pitch.preprocess import enhance_pitch_lines
from football_tracker.reporting import dump_json, report_path

console = Console()


def run(
    source: str,
    pitch_weights: Path,
    max_frames: int | None = None,
    stride: int = 1,
    imgsz: int = 960,            # matches pitch model's training resolution
    device: str | None = None,
    report_dir: Path | None = None,
) -> dict[str, float]:
    """Process `source` through the pitch model, matching the live pipeline's
    preprocessing (green-suppression) + imgsz so the numbers reflect what the
    demo actually sees.
    """
    if not Path(pitch_weights).exists():
        raise SystemExit(f"Pitch weights not found: {pitch_weights}")

    from ultralytics import YOLO

    device = pick_device(device)
    console.log(f"[cyan]Device[/] {device}")
    model = YOLO(str(pitch_weights))
    scheme = load_scheme()
    estimator = DynamicHomographyEstimator()

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    n_processed = 0
    n_fit = 0
    n_fallback = 0
    kpt_counts: list[int] = []
    inlier_counts: list[int] = []
    h_diffs: list[float] = []
    last_H: np.ndarray | None = None
    rejection_reasons: Counter[str] = Counter()
    per_kpt_seen = np.zeros(scheme.num, dtype=np.int64)
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        pres = model.predict(
            enhance_pitch_lines(frame),
            verbose=False, imgsz=imgsz, device=device,
        )[0]

        # Per-keypoint visibility (independent of homography fit)
        if pres.keypoints is not None and pres.keypoints.data is not None:
            kp = pres.keypoints.data.cpu().numpy()
            if kp.shape[0] > 0:
                # pick best detection like the estimator does
                if pres.boxes is not None and pres.boxes.conf is not None:
                    best = int(np.argmax(pres.boxes.conf.cpu().numpy()))
                else:
                    best = 0
                per_kpt_seen += (kp[best, :, 2] > KP_CONF_THRESHOLD).astype(np.int64)

        result = estimator.update(pres, image_shape=frame.shape[:2])
        kpt_counts.append(result.n_visible)
        inlier_counts.append(result.n_inliers)

        if result.homography is not None and not result.used_fallback:
            n_fit += 1
            H = result.homography.H
            if last_H is not None:
                h_diffs.append(float(np.linalg.norm(H - last_H)))
            last_H = H
        else:
            n_fallback += 1
            if result.reason:
                rejection_reasons[result.reason] += 1

        n_processed += 1
        frame_idx += 1
        if max_frames and n_processed >= max_frames:
            break

    cap.release()

    if not kpt_counts:
        raise SystemExit("No frames processed.")

    kpts = np.asarray(kpt_counts)
    inls = np.asarray(inlier_counts)
    fit_rate = n_fit / n_processed
    out = {
        "frames_processed": float(n_processed),
        "fit_rate": fit_rate,
        "fallback_rate": n_fallback / n_processed,
        "kpts_mean": float(kpts.mean()),
        "kpts_std": float(kpts.std()),
        "kpts_min": float(kpts.min()),
        "kpts_max": float(kpts.max()),
        "inliers_mean": float(inls.mean()),
        "h_drift_mean": float(np.mean(h_diffs)) if h_diffs else 0.0,
    }

    _report_summary(out)
    _report_rejections(rejection_reasons, n_processed)
    _report_per_keypoint(per_kpt_seen, n_processed, scheme.names)

    report = {
        "task": "homography_eval",
        "source": source,
        "weights": str(pitch_weights),
        "imgsz": imgsz,
        "stride": stride,
        "device": device,
        "summary": out,
        "rejection_reasons": dict(rejection_reasons),
        "per_keypoint": [
            {"index": i, "name": scheme.names[i],
             "seen_frames": int(per_kpt_seen[i]),
             "seen_rate": float(per_kpt_seen[i] / max(n_processed, 1))}
            for i in range(scheme.num)
        ],
    }
    rp = report_path("eval", "homography", root=report_dir)
    dump_json(report, rp)
    console.log(f"[cyan]Report[/] → {rp}")
    return out


def _report_summary(stats: dict[str, float]) -> None:
    n = int(stats["frames_processed"])
    table = Table(title="Homography stability", show_lines=False)
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("frames processed", f"{n}")
    table.add_row("frames with fitted H", f"{stats['fit_rate']*100:.1f}%")
    table.add_row("frames using fallback H", f"{stats['fallback_rate']*100:.1f}%")
    table.add_row("keypoints visible (mean ± std)",
                 f"{stats['kpts_mean']:.1f} ± {stats['kpts_std']:.1f}")
    table.add_row("keypoints visible (min / max)",
                 f"{int(stats['kpts_min'])} / {int(stats['kpts_max'])}")
    table.add_row("RANSAC inliers (mean)", f"{stats['inliers_mean']:.1f}")
    table.add_row("frame-to-frame H drift (L2)", f"{stats['h_drift_mean']:.4f}")
    console.print(table)


def _report_rejections(reasons: Counter, n_total: int) -> None:
    if not reasons:
        return
    table = Table(title="Why H fits failed (top 5)", show_lines=False)
    table.add_column("reason", style="yellow")
    table.add_column("count", justify="right")
    table.add_column("share", justify="right")
    for reason, count in reasons.most_common(5):
        table.add_row(reason, f"{count}", f"{count/n_total*100:.1f}%")
    console.print(table)


def _report_per_keypoint(seen: np.ndarray, n_total: int, names: list[str]) -> None:
    table = Table(title="Per-keypoint visibility (sorted, top + bottom 8)", show_lines=False)
    table.add_column("idx", justify="right", style="cyan")
    table.add_column("landmark", style="white")
    table.add_column("seen", justify="right")
    rates = (seen / max(n_total, 1) * 100).astype(float)
    order = np.argsort(-rates)
    for i in order[:8]:
        table.add_row(str(i), names[i], f"{rates[i]:.0f}%")
    if len(order) > 16:
        table.add_row("…", "…", "…")
    for i in order[-8:]:
        table.add_row(str(i), names[i], f"{rates[i]:.0f}%")
    console.print(table)

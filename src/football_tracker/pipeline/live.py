"""Live demo: detect (YOLO26) + pitch keypoints (YOLO26-pose) → track (ByteTrack)
→ project (per-frame homography) → analyse (distance/speed) → render
(boxes + trails + minimap).

Run with `aipo demo --source path/to/video.mp4`.

Two-model architecture:
  - `weights`       — fine-tuned YOLO26 detector (player / goalkeeper / referee / ball)
  - `pitch_weights` — fine-tuned YOLO26-pose model (32 pitch keypoints)

Homography mode is chosen automatically:
  1. pitch model + ≥6 visible keypoints  → dynamic (per-frame) homography
  2. else if --homography <file>          → static, loaded from JSON
  3. else                                  → click-calibrate on first frame, save it
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from football_tracker.analytics.distance import DistanceTracker
from football_tracker.pitch.dynamic_homography import DynamicHomographyEstimator
from football_tracker.pitch.homography import Homography, calibrate_interactive
from football_tracker.pitch.minimap import Minimap, _class_colour
from football_tracker.pitch.preprocess import enhance_pitch_lines
from football_tracker.tracking.bytetrack import ByteTracker
from football_tracker.tracking.trail import TrailBuffer

console = Console()

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

# COCO classes used by the pretrained-fallback path (no fine-tuning available yet).
# We re-map COCO 'person' → canonical player(0), 'sports ball' → canonical ball(3).
_COCO_TO_CANONICAL = {0: 0, 32: 3}


def run(
    source: str,
    weights: Path,
    pitch_weights: Path | None = None,
    tracker: str = "bytetrack",
    show_minimap: bool = True,
    show_analytics: bool = True,
    homography_path: Path | None = None,
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    if tracker != "bytetrack":
        raise SystemExit(f"Only bytetrack supported in v0.1 (got: {tracker})")

    # ---------- detector weights ----------
    weights = Path(weights)
    pretrained_fallback = False
    if not weights.exists():
        console.log(
            f"[yellow]No fine-tuned detector at {weights} — "
            "falling back to pretrained yolo26n.pt (COCO person + sports ball).[/]"
        )
        weights = Path("yolo26n.pt")
        pretrained_fallback = True

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from e

    # ---------- video source ----------
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    console.log(f"[cyan]Source[/] {source}  {width}x{height} @ {fps:.1f} fps")

    # ---------- models ----------
    console.log(f"[cyan]Loading detector[/] {weights}")
    detector = YOLO(str(weights))
    predict_classes = list(_COCO_TO_CANONICAL.keys()) if pretrained_fallback else None

    pitch_model = None
    dyn_estimator: DynamicHomographyEstimator | None = None
    if pitch_weights is not None and Path(pitch_weights).exists():
        console.log(f"[cyan]Loading pitch keypoint model[/] {pitch_weights}")
        pitch_model = YOLO(str(pitch_weights))
        dyn_estimator = DynamicHomographyEstimator()
    elif pitch_weights is not None:
        console.log(
            f"[yellow]Pitch weights {pitch_weights} not found — "
            "falling back to static homography.[/]"
        )

    bt = ByteTracker(frame_rate=int(round(fps)))
    trail = TrailBuffer(max_len=int(fps * 2))   # ~2 seconds of trail

    # ---------- homography (static fallback) ----------
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame.")

    static_h: Homography | None = None
    if dyn_estimator is None:
        if homography_path and Path(homography_path).exists():
            static_h = Homography.load(Path(homography_path))
            console.log(f"[cyan]Loaded static homography[/] {homography_path}")
        else:
            console.log("[yellow]No pitch model + no homography file — click 4 pitch corners (TL→TR→BR→BL).[/]")
            static_h = calibrate_interactive(first)
            save_to = Path("configs/homography.json")
            static_h.save(save_to)
            console.log(f"[green]Saved homography[/] → {save_to}")

    # Rewind only if the source is seekable (files); webcams/RTSP just keep streaming.
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_frame_to_process = first if cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0 else None

    # ---------- analytics ----------
    dist = DistanceTracker(fps=fps)

    # ---------- video writer ----------
    minimap = Minimap(width_px=400) if show_minimap else None
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_w = width + (minimap.w if minimap else 0)
        out_h = max(height, minimap.h if minimap else 0)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    # ---------- main loop ----------
    frame_idx = 0
    homo_status = "init"
    try:
        while True:
            if first_frame_to_process is not None:
                frame = first_frame_to_process
                first_frame_to_process = None
                ok = True
            else:
                ok, frame = cap.read()
            if not ok:
                break

            # ----- pitch keypoints → homography (if pitch model present) -----
            # NOTE: keypoint drawing happens AFTER detection so the drawn circles
            # are not fed back into the detector as false positives.
            homo_trusted = False
            pres = None
            if dyn_estimator is not None and pitch_model is not None:
                pres = pitch_model.predict(enhance_pitch_lines(frame), verbose=False, imgsz=960)[0]
                dyn = dyn_estimator.update(pres, image_shape=frame.shape[:2])
                homo = dyn.homography
                homo_trusted = not dyn.used_fallback
                homo_status = (
                    f"dyn ({dyn.n_visible}/32 kpts)"
                    if homo_trusted else f"dyn-fallback ({dyn.n_visible})"
                )
            else:
                homo = static_h
                homo_status = "static"
                homo_trusted = homo is not None

            # ----- detection (on unmodified frame) -----
            yres = detector.predict(
                frame, verbose=False, imgsz=1280, classes=predict_classes
            )[0]
            tracks = bt.update(yres)

            if pretrained_fallback:
                for t in tracks:
                    t.class_id = _COCO_TO_CANONICAL.get(t.class_id, t.class_id)

            # ----- foot points → pitch metres -----
            _PITCH_W, _PITCH_H, _MARGIN = 105.0, 68.0, 3.0
            if tracks and homo is not None:
                foot_pts = np.array(
                    [[(t.xyxy[0] + t.xyxy[2]) / 2, t.xyxy[3]] for t in tracks],
                    dtype=np.float32,
                )
                pitch_pts = homo.project(foot_pts)
                # Only pass through projections that land within the pitch bounds.
                # Keypoints from only one side of the pitch produce a locally-correct
                # but globally-extrapolating homography; out-of-bounds projections
                # must be suppressed before they corrupt analytics and the minimap.
                in_pitch = (
                    (pitch_pts[:, 0] >= -_MARGIN) & (pitch_pts[:, 0] <= _PITCH_W + _MARGIN) &
                    (pitch_pts[:, 1] >= -_MARGIN) & (pitch_pts[:, 1] <= _PITCH_H + _MARGIN)
                )
            else:
                foot_pts = np.empty((0, 2), dtype=np.float32)
                pitch_pts = np.empty((0, 2), dtype=np.float32)
                in_pitch = np.zeros(0, dtype=bool)

            # ----- draw boxes + IDs + trails -----
            for i, (t, foot, pitch) in enumerate(zip(tracks, foot_pts, pitch_pts, strict=False)):
                colour = _class_colour(t.class_id)
                x1, y1, x2, y2 = t.xyxy.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                trail.update(t.track_id, (float(foot[0]), float(foot[1])))
                pts = trail.get(t.track_id).astype(np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, colour, 2)

                label = f"#{t.track_id} {CLASS_NAMES.get(t.class_id, '?')}"
                on_pitch = bool(in_pitch[i]) if in_pitch.size > 0 else False
                if show_analytics and t.class_id != 3 and on_pitch and homo_trusted:
                    total_m, speed = dist.update(t.track_id, pitch)
                    label += f" | {total_m:.0f}m {speed:.1f}km/h"
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
                )

            # HUD: homography status (top-left)
            cv2.putText(
                frame, f"H: {homo_status}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # ----- pitch keypoint overlay (drawn after detection to avoid false positives) -----
            if pres is not None and pres.keypoints is not None and len(pres.keypoints.data) > 0:
                kp_data = pres.keypoints.data.cpu().numpy()[0]
                for kp_i, (kx, ky, kc) in enumerate(kp_data):
                    if kc > 0.5:
                        cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 255), -1)
                        cv2.putText(frame, str(kp_i), (int(kx) + 5, int(ky) - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # ----- minimap composition -----
            if minimap is not None:
                class_ids = np.array([t.class_id for t in tracks], dtype=np.int32)
                mm_pts = pitch_pts[in_pitch] if in_pitch.any() else np.empty((0, 2), np.float32)
                mm_cls = class_ids[in_pitch] if in_pitch.any() else np.empty(0, np.int32)
                mm = minimap.render(mm_pts, mm_cls)
                composed = _stack_side_by_side(frame, mm)
            else:
                composed = frame

            if writer is not None:
                writer.write(composed)
            if show:
                cv2.imshow("football-tracker", composed)
                if cv2.waitKey(1) & 0xFF == 27:    # Esc
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    # ---------- summary ----------
    console.log(f"[green]Processed {frame_idx} frames[/]")
    for tid, stats in dist.summary().items():
        console.log(
            f"  track #{tid}: distance={stats['distance_m']:.1f} m, "
            f"final speed (EMA)={stats['speed_kmh']:.1f} km/h"
        )


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
    return canvas

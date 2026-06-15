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
from football_tracker.device import pick_device
from football_tracker.pitch.dynamic_homography import (
    DynamicHomographyEstimator, STALE_THRESHOLD_FRAMES,
)
from football_tracker.pitch.homography import Homography, calibrate_interactive
from football_tracker.pitch.minimap import Minimap, _class_colour
from football_tracker.pitch.preprocess import enhance_pitch_lines
from football_tracker.reporting import JsonlWriter, dump_json, report_path
from football_tracker.tracking.bytetrack import ByteTracker
from football_tracker.tracking.trail import TrailBuffer

console = Console()

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

# COCO classes used by the pretrained-fallback path (no fine-tuning available yet).
# We re-map COCO 'person' → canonical player(0), 'sports ball' → canonical ball(3).
_COCO_TO_CANONICAL = {0: 0, 32: 3}

# Hot pink for pitch keypoint dots (BGR — OpenCV convention).
_PITCH_KP_COLOUR = (180, 105, 255)


def _faded(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """Halve toward grey — used for untracked detections so they don't dominate the frame."""
    grey = 128
    return tuple(int((c + grey) / 2) for c in colour)


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
    device: str | None = None,
    dump: bool = False,                 # write per-frame JSONL + summary JSON
    report_dir: Path | None = None,
) -> None:
    if tracker != "bytetrack":
        raise SystemExit(f"Only bytetrack supported in v0.1 (got: {tracker})")

    device = pick_device(device)
    console.log(f"[cyan]Device[/] {device}")

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
    trail = TrailBuffer(max_len=int(fps * 3))   # ~3 seconds of fading trail

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

    # ---------- per-keypoint conf EMA (kills the on-screen dot flicker) ----------
    # The pitch model's per-keypoint conf bounces frame-to-frame for boundary
    # keypoints — a kp oscillating around the viz threshold visually flickers
    # on/off. EMA-smooth conf so visibility transitions are gradual.
    kp_conf_ema = np.zeros(32, dtype=np.float32)
    KP_CONF_EMA_ALPHA = 0.4   # 0 = full smoothing, 1 = no smoothing
    KP_VIZ_THRESHOLD = 0.4

    # ---------- per-frame dump ----------
    dump_dir_path: Path | None = None
    jsonl_writer: JsonlWriter | None = None
    homo_fits = 0
    homo_fallbacks = 0
    if dump:
        base = report_path("demo", Path(source).stem, root=report_dir).with_suffix("")
        dump_dir_path = base
        dump_dir_path.mkdir(parents=True, exist_ok=True)
        jsonl_writer = JsonlWriter(dump_dir_path / "frames.jsonl").__enter__()
        console.log(f"[cyan]Per-frame dump[/] → {dump_dir_path}")

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
                # NOTE: ultralytics YOLO-pose models silently reject `augment=True`
                # (warns once per frame and reverts to single-scale). A manual
                # horizontal-flip TTA would mean re-ordering predicted keypoints
                # via flip_idx — non-trivial and unlikely to help while the
                # underlying model has 36% no-detection rate on broadcast.
                # Revisit once the v4 model is trained.
                pres = pitch_model.predict(
                    enhance_pitch_lines(frame),
                    verbose=False, imgsz=960, device=device,
                )[0]
                dyn = dyn_estimator.update(pres, image_shape=frame.shape[:2])
                homo = dyn.homography
                homo_trusted = not dyn.used_fallback
                homo_stale = (
                    dyn.extrapolated
                    and dyn.frames_since_good > STALE_THRESHOLD_FRAMES
                )
                if homo_trusted and dyn.extrapolated:
                    tag = "stale" if homo_stale else "extrap"
                    homo_status = (
                        f"dyn-{tag} {dyn.frames_since_good}f ({dyn.reason or '?'})"
                    )
                elif homo_trusted:
                    homo_status = f"dyn ({dyn.n_visible}vis/{dyn.n_inliers}inl)"
                else:
                    homo_status = f"dyn-fallback ({dyn.n_visible}: {dyn.reason or '?'})"
            else:
                homo = static_h
                homo_status = "static"
                homo_trusted = homo is not None

            # ----- detection (on unmodified frame) -----
            yres = detector.predict(
                frame, verbose=False, imgsz=1280, classes=predict_classes, device=device
            )[0]
            tracks, untracked = bt.update(yres)

            if pretrained_fallback:
                for t in tracks:
                    t.class_id = _COCO_TO_CANONICAL.get(t.class_id, t.class_id)
                for t in untracked:
                    t.class_id = _COCO_TO_CANONICAL.get(t.class_id, t.class_id)

            # ----- foot points → pitch metres (only for trusted H!) -----
            # Untrusted homographies (one-sided / stale / RANSAC-bad) would project
            # players into wildly wrong positions on the minimap. Skip entirely.
            _PITCH_W, _PITCH_H, _MARGIN = 105.0, 68.0, 3.0
            if tracks and homo is not None and homo_trusted:
                foot_pts = np.array(
                    [[(t.xyxy[0] + t.xyxy[2]) / 2, t.xyxy[3]] for t in tracks],
                    dtype=np.float32,
                )
                pitch_pts = homo.project(foot_pts)
                # Only pass through projections that land within the pitch bounds.
                in_pitch = (
                    (pitch_pts[:, 0] >= -_MARGIN) & (pitch_pts[:, 0] <= _PITCH_W + _MARGIN) &
                    (pitch_pts[:, 1] >= -_MARGIN) & (pitch_pts[:, 1] <= _PITCH_H + _MARGIN)
                )
            else:
                foot_pts = np.empty((0, 2), dtype=np.float32)
                pitch_pts = np.empty((0, 2), dtype=np.float32)
                in_pitch = np.zeros(0, dtype=bool)

            # ----- draw untracked detections first (faint, no ID) -----
            # These are detections the tracker couldn't confirm yet; without rendering
            # them the user sees boxes flicker in and out as tracks confirm/decay.
            for u in untracked:
                colour = _class_colour(u.class_id)
                x1, y1, x2, y2 = u.xyxy.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), _faded(colour), 1)

            # ----- draw tracked boxes + IDs + trails -----
            for i, t in enumerate(tracks):
                colour = _class_colour(t.class_id)
                x1, y1, x2, y2 = t.xyxy.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                label = f"#{t.track_id} {CLASS_NAMES.get(t.class_id, '?')}"
                if i < len(foot_pts):
                    foot = foot_pts[i]
                    trail.update(t.track_id, (float(foot[0]), float(foot[1])))
                    pts = trail.get(t.track_id).astype(np.int32)
                    _draw_fading_trail(frame, pts, colour)
                    on_pitch = bool(in_pitch[i]) if in_pitch.size > 0 else False
                    if show_analytics and t.class_id != 3 and on_pitch and homo_trusted:
                        total_m, speed = dist.update(t.track_id, pitch_pts[i])
                        label += f" | {total_m:.0f}m {speed:.1f}km/h"
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
                )

            # HUD: homography status + detection stats (top-left)
            cv2.putText(
                frame, f"H: {homo_status}  det: {len(tracks)}+{len(untracked)} untracked",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # ----- pitch keypoint overlay (drawn after detection to avoid false positives) -----
            # EMA-smooth per-keypoint conf so the on-screen dots don't blink on
            # and off when raw conf oscillates around the viz threshold. The
            # raw position (kx, ky) we still use this frame's value — only the
            # visibility decision is temporally smoothed.
            have_kpts = (
                pres is not None and pres.keypoints is not None
                and len(pres.keypoints.data) > 0
            )
            if have_kpts:
                kp_data = pres.keypoints.data.cpu().numpy()[0]
                raw_conf = kp_data[:, 2]
            else:
                raw_conf = np.zeros(32, dtype=np.float32)
            kp_conf_ema = (
                KP_CONF_EMA_ALPHA * raw_conf
                + (1.0 - KP_CONF_EMA_ALPHA) * kp_conf_ema
            )
            if have_kpts:
                for i, (kx, ky, _) in enumerate(kp_data):
                    if kp_conf_ema[i] > KP_VIZ_THRESHOLD:
                        cv2.circle(frame, (int(kx), int(ky)), 3, _PITCH_KP_COLOUR, -1)

            # ----- minimap composition -----
            if minimap is not None:
                class_ids = np.array([t.class_id for t in tracks], dtype=np.int32)
                mm_pts = pitch_pts[in_pitch] if in_pitch.any() else np.empty((0, 2), np.float32)
                mm_cls = class_ids[in_pitch] if in_pitch.any() else np.empty(0, np.int32)
                mm = minimap.render(mm_pts, mm_cls)
                if homo_stale:
                    # Stale tint: yellow-orange wash so the viewer knows the
                    # geometry hasn't been refreshed in a while.
                    tint = np.full_like(mm, (0, 165, 255))   # BGR amber
                    mm = cv2.addWeighted(mm, 0.7, tint, 0.3, 0)
                composed = _stack_side_by_side(frame, mm)
            else:
                composed = frame

            if writer is not None:
                writer.write(composed)
            if show:
                cv2.imshow("football-tracker", composed)
                if cv2.waitKey(1) & 0xFF == 27:    # Esc
                    break

            # ----- per-frame dump record -----
            if jsonl_writer is not None:
                if homo_trusted:
                    homo_fits += 1
                else:
                    homo_fallbacks += 1
                jsonl_writer.write({
                    "frame": frame_idx,
                    "h_status": homo_status,
                    "h_trusted": bool(homo_trusted),
                    "n_tracked": len(tracks),
                    "n_untracked": len(untracked),
                    "tracks": [
                        {
                            "id": int(t.track_id),
                            "class": int(t.class_id),
                            "conf": float(t.confidence),
                            "bbox": t.xyxy.tolist(),
                            "pitch_xy": (
                                pitch_pts[i].tolist()
                                if i < len(pitch_pts) and bool(in_pitch[i] if in_pitch.size else False)
                                else None
                            ),
                        }
                        for i, t in enumerate(tracks)
                    ],
                })

            frame_idx += 1
    finally:
        cap.release()
        if jsonl_writer is not None:
            jsonl_writer.__exit__(None, None, None)
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

    if dump_dir_path is not None:
        summary = {
            "task": "demo_run",
            "source": str(source),
            "weights": str(weights),
            "pitch_weights": str(pitch_weights) if pitch_weights else None,
            "device": device,
            "fps": float(fps),
            "frame_size": [width, height],
            "frames_processed": frame_idx,
            "homography": {
                "fits": homo_fits,
                "fallbacks": homo_fallbacks,
                "fit_rate": (homo_fits / frame_idx) if frame_idx else 0.0,
            },
            "tracks": {
                str(tid): {
                    "distance_m": float(s["distance_m"]),
                    "final_speed_kmh": float(s["speed_kmh"]),
                }
                for tid, s in dist.summary().items()
            },
        }
        dump_json(summary, dump_dir_path / "summary.json")
        console.log(f"[cyan]Demo summary[/] → {dump_dir_path}/summary.json")


def _draw_fading_trail(
    frame: np.ndarray,
    pts: np.ndarray,
    colour: tuple[int, int, int],
) -> None:
    """Draw a comet-tail trail: newest segment full brightness + 3 px,
    oldest segment dimmed to 25% intensity + 1 px, anti-aliased throughout.
    """
    n = len(pts) - 1
    if n < 1:
        return
    for j in range(n):
        # Progress in [1/n .. 1.0]: 1.0 = newest segment, small = oldest.
        progress = (j + 1) / n
        intensity = 0.25 + 0.75 * progress
        faded = (
            int(colour[0] * intensity),
            int(colour[1] * intensity),
            int(colour[2] * intensity),
        )
        thickness = max(1, int(round(1 + 2 * progress)))
        cv2.line(
            frame, tuple(pts[j]), tuple(pts[j + 1]),
            faded, thickness, cv2.LINE_AA,
        )


def _stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] : left.shape[1] + right.shape[1]] = right
    return canvas

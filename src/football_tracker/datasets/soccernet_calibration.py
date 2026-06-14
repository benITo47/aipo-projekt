"""Convert SoccerNet calibration-2023 dataset to YOLO-pose format.

SoccerNet calibration provides per-frame pitch LINE annotations (normalized
pixel coordinates). We compute line-line intersections to derive our 32
canonical keypoints, then write YOLO-pose labels.

Coordinate systems
------------------
SoccerNet world: origin = pitch centre, +x = right goal, +y = cameras side (bottom),
                 +z = up.
Our world (pitch_keypoints.yaml): origin = bottom-left corner, +x = right, +y = up.

Transform: sn_x = our_x - L/2,  sn_y = W/2 - our_y   (L=105, W=68)
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.progress import track

from football_tracker.pitch.keypoints import load as load_scheme

console = Console()

# ---------------------------------------------------------------------------
# Mapping: keypoint_index → (line_class_A, line_class_B) that intersect there.
# None entries use special extraction (goalpost foot, circle-line, etc.).
# ---------------------------------------------------------------------------
_KPT_LINE_PAIRS: list[tuple[str, str] | None] = [
    # 0  TL
    ("Side line top", "Side line left"),
    # 1  TR
    ("Side line top", "Side line right"),
    # 2  BR
    ("Side line bottom", "Side line right"),
    # 3  BL
    ("Side line bottom", "Side line left"),
    # 4  HALFWAY_TOP
    ("Middle line", "Side line top"),
    # 5  HALFWAY_BOT
    ("Middle line", "Side line bottom"),
    # 6  PEN_L_OUT_TOP
    ("Big rect. left top", "Side line left"),
    # 7  PEN_L_OUT_BOT
    ("Big rect. left bottom", "Side line left"),
    # 8  PEN_L_IN_TOP
    ("Big rect. left top", "Big rect. left main"),
    # 9  PEN_L_IN_BOT
    ("Big rect. left bottom", "Big rect. left main"),
    # 10 PEN_R_IN_TOP
    ("Big rect. right top", "Big rect. right main"),
    # 11 PEN_R_IN_BOT
    ("Big rect. right bottom", "Big rect. right main"),
    # 12 PEN_R_OUT_TOP
    ("Big rect. right top", "Side line right"),
    # 13 PEN_R_OUT_BOT
    ("Big rect. right bottom", "Side line right"),
    # 14 GOAL_L_OUT_TOP
    ("Small rect. left top", "Side line left"),
    # 15 GOAL_L_OUT_BOT
    ("Small rect. left bottom", "Side line left"),
    # 16 GOAL_L_IN_TOP
    ("Small rect. left top", "Small rect. left main"),
    # 17 GOAL_L_IN_BOT
    ("Small rect. left bottom", "Small rect. left main"),
    # 18 GOAL_R_IN_TOP
    ("Small rect. right top", "Small rect. right main"),
    # 19 GOAL_R_IN_BOT
    ("Small rect. right bottom", "Small rect. right main"),
    # 20 GOAL_R_OUT_TOP
    ("Small rect. right top", "Side line right"),
    # 21 GOAL_R_OUT_BOT
    ("Small rect. right bottom", "Side line right"),
    # 22 GOALPOST_L_TOP  – foot of Goal left post right (far post)
    ("Goal left post right", None),
    # 23 GOALPOST_L_BOT  – foot of Goal left post left  (near post)
    ("Goal left post left", None),
    # 24 GOALPOST_R_TOP  – foot of Goal right post left (far post)
    ("Goal right post left", None),
    # 25 GOALPOST_R_BOT  – foot of Goal right post right (near post)
    ("Goal right post right", None),
    # 26 CENTRE_CIRC_TOP – Circle central ∩ Middle line (top, lower y_norm)
    ("Circle central", "Middle line", "top"),   # type: ignore[list-item]
    # 27 CENTRE_CIRC_BOT – Circle central ∩ Middle line (bot, higher y_norm)
    ("Circle central", "Middle line", "bottom"),  # type: ignore[list-item]
    # 28-31: centre/penalty spots and arc apex — not in SN line annotations
    None,  # 28 CENTRE_SPOT
    None,  # 29 PEN_SPOT_L
    None,  # 30 PEN_SPOT_R
    None,  # 31 PEN_ARC_APEX
]


# ---------------------------------------------------------------------------
# Geometry helpers (normalised coords 0-1)
# ---------------------------------------------------------------------------

def _fit_line_hom(pts: list[dict]) -> np.ndarray | None:
    """Fit a homogeneous line [a,b,c] through normalised annotation points."""
    if len(pts) < 2:
        return None
    coords = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float64)
    # Use first and last point for robustness (Roboflow-style ordering)
    p1, p2 = coords[0], coords[-1]
    dx, dy = p2 - p1
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    # Line: dy*x - dx*y + (dx*p1y - dy*p1x) = 0
    a, b, c = dy, -dx, dx * p1[1] - dy * p1[0]
    n = np.sqrt(a**2 + b**2)
    return np.array([a / n, b / n, c / n])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> tuple[float, float] | None:
    """Intersection of two homogeneous lines."""
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-10:
        return None
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def _goalpost_foot(pts: list[dict]) -> tuple[float, float] | None:
    """Return the bottom-most point (highest y_norm) of a goalpost annotation."""
    if not pts:
        return None
    # In image coords y increases downward; foot of post = max y
    foot = max(pts, key=lambda p: p["y"])
    return float(foot["x"]), float(foot["y"])


def _circle_midline_intersections(
    circle_pts: list[dict],
    midline_pts: list[dict],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """
    Return the two intersections of the centre circle with the halfway line.
    Returns (top_pt, bot_pt) where top has smaller y_norm (higher in image).
    """
    if len(circle_pts) < 3 or len(midline_pts) < 2:
        return None, None
    # Fit ellipse to circle points (projected circle → ellipse in image)
    coords = np.array([[p["x"], p["y"]] for p in circle_pts], dtype=np.float32)
    if len(coords) < 5:
        return None, None
    # Scale to pixel-like space for cv2.fitEllipse
    scale = 1000.0
    coords_px = (coords * scale).astype(np.int32)
    try:
        ellipse = cv2.fitEllipse(coords_px)
    except Exception:
        return None, None
    (cx, cy), (ma, mb), angle = ellipse
    # Halfway line in normalised coords
    ml = _fit_line_hom(midline_pts)
    if ml is None:
        return None, None
    # The halfway line is (nearly) vertical in image or at some angle.
    # Find the x position where the midline is at the circle's centre y-range.
    # Approximate: the two intersections are at the circle's extreme y along the midline.
    # We know geometrically they're directly above/below the centre in world coords.
    # Use the fitted midline: x = (-c - b*y) / a  or  y = (-c - a*x) / b
    cx_n, cy_n = cx / scale, cy / scale
    r_avg = ((ma + mb) / 2.0) / scale  # average radius in normalised space
    # Intersection of a line with a circle requires solving a quadratic.
    # Line: a*x + b*y + c = 0  →  parametric x = x0 + b*t, y = y0 - a*t
    a, b, c = ml
    # Centre of fitted ellipse (approximate circle centre)
    # t at closest approach from line to centre:
    x0, y0 = cx_n, cy_n
    # perp dist from (x0,y0) to line
    dist = abs(a * x0 + b * y0 + c)
    if dist > r_avg:
        return None, None
    t0 = -(a * x0 + b * y0 + c)
    dt = np.sqrt(max(r_avg**2 - dist**2, 0))
    t1, t2 = t0 - dt, t0 + dt
    pt1 = (x0 + b * t1, y0 - a * t1)
    pt2 = (x0 + b * t2, y0 - a * t2)
    # Sort by y (top = smaller y in image)
    if pt1[1] > pt2[1]:
        pt1, pt2 = pt2, pt1
    return pt1, pt2


# ---------------------------------------------------------------------------
# Per-frame keypoint extraction
# ---------------------------------------------------------------------------

# Every SoccerNet class name we look up. Used both to normalise annotation
# keys (strip whitespace, case-fold) and to warn on first frame if a release
# silently renames a class.
_EXPECTED_CLASSES: set[str] = {
    "Side line top", "Side line left", "Side line right", "Side line bottom",
    "Middle line",
    "Big rect. left top", "Big rect. left bottom", "Big rect. left main",
    "Big rect. right top", "Big rect. right bottom", "Big rect. right main",
    "Small rect. left top", "Small rect. left bottom", "Small rect. left main",
    "Small rect. right top", "Small rect. right bottom", "Small rect. right main",
    "Goal left post right", "Goal left post left",
    "Goal right post left", "Goal right post right",
    "Circle central",
}


def _norm(s: str) -> str:
    """Canonical form for SoccerNet class name lookup: strip + lowercase + collapse spaces."""
    return " ".join(s.strip().lower().split())


_EXPECTED_NORM: dict[str, str] = {_norm(c): c for c in _EXPECTED_CLASSES}


def _normalise_ann(ann: dict) -> dict:
    """Strip / case-fold annotation keys so lookups are robust to whitespace
    or case drift in future SoccerNet releases (e.g. the historical
    'Goal left post left ' trailing space)."""
    out: dict[str, list] = {}
    for k, v in ann.items():
        if not isinstance(k, str):
            continue
        canon = _EXPECTED_NORM.get(_norm(k), k)
        out[canon] = v
    return out


def _extract_keypoints(
    ann: dict,
) -> tuple[list[float], list[float], list[int]]:
    """
    Given SoccerNet annotation dict, return (xs, ys, visibilities) for our 32 kpts.
    All coordinates are normalised [0,1].
    """
    xs = [0.0] * 32
    ys = [0.0] * 32
    vis = [0] * 32

    # Pre-compute lines we'll need
    _lines: dict[str, np.ndarray | None] = {}

    def _get_line(name: str) -> np.ndarray | None:
        if name not in _lines:
            pts = ann.get(name, [])
            _lines[name] = _fit_line_hom(pts) if pts else None
        return _lines[name]

    for kpt_idx, spec in enumerate(_KPT_LINE_PAIRS):
        if spec is None:
            continue

        # Goalpost foot (single-class, no second line)
        if isinstance(spec, tuple) and len(spec) == 2 and spec[1] is None:
            cls = spec[0]
            pts = ann.get(cls, [])
            result = _goalpost_foot(pts)
            if result is not None:
                x, y = result
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xs[kpt_idx], ys[kpt_idx], vis[kpt_idx] = x, y, 2

        # Circle-midline intersection (3-tuple spec)
        elif isinstance(spec, tuple) and len(spec) == 3:
            circ_pts = ann.get(spec[0], [])
            mid_pts = ann.get(spec[1], [])
            top_pt, bot_pt = _circle_midline_intersections(circ_pts, mid_pts)
            which = spec[2]
            pt = top_pt if which == "top" else bot_pt
            if pt is not None:
                x, y = pt
                if 0 <= x <= 1 and 0 <= y <= 1:
                    xs[kpt_idx], ys[kpt_idx], vis[kpt_idx] = x, y, 2

        # Standard line-line intersection
        else:
            cls_a, cls_b = spec[0], spec[1]
            la = _get_line(cls_a)
            lb = _get_line(cls_b)
            if la is not None and lb is not None:
                result = _intersect(la, lb)
                if result is not None:
                    x, y = result
                    if 0 <= x <= 1 and 0 <= y <= 1:
                        xs[kpt_idx], ys[kpt_idx], vis[kpt_idx] = x, y, 2

    return xs, ys, vis


# ---------------------------------------------------------------------------
# YOLO-pose label writer
# ---------------------------------------------------------------------------

def _write_yolo_label(
    label_path: Path,
    img_w: int,
    img_h: int,
    xs: list[float],
    ys: list[float],
    vis: list[int],
) -> bool:
    """Write a single YOLO-pose label file. Returns False if too few keypoints visible."""
    n_vis = sum(vis)
    # 4 is the geometric minimum to fit a homography. Keep every frame that
    # carries a usable signal — broadcasts often only show one goal area, and
    # the v4 augmentation (crop_fraction=0.7) trains on partial pitches anyway.
    if n_vis < 4:
        return False

    # Bounding box: tight around visible keypoints
    vis_x = [xs[i] for i in range(32) if vis[i]]
    vis_y = [ys[i] for i in range(32) if vis[i]]
    bx = (min(vis_x) + max(vis_x)) / 2
    by = (min(vis_y) + max(vis_y)) / 2
    bw = max(vis_x) - min(vis_x)
    bh = max(vis_y) - min(vis_y)
    # Small padding
    bw = min(bw * 1.05, 1.0)
    bh = min(bh * 1.05, 1.0)

    kpt_str = " ".join(
        f"{xs[i]:.6f} {ys[i]:.6f} {vis[i]}" for i in range(32)
    )
    label_path.write_text(
        f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {kpt_str}\n"
    )
    return True


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def convert(
    soccernet_dir: Path,
    out_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> int:
    """
    Convert one SoccerNet calibration-2023 split to YOLO-pose format.

    Returns the number of converted samples.
    """
    src = soccernet_dir / split
    if not src.exists():
        raise SystemExit(f"SoccerNet split not found: {src}")

    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    frames = sorted(src.glob("*.jpg"))
    if max_samples:
        frames = frames[:max_samples]

    converted = 0
    skipped = 0
    name_check_done = False
    for img_path in track(frames, description=f"[cyan]Converting {split}[/]"):
        ann_path = img_path.with_suffix(".json")
        if not ann_path.exists():
            skipped += 1
            continue

        with ann_path.open() as f:
            ann = json.load(f)
        if not ann:
            skipped += 1
            continue
        ann = _normalise_ann(ann)
        if not name_check_done:
            seen = {_norm(k) for k in ann}
            missing = [c for c in _EXPECTED_CLASSES if _norm(c) not in seen]
            if missing:
                console.log(
                    f"[yellow]SoccerNet class names not seen in first frame:[/] "
                    f"{', '.join(missing[:6])}{'…' if len(missing) > 6 else ''} "
                    "(may be fine if the frame just doesn't show those lines)"
                )
            name_check_done = True

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        xs, ys, vis = _extract_keypoints(ann)
        stem = img_path.stem
        label_path = lbl_out / f"{stem}.txt"

        if not _write_yolo_label(label_path, w, h, xs, ys, vis):
            skipped += 1
            continue

        shutil.copy2(img_path, img_out / img_path.name)
        converted += 1

    console.log(
        f"[green]{split}[/] → converted={converted}, skipped={skipped}"
    )
    return converted


def unzip_and_convert(
    zip_path: Path,
    extract_to: Path,
    out_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> int:
    """Unzip SoccerNet split if needed, then convert."""
    console.log(f"[cyan]Unzipping[/] {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)
    return convert(extract_to, out_dir, split=split, max_samples=max_samples)

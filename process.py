#!/usr/bin/env python3
"""End-to-end football video processor.

Detector (YOLO26) → ByteTrack → pitch-keypoint homography (YOLO26-pose) →
annotated MP4 with player boxes, IDs, fading trails, distance/speed labels,
and a top-down minimap.

Accepts any video format OpenCV / FFmpeg can decode (mp4, mov, mkv, avi,
webm, m4v, ts, flv, wmv, mpg).

Usage:
    python process.py match.mov                       # → match_processed.mp4
    python process.py match.mp4 -o annotated.mp4
    python process.py clip.mkv --device cuda          # use NVIDIA GPU
    python process.py clip.mp4 --no-minimap           # skip top-down panel
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
from rich.console import Console
from rich.table import Table

console = Console()


# Video extensions we accept on input. cv2.VideoCapture handles all of these
# via FFmpeg; this list is purely for error messages.
_SUPPORTED_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
    ".ts", ".flv", ".wmv", ".mpg", ".mpeg",
}


def _probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def _human_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f} s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)} m {sec:.0f} s"
    h, m = divmod(m, 60)
    return f"{int(h)} h {int(m)} m"


def _h_breakdown(frames_jsonl: Path) -> dict[str, int]:
    """Bucket every frame by homography status (fresh / extrap / stale / fallback).
    The summary.json conflates fresh + extrapolated under 'fits' which obscures
    how much of the minimap is being driven by a fresh model fit vs the EMA
    carrying us through brief detection gaps."""
    counts = {"fresh": 0, "extrap": 0, "stale": 0, "fallback": 0}
    if not frames_jsonl.exists():
        return counts
    with frames_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            s = r.get("h_status", "")
            if not r.get("h_trusted", False):
                counts["fallback"] += 1
            elif "extrap" in s:
                counts["extrap"] += 1
            elif "stale" in s:
                counts["stale"] += 1
            else:
                counts["fresh"] += 1
    return counts


def _print_stats(summary: dict, wallclock: float, h_break: dict[str, int]) -> None:
    n_frames = summary["frames_processed"]
    proc_fps = n_frames / wallclock if wallclock else 0.0
    input_dur = n_frames / summary["fps"] if summary["fps"] else 0.0
    realtime_factor = input_dur / wallclock if wallclock else 0.0
    n_tracks = len(summary["tracks"])
    total_dist = sum(t["distance_m"] for t in summary["tracks"].values())
    max_speed = max(
        (t["final_speed_kmh"] for t in summary["tracks"].values()), default=0.0
    )

    def _pct(n: int) -> str:
        return f"{n:>5,} ({n / n_frames:>5.1%})" if n_frames else "—"

    t = Table(show_header=False, title="[bold]Run summary[/]", title_style="cyan")
    t.add_column("metric", style="dim")
    t.add_column("value", justify="right")
    t.add_row("input frames", f"{n_frames:,}")
    t.add_row("input duration", _human_time(input_dur))
    t.add_row("wall-clock", _human_time(wallclock))
    t.add_row("throughput", f"{proc_fps:.1f} fps")
    t.add_row("real-time factor", f"{realtime_factor:.2f}×")
    t.add_section()
    t.add_row("H fresh fits", _pct(h_break["fresh"]))
    t.add_row("H extrapolated", _pct(h_break["extrap"]))
    t.add_row("H stale (>12s)", _pct(h_break["stale"]))
    t.add_row("H fallback (no minimap)", _pct(h_break["fallback"]))
    t.add_section()
    t.add_row("unique player IDs", f"{n_tracks}")
    t.add_row("total distance", f"{total_dist:,.0f} m")
    t.add_row("peak final speed", f"{max_speed:.1f} km/h")
    console.print(t)


def main() -> int:
    p = argparse.ArgumentParser(
        prog="process.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input", type=Path,
                   help="Source video (any format FFmpeg understands).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output MP4 path. Default: <input_stem>_processed.mp4.")
    p.add_argument("--weights", type=Path, default=Path("models/checkpoints/best.pt"),
                   help="Player detector weights (.pt).")
    p.add_argument("--pitch-weights", type=Path, default=Path("models/checkpoints/pitch.pt"),
                   help="Pitch keypoint weights (.pt).")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | mps | cuda | cuda:N | 0 | 0,1,2  (default: auto)")
    p.add_argument("--no-minimap", action="store_true",
                   help="Skip the top-down minimap panel.")
    p.add_argument("--no-analytics", action="store_true",
                   help="Skip per-player distance / speed overlay.")
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.exists():
        raise SystemExit(f"Input video not found: {inp}")
    if inp.is_dir():
        raise SystemExit(f"Input is a directory, not a video: {inp}")
    if inp.suffix.lower() not in _SUPPORTED_EXTS:
        console.print(
            f"[yellow]Warning[/]: unfamiliar extension {inp.suffix!r}; "
            "trying anyway via FFmpeg."
        )
    out = (args.output or inp.parent / f"{inp.stem}_processed.mp4").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    total_t0 = time.perf_counter()
    console.rule("[bold cyan]process.py[/]")

    # ----- Stage 1: probe input + announce paths -----
    console.rule("[bold]Stage 1/3 · setup[/]", style="cyan", align="left")
    t0 = time.perf_counter()
    info = _probe(inp)
    console.print(f"  [bold]input  [/]  {inp}")
    console.print(
        f"  [bold]format [/]  {info['width']}×{info['height']} @ {info['fps']:.1f} fps  "
        f"·  {info['frames']:,} frames  ·  {_human_time(info['duration_s'])}"
    )
    console.print(f"  [bold]output [/]  {out}")
    console.print(f"  [bold]device [/]  {args.device}")
    if not args.weights.exists():
        console.print(
            f"  [yellow]note  [/]  detector weights {args.weights} missing — "
            "stock yolo26n.pt fallback (COCO classes)"
        )
    pitch_path: Path | None = args.pitch_weights
    if not args.pitch_weights.exists():
        console.print(
            f"  [yellow]note  [/]  pitch weights {args.pitch_weights} missing — "
            "homography falls back to click-calibration"
        )
        pitch_path = None
    t_setup = time.perf_counter() - t0
    console.print(f"  [dim]setup took {_human_time(t_setup)}[/]")

    # ----- Stage 2: hand off to the pipeline -----
    console.rule("[bold]Stage 2/3 · processing[/]", style="cyan", align="left")
    console.print(
        "  Detector + pitch model load on first frame; expect a startup delay\n"
        "  before throughput stabilises. Ctrl-C aborts safely."
    )
    from football_tracker.pipeline import live

    t0 = time.perf_counter()
    try:
        live.run(
            source=str(inp),
            weights=args.weights,
            pitch_weights=pitch_path,
            tracker="bytetrack",
            show_minimap=not args.no_minimap,
            show_analytics=not args.no_analytics,
            output_path=out,
            show=False,
            device=args.device,
            dump=True,
            report_dir=None,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/] — partial output may exist.")
        return 130
    t_proc = time.perf_counter() - t0
    console.print(f"  [dim]processing took {_human_time(t_proc)}[/]")

    # ----- Stage 3: load the dumped summary + format -----
    console.rule("[bold]Stage 3/3 · summary[/]", style="cyan", align="left")
    report_root = Path("outputs/reports/demo")
    summary_path: Path | None = None
    if report_root.exists():
        candidates = sorted(
            report_root.glob(f"{inp.stem}-*"), key=lambda p: p.stat().st_mtime
        )
        if candidates:
            cand = candidates[-1] / "summary.json"
            if cand.exists():
                summary_path = cand

    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        h_break = _h_breakdown(summary_path.parent / "frames.jsonl")
        _print_stats(summary, t_proc, h_break)

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(Path.cwd()))
            except ValueError:
                return str(p)
        console.print(f"\n  [bold green]video   →[/]  {_rel(out)}")
        console.print(f"  [bold]summary →[/]  {_rel(summary_path)}")
        console.print(f"  [bold]frames  →[/]  {_rel(summary_path.parent / 'frames.jsonl')}")
    else:
        console.print(
            "  [yellow]No summary JSON found — pipeline may have been interrupted.[/]"
        )
        console.print(f"  [bold]annotated video →[/]  {out}")

    console.rule(
        f"[bold green]done in {_human_time(time.perf_counter() - total_t0)}[/]",
        style="green",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

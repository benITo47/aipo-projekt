#!/usr/bin/env python3
"""Single entry point for evaluating trained models.

Examples:
    python eval.py detector --weights models/checkpoints/best.pt
    python eval.py pitch    --weights models/checkpoints/pitch.pt
    python eval.py homography --source clip.mp4   # per-frame stability stats
    python eval.py all                            # detector + pitch (skips homography)

`--device` accepts auto | cpu | mps | cuda | cuda:N | 0 | 0,1,2 (default: auto).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ---------- handlers ----------

def cmd_detector(args: argparse.Namespace) -> None:
    from football_tracker.training import eval_detector
    eval_detector.run(
        weights=args.weights, data=args.data, imgsz=args.imgsz,
        device=args.device, report_dir=args.report_dir,
    )


def cmd_pitch(args: argparse.Namespace) -> None:
    from football_tracker.training import eval_pitch
    eval_pitch.run(
        weights=args.weights, data=args.data, imgsz=args.imgsz,
        device=args.device, report_dir=args.report_dir,
    )


def cmd_homography(args: argparse.Namespace) -> None:
    from football_tracker.training import eval_homography
    eval_homography.run(
        source=args.source,
        pitch_weights=args.weights,
        max_frames=args.max_frames,
        stride=args.stride,
        imgsz=args.imgsz,
        device=args.device,
        report_dir=args.report_dir,
    )


def cmd_all(args: argparse.Namespace) -> None:
    from football_tracker.training import eval_detector, eval_pitch
    print("\n=== Detector ===")
    eval_detector.run(
        weights=args.detector_weights, data=args.detector_data,
        imgsz=args.detector_imgsz, device=args.device, report_dir=args.report_dir,
    )
    print("\n=== Pitch model ===")
    eval_pitch.run(
        weights=args.pitch_weights, data=args.pitch_data,
        imgsz=args.pitch_imgsz, device=args.device, report_dir=args.report_dir,
    )


# ---------- argparse ----------

def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device", default="auto",
        help="auto | cpu | mps | cuda | cuda:N | 0 | 0,1,2  (default: auto)",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=None,
        help="Where to write the JSON report (default: outputs/reports/).",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detector", help="mAP / PR on detector val split.")
    d.add_argument("--weights", type=Path, required=True)
    d.add_argument("--data", type=Path, default=Path("configs/combined.yaml"))
    d.add_argument("--imgsz", type=int, default=1280)
    _add_common(d)
    d.set_defaults(func=cmd_detector)

    pt = sub.add_parser("pitch", help="Box + keypoint mAP for pitch model.")
    pt.add_argument("--weights", type=Path, required=True)
    pt.add_argument(
        "--data", type=Path,
        default=Path("data/processed/combined_pitch_gs/data.yaml"),
        help="Combined SoccerNet+Roboflow val set by default; pass configs/pitch.yaml for Roboflow only.",
    )
    pt.add_argument("--imgsz", type=int, default=960)   # matches training res
    _add_common(pt)
    pt.set_defaults(func=cmd_pitch)

    h = sub.add_parser("homography", help="Per-frame homography stability on a real video.")
    h.add_argument("--source", required=True, help="Video file or stream URL.")
    h.add_argument("--weights", type=Path, default=Path("models/checkpoints/pitch.pt"))
    h.add_argument("--max-frames", type=int, default=None)
    h.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    h.add_argument("--imgsz", type=int, default=960)
    _add_common(h)
    h.set_defaults(func=cmd_homography)

    al = sub.add_parser("all", help="Detector + pitch eval (skips homography — needs a video).")
    al.add_argument("--detector-weights", type=Path, default=Path("models/checkpoints/best.pt"))
    al.add_argument("--pitch-weights", type=Path, default=Path("models/checkpoints/pitch.pt"))
    al.add_argument("--detector-data", type=Path, default=Path("configs/combined.yaml"))
    al.add_argument(
        "--pitch-data", type=Path,
        default=Path("data/processed/combined_pitch_gs/data.yaml"),
    )
    al.add_argument("--detector-imgsz", type=int, default=1280)
    al.add_argument("--pitch-imgsz", type=int, default=960)
    _add_common(al)
    al.set_defaults(func=cmd_all)

    return p


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

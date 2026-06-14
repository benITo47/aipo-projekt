#!/usr/bin/env python3
"""Single entry point for training both models.

Examples:
    python train.py detector                              # fine-tune YOLO26 detector
    python train.py pitch                                 # fine-tune YOLO26-pose pitch model
    python train.py all                                   # both, sequentially
    python train.py export --weights models/checkpoints/best.pt   # .pt → ONNX
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ---------- handlers ----------

def cmd_detector(args: argparse.Namespace) -> None:
    from football_tracker.training import train_detector
    train_detector.run(config_path=args.config, report_dir=args.report_dir)


def cmd_pitch(args: argparse.Namespace) -> None:
    from football_tracker.training import train_pitch
    train_pitch.run(config_path=args.config, report_dir=args.report_dir)


def cmd_all(args: argparse.Namespace) -> None:
    from football_tracker.training import train_detector, train_pitch
    train_detector.run(config_path=args.detector_config, report_dir=args.report_dir)
    train_pitch.run(config_path=args.pitch_config, report_dir=args.report_dir)


def cmd_export(args: argparse.Namespace) -> None:
    from football_tracker.training import export
    export.run(weights=args.weights, imgsz=args.imgsz, half=args.half, dynamic=args.dynamic)


# ---------- argparse ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detector", help="Fine-tune YOLO26 player detector.")
    d.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    d.add_argument("--report-dir", type=Path, default=None,
                   help="Where to write the JSON report (default: outputs/reports/).")
    d.set_defaults(func=cmd_detector)

    pt = sub.add_parser("pitch", help="Fine-tune YOLO26-pose pitch keypoint model.")
    pt.add_argument("--config", type=Path, default=Path("configs/training_pitch.yaml"))
    pt.add_argument("--report-dir", type=Path, default=None,
                    help="Where to write the JSON report (default: outputs/reports/).")
    pt.set_defaults(func=cmd_pitch)

    al = sub.add_parser("all", help="Train detector + pitch model sequentially.")
    al.add_argument("--detector-config", type=Path, default=Path("configs/training.yaml"))
    al.add_argument("--pitch-config", type=Path, default=Path("configs/training_pitch.yaml"))
    al.add_argument("--report-dir", type=Path, default=None)
    al.set_defaults(func=cmd_all)

    ex = sub.add_parser("export", help="Export a trained .pt → ONNX.")
    ex.add_argument("--weights", type=Path, default=Path("models/checkpoints/best.pt"))
    ex.add_argument("--imgsz", type=int, default=1280)
    ex.add_argument("--half", action="store_true", help="FP16 (CUDA only).")
    ex.add_argument("--no-dynamic", dest="dynamic", action="store_false")
    ex.set_defaults(func=cmd_export, dynamic=True)

    return p


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

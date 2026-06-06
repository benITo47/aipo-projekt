#!/usr/bin/env python3
"""The single point of entry that combines everything.

YOLO26 detector → ByteTrack tracker → YOLO26-pose pitch model → per-frame
homography → top-down minimap + distance/speed HUD.

Examples:
    python demo.py --source clip.mp4
    python demo.py --source 0                          # webcam
    python demo.py --source clip.mp4 --output out.mp4 --no-show
    python demo.py --source clip.mp4 --weights custom.pt --pitch-weights pitch.pt
    python demo.py --source clip.mp4 --no-pitch        # force static click homography
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> int:
    p = argparse.ArgumentParser(
        prog="demo.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", required=True,
                   help="Video file path, RTSP URL, or webcam index (e.g. 0).")
    p.add_argument("--weights", type=Path, default=Path("models/checkpoints/best.pt"),
                   help="Player detector weights (.pt). Missing → pretrained yolo26n COCO fallback.")
    p.add_argument("--pitch-weights", type=Path, default=Path("models/checkpoints/pitch.pt"),
                   help="Pitch keypoint model (.pt). Missing → static click homography.")
    p.add_argument("--no-pitch", action="store_true",
                   help="Force static homography even if pitch weights exist.")
    p.add_argument("--homography", type=Path, default=None,
                   help="Static homography JSON (used only when pitch model absent / disabled).")
    p.add_argument("--no-minimap", dest="minimap", action="store_false")
    p.add_argument("--no-analytics", dest="analytics", action="store_false")
    p.add_argument("--output", type=Path, default=None,
                   help="Write the annotated video here.")
    p.add_argument("--no-show", dest="show", action="store_false",
                   help="Run headless (useful for batch processing).")
    p.set_defaults(minimap=True, analytics=True, show=True)

    args = p.parse_args()

    from football_tracker.pipeline import live
    live.run(
        source=args.source,
        weights=args.weights,
        pitch_weights=None if args.no_pitch else args.pitch_weights,
        tracker="bytetrack",
        show_minimap=args.minimap,
        show_analytics=args.analytics,
        homography_path=args.homography,
        output_path=args.output,
        show=args.show,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Train the YOLO26s-pose pitch model from scratch on the augmented dataset.

Dataset = original combined frames + photometric copies + 11 strategic
partial-pitch crops per source. Produces a model that handles both full-pitch
tactical-cam views and the half / quarter / goal-area broadcast cuts.

Base: ``yolo26s-pose.pt`` (COCO 17-kpt human pose). The keypoint head is
re-initialised for 32 pitch keypoints, so this run learns the head from
scratch — needs lr0=0.001, not the fine-tune lr0=5e-5.

Output: ``models/checkpoints/pitch.pt`` (overwrites if present — the new
model strictly improves on the old).

Usage::

    pip install -e '.[training]'
    python dataset.py augment-pitch                # writes combined_pitch_gs_aug/
    python train_pitch_partials.py                 # ~6-12 h on RTX 3090
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> int:
    p = argparse.ArgumentParser(
        prog="train_pitch_partials.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("configs/training_pitch_partials.yaml"),
    )
    p.add_argument("--report-dir", type=Path, default=None)
    args = p.parse_args()

    dataset_yaml = Path("data/processed/combined_pitch_gs_aug/data.yaml")
    if not dataset_yaml.exists():
        raise SystemExit(
            f"Augmented dataset missing: {dataset_yaml}\n"
            "Generate it first:\n"
            "  python dataset.py augment-pitch"
        )

    from football_tracker.training import train_pitch
    train_pitch.run(args.config, report_dir=args.report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

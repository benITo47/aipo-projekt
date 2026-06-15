#!/usr/bin/env python3
"""Fine-tune the existing pitch model on partial-pitch crops only.

Starts from ``models/checkpoints/pitch.pt`` and trains exclusively on the
11 strategic partial-pitch crops per source (halves, quarters, centre zoom,
goal close-ups). Teaches the model to recognise partial-pitch broadcasts
without forgetting its full-pitch knowledge — short run with a very low
learning rate (5e-5).

Output: ``models/checkpoints/pitch_finetuned.pt`` — does NOT overwrite the
base ``pitch.pt`` by default. Swap it in by hand if eval improves.

Usage::

    pip install -e '.[training]'
    python dataset.py augment-pitch --partials-only \\
           --out data/processed/combined_pitch_gs_partials_only
    python train_pitch_finetune.py                 # ~1-3 h on RTX 3090
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> int:
    p = argparse.ArgumentParser(
        prog="train_pitch_finetune.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        default=Path("configs/training_pitch_partials_finetune.yaml"),
    )
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument(
        "--output", type=Path,
        default=Path("models/checkpoints/pitch_finetuned.pt"),
        help="Output weights path. Default keeps the base pitch.pt intact.",
    )
    args = p.parse_args()

    base_pt = Path("models/checkpoints/pitch.pt")
    if not base_pt.exists():
        raise SystemExit(
            f"Base model missing: {base_pt}\n"
            "Train it first with:\n"
            "  python train_pitch_partials.py"
        )
    dataset_yaml = Path("data/processed/combined_pitch_gs_partials_only/data.yaml")
    if not dataset_yaml.exists():
        raise SystemExit(
            f"Partials-only dataset missing: {dataset_yaml}\n"
            "Generate it first:\n"
            "  python dataset.py augment-pitch --partials-only \\\n"
            "       --out data/processed/combined_pitch_gs_partials_only"
        )

    from football_tracker.training import train_pitch
    train_pitch.run(args.config, report_dir=args.report_dir, output_checkpoint=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Single entry point for all dataset operations.

Examples:
    python dataset.py soccernet --split tracking          # needs $SOCCERNET_PASSWORD
    python dataset.py roboflow                            # needs $ROBOFLOW_API_KEY
    python dataset.py pitch                               # Roboflow pitch keypoints
    python dataset.py youtube https://youtu.be/<id>
    python dataset.py merge                               # players: SoccerNet + Roboflow
    python dataset.py all                                 # roboflow + pitch + merge

All commands write under data/raw/ and emit a dataset YAML under configs/.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running without `pip install -e .` by adding src/ to the path.
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ---------- handlers ----------

def cmd_soccernet(args: argparse.Namespace) -> None:
    from football_tracker.datasets import soccernet
    soccernet.download(split=args.split, out_dir=args.out, password=args.password)
    if args.convert:
        soccernet.convert_to_yolo(out_dir=args.out)


def cmd_roboflow(args: argparse.Namespace) -> None:
    from football_tracker.datasets import roboflow
    roboflow.download(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        out_dir=args.out,
        api_key=args.api_key,
    )


def cmd_pitch(args: argparse.Namespace) -> None:
    from football_tracker.datasets import pitch_roboflow
    pitch_roboflow.download(
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        out_dir=args.out,
        api_key=args.api_key,
    )


def cmd_youtube(args: argparse.Namespace) -> None:
    from football_tracker.datasets import youtube
    youtube.download(url=args.url, out_dir=args.out, resolution=args.resolution)


def cmd_merge(args: argparse.Namespace) -> None:
    from football_tracker.datasets import merge
    # Smart auto-detect: if a path wasn't given, use it iff the default exists.
    sn = args.soccernet or (Path("data/raw/soccernet") if Path("data/raw/soccernet").exists() else None)
    rf = args.roboflow or (Path("data/raw/roboflow") if Path("data/raw/roboflow").exists() else None)
    merge.run(soccernet=sn, roboflow=rf, out_dir=args.out, use_symlinks=not args.copy)


def cmd_all(args: argparse.Namespace) -> None:
    """Quick path: Roboflow players + Roboflow pitch + merge. No NDA needed."""
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY env var or pass --api-key.")

    from football_tracker.datasets import merge, pitch_roboflow, roboflow

    rf_out = Path("data/raw/roboflow")
    pitch_out = Path("data/raw/pitch")
    roboflow.download(
        workspace="roboflow-jvuqo",
        project="football-players-detection-3zvbc",
        version=12,
        out_dir=rf_out,
        api_key=api_key,
    )
    pitch_roboflow.download(
        workspace="roboflow-jvuqo",
        project="football-field-detection-f07vi",
        version=15,
        out_dir=pitch_out,
        api_key=api_key,
    )
    merge.run(
        soccernet=None, roboflow=rf_out,
        out_dir=Path("data/processed/combined"), use_symlinks=True,
    )


# ---------- argparse ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dataset.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sn = sub.add_parser("soccernet", help="Download SoccerNet (needs NDA password).")
    sn.add_argument("--split", default="tracking", choices=["tracking", "tracking-test", "detection"])
    sn.add_argument("--out", type=Path, default=Path("data/raw/soccernet"))
    sn.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"))
    sn.add_argument("--no-convert", dest="convert", action="store_false")
    sn.set_defaults(func=cmd_soccernet, convert=True)

    rf = sub.add_parser("roboflow", help="Download Roboflow players dataset.")
    rf.add_argument("--workspace", default="roboflow-jvuqo")
    rf.add_argument("--project", default="football-players-detection-3zvbc")
    rf.add_argument("--version", type=int, default=12)
    rf.add_argument("--out", type=Path, default=Path("data/raw/roboflow"))
    rf.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    rf.set_defaults(func=cmd_roboflow)

    pt = sub.add_parser("pitch", help="Download Roboflow pitch-keypoints dataset.")
    pt.add_argument("--workspace", default="roboflow-jvuqo")
    pt.add_argument("--project", default="football-field-detection-f07vi")
    pt.add_argument("--version", type=int, default=15)
    pt.add_argument("--out", type=Path, default=Path("data/raw/pitch"))
    pt.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    pt.set_defaults(func=cmd_pitch)

    yt = sub.add_parser("youtube", help="Pull a YouTube match clip via yt-dlp.")
    yt.add_argument("url")
    yt.add_argument("--out", type=Path, default=Path("data/raw/youtube"))
    yt.add_argument("--resolution", default="1080")
    yt.set_defaults(func=cmd_youtube)

    mg = sub.add_parser("merge", help="Merge player datasets into configs/combined.yaml.")
    mg.add_argument("--soccernet", type=Path, default=None)
    mg.add_argument("--roboflow", type=Path, default=None)
    mg.add_argument("--out", type=Path, default=Path("data/processed/combined"))
    mg.add_argument("--copy", action="store_true", help="Copy files instead of symlinking.")
    mg.set_defaults(func=cmd_merge)

    al = sub.add_parser("all", help="Roboflow players + pitch + merge (no NDA needed).")
    al.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    al.set_defaults(func=cmd_all)

    return p


def main() -> int:
    args = build_parser().parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

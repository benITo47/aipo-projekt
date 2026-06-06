"""SoccerNet downloader + MOT-style → YOLO label conversion.

SoccerNet tracking ships gameClips with `gameinfo.ini` and `gt/gt.txt` (MOT format):

    frame, track_id, x, y, w, h, conf, class_id, visibility

where image-coords are top-left pixel anchored. We rewrite to YOLO format:

    cls cx_norm cy_norm w_norm h_norm

per-frame `.txt` files alongside each rendered image. Class IDs are remapped to
the canonical scheme in `configs/classes.yaml`.
"""
from __future__ import annotations

import configparser
import shutil
from pathlib import Path

import cv2
import yaml
from rich.console import Console
from rich.progress import track

console = Console()


# SoccerNet tracking gt.txt class IDs → canonical class IDs (configs/classes.yaml).
# SoccerNet GSR / tracking uses: 1=player, 2=goalkeeper, 3=referee, 4=ball.
_SOCCERNET_TO_CANONICAL = {1: 0, 2: 1, 3: 2, 4: 3}


def download(split: str, out_dir: Path, password: str | None) -> None:
    """Pull a SoccerNet split via the official pip package.

    Requires the NDA password emailed after registering at https://soccer-net.org.
    """
    if not password:
        raise SystemExit(
            "SoccerNet downloads need the NDA password. "
            "Set SOCCERNET_PASSWORD env var or pass --password."
        )
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as e:
        raise SystemExit(
            "Install the SoccerNet package first: pip install SoccerNet"
        ) from e

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dl = SoccerNetDownloader(LocalDirectory=str(out_dir))
    dl.password = password

    console.log(f"[cyan]Downloading SoccerNet split[/] split={split} out={out_dir}")

    if split == "tracking":
        dl.downloadDataSet(task="tracking", split=["train", "test", "challenge"])
    elif split == "tracking-test":
        dl.downloadDataSet(task="tracking", split=["test"])
    elif split == "detection":
        # game-state-reconstruction also provides per-frame detection labels.
        dl.downloadDataSet(task="gamestate-2024", split=["train", "valid", "test"])
    else:
        raise SystemExit(f"Unknown split: {split}")

    console.log("[green]SoccerNet download complete[/]")


def convert_to_yolo(out_dir: Path, dataset_yaml: Path | None = None) -> Path:
    """Walk a downloaded SoccerNet tree and emit a YOLO dataset.

    Output layout (matches Ultralytics expectations):

        <out_dir>/yolo/
            images/{train,val}/*.jpg
            labels/{train,val}/*.txt
            data.yaml
    """
    out_dir = out_dir.resolve()
    yolo_root = out_dir / "yolo"
    (yolo_root / "images/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "images/val").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels/train").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels/val").mkdir(parents=True, exist_ok=True)

    clip_dirs = sorted(
        p.parent for p in out_dir.rglob("gameinfo.ini")
    )
    if not clip_dirs:
        raise SystemExit(f"No SoccerNet clips (gameinfo.ini) found under {out_dir}")

    console.log(f"[cyan]Converting {len(clip_dirs)} clips to YOLO format[/]")

    for clip in track(clip_dirs, description="clips"):
        split = "val" if "test" in clip.parts else "train"
        _convert_clip(clip, yolo_root, split)

    # data.yaml — the file Ultralytics consumes
    dataset_yaml = dataset_yaml or Path("configs/soccernet.yaml")
    dataset_yaml.parent.mkdir(parents=True, exist_ok=True)
    classes = _load_classes()
    yaml_body = {
        "path": str(yolo_root),
        "train": "images/train",
        "val": "images/val",
        "names": classes,
    }
    dataset_yaml.write_text(yaml.safe_dump(yaml_body, sort_keys=False))
    console.log(f"[green]Wrote dataset YAML[/] → {dataset_yaml}")
    return dataset_yaml


def _convert_clip(clip_dir: Path, yolo_root: Path, split: str) -> None:
    """Convert one SoccerNet clip into per-frame YOLO labels + images."""
    img_dir = clip_dir / "img1"
    gt_path = clip_dir / "gt" / "gt.txt"
    if not img_dir.exists() or not gt_path.exists():
        return

    # Frame size — read first image
    sample = next(img_dir.glob("*.jpg"), None)
    if sample is None:
        return
    h, w = cv2.imread(str(sample)).shape[:2]

    info = configparser.ConfigParser()
    info.read(clip_dir / "gameinfo.ini")
    clip_id = clip_dir.name

    # Parse all gt lines and group by frame.
    by_frame: dict[int, list[str]] = {}
    with gt_path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            frame_idx = int(parts[0])
            x, y, bw, bh = map(float, parts[2:6])
            cls_id = int(parts[7])
            if cls_id not in _SOCCERNET_TO_CANONICAL:
                continue
            canon_cls = _SOCCERNET_TO_CANONICAL[cls_id]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            by_frame.setdefault(frame_idx, []).append(
                f"{canon_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
            )

    for frame_path in img_dir.glob("*.jpg"):
        frame_idx = int(frame_path.stem)
        target_name = f"{clip_id}_{frame_idx:06d}.jpg"
        target_img = yolo_root / "images" / split / target_name
        target_lbl = yolo_root / "labels" / split / target_name.replace(".jpg", ".txt")
        if not target_img.exists():
            shutil.copy(frame_path, target_img)
        target_lbl.write_text("\n".join(by_frame.get(frame_idx, [])))

    _ = info  # gameinfo.ini available for downstream (team colors, etc.) if needed.


def _load_classes() -> dict[int, str]:
    path = Path("configs/classes.yaml")
    if not path.exists():
        return {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    body = yaml.safe_load(path.read_text())
    return body["names"]

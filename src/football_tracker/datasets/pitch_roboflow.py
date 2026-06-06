"""Download a Roboflow pitch-keypoints dataset (YOLO-pose format).

Default target: `roboflow-jvuqo/football-field-detection-f07vi` — a public
keypoint dataset built for the Roboflow sports tutorial. It is annotated with
32 pitch landmarks per frame.

**Important:** the keypoint order in the downloaded labels must match
`configs/pitch_keypoints.yaml`. If the chosen Roboflow dataset uses a different
order, you have two options:

  1. Re-order the indices inside each label `.txt` (a small script), OR
  2. Edit `configs/pitch_keypoints.yaml` to mirror the dataset's order.

The downloader injects `kpt_shape` and `flip_idx` into the produced `data.yaml`
so Ultralytics can train a YOLO-pose model directly.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

from football_tracker.pitch.keypoints import load as load_scheme

console = Console()


def download(
    workspace: str,
    project: str,
    version: int,
    out_dir: Path,
    api_key: str | None,
) -> Path:
    if not api_key:
        raise SystemExit(
            "Roboflow needs an API key. Set ROBOFLOW_API_KEY env var or pass --api-key."
        )
    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise SystemExit("Install roboflow first: pip install roboflow") from e

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"[cyan]Downloading Roboflow pose dataset[/] {workspace}/{project}:v{version}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version).download("yolov8-pose", location=str(out_dir))

    return _patch_data_yaml(Path(ds.location))


def _patch_data_yaml(ds_root: Path) -> Path:
    """Ensure the dataset YAML matches our canonical keypoint scheme."""
    scheme = load_scheme()
    data_yaml = ds_root / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"Roboflow export missing data.yaml: {data_yaml}")

    body = yaml.safe_load(data_yaml.read_text())
    body["path"] = str(ds_root)
    body["train"] = body.get("train", "train/images")
    body["val"] = body.get("val", "valid/images")
    body["names"] = {0: "pitch"}            # single bbox class — the whole pitch
    body["nc"] = 1
    body["kpt_shape"] = [scheme.num, 3]
    body["flip_idx"] = scheme.flip_idx

    data_yaml.write_text(yaml.safe_dump(body, sort_keys=False))

    out = Path("configs/pitch.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(body, sort_keys=False))
    console.log(f"[green]Pitch dataset YAML patched[/] → {out}")
    console.log(
        "[yellow]Sanity check:[/] confirm the dataset's keypoint order matches "
        "configs/pitch_keypoints.yaml before training."
    )
    return out

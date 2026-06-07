"""Roboflow Universe downloader.

Default target: `roboflow-jvuqo/football-players-detection-3zvbc` (v12) — a public
YOLO-format dataset with player / goalkeeper / referee / ball classes. The
downloaded `data.yaml` is rewritten to point at canonical class IDs from
`configs/classes.yaml`.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from rich.console import Console

console = Console()


# Roboflow project classes (in the file's own order) → canonical class IDs.
# football-players-detection-3zvbc uses: ['ball', 'goalkeeper', 'player', 'referee']
_ROBOFLOW_NAME_TO_CANONICAL = {
    "ball": 3,
    "goalkeeper": 1,
    "player": 0,
    "referee": 2,
}


def download(
    workspace: str,
    project: str,
    version: int,
    out_dir: Path,
    api_key: str | None,
) -> None:
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

    console.log(f"[cyan]Downloading Roboflow[/] {workspace}/{project}:v{version} → {out_dir}")

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    ds = proj.version(version).download("yolov8", location=str(out_dir), overwrite=True)

    _remap_classes(Path(ds.location))
    console.log("[green]Roboflow download complete[/]")


def _remap_classes(ds_root: Path) -> None:
    """Roboflow ships its own class order; rewrite labels + data.yaml to canonical IDs."""
    data_yaml = ds_root / "data.yaml"
    if not data_yaml.exists():
        return
    body = yaml.safe_load(data_yaml.read_text())
    rf_names: list[str] = body.get("names", [])
    if not rf_names:
        return

    # Map: index in roboflow's order → canonical id
    idx_map: dict[int, int] = {}
    for i, name in enumerate(rf_names):
        key = name.strip().lower()
        if key in _ROBOFLOW_NAME_TO_CANONICAL:
            idx_map[i] = _ROBOFLOW_NAME_TO_CANONICAL[key]

    if not idx_map:
        console.log("[yellow]No class names matched canonical scheme; leaving labels untouched.[/]")
        return

    for split in ("train", "valid", "test"):
        labels_dir = ds_root / split / "labels"
        if not labels_dir.exists():
            continue
        for txt in labels_dir.glob("*.txt"):
            lines = txt.read_text().splitlines()
            out_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                old = int(parts[0])
                if old not in idx_map:
                    continue
                parts[0] = str(idx_map[old])
                out_lines.append(" ".join(parts))
            txt.write_text("\n".join(out_lines))

    # Rewrite data.yaml with canonical names + paths matching Ultralytics expectations
    canonical = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    body["names"] = canonical
    body["nc"] = len(canonical)
    body["path"] = str(ds_root)
    body["train"] = "train/images"
    body["val"] = "valid/images"
    if (ds_root / "test/images").exists():
        body["test"] = "test/images"
    data_yaml.write_text(yaml.safe_dump(body, sort_keys=False))

    # Also drop a copy at configs/roboflow.yaml for convenience
    out = Path("configs/roboflow.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(body, sort_keys=False))
    console.log(f"[green]Roboflow data.yaml remapped to canonical classes[/] → {out}")

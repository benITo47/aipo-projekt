"""Tiny reporting helpers — used by train.py / eval.py / demo.py to dump
structured JSON of every run so we can:

  1. Pinpoint problems after the fact (per-epoch loss curves, per-rejection
     reasons, per-keypoint visibility — without re-running anything).
  2. Drop the numbers straight into the AiPO course report.

Layout:

    outputs/reports/
    ├── eval/
    │   ├── homography-2026-06-14T15-30-00.json
    │   ├── detector-2026-06-14T15-31-00.json
    │   └── pitch-2026-06-14T15-31-30.json
    ├── train/
    │   ├── detector-yolo26m_v2-2026-06-14T16-00-00.json
    │   └── pitch-yolo26m_pitch_v4-2026-06-14T18-00-00.json
    └── demo/
        └── 2026-06-14T17-00-00/
            ├── summary.json
            └── frames.jsonl
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


DEFAULT_ROOT = Path("outputs/reports")


def timestamp_slug() -> str:
    """ISO-8601 with colons swapped to dashes so it's filename-safe."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def report_path(kind: str, slug: str, root: Path | None = None) -> Path:
    """Build a path like `outputs/reports/<kind>/<slug>-<timestamp>.json`."""
    root = root or DEFAULT_ROOT
    out_dir = root / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{slug}-{timestamp_slug()}.json"


def dump_json(data: dict[str, Any], path: Path) -> Path:
    """Atomically write `data` as pretty JSON to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=_json_default, sort_keys=False))
    os.replace(tmp, path)
    return path


def _json_default(obj: Any) -> Any:
    """Make NumPy / Path / datetime serializable."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class JsonlWriter:
    """Append-only JSONL stream — used for per-frame demo dumps.

    Use as a context manager so the file closes cleanly even if the demo
    bails halfway through a video::

        with JsonlWriter(path) as w:
            for frame_record in pipeline_loop(...):
                w.write(frame_record)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: TextIO | None = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlWriter used outside of `with` block")
        self._fh.write(json.dumps(record, default=_json_default))
        self._fh.write("\n")

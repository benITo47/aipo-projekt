"""Canonical pitch-keypoint scheme — loaded once at startup."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml


@dataclass
class KeypointScheme:
    names: list[str]
    world_xy: np.ndarray   # (N, 2), pitch metres
    flip_idx: list[int]
    pitch_size_m: tuple[float, float]

    @property
    def num(self) -> int:
        return len(self.names)


@lru_cache(maxsize=1)
def load(path: str = "configs/pitch_keypoints.yaml") -> KeypointScheme:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Pitch keypoints config missing: {p}")
    body = yaml.safe_load(p.read_text())
    raw = body["keypoints"]
    names = [row[0] for row in raw]
    world = np.array([[row[1], row[2]] for row in raw], dtype=np.float32)
    flip_idx = list(body.get("flip_idx", list(range(len(names)))))
    w, h = body.get("pitch_size_m", [105.0, 68.0])
    if len(names) != body.get("num_keypoints", len(names)):
        raise SystemExit("pitch_keypoints.yaml: num_keypoints does not match list length")
    return KeypointScheme(
        names=names,
        world_xy=world,
        flip_idx=flip_idx,
        pitch_size_m=(float(w), float(h)),
    )

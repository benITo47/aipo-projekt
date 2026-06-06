"""Per-track-ID trail buffer — stores the last N foot positions per player."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class TrailBuffer:
    """Fixed-length deque per track ID. Cheap, O(1) appends + drop-oldest."""

    def __init__(self, max_len: int = 30) -> None:
        self._buf: dict[int, deque] = defaultdict(lambda: deque(maxlen=max_len))
        self._max_len = max_len

    def update(self, track_id: int, point_xy: tuple[float, float]) -> None:
        self._buf[track_id].append(point_xy)

    def get(self, track_id: int) -> np.ndarray:
        return np.asarray(self._buf[track_id], dtype=np.float32)

    def all(self) -> dict[int, np.ndarray]:
        return {tid: np.asarray(pts, dtype=np.float32) for tid, pts in self._buf.items()}

    def drop(self, track_id: int) -> None:
        self._buf.pop(track_id, None)

"""Cumulative occupancy heatmap on the pitch-coords grid.

Per-track-ID or aggregate. Exported as a matplotlib figure at the end of a run.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from football_tracker.pitch.homography import PITCH_HEIGHT_M, PITCH_WIDTH_M


class Heatmap:
    def __init__(self, bins_x: int = 105, bins_y: int = 68) -> None:
        self.bins_x = bins_x
        self.bins_y = bins_y
        self._per_id: dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros((bins_y, bins_x), dtype=np.float32)
        )

    def update(self, track_id: int, pitch_xy: np.ndarray) -> None:
        x, y = pitch_xy
        if not (0 <= x <= PITCH_WIDTH_M and 0 <= y <= PITCH_HEIGHT_M):
            return
        i = min(int(y / PITCH_HEIGHT_M * self.bins_y), self.bins_y - 1)
        j = min(int(x / PITCH_WIDTH_M * self.bins_x), self.bins_x - 1)
        self._per_id[track_id][i, j] += 1

    def aggregate(self) -> np.ndarray:
        if not self._per_id:
            return np.zeros((self.bins_y, self.bins_x), dtype=np.float32)
        return np.sum(np.stack(list(self._per_id.values())), axis=0)

    def per_id(self) -> dict[int, np.ndarray]:
        return dict(self._per_id)

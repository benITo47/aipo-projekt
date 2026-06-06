"""Accumulate distance covered + instantaneous speed per track ID.

Inputs are in pitch metres (post-homography). We smooth speed via an EMA so the
HUD doesn't jitter every frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _PlayerState:
    last_xy: np.ndarray | None = None
    total_m: float = 0.0
    speed_kmh_ema: float = 0.0
    history: list[tuple[float, float]] = field(default_factory=list)


class DistanceTracker:
    def __init__(self, fps: float, ema_alpha: float = 0.25) -> None:
        self.fps = fps
        self.dt = 1.0 / fps
        self.alpha = ema_alpha
        self._state: dict[int, _PlayerState] = {}

    def update(self, track_id: int, pitch_xy: np.ndarray) -> tuple[float, float]:
        """Returns (total_metres, smoothed_speed_kmh)."""
        st = self._state.setdefault(track_id, _PlayerState())
        if st.last_xy is not None:
            delta_m = float(np.linalg.norm(pitch_xy - st.last_xy))
            # Reject unphysical jumps (>15 m between consecutive frames = tracker swap).
            if delta_m < 15.0:
                st.total_m += delta_m
                inst_kmh = (delta_m / self.dt) * 3.6
                st.speed_kmh_ema = (
                    self.alpha * inst_kmh + (1 - self.alpha) * st.speed_kmh_ema
                )
        st.last_xy = pitch_xy.copy()
        st.history.append((float(pitch_xy[0]), float(pitch_xy[1])))
        return st.total_m, st.speed_kmh_ema

    def summary(self) -> dict[int, dict[str, float]]:
        return {
            tid: {"distance_m": s.total_m, "speed_kmh": s.speed_kmh_ema}
            for tid, s in self._state.items()
        }

    def history(self, track_id: int) -> np.ndarray:
        st = self._state.get(track_id)
        if st is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(st.history, dtype=np.float32)

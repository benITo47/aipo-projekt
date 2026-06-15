"""Per-track distance covered + smoothed speed in pitch metres.

The numbers feeding this module are foot-point projections through the per-
frame homography. Two leaks turn naive `delta / dt` into garbage:

  1. **H jitter**: even with EMA smoothing, the homography wobbles a few cm
     frame-to-frame. A stationary player's projected world position moves
     ~2-5 cm/frame in random directions.
  2. **H extrapolation drift**: when the pitch model misses a frame we
     extrapolate from the last good H. If the camera pans during the gap,
     a stationary player's projected world position drifts *with* the
     camera at 10-30 m/s. That's the "supersonic walker" the user saw.

This implementation closes both:

  - **Fresh-fit gating** — distance/speed only update when the homography
    used to project this frame was a fresh per-frame fit (not extrapolated).
    On extrapolated frames we return the last computed values. Eliminates
    the camera-pan-during-extrap leak entirely.
  - **dt-aware sampling** — when a gap of N extrapolated frames separates
    two fresh fits, the delta between them is divided by N/fps seconds, not
    treated as a 1-frame jump. Players can legitimately move during the gap.
  - **Outlier rejection scaled by dt** — single-frame ceiling of MAX_DELTA_M
    becomes MAX_DELTA_M × dt_frames; impossible jumps (track-id swaps,
    detection swaps in scrums) are still cut.
  - **Constant noise floor** — H jitter noise on a delta between two world
    projections is roughly independent of dt (both samples carry the same
    σ regardless of gap size), so the floor stays at NOISE_FLOOR_M.
  - **Sliding-window median** of rate samples — picks the typical speed
    while ignoring outliers that survived the per-sample cap.
  - **Cold-start gate** — speed reads `None` until MIN_SAMPLES_FOR_SPEED
    samples have accumulated; the renderer omits the km/h label rather
    than showing a misleading 0.
  - **Hard display ceiling** at MAX_KMH — belt-and-braces above realistic
    sprint speed.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# Single-frame movement above this = outlier. 0.5 m/frame at 25 fps ≈ 45 km/h —
# comfortably above world-record sprint (~37 km/h) but still flags H glitches
# and track-id swaps which jump many metres.
MAX_DELTA_M = 0.5

# Per-frame movement below this counts as projection jitter, not motion.
# 0.05 m at 25 fps ≈ 4.5 km/h. Combined with the median window this pins
# truly stationary players to exactly 0 km/h. Very slow walking (< 5 km/h)
# reads as stationary — a deliberate trade for a crisp display.
NOISE_FLOOR_M = 0.05

# Realistic ceiling on displayed km/h.
MAX_KMH = 40.0

# Samples (fresh-fit observations of the same track) required before
# reporting speed.
MIN_SAMPLES_FOR_SPEED = 10


@dataclass
class _PlayerState:
    last_xy: np.ndarray | None = None
    last_frame_idx: int = -1
    total_m: float = 0.0
    sample_count: int = 0
    # Sliding window of (delta_m, dt_frames) tuples between consecutive
    # fresh fits. dt_frames is how many video frames elapsed since the
    # previous fresh sample (1 for back-to-back fresh fits, >1 when extrap
    # frames were skipped between them).
    samples: deque = field(default_factory=deque)
    history: list[tuple[float, float]] = field(default_factory=list)


class DistanceTracker:
    def __init__(self, fps: float, window_s: float = 1.0) -> None:
        self.fps = float(fps)
        self._window_size = max(int(round(self.fps * window_s)), 1)
        self._state: dict[int, _PlayerState] = {}

    def update(
        self,
        track_id: int,
        pitch_xy: np.ndarray,
        frame_idx: int,
        is_fresh_fit: bool = True,
    ) -> tuple[float, float | None]:
        """Process one pitch-metres sample for a track ID.

        ``frame_idx`` is the source video's frame number (monotonic).
        ``is_fresh_fit`` is True when this frame's H came from a per-frame
        keypoint solve, False when H was carried over from a past frame.

        Returns ``(total_distance_m, speed_kmh_or_None)``. Speed is ``None``
        until the track has at least ``MIN_SAMPLES_FOR_SPEED`` fresh-fit
        samples. On non-fresh frames distance and speed are held at their
        last computed values — the camera may have panned during the gap,
        making the projected world position untrustworthy.
        """
        st = self._state.get(track_id)
        if st is None:
            st = _PlayerState(samples=deque(maxlen=self._window_size))
            self._state[track_id] = st

        # Always record the raw projected position for trail/heatmap consumers.
        st.history.append((float(pitch_xy[0]), float(pitch_xy[1])))

        # Camera pans during extrap drag the projected world position with
        # them. Skip — return the held values.
        if not is_fresh_fit:
            return st.total_m, self._speed(st)

        # First fresh sample for this track: anchor the baseline, no delta.
        if st.last_xy is None:
            st.last_xy = pitch_xy.copy()
            st.last_frame_idx = frame_idx
            return st.total_m, None

        dt_frames = frame_idx - st.last_frame_idx
        if dt_frames <= 0:
            return st.total_m, self._speed(st)

        raw_delta = float(np.linalg.norm(pitch_xy - st.last_xy))

        # Outlier — the player can't have moved this far in dt_frames frames.
        # Reset baseline without contributing so the next fresh fit measures
        # from the current position rather than a stale anchor.
        if raw_delta > MAX_DELTA_M * dt_frames:
            st.last_xy = pitch_xy.copy()
            st.last_frame_idx = frame_idx
            return st.total_m, self._speed(st)

        # Noise floor is roughly independent of dt (two world-projection
        # samples carry the same σ regardless of gap), so check against the
        # constant floor.
        if raw_delta < NOISE_FLOOR_M:
            contrib = 0.0
        else:
            contrib = raw_delta

        st.total_m += contrib
        st.samples.append((contrib, dt_frames))
        st.last_xy = pitch_xy.copy()
        st.last_frame_idx = frame_idx
        st.sample_count += 1
        return st.total_m, self._speed(st)

    def _speed(self, st: _PlayerState) -> float | None:
        if st.sample_count < MIN_SAMPLES_FOR_SPEED or not st.samples:
            return None
        # Per-sample rate (m/s) = delta_m / (dt_frames / fps) = delta * fps / dt
        rates = np.fromiter(
            (d * self.fps / dt for d, dt in st.samples),
            dtype=np.float32,
            count=len(st.samples),
        )
        med_mps = float(np.median(rates))
        return min(med_mps * 3.6, MAX_KMH)

    def summary(self) -> dict[int, dict[str, float]]:
        """JSON-friendly summary. Speed is 0.0 for tracks that never
        reached MIN_SAMPLES_FOR_SPEED."""
        out: dict[int, dict[str, float]] = {}
        for tid, st in self._state.items():
            sp = self._speed(st)
            out[tid] = {
                "distance_m": st.total_m,
                "speed_kmh": float(sp if sp is not None else 0.0),
            }
        return out

    def history(self, track_id: int) -> np.ndarray:
        st = self._state.get(track_id)
        if st is None:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(st.history, dtype=np.float32)

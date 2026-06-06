"""ByteTrack wrapper — converts YOLO26 detections into persistent track IDs.

Uses `supervision.ByteTrack` rather than the raw ByteTrack repo: it has the
exact same algorithm but plays nicely with `ultralytics.Results`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    class_id: int
    confidence: float
    xyxy: np.ndarray   # shape (4,), pixel coords


class ByteTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 25,
    ) -> None:
        import supervision as sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
        )

    def update(self, ultralytics_result) -> list[TrackedObject]:
        """Take a single ultralytics Results object → list of TrackedObjects."""
        import supervision as sv

        detections = sv.Detections.from_ultralytics(ultralytics_result)
        detections = self._tracker.update_with_detections(detections)

        if detections.tracker_id is None or len(detections) == 0:
            return []

        return [
            TrackedObject(
                track_id=int(tid),
                class_id=int(cid),
                confidence=float(conf),
                xyxy=np.asarray(box, dtype=np.float32),
            )
            for box, tid, cid, conf in zip(
                detections.xyxy,
                detections.tracker_id,
                detections.class_id,
                detections.confidence,
                strict=False,
            )
        ]

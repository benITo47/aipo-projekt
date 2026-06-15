"""ByteTrack wrapper — converts YOLO26 detections into tracked objects.

Uses `supervision.ByteTrack` for the algorithm itself. We return TWO lists:

  - `tracks`: detections with a stable `track_id` (confirmed by ByteTrack)
  - `untracked`: detections the tracker dropped or hasn't confirmed yet

The caller (live.py) renders both — `untracked` gets a faded box with no ID,
`tracks` get the full ID/colour/trail treatment. This is what stops the
"player boxes flicker / disappear" symptom: in the previous design, ~half of
each frame's detections vanished because they hadn't accumulated enough
frames to confirm into a stable track.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedObject:
    track_id: int          # -1 when untracked / unconfirmed
    class_id: int
    confidence: float
    xyxy: np.ndarray       # shape (4,), pixel coords

    @property
    def is_tracked(self) -> bool:
        return self.track_id >= 0


class ByteTracker:
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        # 90 ≈ 3.6 s at 25 fps. Lets a track survive replays, occlusions, and
        # the camera momentarily losing focus without being re-issued a new ID.
        # Was 30 — too short for broadcast TV where ad-board occlusion and
        # close-ups routinely hide a player for 1-2 s.
        lost_track_buffer: int = 90,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 25,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        import supervision as sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(
        self, ultralytics_result
    ) -> tuple[list[TrackedObject], list[TrackedObject]]:
        """Return (tracked, untracked).

        `tracked`   — detections with a valid tracker_id assigned by ByteTrack.
        `untracked` — detections that survived detection but didn't make it
                      through tracking. Same frame, no stable ID.
        """
        import supervision as sv

        raw = sv.Detections.from_ultralytics(ultralytics_result)
        raw_boxes = raw.xyxy
        raw_cls = raw.class_id if raw.class_id is not None else np.zeros(len(raw))
        raw_conf = raw.confidence if raw.confidence is not None else np.zeros(len(raw))

        # Run tracker to get the (subset of) confirmed tracks.
        tracked_dets = self._tracker.update_with_detections(raw)

        tracked: list[TrackedObject] = []
        tracked_boxes_set: set[tuple[float, float, float, float]] = set()
        if tracked_dets.tracker_id is not None and len(tracked_dets) > 0:
            for box, tid, cid, conf in zip(
                tracked_dets.xyxy,
                tracked_dets.tracker_id,
                tracked_dets.class_id,
                tracked_dets.confidence,
                strict=False,
            ):
                if tid is None:
                    continue
                tracked.append(TrackedObject(
                    track_id=int(tid),
                    class_id=int(cid),
                    confidence=float(conf),
                    xyxy=np.asarray(box, dtype=np.float32),
                ))
                tracked_boxes_set.add(tuple(float(x) for x in box))

        # Anything in raw that didn't make it through tracking → untracked.
        untracked: list[TrackedObject] = []
        for box, cid, conf in zip(raw_boxes, raw_cls, raw_conf, strict=False):
            key = tuple(float(x) for x in box)
            if key in tracked_boxes_set:
                continue
            untracked.append(TrackedObject(
                track_id=-1,
                class_id=int(cid),
                confidence=float(conf),
                xyxy=np.asarray(box, dtype=np.float32),
            ))

        return tracked, untracked

"""Per-frame homography from detected pitch keypoints.

Takes the YOLO-pose output for one frame, picks the visible keypoints, and
solves `cv2.findHomography(image_pts, world_pts, RANSAC)`. Falls back to the
last good homography when too few keypoints are visible (occlusion, weird
camera angle).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from football_tracker.pitch.homography import Homography
from football_tracker.pitch.keypoints import KeypointScheme, load as load_scheme


# Minimum keypoints required for a stable homography. 4 is the theoretical
# minimum; we demand 6 for robustness (RANSAC needs slack for outliers).
MIN_KEYPOINTS = 6

# YOLO-pose confidence threshold below which a keypoint is "not visible".
KP_CONF_THRESHOLD = 0.5


@dataclass
class DynamicHomographyResult:
    homography: Homography | None
    n_visible: int
    used_fallback: bool


class DynamicHomographyEstimator:
    """Stateful per-frame homography. Keeps the last good H as a fallback."""

    def __init__(self, scheme: KeypointScheme | None = None) -> None:
        self.scheme = scheme or load_scheme()
        self._last_good: Homography | None = None

    def update(self, pose_result) -> DynamicHomographyResult:
        """`pose_result` is one ultralytics pose Results object."""
        kpts = self._pick_best_pitch_detection(pose_result)
        if kpts is None:
            return DynamicHomographyResult(self._last_good, 0, True)

        # kpts is (N, 3): x_img, y_img, confidence
        confs = kpts[:, 2]
        mask = confs > KP_CONF_THRESHOLD
        n_visible = int(mask.sum())

        if n_visible < MIN_KEYPOINTS:
            return DynamicHomographyResult(self._last_good, n_visible, True)

        image_pts = kpts[mask, :2].astype(np.float32)
        world_pts = self.scheme.world_xy[mask].astype(np.float32)

        H, inlier_mask = cv2.findHomography(image_pts, world_pts, cv2.RANSAC, 5.0)
        if H is None or inlier_mask is None or int(inlier_mask.sum()) < MIN_KEYPOINTS:
            return DynamicHomographyResult(self._last_good, n_visible, True)

        # Approximate image-frame "corners" by back-projecting pitch corners; only
        # used by the saved Homography for inspection.
        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return DynamicHomographyResult(self._last_good, n_visible, True)
        pitch_corners = np.array(
            [
                [0, self.scheme.pitch_size_m[1]],
                [self.scheme.pitch_size_m[0], self.scheme.pitch_size_m[1]],
                [self.scheme.pitch_size_m[0], 0],
                [0, 0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        image_corners = cv2.perspectiveTransform(pitch_corners, inv_H).reshape(-1, 2)

        h = Homography(H=H, image_corners=image_corners)
        self._last_good = h
        return DynamicHomographyResult(h, n_visible, False)

    def save_last(self, path: Path) -> None:
        if self._last_good is not None:
            self._last_good.save(path)

    @staticmethod
    def _pick_best_pitch_detection(pose_result) -> np.ndarray | None:
        """YOLO-pose returns N pitch detections per frame. We want the highest-conf one."""
        if pose_result.keypoints is None or pose_result.keypoints.data is None:
            return None
        kp_data = pose_result.keypoints.data.cpu().numpy()   # (N, K, 3)
        if kp_data.shape[0] == 0:
            return None
        if pose_result.boxes is not None and pose_result.boxes.conf is not None:
            confs = pose_result.boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
        else:
            best = 0
        return kp_data[best]

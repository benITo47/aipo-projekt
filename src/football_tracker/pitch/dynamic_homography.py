"""Per-frame homography from detected pitch keypoints.

Takes the YOLO-pose output for one frame, picks the visible keypoints,
filters out degenerate configurations, and solves
`cv2.findHomography(image_pts, world_pts, RANSAC)`. Optionally EMA-smooths
the result across frames for temporal stability. Falls back to the
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


# Minimum keypoints required to attempt a homography fit.
# Theoretical minimum is 4. We allow 4 because broadcast clips often only show
# half the pitch and we'd reject too much at 6. The geometric-span check below
# is what actually rejects degenerate configurations.
MIN_KEYPOINTS = 4

# YOLO-pose confidence threshold below which a keypoint is "not visible".
# Lowered 0.5 → 0.3 to feed more candidates into RANSAC. The RANSAC outlier
# filter is robust enough to handle the extra noise.
KP_CONF_THRESHOLD = 0.3

# Fraction of pitch dimensions the visible keypoints must span. Rejects
# clustered keypoints that produce a locally-correct but globally-wrong H.
# 0.25 in x = at least 26 m of the 105 m pitch, 0.25 in y = at least 17 m.
MIN_SPAN_X = 0.25
MIN_SPAN_Y = 0.25

# RANSAC reprojection threshold in pixels. Tighter is more accurate but
# rejects more inliers.
RANSAC_REPROJ_PX = 3.0
RANSAC_MAX_ITERS = 2000

# EMA blend factor for the H matrix. 1.0 = no smoothing (raw current frame),
# 0.6 = mostly current with a stabilising tail from the past.
SMOOTHING_ALPHA = 0.6


@dataclass
class DynamicHomographyResult:
    homography: Homography | None
    n_visible: int          # keypoints above KP_CONF_THRESHOLD
    n_inliers: int          # keypoints RANSAC kept
    used_fallback: bool     # True iff this frame's fit failed and we held the last good H
    reason: str | None      # human-readable rejection reason, or None on success


class DynamicHomographyEstimator:
    """Stateful per-frame homography. Keeps the last good H as a fallback and
    EMA-smooths the H matrix across consecutive successful frames."""

    def __init__(
        self,
        scheme: KeypointScheme | None = None,
        smoothing_alpha: float = SMOOTHING_ALPHA,
    ) -> None:
        self.scheme = scheme or load_scheme()
        self._last_good: Homography | None = None
        self._smoothed_H: np.ndarray | None = None
        self._alpha = smoothing_alpha

    def update(
        self,
        pose_result,
        image_shape: tuple[int, int] | None = None,
    ) -> DynamicHomographyResult:
        """Process one frame's pose result; return current-frame H (smoothed)
        or the last good H if this frame's fit failed.

        `image_shape` is (height, width) of the source frame; when provided,
        image corners are back-projected through H to catch one-sided
        homographies that extrapolate wildly outside the pitch.
        """
        kpts = self._pick_best_pitch_detection(pose_result)
        if kpts is None:
            return self._fallback(0, 0, "no pitch detection")

        confs = kpts[:, 2]
        mask = confs > KP_CONF_THRESHOLD
        n_visible = int(mask.sum())

        if n_visible < MIN_KEYPOINTS:
            return self._fallback(n_visible, 0, f"only {n_visible} keypoints above conf")

        image_pts = kpts[mask, :2].astype(np.float32)
        world_pts = self.scheme.world_xy[mask].astype(np.float32)

        # Geometric span check — reject keypoints clustered on one side.
        # If the world points only span one corner of the pitch, RANSAC will fit
        # a locally-correct but globally-divergent H.
        pitch_w, pitch_h = self.scheme.pitch_size_m
        wx_span = float(world_pts[:, 0].max() - world_pts[:, 0].min())
        wy_span = float(world_pts[:, 1].max() - world_pts[:, 1].min())
        if wx_span < MIN_SPAN_X * pitch_w or wy_span < MIN_SPAN_Y * pitch_h:
            return self._fallback(
                n_visible, 0,
                f"keypoints cluster ({wx_span:.0f}×{wy_span:.0f} m)",
            )

        H, inlier_mask = cv2.findHomography(
            image_pts, world_pts, cv2.RANSAC,
            RANSAC_REPROJ_PX, maxIters=RANSAC_MAX_ITERS,
        )
        if H is None or inlier_mask is None:
            return self._fallback(n_visible, 0, "RANSAC failed")

        n_inliers = int(inlier_mask.sum())
        if n_inliers < MIN_KEYPOINTS:
            return self._fallback(n_visible, n_inliers, f"only {n_inliers} inliers")

        # Friend's check: back-project image corners through H and reject if
        # more than one falls far outside the pitch.
        if image_shape is not None:
            h_img, w_img = image_shape
            corners_img = np.array(
                [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32
            ).reshape(-1, 1, 2)
            corners_world = cv2.perspectiveTransform(corners_img, H).reshape(-1, 2)
            margin = 50.0
            n_bad = int(np.sum(
                (corners_world[:, 0] < -margin) | (corners_world[:, 0] > pitch_w + margin) |
                (corners_world[:, 1] < -margin) | (corners_world[:, 1] > pitch_h + margin)
            ))
            if n_bad > 1:
                return self._fallback(n_visible, n_inliers, "H extrapolates outside pitch")

        # EMA-smooth the H matrix for temporal stability. Normalize first to
        # avoid scale-ambiguity issues (H is defined up to a scale factor).
        H_norm = H / H[2, 2]
        if self._smoothed_H is None:
            blended = H_norm
        else:
            blended = self._alpha * H_norm + (1 - self._alpha) * self._smoothed_H
            blended = blended / blended[2, 2]
        self._smoothed_H = blended

        # Compute image-frame "corners" by back-projecting pitch corners through
        # the blended H (used by the saved Homography for inspection).
        try:
            inv_blended = np.linalg.inv(blended)
        except np.linalg.LinAlgError:
            return self._fallback(n_visible, n_inliers, "H singular")
        pitch_corners = np.array(
            [
                [0, pitch_h],
                [pitch_w, pitch_h],
                [pitch_w, 0],
                [0, 0],
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        image_corners = cv2.perspectiveTransform(pitch_corners, inv_blended).reshape(-1, 2)

        good = Homography(H=blended, image_corners=image_corners)
        self._last_good = good
        return DynamicHomographyResult(
            homography=good, n_visible=n_visible, n_inliers=n_inliers,
            used_fallback=False, reason=None,
        )

    def save_last(self, path: Path) -> None:
        if self._last_good is not None:
            self._last_good.save(path)

    def _fallback(self, n_visible: int, n_inliers: int, reason: str) -> DynamicHomographyResult:
        return DynamicHomographyResult(
            homography=self._last_good,
            n_visible=n_visible,
            n_inliers=n_inliers,
            used_fallback=True,
            reason=reason,
        )

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

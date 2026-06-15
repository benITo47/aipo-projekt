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
# 6 (>4 geometric minimum) so RANSAC has an outlier rejection budget;
# inlier requirement is then 5 so RANSAC can drop one bad correspondence
# without killing the frame.
MIN_KEYPOINTS = 6
MIN_INLIERS = 5

# YOLO-pose confidence threshold below which a keypoint is "not visible".
# 0.4 — a balance. The model hallucinates off-screen keypoints in the
# 0.30-0.45 band, but real far-side detections (small line markings in
# tactical-cam) also live in 0.4-0.6. The orientation sanity check below
# is what actually rejects the hallucinated-swap failure mode.
KP_CONF_THRESHOLD = 0.4

# Fraction of pitch dimensions the visible WORLD keypoints must span.
# 0.15 = at least 16 m × 10 m — admits one-goal-area fits, which on a
# wide-angle cam are correct (the model is confident about the close goal
# area but unsure about the far one).
MIN_SPAN_X = 0.15
MIN_SPAN_Y = 0.15

# Fraction of image dimensions the visible PIXEL keypoints must span.
# Hallucinated keypoints typically pile up in a small pixel cluster even
# though their world coordinates (from the scheme) span the full pitch.
MIN_IMAGE_SPAN_X = 0.15
MIN_IMAGE_SPAN_Y = 0.15

# RANSAC reprojection threshold in pixels. Tighter is more accurate but
# rejects more inliers.
RANSAC_REPROJ_PX = 3.0
RANSAC_MAX_ITERS = 2000

# When the four outer corners (TL=0, TR=1, BR=2, BL=3) are visible we can
# verify the index labelling is sane: TL must be LEFT of TR in image pixels,
# BL must be LEFT of BR, TL must be ABOVE BL, TR must be ABOVE BR. Catches
# the failure mode where the model swaps TR↔TL on partial-pitch frames.
_ORIENTATION_PAIRS_X = ((0, 1), (3, 2))   # (left_kpt, right_kpt) — px[left].x < px[right].x
_ORIENTATION_PAIRS_Y = ((0, 3), (1, 2))   # (top_kpt, bot_kpt)    — px[top].y  < px[bot].y

# EMA blend factor for the H matrix. 1.0 = no smoothing (raw current frame),
# 0.6 = mostly current with a stabilising tail from the past.
SMOOTHING_ALPHA = 0.6

# A "wild fit" rejection guard: if a new candidate H jumps the image-centre
# projection by more than this many world metres vs the smoothed H,
# the fit is almost certainly a swap-bug or noisy correspondence set.
# 40 m / fit-event accommodates Sky tactical cam panning (~20-30 m of
# image-centre drift accumulates between consecutive fresh fits) while
# still flagging the truly bad swaps that would project players 60+ m off.
MAX_H_JUMP_M = 40.0

# When a fit fails (too few kpts, cluster, RANSAC), we keep returning the
# last good H indefinitely — until a fresh fit lands. This maximises minimap
# uptime; on a wide-angle camera the geometry never goes stale, and on
# broadcast TV the EMA blends out the old H within a few frames once new
# keypoints arrive. Past STALE_THRESHOLD_FRAMES we still trust the H but
# tint the minimap so the viewer knows it's been a while since a real fit.
STALE_THRESHOLD_FRAMES = 300   # 12 s at 25 fps


@dataclass
class DynamicHomographyResult:
    homography: Homography | None
    n_visible: int          # keypoints above KP_CONF_THRESHOLD
    n_inliers: int          # keypoints RANSAC kept
    # True only when we have NO usable H at all — fresh fit failed AND we're
    # outside the extrapolation budget. The renderer hides projections only
    # in this case. Extrapolated frames have used_fallback=False so the
    # minimap stays populated through brief camera glitches.
    used_fallback: bool
    # True when the H carried by this result was inherited from a recent good
    # fit rather than computed on this frame. The HUD shows a different label;
    # downstream rendering treats it as trusted.
    extrapolated: bool = False
    # Frames elapsed since the last successful fresh fit (0 on a fresh fit).
    frames_since_good: int = 0
    reason: str | None = None


class DynamicHomographyEstimator:
    """Stateful per-frame homography. Keeps the last good H as a fallback and
    EMA-smooths the H matrix across consecutive successful frames."""

    def __init__(
        self,
        scheme: KeypointScheme | None = None,
        smoothing_alpha: float = SMOOTHING_ALPHA,
        stale_threshold: int = STALE_THRESHOLD_FRAMES,
    ) -> None:
        self.scheme = scheme or load_scheme()
        self._last_good: Homography | None = None
        self._smoothed_H: np.ndarray | None = None
        self._alpha = smoothing_alpha
        self._stale_threshold = stale_threshold
        self._frames_since_good = 0

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

        # World-span check — reject when visible keypoints all map to one
        # corner of the pitch (locally-correct but globally-divergent H).
        pitch_w, pitch_h = self.scheme.pitch_size_m
        wx_span = float(world_pts[:, 0].max() - world_pts[:, 0].min())
        wy_span = float(world_pts[:, 1].max() - world_pts[:, 1].min())
        if wx_span < MIN_SPAN_X * pitch_w or wy_span < MIN_SPAN_Y * pitch_h:
            return self._fallback(
                n_visible, 0,
                f"world cluster ({wx_span:.0f}×{wy_span:.0f} m)",
            )

        # Image-span check — catches the failure mode where the model
        # hallucinates keypoints clustered in a small pixel region (worldspans
        # are fine because the scheme covers the pitch even when predictions
        # bunch up). Skip when image_shape isn't known.
        if image_shape is not None:
            ih, iw = image_shape
            ix_span = float(image_pts[:, 0].max() - image_pts[:, 0].min())
            iy_span = float(image_pts[:, 1].max() - image_pts[:, 1].min())
            if ix_span < MIN_IMAGE_SPAN_X * iw or iy_span < MIN_IMAGE_SPAN_Y * ih:
                return self._fallback(
                    n_visible, 0,
                    f"image cluster ({int(ix_span)}×{int(iy_span)} px)",
                )

        # Orientation sanity — if any of the four outer corners are visible
        # AND above conf, their pixel ordering must match the world ordering.
        # Without this, the model swapping TL↔TR on a partial-pitch frame
        # would pass RANSAC with a mirrored H.
        for left_i, right_i in _ORIENTATION_PAIRS_X:
            if confs[left_i] > KP_CONF_THRESHOLD and confs[right_i] > KP_CONF_THRESHOLD:
                if kpts[left_i, 0] >= kpts[right_i, 0]:
                    return self._fallback(
                        n_visible, 0,
                        f"orientation: kpt {left_i} not left of {right_i}",
                    )
        for top_i, bot_i in _ORIENTATION_PAIRS_Y:
            if confs[top_i] > KP_CONF_THRESHOLD and confs[bot_i] > KP_CONF_THRESHOLD:
                if kpts[top_i, 1] >= kpts[bot_i, 1]:
                    return self._fallback(
                        n_visible, 0,
                        f"orientation: kpt {top_i} not above {bot_i}",
                    )

        H, inlier_mask = cv2.findHomography(
            image_pts, world_pts, cv2.RANSAC,
            RANSAC_REPROJ_PX, maxIters=RANSAC_MAX_ITERS,
        )
        if H is None or inlier_mask is None:
            return self._fallback(n_visible, 0, "RANSAC failed")

        n_inliers = int(inlier_mask.sum())
        if n_inliers < MIN_INLIERS:
            return self._fallback(n_visible, n_inliers, f"only {n_inliers} inliers")

        # Sanity check: back-project image corners through H and reject only
        # truly insane H's. Margin is generous (200 m past pitch edge ≈ 3×
        # the pitch length) because wide-angle tactical cams legitimately
        # frame a lot of crowd/stand beyond the pitch and a correct H sends
        # image corners well off-pitch. The orientation + span checks above
        # already reject the swapped-index failure mode; this guard is just
        # for numerical blow-ups.
        if image_shape is not None:
            h_img, w_img = image_shape
            corners_img = np.array(
                [[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]], dtype=np.float32
            ).reshape(-1, 1, 2)
            corners_world = cv2.perspectiveTransform(corners_img, H).reshape(-1, 2)
            margin = 200.0
            n_bad = int(np.sum(
                (corners_world[:, 0] < -margin) | (corners_world[:, 0] > pitch_w + margin) |
                (corners_world[:, 1] < -margin) | (corners_world[:, 1] > pitch_h + margin)
            ))
            if n_bad > 2:
                return self._fallback(n_visible, n_inliers, "H extrapolates outside pitch")

        # EMA-smooth the H matrix for temporal stability. Normalize first to
        # avoid scale-ambiguity issues (H is defined up to a scale factor).
        H_norm = H / H[2, 2]
        if self._smoothed_H is None:
            blended = H_norm
        else:
            # Wild-fit rejection: project the image centre through both the
            # candidate H and the smoothed H; if they disagree by more than
            # MAX_H_JUMP_M metres, this candidate is almost certainly noise
            # (poor inlier set, partial-pitch frame, etc.). Drop it and keep
            # extrapolating from the smoothed H.
            if image_shape is not None:
                h_img, w_img = image_shape
                centre = np.array(
                    [[[w_img / 2.0, h_img / 2.0]]], dtype=np.float32
                )
                old_w = cv2.perspectiveTransform(centre, self._smoothed_H)[0, 0]
                new_w = cv2.perspectiveTransform(centre, H_norm)[0, 0]
                jump = float(np.linalg.norm(new_w - old_w))
                if jump > MAX_H_JUMP_M:
                    return self._fallback(
                        n_visible, n_inliers,
                        f"H jump {jump:.0f}m vs smoothed",
                    )
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
        self._frames_since_good = 0
        return DynamicHomographyResult(
            homography=good, n_visible=n_visible, n_inliers=n_inliers,
            used_fallback=False, extrapolated=False, frames_since_good=0,
            reason=None,
        )

    def save_last(self, path: Path) -> None:
        if self._last_good is not None:
            self._last_good.save(path)

    def _fallback(self, n_visible: int, n_inliers: int, reason: str) -> DynamicHomographyResult:
        self._frames_since_good += 1
        have_last = self._last_good is not None
        return DynamicHomographyResult(
            homography=self._last_good,
            n_visible=n_visible,
            n_inliers=n_inliers,
            # used_fallback only true when we have literally never had a good
            # fit. With a last good H we always extrapolate — the only signal
            # to the renderer is whether the extrapolation has gone stale.
            used_fallback=not have_last,
            extrapolated=have_last,
            frames_since_good=self._frames_since_good,
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

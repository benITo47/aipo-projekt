"""Top-down minimap renderer — draws a FIFA-spec pitch + projected players."""
from __future__ import annotations

import cv2
import numpy as np

from football_tracker.pitch.homography import PITCH_HEIGHT_M, PITCH_WIDTH_M


PITCH_GREEN = (53, 122, 49)
LINE_WHITE = (255, 255, 255)


class Minimap:
    """Pre-renders the empty pitch once; per-frame just blits players on top."""

    def __init__(self, width_px: int = 600) -> None:
        self.scale = width_px / PITCH_WIDTH_M
        self.w = width_px
        self.h = int(PITCH_HEIGHT_M * self.scale)
        self._base = self._render_pitch()

    def render(self, players_pitch_xy: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """`players_pitch_xy` is (N, 2) in metres; `class_ids` is (N,) canonical IDs."""
        canvas = self._base.copy()
        for (x, y), cls in zip(players_pitch_xy, class_ids, strict=False):
            px = int(x * self.scale)
            # Pitch origin is bottom-left, image origin is top-left → flip Y.
            py = int(self.h - y * self.scale)
            colour = _class_colour(int(cls))
            cv2.circle(canvas, (px, py), 6, colour, -1)
            cv2.circle(canvas, (px, py), 6, (0, 0, 0), 1)
        return canvas

    def _render_pitch(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), PITCH_GREEN, dtype=np.uint8)
        s = self.scale

        # Outer boundary
        cv2.rectangle(img, (0, 0), (self.w - 1, self.h - 1), LINE_WHITE, 2)
        # Halfway line
        midx = int(PITCH_WIDTH_M / 2 * s)
        cv2.line(img, (midx, 0), (midx, self.h - 1), LINE_WHITE, 2)
        # Centre circle (radius 9.15 m)
        cv2.circle(img, (midx, self.h // 2), int(9.15 * s), LINE_WHITE, 2)
        cv2.circle(img, (midx, self.h // 2), 3, LINE_WHITE, -1)
        # Penalty boxes (16.5 × 40.32 m)
        pb_h = int(40.32 * s)
        pb_w = int(16.5 * s)
        pb_y0 = (self.h - pb_h) // 2
        cv2.rectangle(img, (0, pb_y0), (pb_w, pb_y0 + pb_h), LINE_WHITE, 2)
        cv2.rectangle(img, (self.w - pb_w, pb_y0), (self.w - 1, pb_y0 + pb_h), LINE_WHITE, 2)
        # Goal boxes (5.5 × 18.32 m)
        gb_h = int(18.32 * s)
        gb_w = int(5.5 * s)
        gb_y0 = (self.h - gb_h) // 2
        cv2.rectangle(img, (0, gb_y0), (gb_w, gb_y0 + gb_h), LINE_WHITE, 2)
        cv2.rectangle(img, (self.w - gb_w, gb_y0), (self.w - 1, gb_y0 + gb_h), LINE_WHITE, 2)
        return img


def _class_colour(class_id: int) -> tuple[int, int, int]:
    # BGR — OpenCV convention.
    return {
        0: (211,   0, 148),  # player     — violet
        1: ( 30, 165, 255),  # goalkeeper — bright orange (high-contrast vs violet)
        2: (  0, 255, 255),  # referee    — yellow (FIFA whistle)
        3: (255, 255, 255),  # ball       — white
    }.get(class_id, (200, 200, 200))

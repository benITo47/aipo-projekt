"""Image → pitch-coords homography.

We use 4 manually-clicked pitch corners (or a saved JSON) to compute a single
perspective transform. The pitch model is the FIFA reference: 105 × 68 m, with
the origin at the bottom-left corner.

A more sophisticated implementation would use detected pitch lines via Hough +
ML keypoints, but for the AiPO demo a static 4-point homography is enough.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PITCH_WIDTH_M = 105.0
PITCH_HEIGHT_M = 68.0

# Order: top-left, top-right, bottom-right, bottom-left of the pitch (in pitch-frame metres).
_PITCH_CORNERS_M = np.array(
    [
        [0.0, PITCH_HEIGHT_M],
        [PITCH_WIDTH_M, PITCH_HEIGHT_M],
        [PITCH_WIDTH_M, 0.0],
        [0.0, 0.0],
    ],
    dtype=np.float32,
)


@dataclass
class Homography:
    H: np.ndarray              # image-pixels → pitch-metres
    image_corners: np.ndarray  # (4, 2) the user-clicked image points

    def project(self, image_points: np.ndarray) -> np.ndarray:
        """Project an (N, 2) array of image points to pitch metres."""
        if image_points.size == 0:
            return image_points.reshape(0, 2)
        pts = image_points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(pts, self.H)
        return out.reshape(-1, 2)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"image_corners": self.image_corners.tolist()}))

    @classmethod
    def load(cls, path: Path) -> "Homography":
        body = json.loads(Path(path).read_text())
        image_corners = np.asarray(body["image_corners"], dtype=np.float32)
        H = cv2.getPerspectiveTransform(image_corners, _PITCH_CORNERS_M)
        return cls(H=H, image_corners=image_corners)


def calibrate_interactive(first_frame: np.ndarray) -> Homography:
    """Pop a window; user clicks 4 pitch corners (TL → TR → BR → BL)."""
    clicks: list[tuple[int, int]] = []
    display = first_frame.copy()
    window = "Click pitch corners: TL → TR → BR → BL  (Enter to confirm)"

    def _on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                display, str(len(clicks)), (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, _on_click)
    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and len(clicks) == 4:  # Enter / Space
            break
        if key == 27:  # Esc
            cv2.destroyWindow(window)
            raise SystemExit("Calibration cancelled.")
    cv2.destroyWindow(window)

    image_corners = np.asarray(clicks, dtype=np.float32)
    H = cv2.getPerspectiveTransform(image_corners, _PITCH_CORNERS_M)
    return Homography(H=H, image_corners=image_corners)

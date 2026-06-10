"""Frame preprocessing for pitch keypoint detection.

Enhances white pitch lines by suppressing the green channel (grass is
G >> R,B; lines are R≈G≈B≈high). Applying this *identically* at training
time and inference time lets the model learn cleaner line features.

Usage at inference::

    from football_tracker.pitch.preprocess import enhance_pitch_lines
    processed = enhance_pitch_lines(frame)
    result = pitch_model.predict(processed, imgsz=960)

To enable during training, pass ``preprocess=True`` to the training config
or call this function inside a custom dataset transform.
"""
from __future__ import annotations

import cv2
import numpy as np


def enhance_pitch_lines(frame: np.ndarray, green_factor: float = 0.75) -> np.ndarray:
    """Suppress the green channel to make white pitch lines stand out.

    Grass pixels have G >> R,B.  White lines have R≈G≈B (high brightness).
    Multiplying the G channel by < 1 darkens grass without affecting lines.

    Args:
        frame: BGR uint8 image.
        green_factor: G channel multiplier (0.0 = fully suppressed, 1.0 = no-op).
                      0.70–0.80 is a good range; larger changes hurt the model
                      if it was not trained with this preprocessing.

    Returns:
        Preprocessed BGR uint8 image, same shape as input.
    """
    out = frame.astype(np.float32)
    out[:, :, 1] *= green_factor          # suppress green
    return np.clip(out, 0, 255).astype(np.uint8)


def enhance_pitch_lines_clahe(frame: np.ndarray, green_factor: float = 0.75) -> np.ndarray:
    """Green suppression + CLAHE on luminance for better local contrast."""
    # Green suppression first
    out = frame.astype(np.float32)
    out[:, :, 1] *= green_factor
    out = np.clip(out, 0, 255).astype(np.uint8)
    # CLAHE on L channel in Lab
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

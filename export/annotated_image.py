"""
annotated_image.py — Save image with measurement overlays burned in.
"""
from __future__ import annotations

import cv2
import numpy as np

from core.stripe_detector import StripeDetectionResult


def draw_stripe_overlays(
    image: np.ndarray,
    detection: StripeDetectionResult,
    roi: tuple[int, int, int, int] | None,
    um_per_px: float = 1.0,
    white_color: tuple = (255, 100, 100),   # BGR
    black_color: tuple = (100, 100, 255),
    edge_color: tuple = (0, 255, 0),
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Return a colour copy of image with stripe overlays and width labels.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    roi_x, roi_y = (roi[0], roi[1]) if roi else (0, 0)
    h = out.shape[0]

    for stripe in detection.stripes:
        lx = int(stripe.left_edge_px) + roi_x
        rx = int(stripe.right_edge_px) + roi_x
        color = white_color if stripe.kind == "white" else black_color

        # Shaded band
        overlay = out.copy()
        cv2.rectangle(overlay, (lx, 0), (rx, h), color, -1)
        out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

        # Left and right edge lines
        cv2.line(out, (lx, 0), (lx, h), edge_color, 1)
        cv2.line(out, (rx, 0), (rx, h), edge_color, 1)

        # Width label
        width_um = stripe.width_px * um_per_px
        label = f"{width_um:.2f}µm"
        cx = (lx + rx) // 2
        cy = h // 4 if stripe.kind == "white" else 3 * h // 4
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = max(0, cx - text_size[0] // 2)
        cv2.putText(out, label, (tx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1, cv2.LINE_AA)

    # ROI rectangle
    if roi:
        x, y, w, h_roi = roi
        cv2.rectangle(out, (x, y), (x + w, y + h_roi), (0, 200, 200), 2)

    return out


def save_annotated_image(
    image: np.ndarray,
    detection: StripeDetectionResult,
    roi: tuple | None,
    output_path: str,
    um_per_px: float = 1.0,
) -> str:
    """Draw overlays and save to output_path (PNG). Returns path."""
    annotated = draw_stripe_overlays(image, detection, roi, um_per_px)
    cv2.imwrite(output_path, annotated)
    return output_path

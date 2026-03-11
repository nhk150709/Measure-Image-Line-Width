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
    direction: str = "horizontal",
    ler_points: list | None = None,
    white_color: tuple = (255, 100, 100),   # BGR
    black_color: tuple = (100, 100, 255),
    edge_color: tuple = (0, 255, 0),
    font_scale: float = 0.45,
) -> np.ndarray:
    """
    Return a colour copy of image with stripe overlays and width labels.

    direction='horizontal' — vertical stripes; bands drawn as vertical columns.
    direction='vertical'   — horizontal stripes; bands drawn as horizontal rows.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        out = image.copy()

    img_h, img_w = out.shape[:2]

    if direction == "horizontal":
        roi_x = roi[0] if roi else 0
        for stripe in detection.stripes:
            lx = int(stripe.left_edge_px) + roi_x
            rx = int(stripe.right_edge_px) + roi_x
            color = white_color if stripe.kind == "white" else black_color

            overlay = out.copy()
            cv2.rectangle(overlay, (lx, 0), (rx, img_h), color, -1)
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

            cv2.line(out, (lx, 0), (lx, img_h), edge_color, 1)
            cv2.line(out, (rx, 0), (rx, img_h), edge_color, 1)

            width_um = stripe.width_px * um_per_px
            label = f"{width_um:.2f}µm"
            cx = (lx + rx) // 2
            cy = img_h // 4 if stripe.kind == "white" else 3 * img_h // 4
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = max(0, cx - text_size[0] // 2)
            cv2.putText(out, label, (tx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1, cv2.LINE_AA)

    else:  # "vertical" direction → horizontal stripes
        roi_y = roi[1] if roi else 0
        for stripe in detection.stripes:
            ty = int(stripe.left_edge_px) + roi_y
            by = int(stripe.right_edge_px) + roi_y
            color = white_color if stripe.kind == "white" else black_color

            overlay = out.copy()
            cv2.rectangle(overlay, (0, ty), (img_w, by), color, -1)
            out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

            cv2.line(out, (0, ty), (img_w, ty), edge_color, 1)
            cv2.line(out, (0, by), (img_w, by), edge_color, 1)

            width_um = stripe.width_px * um_per_px
            label = f"{width_um:.2f}µm"
            cy = (ty + by) // 2
            cx = img_w // 4 if stripe.kind == "white" else 3 * img_w // 4
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx = max(0, cx - text_size[0] // 2)
            cv2.putText(out, label, (tx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1, cv2.LINE_AA)

    # LER sample points: orange for left edge, cyan for right edge
    if ler_points:
        for left_pts, right_pts in ler_points:
            for px, py in left_pts:
                cv2.circle(out, (int(px), int(py)), 2, (0, 140, 255), -1, cv2.LINE_AA)
            for px, py in right_pts:
                cv2.circle(out, (int(px), int(py)), 2, (255, 200, 0), -1, cv2.LINE_AA)

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
    direction: str = "horizontal",
    ler_points: list | None = None,
) -> str:
    """Draw overlays and save to output_path (PNG). Returns path."""
    annotated = draw_stripe_overlays(
        image, detection, roi, um_per_px,
        direction=direction, ler_points=ler_points,
    )
    cv2.imwrite(output_path, annotated)
    return output_path

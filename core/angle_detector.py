"""
angle_detector.py — Detect the dominant orientation of stripes in an image region.

Methods:
  - Hough lines (default, robust)
  - FFT power spectrum peak
Returns angle in degrees (rotation needed to make stripes horizontal).
"""
from __future__ import annotations

import numpy as np
import cv2


def detect_stripe_angle_hough(
    roi: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    min_line_length: int = 50,
    max_line_gap: int = 20,
) -> float | None:
    """
    Detect dominant line angle using Probabilistic Hough Transform.
    Returns angle in degrees relative to horizontal, or None if detection fails.
    """
    edges = cv2.Canny(roi, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    # Cluster around the dominant direction
    angles = np.array(angles)
    # Use circular mean in [-90, 90]
    angles = np.where(angles > 90, angles - 180, angles)
    angles = np.where(angles < -90, angles + 180, angles)
    dominant = float(np.median(angles))
    return dominant


def detect_stripe_angle_fft(roi: np.ndarray) -> float | None:
    """
    Detect dominant stripe direction via 2D FFT power spectrum.
    Returns angle in degrees (direction of the stripe lines, not the perpendicular).
    """
    f = np.fft.fft2(roi.astype(float))
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    # Suppress DC
    cy, cx = np.array(power.shape) // 2
    r = max(5, min(power.shape) // 20)
    cv2.circle(power, (cx, cy), r, 0, -1)

    # Find the peak in the upper half (avoid symmetry duplication)
    half = power[: cy, :]
    idx = np.unravel_index(np.argmax(half), half.shape)
    py, px = idx
    # Angle of the vector from center to peak = perpendicular to stripes
    dy = cy - py
    dx = px - cx
    angle_perp = np.degrees(np.arctan2(dy, dx))
    # Stripe angle = perpendicular + 90
    stripe_angle = angle_perp - 90.0
    # Normalise to [-90, 90]
    stripe_angle = ((stripe_angle + 90) % 180) - 90
    return float(stripe_angle)


def detect_angle(
    roi: np.ndarray,
    method: str = "hough",
    user_angle: float | None = None,
) -> float:
    """
    High-level angle detection.
    method: "hough" | "fft" | "manual"
    user_angle: used when method == "manual"
    Returns angle in degrees (stripes are at this angle from horizontal).
    """
    if method == "manual" and user_angle is not None:
        return user_angle

    if method == "fft":
        angle = detect_stripe_angle_fft(roi)
    else:  # default: hough
        angle = detect_stripe_angle_hough(roi)

    if angle is None:
        # Try fallback
        angle = detect_stripe_angle_fft(roi)
    if angle is None:
        return 0.0
    return angle


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate image so that stripes become VERTICAL (running top→bottom).

    After rotation a horizontal profile (left→right) crosses the stripes
    perpendicularly, which is what extract_averaged_profile with
    direction='horizontal' expects.

    angle_deg is the detected stripe angle relative to horizontal (from Hough/FFT):
      - 90°  → stripes already vertical → no rotation applied
      - 87°  → 3° correction applied
      - 0°   → horizontal stripes → rotated 90° CCW to make them vertical
    """
    # Correction angle to make stripes vertical.
    # Stripe orientation has 180° periodicity, so clamp to [-90, 90].
    # Hough gives the angle of stripe lines; rotating by (angle_deg - 90°)
    # in cv2 image space makes them vertical.
    correction = angle_deg - 90.0
    if correction > 90.0:
        correction -= 180.0
    elif correction < -90.0:
        correction += 180.0
    if abs(correction) < 0.05:
        return image
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), correction, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)
    return rotated

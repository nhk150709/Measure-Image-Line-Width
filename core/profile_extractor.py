"""
profile_extractor.py — Extract averaged intensity profile perpendicular to stripes.
"""
from __future__ import annotations

import numpy as np


def extract_averaged_profile(
    image: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    n_lines: int = 20,
    direction: str = "horizontal",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract an averaged 1D intensity profile from the image.

    Parameters
    ----------
    image : grayscale uint8 ndarray
    roi   : (x, y, w, h) region of interest; None = full image
    n_lines : number of equally-spaced lines to average perpendicular to stripes
    direction : "horizontal" — stripes are vertical, profile goes left→right
                "vertical"   — stripes are horizontal, profile goes top→bottom

    Returns
    -------
    positions : pixel indices along profile axis
    intensity : averaged intensity values
    """
    if roi is not None:
        x, y, w, h = roi
        x, y = max(0, x), max(0, y)
        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)
        region = image[y:y2, x:x2]
    else:
        region = image

    if direction == "horizontal":
        # Average along rows (average many rows, profile goes across columns)
        h_r, w_r = region.shape
        step = max(1, h_r // (n_lines + 1))
        row_indices = [step * i for i in range(1, n_lines + 1) if step * i < h_r]
        if not row_indices:
            row_indices = [h_r // 2]
        rows = region[row_indices, :]
        profile = rows.mean(axis=0)
        positions = np.arange(profile.shape[0], dtype=float)
    else:  # vertical
        h_r, w_r = region.shape
        step = max(1, w_r // (n_lines + 1))
        col_indices = [step * i for i in range(1, n_lines + 1) if step * i < w_r]
        if not col_indices:
            col_indices = [w_r // 2]
        cols = region[:, col_indices]
        profile = cols.mean(axis=1)
        positions = np.arange(profile.shape[0], dtype=float)

    return positions, profile.astype(float)


def extract_line_profile(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    width: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract profile along an arbitrary line (x1,y1)→(x2,y2).
    width > 1 averages over that many parallel pixel lines.

    Returns (distances_px, intensity).
    """
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return np.array([0.0]), np.array([float(image[y1, x1])])

    xs = np.linspace(x1, x2, length).astype(int)
    ys = np.linspace(y1, y2, length).astype(int)

    # Clamp to image bounds
    xs = np.clip(xs, 0, image.shape[1] - 1)
    ys = np.clip(ys, 0, image.shape[0] - 1)

    if width <= 1:
        intensity = image[ys, xs].astype(float)
    else:
        # Average over 'width' parallel lines
        dx = (y2 - y1) / length  # perpendicular direction (rotated 90°)
        dy = -(x2 - x1) / length
        half = width // 2
        profiles = []
        for offset in range(-half, half + 1):
            ox_s = int(np.clip(xs + offset * dx, 0, image.shape[1] - 1))
            oy_s = int(np.clip(ys + offset * dy, 0, image.shape[0] - 1))
            ox_arr = np.clip((xs + offset * dx).astype(int), 0, image.shape[1] - 1)
            oy_arr = np.clip((ys + offset * dy).astype(int), 0, image.shape[0] - 1)
            profiles.append(image[oy_arr, ox_arr].astype(float))
        intensity = np.mean(profiles, axis=0)

    distances = np.arange(length, dtype=float)
    return distances, intensity

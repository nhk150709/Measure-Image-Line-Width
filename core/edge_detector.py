"""
edge_detector.py — Detect edges (transitions) in a 1D intensity profile.

Methods:
  threshold  — crossing of a fixed intensity level (% of min-max range or absolute)
  canny      — position of maximum absolute gradient (derivative peak)
  sigmoid    — fit an error function to each transition (most accurate)

Returns edge positions as pixel indices (float).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# ──────────────────────────────────────────────────────────────────────
# Helper: sigmoid (error function) model
# ──────────────────────────────────────────────────────────────────────

def _erf_model(x, x0, amplitude, width, baseline):
    """Sigmoid (erf) transition from baseline to baseline+amplitude."""
    from scipy.special import erf
    return baseline + amplitude * 0.5 * (1 + erf((x - x0) / (width * np.sqrt(2))))


def _fit_erf_edge(profile: np.ndarray, x: np.ndarray, rising: bool) -> float | None:
    """Fit an erf to one transition region; return the inflection point x0."""
    amp_guess = float(profile.max() - profile.min())
    if not rising:
        amp_guess = -amp_guess
    baseline_guess = float(profile.min() if rising else profile.max())
    x0_guess = float(x[len(x) // 2])
    w_guess = max(1.0, len(x) * 0.1)
    try:
        popt, _ = curve_fit(
            _erf_model, x, profile.astype(float),
            p0=[x0_guess, amp_guess, w_guess, baseline_guess],
            maxfev=2000,
        )
        return float(popt[0])  # x0 = edge position
    except RuntimeError:
        return None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def find_edges_threshold(
    profile: np.ndarray,
    threshold_fraction: float = 0.5,
    threshold_abs: float | None = None,
) -> list[float]:
    """
    Find all threshold crossings in the profile.
    threshold_fraction: fraction of (max-min) range, used if threshold_abs is None.
    Returns list of interpolated pixel positions of crossings (both rising & falling).
    """
    lo, hi = float(profile.min()), float(profile.max())
    level = threshold_abs if threshold_abs is not None else lo + threshold_fraction * (hi - lo)

    crossings = []
    for i in range(len(profile) - 1):
        a, b = float(profile[i]), float(profile[i + 1])
        if (a - level) * (b - level) < 0:  # sign change
            # Linear interpolation
            frac = (level - a) / (b - a)
            crossings.append(i + frac)
    return crossings


def find_edges_canny(
    profile: np.ndarray,
    smooth_sigma: float = 1.0,
) -> list[float]:
    """
    Find edges as peaks of |gradient|.
    Returns sorted list of pixel positions.
    """
    n = len(profile)
    if n < 5:
        return []

    # Smooth then differentiate
    kernel_size = max(5, int(smooth_sigma * 4) | 1)
    kernel_size = min(kernel_size, n - 1 if n % 2 == 0 else n)
    if kernel_size >= n:
        kernel_size = max(3, n - 1 if (n - 1) % 2 != 0 else n - 2)
    if kernel_size < 3:
        return []

    try:
        smoothed = savgol_filter(profile.astype(float), kernel_size, 2)
    except Exception:
        smoothed = profile.astype(float)

    grad = np.gradient(smoothed)
    abs_grad = np.abs(grad)

    # Threshold gradient at 20% of max
    thresh = 0.2 * abs_grad.max()
    in_peak = False
    peak_start = 0
    edges = []
    for i, g in enumerate(abs_grad):
        if g >= thresh and not in_peak:
            in_peak = True
            peak_start = i
        elif g < thresh and in_peak:
            in_peak = False
            segment = abs_grad[peak_start:i]
            local_max = peak_start + int(np.argmax(segment))
            edges.append(float(local_max))
    if in_peak:
        segment = abs_grad[peak_start:]
        local_max = peak_start + int(np.argmax(segment))
        edges.append(float(local_max))

    return edges


def find_edges_sigmoid(
    profile: np.ndarray,
    threshold_fraction: float = 0.5,
    window_half: int = 20,
) -> list[float]:
    """
    Fit an erf to each transition region.  Transitions are first seeded by threshold.
    Returns list of edge positions (sub-pixel accurate).
    """
    seed_edges = find_edges_threshold(profile, threshold_fraction)
    x = np.arange(len(profile), dtype=float)
    result = []
    prev_direction = None
    lo, hi = float(profile.min()), float(profile.max())
    level = lo + threshold_fraction * (hi - lo)

    for pos in seed_edges:
        idx = int(round(pos))
        # Determine if rising or falling
        rising = profile[max(0, idx - 2)] < profile[min(len(profile) - 1, idx + 2)]
        i0 = max(0, idx - window_half)
        i1 = min(len(profile), idx + window_half + 1)
        seg_x = x[i0:i1]
        seg_y = profile[i0:i1]
        edge = _fit_erf_edge(seg_y, seg_x, rising)
        result.append(edge if edge is not None else pos)

    return result


# ──────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ──────────────────────────────────────────────────────────────────────

def find_edges(
    profile: np.ndarray,
    method: str = "canny",
    threshold_fraction: float = 0.5,
    threshold_abs: float | None = None,
    smooth_sigma: float = 1.0,
) -> list[float]:
    """
    Find edges in a 1D intensity profile.
    method: "threshold" | "canny" | "sigmoid"
    Returns list of edge pixel positions (float, sub-pixel for sigmoid/threshold).
    """
    if method == "threshold":
        return find_edges_threshold(profile, threshold_fraction, threshold_abs)
    elif method == "sigmoid":
        return find_edges_sigmoid(profile, threshold_fraction)
    else:  # default: canny
        return find_edges_canny(profile, smooth_sigma)

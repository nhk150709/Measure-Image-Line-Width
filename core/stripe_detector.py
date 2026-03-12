"""
stripe_detector.py — Gradient-based stripe boundary detection for SEM images.

Stripes are detected by finding rising/falling edges in the 1st derivative
of the intensity profile (gradient peaks), or at the zero-crossings of the
2nd derivative (inflection points).  Edge pairs are then assembled into
Stripe objects according to the selected pairing mode.

Suggested methods
-----------------
gradient_peaks (default)
    Peaks of |d I/dx|.  The positive peak marks a B→W (rising) transition;
    the negative peak marks a W→B (falling) transition.  Robust for sharp
    SEM interfaces.  Recommended starting point.

zero_crossing
    Zero-crossings of d²I/dx², i.e. where the gradient is changing fastest.
    Gives the inflection point of each sigmoid-shaped transition.  More
    sensitive to noise, but can locate the exact centre of a smooth edge
    more precisely than gradient_peaks.

Suggested pairing modes
-----------------------
rising_falling  – B→W then W→B  →  white stripe CD  (photoresist lines)
falling_rising  – W→B then B→W  →  black stripe CD  (spaces / trenches)
rising_rising   – B→W then B→W  →  full pitch
falling_falling – W→B then W→B  →  full pitch
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Stripe:
    """Represents one detected stripe (white or black)."""
    kind: str              # "white" | "black"
    center_px: float       # centre of the interval along the profile axis
    left_edge_px: float    # position of the left edge (rising or falling)
    right_edge_px: float   # position of the right edge
    width_px: float        # right_edge_px − left_edge_px
    peak_intensity: float  # mean profile intensity over the stripe region


@dataclass
class StripeDetectionResult:
    stripes: list[Stripe] = field(default_factory=list)
    profile: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    angle_deg: float = 0.0
    # Derivatives (always computed; used by the derivative plot)
    gradient: np.ndarray = field(default_factory=lambda: np.array([]))
    gradient2: np.ndarray = field(default_factory=lambda: np.array([]))
    # Edge positions detected before pairing
    rising_edges: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    falling_edges: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def white_stripes(self) -> list[Stripe]:
        return [s for s in self.stripes if s.kind == "white"]

    @property
    def black_stripes(self) -> list[Stripe]:
        return [s for s in self.stripes if s.kind == "black"]

    def mean_white_width_px(self) -> float | None:
        ws = [s.width_px for s in self.white_stripes]
        return float(np.mean(ws)) if ws else None

    def mean_black_width_px(self) -> float | None:
        bs = [s.width_px for s in self.black_stripes]
        return float(np.mean(bs)) if bs else None

    def mean_pitch_px(self) -> float | None:
        centers = sorted(s.center_px for s in self.white_stripes)
        if len(centers) < 2:
            return None
        return float(np.mean(np.diff(centers)))


# ── Edge detection ─────────────────────────────────────────────────────────────

def _filter_min_distance(peaks: np.ndarray, min_dist: int) -> np.ndarray:
    """Keep peaks that are at least min_dist apart; first peak in each cluster wins."""
    if len(peaks) == 0:
        return peaks
    keep = [int(peaks[0])]
    for p in peaks[1:]:
        if int(p) - keep[-1] >= min_dist:
            keep.append(int(p))
    return np.array(keep, dtype=int)


def _detect_gradient_peaks(
    smoothed: np.ndarray,
    min_distance_px: float,
    prominence_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find rising and falling edges as positive/negative peaks of the 1st derivative.

    Returns (rising_idxs, falling_idxs, gradient_1st, gradient_2nd).
    """
    grad1 = np.gradient(smoothed)
    grad2 = np.gradient(grad1)

    lo, hi = float(smoothed.min()), float(smoothed.max())
    contrast = hi - lo
    prominence = max(0.5, prominence_fraction * contrast)
    dist = max(1, int(min_distance_px))

    rising, _ = find_peaks(grad1, prominence=prominence, distance=dist)
    falling, _ = find_peaks(-grad1, prominence=prominence, distance=dist)

    return rising, falling, grad1, grad2


def _detect_zero_crossings(
    smoothed: np.ndarray,
    min_distance_px: float,
    prominence_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find rising/falling edges as zero-crossings of the 2nd derivative
    (inflection points of the sigmoid-shaped intensity transitions).

    Returns (rising_idxs, falling_idxs, gradient_1st, gradient_2nd).
    """
    grad1 = np.gradient(smoothed)
    grad2 = np.gradient(grad1)

    lo, hi = float(smoothed.min()), float(smoothed.max())
    contrast = hi - lo
    grad_threshold = max(0.5, prominence_fraction * contrast * 0.5)

    sign2 = np.sign(grad2)
    crossings = np.where(np.diff(sign2))[0]

    rising_list: list[int] = []
    falling_list: list[int] = []
    for z in crossings:
        if abs(grad1[z]) < grad_threshold:
            continue
        if grad1[z] > 0:
            rising_list.append(int(z))
        else:
            falling_list.append(int(z))

    dist = max(1, int(min_distance_px))
    rising = _filter_min_distance(np.array(rising_list, dtype=int), dist)
    falling = _filter_min_distance(np.array(falling_list, dtype=int), dist)

    return rising, falling, grad1, grad2


# ── Pairing ────────────────────────────────────────────────────────────────────

def _pair_to_stripes(
    rising: np.ndarray,
    falling: np.ndarray,
    pairing: str,
) -> list[Stripe]:
    """
    Assemble stripe intervals from edge positions.

    Pairing modes
    -------------
    "rising_falling"  – rising → next falling  → white stripe (B→W→B interval)
    "falling_rising"  – falling → next rising  → black stripe (W→B→W interval)
    "rising_rising"   – rising → next rising   → pitch (one full period)
    "falling_falling" – falling → next falling → pitch (one full period)
    """
    stripes: list[Stripe] = []

    def _make(left: float, right: float, kind: str) -> Stripe:
        w = right - left
        return Stripe(
            kind=kind,
            center_px=(left + right) / 2.0,
            left_edge_px=left,
            right_edge_px=right,
            width_px=w,
            peak_intensity=0.0,  # filled in by _classify_by_intensity
        )

    if pairing == "rising_falling":
        for r in rising:
            nxt = falling[falling > r]
            if len(nxt):
                stripes.append(_make(float(r), float(nxt[0]), "white"))

    elif pairing == "falling_rising":
        for f in falling:
            nxt = rising[rising > f]
            if len(nxt):
                stripes.append(_make(float(f), float(nxt[0]), "black"))

    elif pairing == "rising_rising":
        for i in range(len(rising) - 1):
            stripes.append(_make(float(rising[i]), float(rising[i + 1]), "white"))

    elif pairing == "falling_falling":
        for i in range(len(falling) - 1):
            stripes.append(_make(float(falling[i]), float(falling[i + 1]), "black"))

    return stripes


# ── Classification ─────────────────────────────────────────────────────────────

def _classify_by_intensity(stripes: list[Stripe], smoothed: np.ndarray) -> None:
    """
    Compute mean profile intensity for every stripe region, then reclassify
    each stripe as white or black by comparing its mean to the overall profile
    median.  Modifies stripes in-place.
    """
    if not stripes:
        return
    overall_median = float(np.median(smoothed))
    for s in stripes:
        l_int = max(0, int(s.left_edge_px))
        r_int = min(len(smoothed), int(s.right_edge_px) + 1)
        if r_int > l_int:
            s.peak_intensity = float(np.mean(smoothed[l_int:r_int]))
        else:
            s.peak_intensity = float(smoothed[max(0, l_int)])
        s.kind = "white" if s.peak_intensity >= overall_median else "black"


# ── Public API ─────────────────────────────────────────────────────────────────

def detect_stripes(
    profile: np.ndarray,
    positions: np.ndarray | None = None,
    smoothing_sigma: float = 2.0,
    prominence_fraction: float = 0.15,
    edge_detect_method: str = "gradient_peaks",
    edge_pairing: str = "rising_falling",
    min_edge_distance_px: float = 5.0,
) -> StripeDetectionResult:
    """
    Detect stripes from a 1D intensity profile using gradient-based edge detection.

    Parameters
    ----------
    profile               : 1D intensity array (averaged rows / columns from the ROI)
    positions             : pixel positions along the profile axis (defaults to 0..N-1)
    smoothing_sigma       : Gaussian sigma applied to the profile before differentiation
    prominence_fraction   : minimum edge prominence as a fraction of the intensity contrast
    edge_detect_method    : "gradient_peaks"  – peaks of 1st derivative (recommended for
                              sharp SEM transitions; robust and direct)
                            "zero_crossing"   – zero-crossings of 2nd derivative (inflection
                              points; more sensitive for smooth / gradual transitions)
    edge_pairing          : "rising_falling"  → white stripe CD
                            "falling_rising"  → black stripe / space CD
                            "rising_rising"   → pitch
                            "falling_falling" → pitch
    min_edge_distance_px  : edges closer than this many pixels are suppressed (noise rejection)

    Returns
    -------
    StripeDetectionResult containing stripes, profile, both derivative arrays,
    and the raw rising / falling edge positions.
    """
    if positions is None:
        positions = np.arange(len(profile), dtype=float)

    n = len(profile)
    if n < 10:
        return StripeDetectionResult(profile=profile, positions=positions)

    smoothed = gaussian_filter1d(profile.astype(float), max(0.5, smoothing_sigma))

    lo, hi = float(smoothed.min()), float(smoothed.max())
    if hi - lo < 1:
        return StripeDetectionResult(profile=profile, positions=positions)

    if edge_detect_method == "zero_crossing":
        rising, falling, grad1, grad2 = _detect_zero_crossings(
            smoothed, min_edge_distance_px, prominence_fraction
        )
    else:  # "gradient_peaks" (default)
        rising, falling, grad1, grad2 = _detect_gradient_peaks(
            smoothed, min_edge_distance_px, prominence_fraction
        )

    stripes = _pair_to_stripes(rising, falling, edge_pairing)
    _classify_by_intensity(stripes, smoothed)

    return StripeDetectionResult(
        stripes=stripes,
        profile=profile,
        positions=positions,
        gradient=grad1,
        gradient2=grad2,
        rising_edges=rising,
        falling_edges=falling,
    )

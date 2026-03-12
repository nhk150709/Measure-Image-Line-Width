"""
stripe_detector.py — Automatically detect white (photo resist) and black (spacing) stripes.

Works on an angle-corrected ROI where stripes run vertically (profile is horizontal).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.signal import find_peaks, peak_widths


@dataclass
class Stripe:
    """Represents one detected stripe (white or black)."""
    kind: str              # "white" | "black"
    center_px: float       # center position in pixels (along profile axis)
    left_edge_px: float    # left boundary in pixels
    right_edge_px: float   # right boundary in pixels
    width_px: float        # = right - left
    peak_intensity: float  # peak (max for white, min for black)

    @property
    def width_um(self) -> float:
        """Computed externally after calibration is applied."""
        raise AttributeError("Call stripe.width_px * calibration.um_per_px instead")


@dataclass
class StripeDetectionResult:
    stripes: list[Stripe] = field(default_factory=list)
    profile: np.ndarray = field(default_factory=lambda: np.array([]))
    positions: np.ndarray = field(default_factory=lambda: np.array([]))
    angle_deg: float = 0.0

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


def _avg_intensity(smoothed: np.ndarray, left: float, right: float, pk: int) -> float:
    """Return mean profile intensity over the stripe region [left, right]."""
    l_int = max(0, int(left))
    r_int = min(len(smoothed), int(right) + 1)
    if l_int >= r_int:
        return float(smoothed[pk])
    return float(np.mean(smoothed[l_int:r_int]))


def _merge_two(a: Stripe, b: Stripe) -> Stripe:
    """Merge two same-kind stripes into one stripe spanning both regions."""
    left = min(a.left_edge_px, b.left_edge_px)
    right = max(a.right_edge_px, b.right_edge_px)
    width = right - left
    center = (left + right) / 2.0
    # Width-weighted average intensity
    avg_intensity = (
        (a.peak_intensity * a.width_px + b.peak_intensity * b.width_px)
        / (a.width_px + b.width_px)
    )
    return Stripe(
        kind=a.kind,
        center_px=center,
        left_edge_px=left,
        right_edge_px=right,
        width_px=width,
        peak_intensity=avg_intensity,
    )


def _enforce_alternating(candidates: list[Stripe]) -> list[Stripe]:
    """
    Ensure stripes alternate black/white.

    When two consecutive stripes share the same kind, merge them into one
    larger stripe spanning both regions.
    """
    if not candidates:
        return []
    result: list[Stripe] = [candidates[0]]
    for current in candidates[1:]:
        prev = result[-1]
        if current.kind == prev.kind:
            result[-1] = _merge_two(prev, current)
        else:
            result.append(current)
    return result


def detect_stripes(
    profile: np.ndarray,
    positions: np.ndarray | None = None,
    threshold_fraction: float = 0.5,
    min_width_px: float = 3.0,
    smoothing_sigma: float = 2.0,
    prominence_fraction: float = 0.15,
) -> StripeDetectionResult:
    """
    Detect alternating white/black stripes from a 1D intensity profile.

    Parameters
    ----------
    profile          : 1D intensity array (averaged rows from the ROI)
    positions        : x-axis pixel positions (defaults to 0..N-1)
    threshold_fraction : fraction of intensity range for white/black separation
    min_width_px     : minimum stripe width in pixels (rejects noise)
    smoothing_sigma  : Gaussian sigma for smoothing before peak finding
    prominence_fraction : min peak prominence as fraction of intensity range

    Returns
    -------
    StripeDetectionResult with detected stripes list
    """
    if positions is None:
        positions = np.arange(len(profile), dtype=float)

    n = len(profile)
    if n < 10:
        return StripeDetectionResult(profile=profile, positions=positions)

    # Smooth the profile
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(profile.astype(float), max(0.5, smoothing_sigma))

    lo, hi = float(smoothed.min()), float(smoothed.max())
    contrast = hi - lo
    if contrast < 1:
        return StripeDetectionResult(profile=profile, positions=positions)

    prominence = max(5.0, prominence_fraction * contrast)

    # ── Detect white stripes (peaks) and black stripes (valleys) ──────
    white_peaks, _ = find_peaks(smoothed,  prominence=prominence, width=min_width_px)
    black_peaks, _ = find_peaks(-smoothed, prominence=prominence, width=min_width_px)

    # ── Collect all candidates with their regions ──────────────────────
    candidates: list[Stripe] = []

    if len(white_peaks) > 0:
        _, _, left_ips, right_ips = peak_widths(
            smoothed, white_peaks, rel_height=1 - threshold_fraction
        )
        for i, pk in enumerate(white_peaks):
            left = float(left_ips[i])
            right = float(right_ips[i])
            w = right - left
            if w < min_width_px:
                continue
            candidates.append(Stripe(
                kind="white",
                center_px=float(pk),
                left_edge_px=left,
                right_edge_px=right,
                width_px=w,
                peak_intensity=_avg_intensity(smoothed, left, right, pk),
            ))

    if len(black_peaks) > 0:
        _, _, left_ips, right_ips = peak_widths(
            -smoothed, black_peaks, rel_height=1 - threshold_fraction
        )
        for i, pk in enumerate(black_peaks):
            left = float(left_ips[i])
            right = float(right_ips[i])
            w = right - left
            if w < min_width_px:
                continue
            candidates.append(Stripe(
                kind="black",
                center_px=float(pk),
                left_edge_px=left,
                right_edge_px=right,
                width_px=w,
                peak_intensity=_avg_intensity(smoothed, left, right, pk),
            ))

    # ── Sort all candidates by position ───────────────────────────────
    candidates.sort(key=lambda s: s.center_px)

    if not candidates:
        return StripeDetectionResult(profile=profile, positions=positions)

    # ── Reclassify by actual average intensity ─────────────────────────
    # Stripes above the median average intensity are white; below are black.
    # This prevents mislabelling caused by independent peak/valley detection.
    intensity_threshold = float(np.median([s.peak_intensity for s in candidates]))
    for s in candidates:
        s.kind = "white" if s.peak_intensity >= intensity_threshold else "black"

    # ── Enforce strict alternation ─────────────────────────────────────
    stripes = _enforce_alternating(candidates)

    return StripeDetectionResult(stripes=stripes, profile=profile, positions=positions)

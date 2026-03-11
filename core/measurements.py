"""
measurements.py — Compute CD, pitch, LER, LWR, LCDU from stripe detection results.

All outputs are in µm (caller provides Calibration object).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .calibration import Calibration
from .stripe_detector import StripeDetectionResult, Stripe


@dataclass
class CDStats:
    """Statistics for a set of CD (critical dimension) measurements."""
    values_um: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.values_um)

    @property
    def mean(self) -> float | None:
        return float(np.mean(self.values_um)) if self.values_um else None

    @property
    def std(self) -> float | None:
        return float(np.std(self.values_um, ddof=1)) if len(self.values_um) > 1 else 0.0

    @property
    def minimum(self) -> float | None:
        return float(np.min(self.values_um)) if self.values_um else None

    @property
    def maximum(self) -> float | None:
        return float(np.max(self.values_um)) if self.values_um else None

    def to_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}mean_um": self.mean,
            f"{prefix}std_um": self.std,
            f"{prefix}min_um": self.minimum,
            f"{prefix}max_um": self.maximum,
            f"{prefix}n": self.n,
        }


@dataclass
class RoughnessStats:
    """LER / LWR statistics."""
    left_edge_positions_um: list[float] = field(default_factory=list)
    right_edge_positions_um: list[float] = field(default_factory=list)

    @property
    def LER_left_3sigma(self) -> float | None:
        if len(self.left_edge_positions_um) < 3:
            return None
        return float(3 * np.std(self.left_edge_positions_um, ddof=1))

    @property
    def LER_right_3sigma(self) -> float | None:
        if len(self.right_edge_positions_um) < 3:
            return None
        return float(3 * np.std(self.right_edge_positions_um, ddof=1))

    @property
    def LWR_3sigma(self) -> float | None:
        if len(self.left_edge_positions_um) < 3 or len(self.right_edge_positions_um) < 3:
            return None
        n = min(len(self.left_edge_positions_um), len(self.right_edge_positions_um))
        widths = [
            self.right_edge_positions_um[i] - self.left_edge_positions_um[i]
            for i in range(n)
        ]
        return float(3 * np.std(widths, ddof=1))

    def psd(self, sampling_um: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Compute Power Spectral Density of left edge positions."""
        if len(self.left_edge_positions_um) < 4:
            return np.array([]), np.array([])
        arr = np.array(self.left_edge_positions_um) - np.mean(self.left_edge_positions_um)
        n = len(arr)
        fft_vals = np.fft.rfft(arr)
        psd_vals = (np.abs(fft_vals) ** 2) / n
        freqs = np.fft.rfftfreq(n, d=sampling_um)
        return freqs, psd_vals

    def to_dict(self) -> dict:
        return {
            "LER_left_3sigma_um": self.LER_left_3sigma,
            "LER_right_3sigma_um": self.LER_right_3sigma,
            "LWR_3sigma_um": self.LWR_3sigma,
            "n_edge_samples": len(self.left_edge_positions_um),
        }


@dataclass
class MeasurementResult:
    white_cd: CDStats = field(default_factory=CDStats)
    black_cd: CDStats = field(default_factory=CDStats)
    pitch: CDStats = field(default_factory=CDStats)
    roughness: RoughnessStats = field(default_factory=RoughnessStats)
    n_stripes_measured: int = 0
    angle_deg: float = 0.0
    scale_um_per_px: float = 1.0

    def to_dict(self, image_file: str = "") -> dict:
        d = {"image_file": image_file}
        d.update(self.white_cd.to_dict("white_cd_"))
        d.update(self.black_cd.to_dict("black_cd_"))
        d.update(self.pitch.to_dict("pitch_"))
        d.update(self.roughness.to_dict())
        d["n_stripes_measured"] = self.n_stripes_measured
        d["angle_deg"] = round(self.angle_deg, 4)
        d["scale_um_per_px"] = self.scale_um_per_px
        return d


# ──────────────────────────────────────────────────────────────────────

def compute_measurements_from_detection(
    detection: StripeDetectionResult,
    calibration: Calibration,
    image: np.ndarray | None = None,
    roi: tuple[int, int, int, int] | None = None,
    edge_method: str = "canny",
    edge_threshold_fraction: float = 0.5,
) -> MeasurementResult:
    """
    Compute all measurements from a StripeDetectionResult.

    image + roi are optional: if provided, LER/LWR are computed by analyzing
    edge positions across multiple horizontal scan lines within the ROI.
    """
    cal = calibration

    # ── CD from auto-detected stripes ────────────────────────────────
    white_widths = [s.width_px * cal.um_per_px for s in detection.white_stripes]
    black_widths = [s.width_px * cal.um_per_px for s in detection.black_stripes]

    # Pitch from white-center spacing
    white_centers = sorted(s.center_px for s in detection.white_stripes)
    pitches_um = [
        (white_centers[i + 1] - white_centers[i]) * cal.um_per_px
        for i in range(len(white_centers) - 1)
    ]

    result = MeasurementResult(
        white_cd=CDStats(white_widths),
        black_cd=CDStats(black_widths),
        pitch=CDStats(pitches_um),
        n_stripes_measured=len(detection.stripes),
        angle_deg=detection.angle_deg,
        scale_um_per_px=cal.um_per_px,
    )

    # ── LER / LWR from per-row edge analysis ─────────────────────────
    if image is not None and len(detection.white_stripes) > 0:
        result.roughness = _compute_ler_lwr(
            image, roi, detection, calibration, edge_method, edge_threshold_fraction
        )

    return result


def compute_per_stripe_roughness(
    detection: StripeDetectionResult,
    calibration: Calibration,
    image: np.ndarray,
    roi: tuple | None,
    edge_method: str,
    edge_threshold_fraction: float,
    direction: str = "horizontal",
) -> list[RoughnessStats]:
    """
    Compute LER/LWR independently for every stripe in *detection*.

    direction='horizontal': vertical stripes — sample rows, edges are x-positions.
    direction='vertical':   horizontal stripes — sample columns, edges are y-positions.

    Returns a list of RoughnessStats aligned with detection.stripes.
    """
    from .edge_detector import find_edges

    if roi is not None:
        x, y, w, h = roi
        region = image[y: y + h, x: x + w]
    else:
        region = image

    # For horizontal direction sample along rows; for vertical along columns.
    if direction == "horizontal":
        n = region.shape[0]
        def get_profile(idx: int) -> np.ndarray:
            return region[idx, :].astype(float)
    else:
        n = region.shape[1]
        def get_profile(idx: int) -> np.ndarray:
            return region[:, idx].astype(float)

    step = max(1, n // 100)

    results = []
    for stripe in detection.stripes:
        left_edges: list[float] = []
        right_edges: list[float] = []
        for idx in range(0, n, step):
            profile = get_profile(idx)
            edges = find_edges(
                profile,
                method=edge_method,
                threshold_fraction=edge_threshold_fraction,
            )
            if len(edges) < 2:
                continue
            center = stripe.center_px
            lefts = [e for e in edges if e < center]
            rights = [e for e in edges if e >= center]
            if lefts and rights:
                best_left = min(lefts, key=lambda e: abs(e - stripe.left_edge_px))
                best_right = min(rights, key=lambda e: abs(e - stripe.right_edge_px))
                left_edges.append(best_left * calibration.um_per_px)
                right_edges.append(best_right * calibration.um_per_px)
        results.append(RoughnessStats(left_edges, right_edges))
    return results


def collect_per_stripe_ler_points(
    detection: StripeDetectionResult,
    image: np.ndarray,
    roi: tuple | None,
    edge_method: str,
    edge_threshold_fraction: float,
    direction: str = "horizontal",
) -> list[tuple[list[tuple[float, float]], list[tuple[float, float]]]]:
    """
    Collect LER sample point coordinates in image space for visualisation.

    Returns a list aligned with detection.stripes.  Each entry is
    (left_points, right_points) where every point is an (x, y) pixel
    coordinate relative to the full image (ROI offset already applied).

    direction='horizontal': vertical stripes — scans across rows, (edge_x, row_y).
    direction='vertical':   horizontal stripes — scans across cols, (col_x, edge_y).
    """
    from .edge_detector import find_edges

    roi_x = roi[0] if roi else 0
    roi_y = roi[1] if roi else 0

    if roi is not None:
        x, y, w, h = roi
        region = image[y: y + h, x: x + w]
    else:
        region = image

    if direction == "horizontal":
        n = region.shape[0]
        def get_profile(idx: int) -> np.ndarray:
            return region[idx, :].astype(float)
        def to_image_xy(sample_idx: int, edge_pos: float) -> tuple[float, float]:
            return (edge_pos + roi_x, sample_idx + roi_y)
    else:
        n = region.shape[1]
        def get_profile(idx: int) -> np.ndarray:
            return region[:, idx].astype(float)
        def to_image_xy(sample_idx: int, edge_pos: float) -> tuple[float, float]:
            return (sample_idx + roi_x, edge_pos + roi_y)

    step = max(1, n // 100)

    results = []
    for stripe in detection.stripes:
        left_pts: list[tuple[float, float]] = []
        right_pts: list[tuple[float, float]] = []
        for idx in range(0, n, step):
            profile = get_profile(idx)
            edges = find_edges(
                profile,
                method=edge_method,
                threshold_fraction=edge_threshold_fraction,
            )
            if len(edges) < 2:
                continue
            center = stripe.center_px
            lefts = [e for e in edges if e < center]
            rights = [e for e in edges if e >= center]
            if lefts and rights:
                best_left = min(lefts, key=lambda e: abs(e - stripe.left_edge_px))
                best_right = min(rights, key=lambda e: abs(e - stripe.right_edge_px))
                left_pts.append(to_image_xy(idx, best_left))
                right_pts.append(to_image_xy(idx, best_right))
        results.append((left_pts, right_pts))
    return results


def _compute_ler_lwr(
    image: np.ndarray,
    roi: tuple | None,
    detection: StripeDetectionResult,
    calibration: Calibration,
    edge_method: str,
    edge_threshold_fraction: float,
) -> RoughnessStats:
    """
    For each row in the ROI, find edges of the first well-defined white stripe,
    collect left/right edge positions → compute LER/LWR.
    """
    from .edge_detector import find_edges

    if roi is not None:
        x, y, w, h = roi
        region = image[y: y + h, x: x + w]
    else:
        x, y = 0, 0
        region = image

    # Use the widest white stripe as reference
    if not detection.white_stripes:
        return RoughnessStats()
    ref_stripe = max(detection.white_stripes, key=lambda s: s.width_px)

    left_edges = []
    right_edges = []

    n_rows = region.shape[0]
    step = max(1, n_rows // 100)  # sample up to 100 rows

    for row_idx in range(0, n_rows, step):
        row_profile = region[row_idx, :].astype(float)
        edges = find_edges(
            row_profile,
            method=edge_method,
            threshold_fraction=edge_threshold_fraction,
        )
        if len(edges) < 2:
            continue

        # Find edge pair closest to reference stripe
        stripe_l = ref_stripe.left_edge_px
        stripe_r = ref_stripe.right_edge_px

        # Find closest left edge (to the left of center)
        center = ref_stripe.center_px
        lefts = [e for e in edges if e < center]
        rights = [e for e in edges if e >= center]

        if lefts and rights:
            best_left = min(lefts, key=lambda e: abs(e - stripe_l))
            best_right = min(rights, key=lambda e: abs(e - stripe_r))
            left_edges.append(best_left * calibration.um_per_px)
            right_edges.append(best_right * calibration.um_per_px)

    return RoughnessStats(
        left_edge_positions_um=left_edges,
        right_edge_positions_um=right_edges,
    )

"""
recipe.py — Save and load measurement recipes as JSON.

A recipe captures all parameters needed to reproducibly measure stripe widths:
scale, ROI, preprocessing, angle detection, edge method, thresholds, etc.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict


@dataclass
class Recipe:
    # ── Calibration ───────────────────────────────────────────────────
    scale_um_per_px: float = 0.1
    scale_source: str = "manual"          # "manual" | "metadata" | "scalebar"

    # ── ROI (None means full image) ───────────────────────────────────
    roi: dict | None = None               # {"x": 0, "y": 0, "w": 512, "h": 512}

    # ── Angle ─────────────────────────────────────────────────────────
    angle_mode: str = "hough"             # "hough" | "fft" | "manual"
    angle_offset_deg: float = 0.0

    # ── Preprocessing ─────────────────────────────────────────────────
    filter_type: str = "gaussian"         # "gaussian" | "median" | "bilateral" | "none"
    filter_sigma: float = 2.0
    contrast_enhance: str = "none"        # "none" | "clahe" | "histogram_eq" | "normalize"

    # ── Crop (pixels removed from each side, applied before display / analysis)
    crop_top_px: int = 0
    crop_bottom_px: int = 0
    crop_left_px: int = 0
    crop_right_px: int = 0

    # ── Edge / Stripe detection ────────────────────────────────────────
    # Detection method:
    #   "gradient_peaks"  – peaks of 1st derivative; best for sharp SEM edges
    #   "zero_crossing"   – zero-crossings of 2nd derivative; inflection points
    edge_detect_method: str = "gradient_peaks"

    # Pairing mode (determines what constitutes one stripe interval):
    #   "rising_falling"  – B→W then W→B  → white stripe CD
    #   "falling_rising"  – W→B then B→W  → black stripe / space CD
    #   "rising_rising"   – B→W then B→W  → full pitch
    #   "falling_falling" – W→B then W→B  → full pitch
    edge_pairing: str = "rising_falling"

    # Minimum distance between consecutive edges (in µm).
    # Edges closer than this are suppressed.  Raise if sub-stripe spurious
    # edges appear (e.g. from halo artefacts within a stripe).
    min_edge_distance_um: float = 0.05

    # Profile / smoothing
    smoothing_sigma: float = 2.0
    prominence_fraction: float = 0.15    # min edge prominence (fraction of contrast)
    profile_lines: int = 20
    profile_direction: str = "horizontal"  # "horizontal" | "vertical"

    # ── Derivative plot ────────────────────────────────────────────────
    # 0 = hidden, 1 = 1st derivative, 2 = 2nd derivative
    show_derivative_order: int = 1

    # ── LER / edge detection (for roughness measurements) ─────────────
    edge_method: str = "canny"            # "threshold" | "canny" | "sigmoid"
    threshold_fraction: float = 0.5       # threshold used by LER edge finder

    # ── Output ────────────────────────────────────────────────────────
    name: str = "default"
    notes: str = ""

    # ── I/O ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Recipe":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Only pass known fields to avoid breakage on version mismatch
        valid = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    @classmethod
    def default_path_for_directory(cls, directory: str) -> str:
        dirname = os.path.basename(os.path.normpath(directory))
        return os.path.join(directory, f"{dirname}_recipe.json")

    # ── Convenience ──────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def roi_tuple(self) -> tuple[int, int, int, int] | None:
        """Return ROI as (x, y, w, h) tuple or None."""
        if self.roi is None:
            return None
        return (self.roi["x"], self.roi["y"], self.roi["w"], self.roi["h"])

    def set_roi_from_tuple(self, x: int, y: int, w: int, h: int) -> None:
        self.roi = {"x": x, "y": y, "w": w, "h": h}

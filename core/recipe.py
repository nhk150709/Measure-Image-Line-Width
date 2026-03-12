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
    # Calibration
    scale_um_per_px: float = 0.1
    scale_source: str = "manual"  # "manual" | "metadata" | "scalebar"

    # ROI (None means full image)
    roi: dict | None = None  # {"x": 0, "y": 0, "w": 512, "h": 512}

    # Angle
    angle_mode: str = "hough"   # "hough" | "fft" | "manual"
    angle_offset_deg: float = 0.0

    # Preprocessing
    filter_type: str = "gaussian"    # "gaussian" | "median" | "bilateral" | "none"
    filter_sigma: float = 2.0
    contrast_enhance: str = "none"   # "none" | "clahe" | "histogram_eq" | "normalize"

    # Crop (pixels removed from each side before analysis)
    crop_x_px: int = 0  # pixels removed from left and right
    crop_y_px: int = 0  # pixels removed from top and bottom

    # Stripe detection
    threshold_fraction: float = 0.5
    min_stripe_width_px: float = 5.0
    smoothing_sigma: float = 2.0
    profile_lines: int = 20
    profile_direction: str = "horizontal"  # "horizontal" | "vertical"
    min_valley_depth_fraction: float = 0.4  # halo artefact suppression (0 = off)

    # Edge detection
    edge_method: str = "canny"  # "threshold" | "canny" | "sigmoid"

    # Output
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

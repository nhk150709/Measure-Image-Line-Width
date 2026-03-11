"""
calibration.py — Pixel ↔ physical unit (µm) conversion.
"""
from __future__ import annotations

DEFAULT_UM_PER_PX = 1.0  # fallback: 1 px = 1 µm (shows warning)


class Calibration:
    """Stores the scale factor and converts between pixels and µm."""

    def __init__(self, um_per_px: float = DEFAULT_UM_PER_PX, source: str = "default"):
        if um_per_px <= 0:
            raise ValueError("um_per_px must be positive")
        self._um_per_px = um_per_px
        self.source = source  # "manual" | "metadata" | "scalebar" | "default"

    # ------------------------------------------------------------------
    @property
    def um_per_px(self) -> float:
        return self._um_per_px

    @um_per_px.setter
    def um_per_px(self, value: float):
        if value <= 0:
            raise ValueError("um_per_px must be positive")
        self._um_per_px = value

    @property
    def px_per_um(self) -> float:
        return 1.0 / self._um_per_px

    # ------------------------------------------------------------------
    def px_to_um(self, pixels: float) -> float:
        return pixels * self._um_per_px

    def um_to_px(self, um: float) -> float:
        return um / self._um_per_px

    def px_array_to_um(self, arr):
        """Convert numpy array of pixel distances to µm."""
        import numpy as np
        return np.asarray(arr, dtype=float) * self._um_per_px

    # ------------------------------------------------------------------
    @classmethod
    def from_manual(cls, um_per_px: float) -> "Calibration":
        return cls(um_per_px, source="manual")

    @classmethod
    def from_metadata(cls, um_per_px: float) -> "Calibration":
        return cls(um_per_px, source="metadata")

    @classmethod
    def from_scalebar(cls, bar_length_px: float, bar_length_um: float) -> "Calibration":
        """User draws a line over a known scale bar feature."""
        if bar_length_px <= 0:
            raise ValueError("bar_length_px must be positive")
        return cls(bar_length_um / bar_length_px, source="scalebar")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"um_per_px": self._um_per_px, "source": self.source}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(d["um_per_px"], source=d.get("source", "manual"))

    def __repr__(self):
        return f"Calibration({self._um_per_px:.4f} µm/px, source={self.source!r})"

"""
preprocessor.py — Image preprocessing: filtering, contrast enhancement.
"""
from __future__ import annotations

import cv2
import numpy as np


class Preprocessor:
    """Apply a chain of preprocessing steps to a grayscale image."""

    def __init__(
        self,
        filter_type: str = "gaussian",   # "gaussian" | "median" | "bilateral" | "none"
        filter_sigma: float = 2.0,        # Gaussian sigma or bilateral color sigma
        filter_ksize: int = 0,            # 0 = auto from sigma
        contrast_enhance: str = "none",   # "none" | "clahe" | "histogram_eq" | "normalize"
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
    ):
        self.filter_type = filter_type
        self.filter_sigma = filter_sigma
        self.filter_ksize = filter_ksize
        self.contrast_enhance = contrast_enhance
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid

    # ------------------------------------------------------------------
    def process(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline; returns uint8 grayscale."""
        img = image.copy()
        img = self._apply_filter(img)
        img = self._apply_contrast(img)
        return img

    # ------------------------------------------------------------------
    def _apply_filter(self, img: np.ndarray) -> np.ndarray:
        if self.filter_type == "gaussian":
            ksize = self.filter_ksize if self.filter_ksize > 0 else 0
            return cv2.GaussianBlur(img, (ksize, ksize), self.filter_sigma)
        elif self.filter_type == "median":
            ksize = max(3, int(self.filter_sigma * 2 + 1) | 1)
            return cv2.medianBlur(img, ksize)
        elif self.filter_type == "bilateral":
            d = max(5, int(self.filter_sigma * 3) | 1)
            return cv2.bilateralFilter(img, d, self.filter_sigma * 10, self.filter_sigma * 10)
        return img  # "none"

    def _apply_contrast(self, img: np.ndarray) -> np.ndarray:
        if self.contrast_enhance == "clahe":
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip,
                tileGridSize=(self.clahe_grid, self.clahe_grid),
            )
            return clahe.apply(img)
        elif self.contrast_enhance == "histogram_eq":
            return cv2.equalizeHist(img)
        elif self.contrast_enhance == "normalize":
            return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return img  # "none"

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "filter_type": self.filter_type,
            "filter_sigma": self.filter_sigma,
            "filter_ksize": self.filter_ksize,
            "contrast_enhance": self.contrast_enhance,
            "clahe_clip": self.clahe_clip,
            "clahe_grid": self.clahe_grid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})

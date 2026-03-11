"""
batch_processor.py — Apply a saved recipe to all images in a directory.

Emits Qt signals for progress tracking when used from the GUI, or runs
synchronously when called from the command line.
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import Callable

import pandas as pd

from core.image_loader import load_image, list_images_in_directory
from core.calibration import Calibration
from core.preprocessor import Preprocessor
from core.angle_detector import detect_angle, rotate_image
from core.profile_extractor import extract_averaged_profile
from core.stripe_detector import detect_stripes
from core.measurements import compute_measurements_from_detection, MeasurementResult
from core.recipe import Recipe
from export.csv_exporter import export_results_csv


class BatchProcessor:
    """
    Applies a Recipe to every image in a directory.

    Usage (headless):
        bp = BatchProcessor(recipe)
        results = bp.run(image_dir, output_csv)

    Usage (GUI): subclass and override on_progress / on_error,
    or pass callbacks.
    """

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(
        self,
        image_dir: str,
        output_csv: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
    ) -> pd.DataFrame:
        """
        Process all images in image_dir with the stored recipe.

        Parameters
        ----------
        image_dir        : directory containing images
        output_csv       : if given, write results to this path after processing
        progress_callback: called as (current_index, total, filename)
        error_callback   : called as (filename, error_message) on per-image errors

        Returns
        -------
        pd.DataFrame of all measurement results
        """
        self._cancel = False
        image_paths = list_images_in_directory(image_dir)
        total = len(image_paths)
        rows = []

        recipe = self.recipe
        calibration = Calibration.from_dict(
            {"um_per_px": recipe.scale_um_per_px, "source": recipe.scale_source}
        )
        preprocessor = Preprocessor(
            filter_type=recipe.filter_type,
            filter_sigma=recipe.filter_sigma,
            contrast_enhance=recipe.contrast_enhance,
        )

        for idx, path in enumerate(image_paths):
            if self._cancel:
                break

            fname = os.path.basename(path)
            if progress_callback:
                progress_callback(idx, total, fname)

            try:
                row = self._process_one(path, recipe, calibration, preprocessor)
                rows.append(row)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                if error_callback:
                    error_callback(fname, msg)
                else:
                    print(f"[ERROR] {fname}: {msg}")
                    traceback.print_exc()

        if progress_callback:
            progress_callback(total, total, "Done")

        df = pd.DataFrame(rows) if rows else pd.DataFrame()

        if output_csv and not df.empty:
            export_results_csv(df, output_csv)

        return df

    # ──────────────────────────────────────────────────────────────────

    def _process_one(
        self,
        path: str,
        recipe: Recipe,
        calibration: Calibration,
        preprocessor: Preprocessor,
    ) -> dict:
        """Process a single image and return a result dict."""
        img_data = load_image(path)

        # Override calibration if image has embedded metadata
        if img_data.metadata_um_per_px is not None and recipe.scale_source == "metadata":
            calibration = Calibration.from_metadata(img_data.metadata_um_per_px)

        processed = preprocessor.process(img_data.pixels)

        # Apply ROI
        roi = recipe.roi_tuple

        # Detect angle
        if roi is not None:
            x, y, w, h = roi
            roi_pixels = processed[y: y + h, x: x + w]
        else:
            roi_pixels = processed

        angle_deg = detect_angle(
            roi_pixels,
            method=recipe.angle_mode,
            user_angle=recipe.angle_offset_deg if recipe.angle_mode == "manual" else None,
        ) + recipe.angle_offset_deg

        rotated = rotate_image(processed, angle_deg)

        # Extract profile
        positions, profile = extract_averaged_profile(
            rotated,
            roi=roi,
            n_lines=recipe.profile_lines,
            direction=recipe.profile_direction,
        )

        # Detect stripes
        detection = detect_stripes(
            profile,
            positions=positions,
            threshold_fraction=recipe.threshold_fraction,
            min_width_px=recipe.min_stripe_width_px,
            smoothing_sigma=recipe.smoothing_sigma,
        )
        detection.angle_deg = angle_deg

        # Compute measurements
        result = compute_measurements_from_detection(
            detection,
            calibration,
            image=rotated,
            roi=roi,
            edge_method=recipe.edge_method,
            edge_threshold_fraction=recipe.threshold_fraction,
        )

        row = result.to_dict(image_file=os.path.basename(path))
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")
        row["recipe_name"] = recipe.name
        return row

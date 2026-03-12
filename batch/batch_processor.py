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
from core.stripe_detector import detect_stripes, StripeDetectionResult
from core.measurements import (
    compute_measurements_from_detection,
    compute_per_stripe_roughness,
    collect_per_stripe_ler_points,
    MeasurementResult,
)
from core.recipe import Recipe
from export.csv_exporter import export_results_csv, export_stripe_rows_csv
from export.annotated_image import save_annotated_image


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
        annotated_dir: str | None = None,
        output_stripes_csv: str | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        error_callback: Callable[[str, str], None] | None = None,
    ) -> pd.DataFrame:
        """
        Process all images in image_dir with the stored recipe.

        Parameters
        ----------
        image_dir          : directory containing images
        output_csv         : if given, write summary results to this path
        annotated_dir      : if given, save annotated PNG per image here
        output_stripes_csv : if given, write per-stripe rows to this path
        progress_callback  : called as (current_index, total, filename)
        error_callback     : called as (filename, error_message) on per-image errors

        Returns
        -------
        pd.DataFrame of all summary measurement results
        """
        self._cancel = False
        image_paths = list_images_in_directory(image_dir)
        total = len(image_paths)
        rows: list[dict] = []
        stripe_rows: list[dict] = []

        if annotated_dir:
            os.makedirs(annotated_dir, exist_ok=True)

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
                row, per_stripe, rotated, detection, roi, cal, ler_points = self._process_one(
                    path, recipe, calibration, preprocessor
                )
                rows.append(row)
                stripe_rows.extend(per_stripe)

                if annotated_dir:
                    stem = os.path.splitext(fname)[0]
                    out_path = os.path.join(annotated_dir, f"{stem}_annotated.png")
                    save_annotated_image(
                        rotated, detection, roi, out_path,
                        um_per_px=cal.um_per_px,
                        direction=recipe.profile_direction,
                        ler_points=ler_points,
                    )

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

        stripe_df = pd.DataFrame(stripe_rows) if stripe_rows else pd.DataFrame()
        if output_stripes_csv and not stripe_df.empty:
            export_stripe_rows_csv(stripe_df, output_stripes_csv)

        return df

    # ──────────────────────────────────────────────────────────────────

    def _process_one(
        self,
        path: str,
        recipe: Recipe,
        calibration: Calibration,
        preprocessor: Preprocessor,
    ) -> tuple[dict, list[dict], object, StripeDetectionResult, tuple | None, Calibration, list]:
        """Process a single image and return (summary_row, stripe_rows, rotated, detection, roi, cal, ler_points)."""
        img_data = load_image(path)

        # Override calibration if image has embedded metadata
        cal = calibration
        if img_data.metadata_um_per_px is not None and recipe.scale_source == "metadata":
            cal = Calibration.from_metadata(img_data.metadata_um_per_px)

        processed = preprocessor.process(img_data.pixels)

        # Physical 4-directional crop
        t, b, l, r = (recipe.crop_top_px, recipe.crop_bottom_px,
                      recipe.crop_left_px, recipe.crop_right_px)
        if t > 0 or b > 0 or l > 0 or r > 0:
            ph, pw = processed.shape[:2]
            y1, y2 = t, (ph - b if b > 0 else ph)
            x1, x2 = l, (pw - r if r > 0 else pw)
            if y1 < y2 and x1 < x2:
                processed = processed[y1:y2, x1:x2]

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

        # Detect stripes (gradient-based)
        min_edge_px = recipe.min_edge_distance_um / max(recipe.scale_um_per_px, 1e-9)
        detection = detect_stripes(
            profile,
            positions=positions,
            smoothing_sigma=recipe.smoothing_sigma,
            prominence_fraction=recipe.prominence_fraction,
            edge_detect_method=recipe.edge_detect_method,
            edge_pairing=recipe.edge_pairing,
            min_edge_distance_px=min_edge_px,
        )
        detection.angle_deg = angle_deg

        # Compute summary measurements
        result = compute_measurements_from_detection(
            detection, cal,
            image=rotated, roi=roi,
            edge_method=recipe.edge_method,
            edge_threshold_fraction=recipe.threshold_fraction,
        )

        timestamp = datetime.now().isoformat(timespec="seconds")
        fname = os.path.basename(path)

        summary_row = result.to_dict(image_file=fname)
        summary_row["timestamp"] = timestamp
        summary_row["recipe_name"] = recipe.name

        # Per-stripe roughness and LER sample points for annotation
        per_stripe_roughness = compute_per_stripe_roughness(
            detection, cal, rotated, roi,
            edge_method=recipe.edge_method,
            edge_threshold_fraction=recipe.threshold_fraction,
            direction=recipe.profile_direction,
        )
        ler_points = collect_per_stripe_ler_points(
            detection, rotated, roi,
            edge_method=recipe.edge_method,
            edge_threshold_fraction=recipe.threshold_fraction,
            direction=recipe.profile_direction,
        )

        stripe_rows: list[dict] = []
        for i, stripe in enumerate(detection.stripes):
            srow = {
                "image_file": fname,
                "stripe_index": i,
                "kind": stripe.kind,
                "width_um": stripe.width_px * cal.um_per_px,
                "left_edge_um": stripe.left_edge_px * cal.um_per_px,
                "right_edge_um": stripe.right_edge_px * cal.um_per_px,
                "center_um": stripe.center_px * cal.um_per_px,
                "recipe_name": recipe.name,
                "timestamp": timestamp,
            }
            if i < len(per_stripe_roughness):
                srow.update(per_stripe_roughness[i].to_dict())
            stripe_rows.append(srow)

        return summary_row, stripe_rows, rotated, detection, roi, cal, ler_points

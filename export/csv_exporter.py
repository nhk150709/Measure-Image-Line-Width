"""
csv_exporter.py — Export measurement results to CSV.
"""
from __future__ import annotations

import os
import pandas as pd


# Ordered columns for the output CSV
_COLUMN_ORDER = [
    "image_file",
    "white_cd_mean_um", "white_cd_std_um", "white_cd_min_um", "white_cd_max_um", "white_cd_n",
    "black_cd_mean_um", "black_cd_std_um", "black_cd_min_um", "black_cd_max_um", "black_cd_n",
    "pitch_mean_um", "pitch_std_um", "pitch_min_um", "pitch_max_um", "pitch_n",
    "LER_left_3sigma_um", "LER_right_3sigma_um", "LWR_3sigma_um", "n_edge_samples",
    "n_stripes_measured",
    "angle_deg", "scale_um_per_px",
    "recipe_name", "timestamp",
]


def export_results_csv(df: pd.DataFrame, path: str) -> str:
    """
    Write DataFrame to CSV with standardized column ordering.
    Returns the final output path.
    """
    # Reorder columns: known first, then any extra columns
    ordered = [c for c in _COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]

    df.to_csv(path, index=False, float_format="%.6f")
    return path


_STRIPE_COLUMN_ORDER = [
    "image_file", "stripe_index", "kind",
    "width_um", "left_edge_um", "right_edge_um", "center_um",
    "LER_left_3sigma_um", "LER_right_3sigma_um", "LWR_3sigma_um", "n_edge_samples",
    "recipe_name", "timestamp",
]


def export_stripe_rows_csv(df: pd.DataFrame, path: str) -> str:
    """Write per-stripe DataFrame to CSV with standardized column ordering."""
    ordered = [c for c in _STRIPE_COLUMN_ORDER if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    df = df[ordered + extras]
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def export_single_result(result_dict: dict, path: str) -> str:
    """Export a single measurement result dict to a CSV (or append if exists)."""
    df = pd.DataFrame([result_dict])
    if os.path.exists(path):
        existing = pd.read_csv(path)
        df = pd.concat([existing, df], ignore_index=True)
    return export_results_csv(df, path)

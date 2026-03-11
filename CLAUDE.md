# CLAUDE.md — Developer Guide for SEM Stripe Analyzer

## Commands

```bash
# Run the application
python main.py

# Install dependencies
pip install -r requirements.txt

# Generate synthetic test images
python tests/generate_test_images.py

# Run a quick smoke-test import check
python -c "from ui.main_window import MainWindow; print('OK')"
```

There is no automated test suite beyond the synthetic image generator.

---

## Architecture

The project separates the measurement pipeline (`core/`) from the GUI (`ui/`) so that `batch/batch_processor.py` can run headlessly without any Qt dependency.

### Data flow (single image)

```
load_image()
  → Preprocessor.process()        # filter + contrast
  → [crop]                        # recipe.crop_x_px / crop_y_px
  → detect_angle()                # Hough or FFT
  → rotate_image()                # make stripes axis-aligned
  → extract_averaged_profile()    # 1-D intensity profile
  → detect_stripes()              # peak/valley finding → Stripe list
  → compute_measurements_from_detection()  # CD, pitch, LER/LWR
  → compute_per_stripe_roughness()         # LER/LWR per stripe
  → draw_stripe_overlays()        # canvas OR annotated PNG
  → export_results_csv() / export_stripe_rows_csv()
```

### Key types

| Type | Module | Purpose |
|------|--------|---------|
| `Recipe` | `core/recipe.py` | All analysis parameters; JSON serialisable |
| `Stripe` | `core/stripe_detector.py` | Single detected stripe (kind, edges, width) |
| `StripeDetectionResult` | `core/stripe_detector.py` | Full list of stripes + profile |
| `MeasurementResult` | `core/measurements.py` | CD/pitch/LER/LWR statistics |
| `RoughnessStats` | `core/measurements.py` | LER_left, LER_right, LWR (3σ) |
| `CDStats` | `core/measurements.py` | mean/std/min/max for a set of widths |
| `Calibration` | `core/calibration.py` | `um_per_px` conversion |

---

## Important Conventions

### Profile direction vs. stripe orientation

`recipe.profile_direction` describes the axis along which the profile is extracted, **not** the stripe orientation:

| `profile_direction` | Stripe orientation | Profile axis |
|---------------------|-------------------|--------------|
| `"horizontal"` | Vertical stripes | X (left → right) |
| `"vertical"` | Horizontal stripes | Y (top → bottom) |

All code that draws overlays (canvas and annotated image) must respect this direction. `draw_stripe_overlays()` in both `ui/image_canvas.py` and `export/annotated_image.py` accept a `direction` parameter — always pass `recipe.profile_direction`.

### Stripe coordinate space

After `rotate_image()`, stripes are always vertical in the rotated image. `Stripe.left_edge_px` / `right_edge_px` are positions along the **profile axis**:

- `direction="horizontal"` → x-coordinates in the rotated image
- `direction="vertical"` → y-coordinates in the rotated image

The rotated image is what gets annotated and saved.

### Crop is applied before angle detection

In `_run_analysis()` (main_window.py) and `_process_one()` (batch_processor.py), the crop is applied immediately after preprocessing and **before** angle detection and ROI extraction. If crop changes image dimensions, the ROI should be validated/reset.

### LER/LWR sampling

`_compute_ler_lwr()` (aggregate, for summary CSV) uses the **widest white stripe** as the reference and samples up to 100 rows. `compute_per_stripe_roughness()` (for per-stripe CSV) runs the same edge-tracking logic independently for every stripe.

---

## Adding a New Edge Detection Method

1. Add a function `find_edges_mymethod(profile, ...) -> list[float]` in `core/edge_detector.py`
2. Add a branch in `find_edges()` dispatcher at the bottom of the same file
3. Add the string key to the `_edge_method` combobox in `ui/recipe_panel.py`

---

## Adding a New Export Column

Summary columns are ordered by `_COLUMN_ORDER` in `export/csv_exporter.py`. Per-stripe columns by `_STRIPE_COLUMN_ORDER`. Add new keys to the appropriate list and populate them in `MeasurementResult.to_dict()` (summary) or the stripe-row loop in `main_window._run_analysis()` / `batch_processor._process_one()`.

---

## GUI Overlay System

`ImageCanvas` (`ui/image_canvas.py`) maintains three item lists:

| List | Cleared by |
|------|-----------|
| `_roi_items` | `clear_roi_overlays()` |
| `_stripe_items` | `clear_stripe_overlays()` |
| `_measurement_items` | `clear_measurement_overlays()` |

`clear_overlays()` clears all three. Always append new `QGraphicsItem`s to the correct list so they can be removed selectively.

---

## Batch Processing

`BatchProcessor.run()` accepts:

| Parameter | Description |
|-----------|-------------|
| `image_dir` | Directory of images to process |
| `output_csv` | Path for summary CSV |
| `annotated_dir` | Directory for annotated PNGs (created if absent) |
| `output_stripes_csv` | Path for per-stripe CSV |
| `progress_callback` | `(current, total, filename) -> None` |
| `error_callback` | `(filename, error_msg) -> None` |

`_BatchWorker` (batch_panel.py) automatically derives `{stem}_stripes.csv` and `{dir}/annotated/` from the summary CSV path, so no extra UI fields are needed.

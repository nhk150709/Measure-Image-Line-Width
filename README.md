# SEM Stripe Analyzer

A desktop application for measuring line widths, spaces, pitch, and line-edge roughness (LER/LWR) in SEM images of periodic stripe patterns (e.g. photoresist lines, FinFETs, diffraction gratings).

---

## Features

- Interactive image viewer with pan, zoom, and ROI selection
- Automatic stripe orientation detection (Hough or FFT)
- Measurement of white (line) and black (space) widths, pitch, LER, and LWR
- Per-stripe individual measurements with roughness metrics
- Image annotation with width labels burned into exported PNG
- Batch processing of entire directories with annotated image and CSV export
- Recipe system — save/load all parameters as JSON for reproducible runs
- Calibration from manual entry, image metadata, or scale-bar drawing

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

1. **File → Open Directory** (or single image)
2. Draw an ROI on the canvas (optional but recommended)
3. Set scale (µm/px) in the Recipe panel
4. Set **Profile Direction** to match your stripe orientation
5. Click **▶ Run Analysis**
6. Export via **File → Export Summary CSV** or **File → Export Individual Stripes CSV**

---

## How Line Width and Space Width Are Measured

### Step 1 — Angle correction

The stripe angle is detected automatically using either:

- **Hough**: probabilistic Hough line transform on Canny edges — robust for straight, high-contrast stripes
- **FFT**: dominant orientation found in the 2D power spectrum — works even for low-contrast images

The image is then rotated so that all stripes are axis-aligned (vertical), regardless of the original orientation.

### Step 2 — Intensity profile extraction

An averaged 1-D intensity profile is extracted perpendicular to the stripes:

- **Horizontal stripes (profile direction = vertical)**: `n` equally-spaced columns are averaged to produce a profile along the Y axis
- **Vertical stripes (profile direction = horizontal)**: `n` equally-spaced rows are averaged to produce a profile along the X axis

Averaging many lines suppresses noise and pixel-level variation before peak detection.

### Step 3 — Peak / valley detection

The averaged profile is lightly Gaussian-smoothed, then `scipy.signal.find_peaks` is run twice:

| Pass | Target | Input |
|------|--------|-------|
| 1 | White stripes (bright lines) | smoothed profile |
| 2 | Black stripes (dark spaces) | inverted profile |

Peaks must exceed a minimum prominence (default 15% of the intensity range) and a minimum width in pixels to be accepted.

### Step 4 — Edge positions and width

Edge positions are derived using `scipy.signal.peak_widths` at a user-controlled threshold fraction (default 0.5 = 50% of peak height above the surrounding baseline). This gives sub-pixel `left_edge_px` and `right_edge_px` for each stripe:

```
width_px  = right_edge_px − left_edge_px
width_µm  = width_px × scale_µm_per_px
```

**Pitch** is the center-to-center distance between consecutive white stripes:

```
pitch_µm = (center_n+1 − center_n) × scale_µm_per_px
```

---

## How Line-Edge Roughness (LER) and Line-Width Roughness (LWR) Are Measured

LER and LWR quantify how much a stripe edge deviates from a perfectly straight line when measured along its length.

### Edge tracking

For each stripe, the image region is sampled at up to 100 equally-spaced rows (or columns for horizontal stripes). At each row a 1-D intensity profile is extracted and the edge is located using one of three methods:

| Method | Description |
|--------|-------------|
| **Threshold** | Sub-pixel linear interpolation at a fixed intensity level (fraction of min–max range) |
| **Canny** | Position of the maximum absolute gradient after Savitzky–Golay smoothing |
| **Sigmoid** | Error-function (erf) curve fitted to the transition region — most accurate for smooth edges |

The detected edge position at each row is recorded in physical units (µm).

### LER (Line-Edge Roughness)

LER is reported separately for the left and right edge of each stripe as the 3σ standard deviation of the edge-position series along the stripe length:

```
LER_left  = 3 × σ(left_edge_positions_µm)
LER_right = 3 × σ(right_edge_positions_µm)
```

### LWR (Line-Width Roughness)

LWR measures the variation in the instantaneous width at each sampled row:

```
width_i = right_edge_i − left_edge_i   (for each sampled row i)
LWR     = 3 × σ(width_i)
```

When the two edges are uncorrelated: LWR ≈ √(LER_left² + LER_right²).

### Power Spectral Density (PSD)

The PSD of the left-edge position series can be displayed via **View → Show Edge PSD Plot**. It decomposes roughness by spatial frequency, useful for identifying process-induced periodicity.

---

## Output Files

### Summary CSV (`*_results.csv`)

One row per image. Key columns:

| Column | Description |
|--------|-------------|
| `white_cd_mean_um` | Mean white stripe (line) width (µm) |
| `black_cd_mean_um` | Mean black stripe (space) width (µm) |
| `pitch_mean_um` | Mean center-to-center pitch (µm) |
| `LER_left_3sigma_um` | 3σ LER of the left edge of the reference stripe |
| `LER_right_3sigma_um` | 3σ LER of the right edge |
| `LWR_3sigma_um` | 3σ LWR |
| `n_edge_samples` | Number of rows used for roughness sampling |
| `angle_deg` | Detected rotation angle applied to the image |

### Per-Stripe CSV (`*_stripes.csv`)

One row per detected stripe per image. Key columns:

| Column | Description |
|--------|-------------|
| `stripe_index` | Index in detection order (left→right or top→bottom) |
| `kind` | `white` (line) or `black` (space) |
| `width_um` | Individual stripe width (µm) |
| `left_edge_um` / `right_edge_um` | Edge positions in µm |
| `LER_left_3sigma_um` | Per-stripe left-edge roughness |
| `LER_right_3sigma_um` | Per-stripe right-edge roughness |
| `LWR_3sigma_um` | Per-stripe line-width roughness |

### Annotated Images (`annotated/`)

PNG copies of each image with coloured overlays:

- **Red bands** — white (bright) stripes / lines
- **Blue bands** — black (dark) stripes / spaces
- **Green lines** — left and right detected edge positions
- **Yellow labels** — measured width in µm at the centre of each stripe

---

## Recipe Parameters

| Parameter | Description |
|-----------|-------------|
| Scale (µm/px) | Physical scale; loaded from image metadata automatically when available |
| Crop X / Crop Y | Pixels removed from **each** side before analysis (removes noisy borders) |
| Stripe Angle | Auto (Hough), Auto (FFT), or Manual with an additional offset |
| Filter | Gaussian / median / bilateral / none — applied before angle detection |
| Profile Direction | `horizontal` = vertical stripes; `vertical` = horizontal stripes |
| Threshold fraction | Intensity level used to define stripe edges (0 = minimum, 1 = maximum) |
| Min stripe width | Rejects noise peaks narrower than this value |
| Profile lines | Number of rows/columns averaged to build the intensity profile |
| Smoothing sigma | Gaussian sigma applied to the profile before peak finding |
| Edge method | Threshold / Canny / Sigmoid — used for LER/LWR edge tracking |

---

## Batch Processing

Open **Analysis → Batch Processing…** to choose a directory, or use **Analysis → Batch Current Directory…** to prefill the currently loaded directory automatically.

For each image the processor outputs:

1. A summary row appended to `*_results.csv`
2. One row per detected stripe in `*_stripes.csv`
3. An annotated PNG saved to the `annotated/` subdirectory

---

## Project Structure

```
core/                   Pure-Python measurement pipeline (no Qt dependency)
  image_loader.py       TIFF/PNG/BMP loading, metadata extraction
  preprocessor.py       Filtering and contrast enhancement
  angle_detector.py     Hough/FFT angle detection, image rotation
  profile_extractor.py  1-D intensity profile extraction
  stripe_detector.py    Peak/valley finding, edge width computation
  edge_detector.py      Threshold/Canny/sigmoid edge methods
  measurements.py       CD statistics, LER/LWR, per-stripe roughness
  calibration.py        µm ↔ px conversion
  recipe.py             Parameter set, JSON save/load

ui/                     PyQt5 GUI layer
  main_window.py        Application window, menus, analysis wiring
  image_canvas.py       Interactive QGraphicsView with overlays
  recipe_panel.py       Recipe parameter controls
  measurement_panel.py  Results display
  profile_plot_widget.py Embedded matplotlib intensity profile plot
  batch_panel.py        Batch processing dialog

export/
  annotated_image.py    OpenCV overlay rendering, PNG export
  csv_exporter.py       Standardised CSV writing

batch/
  batch_processor.py    Headless batch runner

tests/
  generate_test_images.py  Synthetic stripe image generator
```

---

## Requirements

- Python 3.10+
- PyQt5
- opencv-python
- numpy, scipy, pandas, matplotlib, tifffile

# Microscope Line Width & LER Measurement Tool — Developer Guide

## Project Overview

A PyQt5 desktop application for measuring semiconductor/photolithography line patterns
from microscope images. Measures:

- **Dark stripe width** and **bright stripe width** (median, std, min/max per image)
- **LER** — Line Edge Roughness (3σ of edge position variations, µm)
- **LWR** — Line Width Roughness (3σ of stripe width variations, µm)
- **Pitch** — average period between same-polarity edges (µm)

## Running the Application

```bash
pip install -r requirements.txt
python3 microscope_app.py
```

## File Structure

```
microscope_app.py      # Entire application (single file, ~2360 lines)
requirements.txt       # Python dependencies
measurement_settings.json   # Auto-saved user settings (gitignored)
session_data.json           # Per-image circle/param state (gitignored)
```

## Architecture — Key Classes

| Class | Purpose |
|---|---|
| `EdgeDetector` | Static methods: `extract_profile`, `apply_algorithm`, `find_peaks` |
| `LineMeasurer` | Static methods: `detect_angle`, `measure_widths`, `compute_ler`, `compute_lwr`, `pitch` |
| `ImageProcessor` | Channel extraction + smoothing pipeline |
| `ImageEnhancement` | Brightness / contrast / gamma / CLAHE / invert |
| `MeasurementWorker` | **QObject** — all heavy computation, lives on background QThread |
| `ImageCanvas` | Custom QWidget — displays image, handles draggable/resizable circle ROI |
| `ProfileCanvas` | Matplotlib canvas — intensity profile + edge-signal graph |
| `HistogramCanvas` | Matplotlib canvas — intensity histogram of ROI pixels |
| `BatchProcessThread` | QThread — batch-measures all files with current settings |
| `CameraThread` | QThread — OpenCV live camera feed |
| `MainWindow` | Main Qt window — all UI layout, signal routing, export |

## Threading Model

```
Main Thread (Qt GUI)
  │
  ├── QTimer (debounce 120 ms) ──► _launch_measurement()
  │                                     │
  │                                     └── MeasurementWorker.run()  ← background QThread
  │                                             │ (finished signal)
  │                                     _on_measurement_done()  ← back on main thread
  │
  ├── BatchProcessThread  (independent QThread per batch run)
  └── CameraThread        (independent QThread for live camera)
```

**Rule**: Never call OpenCV, NumPy heavy ops, or matplotlib `draw()` from
signal handlers that are directly connected on the main thread without going
through the worker. All matplotlib draw calls happen in `_on_measurement_done`
which is a Qt slot (safe on main thread, but fast because data is pre-computed).

## Profile Extraction (`EdgeDetector.extract_profile`)

1. Samples N scan-lines inside the circle (capped at 32, min 7) parallel to
   the detected line angle
2. Uses `scipy.ndimage.map_coordinates` (order=1 bilinear) for fast batch
   interpolation — avoids per-pixel Python loops
3. Averages all valid scan-lines → 1-D intensity profile
4. n_samples is capped at `min(512, radius*4)` to stay fast for small circles

## Adding a New Edge Detection Algorithm

1. Add label string to `ALGORITHMS` list (top of file)
2. Add `elif algorithm == "My Algorithm":` branch in `EdgeDetector.apply_algorithm`
3. Return a 1-D numpy array same length as `profile`

## Adding a New Image Processing Step

Extend `ImageProcessor.process` or `ImageEnhancement.apply` — both take a
`params: Dict` so new keys don't break existing callers.

## Session Persistence

- **Per-image state**: circle cx/cy/radius + all param spinbox values are saved
  to `session_data.json` inside the loaded directory. Restored automatically
  when navigating back to an image.
- **Global settings**: saved to `measurement_settings.json` in the working
  directory on exit; loaded on startup.

## Export Format (CSV)

One row per measured image. Key columns:

| Column | Description |
|---|---|
| `dark_median_um` | Median dark stripe width (µm) |
| `bright_median_um` | Median bright stripe width (µm) |
| `ler_3sigma_um` | Line Edge Roughness 3σ (µm) |
| `lwr_3sigma_um` | Line Width Roughness 3σ (µm) |
| `pitch_um` | Average pitch (µm) |
| `dark_width_N_um` | Individual dark stripe widths |
| `bright_width_N_um` | Individual bright stripe widths |

## Known Constraints / Design Decisions

- **Single-file design**: everything in `microscope_app.py` for portability.
  If the file grows beyond ~3000 lines, split into `app/` package.
- **n_scan cap at 32**: prevents slowdown on very large circles / high-res images.
- **matplotlib embedded**: `ProfileCanvas` and `HistogramCanvas` use Qt5Agg
  backend. Do not call `.draw()` from background threads.
- **Camera**: uses OpenCV VideoCapture index 0 by default. To change index,
  modify `CameraThread.__init__` or expose a UI spinner.

## Dependencies

```
PyQt5 >= 5.15
numpy >= 1.21
opencv-python >= 4.5
scipy >= 1.7
matplotlib >= 3.4
pandas >= 1.3
```

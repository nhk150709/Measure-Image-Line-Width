#!/usr/bin/env python3
"""
Microscope Line Width and LER Measurement Tool

Measures:
- Width of dark and bright stripes (lines and spaces)
- Line Edge Roughness (LER) using 3-sigma method
- Supports multiple edge detection algorithms
- Interactive circle ROI with drag/resize
- Scale calibration (pixels per µm)
- Camera capture
- CSV export and labeled image save
"""

import sys
import os
import json
import math
import time
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.stats import linregress
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QTableWidget, QTableWidgetItem,
    QScrollArea, QSlider, QTabWidget, QLineEdit, QProgressBar,
    QMessageBox, QSizePolicy, QHeaderView, QAbstractItemView,
    QStatusBar, QToolBar, QAction, QButtonGroup, QRadioButton,
    QFormLayout, QGridLayout, QDockWidget, QFrame, QDialog,
    QDialogButtonBox, QTextEdit, QColorDialog
)
from PyQt5.QtCore import (
    Qt, QPoint, QRect, QSize, QTimer, QThread, pyqtSignal, QObject,
    QPointF, QRectF, QSettings
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush,
    QCursor, QTransform, QWheelEvent, QMouseEvent, QPalette, QIcon
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
APP_NAME = "Microscope Line Measurement"
SETTINGS_FILE = "measurement_settings.json"
SESSION_FILE  = "session_data.json"

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

ALGORITHMS = [
    "1st Derivative (Gradient)",
    "2nd Derivative (Laplacian)",
    "Gaussian 1st Derivative",
    "Gaussian 2nd Derivative",
    "Canny",
    "Sobel",
    "Prewitt",
    "Zero Crossing (LoG)",
]

PEAK_MODES = [
    "+ peaks",
    "- peaks",
    "+ and - peaks",
    "Zero crossings",
    "Auto (all edges)",
]

IMAGE_CHANNELS = [
    "Grayscale",
    "RGB - Red",
    "RGB - Green",
    "RGB - Blue",
    "YUV - Y (Luma)",
    "YUV - U",
    "YUV - V",
    "HSV - Hue",
    "HSV - Saturation",
    "HSV - Value",
]

SMOOTH_METHODS = [
    "None",
    "Gaussian",
    "Median",
    "Bilateral",
    "Box (Mean)",
]

DARK_BG = "#1e1e1e"
PANEL_BG = "#252526"
BUTTON_BG = "#3a3a3a"
ACCENT   = "#0e8a6e"
TEXT_FG  = "#d4d4d4"


# ─────────────────────────────────────────────────────────────────────────────
# EDGE DETECTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class EdgeDetector:
    """1-D edge detection on intensity profile extracted from circle ROI."""

    @staticmethod
    def extract_profile(
        image_gray: np.ndarray,
        cx: float, cy: float, radius: float,
        angle_deg: float,
        n_samples: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract average intensity profile perpendicular to lines.

        Parameters
        ----------
        image_gray : H×W uint8 array
        cx, cy     : circle centre in image pixels
        radius     : circle radius in pixels
        angle_deg  : line angle (degrees from horizontal)
        n_samples  : number of samples along profile

        Returns
        -------
        t       : positions along perpendicular axis  (pixels, centred on 0)
        profile : average intensity
        """
        h, w = image_gray.shape
        angle_rad = math.radians(angle_deg)

        # unit vector along lines
        lx = math.cos(angle_rad)
        ly = math.sin(angle_rad)
        # unit vector perpendicular to lines (profile direction)
        px = -ly
        py =  lx

        # Cap n_samples to avoid slowness on tiny circles
        n_samples = min(n_samples, max(64, int(radius * 4)))
        t = np.linspace(-radius, radius, n_samples)

        # Cap scan lines: enough for averaging but never > 32
        n_scan = min(32, max(7, int(radius * 0.3)))
        scan_offsets = np.linspace(-radius * 0.85, radius * 0.85, n_scan)

        # Build a (n_scan, n_samples) grid of (row, col) coordinates
        # then use map_coordinates for fast batch bilinear interpolation
        all_rows = []
        all_cols = []
        valid_scans = []

        for k in scan_offsets:
            cols = cx + t * px + k * lx   # x → col
            rows = cy + t * py + k * ly   # y → row
            in_bounds = ((cols >= 0) & (cols < w - 1) &
                         (rows >= 0) & (rows < h - 1))
            if in_bounds.sum() < n_samples * 0.4:
                continue
            all_cols.append(cols)
            all_rows.append(rows)
            valid_scans.append(True)

        if not all_rows:
            return t, np.zeros(n_samples)

        # Stack into 2-D arrays: shape (n_valid_scans, n_samples)
        coords_r = np.vstack(all_rows)   # (S, N)
        coords_c = np.vstack(all_cols)   # (S, N)

        # map_coordinates expects (ndim, npoints); flatten, then reshape
        S, N = coords_r.shape
        flat_r = coords_r.ravel()
        flat_c = coords_c.ravel()

        sampled = ndimage.map_coordinates(
            image_gray.astype(np.float64),
            [flat_r, flat_c],
            order=1,          # bilinear
            mode='nearest',
        ).reshape(S, N)

        return t, sampled.mean(axis=0)

    @staticmethod
    def apply_algorithm(
        profile: np.ndarray,
        algorithm: str,
        sigma: float = 2.0,
    ) -> np.ndarray:
        """Apply 1-D edge detection algorithm."""
        s = profile.copy()

        if algorithm == "1st Derivative (Gradient)":
            return np.gradient(s)

        elif algorithm == "2nd Derivative (Laplacian)":
            return np.gradient(np.gradient(s))

        elif algorithm == "Gaussian 1st Derivative":
            sm = ndimage.gaussian_filter1d(s, sigma)
            return np.gradient(sm)

        elif algorithm == "Gaussian 2nd Derivative":
            sm = ndimage.gaussian_filter1d(s, sigma)
            return np.gradient(np.gradient(sm))

        elif algorithm == "Canny":
            sm = ndimage.gaussian_filter1d(s, sigma)
            return np.gradient(sm)

        elif algorithm == "Sobel":
            k = np.array([-1, 0, 1], dtype=float)
            return np.convolve(s, k, mode='same')

        elif algorithm == "Prewitt":
            k = np.array([-1, 0, 1], dtype=float) / 2.0
            return np.convolve(s, k, mode='same')

        elif algorithm == "Zero Crossing (LoG)":
            sm = ndimage.gaussian_filter1d(s, sigma)
            d2 = np.gradient(np.gradient(sm))
            return d2

        return np.gradient(s)

    @staticmethod
    def find_peaks(
        edge_signal: np.ndarray,
        profile: np.ndarray,
        peak_mode: str,
        threshold_ratio: float = 0.25,
        min_distance: int = 5,
    ) -> List[int]:
        """Find edge positions according to selected mode."""
        if len(edge_signal) == 0:
            return []

        amp = np.max(np.abs(edge_signal))
        if amp < 1e-9:
            return []

        threshold = threshold_ratio * amp

        if peak_mode == "+ peaks":
            peaks, _ = signal.find_peaks(
                edge_signal, height=threshold, distance=min_distance)
            return list(peaks)

        elif peak_mode == "- peaks":
            peaks, _ = signal.find_peaks(
                -edge_signal, height=threshold, distance=min_distance)
            return list(peaks)

        elif peak_mode == "+ and - peaks":
            pos, _ = signal.find_peaks(
                 edge_signal, height=threshold, distance=min_distance)
            neg, _ = signal.find_peaks(
                -edge_signal, height=threshold, distance=min_distance)
            return sorted(list(pos) + list(neg))

        elif peak_mode == "Zero crossings":
            crossings = []
            for i in range(len(edge_signal) - 1):
                if edge_signal[i] * edge_signal[i + 1] < 0:
                    # sub-pixel interpolation
                    f0, f1 = edge_signal[i], edge_signal[i + 1]
                    frac = f0 / (f0 - f1)
                    crossings.append(int(round(i + frac)))
            return crossings

        else:  # Auto
            pos, _ = signal.find_peaks(
                 edge_signal, height=threshold, distance=min_distance)
            neg, _ = signal.find_peaks(
                -edge_signal, height=threshold, distance=min_distance)
            return sorted(list(pos) + list(neg))


# ─────────────────────────────────────────────────────────────────────────────
# LINE MEASUREMENT
# ─────────────────────────────────────────────────────────────────────────────
class LineMeasurer:
    """Measures stripe widths and LER from detected edge positions."""

    @staticmethod
    def detect_angle(image_gray: np.ndarray) -> float:
        """Detect dominant line angle using Hough transform."""
        # Blur slightly to reduce noise
        blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)

        lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=max(30, min(image_gray.shape) // 4))
        if lines is None or len(lines) == 0:
            # Fallback: FFT
            return LineMeasurer._fft_angle(image_gray)

        angles = []
        for line in lines:
            rho, theta = line[0]
            # theta ∈ [0, π]. Lines are perpendicular to (rho, theta) direction.
            # Line angle from horizontal = theta - π/2
            a = math.degrees(theta) - 90.0
            angles.append(a)

        angles = np.array(angles)
        # Histogram vote
        hist, bins = np.histogram(angles, bins=360, range=(-90, 90))
        best_bin = np.argmax(hist)
        dominant = (bins[best_bin] + bins[best_bin + 1]) / 2
        return float(dominant)

    @staticmethod
    def _fft_angle(image_gray: np.ndarray) -> float:
        """FFT-based angle detection fallback."""
        f = np.fft.fft2(image_gray.astype(float))
        fs = np.fft.fftshift(f)
        mag = np.abs(fs)
        h, w = mag.shape
        # Mask DC
        cy, cx = h // 2, w // 2
        mag[cy - 5:cy + 5, cx - 5:cx + 5] = 0
        y, x = np.unravel_index(np.argmax(mag), mag.shape)
        dy, dx = y - cy, x - cx
        if dx == 0:
            return 90.0
        angle = math.degrees(math.atan2(dy, dx))
        # Lines are perpendicular to the dominant frequency direction
        return float(angle + 90.0) % 180 - 90

    @staticmethod
    def measure_widths(
        t: np.ndarray,
        edge_indices: List[int],
        profile: np.ndarray,
        pixels_per_um: float,
    ) -> Dict[str, Any]:
        """
        Classify edges and compute dark/bright stripe widths.

        Returns dict with dark_widths, bright_widths, medians.
        """
        if len(edge_indices) < 2:
            return {
                'dark_widths': [], 'bright_widths': [],
                'dark_median': 0.0, 'bright_median': 0.0,
                'all_widths': [],
            }

        # Sort edges by t position
        edges_sorted = sorted(edge_indices, key=lambda i: t[i] if 0 <= i < len(t) else 0)

        dark_threshold = (np.max(profile) + np.min(profile)) / 2.0

        dark_widths = []
        bright_widths = []
        all_widths = []

        for k in range(len(edges_sorted) - 1):
            i1, i2 = edges_sorted[k], edges_sorted[k + 1]
            i1 = max(0, min(len(t) - 1, i1))
            i2 = max(0, min(len(t) - 1, i2))
            width_px = abs(t[i2] - t[i1])
            width_um = width_px / pixels_per_um

            mid_idx = (i1 + i2) // 2
            if 0 <= mid_idx < len(profile):
                val = profile[mid_idx]
                if val < dark_threshold:
                    dark_widths.append(width_um)
                else:
                    bright_widths.append(width_um)
            all_widths.append(width_um)

        return {
            'dark_widths':   dark_widths,
            'bright_widths': bright_widths,
            'dark_median':   float(np.median(dark_widths))  if dark_widths   else 0.0,
            'bright_median': float(np.median(bright_widths)) if bright_widths else 0.0,
            'all_widths':    all_widths,
        }

    @staticmethod
    def compute_ler(
        t: np.ndarray,
        edge_indices: List[int],
        pixels_per_um: float,
    ) -> float:
        """LER = 3σ of edge position variations (µm)."""
        if len(edge_indices) < 3:
            return 0.0
        positions = np.array([t[i] for i in edge_indices
                               if 0 <= i < len(t)], dtype=float)
        if len(positions) < 3:
            return 0.0
        # Detrend: fit line and remove
        x = np.arange(len(positions), dtype=float)
        slope, intercept, *_ = linregress(x, positions)
        residuals = positions - (slope * x + intercept)
        ler_px = 3.0 * np.std(residuals)
        return ler_px / pixels_per_um

    @staticmethod
    def compute_lwr(
        t: np.ndarray,
        edge_indices: List[int],
        profile: np.ndarray,
        pixels_per_um: float,
    ) -> float:
        """
        Line Width Roughness (LWR) = 3σ of individual stripe width variations.

        LWR measures how much the width of a single stripe type varies,
        while LER measures positional jitter of individual edges.
        """
        if len(edge_indices) < 3:
            return 0.0
        edges_sorted = sorted(edge_indices, key=lambda i: t[i] if 0 <= i < len(t) else 0)
        dark_threshold = (np.max(profile) + np.min(profile)) / 2.0
        dark_widths = []
        bright_widths = []
        for k in range(len(edges_sorted) - 1):
            i1, i2 = edges_sorted[k], edges_sorted[k + 1]
            i1 = max(0, min(len(t) - 1, i1))
            i2 = max(0, min(len(t) - 1, i2))
            width_px = abs(t[i2] - t[i1])
            mid = (i1 + i2) // 2
            if 0 <= mid < len(profile):
                if profile[mid] < dark_threshold:
                    dark_widths.append(width_px)
                else:
                    bright_widths.append(width_px)
        # LWR on the type with more samples
        widths = dark_widths if len(dark_widths) >= len(bright_widths) else bright_widths
        if len(widths) < 2:
            return 0.0
        return 3.0 * float(np.std(widths)) / pixels_per_um

    @staticmethod
    def pitch(
        t: np.ndarray,
        edge_indices: List[int],
        pixels_per_um: float,
    ) -> float:
        """Average pitch = average distance between same-polarity edges (µm)."""
        if len(edge_indices) < 3:
            return 0.0
        edges_sorted = sorted([t[i] for i in edge_indices if 0 <= i < len(t)])
        # pitch = step between every other pair (same-side edges)
        pitches = [abs(edges_sorted[i + 2] - edges_sorted[i])
                   for i in range(len(edges_sorted) - 2)]
        return float(np.mean(pitches)) / pixels_per_um if pitches else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSOR
# ─────────────────────────────────────────────────────────────────────────────
class ImageProcessor:
    """Pre-processing pipeline applied before measurement."""

    @staticmethod
    def process(image_bgr: np.ndarray, params: Dict) -> np.ndarray:
        """Return processed grayscale image."""
        channel = params.get('channel', 'Grayscale')
        smooth  = params.get('smooth_method', 'None')
        smooth_k = int(params.get('smooth_kernel', 3))
        if smooth_k % 2 == 0:
            smooth_k += 1

        # ── channel extraction ──
        if channel == 'Grayscale' or len(image_bgr.shape) == 2:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) \
                   if len(image_bgr.shape) == 3 else image_bgr.copy()

        elif channel.startswith('RGB'):
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            ch_map = {'RGB - Red': 0, 'RGB - Green': 1, 'RGB - Blue': 2}
            gray = rgb[:, :, ch_map.get(channel, 0)]

        elif channel.startswith('YUV'):
            yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
            ch_map = {'YUV - Y (Luma)': 0, 'YUV - U': 1, 'YUV - V': 2}
            gray = yuv[:, :, ch_map.get(channel, 0)]

        elif channel.startswith('HSV'):
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            ch_map = {'HSV - Hue': 0, 'HSV - Saturation': 1, 'HSV - Value': 2}
            gray = hsv[:, :, ch_map.get(channel, 2)]

        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # ── smoothing ──
        if smooth == 'Gaussian':
            gray = cv2.GaussianBlur(gray, (smooth_k, smooth_k), 0)
        elif smooth == 'Median':
            gray = cv2.medianBlur(gray, smooth_k)
        elif smooth == 'Bilateral':
            gray = cv2.bilateralFilter(gray, smooth_k, 75, 75)
        elif smooth == 'Box (Mean)':
            gray = cv2.blur(gray, (smooth_k, smooth_k))

        return gray.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE ENHANCEMENT (brightness / contrast / gamma / CLAHE)
# ─────────────────────────────────────────────────────────────────────────────
class ImageEnhancement:
    """
    Post-process a grayscale image for better visual contrast and measurement.
    All operations are non-destructive (applied to display / proc copy only).
    """

    @staticmethod
    def apply(gray: np.ndarray, params: Dict) -> np.ndarray:
        brightness = int(params.get('brightness', 0))         # -127..+127
        contrast   = float(params.get('contrast',  1.0))      # 0.1..4.0
        gamma      = float(params.get('gamma',     1.0))      # 0.1..4.0
        clahe_en   = bool(params.get('clahe',      False))
        clahe_clip = float(params.get('clahe_clip', 2.0))
        invert     = bool(params.get('invert',     False))

        out = gray.astype(np.float32)

        # brightness
        out = out + brightness

        # contrast (pivot at 128)
        out = (out - 128.0) * contrast + 128.0

        out = np.clip(out, 0, 255).astype(np.uint8)

        # gamma
        if abs(gamma - 1.0) > 0.01:
            lut = np.array(
                [min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                 for i in range(256)], dtype=np.uint8)
            out = cv2.LUT(out, lut)

        # CLAHE
        if clahe_en:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            out = clahe.apply(out)

        # invert
        if invert:
            out = 255 - out

        return out


# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAM CANVAS (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
class HistogramCanvas(FigureCanvas):
    """Shows intensity histogram of the image (optionally restricted to ROI)."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 2), facecolor='#252526')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self._style()
        self.fig.tight_layout(pad=0.8)

    def _style(self):
        self.ax.set_facecolor('#1e1e1e')
        for spine in self.ax.spines.values():
            spine.set_color('#555')
        self.ax.tick_params(colors='#aaa', labelsize=7)
        self.ax.title.set_color('#ccc')
        self.ax.xaxis.label.set_color('#ccc')
        self.ax.yaxis.label.set_color('#ccc')

    def plot(self, gray: np.ndarray, roi_mask: Optional[np.ndarray] = None):
        data = gray[roi_mask] if roi_mask is not None else gray.ravel()
        self.plot_data(data)

    def plot_data(self, data: np.ndarray):
        """Plot pre-extracted pixel values (called from main thread after worker)."""
        self.ax.cla()
        self._style()
        if data.size == 0:
            self.draw()
            return
        self.ax.hist(data, bins=128, range=(0, 256),
                     color='#4fc3f7', alpha=0.8, density=True, histtype='stepfilled')
        self.ax.set_xlim(0, 255)
        self.ax.set_xlabel('Intensity', fontsize=7)
        self.ax.set_title('Histogram (ROI)', fontsize=8)
        self.ax.grid(True, alpha=0.2, color='#444')
        self.fig.tight_layout(pad=0.8)
        self.draw()

    def clear(self):
        self.ax.cla()
        self._style()
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSOR THREAD
# ─────────────────────────────────────────────────────────────────────────────
class BatchProcessThread(QThread):
    """
    Runs measurement on every image in the file list using current parameters.
    Emits progress and per-file results.
    """
    progress   = pyqtSignal(int, int)               # current, total
    result     = pyqtSignal(str, dict)              # filepath, result dict
    finished_all = pyqtSignal()

    def __init__(self, files: List[str], params: Dict):
        super().__init__()
        self.files  = files
        self.params = params
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        total = len(self.files)
        for i, fp in enumerate(self.files):
            if self._stop:
                break
            self.progress.emit(i + 1, total)
            try:
                img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                gray = ImageProcessor.process(img, self.params)

                enh = self.params.get('enhancement', {})
                if any(v != ImageEnhancement.apply.__defaults__ and k in enh
                       for k, v in enh.items()):
                    gray = ImageEnhancement.apply(gray, enh)

                cx     = self.params['cx']
                cy     = self.params['cy']
                radius = self.params['radius']
                angle  = self.params['angle']
                algo   = self.params['algo']
                sigma  = self.params['sigma']
                peaks  = self.params['peak_mode']
                thresh = self.params['threshold']
                ppu    = self.params['pixels_per_um']

                t, profile = EdgeDetector.extract_profile(gray, cx, cy, radius, angle)
                edge_sig   = EdgeDetector.apply_algorithm(profile, algo, sigma)
                peak_idx   = EdgeDetector.find_peaks(edge_sig, profile, peaks, thresh)

                meas = LineMeasurer.measure_widths(t, peak_idx, profile, ppu)
                ler  = LineMeasurer.compute_ler(t, peak_idx, ppu)
                lwr  = LineMeasurer.compute_lwr(t, peak_idx, profile, ppu)
                pit  = LineMeasurer.pitch(t, peak_idx, ppu)

                res = {
                    'filename':      Path(fp).name,
                    'dark_median':   meas['dark_median'],
                    'bright_median': meas['bright_median'],
                    'dark_widths':   meas['dark_widths'],
                    'bright_widths': meas['bright_widths'],
                    'ler':           ler,
                    'lwr':           lwr,
                    'pitch':         pit,
                    'angle':         angle,
                    'n_edges':       len(peak_idx),
                    'pixels_per_um': ppu,
                }
                self.result.emit(fp, res)
            except Exception as e:
                self.result.emit(fp, {'filename': Path(fp).name, 'error': str(e)})

        self.finished_all.emit()


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA THREAD
# ─────────────────────────────────────────────────────────────────────────────
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error       = pyqtSignal(str)

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self.camera_index = camera_index
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera {self.camera_index}")
            return
        while self._running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame.copy())
            self.msleep(33)  # ~30 fps
        cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CANVAS WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class ImageCanvas(QWidget):
    """Image display with interactive draggable/resizable green circle ROI."""

    circle_changed = pyqtSignal(float, float, float)   # cx, cy, radius

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 350)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._image_bgr: Optional[np.ndarray] = None   # original BGR
        self._qimage:    Optional[QImage]      = None

        # View transform
        self._scale  = 1.0
        self._offset = QPoint(0, 0)

        # Circle ROI (image coords)
        self.cx     = 200.0
        self.cy     = 200.0
        self.radius = 80.0

        # Drag state
        self._dragging  = False
        self._drag_mode = None          # 'move' | 'resize'
        self._drag_start_w  = QPoint()
        self._drag_start_cx = 0.0
        self._drag_start_cy = 0.0
        self._drag_start_r  = 0.0

        # Overlays
        self.edge_t_positions: List[float] = []    # positions in pixels relative to centre
        self.line_angle_deg = 0.0
        self.show_boundaries = True

        self.setFocusPolicy(Qt.StrongFocus)

    # ── image loading ──────────────────────────────────────────────────────
    def set_image(self, image_bgr: np.ndarray):
        self._image_bgr = image_bgr
        self._build_qimage()
        self._fit_to_window()
        self.update()

    def get_image_bgr(self) -> Optional[np.ndarray]:
        return self._image_bgr

    def _build_qimage(self):
        if self._image_bgr is None:
            self._qimage = None
            return
        img = self._image_bgr
        if len(img.shape) == 2:
            h, w = img.shape
            self._qimage = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            self._qimage = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()

    def _fit_to_window(self):
        if self._image_bgr is None:
            return
        ih, iw = self._image_bgr.shape[:2]
        ww, wh = self.width(), self.height()
        if ww == 0 or wh == 0:
            return
        self._scale = min(ww / iw, wh / ih)
        self._offset = QPoint(
            int((ww - iw * self._scale) / 2),
            int((wh - ih * self._scale) / 2),
        )

    def resizeEvent(self, event):
        self._fit_to_window()
        super().resizeEvent(event)

    # ── coordinate helpers ─────────────────────────────────────────────────
    def _img_to_widget(self, ix: float, iy: float) -> QPointF:
        return QPointF(
            ix * self._scale + self._offset.x(),
            iy * self._scale + self._offset.y(),
        )

    def _widget_to_img(self, wx: float, wy: float) -> QPointF:
        return QPointF(
            (wx - self._offset.x()) / self._scale,
            (wy - self._offset.y()) / self._scale,
        )

    # ── painting ───────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._qimage is None:
            painter.setPen(QColor(120, 120, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        if self._image_bgr is not None:
            ih, iw = self._image_bgr.shape[:2]
            dst = QRect(
                self._offset.x(), self._offset.y(),
                int(iw * self._scale), int(ih * self._scale),
            )
            painter.drawImage(dst, self._qimage)

        if self.show_boundaries:
            self._paint_boundaries(painter)

        self._paint_circle(painter)

    def _paint_boundaries(self, painter: QPainter):
        if not self.edge_t_positions or self._image_bgr is None:
            return
        ih, iw = self._image_bgr.shape[:2]
        a = math.radians(self.line_angle_deg)
        lx, ly = math.cos(a), math.sin(a)   # along lines
        px, py = -ly, lx                     # perpendicular (profile direction)
        pen = QPen(QColor(255, 140, 0), 1.5, Qt.DashLine)
        painter.setPen(pen)
        ext = max(iw, ih) * 2.0
        for t_pos in self.edge_t_positions:
            # Point on perpendicular axis
            bx = self.cx + t_pos * px
            by = self.cy + t_pos * py
            # Extend line along line direction
            x1, y1 = bx - ext * lx, by - ext * ly
            x2, y2 = bx + ext * lx, by + ext * ly
            p1 = self._img_to_widget(x1, y1)
            p2 = self._img_to_widget(x2, y2)
            painter.drawLine(p1.toPoint(), p2.toPoint())

    def _paint_circle(self, painter: QPainter):
        centre = self._img_to_widget(self.cx, self.cy)
        rw = self.radius * self._scale
        pen = QPen(QColor(0, 255, 80), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(centre, rw, rw)
        # crosshair
        cs = 7
        painter.drawLine(QPointF(centre.x() - cs, centre.y()),
                         QPointF(centre.x() + cs, centre.y()))
        painter.drawLine(QPointF(centre.x(), centre.y() - cs),
                         QPointF(centre.x(), centre.y() + cs))

    # ── mouse events ───────────────────────────────────────────────────────
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        wp = event.pos()
        ip = self._widget_to_img(wp.x(), wp.y())
        dx = ip.x() - self.cx
        dy = ip.y() - self.cy
        dist = math.hypot(dx, dy)
        rim  = max(12.0 / self._scale, self.radius * 0.12)

        self._drag_start_w  = wp
        self._drag_start_cx = self.cx
        self._drag_start_cy = self.cy
        self._drag_start_r  = self.radius

        if abs(dist - self.radius) < rim:
            self._drag_mode = 'resize'
            self._dragging  = True
        elif dist < self.radius:
            self._drag_mode = 'move'
            self._dragging  = True

    def mouseMoveEvent(self, event: QMouseEvent):
        wp = event.pos()
        ip = self._widget_to_img(wp.x(), wp.y())
        dx = ip.x() - self.cx
        dy = ip.y() - self.cy
        dist = math.hypot(dx, dy)
        rim  = max(12.0 / self._scale, self.radius * 0.12)

        # cursor feedback
        if abs(dist - self.radius) < rim:
            self.setCursor(Qt.SizeBDiagCursor)
        elif dist < self.radius:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.CrossCursor)

        if not self._dragging:
            return

        if self._drag_mode == 'move':
            delta = wp - self._drag_start_w
            self.cx = self._drag_start_cx + delta.x() / self._scale
            self.cy = self._drag_start_cy + delta.y() / self._scale
            if self._image_bgr is not None:
                ih, iw = self._image_bgr.shape[:2]
                self.cx = max(0.0, min(float(iw - 1), self.cx))
                self.cy = max(0.0, min(float(ih - 1), self.cy))

        elif self._drag_mode == 'resize':
            ip0 = self._widget_to_img(self._drag_start_w.x(), self._drag_start_w.y())
            ip1 = self._widget_to_img(wp.x(), wp.y())
            # new radius = distance from centre to current mouse
            new_r = math.hypot(ip1.x() - self._drag_start_cx,
                               ip1.y() - self._drag_start_cy)
            self.radius = max(20.0, new_r)

        self.update()
        self.circle_changed.emit(self.cx, self.cy, self.radius)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging  = False
            self._drag_mode = None

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._scale = max(0.05, min(20.0, self._scale * factor))
        if self._image_bgr is not None:
            ih, iw = self._image_bgr.shape[:2]
            ww, wh = self.width(), self.height()
            self._offset = QPoint(
                int((ww - iw * self._scale) / 2),
                int((wh - ih * self._scale) / 2),
            )
        self.update()

    # ── public helpers ─────────────────────────────────────────────────────
    def set_edge_positions(self, t: np.ndarray, edge_indices: List[int]):
        if len(edge_indices) == 0 or t is None or len(t) == 0:
            self.edge_t_positions = []
        else:
            valid = [i for i in edge_indices if 0 <= i < len(t)]
            self.edge_t_positions = [float(t[i]) for i in valid]
        self.update()

    def get_labeled_image(self) -> Optional[np.ndarray]:
        """Return a copy of the image with circle and boundaries drawn."""
        if self._image_bgr is None:
            return None
        out = self._image_bgr.copy()
        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        ih, iw = out.shape[:2]
        a  = math.radians(self.line_angle_deg)
        lx, ly = math.cos(a), math.sin(a)
        px, py = -ly, lx
        ext = max(iw, ih) * 2.0

        # Draw boundary lines
        for t_pos in self.edge_t_positions:
            bx = self.cx + t_pos * px
            by = self.cy + t_pos * py
            x1, y1 = int(bx - ext * lx), int(by - ext * ly)
            x2, y2 = int(bx + ext * lx), int(by + ext * ly)
            cv2.line(out, (x1, y1), (x2, y2), (0, 140, 255), 1, cv2.LINE_AA)

        # Draw circle
        cv2.circle(out, (int(self.cx), int(self.cy)), int(self.radius),
                   (0, 255, 80), 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(self.cx), int(self.cy)),
                       (0, 255, 80), cv2.MARKER_CROSS, 14, 1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# PROFILE CANVAS (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
class ProfileCanvas(FigureCanvas):
    """Matplotlib widget showing intensity profile and edge-detection signal."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 4), facecolor='#252526')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self._style()
        self.fig.tight_layout(pad=1.5)

    def _style(self):
        for ax in (self.ax1, self.ax2):
            ax.set_facecolor('#1e1e1e')
            for spine in ax.spines.values():
                spine.set_color('#555')
            ax.tick_params(colors='#aaa', labelsize=7)
            ax.title.set_color('#ccc')
            ax.xaxis.label.set_color('#ccc')
            ax.yaxis.label.set_color('#ccc')

    def plot(
        self,
        t: np.ndarray,
        profile: np.ndarray,
        edge_signal: np.ndarray,
        peak_indices: List[int],
        pixels_per_um: float = 1.0,
    ):
        self.ax1.cla()
        self.ax2.cla()
        self._style()

        t_um = t / pixels_per_um if pixels_per_um > 0 else t
        xlabel = 'Position (µm)' if pixels_per_um != 1.0 else 'Position (px)'

        self.ax1.plot(t_um, profile, color='#4fc3f7', lw=1)
        self.ax1.set_ylabel('Intensity', fontsize=7)
        self.ax1.set_title('Intensity Profile', fontsize=8)
        self.ax1.grid(True, alpha=0.25, color='#444')

        self.ax2.plot(t_um, edge_signal, color='#ef5350', lw=1)
        self.ax2.axhline(0, color='#555', lw=0.5)
        self.ax2.set_ylabel('Edge Signal', fontsize=7)
        self.ax2.set_title('Edge Detection Signal', fontsize=8)
        self.ax2.set_xlabel(xlabel, fontsize=7)
        self.ax2.grid(True, alpha=0.25, color='#444')

        if peak_indices:
            valid = [i for i in peak_indices if 0 <= i < len(t)]
            if valid:
                self.ax2.scatter(t_um[valid], edge_signal[valid],
                                 c='#ffee58', s=35, zorder=5)
                for pi in valid:
                    self.ax1.axvline(t_um[pi], color='#ff7043',
                                     alpha=0.6, lw=0.8)

        self.fig.tight_layout(pad=1.5)
        self.draw()

    def clear(self):
        self.ax1.cla()
        self.ax2.cla()
        self._style()
        self.draw()


# ─────────────────────────────────────────────────────────────────────────────
# MEASUREMENT WORKER  (runs on a background QThread)
# ─────────────────────────────────────────────────────────────────────────────
class MeasurementWorker(QObject):
    """
    All heavy computation lives here so the Qt main thread stays responsive.
    Signals carry results back to the GUI thread.
    """
    finished = pyqtSignal(dict)   # emits result dict on success
    error    = pyqtSignal(str)    # emits error string on failure

    def __init__(self):
        super().__init__()
        self._params: Optional[Dict] = None
        self._cancelled = False

    def set_params(self, params: Dict):
        self._params = params
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        if self._params is None or self._cancelled:
            return
        p = self._params
        try:
            gray      = p['gray']
            cx        = p['cx']
            cy        = p['cy']
            r         = p['radius']
            ang       = p['angle']
            algo      = p['algo']
            sigma     = p['sigma']
            peak_mode = p['peak_mode']
            threshold = p['threshold']
            ppu       = p['pixels_per_um']

            t, profile   = EdgeDetector.extract_profile(gray, cx, cy, r, ang)
            if self._cancelled:
                return
            edge_signal  = EdgeDetector.apply_algorithm(profile, algo, sigma)
            peak_indices = EdgeDetector.find_peaks(
                edge_signal, profile, peak_mode, threshold)

            meas = LineMeasurer.measure_widths(t, peak_indices, profile, ppu)
            ler  = LineMeasurer.compute_ler(t, peak_indices, ppu)
            lwr  = LineMeasurer.compute_lwr(t, peak_indices, profile, ppu)
            pit  = LineMeasurer.pitch(t, peak_indices, ppu)

            # Build histogram data inside the worker (avoids large mask on main thread)
            h, w = gray.shape
            r_int = int(r)
            cx_i, cy_i = int(cx), int(cy)
            y0 = max(0, cy_i - r_int)
            y1 = min(h, cy_i + r_int + 1)
            x0 = max(0, cx_i - r_int)
            x1 = min(w, cx_i + r_int + 1)
            roi_crop = gray[y0:y1, x0:x1]
            # Build mask only on the small crop
            if roi_crop.size > 0:
                Yc, Xc = np.ogrid[:roi_crop.shape[0], :roi_crop.shape[1]]
                mask = ((Xc - (cx - x0)) ** 2 +
                        (Yc - (cy - y0)) ** 2) <= r ** 2
                hist_data = roi_crop[mask].ravel()
            else:
                hist_data = gray.ravel()

            self.finished.emit({
                't':            t,
                'profile':      profile,
                'edge_signal':  edge_signal,
                'peak_indices': peak_indices,
                'meas':         meas,
                'ler':          ler,
                'lwr':          lwr,
                'pitch':        pit,
                'angle':        ang,
                'hist_data':    hist_data,
                'ppu':          ppu,
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 900)

        # ── state ──────────────────────────────────────────────────────────
        self._directory: str = ""
        self._image_files: List[str] = []
        self._current_index: int = -1

        self._raw_image: Optional[np.ndarray] = None    # original BGR load
        self._proc_image:     Optional[np.ndarray] = None  # channel-extracted gray
        self._enhanced_image: Optional[np.ndarray] = None  # after enhancement

        self._camera_thread: Optional[CameraThread] = None
        self._camera_mode = False
        self._camera_captured: Optional[np.ndarray] = None

        self._batch_thread: Optional[BatchProcessThread] = None

        # per-image session data  { filename -> {...} }
        self._session: Dict[str, Any] = {}
        # measurement results per file
        self._results: Dict[str, Dict] = {}

        # last measurement output
        self._last_t:            Optional[np.ndarray] = None
        self._last_profile:      Optional[np.ndarray] = None
        self._last_edge_signal:  Optional[np.ndarray] = None
        self._last_peak_indices: List[int] = []

        self._build_ui()
        self._apply_dark_theme()
        self._load_settings()

        # ── Background measurement worker ──────────────────────────────────
        self._meas_worker  = MeasurementWorker()
        self._meas_thread  = QThread(self)
        self._meas_worker.moveToThread(self._meas_thread)
        self._meas_thread.started.connect(self._meas_worker.run)
        self._meas_worker.finished.connect(self._on_measurement_done)
        self._meas_worker.error.connect(self._on_measurement_error)
        self._meas_thread.start()

        # Debounce timer: waits 120 ms of inactivity before triggering worker
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(120)
        self._update_timer.timeout.connect(self._launch_measurement)

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left panel: file list
        splitter.addWidget(self._build_file_panel())

        # Centre: image + profile
        centre_widget = QWidget()
        centre_vbox   = QVBoxLayout(centre_widget)
        centre_vbox.setContentsMargins(0, 0, 0, 0)
        centre_vbox.setSpacing(3)

        # navigation bar
        nav = QHBoxLayout()
        self.btn_prev  = QPushButton("◀ Prev")
        self.btn_next  = QPushButton("Next ▶")
        self.lbl_fname = QLabel("No image")
        self.lbl_fname.setAlignment(Qt.AlignCenter)
        btn_fit = QPushButton("Fit")
        btn_fit.setToolTip("Fit image to window")
        btn_fit.setFixedWidth(50)
        btn_fit.clicked.connect(self._fit_image)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.lbl_fname, 1)
        nav.addWidget(self.btn_next)
        nav.addWidget(btn_fit)
        centre_vbox.addLayout(nav)

        # image canvas
        self.image_canvas = ImageCanvas()
        self.image_canvas.circle_changed.connect(self._on_circle_changed)

        # profile canvas
        # Bottom tab: profile + histogram
        bottom_tabs = QTabWidget()
        bottom_tabs.setMinimumHeight(200)
        bottom_tabs.setMaximumHeight(280)
        bottom_tabs.setTabPosition(QTabWidget.South)

        self.profile_canvas = ProfileCanvas()
        bottom_tabs.addTab(self.profile_canvas, "Edge Profile")

        self.histogram_canvas = HistogramCanvas()
        bottom_tabs.addTab(self.histogram_canvas, "Histogram")

        img_splitter = QSplitter(Qt.Vertical)
        img_splitter.addWidget(self.image_canvas)
        img_splitter.addWidget(bottom_tabs)
        img_splitter.setSizes([600, 220])
        centre_vbox.addWidget(img_splitter, 1)

        splitter.addWidget(centre_widget)

        # Right panel: controls + results
        splitter.addWidget(self._build_control_panel())

        splitter.setSizes([230, 800, 320])

        # toolbar / menu
        self._build_menu()

        # status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.lbl_status = QLabel("Ready")
        self.status.addWidget(self.lbl_status, 1)

        # button connections
        self.btn_prev.clicked.connect(self._prev_image)
        self.btn_next.clicked.connect(self._next_image)

    # ── file panel ─────────────────────────────────────────────────────────
    def _build_file_panel(self) -> QWidget:
        w = QWidget()
        w.setMinimumWidth(200)
        w.setMaximumWidth(300)
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(4, 4, 4, 4)
        vbox.setSpacing(4)

        # directory picker
        hb = QHBoxLayout()
        btn_dir = QPushButton("Open Dir")
        btn_dir.clicked.connect(self._open_directory)
        hb.addWidget(btn_dir)
        btn_cam = QPushButton("Camera")
        btn_cam.clicked.connect(self._toggle_camera)
        hb.addWidget(btn_cam)
        self.btn_cam = btn_cam
        vbox.addLayout(hb)

        # file table
        self.file_table = QTableWidget(0, 3)
        self.file_table.setHorizontalHeaderLabels(["File", "Dark (µm)", "Bright (µm)"])
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_table.verticalHeader().setDefaultSectionSize(20)
        self.file_table.itemSelectionChanged.connect(self._on_file_selected)
        vbox.addWidget(self.file_table, 1)

        # export
        hb2 = QHBoxLayout()
        btn_csv = QPushButton("Export CSV")
        btn_csv.clicked.connect(self._export_csv)
        btn_imgs = QPushButton("Save Images")
        btn_imgs.clicked.connect(self._save_labeled_images)
        hb2.addWidget(btn_csv)
        hb2.addWidget(btn_imgs)
        vbox.addLayout(hb2)

        return w

    # ── control panel ──────────────────────────────────────────────────────
    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        scroll.setMaximumWidth(380)

        inner = QWidget()
        scroll.setWidget(inner)
        vbox = QVBoxLayout(inner)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(6)

        # ── Scale ──
        grp_scale = QGroupBox("Scale Calibration")
        fl = QFormLayout(grp_scale)
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.001, 100000.0)
        self.spin_scale.setValue(1.0)
        self.spin_scale.setSuffix(" px/µm")
        self.spin_scale.setDecimals(4)
        self.spin_scale.valueChanged.connect(self._schedule_measurement)
        fl.addRow("Scale:", self.spin_scale)
        vbox.addWidget(grp_scale)

        # ── Angle ──
        grp_angle = QGroupBox("Line Angle")
        fl2 = QFormLayout(grp_angle)
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(-90.0, 90.0)
        self.spin_angle.setValue(0.0)
        self.spin_angle.setSuffix("°")
        self.spin_angle.setDecimals(2)
        self.spin_angle.valueChanged.connect(self._on_angle_changed)
        fl2.addRow("Angle:", self.spin_angle)
        btn_detect = QPushButton("Auto-Detect Angle")
        btn_detect.clicked.connect(self._auto_detect_angle)
        fl2.addRow(btn_detect)
        vbox.addWidget(grp_angle)

        # ── Circle ROI ──
        grp_roi = QGroupBox("Circle ROI")
        fl3 = QFormLayout(grp_roi)
        self.spin_cx = QDoubleSpinBox()
        self.spin_cx.setRange(0, 99999)
        self.spin_cx.valueChanged.connect(self._on_roi_spinbox_changed)
        self.spin_cy = QDoubleSpinBox()
        self.spin_cy.setRange(0, 99999)
        self.spin_cy.valueChanged.connect(self._on_roi_spinbox_changed)
        self.spin_r  = QDoubleSpinBox()
        self.spin_r.setRange(10, 99999)
        self.spin_r.setValue(80)
        self.spin_r.valueChanged.connect(self._on_roi_spinbox_changed)
        fl3.addRow("Centre X:", self.spin_cx)
        fl3.addRow("Centre Y:", self.spin_cy)
        fl3.addRow("Radius:",   self.spin_r)
        vbox.addWidget(grp_roi)

        # ── Image Processing ──
        grp_proc = QGroupBox("Image Processing")
        fl4 = QFormLayout(grp_proc)
        self.combo_channel = QComboBox()
        self.combo_channel.addItems(IMAGE_CHANNELS)
        self.combo_channel.currentIndexChanged.connect(self._on_proc_changed)
        fl4.addRow("Channel:", self.combo_channel)
        self.combo_smooth = QComboBox()
        self.combo_smooth.addItems(SMOOTH_METHODS)
        self.combo_smooth.currentIndexChanged.connect(self._on_proc_changed)
        fl4.addRow("Smooth:", self.combo_smooth)
        self.spin_smooth_k = QSpinBox()
        self.spin_smooth_k.setRange(1, 51)
        self.spin_smooth_k.setValue(3)
        self.spin_smooth_k.setSingleStep(2)
        self.spin_smooth_k.valueChanged.connect(self._on_proc_changed)
        fl4.addRow("Kernel:", self.spin_smooth_k)
        vbox.addWidget(grp_proc)

        # ── Edge Detection ──
        grp_edge = QGroupBox("Edge Detection")
        vb_edge = QVBoxLayout(grp_edge)
        fl5 = QFormLayout()
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(ALGORITHMS)
        self.combo_algo.currentIndexChanged.connect(self._schedule_measurement)
        fl5.addRow("Algorithm:", self.combo_algo)
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 20.0)
        self.spin_sigma.setValue(2.0)
        self.spin_sigma.setSuffix(" σ")
        self.spin_sigma.valueChanged.connect(self._schedule_measurement)
        fl5.addRow("Sigma:", self.spin_sigma)
        vb_edge.addLayout(fl5)

        # Peak mode
        self.combo_peaks = QComboBox()
        self.combo_peaks.addItems(PEAK_MODES)
        self.combo_peaks.setCurrentIndex(2)  # + and - peaks default
        self.combo_peaks.currentIndexChanged.connect(self._schedule_measurement)
        fl5b = QFormLayout()
        fl5b.addRow("Peak mode:", self.combo_peaks)
        vb_edge.addLayout(fl5b)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 1.0)
        self.spin_threshold.setValue(0.25)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.valueChanged.connect(self._schedule_measurement)
        fl5c = QFormLayout()
        fl5c.addRow("Threshold:", self.spin_threshold)
        vb_edge.addLayout(fl5c)

        vbox.addWidget(grp_edge)

        # ── Enhancement ──
        grp_enh = QGroupBox("Image Enhancement")
        fl_enh = QFormLayout(grp_enh)

        self.slider_brightness = QSlider(Qt.Horizontal)
        self.slider_brightness.setRange(-127, 127)
        self.slider_brightness.setValue(0)
        self.slider_brightness.setTickInterval(32)
        self.slider_brightness.valueChanged.connect(self._on_enhancement_changed)
        fl_enh.addRow("Brightness:", self.slider_brightness)

        self.slider_contrast = QSlider(Qt.Horizontal)
        self.slider_contrast.setRange(10, 400)   # 0.10 … 4.00 × 100
        self.slider_contrast.setValue(100)
        self.slider_contrast.valueChanged.connect(self._on_enhancement_changed)
        fl_enh.addRow("Contrast:", self.slider_contrast)

        self.slider_gamma = QSlider(Qt.Horizontal)
        self.slider_gamma.setRange(10, 400)       # 0.10 … 4.00 × 100
        self.slider_gamma.setValue(100)
        self.slider_gamma.valueChanged.connect(self._on_enhancement_changed)
        fl_enh.addRow("Gamma:", self.slider_gamma)

        self.chk_clahe = QCheckBox("CLAHE (auto contrast)")
        self.chk_clahe.stateChanged.connect(self._on_enhancement_changed)
        fl_enh.addRow(self.chk_clahe)

        self.spin_clahe_clip = QDoubleSpinBox()
        self.spin_clahe_clip.setRange(0.5, 20.0)
        self.spin_clahe_clip.setValue(2.0)
        self.spin_clahe_clip.valueChanged.connect(self._on_enhancement_changed)
        fl_enh.addRow("CLAHE clip:", self.spin_clahe_clip)

        self.chk_invert = QCheckBox("Invert image")
        self.chk_invert.stateChanged.connect(self._on_enhancement_changed)
        fl_enh.addRow(self.chk_invert)

        btn_reset_enh = QPushButton("Reset Enhancement")
        btn_reset_enh.clicked.connect(self._reset_enhancement)
        fl_enh.addRow(btn_reset_enh)
        vbox.addWidget(grp_enh)

        # ── Measurement Results ──
        grp_res = QGroupBox("Measurement Results")
        fl6 = QFormLayout(grp_res)

        self.lbl_angle_val  = QLabel("—")
        self.lbl_dark_med   = QLabel("—")
        self.lbl_bright_med = QLabel("—")
        self.lbl_ler        = QLabel("—")
        self.lbl_lwr        = QLabel("—")
        self.lbl_pitch      = QLabel("—")
        self.lbl_n_edges    = QLabel("—")

        for lbl in (self.lbl_angle_val, self.lbl_dark_med, self.lbl_bright_med,
                    self.lbl_ler, self.lbl_lwr, self.lbl_pitch, self.lbl_n_edges):
            lbl.setFont(QFont("Courier", 10))
            lbl.setStyleSheet("color: #4fc3f7;")

        fl6.addRow("Angle:",         self.lbl_angle_val)
        fl6.addRow("Dark median:",   self.lbl_dark_med)
        fl6.addRow("Bright median:", self.lbl_bright_med)
        fl6.addRow("LER (3σ):",      self.lbl_ler)
        fl6.addRow("LWR (3σ):",      self.lbl_lwr)
        fl6.addRow("Pitch:",         self.lbl_pitch)
        fl6.addRow("Edge count:",    self.lbl_n_edges)

        btn_measure = QPushButton("Re-Measure Now")
        btn_measure.clicked.connect(self._run_measurement)
        fl6.addRow(btn_measure)

        btn_batch = QPushButton("Batch Measure All")
        btn_batch.setToolTip("Measure all loaded images with current settings")
        btn_batch.clicked.connect(self._start_batch_measurement)
        fl6.addRow(btn_batch)
        self.btn_batch = btn_batch

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        fl6.addRow(self.progress_bar)
        vbox.addWidget(grp_res)

        # ── Display options ──
        grp_disp = QGroupBox("Display")
        fl7 = QFormLayout(grp_disp)
        self.chk_boundaries = QCheckBox("Show boundary lines")
        self.chk_boundaries.setChecked(True)
        self.chk_boundaries.stateChanged.connect(self._on_display_changed)
        fl7.addRow(self.chk_boundaries)
        vbox.addWidget(grp_disp)

        vbox.addStretch(1)
        return scroll

    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("File")
        a = fm.addAction("Open Directory")
        a.triggered.connect(self._open_directory)
        a = fm.addAction("Export CSV")
        a.triggered.connect(self._export_csv)
        a = fm.addAction("Save Labeled Images")
        a.triggered.connect(self._save_labeled_images)
        fm.addSeparator()
        a = fm.addAction("Save Settings")
        a.triggered.connect(self._save_settings)
        a = fm.addAction("Load Settings")
        a.triggered.connect(self._load_settings_dialog)
        fm.addSeparator()
        a = fm.addAction("Exit")
        a.triggered.connect(self.close)

        # Camera
        cm = mb.addMenu("Camera")
        a = cm.addAction("Toggle Camera")
        a.triggered.connect(self._toggle_camera)
        a = cm.addAction("Capture Frame")
        a.triggered.connect(self._capture_frame)

        # Help
        hm = mb.addMenu("Help")
        a = hm.addAction("About")
        a.triggered.connect(self._show_about)

    # ── dark theme ─────────────────────────────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background-color: {DARK_BG};
                color: {TEXT_FG};
            }}
            QGroupBox {{
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 8px;
                font-weight: bold;
                color: #aaa;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            QPushButton {{
                background-color: {BUTTON_BG};
                color: {TEXT_FG};
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                min-height: 22px;
            }}
            QPushButton:hover {{
                background-color: #505050;
            }}
            QPushButton:pressed {{
                background-color: {ACCENT};
            }}
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
                background-color: #2d2d2d;
                color: {TEXT_FG};
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px 4px;
            }}
            QTableWidget {{
                background-color: #1e1e1e;
                color: {TEXT_FG};
                gridline-color: #333;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: #094771;
            }}
            QHeaderView::section {{
                background-color: #2d2d2d;
                color: #aaa;
                border: 1px solid #444;
                padding: 3px;
            }}
            QScrollArea {{ border: none; }}
            QScrollBar:vertical {{
                background: #2a2a2a;
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: #555;
                border-radius: 4px;
            }}
            QSplitter::handle {{
                background: #3a3a3a;
            }}
            QMenuBar {{
                background-color: #2d2d2d;
                color: {TEXT_FG};
            }}
            QMenu {{
                background-color: #2d2d2d;
                color: {TEXT_FG};
                border: 1px solid #555;
            }}
            QMenu::item:selected {{
                background-color: #094771;
            }}
            QStatusBar {{
                background-color: #007acc;
                color: white;
            }}
            QCheckBox {{ color: {TEXT_FG}; }}
        """)

    # ── directory / file handling ──────────────────────────────────────────
    def _open_directory(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select Image Directory",
            self._directory or str(Path.home()))
        if not d:
            return
        self._directory = d
        self._load_directory(d)

    def _load_directory(self, path: str):
        self._image_files = sorted([
            str(p) for p in Path(path).iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])
        self._populate_file_table()
        if self._image_files:
            self._goto_image(0)

    def _populate_file_table(self):
        self.file_table.setRowCount(0)
        for fp in self._image_files:
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            name = Path(fp).name
            self.file_table.setItem(row, 0, QTableWidgetItem(name))
            res = self._results.get(fp, {})
            dm = res.get('dark_median', '')
            bm = res.get('bright_median', '')
            self.file_table.setItem(row, 1, QTableWidgetItem(
                f"{dm:.3f}" if isinstance(dm, float) else ""))
            self.file_table.setItem(row, 2, QTableWidgetItem(
                f"{bm:.3f}" if isinstance(bm, float) else ""))

    def _on_file_selected(self):
        rows = self.file_table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if idx != self._current_index:
            self._goto_image(idx)

    def _prev_image(self):
        if self._current_index > 0:
            self._goto_image(self._current_index - 1)

    def _next_image(self):
        if self._current_index < len(self._image_files) - 1:
            self._goto_image(self._current_index + 1)

    def _goto_image(self, index: int):
        if not self._image_files or not (0 <= index < len(self._image_files)):
            return
        self._save_session_current()
        self._current_index = index
        fp = self._image_files[index]

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.lbl_status.setText(f"Failed to load: {fp}")
            return

        # normalise to 8-bit BGR
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self._raw_image = img
        self._camera_mode = False

        # restore session
        self._restore_session(fp)

        self._apply_processing()
        self._apply_enhancement()
        # Show original raw image as base; enhancement is shown via _apply_enhancement
        self.image_canvas.set_image(self._raw_image)
        self.lbl_fname.setText(Path(fp).name)
        self.file_table.selectRow(index)

        self._schedule_measurement()
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        self.btn_prev.setEnabled(self._current_index > 0)
        self.btn_next.setEnabled(
            self._current_index < len(self._image_files) - 1)

    def _fit_image(self):
        self.image_canvas._fit_to_window()
        self.image_canvas.update()

    # ── session save / restore ─────────────────────────────────────────────
    def _save_session_current(self):
        if self._current_index < 0 or not self._image_files:
            return
        fp = self._image_files[self._current_index]
        self._session[fp] = {
            'cx':     self.image_canvas.cx,
            'cy':     self.image_canvas.cy,
            'radius': self.image_canvas.radius,
            'angle':  self.spin_angle.value(),
            'algo':   self.combo_algo.currentIndex(),
            'sigma':  self.spin_sigma.value(),
            'peaks':  self.combo_peaks.currentIndex(),
            'thresh': self.spin_threshold.value(),
            'channel': self.combo_channel.currentIndex(),
            'smooth':  self.combo_smooth.currentIndex(),
            'smooth_k': self.spin_smooth_k.value(),
        }
        self._write_session_file()

    def _restore_session(self, fp: str):
        s = self._session.get(fp)
        if s is None:
            return
        # Block signals while restoring
        self._block_signals(True)
        self.image_canvas.cx     = s.get('cx',     self.image_canvas.cx)
        self.image_canvas.cy     = s.get('cy',     self.image_canvas.cy)
        self.image_canvas.radius = s.get('radius', self.image_canvas.radius)
        self.spin_angle.setValue(s.get('angle', 0.0))
        self.combo_algo.setCurrentIndex(s.get('algo', 0))
        self.spin_sigma.setValue(s.get('sigma', 2.0))
        self.combo_peaks.setCurrentIndex(s.get('peaks', 2))
        self.spin_threshold.setValue(s.get('thresh', 0.25))
        self.combo_channel.setCurrentIndex(s.get('channel', 0))
        self.combo_smooth.setCurrentIndex(s.get('smooth', 0))
        self.spin_smooth_k.setValue(s.get('smooth_k', 3))
        self._sync_roi_spinboxes()
        self._block_signals(False)

    def _block_signals(self, block: bool):
        for w in (self.spin_angle, self.combo_algo, self.spin_sigma,
                  self.combo_peaks, self.spin_threshold, self.combo_channel,
                  self.combo_smooth, self.spin_smooth_k,
                  self.spin_cx, self.spin_cy, self.spin_r):
            w.blockSignals(block)

    def _write_session_file(self):
        try:
            path = Path(self._directory) / SESSION_FILE if self._directory else Path(SESSION_FILE)
            with open(path, 'w') as f:
                json.dump(self._session, f, indent=2)
        except Exception:
            pass

    def _read_session_file(self):
        try:
            candidates = []
            if self._directory:
                candidates.append(Path(self._directory) / SESSION_FILE)
            candidates.append(Path(SESSION_FILE))
            for p in candidates:
                if p.exists():
                    with open(p) as f:
                        self._session = json.load(f)
                    return
        except Exception:
            pass

    # ── processing ────────────────────────────────────────────────────────
    def _get_proc_params(self) -> Dict:
        return {
            'channel':      self.combo_channel.currentText(),
            'smooth_method': self.combo_smooth.currentText(),
            'smooth_kernel': self.spin_smooth_k.value(),
        }

    def _apply_processing(self):
        if self._raw_image is None:
            return
        self._proc_image = ImageProcessor.process(
            self._raw_image, self._get_proc_params())
        self._enhanced_image = self._proc_image  # default: no enhancement

    def _on_proc_changed(self):
        self._apply_processing()
        self._apply_enhancement()
        self._schedule_measurement()

    # ── enhancement ────────────────────────────────────────────────────────
    def _get_enhancement_params(self) -> Dict:
        return {
            'brightness': self.slider_brightness.value(),
            'contrast':   self.slider_contrast.value() / 100.0,
            'gamma':      self.slider_gamma.value() / 100.0,
            'clahe':      self.chk_clahe.isChecked(),
            'clahe_clip': self.spin_clahe_clip.value(),
            'invert':     self.chk_invert.isChecked(),
        }

    def _apply_enhancement(self):
        """Apply enhancement on top of processed image for display and measurement."""
        if self._proc_image is None:
            return
        params = self._get_enhancement_params()
        # Only apply if any non-default value is set
        non_default = (
            params['brightness'] != 0 or
            abs(params['contrast'] - 1.0) > 0.01 or
            abs(params['gamma'] - 1.0) > 0.01 or
            params['clahe'] or params['invert']
        )
        if non_default:
            self._enhanced_image = ImageEnhancement.apply(self._proc_image, params)
        else:
            self._enhanced_image = self._proc_image
        # Update display overlay (show enhanced as BGR)
        if self._raw_image is not None:
            enh_bgr = cv2.cvtColor(self._enhanced_image, cv2.COLOR_GRAY2BGR)
            self.image_canvas.set_image(enh_bgr)

    def _on_enhancement_changed(self):
        self._apply_enhancement()
        self._schedule_measurement()

    def _reset_enhancement(self):
        self.slider_brightness.setValue(0)
        self.slider_contrast.setValue(100)
        self.slider_gamma.setValue(100)
        self.chk_clahe.setChecked(False)
        self.spin_clahe_clip.setValue(2.0)
        self.chk_invert.setChecked(False)

    # ── angle ──────────────────────────────────────────────────────────────
    def _on_angle_changed(self):
        angle = self.spin_angle.value()
        self.image_canvas.line_angle_deg = angle
        self.lbl_angle_val.setText(f"{angle:.2f}°")
        self._schedule_measurement()

    def _auto_detect_angle(self):
        if self._proc_image is None:
            return
        angle = LineMeasurer.detect_angle(self._proc_image)
        self.spin_angle.setValue(angle)
        self.lbl_status.setText(f"Auto-detected angle: {angle:.2f}°")

    # ── ROI spinboxes ──────────────────────────────────────────────────────
    def _sync_roi_spinboxes(self):
        self.spin_cx.blockSignals(True)
        self.spin_cy.blockSignals(True)
        self.spin_r.blockSignals(True)
        self.spin_cx.setValue(self.image_canvas.cx)
        self.spin_cy.setValue(self.image_canvas.cy)
        self.spin_r.setValue(self.image_canvas.radius)
        self.spin_cx.blockSignals(False)
        self.spin_cy.blockSignals(False)
        self.spin_r.blockSignals(False)

    def _on_roi_spinbox_changed(self):
        self.image_canvas.cx     = self.spin_cx.value()
        self.image_canvas.cy     = self.spin_cy.value()
        self.image_canvas.radius = self.spin_r.value()
        self.image_canvas.update()
        self._schedule_measurement()

    def _on_circle_changed(self, cx: float, cy: float, r: float):
        self._sync_roi_spinboxes()
        self._schedule_measurement()

    # ── measurement scheduling ─────────────────────────────────────────────
    def _schedule_measurement(self):
        """Restart debounce timer; actual work fires after 120 ms of quiet."""
        self._update_timer.start()

    def _launch_measurement(self):
        """Called by debounce timer — packages params and signals the worker."""
        gray = getattr(self, '_enhanced_image', None) or self._proc_image
        if gray is None:
            return
        self._meas_worker.cancel()   # discard any in-flight computation
        params = {
            'gray':         gray,
            'cx':           self.image_canvas.cx,
            'cy':           self.image_canvas.cy,
            'radius':       self.image_canvas.radius,
            'angle':        self.spin_angle.value(),
            'algo':         self.combo_algo.currentText(),
            'sigma':        self.spin_sigma.value(),
            'peak_mode':    self.combo_peaks.currentText(),
            'threshold':    self.spin_threshold.value(),
            'pixels_per_um': self.spin_scale.value(),
        }
        self._meas_worker.set_params(params)
        self.lbl_status.setText("Measuring…")
        # Re-trigger the thread's event loop by invoking run() via a queued call
        QTimer.singleShot(0, self._meas_worker.run)

    # Kept for "Re-Measure Now" button — just restarts the debounce immediately
    def _run_measurement(self):
        self._update_timer.stop()
        self._launch_measurement()

    def _on_measurement_done(self, result: Dict):
        """Slot called on main thread when worker emits finished."""
        t            = result['t']
        profile      = result['profile']
        edge_signal  = result['edge_signal']
        peak_indices = result['peak_indices']
        meas         = result['meas']
        ler          = result['ler']
        lwr          = result['lwr']
        pit          = result['pitch']
        ppu          = result['ppu']
        ang          = result['angle']
        hist_data    = result['hist_data']

        self._last_t            = t
        self._last_profile      = profile
        self._last_edge_signal  = edge_signal
        self._last_peak_indices = peak_indices

        # Labels
        self.lbl_dark_med.setText(
            f"{meas['dark_median']:.3f} µm" if meas['dark_median'] else "—")
        self.lbl_bright_med.setText(
            f"{meas['bright_median']:.3f} µm" if meas['bright_median'] else "—")
        self.lbl_ler.setText(f"{ler:.3f} µm" if ler else "—")
        self.lbl_lwr.setText(f"{lwr:.3f} µm" if lwr else "—")
        self.lbl_pitch.setText(f"{pit:.3f} µm" if pit else "—")
        self.lbl_n_edges.setText(str(len(peak_indices)))
        self.lbl_angle_val.setText(f"{ang:.2f}°")

        # Canvas overlays
        self.image_canvas.line_angle_deg = ang
        self.image_canvas.set_edge_positions(t, peak_indices)

        # Profile plot (matplotlib — keep lightweight)
        self.profile_canvas.plot(t, profile, edge_signal, peak_indices, ppu)

        # Histogram — data already computed in worker
        self.histogram_canvas.plot_data(hist_data)

        # Persist result
        if self._current_index >= 0 and self._image_files:
            fp = self._image_files[self._current_index]
            self._results[fp] = {
                'filename':      Path(fp).name,
                'dark_median':   meas['dark_median'],
                'bright_median': meas['bright_median'],
                'dark_widths':   meas['dark_widths'],
                'bright_widths': meas['bright_widths'],
                'ler':           ler,
                'lwr':           lwr,
                'pitch':         pit,
                'angle':         ang,
                'n_edges':       len(peak_indices),
                'pixels_per_um': ppu,
            }
            self._update_file_table_row(self._current_index, meas)

        self.lbl_status.setText(
            f"Edges: {len(peak_indices)} | "
            f"Dark: {meas['dark_median']:.3f} µm | "
            f"Bright: {meas['bright_median']:.3f} µm | "
            f"LER: {ler:.3f} µm  LWR: {lwr:.3f} µm")

    def _on_measurement_error(self, msg: str):
        self.lbl_status.setText(f"Measurement error — see console")
        print(msg, file=sys.stderr)

    def _update_file_table_row(self, idx: int, meas: Dict):
        if 0 <= idx < self.file_table.rowCount():
            dm = meas.get('dark_median', 0.0)
            bm = meas.get('bright_median', 0.0)
            self.file_table.setItem(idx, 1, QTableWidgetItem(
                f"{dm:.3f}" if dm else ""))
            self.file_table.setItem(idx, 2, QTableWidgetItem(
                f"{bm:.3f}" if bm else ""))

    # ── display options ────────────────────────────────────────────────────
    def _on_display_changed(self):
        self.image_canvas.show_boundaries = self.chk_boundaries.isChecked()
        self.image_canvas.update()

    # ── batch processing ────────────────────────────────────────────────────
    def _start_batch_measurement(self):
        if not self._image_files:
            QMessageBox.information(self, "No Files", "No image files loaded.")
            return

        params = {
            'cx':           self.image_canvas.cx,
            'cy':           self.image_canvas.cy,
            'radius':       self.image_canvas.radius,
            'angle':        self.spin_angle.value(),
            'algo':         self.combo_algo.currentText(),
            'sigma':        self.spin_sigma.value(),
            'peak_mode':    self.combo_peaks.currentText(),
            'threshold':    self.spin_threshold.value(),
            'pixels_per_um': self.spin_scale.value(),
            'channel':      self.combo_channel.currentText(),
            'smooth_method': self.combo_smooth.currentText(),
            'smooth_kernel': self.spin_smooth_k.value(),
            'enhancement':  self._get_enhancement_params(),
        }

        self._batch_thread = BatchProcessThread(self._image_files, params)
        self._batch_thread.progress.connect(self._on_batch_progress)
        self._batch_thread.result.connect(self._on_batch_result)
        self._batch_thread.finished_all.connect(self._on_batch_done)
        self.progress_bar.setRange(0, len(self._image_files))
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_batch.setText("Stop Batch")
        self.btn_batch.clicked.disconnect()
        self.btn_batch.clicked.connect(self._stop_batch)
        self._batch_thread.start()
        self.lbl_status.setText("Batch measurement started…")

    def _stop_batch(self):
        if hasattr(self, '_batch_thread') and self._batch_thread.isRunning():
            self._batch_thread.stop()
        self._on_batch_done()

    def _on_batch_progress(self, current: int, total: int):
        self.progress_bar.setValue(current)
        self.lbl_status.setText(f"Batch: {current}/{total}")

    def _on_batch_result(self, fp: str, result: Dict):
        self._results[fp] = result
        # find row index
        try:
            idx = self._image_files.index(fp)
            self._update_file_table_row(idx, result)
        except ValueError:
            pass

    def _on_batch_done(self):
        self.progress_bar.setVisible(False)
        self.btn_batch.setText("Batch Measure All")
        self.btn_batch.clicked.disconnect()
        self.btn_batch.clicked.connect(self._start_batch_measurement)
        n = sum(1 for r in self._results.values() if 'error' not in r)
        self.lbl_status.setText(f"Batch complete. {n} images measured.")
        QMessageBox.information(self, "Batch Done",
                                f"Measured {n} / {len(self._image_files)} images.\n"
                                f"Use Export CSV to save results.")

    # ── camera ─────────────────────────────────────────────────────────────
    def _toggle_camera(self):
        if self._camera_thread and self._camera_thread.isRunning():
            self._camera_thread.stop()
            self._camera_thread = None
            self.btn_cam.setText("Camera")
            self._camera_mode = False
            self.lbl_status.setText("Camera stopped")
        else:
            self._camera_mode = True
            self._camera_thread = CameraThread(0)
            self._camera_thread.frame_ready.connect(self._on_camera_frame)
            self._camera_thread.error.connect(
                lambda e: self.lbl_status.setText(e))
            self._camera_thread.start()
            self.btn_cam.setText("Stop Cam")
            self.lbl_fname.setText("[Camera]")
            self.lbl_status.setText("Camera running — click 'Capture' to freeze frame")

    def _on_camera_frame(self, frame: np.ndarray):
        if not self._camera_mode:
            return
        self._raw_image = frame
        self._apply_processing()
        self.image_canvas.set_image(frame)
        self._schedule_measurement()

    def _capture_frame(self):
        if self._raw_image is None:
            return
        self._camera_captured = self._raw_image.copy()
        # stop camera
        if self._camera_thread:
            self._camera_thread.stop()
            self._camera_thread = None
            self.btn_cam.setText("Camera")
            self._camera_mode = False
        self.lbl_status.setText("Frame captured. Processing...")
        self._apply_processing()
        self._schedule_measurement()

    # ── export ─────────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self._results:
            QMessageBox.information(self, "No Data", "No measurements to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "measurements.csv", "CSV files (*.csv)")
        if not path:
            return
        rows = []
        for fp, r in self._results.items():
            if 'error' in r:
                rows.append({'filename': r.get('filename', fp), 'error': r['error']})
                continue
            base = {
                'filename':         r.get('filename', ''),
                'angle_deg':        r.get('angle', 0),
                'pixels_per_um':    r.get('pixels_per_um', 1),
                'dark_median_um':   r.get('dark_median', 0),
                'bright_median_um': r.get('bright_median', 0),
                'ler_3sigma_um':    r.get('ler', 0),
                'lwr_3sigma_um':    r.get('lwr', 0),
                'pitch_um':         r.get('pitch', 0),
                'n_edges':          r.get('n_edges', 0),
                'dark_std_um':      float(np.std(r['dark_widths']))  if r.get('dark_widths')   else 0,
                'bright_std_um':    float(np.std(r['bright_widths'])) if r.get('bright_widths') else 0,
                'dark_min_um':      float(np.min(r['dark_widths']))  if r.get('dark_widths')   else 0,
                'dark_max_um':      float(np.max(r['dark_widths']))  if r.get('dark_widths')   else 0,
                'bright_min_um':    float(np.min(r['bright_widths'])) if r.get('bright_widths') else 0,
                'bright_max_um':    float(np.max(r['bright_widths'])) if r.get('bright_widths') else 0,
            }
            # Expand individual widths
            dw = r.get('dark_widths', [])
            bw = r.get('bright_widths', [])
            for i, w in enumerate(dw):
                base[f'dark_width_{i+1}_um'] = w
            for i, w in enumerate(bw):
                base[f'bright_width_{i+1}_um'] = w
            rows.append(base)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        self.lbl_status.setText(f"CSV exported: {path}")
        QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def _save_labeled_images(self):
        if not self._image_files:
            QMessageBox.information(self, "No Images", "No images loaded.")
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory")
        if not out_dir:
            return
        saved = 0
        for i, fp in enumerate(self._image_files):
            if fp not in self._results:
                continue
            # Load original
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # For current image, use canvas overlay
            if i == self._current_index:
                out_img = self.image_canvas.get_labeled_image()
            else:
                out_img = img
            if out_img is None:
                continue
            stem = Path(fp).stem
            out_path = str(Path(out_dir) / f"{stem}_labeled.png")
            cv2.imwrite(out_path, out_img)
            saved += 1

        self.lbl_status.setText(f"Saved {saved} labeled image(s) to {out_dir}")
        QMessageBox.information(self, "Done", f"Saved {saved} image(s) to:\n{out_dir}")

    # ── settings persist ───────────────────────────────────────────────────
    def _collect_settings(self) -> Dict:
        return {
            'scale':       self.spin_scale.value(),
            'angle':       self.spin_angle.value(),
            'algo':        self.combo_algo.currentIndex(),
            'sigma':       self.spin_sigma.value(),
            'peaks':       self.combo_peaks.currentIndex(),
            'threshold':   self.spin_threshold.value(),
            'channel':     self.combo_channel.currentIndex(),
            'smooth':      self.combo_smooth.currentIndex(),
            'smooth_k':    self.spin_smooth_k.value(),
            'cx':          self.image_canvas.cx,
            'cy':          self.image_canvas.cy,
            'radius':      self.image_canvas.radius,
            'directory':   self._directory,
        }

    def _apply_settings(self, s: Dict):
        self._block_signals(True)
        self.spin_scale.setValue(s.get('scale', 1.0))
        self.spin_angle.setValue(s.get('angle', 0.0))
        self.combo_algo.setCurrentIndex(s.get('algo', 0))
        self.spin_sigma.setValue(s.get('sigma', 2.0))
        self.combo_peaks.setCurrentIndex(s.get('peaks', 2))
        self.spin_threshold.setValue(s.get('threshold', 0.25))
        self.combo_channel.setCurrentIndex(s.get('channel', 0))
        self.combo_smooth.setCurrentIndex(s.get('smooth', 0))
        self.spin_smooth_k.setValue(s.get('smooth_k', 3))
        self.image_canvas.cx     = s.get('cx',     200.0)
        self.image_canvas.cy     = s.get('cy',     200.0)
        self.image_canvas.radius = s.get('radius',  80.0)
        self._sync_roi_spinboxes()
        self._block_signals(False)

    def _save_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Settings", SETTINGS_FILE, "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, 'w') as f:
                json.dump(self._collect_settings(), f, indent=2)
            self.lbl_status.setText(f"Settings saved: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _load_settings(self):
        """Load settings from default file on startup."""
        if Path(SETTINGS_FILE).exists():
            try:
                with open(SETTINGS_FILE) as f:
                    s = json.load(f)
                self._apply_settings(s)
                d = s.get('directory', '')
                if d and Path(d).is_dir():
                    self._directory = d
                    self._read_session_file()
                    self._load_directory(d)
            except Exception:
                pass

    def _load_settings_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path) as f:
                s = json.load(f)
            self._apply_settings(s)
            self.lbl_status.setText(f"Settings loaded: {path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    # ── about ──────────────────────────────────────────────────────────────
    def _show_about(self):
        QMessageBox.about(self, "About", (
            "<b>Microscope Line Width & LER Measurement Tool</b><br><br>"
            "Measures dark/bright stripe widths and Line Edge Roughness (3σ) "
            "from SEM/optical microscope images of parallel line patterns.<br><br>"
            "<b>Usage:</b><br>"
            "1. Open a directory with images<br>"
            "2. Drag the green circle over the stripe pattern<br>"
            "3. Auto-detect angle or enter manually<br>"
            "4. Select edge-detection algorithm and peak mode<br>"
            "5. Set scale (pixels/µm) for real-unit measurements<br>"
            "6. Export CSV and labeled images when done<br><br>"
            "© 2025"
        ))

    # ── close ──────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._save_session_current()
        # Save default settings
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self._collect_settings(), f, indent=2)
        except Exception:
            pass
        if self._camera_thread:
            self._camera_thread.stop()
        if self._batch_thread and self._batch_thread.isRunning():
            self._batch_thread.stop()
        # Shut down the measurement worker thread cleanly
        self._meas_worker.cancel()
        self._meas_thread.quit()
        self._meas_thread.wait(2000)
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")

    # Base dark palette for Fusion style
    palette = QPalette()
    palette.setColor(QPalette.Window,          QColor(30,  30,  30))
    palette.setColor(QPalette.WindowText,      QColor(212, 212, 212))
    palette.setColor(QPalette.Base,            QColor(25,  25,  25))
    palette.setColor(QPalette.AlternateBase,   QColor(40,  40,  40))
    palette.setColor(QPalette.ToolTipBase,     QColor(45,  45,  45))
    palette.setColor(QPalette.ToolTipText,     QColor(212, 212, 212))
    palette.setColor(QPalette.Text,            QColor(212, 212, 212))
    palette.setColor(QPalette.Button,          QColor(55,  55,  55))
    palette.setColor(QPalette.ButtonText,      QColor(212, 212, 212))
    palette.setColor(QPalette.BrightText,      Qt.red)
    palette.setColor(QPalette.Link,            QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight,       QColor(9,  71, 113))
    palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

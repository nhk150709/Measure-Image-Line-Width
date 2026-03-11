"""
main_window.py — Main application window for the SEM Stripe Analyzer.
"""
from __future__ import annotations

import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QListWidget, QListWidgetItem,
    QDockWidget, QToolBar, QStatusBar, QAction, QFileDialog,
    QMessageBox, QInputDialog, QLabel, QSizePolicy, QVBoxLayout,
    QAbstractItemView,
)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QIcon, QKeySequence

import numpy as np

from core.image_loader import load_image, list_images_in_directory, ImageData
from core.calibration import Calibration
from core.preprocessor import Preprocessor
from core.angle_detector import detect_angle, rotate_image
from core.profile_extractor import extract_averaged_profile, extract_line_profile
from core.stripe_detector import detect_stripes, StripeDetectionResult
from core.measurements import compute_measurements_from_detection, MeasurementResult
from core.recipe import Recipe

from ui.image_canvas import ImageCanvas
from ui.roi_tools import RoiToolbar
from ui.measurement_panel import MeasurementPanel
from ui.profile_plot_widget import ProfilePlotWidget
from ui.recipe_panel import RecipePanel
from ui.batch_panel import BatchDialog
from export.csv_exporter import export_results_csv, export_single_result
from export.annotated_image import save_annotated_image

import pandas as pd


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SEM Stripe Analyzer")
        self.resize(1400, 860)

        # State
        self._image_dir: str | None = None
        self._image_paths: list[str] = []
        self._current_img: ImageData | None = None
        self._current_roi: tuple[int, int, int, int] | None = None
        self._rotated_image: np.ndarray | None = None
        self._detection: StripeDetectionResult | None = None
        self._result: MeasurementResult | None = None
        self._angle_deg: float = 0.0
        self._session_results: list[dict] = []

        self._build_ui()
        self._apply_dark_style()

    # ── UI Construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Central splitter: [image list | canvas+plot | panels]
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left: image file list ──────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("Images"))
        self._img_list = QListWidget()
        self._img_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._img_list.currentItemChanged.connect(self._on_image_selected)
        left_layout.addWidget(self._img_list)
        left_widget.setMaximumWidth(220)
        splitter.addWidget(left_widget)

        # ── Centre: canvas + profile plot ─────────────────────────
        centre = QSplitter(Qt.Vertical)

        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # Tool bar (vertical strip left of canvas)
        canvas_row = QSplitter(Qt.Horizontal)
        self._canvas = ImageCanvas()
        self._canvas.roi_defined.connect(self._on_roi_defined)
        self._canvas.line_defined.connect(self._on_line_defined)
        self._canvas.pixel_hover.connect(self._on_pixel_hover)
        self._canvas.stripe_clicked.connect(self._on_stripe_clicked)

        self._roi_toolbar = RoiToolbar(self._canvas)
        canvas_row.addWidget(self._roi_toolbar)
        canvas_row.addWidget(self._canvas)
        canvas_row.setSizes([44, 900])
        canvas_layout.addWidget(canvas_row)
        canvas_widget.setLayout(canvas_layout)
        centre.addWidget(canvas_widget)

        # Profile plot at bottom of centre
        self._profile_plot = ProfilePlotWidget()
        self._profile_plot.setMaximumHeight(220)
        centre.addWidget(self._profile_plot)
        centre.setSizes([600, 200])
        splitter.addWidget(centre)

        # ── Right: recipe + measurements ──────────────────────────
        right_splitter = QSplitter(Qt.Vertical)
        self._recipe_panel = RecipePanel()
        self._recipe_panel.recipe_changed.connect(self._on_recipe_changed)
        self._recipe_panel.run_analysis_requested.connect(self._run_analysis)
        right_splitter.addWidget(self._recipe_panel)

        self._meas_panel = MeasurementPanel()
        self._meas_panel.export_requested.connect(self._export_csv)
        right_splitter.addWidget(self._meas_panel)
        right_splitter.setSizes([420, 380])
        right_splitter.setMaximumWidth(280)
        splitter.addWidget(right_splitter)

        splitter.setSizes([200, 950, 260])

        # ── Menu bar ──────────────────────────────────────────────
        self._build_menus()

        # ── Status bar ────────────────────────────────────────────
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status_label = QLabel("Ready")
        self._coord_label = QLabel("")
        self._status.addWidget(self._status_label, 1)
        self._status.addPermanentWidget(self._coord_label)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        act_open = QAction("Open Directory…", self, shortcut="Ctrl+O")
        act_open.triggered.connect(self._open_directory)
        act_open_img = QAction("Open Single Image…", self, shortcut="Ctrl+Shift+O")
        act_open_img.triggered.connect(self._open_single_image)
        act_export = QAction("Export CSV…", self, shortcut="Ctrl+E")
        act_export.triggered.connect(self._export_csv)
        act_export_img = QAction("Save Annotated Image…", self)
        act_export_img.triggered.connect(self._save_annotated)
        act_quit = QAction("Quit", self, shortcut="Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addActions([act_open, act_open_img])
        file_menu.addSeparator()
        file_menu.addActions([act_export, act_export_img])
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        # View
        view_menu = mb.addMenu("&View")
        act_fit = QAction("Fit Image", self, shortcut="F")
        act_fit.triggered.connect(self._canvas.fit_in_view)
        act_clear_ov = QAction("Clear Overlays", self, shortcut="Escape")
        act_clear_ov.triggered.connect(self._clear_overlays)
        act_psd = QAction("Show Edge PSD Plot", self)
        act_psd.triggered.connect(self._show_psd)
        view_menu.addActions([act_fit, act_clear_ov, act_psd])

        # Analysis
        analysis_menu = mb.addMenu("&Analysis")
        act_run = QAction("▶ Run Analysis", self, shortcut="Ctrl+R")
        act_run.triggered.connect(self._run_analysis)
        act_batch = QAction("Batch Processing…", self, shortcut="Ctrl+B")
        act_batch.triggered.connect(self._open_batch)
        act_cal = QAction("Set Scale from Scale Bar…", self)
        act_cal.triggered.connect(self._set_scale_from_scalebar)
        analysis_menu.addActions([act_run, act_batch])
        analysis_menu.addSeparator()
        analysis_menu.addAction(act_cal)

        # Recipe
        recipe_menu = mb.addMenu("&Recipe")
        act_save_r = QAction("Save Recipe…", self, shortcut="Ctrl+S")
        act_save_r.triggered.connect(self._save_recipe)
        act_load_r = QAction("Load Recipe…", self, shortcut="Ctrl+L")
        act_load_r.triggered.connect(self._load_recipe)
        recipe_menu.addActions([act_save_r, act_load_r])

        # Help
        help_menu = mb.addMenu("&Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ── Slots ─────────────────────────────────────────────────────────

    @pyqtSlot()
    def _open_directory(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if d:
            self._image_dir = d
            self._image_paths = list_images_in_directory(d)
            self._img_list.clear()
            for p in self._image_paths:
                self._img_list.addItem(QListWidgetItem(os.path.basename(p)))
            self._status_label.setText(
                f"Loaded {len(self._image_paths)} image(s) from: {d}"
            )
            if self._image_paths:
                self._img_list.setCurrentRow(0)

    @pyqtSlot()
    def _open_single_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)"
        )
        if path:
            self._image_dir = os.path.dirname(path)
            self._image_paths = [path]
            self._img_list.clear()
            self._img_list.addItem(QListWidgetItem(os.path.basename(path)))
            self._img_list.setCurrentRow(0)

    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def _on_image_selected(self, current: QListWidgetItem, _) -> None:
        if current is None:
            return
        idx = self._img_list.row(current)
        if 0 <= idx < len(self._image_paths):
            self._load_image(self._image_paths[idx])

    def _load_image(self, path: str) -> None:
        try:
            self._current_img = load_image(path)
            self._canvas.set_image(self._current_img.pixels)
            self._canvas.clear_overlays()
            self._current_roi = None
            self._detection = None
            self._result = None
            self._meas_panel.clear()
            self._profile_plot.clear()

            # Auto-apply metadata scale
            if self._current_img.metadata_um_per_px is not None:
                recipe = self._recipe_panel.get_recipe()
                recipe.scale_um_per_px = self._current_img.metadata_um_per_px
                recipe.scale_source = "metadata"
                self._recipe_panel.set_recipe(recipe)
                self._status_label.setText(
                    f"Loaded: {self._current_img.filename}  |  "
                    f"Scale from metadata: {recipe.scale_um_per_px:.4f} µm/px"
                )
            else:
                self._status_label.setText(f"Loaded: {self._current_img.filename}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))

    @pyqtSlot(int, int, int, int)
    def _on_roi_defined(self, x: int, y: int, w: int, h: int) -> None:
        self._current_roi = (x, y, w, h)
        recipe = self._recipe_panel.get_recipe()
        recipe.set_roi_from_tuple(x, y, w, h)
        self._recipe_panel.set_recipe(recipe)
        self._status_label.setText(f"ROI set: ({x},{y}) {w}×{h} px")

    @pyqtSlot(int, int, int, int)
    def _on_line_defined(self, x1: int, y1: int, x2: int, y2: int) -> None:
        tool = self._canvas._current_tool
        if tool == ImageCanvas.TOOL_ANGLE_LINE:
            import math
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            recipe = self._recipe_panel.get_recipe()
            recipe.angle_mode = "manual"
            recipe.angle_offset_deg = angle
            self._recipe_panel.set_recipe(recipe)
            self._status_label.setText(f"Manual angle set: {angle:.2f}°")
        elif tool == ImageCanvas.TOOL_SCALE_BAR:
            self._calibrate_from_line(x1, y1, x2, y2)
        elif tool == ImageCanvas.TOOL_LINE and self._current_img is not None:
            # Extract and plot profile along this line
            distances, intensity = extract_line_profile(
                self._current_img.pixels, x1, y1, x2, y2, width=3
            )
            recipe = self._recipe_panel.get_recipe()
            cal = Calibration.from_dict(
                {"um_per_px": recipe.scale_um_per_px, "source": recipe.scale_source}
            )
            self._profile_plot.plot_profile(distances, intensity, um_per_px=cal.um_per_px)
            self._canvas.draw_measurement_line(x1, y1, x2, y2, label="Profile")

    def _calibrate_from_line(self, x1: int, y1: int, x2: int, y2: int) -> None:
        import math
        length_px = math.hypot(x2 - x1, y2 - y1)
        length_um, ok = QInputDialog.getDouble(
            self, "Scale Bar",
            f"Drawn line = {length_px:.1f} px.\nEnter physical length (µm):",
            1.0, 0.001, 1e6, 3
        )
        if ok and length_px > 0:
            um_per_px = length_um / length_px
            recipe = self._recipe_panel.get_recipe()
            recipe.scale_um_per_px = um_per_px
            recipe.scale_source = "scalebar"
            self._recipe_panel.set_recipe(recipe)
            self._status_label.setText(
                f"Scale set from scale bar: {um_per_px:.5f} µm/px"
            )

    @pyqtSlot(int, int, int)
    def _on_pixel_hover(self, x: int, y: int, intensity: int) -> None:
        recipe = self._recipe_panel.get_recipe()
        um_per_px = recipe.scale_um_per_px
        self._coord_label.setText(
            f"x={x} ({x*um_per_px:.2f}µm)  y={y} ({y*um_per_px:.2f}µm)  I={intensity}"
        )

    @pyqtSlot(float)
    def _on_stripe_clicked(self, x_px: float) -> None:
        """User clicked a specific stripe — highlight it and show its CD."""
        if self._detection is None:
            return
        # Find nearest stripe
        best = min(self._detection.stripes, key=lambda s: abs(s.center_px - x_px), default=None)
        if best is None:
            return
        recipe = self._recipe_panel.get_recipe()
        um_per_px = recipe.scale_um_per_px
        roi_x = self._current_roi[0] if self._current_roi else 0
        h = self._current_img.pixels.shape[0] if self._current_img else 512
        lx = int(best.left_edge_px) + roi_x
        rx = int(best.right_edge_px) + roi_x
        width_um = best.width_px * um_per_px
        self._canvas.draw_measurement_line(
            lx, h // 2, rx, h // 2,
            label=f"{width_um:.3f} µm",
            color=None,
        )
        self._status_label.setText(
            f"Selected {best.kind} stripe: CD = {width_um:.3f} µm  "
            f"({best.width_px:.1f} px)"
        )

    @pyqtSlot(Recipe)
    def _on_recipe_changed(self, recipe: Recipe) -> None:
        pass  # Could auto-run preview; currently on-demand

    @pyqtSlot()
    def _run_analysis(self) -> None:
        if self._current_img is None:
            QMessageBox.information(self, "No Image", "Please load an image first.")
            return

        self._status_label.setText("Running analysis…")
        try:
            recipe = self._recipe_panel.get_recipe()
            calibration = Calibration.from_dict(
                {"um_per_px": recipe.scale_um_per_px, "source": recipe.scale_source}
            )
            preprocessor = Preprocessor(
                filter_type=recipe.filter_type,
                filter_sigma=recipe.filter_sigma,
                contrast_enhance=recipe.contrast_enhance,
            )
            processed = preprocessor.process(self._current_img.pixels)
            roi = self._current_roi or recipe.roi_tuple

            # Angle detection
            roi_region = processed
            if roi:
                x, y, w, h = roi
                roi_region = processed[y: y + h, x: x + w]

            self._angle_deg = detect_angle(
                roi_region,
                method=recipe.angle_mode,
                user_angle=recipe.angle_offset_deg,
            ) + recipe.angle_offset_deg

            rotated = rotate_image(processed, self._angle_deg)
            self._rotated_image = rotated

            # Profile extraction
            positions, profile = extract_averaged_profile(
                rotated, roi=roi,
                n_lines=recipe.profile_lines,
                direction=recipe.profile_direction,
            )

            # Stripe detection
            self._detection = detect_stripes(
                profile, positions=positions,
                threshold_fraction=recipe.threshold_fraction,
                min_width_px=recipe.min_stripe_width_px,
                smoothing_sigma=recipe.smoothing_sigma,
            )
            self._detection.angle_deg = self._angle_deg

            # Compute measurements
            self._result = compute_measurements_from_detection(
                self._detection, calibration,
                image=rotated, roi=roi,
                edge_method=recipe.edge_method,
                edge_threshold_fraction=recipe.threshold_fraction,
            )

            # Update displays
            self._meas_panel.update_results(self._result, self._detection)
            self._profile_plot.plot_profile(
                positions, profile,
                stripes=self._detection.stripes,
                um_per_px=calibration.um_per_px,
            )

            roi_offset = (roi[0], roi[1]) if roi else (0, 0)
            img_h = self._current_img.pixels.shape[0]
            self._canvas.draw_stripe_overlays(
                self._detection.stripes, roi_offset=roi_offset, image_height=img_h
            )

            # Store result for export
            row = self._result.to_dict(
                image_file=self._current_img.filename
            )
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            row["recipe_name"] = recipe.name
            self._session_results.append(row)

            n_white = len(self._detection.white_stripes)
            n_black = len(self._detection.black_stripes)
            self._status_label.setText(
                f"Analysis done — {n_white} white, {n_black} black stripes | "
                f"Angle: {self._angle_deg:.2f}°"
            )

        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Analysis Error", str(exc))
            traceback.print_exc()
            self._status_label.setText("Analysis failed.")

    @pyqtSlot()
    def _export_csv(self) -> None:
        if not self._session_results:
            QMessageBox.information(self, "No Data", "Run analysis on at least one image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results CSV", "", "CSV Files (*.csv)"
        )
        if path:
            df = pd.DataFrame(self._session_results)
            export_results_csv(df, path)
            self._status_label.setText(f"Exported {len(df)} result(s) to {path}")

    @pyqtSlot()
    def _save_annotated(self) -> None:
        if self._current_img is None or self._detection is None:
            QMessageBox.information(self, "No Data", "Run analysis first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Annotated Image", "", "PNG Image (*.png)"
        )
        if path:
            recipe = self._recipe_panel.get_recipe()
            save_annotated_image(
                self._rotated_image or self._current_img.pixels,
                self._detection,
                self._current_roi,
                path,
                um_per_px=recipe.scale_um_per_px,
            )
            self._status_label.setText(f"Saved annotated image: {path}")

    @pyqtSlot()
    def _open_batch(self) -> None:
        recipe = self._recipe_panel.get_recipe()
        dialog = BatchDialog(recipe, parent=self)
        dialog.exec_()

    @pyqtSlot()
    def _clear_overlays(self) -> None:
        self._canvas.clear_overlays()
        self._current_roi = None
        self._detection = None

    @pyqtSlot()
    def _show_psd(self) -> None:
        if self._result is None:
            return
        freqs, psd = self._result.roughness.psd(
            sampling_um=self._recipe_panel.get_recipe().scale_um_per_px
        )
        self._profile_plot.plot_psd(freqs, psd)

    @pyqtSlot()
    def _save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recipe", "", "JSON Recipe (*.json)"
        )
        if path:
            self._recipe_panel.get_recipe().save(path)

    @pyqtSlot()
    def _load_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "", "JSON Recipe (*.json)"
        )
        if path:
            try:
                recipe = Recipe.load(path)
                self._recipe_panel.set_recipe(recipe)
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", str(exc))

    @pyqtSlot()
    def _set_scale_from_scalebar(self) -> None:
        self._canvas.set_tool(ImageCanvas.TOOL_SCALE_BAR)
        self._status_label.setText(
            "Draw a line over a known scale bar feature, then enter the length."
        )

    @pyqtSlot()
    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About SEM Stripe Analyzer",
            "<b>SEM Stripe Analyzer</b><br>"
            "Version 1.0<br><br>"
            "Measures white (photo resist) and black (spacing) stripe widths,<br>"
            "Line Edge Roughness (LER), and Line Width Roughness (LWR)<br>"
            "in SEM images.<br><br>"
            "Supports batch processing, recipe save/load, and CSV export.",
        )

    # ── Styling ───────────────────────────────────────────────────────

    def _apply_dark_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e1e;
                color: #ddd;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                font-size: 11px;
                color: #aaa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
            QPushButton {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ddd;
            }
            QPushButton:hover { background-color: #444; }
            QPushButton:pressed { background-color: #222; }
            QListWidget {
                background-color: #252525;
                border: 1px solid #444;
                color: #ddd;
            }
            QListWidget::item:selected { background-color: #3a5a8a; }
            QTableWidget {
                background-color: #252525;
                alternate-background-color: #2a2a2a;
                color: #ddd;
                gridline-color: #444;
            }
            QHeaderView::section {
                background-color: #333;
                color: #aaa;
                border: 1px solid #444;
                padding: 2px;
            }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
                background-color: #2d2d2d;
                border: 1px solid #555;
                color: #ddd;
                padding: 2px;
            }
            QToolButton {
                background-color: #333;
                border: 1px solid #555;
                border-radius: 3px;
                color: #ddd;
            }
            QToolButton:checked {
                background-color: #3a5a8a;
                border: 1px solid #6699cc;
            }
            QSplitter::handle { background-color: #444; }
            QStatusBar { color: #aaa; font-size: 11px; }
            QMenuBar {
                background-color: #2b2b2b;
                color: #ddd;
            }
            QMenuBar::item:selected { background-color: #3a5a8a; }
            QMenu {
                background-color: #2b2b2b;
                color: #ddd;
                border: 1px solid #555;
            }
            QMenu::item:selected { background-color: #3a5a8a; }
            QProgressBar {
                border: 1px solid #555;
                background-color: #2d2d2d;
                color: #ddd;
                text-align: center;
            }
            QProgressBar::chunk { background-color: #3a7a3a; }
        """)

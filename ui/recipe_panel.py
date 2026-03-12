"""
recipe_panel.py — Recipe editor side panel.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton,
    QHBoxLayout, QFileDialog, QMessageBox, QScrollArea,
)
from PyQt5.QtCore import pyqtSignal, Qt

from core.recipe import Recipe


class RecipePanel(QWidget):
    recipe_changed = pyqtSignal(Recipe)
    run_analysis_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recipe = Recipe()
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        # ── Scale ──────────────────────────────────────────────────────
        grp = QGroupBox("Scale")
        f = QFormLayout(grp)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.00001, 1000.0)
        self._scale_spin.setDecimals(5)
        self._scale_spin.setValue(0.1)
        self._scale_spin.setSuffix(" µm/px")
        f.addRow("µm/px:", self._scale_spin)
        layout.addWidget(grp)

        # ── Angle detection ────────────────────────────────────────────
        grp = QGroupBox("Angle Detection")
        f = QFormLayout(grp)
        self._angle_mode = QComboBox()
        self._angle_mode.addItems(["Hough", "FFT", "Manual"])
        self._angle_offset = QDoubleSpinBox()
        self._angle_offset.setRange(-90.0, 90.0)
        self._angle_offset.setDecimals(2)
        self._angle_offset.setSuffix("°")
        f.addRow("Method:", self._angle_mode)
        f.addRow("Angle offset:", self._angle_offset)
        layout.addWidget(grp)

        # ── Preprocessing ──────────────────────────────────────────────
        grp = QGroupBox("Preprocessing")
        f = QFormLayout(grp)
        self._filter_type = QComboBox()
        self._filter_type.addItems(["gaussian", "median", "bilateral", "none"])
        self._filter_sigma = QDoubleSpinBox()
        self._filter_sigma.setRange(0.1, 20.0)
        self._filter_sigma.setValue(2.0)
        self._contrast = QComboBox()
        self._contrast.addItems(["none", "clahe", "histogram_eq", "normalize"])
        f.addRow("Filter:", self._filter_type)
        f.addRow("Sigma:", self._filter_sigma)
        f.addRow("Contrast:", self._contrast)
        layout.addWidget(grp)

        # ── Crop (per-side, applied before display & analysis) ─────────
        grp = QGroupBox("Crop — applied before display & analysis")
        f = QFormLayout(grp)
        def _crop_sb():
            sb = QSpinBox()
            sb.setRange(0, 4000)
            sb.setSuffix(" px")
            return sb
        self._crop_top    = _crop_sb()
        self._crop_bottom = _crop_sb()
        self._crop_left   = _crop_sb()
        self._crop_right  = _crop_sb()
        f.addRow("Top:", self._crop_top)
        f.addRow("Bottom:", self._crop_bottom)
        f.addRow("Left:", self._crop_left)
        f.addRow("Right:", self._crop_right)
        layout.addWidget(grp)

        # ── Edge & stripe detection ────────────────────────────────────
        grp = QGroupBox("Edge & Stripe Detection")
        f = QFormLayout(grp)

        self._edge_detect_method = QComboBox()
        self._edge_detect_method.addItems(["gradient_peaks", "zero_crossing"])
        self._edge_detect_method.setToolTip(
            "gradient_peaks  — peaks of 1st derivative.\n"
            "  ▸ Recommended for sharp SEM interfaces. Each B→W transition\n"
            "    produces a positive peak; each W→B produces a negative peak.\n\n"
            "zero_crossing   — zero-crossings of 2nd derivative.\n"
            "  ▸ Locates the inflection point of each sigmoid transition.\n"
            "    Better for smooth / gradual edges; more sensitive to noise."
        )

        self._edge_pairing = QComboBox()
        self._edge_pairing.addItems([
            "rising_falling",
            "falling_rising",
            "rising_rising",
            "falling_falling",
        ])
        self._edge_pairing.setToolTip(
            "How consecutive edges are paired to form one stripe interval:\n\n"
            "rising_falling   B→W then W→B  →  white stripe CD  (photoresist lines)\n"
            "falling_rising   W→B then B→W  →  black stripe CD  (spaces / trenches)\n"
            "rising_rising    B→W then B→W  →  full pitch  (one period)\n"
            "falling_falling  W→B then W→B  →  full pitch  (one period)"
        )

        self._min_edge_dist = QDoubleSpinBox()
        self._min_edge_dist.setRange(0.001, 500.0)
        self._min_edge_dist.setDecimals(3)
        self._min_edge_dist.setValue(0.05)
        self._min_edge_dist.setSuffix(" µm")
        self._min_edge_dist.setToolTip(
            "Minimum allowed distance between two consecutive detected edges.\n"
            "Edges closer than this are suppressed (noise / halo rejection).\n"
            "Raise if spurious edges appear within a single physical stripe."
        )

        self._smoothing = QDoubleSpinBox()
        self._smoothing.setRange(0.1, 20.0)
        self._smoothing.setValue(2.0)
        self._smoothing.setToolTip(
            "Gaussian smoothing sigma applied to the profile before differentiation.\n"
            "Higher values reduce noise sensitivity but may blur closely-spaced edges."
        )

        self._prominence = QDoubleSpinBox()
        self._prominence.setRange(0.01, 1.0)
        self._prominence.setDecimals(2)
        self._prominence.setSingleStep(0.05)
        self._prominence.setValue(0.15)
        self._prominence.setToolTip(
            "Minimum edge prominence as a fraction of the total intensity contrast.\n"
            "Raise to ignore weak / noisy edges; lower to detect subtle transitions."
        )

        self._profile_lines = QSpinBox()
        self._profile_lines.setRange(1, 200)
        self._profile_lines.setValue(20)
        self._profile_lines.setToolTip(
            "Number of image rows (or columns) averaged to build the 1D profile."
        )

        self._profile_direction = QComboBox()
        self._profile_direction.addItems(["horizontal", "vertical"])
        self._profile_direction.setToolTip(
            "horizontal — profile runs left→right  (for vertical stripes)\n"
            "vertical   — profile runs top→bottom  (for horizontal stripes)"
        )

        f.addRow("Detection method:", self._edge_detect_method)
        f.addRow("Edge pairing:", self._edge_pairing)
        f.addRow("Min edge distance:", self._min_edge_dist)
        f.addRow("Smoothing sigma:", self._smoothing)
        f.addRow("Prominence:", self._prominence)
        f.addRow("Profile lines:", self._profile_lines)
        f.addRow("Profile direction:", self._profile_direction)
        layout.addWidget(grp)

        # ── Derivative plot ────────────────────────────────────────────
        grp = QGroupBox("Derivative Plot")
        f = QFormLayout(grp)
        self._deriv_order = QComboBox()
        self._deriv_order.addItems(["None", "1st derivative", "2nd derivative"])
        self._deriv_order.setCurrentIndex(1)
        self._deriv_order.setToolTip(
            "Show a derivative of the intensity profile in a second subplot.\n\n"
            "1st derivative  — highlights B→W (positive peak) and W→B (negative peak).\n"
            "  Orange dots = rising edges,  red dots = falling edges.\n\n"
            "2nd derivative  — shows where the gradient is changing fastest;\n"
            "  zero-crossings mark the inflection points of edge transitions."
        )
        f.addRow("Show:", self._deriv_order)
        layout.addWidget(grp)

        # ── LER / edge detection ───────────────────────────────────────
        grp = QGroupBox("LER / Edge Detection")
        f = QFormLayout(grp)
        self._edge_method = QComboBox()
        self._edge_method.addItems(["canny", "threshold", "sigmoid"])
        self._ler_threshold = QDoubleSpinBox()
        self._ler_threshold.setRange(0.05, 0.95)
        self._ler_threshold.setDecimals(2)
        self._ler_threshold.setSingleStep(0.05)
        self._ler_threshold.setValue(0.5)
        self._ler_threshold.setToolTip(
            "Threshold fraction used by the LER edge finder\n"
            "(threshold / canny methods)."
        )
        f.addRow("Method:", self._edge_method)
        f.addRow("Edge threshold:", self._ler_threshold)
        layout.addWidget(grp)

        # ── Actions ───────────────────────────────────────────────────
        btn_run = QPushButton("▶ Run Analysis")
        btn_run.setStyleSheet("font-weight: bold; background-color: #2d6a2d;")
        btn_run.clicked.connect(self.run_analysis_requested)
        layout.addWidget(btn_run)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save Recipe…")
        btn_load = QPushButton("Load Recipe…")
        btn_save.clicked.connect(self._save_recipe)
        btn_load.clicked.connect(self._load_recipe)
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_load)
        layout.addLayout(btn_row)
        layout.addStretch()

        # ── Connect change signals ────────────────────────────────────
        for w in [
            self._scale_spin, self._angle_offset, self._filter_sigma,
            self._smoothing, self._prominence, self._ler_threshold,
            self._min_edge_dist,
            self._crop_top, self._crop_bottom, self._crop_left, self._crop_right,
            self._profile_lines,
        ]:
            w.valueChanged.connect(self._emit_changed)
        for cb in [
            self._angle_mode, self._filter_type, self._contrast, self._edge_method,
            self._edge_detect_method, self._edge_pairing, self._deriv_order,
            self._profile_direction,
        ]:
            cb.currentTextChanged.connect(self._emit_changed)

    # ── Internal ──────────────────────────────────────────────────────

    def _emit_changed(self, *_) -> None:
        self._recipe = self._build_recipe()
        self.recipe_changed.emit(self._recipe)

    def _build_recipe(self) -> Recipe:
        r = Recipe()
        r.scale_um_per_px = self._scale_spin.value()
        mode_map = {0: "hough", 1: "fft", 2: "manual"}
        r.angle_mode = mode_map.get(self._angle_mode.currentIndex(), "hough")
        r.angle_offset_deg = self._angle_offset.value()
        r.filter_type = self._filter_type.currentText()
        r.filter_sigma = self._filter_sigma.value()
        r.contrast_enhance = self._contrast.currentText()
        r.crop_top_px    = self._crop_top.value()
        r.crop_bottom_px = self._crop_bottom.value()
        r.crop_left_px   = self._crop_left.value()
        r.crop_right_px  = self._crop_right.value()
        r.edge_detect_method   = self._edge_detect_method.currentText()
        r.edge_pairing         = self._edge_pairing.currentText()
        r.min_edge_distance_um = self._min_edge_dist.value()
        r.smoothing_sigma      = self._smoothing.value()
        r.prominence_fraction  = self._prominence.value()
        r.profile_lines        = self._profile_lines.value()
        r.profile_direction    = self._profile_direction.currentText()
        deriv_map = {"None": 0, "1st derivative": 1, "2nd derivative": 2}
        r.show_derivative_order = deriv_map.get(self._deriv_order.currentText(), 1)
        r.edge_method        = self._edge_method.currentText()
        r.threshold_fraction = self._ler_threshold.value()
        return r

    def get_recipe(self) -> Recipe:
        return self._build_recipe()

    def set_recipe(self, recipe: Recipe) -> None:
        """Load recipe values into UI widgets without triggering extra signals."""
        all_spins = [
            self._scale_spin, self._angle_offset, self._filter_sigma,
            self._smoothing, self._prominence, self._ler_threshold,
            self._min_edge_dist,
            self._crop_top, self._crop_bottom, self._crop_left, self._crop_right,
            self._profile_lines,
        ]
        all_combos = [
            self._angle_mode, self._filter_type, self._contrast, self._edge_method,
            self._edge_detect_method, self._edge_pairing, self._deriv_order,
            self._profile_direction,
        ]
        for w in all_spins:
            w.blockSignals(True)
        for cb in all_combos:
            cb.blockSignals(True)

        self._scale_spin.setValue(recipe.scale_um_per_px)
        mode_idx = {"hough": 0, "fft": 1, "manual": 2}.get(recipe.angle_mode, 0)
        self._angle_mode.setCurrentIndex(mode_idx)
        self._angle_offset.setValue(recipe.angle_offset_deg)
        self._filter_type.setCurrentText(recipe.filter_type)
        self._filter_sigma.setValue(recipe.filter_sigma)
        self._contrast.setCurrentText(recipe.contrast_enhance)
        self._crop_top.setValue(recipe.crop_top_px)
        self._crop_bottom.setValue(recipe.crop_bottom_px)
        self._crop_left.setValue(recipe.crop_left_px)
        self._crop_right.setValue(recipe.crop_right_px)
        self._edge_detect_method.setCurrentText(recipe.edge_detect_method)
        self._edge_pairing.setCurrentText(recipe.edge_pairing)
        self._min_edge_dist.setValue(recipe.min_edge_distance_um)
        self._smoothing.setValue(recipe.smoothing_sigma)
        self._prominence.setValue(recipe.prominence_fraction)
        self._profile_lines.setValue(recipe.profile_lines)
        self._profile_direction.setCurrentText(recipe.profile_direction)
        deriv_map_rev = {0: "None", 1: "1st derivative", 2: "2nd derivative"}
        self._deriv_order.setCurrentText(
            deriv_map_rev.get(recipe.show_derivative_order, "1st derivative")
        )
        self._edge_method.setCurrentText(recipe.edge_method)
        self._ler_threshold.setValue(recipe.threshold_fraction)

        for w in all_spins:
            w.blockSignals(False)
        for cb in all_combos:
            cb.blockSignals(False)

    def _save_recipe(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recipe", "", "JSON Recipe (*.json)"
        )
        if path:
            self._build_recipe().save(path)

    def _load_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Recipe", "", "JSON Recipe (*.json)"
        )
        if path:
            try:
                r = Recipe.load(path)
                self.set_recipe(r)
                self._emit_changed()
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", str(exc))

"""
recipe_panel.py — Dock panel for viewing/editing recipe parameters.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox,
    QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QMessageBox, QCheckBox, QLineEdit,
)
from PyQt5.QtCore import pyqtSignal

from core.recipe import Recipe


class RecipePanel(QWidget):
    """
    Displays and edits the current recipe.  Emits recipe_changed when any
    parameter is changed by the user.
    """
    recipe_changed = pyqtSignal(Recipe)
    run_analysis_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recipe = Recipe()
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Calibration ───────────────────────────────────────────────
        grp_cal = QGroupBox("Calibration")
        cal_form = QFormLayout(grp_cal)
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(1e-6, 1000.0)
        self._scale_spin.setDecimals(6)
        self._scale_spin.setSingleStep(0.01)
        self._scale_spin.setValue(0.1)
        self._scale_spin.setSuffix(" µm/px")
        cal_form.addRow("Scale:", self._scale_spin)
        layout.addWidget(grp_cal)

        # ── Angle ─────────────────────────────────────────────────────
        grp_ang = QGroupBox("Stripe Angle")
        ang_form = QFormLayout(grp_ang)
        self._angle_mode = QComboBox()
        self._angle_mode.addItems(["Auto (Hough)", "Auto (FFT)", "Manual"])
        ang_form.addRow("Mode:", self._angle_mode)
        self._angle_offset = QDoubleSpinBox()
        self._angle_offset.setRange(-90.0, 90.0)
        self._angle_offset.setDecimals(2)
        self._angle_offset.setSuffix(" °")
        ang_form.addRow("Offset / manual:", self._angle_offset)
        layout.addWidget(grp_ang)

        # ── Preprocessing ─────────────────────────────────────────────
        grp_pre = QGroupBox("Preprocessing")
        pre_form = QFormLayout(grp_pre)
        self._filter_type = QComboBox()
        self._filter_type.addItems(["gaussian", "median", "bilateral", "none"])
        self._filter_sigma = QDoubleSpinBox()
        self._filter_sigma.setRange(0.1, 20.0)
        self._filter_sigma.setSingleStep(0.5)
        self._filter_sigma.setValue(2.0)
        self._contrast = QComboBox()
        self._contrast.addItems(["none", "clahe", "histogram_eq", "normalize"])
        pre_form.addRow("Filter:", self._filter_type)
        pre_form.addRow("Filter sigma:", self._filter_sigma)
        pre_form.addRow("Contrast:", self._contrast)
        layout.addWidget(grp_pre)

        # ── Stripe Detection ──────────────────────────────────────────
        grp_sd = QGroupBox("Stripe Detection")
        sd_form = QFormLayout(grp_sd)
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.1, 0.9)
        self._threshold.setDecimals(2)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(0.5)
        self._min_width = QDoubleSpinBox()
        self._min_width.setRange(1.0, 500.0)
        self._min_width.setValue(5.0)
        self._min_width.setSuffix(" px")
        self._profile_lines = QSpinBox()
        self._profile_lines.setRange(1, 200)
        self._profile_lines.setValue(20)
        self._smoothing = QDoubleSpinBox()
        self._smoothing.setRange(0.1, 20.0)
        self._smoothing.setValue(2.0)
        sd_form.addRow("Threshold (fraction):", self._threshold)
        sd_form.addRow("Min stripe width:", self._min_width)
        sd_form.addRow("Profile lines:", self._profile_lines)
        sd_form.addRow("Smoothing sigma:", self._smoothing)
        layout.addWidget(grp_sd)

        # ── Edge Detection ────────────────────────────────────────────
        grp_ed = QGroupBox("Edge Detection (LER)")
        ed_form = QFormLayout(grp_ed)
        self._edge_method = QComboBox()
        self._edge_method.addItems(["canny", "threshold", "sigmoid"])
        ed_form.addRow("Method:", self._edge_method)
        layout.addWidget(grp_ed)

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

        # ── Connect changes ───────────────────────────────────────────
        for widget in [
            self._scale_spin, self._angle_offset, self._filter_sigma,
            self._threshold, self._min_width, self._smoothing,
        ]:
            widget.valueChanged.connect(self._emit_changed)
        for cb in [self._angle_mode, self._filter_type, self._contrast, self._edge_method]:
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
        r.threshold_fraction = self._threshold.value()
        r.min_stripe_width_px = self._min_width.value()
        r.profile_lines = self._profile_lines.value()
        r.smoothing_sigma = self._smoothing.value()
        r.edge_method = self._edge_method.currentText()
        return r

    def get_recipe(self) -> Recipe:
        return self._build_recipe()

    def set_recipe(self, recipe: Recipe) -> None:
        """Load recipe values into UI widgets (without triggering extra signals)."""
        for widget in [
            self._scale_spin, self._angle_offset, self._filter_sigma,
            self._threshold, self._min_width, self._smoothing,
        ]:
            widget.blockSignals(True)
        for cb in [self._angle_mode, self._filter_type, self._contrast, self._edge_method]:
            cb.blockSignals(True)

        self._scale_spin.setValue(recipe.scale_um_per_px)
        mode_idx = {"hough": 0, "fft": 1, "manual": 2}.get(recipe.angle_mode, 0)
        self._angle_mode.setCurrentIndex(mode_idx)
        self._angle_offset.setValue(recipe.angle_offset_deg)
        self._filter_type.setCurrentText(recipe.filter_type)
        self._filter_sigma.setValue(recipe.filter_sigma)
        self._contrast.setCurrentText(recipe.contrast_enhance)
        self._threshold.setValue(recipe.threshold_fraction)
        self._min_width.setValue(recipe.min_stripe_width_px)
        self._profile_lines.setValue(recipe.profile_lines)
        self._smoothing.setValue(recipe.smoothing_sigma)
        self._edge_method.setCurrentText(recipe.edge_method)

        for widget in [
            self._scale_spin, self._angle_offset, self._filter_sigma,
            self._threshold, self._min_width, self._smoothing,
        ]:
            widget.blockSignals(False)
        for cb in [self._angle_mode, self._filter_type, self._contrast, self._edge_method]:
            cb.blockSignals(False)

        self._recipe = recipe

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
                recipe = Recipe.load(path)
                self.set_recipe(recipe)
                self._recipe = recipe
                self.recipe_changed.emit(recipe)
            except Exception as exc:
                QMessageBox.warning(self, "Load Error", str(exc))

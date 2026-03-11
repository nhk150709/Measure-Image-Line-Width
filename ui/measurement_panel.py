"""
measurement_panel.py — Right-side panel showing measurement results and stats.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QPushButton, QGroupBox, QHBoxLayout, QSizePolicy, QHeaderView,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from core.measurements import MeasurementResult


_LABEL_STYLE = "font-size: 11px; color: #aaa;"
_VALUE_STYLE = "font-size: 13px; font-weight: bold; color: #fff;"


class ValueRow(QWidget):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        self._lbl = QLabel(label)
        self._lbl.setStyleSheet(_LABEL_STYLE)
        self._val = QLabel("—")
        self._val.setStyleSheet(_VALUE_STYLE)
        self._val.setAlignment(Qt.AlignRight)
        layout.addWidget(self._lbl)
        layout.addStretch()
        layout.addWidget(self._val)

    def set_value(self, val: float | None, unit: str = "µm", decimals: int = 3) -> None:
        if val is None:
            self._val.setText("—")
        else:
            self._val.setText(f"{val:.{decimals}f} {unit}")

    def set_text(self, text: str) -> None:
        self._val.setText(text)


class MeasurementPanel(QWidget):
    export_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # ── Summary group ─────────────────────────────────────────────
        grp_cd = QGroupBox("Critical Dimensions")
        grp_layout = QVBoxLayout(grp_cd)
        self._white_mean = ValueRow("White CD (mean)")
        self._white_std  = ValueRow("White CD (±std)")
        self._black_mean = ValueRow("Black CD (mean)")
        self._black_std  = ValueRow("Black CD (±std)")
        self._pitch_mean = ValueRow("Pitch (mean)")
        for w in [self._white_mean, self._white_std,
                  self._black_mean, self._black_std, self._pitch_mean]:
            grp_layout.addWidget(w)
        layout.addWidget(grp_cd)

        # ── Roughness group ───────────────────────────────────────────
        grp_ler = QGroupBox("Line Roughness")
        ler_layout = QVBoxLayout(grp_ler)
        self._ler_left  = ValueRow("LER left (3σ)")
        self._ler_right = ValueRow("LER right (3σ)")
        self._lwr       = ValueRow("LWR (3σ)")
        for w in [self._ler_left, self._ler_right, self._lwr]:
            ler_layout.addWidget(w)
        layout.addWidget(grp_ler)

        # ── Measurement info ──────────────────────────────────────────
        grp_info = QGroupBox("Info")
        info_layout = QVBoxLayout(grp_info)
        self._n_stripes = ValueRow("Stripes measured")
        self._angle     = ValueRow("Stripe angle")
        self._scale     = ValueRow("Scale")
        for w in [self._n_stripes, self._angle, self._scale]:
            info_layout.addWidget(w)
        layout.addWidget(grp_info)

        # ── Per-stripe table ──────────────────────────────────────────
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Type", "Width (µm)", "Left (µm)", "Right (µm)"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setMaximumHeight(200)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(QLabel("Per-stripe detail:"))
        layout.addWidget(self._table)

        # ── Export button ─────────────────────────────────────────────
        btn = QPushButton("Export CSV…")
        btn.clicked.connect(self.export_requested)
        layout.addWidget(btn)
        layout.addStretch()

    # ──────────────────────────────────────────────────────────────────

    def update_results(self, result: MeasurementResult, detection=None) -> None:
        self._white_mean.set_value(result.white_cd.mean)
        self._white_std.set_value(result.white_cd.std)
        self._black_mean.set_value(result.black_cd.mean)
        self._black_std.set_value(result.black_cd.std)
        self._pitch_mean.set_value(result.pitch.mean)

        self._ler_left.set_value(result.roughness.LER_left_3sigma)
        self._ler_right.set_value(result.roughness.LER_right_3sigma)
        self._lwr.set_value(result.roughness.LWR_3sigma)

        self._n_stripes.set_text(str(result.n_stripes_measured))
        self._angle.set_value(result.angle_deg, unit="°", decimals=2)
        self._scale.set_value(result.scale_um_per_px, unit="µm/px", decimals=4)

        if detection is not None:
            self._fill_table(detection.stripes, result.scale_um_per_px)

    def _fill_table(self, stripes, um_per_px: float) -> None:
        self._table.setRowCount(0)
        for stripe in stripes:
            row = self._table.rowCount()
            self._table.insertRow(row)
            kind_item = QTableWidgetItem(stripe.kind.capitalize())
            kind_item.setForeground(
                QColor(255, 120, 120) if stripe.kind == "white" else QColor(120, 120, 255)
            )
            self._table.setItem(row, 0, kind_item)
            self._table.setItem(row, 1, QTableWidgetItem(f"{stripe.width_px * um_per_px:.3f}"))
            self._table.setItem(row, 2, QTableWidgetItem(f"{stripe.left_edge_px * um_per_px:.3f}"))
            self._table.setItem(row, 3, QTableWidgetItem(f"{stripe.right_edge_px * um_per_px:.3f}"))

    def clear(self) -> None:
        for w in [self._white_mean, self._white_std, self._black_mean,
                  self._black_std, self._pitch_mean, self._ler_left,
                  self._ler_right, self._lwr]:
            w.set_value(None)
        self._n_stripes.set_text("—")
        self._angle.set_text("—")
        self._scale.set_text("—")
        self._table.setRowCount(0)

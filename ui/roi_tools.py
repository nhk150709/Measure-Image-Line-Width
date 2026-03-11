"""
roi_tools.py — Tool-button group for selecting the active canvas tool.
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QToolButton, QButtonGroup, QLabel,
    QSizePolicy,
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon

from ui.image_canvas import ImageCanvas


_TOOLS = [
    (ImageCanvas.TOOL_SELECT,     "↖",  "Select / Pan"),
    (ImageCanvas.TOOL_ROI_RECT,   "▭",  "Rectangle ROI\n(defines analysis region)"),
    (ImageCanvas.TOOL_ROI_CIRCLE, "◯",  "Circle ROI"),
    (ImageCanvas.TOOL_LINE,       "╱",  "Line Profile\n(draw line → extract profile)"),
    (ImageCanvas.TOOL_ANGLE_LINE, "∠",  "Set Stripe Angle\n(draw along a stripe)"),
    (ImageCanvas.TOOL_SCALE_BAR,  "↔",  "Scale Bar\n(draw over known length)"),
    (ImageCanvas.TOOL_PICK_STRIPE,"✦",  "Pick Stripe\n(click a stripe to measure it)"),
]


class RoiToolbar(QWidget):
    """Vertical toolbar of tool buttons."""
    tool_changed = pyqtSignal(str)

    def __init__(self, canvas: ImageCanvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Tools")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for tool_id, symbol, tooltip in _TOOLS:
            btn = QToolButton()
            btn.setText(symbol)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setFixedSize(36, 36)
            btn.setStyleSheet("font-size: 16px;")
            self._group.addButton(btn)
            layout.addWidget(btn)
            btn.clicked.connect(lambda checked, tid=tool_id: self._on_tool(tid))

        # Default: select
        self._group.buttons()[0].setChecked(True)
        layout.addStretch()
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def _on_tool(self, tool_id: str) -> None:
        self._canvas.set_tool(tool_id)
        self.tool_changed.emit(tool_id)

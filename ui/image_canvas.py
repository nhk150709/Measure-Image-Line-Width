"""
image_canvas.py — Interactive image display widget.

Features:
  - Zoom (Ctrl+scroll) and Pan (middle-button drag or space+left drag)
  - Overlay items: ROI rectangles, measurement lines, stripe markers, scale bar
  - Cursor crosshair with pixel-coordinate readout
"""
from __future__ import annotations

from PyQt5.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsLineItem, QGraphicsEllipseItem,
    QGraphicsTextItem,
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal, QLineF
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPen, QBrush, QFont, QCursor,
    QTransform, QPainter,
)
import numpy as np


class ImageCanvas(QGraphicsView):
    """
    Central image display with zoom, pan, and overlay support.

    Signals
    -------
    roi_defined(int, int, int, int)        — emitted when user finishes drawing an ROI rect
    line_defined(int, int, int, int)       — emitted when user finishes drawing a line
    pixel_hover(int, int, int)             — (x, y, intensity) as mouse moves
    stripe_clicked(float)                  — x-position of a user-selected stripe click
    """
    roi_defined = pyqtSignal(int, int, int, int)
    line_defined = pyqtSignal(int, int, int, int)
    pixel_hover = pyqtSignal(int, int, int)
    stripe_clicked = pyqtSignal(float)

    # Tool modes
    TOOL_SELECT = "select"
    TOOL_ROI_RECT = "roi_rect"
    TOOL_ROI_CIRCLE = "roi_circle"
    TOOL_LINE = "line"
    TOOL_ANGLE_LINE = "angle_line"
    TOOL_SCALE_BAR = "scale_bar"
    TOOL_PICK_STRIPE = "pick_stripe"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_np: np.ndarray | None = None
        self._current_tool = self.TOOL_SELECT

        # Drawing state
        self._draw_start: QPointF | None = None
        self._temp_item = None

        # Overlays registry
        self._roi_items: list = []
        self._stripe_items: list = []
        self._measurement_items: list = []
        self._scalebar_item = None

        # Pan state
        self._panning = False
        self._pan_start_pos = None

        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setMouseTracking(True)

    # ── Image loading ─────────────────────────────────────────────────

    def set_image(self, image: np.ndarray) -> None:
        """Display a grayscale uint8 numpy array."""
        self._image_np = image
        h, w = image.shape
        q_image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)

        if self._pixmap_item is None:
            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._scene.addItem(self._pixmap_item)
        else:
            self._pixmap_item.setPixmap(pixmap)

        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view()

    def fit_in_view(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ── Tool selection ────────────────────────────────────────────────

    def set_tool(self, tool: str) -> None:
        self._current_tool = tool
        if tool == self.TOOL_SELECT:
            self.setCursor(Qt.ArrowCursor)
        elif tool in (self.TOOL_ROI_RECT, self.TOOL_ROI_CIRCLE,
                      self.TOOL_LINE, self.TOOL_ANGLE_LINE, self.TOOL_SCALE_BAR):
            self.setCursor(Qt.CrossCursor)
        elif tool == self.TOOL_PICK_STRIPE:
            self.setCursor(Qt.PointingHandCursor)

    # ── Overlay management ────────────────────────────────────────────

    def clear_overlays(self) -> None:
        for item in self._roi_items + self._stripe_items + self._measurement_items:
            self._scene.removeItem(item)
        self._roi_items.clear()
        self._stripe_items.clear()
        self._measurement_items.clear()
        if self._scalebar_item:
            self._scene.removeItem(self._scalebar_item)
            self._scalebar_item = None

    def clear_stripe_overlays(self) -> None:
        for item in self._stripe_items:
            self._scene.removeItem(item)
        self._stripe_items.clear()

    def draw_roi_rect(self, x: int, y: int, w: int, h: int) -> None:
        pen = QPen(QColor(0, 220, 220), 2, Qt.DashLine)
        item = self._scene.addRect(x, y, w, h, pen)
        self._roi_items.append(item)

    def draw_stripe_overlays(
        self, stripes, roi_offset=(0, 0),
        image_height=512, image_width=512,
        direction="horizontal", um_per_px: float = 1.0,
    ) -> None:
        """Draw coloured bands for each detected stripe with width labels.

        direction='horizontal' — vertical stripes; bands are vertical columns.
        direction='vertical'   — horizontal stripes; bands are horizontal rows.
        """
        self.clear_stripe_overlays()
        edge_pen = QPen(QColor(0, 255, 100), 1)
        label_color = QColor(255, 230, 0)
        label_font = QFont("Arial", 8)

        if direction == "horizontal":
            ox = roi_offset[0]
            h = image_height
            for stripe in stripes:
                lx = stripe.left_edge_px + ox
                rx = stripe.right_edge_px + ox
                color = QColor(255, 80, 80, 60) if stripe.kind == "white" else QColor(80, 80, 255, 60)
                brush = QBrush(color)
                band = self._scene.addRect(lx, 0, rx - lx, h, QPen(Qt.NoPen), brush)
                self._stripe_items.append(band)
                l_line = self._scene.addLine(lx, 0, lx, h, edge_pen)
                r_line = self._scene.addLine(rx, 0, rx, h, edge_pen)
                self._stripe_items.extend([l_line, r_line])
                # Width label at centre of band
                cx = (lx + rx) / 2
                cy = h / 4 if stripe.kind == "white" else 3 * h / 4
                txt = self._scene.addText(f"{stripe.width_px * um_per_px:.2f}µm", label_font)
                txt.setDefaultTextColor(label_color)
                txt.setPos(cx - txt.boundingRect().width() / 2, cy - 8)
                self._stripe_items.append(txt)
        else:  # "vertical" direction → horizontal stripes
            oy = roi_offset[1]
            w = image_width
            for stripe in stripes:
                ty = stripe.left_edge_px + oy
                by = stripe.right_edge_px + oy
                color = QColor(255, 80, 80, 60) if stripe.kind == "white" else QColor(80, 80, 255, 60)
                brush = QBrush(color)
                band = self._scene.addRect(0, ty, w, by - ty, QPen(Qt.NoPen), brush)
                self._stripe_items.append(band)
                t_line = self._scene.addLine(0, ty, w, ty, edge_pen)
                b_line = self._scene.addLine(0, by, w, by, edge_pen)
                self._stripe_items.extend([t_line, b_line])
                # Width label at centre of band
                cy = (ty + by) / 2
                cx = w / 4 if stripe.kind == "white" else 3 * w / 4
                txt = self._scene.addText(f"{stripe.width_px * um_per_px:.2f}µm", label_font)
                txt.setDefaultTextColor(label_color)
                txt.setPos(cx - txt.boundingRect().width() / 2, cy - 8)
                self._stripe_items.append(txt)

    def draw_measurement_line(self, x1: int, y1: int, x2: int, y2: int,
                              label: str = "", color: QColor | None = None) -> None:
        c = color or QColor(255, 200, 0)
        pen = QPen(c, 2)
        item = self._scene.addLine(x1, y1, x2, y2, pen)
        self._measurement_items.append(item)
        if label:
            txt = self._scene.addText(label, QFont("Arial", 9))
            txt.setDefaultTextColor(c)
            txt.setPos((x1 + x2) / 2, (y1 + y2) / 2 - 14)
            self._measurement_items.append(txt)

    def draw_scale_bar(self, x: int, y: int, length_px: int, label: str) -> None:
        pen = QPen(QColor(255, 255, 255), 3)
        bar = self._scene.addLine(x, y, x + length_px, y, pen)
        tick_l = self._scene.addLine(x, y - 5, x, y + 5, pen)
        tick_r = self._scene.addLine(x + length_px, y - 5, x + length_px, y + 5, pen)
        txt = self._scene.addText(label, QFont("Arial", 10))
        txt.setDefaultTextColor(QColor(255, 255, 255))
        txt.setPos(x, y + 8)
        self._scalebar_item = (bar, tick_l, tick_r, txt)

    # ── Mouse events ──────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if self._current_tool == self.TOOL_PICK_STRIPE:
                self.stripe_clicked.emit(scene_pos.x())
                return

            if self._current_tool in (
                self.TOOL_ROI_RECT, self.TOOL_ROI_CIRCLE,
                self.TOOL_LINE, self.TOOL_ANGLE_LINE, self.TOOL_SCALE_BAR
            ):
                self._draw_start = scene_pos
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Update pixel info
        scene_pos = self.mapToScene(event.pos())
        ix, iy = int(scene_pos.x()), int(scene_pos.y())
        if self._image_np is not None:
            h, w = self._image_np.shape
            if 0 <= ix < w and 0 <= iy < h:
                intensity = int(self._image_np[iy, ix])
                self.pixel_hover.emit(ix, iy, intensity)

        # Pan
        if self._panning and self._pan_start_pos is not None:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        # Rubber-band draw preview
        if self._draw_start and event.buttons() & Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if self._temp_item:
                self._scene.removeItem(self._temp_item)
                self._temp_item = None
            pen = QPen(QColor(0, 220, 220), 1, Qt.DashLine)
            sx, sy = self._draw_start.x(), self._draw_start.y()
            ex, ey = scene_pos.x(), scene_pos.y()
            if self._current_tool in (self.TOOL_ROI_RECT,):
                self._temp_item = self._scene.addRect(
                    min(sx, ex), min(sy, ey), abs(ex - sx), abs(ey - sy), pen
                )
            elif self._current_tool in (
                self.TOOL_LINE, self.TOOL_ANGLE_LINE, self.TOOL_SCALE_BAR
            ):
                self._temp_item = self._scene.addLine(sx, sy, ex, ey, pen)
            elif self._current_tool == self.TOOL_ROI_CIRCLE:
                r = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
                self._temp_item = self._scene.addEllipse(
                    sx - r, sy - r, 2 * r, 2 * r, pen
                )

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = False
            self.set_tool(self._current_tool)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._draw_start:
            scene_pos = self.mapToScene(event.pos())
            sx, sy = int(self._draw_start.x()), int(self._draw_start.y())
            ex, ey = int(scene_pos.x()), int(scene_pos.y())

            if self._temp_item:
                self._scene.removeItem(self._temp_item)
                self._temp_item = None

            if self._current_tool == self.TOOL_ROI_RECT:
                x, y = min(sx, ex), min(sy, ey)
                w, h = abs(ex - sx), abs(ey - sy)
                if w > 5 and h > 5:
                    self.draw_roi_rect(x, y, w, h)
                    self.roi_defined.emit(x, y, w, h)

            elif self._current_tool in (self.TOOL_LINE, self.TOOL_ANGLE_LINE,
                                        self.TOOL_SCALE_BAR):
                self.line_defined.emit(sx, sy, ex, ey)

            self._draw_start = None

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_F:
            self.fit_in_view()
        super().keyPressEvent(event)

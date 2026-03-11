"""
profile_plot_widget.py — Embedded matplotlib widget showing intensity profile.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure


class ProfilePlotWidget(QWidget):
    """Shows the averaged intensity profile with detected edge markers."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 2.5), tight_layout=True)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavToolbar(self._canvas, self))
        layout.addWidget(self._canvas)

        self._ax.set_xlabel("Position (px)")
        self._ax.set_ylabel("Intensity")
        self._ax.set_title("Intensity Profile")
        self._fig.patch.set_facecolor("#2b2b2b")
        self._ax.set_facecolor("#2b2b2b")
        self._ax.tick_params(colors="white")
        self._ax.xaxis.label.set_color("white")
        self._ax.yaxis.label.set_color("white")
        self._ax.title.set_color("white")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("gray")

    def plot_profile(
        self,
        positions: np.ndarray,
        profile: np.ndarray,
        stripes=None,
        um_per_px: float = 1.0,
        show_um: bool = True,
    ) -> None:
        """
        Plot the profile and optionally mark stripe boundaries.

        stripes : list of Stripe objects (from stripe_detector)
        """
        self._ax.cla()

        x = positions * um_per_px if show_um else positions
        xlabel = "Position (µm)" if show_um else "Position (px)"

        self._ax.plot(x, profile, color="#88ccff", linewidth=1, label="Profile")

        if stripes:
            for stripe in stripes:
                lx = stripe.left_edge_px * um_per_px if show_um else stripe.left_edge_px
                rx = stripe.right_edge_px * um_per_px if show_um else stripe.right_edge_px
                cx = stripe.center_px * um_per_px if show_um else stripe.center_px
                color = "#ff6666" if stripe.kind == "white" else "#6688ff"
                self._ax.axvspan(lx, rx, alpha=0.2, color=color)
                self._ax.axvline(lx, color="#00ff88", linewidth=0.8, linestyle="--")
                self._ax.axvline(rx, color="#00ff88", linewidth=0.8, linestyle="--")

        self._ax.set_xlabel(xlabel, color="white")
        self._ax.set_ylabel("Intensity", color="white")
        self._ax.set_title("Intensity Profile", color="white")
        self._ax.tick_params(colors="white")
        self._fig.patch.set_facecolor("#2b2b2b")
        self._ax.set_facecolor("#2b2b2b")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("gray")

        self._canvas.draw()

    def plot_psd(self, freqs: np.ndarray, psd: np.ndarray) -> None:
        """Plot Power Spectral Density of edge positions."""
        self._ax.cla()
        if len(freqs) == 0:
            self._canvas.draw()
            return
        valid = freqs > 0
        self._ax.loglog(freqs[valid], psd[valid], color="#ff9944")
        self._ax.set_xlabel("Spatial Frequency (1/µm)", color="white")
        self._ax.set_ylabel("PSD (µm²·µm)", color="white")
        self._ax.set_title("Edge PSD (LER)", color="white")
        self._ax.tick_params(colors="white")
        self._fig.patch.set_facecolor("#2b2b2b")
        self._ax.set_facecolor("#2b2b2b")
        for spine in self._ax.spines.values():
            spine.set_edgecolor("gray")
        self._canvas.draw()

    def clear(self) -> None:
        self._ax.cla()
        self._canvas.draw()

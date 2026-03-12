"""
profile_plot_widget.py — Embedded matplotlib widget showing intensity profile
and optionally the 1st or 2nd derivative below it.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

_BG = "#2b2b2b"
_FG = "white"


def _style_ax(ax) -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")


class ProfilePlotWidget(QWidget):
    """Shows the averaged intensity profile with detected edge markers,
    and optionally the 1st or 2nd derivative in a second subplot below."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fig = Figure(figsize=(5, 2.5), tight_layout=True)
        self._fig.patch.set_facecolor(_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(NavToolbar(self._canvas, self))
        layout.addWidget(self._canvas)

        # Initial single subplot
        ax = self._fig.add_subplot(111)
        ax.set_xlabel("Position (px)", color=_FG)
        ax.set_ylabel("Intensity", color=_FG)
        ax.set_title("Intensity Profile", color=_FG)
        _style_ax(ax)

    def plot_profile(
        self,
        positions: np.ndarray,
        profile: np.ndarray,
        stripes=None,
        um_per_px: float = 1.0,
        show_um: bool = True,
        derivative: np.ndarray | None = None,
        derivative_label: str = "Derivative",
        rising_edges: np.ndarray | None = None,
        falling_edges: np.ndarray | None = None,
    ) -> None:
        """
        Plot the intensity profile and, optionally, a derivative below it.

        Parameters
        ----------
        stripes          : list of Stripe objects; drawn as shaded spans + dashed edge lines
        derivative       : 1D array (grad1 or grad2); if None no second subplot is drawn
        derivative_label : y-axis / title label for the derivative subplot
        rising_edges     : indices of rising-edge positions (orange markers)
        falling_edges    : indices of falling-edge positions (red markers)
        """
        self._fig.clear()

        show_deriv = derivative is not None and len(derivative) > 0
        if show_deriv:
            ax1 = self._fig.add_subplot(211)
            ax2 = self._fig.add_subplot(212, sharex=ax1)
        else:
            ax1 = self._fig.add_subplot(111)
            ax2 = None

        x = positions * um_per_px if show_um else positions
        xlabel = "Position (µm)" if show_um else "Position (px)"

        # ── Intensity profile ─────────────────────────────────────────
        ax1.plot(x, profile, color="#88ccff", linewidth=1, label="Profile")

        if stripes:
            for stripe in stripes:
                lx = stripe.left_edge_px * um_per_px if show_um else stripe.left_edge_px
                rx = stripe.right_edge_px * um_per_px if show_um else stripe.right_edge_px
                color = "#ff6666" if stripe.kind == "white" else "#6688ff"
                ax1.axvspan(lx, rx, alpha=0.2, color=color)
                ax1.axvline(lx, color="#00ff88", linewidth=0.8, linestyle="--")
                ax1.axvline(rx, color="#00ff88", linewidth=0.8, linestyle="--")

        # Mark raw edge detections on the profile plot
        if rising_edges is not None and len(rising_edges):
            valid = rising_edges[rising_edges < len(positions)]
            for rv in (positions[valid] * um_per_px if show_um else positions[valid]):
                ax1.axvline(rv, color="#ffaa00", linewidth=0.7, linestyle=":",
                            alpha=0.8)
        if falling_edges is not None and len(falling_edges):
            valid = falling_edges[falling_edges < len(positions)]
            for fv in (positions[valid] * um_per_px if show_um else positions[valid]):
                ax1.axvline(fv, color="#ff4444", linewidth=0.7, linestyle=":",
                            alpha=0.8)

        ax1.set_ylabel("Intensity", color=_FG)
        if not show_deriv:
            ax1.set_xlabel(xlabel, color=_FG)
        ax1.set_title("Intensity Profile", color=_FG)
        _style_ax(ax1)

        # ── Derivative subplot ────────────────────────────────────────
        if show_deriv and ax2 is not None:
            deriv_x = positions * um_per_px if show_um else positions
            ax2.plot(deriv_x, derivative, color="#ffcc44", linewidth=1)
            ax2.axhline(0, color="gray", linewidth=0.5)

            if rising_edges is not None and len(rising_edges):
                valid = rising_edges[rising_edges < len(derivative)]
                rx_vals = positions[valid] * um_per_px if show_um else positions[valid]
                ax2.scatter(rx_vals, derivative[valid], color="#ffaa00",
                            s=35, zorder=5, label="Rising ↑")
            if falling_edges is not None and len(falling_edges):
                valid = falling_edges[falling_edges < len(derivative)]
                fx_vals = positions[valid] * um_per_px if show_um else positions[valid]
                ax2.scatter(fx_vals, derivative[valid], color="#ff4444",
                            s=35, zorder=5, label="Falling ↓")

            ax2.set_xlabel(xlabel, color=_FG)
            ax2.set_ylabel(derivative_label, color=_FG)
            ax2.set_title(derivative_label, color=_FG)
            _style_ax(ax2)
            if rising_edges is not None or falling_edges is not None:
                ax2.legend(fontsize=8, facecolor="#333", edgecolor="gray",
                           labelcolor=_FG)

        self._fig.patch.set_facecolor(_BG)
        self._fig.tight_layout()
        self._canvas.draw()

    def plot_psd(self, freqs: np.ndarray, psd: np.ndarray) -> None:
        """Plot Power Spectral Density of edge positions."""
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        if len(freqs) > 0:
            valid = freqs > 0
            ax.loglog(freqs[valid], psd[valid], color="#ff9944")
        ax.set_xlabel("Spatial Frequency (1/µm)", color=_FG)
        ax.set_ylabel("PSD (µm²·µm)", color=_FG)
        ax.set_title("Edge PSD (LER)", color=_FG)
        _style_ax(ax)
        self._fig.patch.set_facecolor(_BG)
        self._canvas.draw()

    def clear(self) -> None:
        self._fig.clear()
        self._canvas.draw()

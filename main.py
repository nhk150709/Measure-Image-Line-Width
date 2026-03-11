"""
main.py — Entry point for SEM Stripe Analyzer.

Usage:
    python main.py
"""
import sys
import os

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from ui.main_window import MainWindow


def main() -> int:
    # High-DPI support (Windows)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("SEM Stripe Analyzer")
    app.setOrganizationName("SEM Tools")

    # Base font
    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

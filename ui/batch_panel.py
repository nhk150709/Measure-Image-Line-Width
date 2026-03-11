"""
batch_panel.py — Dialog for batch processing a directory of images.
"""
from __future__ import annotations

import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QProgressBar, QTextEdit, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject

from core.recipe import Recipe
from batch.batch_processor import BatchProcessor


class _BatchWorker(QObject):
    """Runs BatchProcessor in a background thread."""
    progress = pyqtSignal(int, int, str)   # current, total, filename
    error = pyqtSignal(str, str)           # filename, error_msg
    finished = pyqtSignal(str, str, str)   # output_csv, stripes_csv, annotated_dir

    def __init__(self, recipe: Recipe, image_dir: str, output_csv: str):
        super().__init__()
        self._recipe = recipe
        self._image_dir = image_dir
        self._output_csv = output_csv
        # Derive companion output paths from the summary CSV path
        stem, _ = os.path.splitext(output_csv)
        self._stripes_csv = f"{stem}_stripes.csv"
        self._annotated_dir = os.path.join(os.path.dirname(output_csv), "annotated")
        self._processor = BatchProcessor(recipe)

    def run(self) -> None:
        self._processor.run(
            self._image_dir,
            output_csv=self._output_csv,
            annotated_dir=self._annotated_dir,
            output_stripes_csv=self._stripes_csv,
            progress_callback=lambda cur, tot, fn: self.progress.emit(cur, tot, fn),
            error_callback=lambda fn, msg: self.error.emit(fn, msg),
        )
        self.finished.emit(self._output_csv, self._stripes_csv, self._annotated_dir)

    def cancel(self) -> None:
        self._processor.cancel()


class BatchDialog(QDialog):
    """Batch processing dialog."""

    def __init__(self, recipe: Recipe, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(520, 440)
        self._recipe = recipe
        self._worker: _BatchWorker | None = None
        self._thread: QThread | None = None

        layout = QVBoxLayout(self)

        # ── Directory selector ────────────────────────────────────────
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Image directory…")
        btn_browse_dir = QPushButton("Browse…")
        btn_browse_dir.clicked.connect(self._browse_dir)
        dir_row.addWidget(QLabel("Directory:"))
        dir_row.addWidget(self._dir_edit)
        dir_row.addWidget(btn_browse_dir)
        layout.addLayout(dir_row)

        # ── Output CSV ────────────────────────────────────────────────
        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Output CSV path…")
        btn_browse_out = QPushButton("Browse…")
        btn_browse_out.clicked.connect(self._browse_out)
        out_row.addWidget(QLabel("Output CSV:"))
        out_row.addWidget(self._out_edit)
        out_row.addWidget(btn_browse_out)
        layout.addLayout(out_row)

        # ── Progress ──────────────────────────────────────────────────
        self._progress_bar = QProgressBar()
        self._progress_bar.setTextVisible(True)
        layout.addWidget(self._progress_bar)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        layout.addWidget(self._log)

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("▶ Start Batch")
        self._btn_run.setStyleSheet("font-weight: bold; background-color: #2d6a2d;")
        self._btn_run.clicked.connect(self._start_batch)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_batch)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_row.addWidget(self._btn_run)
        btn_row.addWidget(self._btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    # ──────────────────────────────────────────────────────────────────

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if d:
            self._dir_edit.setText(d)
            # Auto-suggest CSV name
            if not self._out_edit.text():
                base = os.path.basename(d.rstrip("/\\"))
                self._out_edit.setText(os.path.join(d, f"{base}_results.csv"))

    def _browse_out(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results CSV", "", "CSV Files (*.csv)"
        )
        if path:
            self._out_edit.setText(path)

    def _start_batch(self) -> None:
        image_dir = self._dir_edit.text().strip()
        output_csv = self._out_edit.text().strip()

        if not image_dir or not os.path.isdir(image_dir):
            QMessageBox.warning(self, "Error", "Please select a valid image directory.")
            return
        if not output_csv:
            QMessageBox.warning(self, "Error", "Please specify an output CSV path.")
            return

        self._log.clear()
        self._progress_bar.setValue(0)
        self._btn_run.setEnabled(False)
        self._btn_cancel.setEnabled(True)

        self._worker = _BatchWorker(self._recipe, image_dir, output_csv)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_finished)  # type: ignore[arg-type]
        self._thread.start()

    def _cancel_batch(self) -> None:
        if self._worker:
            self._worker.cancel()
        self._log.append("Cancelling…")
        self._btn_cancel.setEnabled(False)

    def _on_progress(self, current: int, total: int, filename: str) -> None:
        if total > 0:
            self._progress_bar.setMaximum(total)
            self._progress_bar.setValue(current)
        self._log.append(f"[{current}/{total}] {filename}")
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_error(self, filename: str, msg: str) -> None:
        self._log.append(f"  ⚠ ERROR in {filename}: {msg}")

    def _on_finished(self, csv_path: str, stripes_csv: str, annotated_dir: str) -> None:
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._log.append(
            f"\n✓ Batch complete.\n"
            f"  Summary CSV:   {csv_path}\n"
            f"  Stripes CSV:   {stripes_csv}\n"
            f"  Annotated images: {annotated_dir}"
        )
        QMessageBox.information(
            self, "Batch Complete",
            f"Processing finished.\n\n"
            f"Summary CSV:\n  {csv_path}\n\n"
            f"Per-stripe CSV:\n  {stripes_csv}\n\n"
            f"Annotated images:\n  {annotated_dir}"
        )

from __future__ import annotations

from collections import deque
from typing import Iterable

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QComboBox,
    QPushButton,
    QFileDialog,
    QPlainTextEdit,
    QCheckBox,
)

from hearts_ai.gui.services.log_bridge import LogEntry
from hearts_ai.util.timeutil import format_time


class LogView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._buffer = deque(maxlen=2000)
        self._pending = False

        self._level_filter = QComboBox()
        self._level_filter.addItems(["ALL", "INFO", "WARNING", "ERROR", "DEBUG"])
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search logs")
        self._auto_follow = QCheckBox("Auto-follow")
        self._auto_follow.setChecked(True)
        self._export = QPushButton("Export log to file")

        filter_row = QHBoxLayout()
        filter_row.addWidget(self._level_filter)
        filter_row.addWidget(self._search)
        filter_row.addWidget(self._auto_follow)
        filter_row.addWidget(self._export)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addLayout(filter_row)
        layout.addWidget(self._text)
        self.setLayout(layout)

        self._level_filter.currentTextChanged.connect(self._schedule_refresh)
        self._search.textChanged.connect(self._schedule_refresh)
        self._export.clicked.connect(self._export_logs)

        self._timer = QTimer()
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def add_entries(self, entries: list[LogEntry]) -> None:
        for entry in entries:
            self._buffer.append(entry)
        self._pending = True

    def _filtered_entries(self) -> Iterable[LogEntry]:
        level = self._level_filter.currentText()
        query = self._search.text().lower().strip()
        for entry in self._buffer:
            if level != "ALL" and entry.level != level:
                continue
            if query and query not in entry.message.lower() and query not in entry.source.lower():
                continue
            yield entry

    def _schedule_refresh(self) -> None:
        self._pending = True

    def _refresh(self) -> None:
        if not self._pending:
            return
        self._pending = False
        lines = [
            f"{format_time(entry.timestamp)} | {entry.level} | {entry.source} | {entry.message}"
            for entry in self._filtered_entries()
        ]
        self._text.setUpdatesEnabled(False)
        self._text.setPlainText("\n".join(lines))
        if self._auto_follow.isChecked():
            self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())
        self._text.setUpdatesEnabled(True)

    def _export_logs(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Logs", "hearts_logs.txt")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(
                    f"{format_time(entry.timestamp)}\t{entry.level}\t{entry.source}\t{entry.message}\n"
                )

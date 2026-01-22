from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout

from hearts_ai.gui.widgets.log_view import LogView


class LogsPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.log_view = LogView()
        layout.addWidget(self.log_view)
        self.setLayout(layout)

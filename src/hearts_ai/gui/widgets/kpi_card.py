from __future__ import annotations

from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QLabel

from hearts_ai.gui.theme import THEME


class KpiCard(QFrame):
    def __init__(self, title: str, accent: str) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ padding: 6px; border-radius: 6px; background: {THEME.card}; "
            f"border: 1px solid {THEME.border}; }} QLabel {{ font-size: 11px; }}"
        )
        self.setMinimumHeight(115)
        self._accent = QFrame()
        self._accent.setFixedWidth(6)
        self._accent.setStyleSheet(f"QFrame {{ background: {accent}; border-radius: 3px; }}")
        self._title = QLabel(title)
        self._value = QLabel("-")
        self._value.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {THEME.text};")
        self._delta = QLabel("")
        self._title.setStyleSheet(f"color: {THEME.text_muted};")
        self._delta.setStyleSheet(f"color: {THEME.text_muted};")

        col = QVBoxLayout()
        col.addWidget(self._title)
        col.addWidget(self._value)
        col.addWidget(self._delta)

        layout = QHBoxLayout()
        layout.addWidget(self._accent)
        layout.addLayout(col)
        self.setLayout(layout)

    def set_value(self, text: str, delta: str = "") -> None:
        self._value.setText(text)
        self._delta.setText(delta)

    def set_delta(self, text: str) -> None:
        self._delta.setText(text)

    def value_text(self) -> str:
        return self._value.text()

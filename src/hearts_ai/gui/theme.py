from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    bg: str
    panel: str
    card: str
    card2: str
    border: str
    text: str
    text_muted: str
    accent_train: str
    accent_eval: str
    accent_holdout: str
    accent_baseline: str
    good: str
    warn: str
    bad: str
    train_line: str
    eval_points: str
    holdout_line: str
    baseline_line: str
    accent_ok: str
    accent_warn: str


THEME = Theme(
    bg="#1E1F22",
    panel="#26282D",
    card="#2B2E34",
    card2="#30343B",
    border="#3A3F46",
    text="#ECEDEE",
    text_muted="#B3B7BF",
    accent_train="#8ECAE6",
    accent_eval="#FFB3C1",
    accent_holdout="#CDB4DB",
    accent_baseline="#B7B7A4",
    good="#BDE0FE",
    warn="#FFD6A5",
    bad="#FFADAD",
    train_line="#8ECAE6",
    eval_points="#FFB3C1",
    holdout_line="#CDB4DB",
    baseline_line="#B7B7A4",
    accent_ok="#BDE0FE",
    accent_warn="#FFD6A5",
)


def build_stylesheet(theme: Theme) -> str:
    return f"""
    QWidget {{
        background: {theme.bg};
        color: {theme.text};
        font-size: 13px;
    }}
    QGroupBox {{
        background: {theme.panel};
        border: 1px solid {theme.border};
        border-radius: 8px;
        margin-top: 10px;
        padding: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 6px;
        color: {theme.text_muted};
    }}
    QFrame {{
        background: {theme.card};
        border: 1px solid {theme.border};
        border-radius: 6px;
    }}
    QLabel {{
        color: {theme.text};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit {{
        background: {theme.card2};
        border: 1px solid {theme.border};
        border-radius: 6px;
        padding: 4px 6px;
        color: {theme.text};
    }}
    QComboBox::drop-down {{
        border: 0px;
    }}
    QComboBox QAbstractItemView {{
        background: {theme.card2};
        color: {theme.text};
        selection-background-color: {theme.accent_train};
    }}
    QPushButton {{
        background: {theme.card2};
        border: 1px solid {theme.border};
        border-radius: 6px;
        padding: 6px 10px;
    }}
    QPushButton:hover {{
        border: 1px solid {theme.accent_train};
    }}
    QPushButton:pressed {{
        background: {theme.panel};
    }}
    QHeaderView::section {{
        background: {theme.panel};
        color: {theme.text_muted};
        border: 1px solid {theme.border};
        padding: 4px 6px;
    }}
    QTableWidget {{
        gridline-color: {theme.border};
        background: {theme.card};
        alternate-background-color: {theme.card2};
    }}
    QScrollArea {{
        border: 0px;
    }}
    """

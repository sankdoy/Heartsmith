from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QFileDialog,
    QGroupBox,
)

from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trace import generate_trace


class DecisionInspector(QGroupBox):
    def __init__(self, params: ParameterSet) -> None:
        super().__init__("Decision Inspector")
        self._params = params
        self._trace = []
        self._trace_index = 0
        self._trace_path: Path | None = None

        controls = QHBoxLayout()
        controls.addWidget(QLabel("seed"))
        self._seed = QSpinBox()
        self._seed.setRange(0, 999999)
        self._seed.setValue(42)
        controls.addWidget(self._seed)

        controls.addWidget(QLabel("hand_index"))
        self._hand_index = QSpinBox()
        self._hand_index.setRange(0, 1000000)
        self._hand_index.setValue(0)
        controls.addWidget(self._hand_index)

        self._generate = QPushButton("Generate trace")
        self._load = QPushButton("Load trace…")
        controls.addWidget(self._generate)
        controls.addWidget(self._load)

        nav = QHBoxLayout()
        self._prev = QPushButton("Prev")
        self._next = QPushButton("Next")
        self._index_label = QLabel("0/0")
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addWidget(self._index_label)
        nav.addStretch(1)

        self._summary = QLabel("No trace loaded.")
        self._summary.setWordWrap(True)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMinimumHeight(260)

        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addLayout(nav)
        layout.addWidget(self._summary)
        layout.addWidget(self._details)
        self.setLayout(layout)

        self._generate.clicked.connect(self._generate_trace)
        self._load.clicked.connect(self._load_trace_dialog)
        self._prev.clicked.connect(lambda: self._step(-1))
        self._next.clicked.connect(lambda: self._step(1))

    def _generate_trace(self) -> None:
        seed = self._seed.value()
        hand_index = self._hand_index.value()
        output_dir = Path.cwd() / "runs" / "inspector"
        output_path = output_dir / f"trace_{seed}_{hand_index}.json"
        generate_trace(self._params.copy(), seed, hand_index, output_path)
        self._load_trace_file(output_path)

    def _load_trace_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open trace",
            str(Path.cwd() / "runs"),
            "Trace JSON (*.json)",
        )
        if path:
            self._load_trace_file(Path(path))

    def _load_trace_file(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._summary.setText("Failed to load trace.")
            return
        self._trace_path = path
        self._trace = payload.get("trace", [])
        self._trace_index = 0
        points = payload.get("points")
        raw_points = payload.get("raw_points")
        seed = payload.get("seed")
        hand_index = payload.get("hand_index")
        self._summary.setText(
            f"Trace: seed={seed} hand_index={hand_index} points={points} raw={raw_points}"
        )
        self._update_trace_display()

    def _step(self, delta: int) -> None:
        if not self._trace:
            return
        self._trace_index = max(0, min(len(self._trace) - 1, self._trace_index + delta))
        self._update_trace_display()

    def _update_trace_display(self) -> None:
        total = len(self._trace)
        if total == 0:
            self._details.setPlainText("No trace loaded.")
            self._index_label.setText("0/0")
            return
        entry = self._trace[self._trace_index]
        self._index_label.setText(f"{self._trace_index + 1}/{total}")

        lead = entry.get("lead_suit")
        hearts_broken = entry.get("hearts_broken")
        points_on_table = entry.get("points_on_table")
        chosen = entry.get("chosen")
        chosen_score = entry.get("chosen_score")
        legal = entry.get("legal_moves", [])
        trick_cards = entry.get("trick_cards", [])
        hand = entry.get("hand", [])
        must_follow = entry.get("must_follow")

        lines = [
            f"Trick {entry.get('trick_index')} lead={lead} hearts_broken={hearts_broken} "
            f"points_on_table={points_on_table:.2f}" if isinstance(points_on_table, float) else
            f"Trick {entry.get('trick_index')} lead={lead} hearts_broken={hearts_broken}",
            f"Hand size: {entry.get('hand_size')} must_follow: {must_follow}",
            f"Hand: {', '.join(hand)}",
            "Trick cards: " + ", ".join([f"P{c['player']}:{c['card']}" for c in trick_cards]),
            f"Legal moves ({len(legal)}): {', '.join(legal)}",
            f"Chosen: {chosen} score={chosen_score:.3f}" if isinstance(chosen_score, float) else f"Chosen: {chosen}",
            "",
            "Top candidates:",
        ]

        for idx, cand in enumerate(entry.get("candidates", []), start=1):
            terms = cand.get("terms", [])
            term_text = ", ".join([f"{name}:{value:.2f}" for name, value in terms])
            lines.append(f"{idx}. {cand.get('card')} score={cand.get('score'):.3f}")
            if term_text:
                lines.append(f"   terms: {term_text}")

        self._details.setPlainText("\n".join(lines))


class PlayPage(QWidget):
    def __init__(self, params: ParameterSet) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        layout.addWidget(QLabel("Play mode (inspect decisions)."))
        layout.addWidget(DecisionInspector(params))
        self.setLayout(layout)

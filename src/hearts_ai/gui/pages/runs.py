from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QTextEdit,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QMessageBox,
)

from hearts_ai.gui.pages.params import ParamsPage
from hearts_ai.gui.services.training_worker import TrainingWorker
from hearts_ai.training.params import ParameterSet


class RunsPage(QWidget):
    def __init__(self, store: ParameterSet, worker: TrainingWorker, params_page: ParamsPage) -> None:
        super().__init__()
        self._store = store
        self._worker = worker
        self._params_page = params_page
        self._runs_dir = Path.cwd() / "runs"

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._summary_a = QTextEdit()
        self._summary_a.setReadOnly(True)
        self._summary_b = QTextEdit()
        self._summary_b.setReadOnly(True)
        self._load_a = QPushButton("Load A params")
        self._load_b = QPushButton("Load B params")
        self._reload_btn = QPushButton("Reload runs")
        self._btn_open = QPushButton("Open folder")
        self._btn_delete = QPushButton("Delete selected")
        self._btn_clear = QPushButton("Clear all")
        self._status = "idle"

        self._diff_table = QTableWidget(0, 4)
        self._diff_table.setHorizontalHeaderLabels(["Name", "Run A", "Run B", "Delta"])
        self._diff_table.horizontalHeader().setStretchLastSection(True)

        left = QVBoxLayout()
        toolbar = QHBoxLayout()
        toolbar.addWidget(self._btn_open)
        toolbar.addWidget(self._btn_delete)
        toolbar.addWidget(self._btn_clear)
        left.addWidget(QLabel("Runs (select up to 2)"))
        left.addLayout(toolbar)
        left.addWidget(self._list)
        left.addWidget(self._reload_btn)

        summaries = QHBoxLayout()
        summaries.addWidget(self._summary_a)
        summaries.addWidget(self._summary_b)

        right = QVBoxLayout()
        right.addLayout(summaries)
        right.addWidget(self._diff_table)
        right.addWidget(self._load_a)
        right.addWidget(self._load_b)

        layout = QHBoxLayout()
        layout.addLayout(left, 1)
        layout.addLayout(right, 3)
        self.setLayout(layout)

        self._list.itemSelectionChanged.connect(self._refresh_view)
        self._load_a.clicked.connect(lambda: self._load_params(0))
        self._load_b.clicked.connect(lambda: self._load_params(1))
        self._reload_btn.clicked.connect(self._reload_runs)
        self._btn_open.clicked.connect(self._open_folder)
        self._btn_delete.clicked.connect(self._delete_selected)
        self._btn_clear.clicked.connect(self._clear_all)
        self._worker.status_updated.connect(self._update_status)

        self._reload_runs()

    def _reload_runs(self) -> None:
        self._list.clear()
        if not self._runs_dir.exists():
            return
        runs = sorted(self._runs_dir.glob("run_*"), reverse=True)
        for run in runs:
            self._list.addItem(run.name)
        self._update_buttons()

    def _selected_runs(self) -> list[Path]:
        items = self._list.selectedItems()
        run_dirs = [self._runs_dir / item.text() for item in items]
        return run_dirs[:2]

    def select_run(self, run_id: str) -> None:
        for row in range(self._list.count()):
            item = self._list.item(row)
            if item.text() == run_id:
                self._list.setCurrentItem(item)
                break

    def _refresh_view(self) -> None:
        runs = self._selected_runs()
        self._summary_a.setPlainText(self._summary_for_run(runs[0]) if len(runs) > 0 else "")
        self._summary_b.setPlainText(self._summary_for_run(runs[1]) if len(runs) > 1 else "")
        self._populate_diff(runs)
        self._update_buttons()

    def _summary_for_run(self, run_dir: Path) -> str:
        meta = self._read_json(run_dir / "run_meta.json")
        run_config = self._read_json(run_dir / "run_config.json")
        train_last = self._read_last_jsonl(run_dir / "metrics_train.jsonl")
        eval_last = self._read_last_jsonl(run_dir / "metrics_eval.jsonl")
        if not meta:
            return "No run metadata."
        lines = [json.dumps(meta, indent=2)]
        if run_config:
            lines.append("\nRun config:\n" + json.dumps(run_config, indent=2))
        if train_last:
            lines.append("\nTrain final:\n" + json.dumps(train_last, indent=2))
        if eval_last:
            lines.append("\nEval final:\n" + json.dumps(eval_last, indent=2))
        return "\n".join(lines)

    def _populate_diff(self, runs: list[Path]) -> None:
        self._diff_table.setRowCount(0)
        if len(runs) < 2:
            return
        params_a = self._read_params(runs[0] / "params_best.json")
        params_b = self._read_params(runs[1] / "params_best.json")
        if not params_a or not params_b:
            return
        names = sorted(set(params_a.keys()) | set(params_b.keys()))
        self._diff_table.setRowCount(len(names))
        for row, name in enumerate(names):
            a_val = params_a.get(name)
            b_val = params_b.get(name)
            delta = None
            if a_val is not None and b_val is not None:
                delta = b_val - a_val
            self._diff_table.setItem(row, 0, QTableWidgetItem(name))
            self._diff_table.setItem(row, 1, QTableWidgetItem(self._format_value(a_val)))
            self._diff_table.setItem(row, 2, QTableWidgetItem(self._format_value(b_val)))
            self._diff_table.setItem(row, 3, QTableWidgetItem(self._format_value(delta)))

    def _load_params(self, index: int) -> None:
        runs = self._selected_runs()
        if len(runs) <= index:
            return
        params_path = runs[index] / "params_best.json"
        if not params_path.exists():
            return
        payload = params_path.read_text(encoding="utf-8")
        loaded = ParameterSet.from_json(payload)
        self._store.apply(loaded)
        for param in self._store.all():
            self._worker.update_param(param.name, param.value)
            self._worker.update_locked(param.name, param.locked)
        self._params_page.refresh()

    def _read_json(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _read_last_jsonl(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            if not lines:
                return None
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            return None

    def _read_params(self, path: Path) -> dict[str, float] | None:
        data = self._read_json(path)
        if not data:
            return None
        return {name: float(entry.get("value", 0.0)) for name, entry in data.items()}

    def _format_value(self, value) -> str:
        if value is None:
            return "-"
        return f"{value:.3f}" if isinstance(value, float) else str(value)

    def _update_status(self, status: str) -> None:
        self._status = status
        self._update_buttons()

    def _update_buttons(self) -> None:
        has_selection = len(self._list.selectedItems()) > 0
        busy = self._status in {"running", "stopping"}
        self._btn_open.setEnabled(has_selection)
        self._btn_delete.setEnabled(has_selection and not busy)
        self._btn_clear.setEnabled(not busy)

    def _open_folder(self) -> None:
        runs = self._selected_runs()
        if not runs:
            return
        path = runs[0]
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _delete_selected(self) -> None:
        runs = self._selected_runs()
        if not runs:
            return
        if self._is_active_run_selected(runs):
            QMessageBox.warning(self, "Run active", "Cannot delete the active run while training.")
            return
        details = self._delete_summary(runs)
        reply = QMessageBox.question(
            self,
            "Delete selected runs",
            f"Move selected runs to trash?\n\n{details}",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        trash_root = self._trash_root()
        for run_dir in runs:
            try:
                move_to_trash(run_dir, trash_root, self._runs_dir)
            except Exception as exc:
                QMessageBox.warning(self, "Delete failed", f"{run_dir.name}: {exc}")
        self._reload_runs()

    def _clear_all(self) -> None:
        if self._status in {"running", "stopping"}:
            QMessageBox.warning(self, "Training running", "Stop training before clearing runs.")
            return
        runs = sorted(self._runs_dir.glob("run_*"))
        if not runs:
            return
        reply = QMessageBox.question(
            self,
            "Clear all runs",
            "This will move ALL runs to runs/_trash. Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        trash_root = self._trash_root()
        for run_dir in runs:
            try:
                move_to_trash(run_dir, trash_root, self._runs_dir)
            except Exception as exc:
                QMessageBox.warning(self, "Clear failed", f"{run_dir.name}: {exc}")
        self._reload_runs()

    def _trash_root(self) -> Path:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trash_root = self._runs_dir / "_trash" / timestamp
        trash_root.mkdir(parents=True, exist_ok=True)
        return trash_root

    def _delete_summary(self, runs: list[Path]) -> str:
        lines = []
        for run_dir in runs:
            meta = self._read_json(run_dir / "run_meta.json") or {}
            train_last = self._read_last_jsonl(run_dir / "metrics_train.jsonl") or {}
            summary = (
                f"{run_dir.name} start={meta.get('start_time','-')} "
                f"preset={meta.get('preset_id','-')} "
                f"hands={train_last.get('hands_done','-')}"
            )
            lines.append(summary)
        return "\n".join(lines)

    def _is_active_run_selected(self, runs: list[Path]) -> bool:
        if self._status not in {"running", "stopping"}:
            return False
        current = self._worker.current_run_id()
        if not current:
            return False
        return any(run.name == current for run in runs)


def move_to_trash(run_dir: Path, trash_root: Path, runs_root: Path) -> Path:
    import shutil

    run_dir = run_dir.resolve()
    runs_root = runs_root.resolve()
    if not run_dir.is_relative_to(runs_root):
        raise ValueError("Refusing to move path outside runs directory")
    destination = trash_root / run_dir.name
    try:
        return Path(run_dir).rename(destination)
    except OSError:
        return Path(shutil.move(str(run_dir), str(destination)))

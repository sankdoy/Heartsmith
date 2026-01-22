from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QStackedWidget,
    QLabel,
)

from hearts_ai.gui.pages.about import AboutPage
from hearts_ai.gui.pages.logs import LogsPage
from hearts_ai.gui.pages.params import ParamsPage
from hearts_ai.gui.pages.play import PlayPage
from hearts_ai.gui.pages.runs import RunsPage
from hearts_ai.gui.pages.train import TrainPage
from hearts_ai.gui.pages.train_simple import SimpleTrainPage
from hearts_ai.gui.services.log_bridge import QtLogHandler
from hearts_ai.gui.services.training_worker import TrainingWorker
from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trainer import TrainingConfig


class MainWindow(QMainWindow):
    def __init__(self, log_handler: QtLogHandler) -> None:
        super().__init__()
        self.setWindowTitle("Hearts AI")

        self._params_store = ParameterSet()
        self._training_config = TrainingConfig()
        self._worker = TrainingWorker(self._params_store, self._training_config)

        self._last_metrics_time = time.time()
        self._hands_per_tick = self._training_config.hands_per_tick

        container = QWidget()
        layout = QHBoxLayout()
        container.setLayout(layout)

        nav = QVBoxLayout()
        nav_widget = QWidget()
        nav_widget.setLayout(nav)
        nav_widget.setFixedWidth(160)

        self._btn_play = QPushButton("Play")
        self._btn_train_simple = QPushButton("Training (simple)")
        self._btn_train = QPushButton("Training (advanced)")
        self._btn_params = QPushButton("Parameters")
        self._btn_logs = QPushButton("Logs")
        self._btn_runs = QPushButton("Runs")
        self._btn_about = QPushButton("About/Rules")

        for btn in (
            self._btn_play,
            self._btn_train_simple,
            self._btn_train,
            self._btn_params,
            self._btn_logs,
            self._btn_runs,
            self._btn_about,
        ):
            btn.setCursor(Qt.PointingHandCursor)
            nav.addWidget(btn)
        nav.addStretch(1)

        self._stack = QStackedWidget()
        self._page_play = PlayPage(self._params_store)
        self._page_train_simple = SimpleTrainPage(self._worker)
        self._page_train = TrainPage(self._worker)
        self._page_params = ParamsPage(self._params_store, self._worker)
        self._page_logs = LogsPage()
        self._page_runs = RunsPage(self._params_store, self._worker, self._page_params)
        self._page_about = AboutPage()

        self._stack.addWidget(self._page_play)
        self._stack.addWidget(self._page_train_simple)
        self._stack.addWidget(self._page_train)
        self._stack.addWidget(self._page_params)
        self._stack.addWidget(self._page_logs)
        self._stack.addWidget(self._page_runs)
        self._stack.addWidget(self._page_about)

        layout.addWidget(nav_widget)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(container)

        self._status_hands = QLabel("hands/sec: 0")
        self._status_iter = QLabel("iter: 0")
        self._status_best = QLabel("best: -")
        self.statusBar().addPermanentWidget(self._status_hands)
        self.statusBar().addPermanentWidget(self._status_iter)
        self.statusBar().addPermanentWidget(self._status_best)

        self._btn_play.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_play))
        self._btn_train_simple.clicked.connect(
            lambda: self._stack.setCurrentWidget(self._page_train_simple)
        )
        self._btn_train.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_train))
        self._btn_params.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_params))
        self._btn_logs.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_logs))
        self._btn_runs.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_runs))
        self._btn_about.clicked.connect(lambda: self._stack.setCurrentWidget(self._page_about))

        self._worker.metrics_updated.connect(self._handle_metrics)
        self._worker.best_params_updated.connect(self._page_train.on_best_params)
        self._worker.eval_metrics_updated.connect(self._page_train.on_eval_metrics)
        self._worker.holdout_metrics_updated.connect(self._page_train.on_holdout_metrics)
        self._worker.status_updated.connect(self._page_train.on_status)
        self._worker.phase_updated.connect(self._page_train.on_phase)
        self._worker.metrics_updated.connect(self._page_train_simple.on_metrics)
        self._worker.eval_metrics_updated.connect(self._page_train_simple.on_eval_metrics)
        self._worker.holdout_metrics_updated.connect(self._page_train_simple.on_holdout_metrics)
        self._worker.status_updated.connect(self._page_train_simple.on_status)
        self._worker.eval_status_updated.connect(self._page_train_simple.on_eval_status)
        self._worker.phase_updated.connect(self._page_train_simple.on_phase)
        log_handler.log_batch_received.connect(self._page_logs.log_view.add_entries)
        self._page_train.resume_last_best.connect(self._load_latest_params)
        self._page_train.open_run_requested.connect(self._open_run)
        self._page_train_simple.open_advanced_requested.connect(
            lambda: self._stack.setCurrentWidget(self._page_train)
        )
        self._page_train_simple.open_run_requested.connect(self._open_run)

        self._load_latest_params()

        self._page_train._hands_per_tick.valueChanged.connect(self._update_hands_per_tick)

    def _handle_metrics(self, metrics) -> None:
        now = time.time()
        dt = max(0.001, now - self._last_metrics_time)
        hands_per_sec = self._hands_per_tick / dt
        self._last_metrics_time = now
        self._status_hands.setText(f"hands/sec: {hands_per_sec:.1f}")
        self._status_iter.setText(f"iter: {metrics.iteration}")
        self._status_best.setText(f"best: {metrics.best_score:.2f}")
        self._page_train.on_metrics(metrics)

    def _update_hands_per_tick(self, value: int) -> None:
        self._hands_per_tick = value

    def closeEvent(self, event) -> None:
        if self._worker.is_running():
            stopped = self._worker.stop_and_wait(2000)
            if not stopped:
                self.statusBar().showMessage("Training still stopping. Please wait…", 5000)
                event.ignore()
                return
        event.accept()

    def _load_latest_params(self) -> None:
        from pathlib import Path
        import json

        latest_path = Path.cwd() / "runs" / "latest.json"
        if not latest_path.exists():
            return
        try:
            payload = json.loads(latest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        params_path = Path(payload.get("path", ""))
        if not params_path.exists():
            return
        try:
            params_json = params_path.read_text(encoding="utf-8")
        except OSError:
            return
        loaded = ParameterSet.from_json(params_json)
        self._params_store.apply(loaded)
        for param in self._params_store.all():
            self._worker.update_param(param.name, param.value)
            self._worker.update_locked(param.name, param.locked)
        self._page_params.refresh()
        self._page_train.on_best_params(params_json)
        run_id = payload.get("run_id", "unknown")
        self.statusBar().showMessage(f"Loaded best params from run {run_id}", 5000)
        self._page_train.set_loaded_message(f"Loaded best params from run {run_id}")
        if run_id and run_id != "unknown":
            self._page_train.show_loaded_banner(run_id)
            self._page_train_simple.show_loaded_banner(run_id)

    def _open_run(self, run_id: str) -> None:
        self._stack.setCurrentWidget(self._page_runs)
        if hasattr(self._page_runs, "select_run"):
            self._page_runs.select_run(run_id)

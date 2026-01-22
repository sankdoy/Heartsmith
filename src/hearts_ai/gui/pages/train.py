from __future__ import annotations

import logging
import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
    QGroupBox,
    QCheckBox,
    QTextEdit,
    QDoubleSpinBox,
    QComboBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QFrame,
    QScrollArea,
    QDialog,
    QApplication,
    QGridLayout,
)

from hearts_ai.gui.widgets.metric_plot import MetricPlot
from hearts_ai.gui.widgets.kpi_card import KpiCard
from hearts_ai.gui.services.training_worker import TrainingWorker
from hearts_ai.gui.theme import THEME
from hearts_ai.training.run_config import build_run_config, write_run_config
from hearts_ai.training.metrics import EvalSnapshot, HoldoutSnapshot, MetricsSnapshot


class TrainPage(QWidget):
    resume_last_best = Signal()
    open_run_requested = Signal(str)
    def __init__(self, worker: TrainingWorker) -> None:
        super().__init__()
        self._worker = worker
        self._last_train_time = time.time()
        self._last_eval: EvalSnapshot | None = None
        self._last_holdout_mean: float | None = None
        self._last_holdout_se: float | None = None
        self._best_eval_mean: float | None = None
        self._resume_last_best = QPushButton("Resume last best")
        self._banner = QFrame()
        self._banner.setStyleSheet(f"QFrame {{ background: {THEME.card2}; border-radius: 6px; }}")
        banner_layout = QHBoxLayout()
        self._banner_label = QLabel("")
        self._banner_open = QPushButton("Open run")
        self._banner_close = QPushButton("✕")
        self._banner_close.setFixedWidth(28)
        banner_layout.addWidget(self._banner_label)
        banner_layout.addWidget(self._banner_open)
        banner_layout.addWidget(self._banner_close)
        self._banner.setLayout(banner_layout)
        self._banner.hide()

        controls = QHBoxLayout()
        self._start = QPushButton("Start")
        self._stop = QPushButton("Stop")
        self._pause = QPushButton("Pause")
        self._reset = QPushButton("Reset")
        self._save = QPushButton("Save Run")
        self._run_eval = QPushButton("Run evaluation now")
        controls.addWidget(self._start)
        controls.addWidget(self._stop)
        controls.addWidget(self._pause)
        controls.addWidget(self._reset)
        controls.addWidget(self._save)
        controls.addWidget(self._run_eval)

        settings = QGroupBox("Training")
        settings_layout = QVBoxLayout()
        hands_row = QHBoxLayout()
        hands_row.addWidget(QLabel("hands_per_tick"))
        self._hands_per_tick = QSpinBox()
        self._hands_per_tick.setRange(10, 5000)
        self._hands_per_tick.setValue(200)
        self._hands_per_tick.setToolTip("Hands simulated per training tick.")
        hands_row.addWidget(self._hands_per_tick)
        settings_layout.addLayout(hands_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("base_seed"))
        self._seed = QSpinBox()
        self._seed.setRange(0, 999999)
        self._seed.setValue(42)
        self._seed.setToolTip("Base seed for training; combined with hand_index.")
        seed_row.addWidget(self._seed)
        settings_layout.addLayout(seed_row)

        seed_mode_row = QHBoxLayout()
        seed_mode_row.addWidget(QLabel("train_seed_mode"))
        self._seed_mode = QComboBox()
        self._seed_mode.addItems(["cycle", "random", "fixed"])
        self._seed_mode.setToolTip("Seed strategy for training runs.")
        seed_mode_row.addWidget(self._seed_mode)
        settings_layout.addLayout(seed_mode_row)

        rate_row = QHBoxLayout()
        rate_row.addWidget(QLabel("updates_per_sec"))
        self._updates_per_sec = QDoubleSpinBox()
        self._updates_per_sec.setRange(1.0, 20.0)
        self._updates_per_sec.setSingleStep(1.0)
        self._updates_per_sec.setValue(5.0)
        self._updates_per_sec.setToolTip("UI update cadence during training.")
        rate_row.addWidget(self._updates_per_sec)
        settings_layout.addLayout(rate_row)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("hands_per_second_target"))
        self._hands_per_second_target = QDoubleSpinBox()
        self._hands_per_second_target.setRange(0.0, 10000.0)
        self._hands_per_second_target.setSingleStep(50.0)
        self._hands_per_second_target.setValue(0.0)
        self._hands_per_second_target.setToolTip("Adaptive speed target; 0 disables.")
        target_row.addWidget(self._hands_per_second_target)
        settings_layout.addLayout(target_row)

        self._fast_mode = QCheckBox("Fast mode")
        settings_layout.addWidget(self._fast_mode)

        capture_row = QHBoxLayout()
        self._capture_enabled = QCheckBox("Capture sample hand every")
        self._capture_every = QSpinBox()
        self._capture_every.setRange(100, 50000)
        self._capture_every.setValue(5000)
        self._capture_every.setToolTip("Sampling frequency; ignored in Fast mode.")
        capture_row.addWidget(self._capture_enabled)
        capture_row.addWidget(self._capture_every)
        capture_row.addWidget(QLabel("hands"))
        settings_layout.addLayout(capture_row)

        self._capture_now = QPushButton("Capture next hand now")
        settings_layout.addWidget(self._capture_now)

        log_row = QHBoxLayout()
        log_row.addWidget(QLabel("log_every_n_hands"))
        self._log_every_n = QSpinBox()
        self._log_every_n.setRange(10, 10000)
        self._log_every_n.setValue(100)
        log_row.addWidget(self._log_every_n)
        settings_layout.addLayout(log_row)

        verbosity_row = QHBoxLayout()
        verbosity_row.addWidget(QLabel("log_verbosity"))
        self._log_verbosity = QComboBox()
        self._log_verbosity.addItems(["Quiet", "Normal", "Verbose"])
        self._log_verbosity.setCurrentText("Normal")
        verbosity_row.addWidget(self._log_verbosity)
        settings_layout.addLayout(verbosity_row)

        settings.setLayout(settings_layout)

        eval_group = QGroupBox("Evaluation")
        eval_layout = QVBoxLayout()
        eval_hands_row = QHBoxLayout()
        eval_hands_row.addWidget(QLabel("eval_hands_per_seed"))
        self._eval_hands = QSpinBox()
        self._eval_hands.setRange(50, 5000)
        self._eval_hands.setValue(200)
        self._eval_hands.setToolTip("Hands simulated per evaluation seed.")
        eval_hands_row.addWidget(self._eval_hands)
        eval_layout.addLayout(eval_hands_row)

        eval_seed_row = QHBoxLayout()
        eval_seed_row.addWidget(QLabel("eval_seed_preset"))
        self._eval_seed_preset = QComboBox()
        self._eval_seed_preset.addItems(["quick", "standard", "thorough"])
        self._eval_seed_preset.setToolTip("Seed list size for evaluation.")
        eval_seed_row.addWidget(self._eval_seed_preset)
        eval_layout.addLayout(eval_seed_row)

        self._eval_status = QLabel("eval status: idle")
        eval_layout.addWidget(self._eval_status)

        eval_group.setLayout(eval_layout)

        opponents = QGroupBox("Opponent pools")
        opp_layout = QHBoxLayout()
        train_opp = QVBoxLayout()
        train_opp.addWidget(QLabel("Training"))
        self._train_random = QCheckBox("RandomBot")
        self._train_safe = QCheckBox("SafeBot")
        self._train_best = QCheckBox("BestSnapshotBot")
        self._train_random.setChecked(True)
        self._train_safe.setChecked(True)
        self._train_best.setChecked(True)
        train_opp.addWidget(self._train_random)
        train_opp.addWidget(self._train_safe)
        train_opp.addWidget(self._train_best)

        eval_opp = QVBoxLayout()
        eval_opp.addWidget(QLabel("Evaluation"))
        self._eval_random = QCheckBox("RandomBot")
        self._eval_safe = QCheckBox("SafeBot")
        self._eval_best = QCheckBox("BestSnapshotBot")
        self._eval_safe.setChecked(True)
        self._eval_best.setChecked(True)
        eval_opp.addWidget(self._eval_random)
        eval_opp.addWidget(self._eval_safe)
        eval_opp.addWidget(self._eval_best)

        opp_layout.addLayout(train_opp)
        opp_layout.addLayout(eval_opp)
        opponents.setLayout(opp_layout)

        self._overfit_label = QLabel("Overfit risk: LOW")
        self._status_label = QLabel("Status: Idle")
        self._phase_label = QLabel("Phase: Training")
        self._seed_strategy = QLabel("Training seed: base_seed + hand_index")
        self._hand_index_label = QLabel("hand_index: 0")
        self._seed_warning = QLabel("")

        self._plot_penalty = MetricPlot(
            "Mean points/hand",
            y_min=0,
            y_max=26,
            y_label="points/hand",
        )
        self._plot_win = MetricPlot("Win rate", y_min=0, y_max=1, y_label="rate")
        self._plot_qs = MetricPlot("Q♠ taken rate", y_min=0, y_max=1, y_label="rate")
        self._plot_penalty.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_win.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_qs.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_penalty.add_series("train", THEME.train_line)
        self._plot_win.add_series("train", THEME.train_line)
        self._plot_qs.add_series("train", THEME.train_line)
        self._plot_penalty.set_baseline("avg_points", 6.5, THEME.baseline_line)
        self._plot_delta = MetricPlot("Δ points/hand vs SafeBot", y_label="points/hand")
        self._plot_delta.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_delta.add_series("train", THEME.train_line)

        plots = QGridLayout()
        plots.addWidget(self._plot_penalty, 0, 0)
        plots.addWidget(self._plot_win, 0, 1)
        plots.addWidget(self._plot_qs, 1, 0)
        plots.addWidget(self._plot_delta, 1, 1)

        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("window"))
        self._window_select = QComboBox()
        self._window_select.addItems(["200", "500", "1000", "all"])
        self._window_select.setCurrentText("500")
        self._window_select.setToolTip("Window of recent ticks shown.")
        plot_controls.addWidget(self._window_select)

        plot_controls.addWidget(QLabel("smoothing"))
        self._smoothing_mode = QComboBox()
        self._smoothing_mode.addItems(["ema", "rolling", "none"])
        self._smoothing_mode.setToolTip("Smoothing for plots.")
        plot_controls.addWidget(self._smoothing_mode)

        self._smoothing_value = QDoubleSpinBox()
        self._smoothing_value.setRange(0.0, 1.0)
        self._smoothing_value.setSingleStep(0.05)
        self._smoothing_value.setValue(0.2)
        self._smoothing_value.setToolTip("EMA alpha or rolling window size.")
        plot_controls.addWidget(self._smoothing_value)

        self._eval_settings_label = QLabel("eval: -")

        self._kpi_grid = QGridLayout()
        self._kpi_train = KpiCard("Train points/hand", THEME.train_line)
        self._kpi_eval = KpiCard("Eval points/hand", THEME.eval_points)
        self._kpi_holdout = KpiCard("Holdout points/hand", THEME.holdout_line)
        self._kpi_best_eval = KpiCard("Best eval points/hand", THEME.eval_points)
        self._kpi_eval_win = KpiCard("Eval win rate", THEME.eval_points)
        self._kpi_eval_qs = KpiCard("Eval Q♠ rate", THEME.eval_points)
        self._kpi_hands_sec = KpiCard("Hands/sec", THEME.train_line)
        self._kpi_cards = [
            self._kpi_train,
            self._kpi_eval,
            self._kpi_holdout,
            self._kpi_best_eval,
            self._kpi_eval_win,
            self._kpi_eval_qs,
            self._kpi_hands_sec,
        ]
        for card in self._kpi_cards:
            card.setMinimumWidth(180)

        self._opponent_table = QTableWidget(0, 3)
        self._opponent_table.setHorizontalHeaderLabels(["Opponent", "Mean points/hand", "Win rate"])
        self._opponent_table.horizontalHeader().setStretchLastSection(True)
        self._opponent_table.setAlternatingRowColors(True)

        self._view_best_params = QPushButton("View best params JSON")
        self._copy_best_params = QPushButton("Copy best params")
        self._best_params_json = ""
        self._load_status = QLabel("")

        settings_container = QWidget()
        settings_layout_outer = QVBoxLayout()
        settings_layout_outer.addLayout(controls)
        settings_layout_outer.addWidget(settings)
        settings_layout_outer.addWidget(eval_group)
        settings_layout_outer.addWidget(opponents)
        settings_layout_outer.addWidget(self._overfit_label)
        settings_layout_outer.addWidget(self._status_label)
        settings_layout_outer.addWidget(self._phase_label)
        settings_layout_outer.addWidget(self._banner)
        settings_layout_outer.addWidget(self._seed_strategy)
        settings_layout_outer.addWidget(self._hand_index_label)
        settings_layout_outer.addWidget(self._seed_warning)
        settings_layout_outer.addWidget(self._load_status)
        settings_layout_outer.addWidget(self._view_best_params)
        settings_layout_outer.addWidget(self._copy_best_params)
        settings_layout_outer.addWidget(self._resume_last_best)
        settings_container.setLayout(settings_layout_outer)

        plot_container = QWidget()
        plot_layout = QVBoxLayout()
        plot_layout.addLayout(self._kpi_grid)
        plot_layout.addWidget(self._eval_settings_label)
        plot_layout.addWidget(self._opponent_table)
        plot_layout.addLayout(plot_controls)
        plot_layout.addLayout(plots)
        plot_container.setLayout(plot_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(settings_container)
        splitter.addWidget(plot_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)

        container = QWidget()
        container_layout = QVBoxLayout()
        container_layout.addWidget(splitter)
        container.setLayout(container_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(container)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

        self._start.clicked.connect(self._start_training)
        self._stop.clicked.connect(self._worker.stop)
        self._pause.clicked.connect(self._toggle_pause)
        self._reset.clicked.connect(self._worker.reset)
        self._run_eval.clicked.connect(self._worker.run_eval_now)

        self._hands_per_tick.valueChanged.connect(self._worker.update_hands_per_tick)
        self._seed.valueChanged.connect(self._worker.update_seed)
        self._seed_mode.currentTextChanged.connect(self._worker.update_seed_mode)
        self._seed_mode.currentTextChanged.connect(self._update_seed_warning)
        self._updates_per_sec.valueChanged.connect(self._worker.update_updates_per_sec)
        self._hands_per_second_target.valueChanged.connect(self._worker.update_hands_per_second_target)
        self._fast_mode.toggled.connect(self._worker.update_fast_mode)
        self._log_every_n.valueChanged.connect(self._worker.update_log_every_n_hands)
        self._log_verbosity.currentTextChanged.connect(self._set_log_verbosity)
        self._capture_enabled.toggled.connect(self._worker.update_capture_sample_enabled)
        self._capture_every.valueChanged.connect(self._worker.update_capture_sample_every)
        self._capture_now.clicked.connect(self._worker.capture_next_sample)
        self._eval_hands.valueChanged.connect(self._worker.update_eval_hands_per_seed)
        self._eval_seed_preset.currentTextChanged.connect(self._handle_eval_seed_preset)
        self._window_select.currentTextChanged.connect(self._update_plot_window)
        self._smoothing_mode.currentTextChanged.connect(self._on_smoothing_mode_changed)
        self._smoothing_value.valueChanged.connect(self._update_smoothing)
        self._view_best_params.clicked.connect(self._show_best_params)
        self._copy_best_params.clicked.connect(self._copy_best_params_json)
        self._resume_last_best.clicked.connect(self._emit_resume_last_best)
        self._save.clicked.connect(self._save_run_config)
        self._banner_close.clicked.connect(self._banner.hide)
        self._banner_open.clicked.connect(self._emit_open_run)

        for box in (
            self._train_random,
            self._train_safe,
            self._train_best,
            self._eval_random,
            self._eval_safe,
            self._eval_best,
        ):
            box.stateChanged.connect(self._update_opponent_pools)

        self._worker.eval_status_updated.connect(self._set_eval_status)
        self._worker.overfit_status_updated.connect(self._set_overfit_status)

        self._handle_eval_seed_preset(self._eval_seed_preset.currentText())
        self._update_opponent_pools()
        self._set_log_verbosity(self._log_verbosity.currentText())
        self.on_status("idle")
        self._set_overfit_status("LOW")
        self._update_plot_window(self._window_select.currentText())
        self._on_smoothing_mode_changed(self._smoothing_mode.currentText())
        self._kpi_holdout.set_value("—")
        self._layout_kpis()
        self._load_latest_seed_mode()
        self._update_seed_warning()

    def on_metrics(self, metrics: MetricsSnapshot) -> None:
        self._plot_penalty.add_point("train", metrics.iteration, metrics.mean_penalty)
        self._plot_win.add_point("train", metrics.iteration, metrics.win_rate)
        self._plot_qs.add_point("train", metrics.iteration, metrics.qs_rate)
        now = time.time()
        dt = max(1e-6, now - self._last_train_time)
        hands_sec = self._hands_per_tick.value() / dt
        self._last_train_time = now
        self._kpi_train.set_value(f"{metrics.mean_penalty:.2f} pts/hand")
        self._kpi_hands_sec.set_value(f"{hands_sec:.1f}")
        self._hand_index_label.setText(f"hand_index: {metrics.hand_index}")
        if self._last_eval:
            self._update_delta_plot(metrics, self._last_eval.opponent_breakdown, is_eval=False)

    def on_eval_metrics(self, metrics: EvalSnapshot) -> None:
        self._plot_penalty.add_point("eval", metrics.iteration, metrics.mean_penalty)
        self._plot_win.add_point("eval", metrics.iteration, metrics.win_rate)
        self._plot_qs.add_point("eval", metrics.iteration, metrics.qs_rate)
        self._plot_penalty.add_event_marker(metrics.iteration)
        self._plot_win.add_event_marker(metrics.iteration)
        self._plot_qs.add_event_marker(metrics.iteration)
        self._last_eval = metrics
        if self._best_eval_mean is None or metrics.mean_penalty < self._best_eval_mean:
            self._best_eval_mean = metrics.mean_penalty

        se_text = f" ±{metrics.mean_penalty_se:.2f}" if metrics.mean_penalty_se > 0 else ""
        self._kpi_eval.set_value(f"{metrics.mean_penalty:.2f} pts/hand{se_text}")
        self._kpi_best_eval.set_value(f"{self._best_eval_mean:.2f} pts/hand")
        self._kpi_eval_win.set_value(f"{metrics.win_rate:.2f}")
        self._kpi_eval_qs.set_value(f"{metrics.qs_rate:.2f}")
        self._eval_settings_label.setText(
            f"eval: {metrics.seeds_count} seeds x {metrics.hands_per_seed} hands"
        )
        self._update_opponent_table(metrics.opponent_breakdown)
        self._update_safe_delta(metrics.opponent_breakdown, metrics)
        self._update_safe_baseline(metrics.opponent_breakdown)
        self._update_delta_plot(metrics, metrics.opponent_breakdown, is_eval=True)

    def on_best_params(self, text: str) -> None:
        self._best_params_json = text

    def set_loaded_message(self, message: str) -> None:
        self._load_status.setText(message)

    def on_holdout_metrics(self, metrics: HoldoutSnapshot) -> None:
        self._last_holdout_mean = metrics.mean_penalty
        self._last_holdout_se = metrics.mean_penalty_se
        se_text = f" ±{metrics.mean_penalty_se:.2f}" if metrics.mean_penalty_se > 0 else ""
        self._kpi_holdout.set_value(f"{metrics.mean_penalty:.2f} pts/hand{se_text}")

    def _toggle_pause(self) -> None:
        if self._pause.text() == "Pause":
            self._worker.pause()
            self._pause.setText("Resume")
        else:
            self._worker.resume()
            self._pause.setText("Pause")

    def _start_training(self) -> None:
        seed = self._apply_seed_mode()
        self._worker.update_hands_per_tick(self._hands_per_tick.value())
        self._worker.update_seed(seed)
        self._worker.update_seed_mode(self._seed_mode.currentText())
        self._worker.update_updates_per_sec(self._updates_per_sec.value())
        self._worker.update_hands_per_second_target(self._hands_per_second_target.value())
        self._worker.update_fast_mode(self._fast_mode.isChecked())
        self._worker.update_log_every_n_hands(self._log_every_n.value())
        self._set_log_verbosity(self._log_verbosity.currentText())
        self._worker.update_capture_sample_enabled(self._capture_enabled.isChecked())
        self._worker.update_capture_sample_every(self._capture_every.value())
        self._worker.update_eval_hands_per_seed(self._eval_hands.value())
        self._handle_eval_seed_preset(self._eval_seed_preset.currentText())
        self._update_opponent_pools()
        self._worker.start()

    def _handle_eval_seed_preset(self, preset: str) -> None:
        if preset == "quick":
            seeds = list(range(1, 6))
        elif preset == "standard":
            seeds = list(range(1, 21))
        else:
            seeds = list(range(1, 101))
        self._worker.update_eval_seeds(seeds)

    def _update_opponent_pools(self) -> None:
        train_pool = []
        if self._train_random.isChecked():
            train_pool.append("RandomBot")
        if self._train_safe.isChecked():
            train_pool.append("SafeBot")
        if self._train_best.isChecked():
            train_pool.append("BestSnapshotBot")

        eval_pool = []
        if self._eval_random.isChecked():
            eval_pool.append("RandomBot")
        if self._eval_safe.isChecked():
            eval_pool.append("SafeBot")
        if self._eval_best.isChecked():
            eval_pool.append("BestSnapshotBot")

        self._worker.update_train_opponents(train_pool)
        self._worker.update_eval_opponents(eval_pool)

    def _set_eval_status(self, status: str) -> None:
        text = status if status else "idle"
        self._eval_status.setText(f"eval status: {text}")

    def _set_overfit_status(self, status: str) -> None:
        level = status.upper() if status else "LOW"
        color = THEME.good
        text_color = "#1E1F22"
        if level == "MED":
            color = THEME.warn
        elif level == "HIGH":
            color = THEME.bad
        self._overfit_label.setText(f"Overfit risk: {level}")
        self._overfit_label.setStyleSheet(
            f"background-color: {color}; color: {text_color}; padding: 4px 10px; "
            "border-radius: 10px; font-weight: bold;"
        )

    def on_status(self, status: str) -> None:
        label = status.capitalize() if status else "Idle"
        self._status_label.setText(f"Status: {label}")
        running = status in {"running", "paused", "stopping"}
        self._start.setEnabled(not running)
        self._stop.setEnabled(running)
        self._pause.setEnabled(status in {"running", "paused"})
        if status != "paused":
            self._pause.setText("Pause")

    def on_phase(self, phase: str) -> None:
        label = phase.capitalize() if phase else "Training"
        self._phase_label.setText(f"Phase: {label}")

    def _set_log_verbosity(self, level: str) -> None:
        if level == "Quiet":
            logging.getLogger().setLevel(logging.WARNING)
            self._worker.update_debug_explain_enabled(False)
        elif level == "Verbose":
            logging.getLogger().setLevel(logging.DEBUG)
            self._worker.update_debug_explain_enabled(True)
        else:
            logging.getLogger().setLevel(logging.INFO)
            self._worker.update_debug_explain_enabled(False)

    def _update_plot_window(self, text: str) -> None:
        if text == "all":
            window = None
        else:
            window = int(text)
        for plot in (self._plot_penalty, self._plot_win, self._plot_qs):
            plot.set_window(window)

    def _update_smoothing(self) -> None:
        mode = self._smoothing_mode.currentText()
        value = self._smoothing_value.value()
        for plot in (self._plot_penalty, self._plot_win, self._plot_qs):
            plot.set_smoothing(mode, value)

    def _on_smoothing_mode_changed(self, mode: str) -> None:
        if mode == "rolling":
            self._smoothing_value.setRange(1.0, 50.0)
            self._smoothing_value.setSingleStep(1.0)
            if self._smoothing_value.value() < 2:
                self._smoothing_value.setValue(5.0)
        elif mode == "ema":
            self._smoothing_value.setRange(0.0, 1.0)
            self._smoothing_value.setSingleStep(0.05)
            if self._smoothing_value.value() == 0:
                self._smoothing_value.setValue(0.2)
        else:
            self._smoothing_value.setRange(0.0, 1.0)
            self._smoothing_value.setSingleStep(0.05)
            self._smoothing_value.setValue(0.0)
        self._update_smoothing()

    def _update_opponent_table(self, breakdown: dict) -> None:
        items = [("RandomBot", breakdown.get("RandomBot")), ("SafeBot", breakdown.get("SafeBot")),
                 ("BestSnapshotBot", breakdown.get("BestSnapshotBot"))]
        items = [(name, stats) for name, stats in items if stats is not None]
        self._opponent_table.setRowCount(len(items))
        for row, (name, stats) in enumerate(items):
            self._opponent_table.setItem(row, 0, QTableWidgetItem(name))
            self._opponent_table.setItem(
                row, 1, QTableWidgetItem(f"{stats.get('mean_penalty', 0.0):.2f}")
            )
            self._opponent_table.setItem(
                row, 2, QTableWidgetItem(f"{stats.get('win_rate', 0.0):.2f}")
            )

    def _update_safe_delta(self, breakdown: dict, metrics: EvalSnapshot) -> None:
        safe = breakdown.get("SafeBot")
        if not safe:
            self._kpi_eval.set_delta("Δ vs SafeBot: -")
            self._kpi_eval_win.set_delta("Δ vs SafeBot: -")
            return
        delta = metrics.mean_penalty - safe.get("mean_penalty", metrics.mean_penalty)
        self._kpi_eval.set_delta(f"Δ vs SafeBot: {delta:+.2f} points/hand (lower is better)")
        win_delta = metrics.win_rate - safe.get("win_rate", metrics.win_rate)
        self._kpi_eval_win.set_delta(f"Δ vs SafeBot: {win_delta:+.2f}")

    def _update_safe_baseline(self, breakdown: dict) -> None:
        safe = breakdown.get("SafeBot")
        if not safe:
            return
        self._plot_penalty.set_baseline("safe_bot", safe.get("mean_penalty", 0.0), THEME.accent_warn)

    def _update_delta_plot(self, metrics, breakdown: dict, is_eval: bool) -> None:
        safe = breakdown.get("SafeBot")
        if not safe:
            return
        delta = metrics.mean_penalty - safe.get("mean_penalty", metrics.mean_penalty)
        series = "eval" if is_eval else "train"
        self._plot_delta.add_point(series, metrics.iteration, delta)

    def _show_best_params(self) -> None:
        if not self._best_params_json:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Best Params JSON")
        layout = QVBoxLayout()
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(self._best_params_json)
        layout.addWidget(text)
        dialog.setLayout(layout)
        dialog.resize(720, 480)
        dialog.exec()

    def _copy_best_params_json(self) -> None:
        if not self._best_params_json:
            return
        clipboard = QApplication.clipboard()
        clipboard.setText(self._best_params_json)

    def _emit_resume_last_best(self) -> None:
        self.resume_last_best.emit()

    def show_loaded_banner(self, run_id: str) -> None:
        self._banner_label.setText(f"Loaded best params from run {run_id} — click to open run")
        self._banner.setVisible(True)
        self._banner.setProperty("run_id", run_id)

    def _emit_open_run(self) -> None:
        run_id = self._banner.property("run_id")
        if run_id:
            self.open_run_requested.emit(str(run_id))

    def _apply_seed_mode(self) -> int:
        mode = self._seed_mode.currentText()
        seed = self._seed.value()
        if mode == "random":
            seed = int(time.time()) % 1000000
            self._seed.setValue(seed)
        elif mode == "cycle":
            seed += 1
            self._seed.setValue(seed)
        self._persist_latest_seed(seed, mode)
        return seed

    def _persist_latest_seed(self, seed: int, mode: str) -> None:
        from pathlib import Path
        import json

        path = Path.cwd() / "runs" / "latest_seed.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed": seed, "mode": mode}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_latest_seed_mode(self) -> None:
        from pathlib import Path
        import json

        path = Path.cwd() / "runs" / "latest_seed.json"
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        seed = payload.get("seed")
        mode = payload.get("mode")
        if isinstance(seed, int):
            self._seed.setValue(seed)
        if isinstance(mode, str) and mode in {"cycle", "random", "fixed"}:
            self._seed_mode.setCurrentText(mode)

    def _update_seed_warning(self) -> None:
        mode = self._seed_mode.currentText()
        if mode == "fixed":
            self._seed_warning.setText("Warning: fixed seed may overfit.")
        else:
            self._seed_warning.setText("")

    def _save_run_config(self) -> None:
        from pathlib import Path
        import json
        from datetime import datetime

        runs_dir = Path.cwd() / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        latest_path = runs_dir / "latest.json"
        run_id = None
        run_dir = None
        if latest_path.exists():
            try:
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                run_id = latest.get("run_id")
                if run_id:
                    run_dir = runs_dir / run_id
            except json.JSONDecodeError:
                run_dir = None
        if run_dir is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

        seed_mode = self._seed_mode.currentText()
        latest_seed_path = runs_dir / "latest_seed.json"
        latest_seed = {}
        if latest_seed_path.exists():
            try:
                latest_seed = json.loads(latest_seed_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                latest_seed = {}

        config = build_run_config(
            train_config={
                "hands_per_tick": self._hands_per_tick.value(),
                "updates_per_sec": self._updates_per_sec.value(),
                "hands_per_second_target": self._hands_per_second_target.value(),
                "fast_mode": self._fast_mode.isChecked(),
                "log_every_n_hands": self._log_every_n.value(),
                "log_verbosity": self._log_verbosity.currentText(),
            },
            eval_config={
                "eval_seed_preset": self._eval_seed_preset.currentText(),
                "eval_hands_per_seed": self._eval_hands.value(),
            },
            opponent_pools={
                "train": {
                    "RandomBot": self._train_random.isChecked(),
                    "SafeBot": self._train_safe.isChecked(),
                    "BestSnapshotBot": self._train_best.isChecked(),
                },
                "eval": {
                    "RandomBot": self._eval_random.isChecked(),
                    "SafeBot": self._eval_safe.isChecked(),
                    "BestSnapshotBot": self._eval_best.isChecked(),
                },
            },
            seed_mode=seed_mode,
            base_seed=self._seed.value(),
            latest_seed=latest_seed,
        )
        write_run_config(run_dir, config)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._layout_kpis()

    def _layout_kpis(self) -> None:
        width = self.width()
        if width < 1100:
            cols = 2
        elif width < 1400:
            cols = 3
        else:
            cols = 4
        for i in reversed(range(self._kpi_grid.count())):
            item = self._kpi_grid.itemAt(i)
            if item:
                self._kpi_grid.removeItem(item)
        for idx, card in enumerate(self._kpi_cards):
            self._kpi_grid.addWidget(card, idx // cols, idx % cols)

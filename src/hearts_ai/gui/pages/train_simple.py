from __future__ import annotations

import logging
import time
import re

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QFrame,
    QScrollArea,
    QProgressBar,
)

from hearts_ai.gui.services.training_worker import TrainingWorker
from hearts_ai.gui.theme import THEME
from hearts_ai.gui.widgets.metric_plot import MetricPlot
from hearts_ai.gui.widgets.kpi_card import KpiCard
from hearts_ai.training.autopolicy import AutoPolicy, PolicyAction
from hearts_ai.training.metrics import EvalSnapshot, HoldoutSnapshot, MetricsSnapshot
from hearts_ai.training.presets import (
    TrainingPreset,
    get_presets,
    seed_preset_to_list,
    apply_eval_budget,
    max_eval_hands_per_seed,
)


class SimpleTrainPage(QWidget):
    open_advanced_requested = Signal()
    open_run_requested = Signal(str)

    def __init__(self, worker: TrainingWorker) -> None:
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._worker = worker
        self._last_train_time = time.time()
        self._last_eval: EvalSnapshot | None = None
        self._last_holdout: HoldoutSnapshot | None = None
        self._best_eval_mean: float | None = None
        self._autopolicy = AutoPolicy()
        self._autopolicy_enabled = True
        self._autopolicy.set_allow_seed_preset_changes(False)
        self._presets = get_presets()
        self._run_length_mode = "10 min"
        self._run_started_at: float | None = None
        self._hands_per_tick = 200
        self._last_actions: dict[str, object] = {}
        self._last_metrics: MetricsSnapshot | None = None
        self._stop_reason: str | None = None
        self._last_hands_sec = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._tick_timer)
        self._eval_phase = ""
        self._status_state = "Idle"
        self._is_running = False
        self._eval_progress_total = 0
        self._eval_progress_done = 0
        self._eval_progress_started_at: float | None = None

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

        controls = QGroupBox("Training (simple)")
        controls_layout = QVBoxLayout()

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset"))
        self._preset_select = QComboBox()
        for preset in self._presets:
            self._preset_select.addItem(preset.name, preset.preset_id)
            self._preset_select.setItemData(
                self._preset_select.count() - 1, preset.description, Qt.ToolTipRole
            )
        preset_row.addWidget(self._preset_select)
        controls_layout.addLayout(preset_row)

        self._auto_mode = QCheckBox("Auto mode")
        self._auto_mode.setChecked(True)
        controls_layout.addWidget(self._auto_mode)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Run length"))
        self._run_length = QComboBox()
        self._run_length.addItems(["2 min", "10 min", "30 min", "until plateau"])
        self._run_length.setCurrentText("10 min")
        run_row.addWidget(self._run_length)
        controls_layout.addLayout(run_row)

        self._eval_budget_label = QLabel("Eval budget: -")
        controls_layout.addWidget(self._eval_budget_label)

        buttons = QHBoxLayout()
        self._start = QPushButton("Start")
        self._stop = QPushButton("Stop")
        self._run_eval = QPushButton("Run evaluation now")
        self._open_advanced = QPushButton("Open advanced settings")
        buttons.addWidget(self._start)
        buttons.addWidget(self._stop)
        buttons.addWidget(self._run_eval)
        buttons.addWidget(self._open_advanced)
        controls_layout.addLayout(buttons)

        self._suggestion = QLabel("Suggestions will appear here.")
        self._suggestion.setWordWrap(True)
        controls_layout.addWidget(self._suggestion)

        self._warning = QLabel("")
        self._warning.setWordWrap(True)
        controls_layout.addWidget(self._warning)

        self._preset_notice = QLabel("")
        self._preset_notice.setWordWrap(True)
        controls_layout.addWidget(self._preset_notice)

        status_row = QHBoxLayout()
        self._status_label = QLabel("Status: Idle")
        self._countdown_label = QLabel("")
        status_row.addWidget(self._status_label)
        status_row.addWidget(self._countdown_label)
        status_row.addStretch(1)
        controls_layout.addLayout(status_row)

        self._eval_progress = QProgressBar()
        self._eval_progress.setVisible(False)
        controls_layout.addWidget(self._eval_progress)

        self._perf_debug = QCheckBox("Perf debug")
        controls_layout.addWidget(self._perf_debug)

        controls.setLayout(controls_layout)

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

        self._plot_penalty = MetricPlot("Mean points/hand", y_min=0, y_max=26, y_label="points/hand")
        self._plot_win = MetricPlot("Win rate", y_min=0, y_max=1, y_label="rate")
        self._plot_qs = MetricPlot("Q♠ taken rate", y_min=0, y_max=1, y_label="rate")
        self._plot_penalty.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_win.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_qs.add_series("eval", THEME.eval_points, style=Qt.DashLine, symbol="o")
        self._plot_penalty.add_series("train", THEME.train_line)
        self._plot_win.add_series("train", THEME.train_line)
        self._plot_qs.add_series("train", THEME.train_line)
        self._plot_penalty.set_baseline("avg_points", 6.5, THEME.baseline_line)

        plots = QGridLayout()
        plots.addWidget(self._plot_penalty, 0, 0)
        plots.addWidget(self._plot_win, 0, 1)
        plots.addWidget(self._plot_qs, 1, 0)

        self._opponent_table = QTableWidget(0, 3)
        self._opponent_table.setHorizontalHeaderLabels(["Opponent", "Mean points/hand", "Win rate"])
        self._opponent_table.horizontalHeader().setStretchLastSection(True)
        self._opponent_table.setAlternatingRowColors(True)

        main = QVBoxLayout()
        main.addWidget(self._banner)
        main.addWidget(controls)
        main.addLayout(self._kpi_grid)
        main.addWidget(self._opponent_table)
        main.addLayout(plots)

        container = QWidget()
        container.setLayout(main)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(container)

        layout = QVBoxLayout()
        layout.addWidget(scroll)
        self.setLayout(layout)

        self._start.clicked.connect(self._start_training)
        self._stop.clicked.connect(self._handle_stop_clicked)
        self._run_eval.clicked.connect(self._worker.run_eval_now)
        self._open_advanced.clicked.connect(lambda: self.open_advanced_requested.emit())
        self._auto_mode.toggled.connect(self._toggle_auto_mode)
        self._preset_select.currentIndexChanged.connect(self._handle_preset_change)
        self._run_length.currentTextChanged.connect(self._update_run_length)
        self._banner_close.clicked.connect(self._banner.hide)
        self._banner_open.clicked.connect(self._emit_open_run)
        self._perf_debug.toggled.connect(self._worker.update_perf_log_enabled)

        self._kpi_holdout.set_value("—")
        self._layout_kpis()
        self._apply_selected_preset()
        self._autopolicy.set_run_length_mode(self._run_length_mode)
        self._perf_debug.setChecked(False)

    def on_metrics(self, metrics: MetricsSnapshot) -> None:
        self._last_metrics = metrics
        self._plot_penalty.add_point("train", metrics.iteration, metrics.mean_penalty)
        self._plot_win.add_point("train", metrics.iteration, metrics.win_rate)
        self._plot_qs.add_point("train", metrics.iteration, metrics.qs_rate)
        now = time.time()
        dt = max(1e-6, now - self._last_train_time)
        hands_sec = self._hands_per_tick / dt if self._hands_per_tick > 0 else 0.0
        self._last_hands_sec = hands_sec
        self._last_train_time = now
        self._kpi_train.set_value(f"{metrics.mean_penalty:.2f} pts/hand")
        self._kpi_hands_sec.set_value(f"{hands_sec:.1f}")
        self._update_status_row(hands_sec)
        self._check_small_run_warning(metrics.hands_done)

        if self._autopolicy_enabled:
            result = self._autopolicy.update(None, None, metrics)
            self._apply_policy_actions(result.actions, metrics=metrics)

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
        self._update_opponent_table(metrics.opponent_breakdown)

        if self._autopolicy_enabled:
            self._autopolicy.set_run_length_mode(self._run_length_mode)
            result = self._autopolicy.update(metrics, self._last_holdout, None)
            self._apply_policy_actions(result.actions, eval_snapshot=metrics)
            if result.note:
                self._suggestion.setText(result.note)
                self._record_autopolicy_decision("note", result.note)
            if result.suggestion:
                self._suggestion.setText(result.suggestion)
                self._record_autopolicy_decision("suggestion", result.suggestion)
                if self._run_length_mode == "until plateau":
                    self._request_stop("plateau")

    def on_holdout_metrics(self, metrics: HoldoutSnapshot) -> None:
        self._last_holdout = metrics
        se_text = f" ±{metrics.mean_penalty_se:.2f}" if metrics.mean_penalty_se > 0 else ""
        self._kpi_holdout.set_value(f"{metrics.mean_penalty:.2f} pts/hand{se_text}")
        if self._autopolicy_enabled and self._last_eval:
            result = self._autopolicy.update(self._last_eval, metrics, None)
            self._apply_policy_actions(result.actions, eval_snapshot=self._last_eval)

    def on_status(self, status: str) -> None:
        running = status in {"running", "paused", "stopping"}
        self._is_running = running
        self._start.setEnabled(not running)
        self._stop.setEnabled(running)
        self._preset_select.setEnabled(not running)
        self._auto_mode.setEnabled(not running)
        if status == "running":
            self._run_started_at = self._run_started_at or time.time()
            self._timer.start()
        elif status in {"idle", "stopping"}:
            if status == "idle" and self._stop_reason:
                self._show_stop_banner(self._stop_reason)
            self._run_started_at = None
            self._timer.stop()
        self._status_state = status.capitalize() if status else "Idle"
        self._status_label.setText(f"Status: {self._status_state}")

    def show_loaded_banner(self, run_id: str) -> None:
        self._banner_label.setText(f"Loaded best params from run {run_id} — click to open run")
        self._banner.setVisible(True)
        self._banner.setProperty("run_id", run_id)

    def _emit_open_run(self) -> None:
        run_id = self._banner.property("run_id")
        if run_id:
            self.open_run_requested.emit(str(run_id))

    def _toggle_auto_mode(self, enabled: bool) -> None:
        if self._is_running:
            self._preset_notice.setText("Auto mode changes apply next run.")
            return
        self._autopolicy_enabled = enabled
        self._worker.update_autopolicy_enabled(enabled)

    def _update_run_length(self, text: str) -> None:
        self._run_length_mode = text
        self._autopolicy.set_run_length_mode(text)
        seconds = self._run_length_seconds()
        self._worker.update_run_length_mode(text)
        self._worker.update_run_length_seconds(seconds or 0)
        self._worker.update_eval_interval_seconds(self._eval_interval_seconds_for_run_length())
        preset = self._current_preset()
        max_per_seed = max_eval_hands_per_seed(text)
        seed_preset = preset.eval_seed_preset
        if text == "2 min" and preset.preset_id != "thorough":
            seed_preset = "quick"
        seeds_count = len(seed_preset_to_list(seed_preset))
        total = max_per_seed * seeds_count
        self._eval_budget_label.setText(f"Eval budget: {total} hands per eval")

    def _check_run_length(self) -> None:
        if self._run_length_mode == "until plateau" or self._run_started_at is None:
            return
        seconds = self._run_length_seconds()
        if seconds and time.time() - self._run_started_at >= seconds:
            self._suggestion.setText("Run length reached → stopping.")
            self._request_stop("time_limit")
            self._run_started_at = None

    def _apply_selected_preset(self) -> None:
        preset = self._current_preset()
        hands_per_tick = max(50, preset.hands_per_tick)
        eval_hands = max(200, preset.eval_hands_per_seed)
        eval_seed_preset = preset.eval_seed_preset
        eval_interval = preset.eval_interval
        holdout_enabled = preset.holdout_enabled
        if self._run_length_mode == "2 min" and preset.preset_id != "thorough":
            eval_seed_preset = "quick"
            eval_interval = max(20, eval_interval)
            holdout_enabled = False
            eval_hands = min(eval_hands, 200)
        eval_interval_seconds = self._eval_interval_seconds_for_run_length()
        eval_seed_preset, eval_hands, total_eval_hands = apply_eval_budget(
            eval_seed_preset, eval_hands, self._run_length_mode
        )
        if eval_hands < 200:
            eval_hands = 200
            seeds_count = len(seed_preset_to_list(eval_seed_preset))
            total_eval_hands = seeds_count * eval_hands
            max_seeds = max(1, total_eval_hands // eval_hands)
            if max_seeds <= 5:
                eval_seed_preset = "quick"
            elif max_seeds <= 20:
                eval_seed_preset = "standard"
            else:
                eval_seed_preset = "thorough"
        total_eval_hands = len(seed_preset_to_list(eval_seed_preset)) * eval_hands
        if (
            hands_per_tick != preset.hands_per_tick
            or eval_hands != preset.eval_hands_per_seed
            or eval_seed_preset != preset.eval_seed_preset
            or eval_interval != preset.eval_interval
            or holdout_enabled != preset.holdout_enabled
        ):
            preset = TrainingPreset(
                preset_id=preset.preset_id,
                name=preset.name,
                description=preset.description,
                hands_per_tick=hands_per_tick,
                updates_per_sec=preset.updates_per_sec,
                hands_per_second_target=preset.hands_per_second_target,
                fast_mode=preset.fast_mode,
                train_seed_mode=preset.train_seed_mode,
                train_opponents=preset.train_opponents,
                eval_opponents=preset.eval_opponents,
                eval_interval=eval_interval,
                eval_hands_per_seed=eval_hands,
                eval_seed_preset=eval_seed_preset,
                holdout_enabled=holdout_enabled,
            )
        self._hands_per_tick = hands_per_tick
        self._worker.apply_preset(preset)
        self._worker.update_autopolicy_enabled(self._auto_mode.isChecked())
        self._worker.update_run_length_mode(self._run_length_mode)
        self._worker.update_run_length_seconds(self._run_length_seconds() or 0)
        self._worker.update_eval_interval_seconds(eval_interval_seconds)
        self._worker.update_simple_max_tick_seconds(0.25)
        self._eval_budget_label.setText(f"Eval budget: {total_eval_hands} hands per eval")
        self._logger.info(
            "Simple preset applied: %s (hands_per_tick=%d, eval=%dx%d, holdout=%dx%d, fast_mode=%s)",
            preset.preset_id,
            hands_per_tick,
            len(seed_preset_to_list(preset.eval_seed_preset)),
            eval_hands,
            5 if preset.holdout_enabled else 0,
            max(1, eval_hands // 4) if preset.holdout_enabled else 0,
            preset.fast_mode,
        )

    def _apply_policy_actions(
        self,
        actions: list[PolicyAction],
        eval_snapshot: EvalSnapshot | None = None,
        metrics: MetricsSnapshot | None = None,
    ) -> None:
        for action in actions:
            if self._last_actions.get(action.action) == action.value:
                continue
            if action.action == "seed_mode":
                self._worker.update_seed_mode(str(action.value))
            elif action.action == "eval_seed_preset":
                self._worker.update_eval_seeds(seed_preset_to_list(str(action.value)))
            elif action.action == "eval_hands_per_seed":
                self._worker.update_eval_hands_per_seed(int(action.value))
            elif action.action == "eval_opponents":
                self._worker.update_eval_opponents(list(action.value))
            elif action.action == "train_opponents":
                self._worker.update_train_opponents(list(action.value))
            elif action.action == "hands_per_tick":
                self._hands_per_tick = int(action.value)
                self._worker.update_hands_per_tick(self._hands_per_tick)
            elif action.action == "hands_per_second_target":
                self._worker.update_hands_per_second_target(float(action.value))
            elif action.action == "fast_mode":
                self._worker.update_fast_mode(bool(action.value))
            else:
                continue
            self._last_actions[action.action] = action.value
            self._record_autopolicy_decision(action.action, action.reason)

    def _start_training(self) -> None:
        self._apply_selected_preset()
        if not self._presets:
            self._presets = get_presets()
            if not self._presets:
                return
        self._worker.update_stop_reason("")
        self._worker.update_stop_requested_at("")
        self._preset_notice.setText("")
        self._worker.start()
        self._stop_reason = None

    def _update_opponent_table(self, breakdown: dict) -> None:
        items = [
            ("RandomBot", breakdown.get("RandomBot")),
            ("SafeBot", breakdown.get("SafeBot")),
            ("BestSnapshotBot", breakdown.get("BestSnapshotBot")),
        ]
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

    def _handle_stop_clicked(self) -> None:
        self._request_stop("user")

    def _request_stop(self, reason: str) -> None:
        if self._stop_reason:
            return
        self._stop_reason = reason
        self._worker.update_stop_reason(reason)
        self._worker.update_stop_requested_at(self._format_elapsed())
        self._worker.stop()

    def _run_length_seconds(self) -> int | None:
        if self._run_length_mode == "2 min":
            return 120
        if self._run_length_mode == "10 min":
            return 600
        if self._run_length_mode == "30 min":
            return 1800
        return None

    def _eval_interval_seconds_for_run_length(self) -> float:
        if self._run_length_mode == "2 min":
            return 15.0
        if self._run_length_mode == "10 min":
            return 30.0
        if self._run_length_mode == "30 min":
            return 45.0
        return 30.0

    def _format_elapsed(self) -> str:
        if not self._run_started_at:
            return "00:00"
        elapsed = max(0, int(time.time() - self._run_started_at))
        minutes = elapsed // 60
        seconds = elapsed % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _tick_timer(self) -> None:
        if not self._run_started_at:
            return
        if self._run_length_mode != "until plateau":
            self._check_run_length()
        self._update_status_row()

    def _update_status_row(self, hands_sec: float | None = None) -> None:
        elapsed = self._format_elapsed()
        hands_done = self._last_metrics.hands_done if self._last_metrics else 0
        if hands_sec is None:
            hands_sec = self._last_hands_sec
        target = self._run_length_seconds()
        limit_text = f" / {target // 60:02d}:{target % 60:02d}" if target else ""
        status = self._status_state
        stop_text = self._stop_reason or "none yet"
        phase_note = " (training paused during evaluation)" if self._eval_phase else ""
        eta_text = ""
        if self._eval_progress_total and self._eval_progress_started_at:
            elapsed = max(1e-6, time.time() - self._eval_progress_started_at)
            rate = self._eval_progress_done / elapsed
            remaining = max(0, self._eval_progress_total - self._eval_progress_done)
            eta = int(remaining / rate) if rate > 0 else 0
            eta_text = f" • ETA {eta // 60:02d}:{eta % 60:02d}"
        phase = f" • {self._eval_phase}{phase_note}{eta_text}" if self._eval_phase else ""
        self._status_label.setText(
            f"{status}{phase} • elapsed {elapsed}{limit_text} • hands {hands_done:,} "
            f"• hands/s {hands_sec:.1f} • hands_per_tick {self._hands_per_tick} • stop: {stop_text}"
        )
        if target and self._run_started_at:
            remaining = max(0, target - int(time.time() - self._run_started_at))
            self._countdown_label.setText(f"Remaining {remaining // 60:02d}:{remaining % 60:02d}")
        else:
            self._countdown_label.setText("")

    def _record_autopolicy_decision(self, action: str, reason: str) -> None:
        payload = {"t": self._format_elapsed(), "action": action, "reason": reason}
        self._worker.add_autopolicy_decision(payload)

    def _check_small_run_warning(self, hands_done: int) -> None:
        if hands_done < 1000:
            self._warning.setText(
                "This run is too small to be meaningful. Increase run length or hands_per_tick."
            )
        else:
            self._warning.setText("")

    def _show_stop_banner(self, reason: str) -> None:
        message = "Stopped."
        if reason == "time_limit":
            limit = self._run_length_mode
            message = f"Stopped: time limit reached ({limit})."
        elif reason == "plateau":
            message = "Stopped: plateau detected."
        elif reason == "user":
            message = "Stopped: user requested."
        self._banner_label.setText(message)
        self._banner.setVisible(True)

    def _handle_preset_change(self) -> None:
        if self._is_running:
            self._preset_notice.setText("Preset changes apply next run.")
            return
        self._apply_selected_preset()

    def _current_preset(self) -> TrainingPreset:
        if not self._presets:
            self._presets = get_presets()
        idx = self._preset_select.currentIndex()
        if 0 <= idx < len(self._presets):
            return self._presets[idx]
        for preset in self._presets:
            if preset.preset_id == "balanced":
                return preset
        return self._presets[0]

    def on_eval_status(self, status: str) -> None:
        if not status or status == "idle":
            self._eval_phase = ""
            self._eval_progress.setVisible(False)
            self._eval_progress.reset()
            self._eval_progress_total = 0
            self._eval_progress_done = 0
            self._eval_progress_started_at = None
            self._update_status_row()
            return
        phase, progress = self._parse_eval_progress(status)
        if progress:
            current, total = progress
            self._eval_progress.setMaximum(total)
            self._eval_progress.setValue(current)
            self._eval_progress.setVisible(True)
            if total != self._eval_progress_total:
                self._eval_progress_total = total
                self._eval_progress_done = current
                self._eval_progress_started_at = time.time()
            else:
                self._eval_progress_done = current
        else:
            self._eval_progress.setVisible(False)
        self._update_status_row()

    def on_phase(self, phase: str) -> None:
        if phase == "evaluating":
            self._eval_phase = "Evaluating"
        elif phase == "holdout":
            self._eval_phase = "Holdout"
        elif phase in {"training", "idle"}:
            self._eval_phase = ""
        self._update_status_row()

    def _parse_eval_progress(self, status: str) -> tuple[str, tuple[int, int] | None]:
        phase = "Evaluating"
        text = status
        if status.startswith("holdout "):
            phase = "Holdout"
            text = status.replace("holdout ", "", 1)
        match = re.search(r"seed (\d+)/(\d+) hand (\d+)/(\d+) total (\d+)/(\d+)", text)
        if not match:
            return phase, None
        total_done = int(match.group(5))
        total_total = int(match.group(6))
        seed_i = match.group(1)
        seed_total = match.group(2)
        hand_i = match.group(3)
        hand_total = match.group(4)
        self._eval_phase = f"{phase} (seed {seed_i}/{seed_total}, hand {hand_i}/{hand_total})"
        return phase, (total_done, total_total)

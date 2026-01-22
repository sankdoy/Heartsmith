from __future__ import annotations

import logging

from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt

from hearts_ai.training.control import TrainControl
from hearts_ai.training.metrics import EvalSnapshot, HoldoutSnapshot, MetricsSnapshot
from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trainer import Trainer, TrainingConfig
from hearts_ai.training.presets import TrainingPreset, preset_to_values


class TrainerRunner(QObject):
    metrics_updated = Signal(MetricsSnapshot)
    best_params_updated = Signal(str)
    eval_metrics_updated = Signal(EvalSnapshot)
    holdout_metrics_updated = Signal(HoldoutSnapshot)
    eval_status_updated = Signal(str)
    overfit_status_updated = Signal(str)
    error_occurred = Signal(str)
    eval_finished = Signal()
    run_started = Signal(str)
    phase_updated = Signal(str)
    finished = Signal()

    def __init__(self, params: ParameterSet, config: TrainingConfig, control: TrainControl) -> None:
        super().__init__()
        self._params = params
        self._config = config
        self._control = control
        self._trainer = Trainer(self._params, self._config, control=self._control)
    @Slot()
    def start_run(self) -> None:
        try:
            self._trainer.run(
                self._emit_metrics,
                self._emit_best_params,
                self._emit_eval,
                self._emit_eval_status,
                self._emit_overfit,
                self._emit_holdout,
                self._emit_run_started,
                self._emit_phase,
            )
        except Exception:
            logging.exception("Training worker crashed")
            self.error_occurred.emit("Training worker crashed")
        finally:
            self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self._trainer.stop()

    @Slot()
    def pause(self) -> None:
        self._trainer.pause()

    @Slot()
    def resume(self) -> None:
        self._trainer.resume()

    @Slot()
    def reset(self) -> None:
        self._trainer.reset()

    @Slot()
    def run_eval_now(self) -> None:
        self._emit_phase("evaluating")
        self._trainer.run_eval_now(self._emit_eval, self._emit_eval_status)
        self.eval_finished.emit()
        self._emit_phase("training")

    @Slot(int)
    def update_hands_per_tick(self, hands_per_tick: int) -> None:
        self._config.hands_per_tick = hands_per_tick

    @Slot(int)
    def update_seed(self, seed: int) -> None:
        self._config.seed = seed

    @Slot(str)
    def update_seed_mode(self, mode: str) -> None:
        self._config.seed_mode = mode

    @Slot(float)
    def update_updates_per_sec(self, updates_per_sec: float) -> None:
        self._config.updates_per_sec = max(1.0, updates_per_sec)

    @Slot(float)
    def update_hands_per_second_target(self, target: float) -> None:
        self._config.hands_per_second_target = target if target > 0 else None

    @Slot(bool)
    def update_fast_mode(self, enabled: bool) -> None:
        self._config.fast_mode = enabled

    @Slot(bool)
    def update_debug_explain_enabled(self, enabled: bool) -> None:
        self._config.debug_explain_enabled = enabled

    @Slot(int)
    def update_log_every_n_hands(self, count: int) -> None:
        self._config.log_every_n_hands = max(1, count)

    @Slot(bool)
    def update_capture_sample_enabled(self, enabled: bool) -> None:
        self._config.capture_sample_enabled = enabled

    @Slot(int)
    def update_capture_sample_every(self, count: int) -> None:
        self._config.capture_sample_every = max(1, count)

    @Slot()
    def request_capture_next(self) -> None:
        self._trainer.request_capture_next()

    @Slot(object)
    def update_eval_seeds(self, seeds: list[int]) -> None:
        self._config.eval_seeds = seeds

    @Slot(int)
    def update_eval_hands_per_seed(self, count: int) -> None:
        self._config.eval_hands_per_seed = count

    @Slot(int)
    def update_eval_interval(self, interval: int) -> None:
        self._config.eval_interval = max(1, interval)

    @Slot(float)
    def update_eval_interval_seconds(self, interval: float) -> None:
        self._config.eval_interval_seconds = interval if interval > 0 else None

    @Slot(str)
    def update_preset_id(self, preset_id: str) -> None:
        self._config.preset_id = preset_id

    @Slot(bool)
    def update_autopolicy_enabled(self, enabled: bool) -> None:
        self._config.autopolicy_enabled = enabled

    @Slot(str)
    def add_autopolicy_decision(self, decision: dict) -> None:
        self._config.autopolicy_decisions.append(decision)

    @Slot(str)
    def update_run_length_mode(self, mode: str) -> None:
        self._config.run_length_mode = mode

    @Slot(int)
    def update_run_length_seconds(self, seconds: int) -> None:
        self._config.run_length_seconds = seconds

    @Slot(str)
    def update_stop_reason(self, reason: str) -> None:
        self._config.stop_reason = reason

    @Slot(str)
    def update_stop_requested_at(self, timestamp: str) -> None:
        self._config.stop_requested_at = timestamp

    @Slot(float)
    def update_simple_max_tick_seconds(self, seconds: float) -> None:
        self._config.simple_max_tick_seconds = seconds if seconds > 0 else None

    @Slot(bool)
    def update_perf_log_enabled(self, enabled: bool) -> None:
        self._config.perf_log_enabled = enabled

    @Slot(object)
    def update_train_opponents(self, pool: list[str]) -> None:
        self._config.train_opponents = pool

    @Slot(object)
    def update_eval_opponents(self, pool: list[str]) -> None:
        self._config.eval_opponents = pool

    @Slot(object)
    def update_holdout_seeds(self, seeds: list[int]) -> None:
        self._config.holdout_seeds = seeds

    @Slot(str, float)
    def update_param(self, name: str, value: float) -> None:
        self._params.update(name, value)
        self._trainer.update_params(self._params)

    @Slot(str, bool)
    def update_locked(self, name: str, locked: bool) -> None:
        self._params.set_locked(name, locked)

    def _emit_metrics(self, metrics: MetricsSnapshot) -> None:
        self.metrics_updated.emit(metrics)

    def _emit_best_params(self, params: ParameterSet) -> None:
        self.best_params_updated.emit(params.to_json())

    def _emit_eval(self, metrics: EvalSnapshot) -> None:
        self.eval_metrics_updated.emit(metrics)

    def _emit_holdout(self, metrics: HoldoutSnapshot) -> None:
        self.holdout_metrics_updated.emit(metrics)

    def _emit_eval_status(self, status: str) -> None:
        self.eval_status_updated.emit(status)

    def _emit_overfit(self, status: str) -> None:
        self.overfit_status_updated.emit(status)

    def _emit_run_started(self, run_id: str) -> None:
        self.run_started.emit(run_id)

    def _emit_phase(self, phase: str) -> None:
        self.phase_updated.emit(phase)


class TrainingWorker(QObject):
    metrics_updated = Signal(MetricsSnapshot)
    best_params_updated = Signal(str)
    eval_metrics_updated = Signal(EvalSnapshot)
    holdout_metrics_updated = Signal(HoldoutSnapshot)
    eval_status_updated = Signal(str)
    overfit_status_updated = Signal(str)
    error_occurred = Signal(str)
    status_updated = Signal(str)
    run_started = Signal(str)
    phase_updated = Signal(str)

    request_start = Signal()
    request_stop = Signal()
    request_pause = Signal()
    request_resume = Signal()
    request_reset = Signal()
    request_run_eval_now = Signal()
    request_hands_per_tick = Signal(int)
    request_seed = Signal(int)
    request_seed_mode = Signal(str)
    request_param_update = Signal(str, float)
    request_updates_per_sec = Signal(float)
    request_locked_update = Signal(str, bool)
    request_eval_seeds = Signal(object)
    request_eval_hands_per_seed = Signal(int)
    request_eval_interval = Signal(int)
    request_eval_interval_seconds = Signal(float)
    request_train_opponents = Signal(object)
    request_eval_opponents = Signal(object)
    request_holdout_seeds = Signal(object)
    request_hands_per_second_target = Signal(float)
    request_fast_mode = Signal(bool)
    request_debug_explain_enabled = Signal(bool)
    request_log_every_n_hands = Signal(int)
    request_capture_sample_enabled = Signal(bool)
    request_capture_sample_every = Signal(int)
    request_capture_next = Signal()
    request_preset_id = Signal(str)
    request_autopolicy_enabled = Signal(bool)
    request_autopolicy_decision = Signal(object)
    request_run_length_mode = Signal(str)
    request_run_length_seconds = Signal(int)
    request_stop_reason = Signal(str)
    request_stop_requested_at = Signal(str)
    request_simple_max_tick_seconds = Signal(float)
    request_perf_log_enabled = Signal(bool)

    def __init__(self, params: ParameterSet, config: TrainingConfig) -> None:
        super().__init__()
        self._params = params
        self._config = config
        self._thread: QThread | None = None
        self._runner: TrainerRunner | None = None
        self._running = False
        self._control: TrainControl | None = None
        self._current_run_id: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.isRunning():
            return
        self._config.autopolicy_decisions = []
        self._thread = QThread()
        self._control = TrainControl()
        self._runner = TrainerRunner(self._params, self._config, self._control)
        self._runner.moveToThread(self._thread)

        self.request_start.connect(self._runner.start_run, Qt.QueuedConnection)
        self.request_stop.connect(self._runner.stop, Qt.DirectConnection)
        self.request_pause.connect(self._runner.pause, Qt.DirectConnection)
        self.request_resume.connect(self._runner.resume, Qt.DirectConnection)
        self.request_reset.connect(self._runner.reset, Qt.DirectConnection)
        self.request_run_eval_now.connect(self._runner.run_eval_now, Qt.QueuedConnection)
        self.request_hands_per_tick.connect(self._runner.update_hands_per_tick, Qt.QueuedConnection)
        self.request_seed.connect(self._runner.update_seed, Qt.QueuedConnection)
        self.request_seed_mode.connect(self._runner.update_seed_mode, Qt.QueuedConnection)
        self.request_param_update.connect(self._runner.update_param, Qt.QueuedConnection)
        self.request_updates_per_sec.connect(self._runner.update_updates_per_sec, Qt.QueuedConnection)
        self.request_locked_update.connect(self._runner.update_locked, Qt.QueuedConnection)
        self.request_eval_seeds.connect(self._runner.update_eval_seeds, Qt.QueuedConnection)
        self.request_eval_hands_per_seed.connect(self._runner.update_eval_hands_per_seed, Qt.QueuedConnection)
        self.request_eval_interval.connect(self._runner.update_eval_interval, Qt.QueuedConnection)
        self.request_eval_interval_seconds.connect(
            self._runner.update_eval_interval_seconds, Qt.QueuedConnection
        )
        self.request_train_opponents.connect(self._runner.update_train_opponents, Qt.QueuedConnection)
        self.request_eval_opponents.connect(self._runner.update_eval_opponents, Qt.QueuedConnection)
        self.request_holdout_seeds.connect(self._runner.update_holdout_seeds, Qt.QueuedConnection)
        self.request_hands_per_second_target.connect(
            self._runner.update_hands_per_second_target, Qt.QueuedConnection
        )
        self.request_fast_mode.connect(self._runner.update_fast_mode, Qt.QueuedConnection)
        self.request_debug_explain_enabled.connect(
            self._runner.update_debug_explain_enabled, Qt.QueuedConnection
        )
        self.request_log_every_n_hands.connect(self._runner.update_log_every_n_hands, Qt.QueuedConnection)
        self.request_capture_sample_enabled.connect(
            self._runner.update_capture_sample_enabled, Qt.QueuedConnection
        )
        self.request_capture_sample_every.connect(
            self._runner.update_capture_sample_every, Qt.QueuedConnection
        )
        self.request_capture_next.connect(self._runner.request_capture_next, Qt.QueuedConnection)
        self.request_preset_id.connect(self._runner.update_preset_id, Qt.QueuedConnection)
        self.request_autopolicy_enabled.connect(
            self._runner.update_autopolicy_enabled, Qt.QueuedConnection
        )
        self.request_autopolicy_decision.connect(
            self._runner.add_autopolicy_decision, Qt.QueuedConnection
        )
        self.request_run_length_mode.connect(self._runner.update_run_length_mode, Qt.QueuedConnection)
        self.request_run_length_seconds.connect(
            self._runner.update_run_length_seconds, Qt.QueuedConnection
        )
        self.request_stop_reason.connect(self._runner.update_stop_reason, Qt.QueuedConnection)
        self.request_stop_requested_at.connect(
            self._runner.update_stop_requested_at, Qt.QueuedConnection
        )
        self.request_simple_max_tick_seconds.connect(
            self._runner.update_simple_max_tick_seconds, Qt.QueuedConnection
        )
        self.request_perf_log_enabled.connect(
            self._runner.update_perf_log_enabled, Qt.QueuedConnection
        )

        self._runner.metrics_updated.connect(self.metrics_updated)
        self._runner.best_params_updated.connect(self.best_params_updated)
        self._runner.eval_metrics_updated.connect(self.eval_metrics_updated)
        self._runner.holdout_metrics_updated.connect(self.holdout_metrics_updated)
        self._runner.eval_status_updated.connect(self.eval_status_updated)
        self._runner.overfit_status_updated.connect(self.overfit_status_updated)
        self._runner.error_occurred.connect(self.error_occurred)
        self._runner.run_started.connect(self._set_current_run)
        self._runner.run_started.connect(self.run_started)
        self._runner.phase_updated.connect(self.phase_updated)
        self._runner.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._clear_runner)

        self._thread.started.connect(self.request_start)
        self._thread.start()
        self._running = True
        self.status_updated.emit("running")

    def stop(self) -> None:
        if not self._runner:
            return
        if self._control:
            self._control.request_stop()
        self.status_updated.emit("stopping")

    def pause(self) -> None:
        if not self._runner:
            return
        if self._control:
            self._control.request_pause(True)
        self.status_updated.emit("paused")

    def resume(self) -> None:
        if not self._runner:
            return
        if self._control:
            self._control.request_pause(False)
        self.status_updated.emit("running")

    def reset(self) -> None:
        if not self._runner:
            return
        if self._control:
            self._control.request_pause(False)
        self._runner.reset()
        self.status_updated.emit("reset")

    def run_eval_now(self) -> None:
        if not self._runner:
            self._thread = QThread()
            self._control = TrainControl()
            self._runner = TrainerRunner(self._params, self._config, self._control)
            self._runner.moveToThread(self._thread)
            self.request_run_eval_now.connect(self._runner.run_eval_now, Qt.QueuedConnection)
            self._runner.eval_metrics_updated.connect(self.eval_metrics_updated)
            self._runner.eval_status_updated.connect(self.eval_status_updated)
            self._runner.error_occurred.connect(self.error_occurred)
            self._runner.eval_finished.connect(self._thread.quit)
            self._thread.finished.connect(self._clear_runner)
            self._thread.started.connect(self.request_run_eval_now)
            self._thread.start()
            self._running = True
            return
        self.request_run_eval_now.emit()

    def update_hands_per_tick(self, hands_per_tick: int) -> None:
        if self._runner is None:
            self._config.hands_per_tick = hands_per_tick
            return
        self.request_hands_per_tick.emit(hands_per_tick)

    def update_seed(self, seed: int) -> None:
        if self._runner is None:
            self._config.seed = seed
            return
        self.request_seed.emit(seed)

    def update_seed_mode(self, mode: str) -> None:
        if self._runner is None:
            self._config.seed_mode = mode
            return
        self.request_seed_mode.emit(mode)

    def update_param(self, name: str, value: float) -> None:
        if self._runner is None:
            self._params.update(name, value)
            return
        self.request_param_update.emit(name, value)

    def update_locked(self, name: str, locked: bool) -> None:
        if self._runner is None:
            self._params.set_locked(name, locked)
            return
        self.request_locked_update.emit(name, locked)

    def update_updates_per_sec(self, updates_per_sec: float) -> None:
        if self._runner is None:
            self._config.updates_per_sec = max(1.0, updates_per_sec)
            return
        self.request_updates_per_sec.emit(updates_per_sec)

    def update_hands_per_second_target(self, target: float) -> None:
        if self._runner is None:
            self._config.hands_per_second_target = target if target > 0 else None
            return
        self.request_hands_per_second_target.emit(target)

    def update_fast_mode(self, enabled: bool) -> None:
        if self._runner is None:
            self._config.fast_mode = enabled
            return
        self.request_fast_mode.emit(enabled)

    def update_debug_explain_enabled(self, enabled: bool) -> None:
        if self._runner is None:
            self._config.debug_explain_enabled = enabled
            return
        self.request_debug_explain_enabled.emit(enabled)

    def update_log_every_n_hands(self, count: int) -> None:
        if self._runner is None:
            self._config.log_every_n_hands = max(1, count)
            return
        self.request_log_every_n_hands.emit(count)

    def update_capture_sample_enabled(self, enabled: bool) -> None:
        if self._runner is None:
            self._config.capture_sample_enabled = enabled
            return
        self.request_capture_sample_enabled.emit(enabled)

    def update_capture_sample_every(self, count: int) -> None:
        if self._runner is None:
            self._config.capture_sample_every = max(1, count)
            return
        self.request_capture_sample_every.emit(count)

    def capture_next_sample(self) -> None:
        if self._runner is None:
            return
        self.request_capture_next.emit()

    def update_eval_seeds(self, seeds: list[int]) -> None:
        if self._runner is None:
            self._config.eval_seeds = seeds
            return
        self.request_eval_seeds.emit(seeds)

    def update_eval_hands_per_seed(self, count: int) -> None:
        if self._runner is None:
            self._config.eval_hands_per_seed = count
            return
        self.request_eval_hands_per_seed.emit(count)

    def update_eval_interval(self, interval: int) -> None:
        if self._runner is None:
            self._config.eval_interval = max(1, interval)
            return
        self.request_eval_interval.emit(interval)

    def update_eval_interval_seconds(self, interval: float) -> None:
        if self._runner is None:
            self._config.eval_interval_seconds = interval if interval > 0 else None
            return
        self.request_eval_interval_seconds.emit(interval)

    def update_preset_id(self, preset_id: str) -> None:
        if self._runner is None:
            self._config.preset_id = preset_id
            return
        self.request_preset_id.emit(preset_id)

    def update_autopolicy_enabled(self, enabled: bool) -> None:
        if self._runner is None:
            self._config.autopolicy_enabled = enabled
            return
        self.request_autopolicy_enabled.emit(enabled)

    def add_autopolicy_decision(self, decision: dict) -> None:
        if self._runner is None:
            self._config.autopolicy_decisions.append(decision)
            return
        self.request_autopolicy_decision.emit(decision)

    def update_run_length_mode(self, mode: str) -> None:
        if self._runner is None:
            self._config.run_length_mode = mode
            return
        self.request_run_length_mode.emit(mode)

    def update_run_length_seconds(self, seconds: int) -> None:
        if self._runner is None:
            self._config.run_length_seconds = seconds
            return
        self.request_run_length_seconds.emit(seconds)

    def update_stop_reason(self, reason: str) -> None:
        if self._runner is None:
            self._config.stop_reason = reason
            return
        self.request_stop_reason.emit(reason)

    def update_stop_requested_at(self, timestamp: str) -> None:
        if self._runner is None:
            self._config.stop_requested_at = timestamp
            return
        self.request_stop_requested_at.emit(timestamp)

    def update_simple_max_tick_seconds(self, seconds: float) -> None:
        if self._runner is None:
            self._config.simple_max_tick_seconds = seconds if seconds > 0 else None
            return
        self.request_simple_max_tick_seconds.emit(seconds)

    def update_perf_log_enabled(self, enabled: bool) -> None:
        if self._runner is None:
            self._config.perf_log_enabled = enabled
            return
        self.request_perf_log_enabled.emit(enabled)

    def update_train_opponents(self, pool: list[str]) -> None:
        if self._runner is None:
            self._config.train_opponents = pool
            return
        self.request_train_opponents.emit(pool)

    def update_eval_opponents(self, pool: list[str]) -> None:
        if self._runner is None:
            self._config.eval_opponents = pool
            return
        self.request_eval_opponents.emit(pool)

    def update_holdout_seeds(self, seeds: list[int]) -> None:
        if self._runner is None:
            self._config.holdout_seeds = seeds
            return
        self.request_holdout_seeds.emit(seeds)

    def apply_preset(self, preset: TrainingPreset) -> None:
        values = preset_to_values(preset)
        self.update_preset_id(str(values["preset_id"]))
        self.update_hands_per_tick(int(values["hands_per_tick"]))
        self.update_updates_per_sec(float(values["updates_per_sec"]))
        target = values["hands_per_second_target"]
        self.update_hands_per_second_target(float(target) if target else 0.0)
        self.update_fast_mode(bool(values["fast_mode"]))
        self.update_seed_mode(str(values["train_seed_mode"]))
        self.update_train_opponents(list(values["train_opponents"]))
        self.update_eval_opponents(list(values["eval_opponents"]))
        self.update_eval_interval(int(values["eval_interval"]))
        self.update_eval_hands_per_seed(int(values["eval_hands_per_seed"]))
        self.update_eval_seeds(list(values["eval_seeds"]))
        self.update_holdout_seeds(list(values["holdout_seeds"]))

    def _clear_runner(self) -> None:
        self._runner = None
        self._thread = None
        self._running = False
        self._control = None
        self._current_run_id = None
        self.status_updated.emit("idle")

    def stop_and_wait(self, timeout_ms: int = 2000) -> bool:
        if self._runner and self._thread and self._thread.isRunning():
            if self._control:
                self._control.request_stop()
            self._thread.wait(timeout_ms)
        return not (self._thread and self._thread.isRunning())

    def is_running(self) -> bool:
        return self._running or (self._thread is not None and self._thread.isRunning())

    def current_run_id(self) -> str | None:
        return self._current_run_id

    def _set_current_run(self, run_id: str) -> None:
        self._current_run_id = run_id

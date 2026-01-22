from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hearts_ai.bots.heuristic_bot import HeuristicBot
from hearts_ai.bots.random_bot import RandomBot
from hearts_ai.bots.safe_bot import SafeBot
from hearts_ai.core.sim import simulate_hand
from hearts_ai.training.control import TrainControl
from hearts_ai.training.eval import evaluate
from hearts_ai.training.metrics import EvalSnapshot, HoldoutSnapshot, MetricsSnapshot
from hearts_ai.training.optimizers.random_search import RandomSearchOptimizer
from hearts_ai.training.params import ParameterSet
from hearts_ai.util.rng import RNG, create_rng
from hearts_ai.util.version import get_version_string

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    hands_per_tick: int = 200
    seed: int | None = None
    seed_mode: str = "cycle"
    updates_per_sec: float = 5.0
    eval_interval: int = 5
    eval_hands_per_seed: int = 200
    eval_min_delta: float = 0.05
    eval_interval_seconds: float | None = None
    eval_seeds: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    holdout_seeds: list[int] = field(default_factory=lambda: [101, 102, 103, 104, 105])
    holdout_regression_threshold: float = 0.02
    train_opponents: list[str] = field(
        default_factory=lambda: ["RandomBot", "SafeBot", "BestSnapshotBot"]
    )
    eval_opponents: list[str] = field(
        default_factory=lambda: ["SafeBot", "BestSnapshotBot"]
    )
    hands_per_second_target: float | None = None
    fast_mode: bool = False
    debug_explain_enabled: bool = False
    log_every_n_hands: int = 100
    capture_sample_enabled: bool = False
    capture_sample_every: int = 5000
    preset_id: str | None = None
    autopolicy_enabled: bool = False
    autopolicy_decisions: list[dict] = field(default_factory=list)
    run_length_mode: str | None = None
    run_length_seconds: int | None = None
    stop_reason: str | None = None
    stop_requested_at: str | None = None
    stop_completed_at: str | None = None
    simple_max_tick_seconds: float | None = None
    perf_log_enabled: bool = False
    perf_log_interval_seconds: float = 30.0


class Trainer:
    def __init__(self, params: ParameterSet, config: TrainingConfig, control: TrainControl | None = None) -> None:
        self._params = params
        self._config = config
        self._control = control or TrainControl()
        self._iteration = 0
        self._best_train_score = float("inf")
        self._best_eval_score: float | None = None
        self._best_eval_se: float | None = None
        self._best_eval_params: ParameterSet | None = None
        self._rng: RNG = create_rng(config.seed)
        self._optimizer = RandomSearchOptimizer(self._rng)
        self._best_holdout: float | None = None
        self._overfit_level = "LOW"
        self._hands_done = 0
        self._last_eval_snapshot: EvalSnapshot | None = None
        self._global_hand_index = 0
        self._capture_next_requested = False
        self._run_dir: Path | None = None
        self._last_train_tick_s = 0.0
        self._last_eval_s = 0.0
        self._last_holdout_s = 0.0
        self._last_train_hands_per_s = 0.0
        self._last_eval_hands_per_s = 0.0
        self._last_eval_time: float | None = None
        self._train_time_accum = 0.0
        self._eval_time_accum = 0.0
        self._holdout_time_accum = 0.0
        self._last_perf_log_time: float | None = None
        self._train_ticks = 0

    def stop(self) -> None:
        self._control.request_stop()

    def pause(self) -> None:
        self._control.request_pause(True)

    def resume(self) -> None:
        self._control.request_pause(False)

    def reset(self) -> None:
        self._iteration = 0
        self._best_train_score = float("inf")
        self._best_eval_score = None
        self._best_eval_se = None
        self._best_eval_params = None

    def update_params(self, params: ParameterSet) -> None:
        self._params = params

    def request_capture_next(self) -> None:
        self._capture_next_requested = True

    @property
    def hands_done(self) -> int:
        return self._hands_done

    def run(
        self,
        on_metrics,
        on_best,
        on_eval=None,
        on_eval_status=None,
        on_overfit=None,
        on_holdout=None,
        on_run_started=None,
        on_phase=None,
    ) -> None:
        logger.info("Trainer started")
        run_start = time.perf_counter()
        run_dir = self._init_run_dir()
        self._run_dir = run_dir
        if on_run_started:
            on_run_started(run_dir.name)
        if on_phase:
            on_phase("training")
        # Write a baseline params file so latest.json always exists even before first eval.
        if not (run_dir / "params_best.json").exists():
            self._write_json(run_dir / "params_best.json", self._params.to_json())
            self._write_latest(run_dir)
        if self._config.eval_interval_seconds:
            self._last_eval_time = time.perf_counter()
        self._last_perf_log_time = time.perf_counter()
        train_log = (run_dir / "metrics_train.jsonl").open("a", encoding="utf-8")
        eval_log = (run_dir / "metrics_eval.jsonl").open("a", encoding="utf-8")
        if on_overfit:
            on_overfit(self._overfit_level)
        while not self._control.check_stop():
            self._control.wait_if_paused()

            tick_start = time.perf_counter()
            self._iteration += 1

            base_params = self._best_eval_params_or_current()
            candidate = self._optimizer.mutate(base_params)
            metrics = self._evaluate(candidate)
            elapsed = time.perf_counter() - tick_start
            self._last_train_tick_s = elapsed
            if elapsed > 0:
                self._last_train_hands_per_s = self._config.hands_per_tick / elapsed
            self._train_time_accum += elapsed
            self._train_ticks += 1
            self._hands_done += self._config.hands_per_tick

            if metrics.mean_penalty < self._best_train_score:
                self._best_train_score = metrics.mean_penalty

            if not self._config.fast_mode or self._iteration % 2 == 0:
                on_metrics(metrics)
            self._append_jsonl(train_log, metrics.__dict__)

            eval_due = False
            if self._config.eval_interval_seconds:
                now = time.perf_counter()
                if self._last_eval_time is None:
                    self._last_eval_time = now
                eval_due = (now - self._last_eval_time) >= self._config.eval_interval_seconds
            else:
                eval_due = self._iteration % self._config.eval_interval == 0

            if on_eval and eval_due:
                if on_phase:
                    on_phase("evaluating")
                logger.info(
                    "Eval start seeds=%d hands_per_seed=%d",
                    len(self._config.eval_seeds),
                    self._config.eval_hands_per_seed,
                )
                eval_start = time.perf_counter()
                eval_metrics = evaluate(
                    candidate,
                    self._best_eval_params_or_current(),
                    self._config.eval_seeds,
                    self._config.eval_hands_per_seed,
                    self._config.eval_opponents,
                    on_progress=on_eval_status,
                    control=self._control,
                )
                eval_elapsed = time.perf_counter() - eval_start
                self._last_eval_time = time.perf_counter()
                self._last_eval_s = eval_elapsed
                total_eval_hands = eval_metrics.seeds_count * eval_metrics.hands_per_seed
                self._last_eval_hands_per_s = (
                    total_eval_hands / eval_elapsed if eval_elapsed > 0 else 0.0
                )
                self._eval_time_accum += eval_elapsed
                logger.info(
                    "Eval done duration=%.2fs mean=%.3f ±%.3f",
                    eval_elapsed,
                    eval_metrics.mean_penalty,
                    eval_metrics.mean_penalty_se,
                )
                eval_snapshot = EvalSnapshot(
                    iteration=self._iteration,
                    mean_penalty=eval_metrics.mean_penalty,
                    mean_penalty_se=eval_metrics.mean_penalty_se,
                    win_rate=eval_metrics.win_rate,
                    qs_rate=eval_metrics.qs_rate,
                    hearts_rate=eval_metrics.hearts_rate,
                    moon_conceded_rate=eval_metrics.moon_conceded_rate,
                    seeds_count=eval_metrics.seeds_count,
                    hands_per_seed=eval_metrics.hands_per_seed,
                    opponent_breakdown=eval_metrics.opponent_breakdown,
                )
                on_eval(eval_snapshot)
                self._append_jsonl(eval_log, eval_snapshot.__dict__)
                self._last_eval_snapshot = eval_snapshot
                if self._is_new_best_eval(eval_snapshot):
                    self._best_eval_score = eval_snapshot.mean_penalty
                    self._best_eval_se = eval_snapshot.mean_penalty_se
                    self._best_eval_params = candidate.copy()
                    on_best(self._best_eval_params)
                    self._write_json(run_dir / "params_best.json", self._best_eval_params.to_json())
                    self._write_latest(run_dir)
                    logger.info(
                        "New best eval: %.3f ±%.3f",
                        eval_snapshot.mean_penalty,
                        eval_snapshot.mean_penalty_se,
                    )
                    def _holdout_status(message: str) -> None:
                        if on_eval_status:
                            on_eval_status(f"holdout {message}")

                    if on_phase:
                        on_phase("holdout")
                    logger.info(
                        "Holdout start seeds=%d hands_per_seed=%d",
                        len(self._config.holdout_seeds),
                        max(1, min(50, self._config.eval_hands_per_seed // 4)),
                    )
                    holdout = self._run_holdout_eval(_holdout_status if on_eval_status else None)
                    if holdout is not None:
                        logger.info(
                            "Holdout done duration=%.2fs mean=%.3f ±%.3f",
                            self._last_holdout_s,
                            holdout.mean_penalty,
                            holdout.mean_penalty_se,
                        )
                        self._holdout_time_accum += self._last_holdout_s
                        self._update_overfit(holdout.mean_penalty, on_overfit)
                        if on_holdout:
                            on_holdout(
                                HoldoutSnapshot(
                                    mean_penalty=holdout.mean_penalty,
                                    mean_penalty_se=holdout.mean_penalty_se,
                                    seeds_count=holdout.seeds_count,
                                    hands_per_seed=holdout.hands_per_seed,
                                )
                            )
                    if on_phase:
                        on_phase("evaluating")
                if on_phase:
                    on_phase("training")

            if self._config.log_every_n_hands > 0:
                if self._hands_done % self._config.log_every_n_hands == 0:
                    hands_sec = (
                        self._config.hands_per_tick / max(elapsed, 1e-6) if elapsed > 0 else 0.0
                    )
                    eval_mean = self._last_eval_snapshot.mean_penalty if self._last_eval_snapshot else 0.0
                    logger.info(
                        "hands=%d hands_sec=%.1f best_eval=%.3f train=%.3f eval=%.3f win=%.3f qs=%.3f",
                        self._hands_done,
                        hands_sec,
                        self._best_eval_score or self._best_train_score,
                        metrics.mean_penalty,
                        eval_mean,
                        metrics.win_rate,
                        metrics.qs_rate,
                    )

            if eval_due:
                self._params = self._best_eval_params_or_current()
            else:
                self._params = base_params

            interval = 1.0 / max(1.0, self._effective_updates_per_sec())
            if elapsed < interval:
                time.sleep(interval - elapsed)
            self._adapt_hands_per_tick(elapsed)
            self._maybe_log_perf()

        train_log.close()
        eval_log.close()
        if on_phase:
            on_phase("idle")
        if not self._config.stop_completed_at:
            self._config.stop_completed_at = _format_elapsed(time.perf_counter() - run_start)
        self._update_run_meta_end(run_dir)
        logger.info("Trainer stopped")

    def _evaluate(self, params: ParameterSet) -> MetricsSnapshot:
        total_penalty = 0
        wins = 0
        qs_taken = 0
        hands_count = 0

        for hand_index in range(self._config.hands_per_tick):
            if self._control.check_stop():
                break
            self._control.wait_if_paused()
            hand_seed = (self._config.seed or 0) + self._global_hand_index
            hand_rng = create_rng(hand_seed)
            bots = self._build_bots(
                params,
                self._best_eval_params_or_current(),
                self._config.train_opponents,
                hand_rng,
            )
            result = simulate_hand(bots, hand_rng, hand_index)
            total_penalty += result.points[0]
            if result.points[0] == min(result.points):
                wins += 1
            if result.qs_taken[0]:
                qs_taken += 1
            if self._should_capture_sample(self._global_hand_index):
                self._capture_sample(self._global_hand_index, hand_seed)
            self._global_hand_index += 1
            hands_count += 1

        divisor = max(1, hands_count)
        mean_penalty = total_penalty / divisor
        win_rate = wins / divisor
        qs_rate = qs_taken / divisor

        return MetricsSnapshot(
            iteration=self._iteration,
            mean_penalty=mean_penalty,
            win_rate=win_rate,
            qs_rate=qs_rate,
            best_score=self._best_eval_score or self._best_train_score,
            hands_done=self._hands_done + hands_count,
            hand_index=self._global_hand_index,
            best_train_mean=self._best_train_score,
            best_eval_mean=self._best_eval_score or 0.0,
            phase="train",
            last_train_tick_s=self._last_train_tick_s,
            last_eval_s=self._last_eval_s,
            last_holdout_s=self._last_holdout_s,
            train_hands_per_s=self._last_train_hands_per_s,
            eval_hands_per_s=self._last_eval_hands_per_s,
            hands_per_tick=self._config.hands_per_tick,
        )

    def run_eval_now(self, on_eval, on_eval_status=None) -> None:
        if on_eval_status:
            on_eval_status("evaluating")
        eval_metrics = evaluate(
            self._params,
            self._best_eval_params_or_current(),
            self._config.eval_seeds,
            self._config.eval_hands_per_seed,
            self._config.eval_opponents,
            on_progress=on_eval_status,
            control=self._control,
        )
        eval_snapshot = EvalSnapshot(
            iteration=self._iteration,
            mean_penalty=eval_metrics.mean_penalty,
            mean_penalty_se=eval_metrics.mean_penalty_se,
            win_rate=eval_metrics.win_rate,
            qs_rate=eval_metrics.qs_rate,
            hearts_rate=eval_metrics.hearts_rate,
            moon_conceded_rate=eval_metrics.moon_conceded_rate,
            seeds_count=eval_metrics.seeds_count,
            hands_per_seed=eval_metrics.hands_per_seed,
            opponent_breakdown=eval_metrics.opponent_breakdown,
        )
        on_eval(eval_snapshot)

    def _init_run_dir(self) -> Path:
        runs_dir = Path.cwd() / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": self._config.seed,
            "seed_mode": self._config.seed_mode,
            "start_time": timestamp,
            "hands_per_tick": self._config.hands_per_tick,
            "updates_per_sec": self._config.updates_per_sec,
            "eval_interval": self._config.eval_interval,
            "eval_interval_seconds": self._config.eval_interval_seconds,
            "eval_hands_per_seed": self._config.eval_hands_per_seed,
            "eval_seeds": self._config.eval_seeds,
            "holdout_seeds": self._config.holdout_seeds,
            "train_opponents": self._config.train_opponents,
            "eval_opponents": self._config.eval_opponents,
            "hands_per_second_target": self._config.hands_per_second_target,
            "fast_mode": self._config.fast_mode,
            "preset_id": self._config.preset_id,
            "autopolicy_enabled": self._config.autopolicy_enabled,
            "autopolicy_decisions": list(self._config.autopolicy_decisions),
            "run_length_mode": self._config.run_length_mode,
            "run_length_seconds": self._config.run_length_seconds,
            "stop_reason": self._config.stop_reason,
            "stop_requested_at": self._config.stop_requested_at,
            "stop_completed_at": self._config.stop_completed_at,
            "simple_max_tick_seconds": self._config.simple_max_tick_seconds,
            "perf_log_enabled": self._config.perf_log_enabled,
            "perf_log_interval_seconds": self._config.perf_log_interval_seconds,
            "version": get_version_string(),
            "params_snapshot": json.loads(self._params.to_json()),
            "start_hand_index": self._global_hand_index,
        }
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return run_dir

    def _build_bots(
        self,
        params: ParameterSet,
        best_params: ParameterSet,
        pool: list[str],
        rng: RNG,
    ):
        bots = []
        explain = False
        for name in pool:
            if name == "RandomBot":
                bots.append(RandomBot(rng))
            elif name == "SafeBot":
                bots.append(SafeBot(rng))
            elif name == "BestSnapshotBot":
                bots.append(HeuristicBot(best_params.copy(), rng, explain_enabled=explain))
        while len(bots) < 3:
            bots.append(RandomBot(rng))
        our_bot = HeuristicBot(params.copy(), rng, explain_enabled=explain)
        return [our_bot] + bots[:3]

    def _should_capture_sample(self, hand_index: int) -> bool:
        if self._config.fast_mode:
            if self._capture_next_requested:
                self._capture_next_requested = False
            return False
        if self._capture_next_requested:
            return True
        if not self._config.capture_sample_enabled:
            return False
        every = max(1, self._config.capture_sample_every)
        return hand_index % every == 0

    def _capture_sample(self, hand_index: int, hand_seed: int) -> None:
        from hearts_ai.training.trace import generate_trace

        if not self._run_dir:
            return
        if self._capture_next_requested:
            self._capture_next_requested = False
        samples_dir = self._run_dir / "samples"
        output_path = samples_dir / f"sample_{hand_index:06d}.json"
        try:
            generate_trace(self._params.copy(), hand_seed, hand_index, output_path)
            logger.info("Captured sample trace %s", output_path.name)
        except Exception:
            logger.exception("Failed to capture trace sample")

    def _run_holdout_eval(self, on_eval_status=None):
        if not self._config.holdout_seeds:
            return None
        holdout_start = time.perf_counter()
        metrics = evaluate(
            self._best_eval_params_or_current(),
            self._best_eval_params_or_current(),
            self._config.holdout_seeds,
            max(1, min(50, self._config.eval_hands_per_seed // 4)),
            self._config.eval_opponents,
            on_progress=on_eval_status,
            control=self._control,
        )
        self._last_holdout_s = time.perf_counter() - holdout_start
        return metrics

    def _update_overfit(self, holdout_mean: float, on_overfit) -> None:
        if self._best_holdout is None:
            self._best_holdout = holdout_mean
            if on_overfit:
                on_overfit(self._overfit_level)
            return
        if holdout_mean > self._best_holdout * (1.0 + self._config.holdout_regression_threshold):
            self._overfit_level = "HIGH"
            logger.warning("possible overfit/regression")
        else:
            self._overfit_level = "LOW"
            self._best_holdout = min(self._best_holdout, holdout_mean)
        if on_overfit:
            on_overfit(self._overfit_level)

    def _effective_updates_per_sec(self) -> float:
        if self._config.fast_mode:
            return max(self._config.updates_per_sec, 10.0)
        return self._config.updates_per_sec

    def _adapt_hands_per_tick(self, elapsed: float) -> None:
        if self._config.hands_per_second_target and elapsed > 0:
            measured = self._config.hands_per_tick / elapsed
            if measured > 0:
                scale = self._config.hands_per_second_target / measured
                scale = max(0.75, min(1.25, scale))
                new_value = int(self._config.hands_per_tick * scale)
                desired = int(
                    self._config.hands_per_second_target / max(1.0, self._effective_updates_per_sec())
                )
                lerp_value = int(
                    self._config.hands_per_tick + 0.2 * (desired - self._config.hands_per_tick)
                )
                blended = int((new_value + lerp_value) / 2)
                self._config.hands_per_tick = max(10, min(5000, blended))
        self._apply_simple_tick_cap()

    def _apply_simple_tick_cap(self) -> None:
        if not self._config.simple_max_tick_seconds:
            return
        if self._last_train_hands_per_s <= 0:
            return
        max_hands = int(self._last_train_hands_per_s * self._config.simple_max_tick_seconds)
        max_hands = max(50, max_hands)
        if self._config.hands_per_tick > max_hands:
            self._config.hands_per_tick = max_hands

    def _maybe_log_perf(self) -> None:
        if not self._config.perf_log_enabled:
            return
        if self._last_perf_log_time is None:
            self._last_perf_log_time = time.perf_counter()
            return
        now = time.perf_counter()
        if now - self._last_perf_log_time < self._config.perf_log_interval_seconds:
            return
        total = self._train_time_accum + self._eval_time_accum + self._holdout_time_accum
        if total <= 0:
            return
        train_pct = (self._train_time_accum / total) * 100.0
        eval_pct = (self._eval_time_accum / total) * 100.0
        hold_pct = (self._holdout_time_accum / total) * 100.0
        avg_tick = self._train_time_accum / max(1, self._train_ticks)
        logger.info(
            "Perf breakdown train=%.0f%% eval=%.0f%% holdout=%.0f%% avg_tick=%.3fs hands_per_tick=%d",
            train_pct,
            eval_pct,
            hold_pct,
            avg_tick,
            self._config.hands_per_tick,
        )
        self._last_perf_log_time = now

    def _append_jsonl(self, handle, payload: dict) -> None:
        handle.write(json.dumps(payload) + "\n")
        handle.flush()

    def _write_json(self, path: Path, payload: str) -> None:
        path.write_text(payload, encoding="utf-8")

    def _write_latest(self, run_dir: Path) -> None:
        runs_dir = Path.cwd() / "runs"
        latest = {
            "run_id": run_dir.name,
            "path": str(run_dir / "params_best.json"),
        }
        (runs_dir / "latest.json").write_text(json.dumps(latest, indent=2), encoding="utf-8")

    def _update_run_meta_end(self, run_dir: Path) -> None:
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        meta["final_hand_index"] = self._global_hand_index
        meta["autopolicy_decisions"] = list(self._config.autopolicy_decisions)
        meta["run_length_mode"] = self._config.run_length_mode
        meta["run_length_seconds"] = self._config.run_length_seconds
        meta["stop_reason"] = self._config.stop_reason
        meta["stop_requested_at"] = self._config.stop_requested_at
        meta["stop_completed_at"] = self._config.stop_completed_at
        meta["best_train_mean"] = self._best_train_score
        meta["best_eval_mean"] = self._best_eval_score
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _best_eval_params_or_current(self) -> ParameterSet:
        if self._best_eval_params is not None:
            return self._best_eval_params
        return self._params

    def _is_new_best_eval(self, snapshot: EvalSnapshot) -> bool:
        if self._best_eval_score is None:
            return True
        delta = self._best_eval_score - snapshot.mean_penalty
        if delta >= self._config.eval_min_delta:
            return True
        se = snapshot.mean_penalty_se
        if se > 0 and delta > 2.0 * se:
            return True
        return False


def _format_elapsed(elapsed: float) -> str:
    total = max(0, int(elapsed))
    minutes = total // 60
    seconds = total % 60
    return f"{minutes:02d}:{seconds:02d}"

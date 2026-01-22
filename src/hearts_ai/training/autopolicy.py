from __future__ import annotations

from dataclasses import dataclass, field

from hearts_ai.training.metrics import EvalSnapshot, HoldoutSnapshot, MetricsSnapshot


@dataclass
class PolicyAction:
    action: str
    value: object
    reason: str


@dataclass
class PolicyResult:
    actions: list[PolicyAction] = field(default_factory=list)
    suggestion: str | None = None
    note: str | None = None


class AutoPolicy:
    def __init__(
        self,
        plateau_window: int = 5,
        plateau_min_improvement: float = 0.05,
        overfit_gap: float = 0.3,
        noisy_se_threshold: float = 0.35,
        cooldown_evals: int = 3,
    ) -> None:
        self._eval_history: list[EvalSnapshot] = []
        self._holdout: HoldoutSnapshot | None = None
        self._plateau_window = plateau_window
        self._plateau_min_improvement = plateau_min_improvement
        self._overfit_gap = overfit_gap
        self._noisy_se_threshold = noisy_se_threshold
        self._cooldown_evals = cooldown_evals
        self._cooldown_remaining = 0
        self._consec_noisy = 0
        self._run_length_mode = "10 min"
        self._allow_seed_preset_changes = True

    def set_run_length_mode(self, mode: str) -> None:
        self._run_length_mode = mode

    def set_allow_seed_preset_changes(self, enabled: bool) -> None:
        self._allow_seed_preset_changes = enabled

    def update(
        self,
        eval_snapshot: EvalSnapshot | None,
        holdout_snapshot: HoldoutSnapshot | None,
        metrics_snapshot: MetricsSnapshot | None,
    ) -> PolicyResult:
        result = PolicyResult()
        if holdout_snapshot:
            self._holdout = holdout_snapshot
        if eval_snapshot:
            self._eval_history.append(eval_snapshot)

        if eval_snapshot and holdout_snapshot:
            gap = holdout_snapshot.mean_penalty - eval_snapshot.mean_penalty
            if gap > self._overfit_gap or holdout_snapshot.mean_penalty > eval_snapshot.mean_penalty * 1.02:
                result.actions.extend(
                    [
                        PolicyAction("eval_seed_preset", "standard", "overfit risk rising"),
                        PolicyAction("eval_hands_per_seed", max(300, eval_snapshot.hands_per_seed), "reduce noise"),
                        PolicyAction("seed_mode", "random", "diversify deals"),
                        PolicyAction(
                            "eval_opponents",
                            ["SafeBot", "BestSnapshotBot"],
                            "tighten evaluation pool",
                        ),
                    ]
                )
                result.note = "Overfit rising → tightened evaluation settings."

        if eval_snapshot:
            total_eval_hands = eval_snapshot.seeds_count * eval_snapshot.hands_per_seed
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
            if total_eval_hands >= 1000 and self._cooldown_remaining == 0:
                noisy = eval_snapshot.mean_penalty_se > self._noisy_se_threshold
                if noisy:
                    self._consec_noisy += 1
                else:
                    self._consec_noisy = 0
                if self._consec_noisy >= 2 and self._cooldown_remaining == 0:
                    cap = _cap_for_run_length(self._run_length_mode)
                    if eval_snapshot.seeds_count < 10 and self._allow_seed_preset_changes:
                        result.actions.append(
                            PolicyAction("eval_seed_preset", "standard", "eval too noisy")
                        )
                    else:
                        new_value = min(
                            cap,
                            max(
                                eval_snapshot.hands_per_seed + 100,
                                int(eval_snapshot.hands_per_seed * 1.25),
                            ),
                        )
                        if new_value > eval_snapshot.hands_per_seed:
                            result.actions.append(
                                PolicyAction("eval_hands_per_seed", new_value, "eval too noisy")
                            )
                    self._cooldown_remaining = self._cooldown_evals
                    self._consec_noisy = 0
                    result.note = "Eval noise high → adjusted eval cost."

        if len(self._eval_history) >= self._plateau_window:
            recent = self._eval_history[-self._plateau_window :]
            best = min(s.mean_penalty for s in recent)
            first = recent[0].mean_penalty
            if first - best < self._plateau_min_improvement:
                result.suggestion = "Plateau detected, consider stopping."

        if metrics_snapshot and metrics_snapshot.iteration < 10:
            result.actions.append(
                PolicyAction("train_opponents", ["SafeBot"], "early training schedule")
            )
        elif metrics_snapshot and metrics_snapshot.iteration < 30:
            result.actions.append(
                PolicyAction("train_opponents", ["SafeBot", "RandomBot"], "mid training schedule")
            )
        elif metrics_snapshot:
            result.actions.append(
                PolicyAction(
                    "train_opponents",
                    ["SafeBot", "BestSnapshotBot"],
                    "late training schedule",
                )
            )

        return result


def _cap_for_run_length(mode: str) -> int:
    if mode == "2 min":
        return 300
    if mode == "30 min":
        return 600
    if mode == "10 min":
        return 400
    return 400

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricsSnapshot:
    iteration: int
    mean_penalty: float
    win_rate: float
    qs_rate: float
    best_score: float
    hands_done: int
    hand_index: int
    best_train_mean: float = 0.0
    best_eval_mean: float = 0.0
    phase: str = "train"
    last_train_tick_s: float = 0.0
    last_eval_s: float = 0.0
    last_holdout_s: float = 0.0
    train_hands_per_s: float = 0.0
    eval_hands_per_s: float = 0.0
    hands_per_tick: int = 0


@dataclass
class EvalSnapshot:
    iteration: int
    mean_penalty: float
    mean_penalty_se: float
    win_rate: float
    qs_rate: float
    hearts_rate: float
    moon_conceded_rate: float
    seeds_count: int
    hands_per_seed: int
    opponent_breakdown: dict


@dataclass
class HoldoutSnapshot:
    mean_penalty: float
    mean_penalty_se: float
    seeds_count: int
    hands_per_seed: int


def format_optional_points(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"

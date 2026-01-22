from __future__ import annotations

from dataclasses import dataclass
import statistics
import time

from hearts_ai.bots.heuristic_bot import HeuristicBot
from hearts_ai.bots.random_bot import RandomBot
from hearts_ai.bots.safe_bot import SafeBot
from hearts_ai.core.sim import simulate_hand
from hearts_ai.training.params import ParameterSet
from hearts_ai.util.rng import RNG, create_rng


@dataclass
class EvalMetrics:
    mean_penalty: float
    mean_penalty_se: float
    win_rate: float
    qs_rate: float
    hearts_rate: float
    moon_conceded_rate: float
    seeds_count: int
    hands_per_seed: int
    opponent_breakdown: dict


def evaluate(
    params: ParameterSet,
    best_params: ParameterSet,
    seeds: list[int],
    hands_per_seed: int,
    opponents: list[str],
    on_progress=None,
    control=None,
) -> EvalMetrics:
    total_penalty = 0
    wins = 0
    qs_taken = 0
    hearts_taken = 0
    moon_conceded = 0
    total_hands = 0
    seed_means: list[float] = []
    opponent_stats: dict[str, dict[str, float]] = {}
    total_eval_hands = len(seeds) * hands_per_seed
    last_progress_time = 0.0
    progress_every_hands = 50
    progress_min_interval = 0.2

    total_seeds = len(seeds)
    for idx, seed in enumerate(seeds, start=1):
        if control and control.check_stop():
            break
        if control:
            control.wait_if_paused()
        rng: RNG = create_rng(seed)
        if on_progress:
            on_progress(f"seed {idx}/{total_seeds} hand 0/{hands_per_seed} total 0/{total_eval_hands}")
        per_seed_best = best_params.copy()

        seed_penalty = 0
        for hand_index in range(hands_per_seed):
            if control and control.check_stop():
                break
            if control:
                control.wait_if_paused()
            bots, types, my_idx = _build_bots(params, per_seed_best, opponents, rng, hand_index)
            if len(bots) != 4:
                raise ValueError(f"Expected 4 bots, got {len(bots)}")
            result = simulate_hand(bots, rng, hand_index)
            total_hands += 1
            total_penalty += result.points[my_idx]
            seed_penalty += result.points[my_idx]
            if result.points[my_idx] == min(result.points):
                wins += 1
            if result.qs_taken[my_idx]:
                qs_taken += 1
            hearts_taken += result.hearts_taken[my_idx]
            if result.moon_shooter is not None and result.moon_shooter != my_idx:
                moon_conceded += 1

            min_points = min(result.points)
            for idx, bot_type in enumerate(types):
                if idx == my_idx:
                    continue
                stats = opponent_stats.setdefault(
                    bot_type,
                    {"points": 0.0, "wins": 0.0, "hands": 0.0},
                )
                stats["points"] += result.points[idx]
                stats["hands"] += 1
                if result.points[idx] == min_points:
                    stats["wins"] += 1

            if on_progress:
                should_emit = (hand_index + 1) % progress_every_hands == 0
                now = time.perf_counter()
                if should_emit and (now - last_progress_time) >= progress_min_interval:
                    on_progress(
                        f"seed {idx}/{total_seeds} hand {hand_index + 1}/{hands_per_seed} "
                        f"total {total_hands}/{total_eval_hands}"
                    )
                    last_progress_time = now

        if hands_per_seed > 0:
            seed_means.append(seed_penalty / hands_per_seed)

    if total_hands == 0:
        if on_progress:
            on_progress("idle")
        return EvalMetrics(0, 0, 0, 0, 0, 0, 0, 0, {})

    mean_penalty = total_penalty / total_hands
    if len(seed_means) > 1:
        mean_penalty_se = statistics.pstdev(seed_means) / (len(seed_means) ** 0.5)
    else:
        mean_penalty_se = 0.0

    breakdown = {}
    for name, stats in opponent_stats.items():
        if stats["hands"] <= 0:
            continue
        breakdown[name] = {
            "mean_penalty": stats["points"] / stats["hands"],
            "win_rate": stats["wins"] / stats["hands"],
        }

    if on_progress:
        on_progress("idle")
    return EvalMetrics(
        mean_penalty=mean_penalty,
        mean_penalty_se=mean_penalty_se,
        win_rate=wins / total_hands,
        qs_rate=qs_taken / total_hands,
        hearts_rate=hearts_taken / (13 * total_hands),
        moon_conceded_rate=moon_conceded / total_hands,
        seeds_count=len(seed_means),
        hands_per_seed=hands_per_seed,
        opponent_breakdown=breakdown,
    )


def _build_bots(
    params: ParameterSet,
    best_params: ParameterSet,
    opponents: list[str],
    rng: RNG,
    hand_index: int,
):
    factories = []
    names: list[str] = []
    for name in opponents:
        if name == "RandomBot":
            factories.append(lambda r=rng: RandomBot(r))
            names.append("RandomBot")
        elif name == "SafeBot":
            factories.append(lambda r=rng: SafeBot(r))
            names.append("SafeBot")
        elif name == "BestSnapshotBot":
            factories.append(
                lambda r=rng, p=best_params: HeuristicBot(params=p.copy(), rng=r, explain_enabled=False)
            )
            names.append("BestSnapshotBot")

    if not factories:
        factories = [lambda r=rng: RandomBot(r)]
        names = ["RandomBot"]

    opps = [(factories[i % len(factories)](), names[i % len(names)]) for i in range(3)]
    bots = [bot for bot, _ in opps]
    types = [name for _, name in opps]
    seat = hand_index % 4
    bots.insert(seat, HeuristicBot(params=params.copy(), rng=rng, explain_enabled=False))
    types.insert(seat, "HeuristicBot")
    return bots[:4], types[:4], seat

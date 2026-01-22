from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HOLDOUT_SEEDS = [101, 102, 103, 104, 105]


@dataclass(frozen=True)
class TrainingPreset:
    preset_id: str
    name: str
    description: str
    hands_per_tick: int
    updates_per_sec: float
    hands_per_second_target: float | None
    fast_mode: bool
    train_seed_mode: str
    train_opponents: list[str]
    eval_opponents: list[str]
    eval_interval: int
    eval_hands_per_seed: int
    eval_seed_preset: str
    holdout_enabled: bool = True


def get_presets() -> list[TrainingPreset]:
    return [
        TrainingPreset(
            preset_id="quick_boost",
            name="Quick Boost",
            description="Fast feedback with low-cost eval.",
            hands_per_tick=300,
            updates_per_sec=8.0,
            hands_per_second_target=2500.0,
            fast_mode=True,
            train_seed_mode="random",
            train_opponents=["SafeBot"],
            eval_opponents=["SafeBot", "BestSnapshotBot"],
            eval_interval=8,
            eval_hands_per_seed=100,
            eval_seed_preset="quick",
            holdout_enabled=False,
        ),
        TrainingPreset(
            preset_id="balanced",
            name="Balanced",
            description="Default balance of speed and reliability.",
            hands_per_tick=200,
            updates_per_sec=5.0,
            hands_per_second_target=2000.0,
            fast_mode=False,
            train_seed_mode="cycle",
            train_opponents=["RandomBot", "SafeBot", "BestSnapshotBot"],
            eval_opponents=["SafeBot", "BestSnapshotBot"],
            eval_interval=5,
            eval_hands_per_seed=200,
            eval_seed_preset="standard",
            holdout_enabled=True,
        ),
        TrainingPreset(
            preset_id="anti_overfit",
            name="Anti-Overfit",
            description="Stricter eval and holdout checks.",
            hands_per_tick=160,
            updates_per_sec=4.0,
            hands_per_second_target=1500.0,
            fast_mode=False,
            train_seed_mode="random",
            train_opponents=["SafeBot", "BestSnapshotBot"],
            eval_opponents=["SafeBot", "BestSnapshotBot"],
            eval_interval=3,
            eval_hands_per_seed=400,
            eval_seed_preset="standard",
            holdout_enabled=True,
        ),
        TrainingPreset(
            preset_id="thorough",
            name="Thorough",
            description="Slow but trustworthy eval metrics.",
            hands_per_tick=120,
            updates_per_sec=3.0,
            hands_per_second_target=1200.0,
            fast_mode=False,
            train_seed_mode="cycle",
            train_opponents=["SafeBot", "BestSnapshotBot"],
            eval_opponents=["SafeBot", "BestSnapshotBot"],
            eval_interval=2,
            eval_hands_per_seed=600,
            eval_seed_preset="thorough",
            holdout_enabled=True,
        ),
    ]


def seed_preset_to_list(preset: str) -> list[int]:
    if preset == "quick":
        return list(range(1, 6))
    if preset == "thorough":
        return list(range(1, 101))
    return list(range(1, 21))


def preset_to_values(preset: TrainingPreset) -> dict[str, object]:
    return {
        "preset_id": preset.preset_id,
        "hands_per_tick": preset.hands_per_tick,
        "updates_per_sec": preset.updates_per_sec,
        "hands_per_second_target": preset.hands_per_second_target,
        "fast_mode": preset.fast_mode,
        "train_seed_mode": preset.train_seed_mode,
        "train_opponents": list(preset.train_opponents),
        "eval_opponents": list(preset.eval_opponents),
        "eval_interval": preset.eval_interval,
        "eval_hands_per_seed": preset.eval_hands_per_seed,
        "eval_seeds": seed_preset_to_list(preset.eval_seed_preset),
        "holdout_seeds": list(DEFAULT_HOLDOUT_SEEDS) if preset.holdout_enabled else [],
    }


def max_eval_hands_per_seed(mode: str) -> int:
    if mode == "2 min":
        return 300
    if mode == "30 min":
        return 600
    if mode == "10 min":
        return 400
    return 400


def apply_eval_budget(
    seed_preset: str,
    hands_per_seed: int,
    run_length_mode: str,
) -> tuple[str, int, int]:
    max_per_seed = max_eval_hands_per_seed(run_length_mode)
    hands_per_seed = min(hands_per_seed, max_per_seed)
    seeds = seed_preset_to_list(seed_preset)
    total = len(seeds) * hands_per_seed
    return seed_preset, hands_per_seed, total

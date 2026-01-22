from __future__ import annotations

from dataclasses import dataclass

from hearts_ai.core.game import play_hand
from hearts_ai.util.rng import RNG, create_rng


@dataclass
class SimResult:
    points: list[int]
    qs_taken: list[bool]
    raw_points: list[int]
    hearts_taken: list[int]
    moon_shooter: int | None


def simulate_hand(bots, rng: RNG | None = None, hand_index: int = 0, seed: int | None = None) -> SimResult:
    rng = rng or create_rng(seed)
    result = play_hand(bots, rng, hand_index)
    return SimResult(
        points=result.points,
        qs_taken=result.qs_taken,
        raw_points=result.raw_points,
        hearts_taken=result.hearts_taken,
        moon_shooter=result.moon_shooter,
    )

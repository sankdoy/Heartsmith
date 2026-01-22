from __future__ import annotations

from hearts_ai.training.params import ParameterSet
from hearts_ai.util.rng import RNG, create_rng


class RandomSearchOptimizer:
    def __init__(self, rng: RNG | None = None) -> None:
        self._rng = rng or create_rng()

    def mutate(self, params: ParameterSet) -> ParameterSet:
        mutated = params.copy()
        candidates = [p for p in mutated.all() if not p.locked]
        if not candidates:
            return mutated
        k = self._rng.randint(3, min(8, len(candidates)))
        picks = self._rng.sample(candidates, k=k)
        for param in picks:
            span = param.max_value - param.min_value
            delta = self._rng.gauss(0.0, span * 0.05)
            param.value = max(param.min_value, min(param.max_value, param.value + delta))
        return mutated

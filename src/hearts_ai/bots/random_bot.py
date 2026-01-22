from __future__ import annotations

from hearts_ai.util.rng import RNG, create_rng

from hearts_ai.bots.base import Bot
from hearts_ai.core.cards import Card
from hearts_ai.core.state import TrickState


class RandomBot(Bot):
    def __init__(self, rng: RNG | None = None) -> None:
        self._rng = rng or create_rng()

    def choose_pass(self, hand: list[Card]) -> list[Card]:
        if len(hand) <= 3:
            return list(hand)
        return self._rng.sample(hand, 3)

    def choose_card(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
    ) -> Card:
        return self._rng.choice(legal_moves)

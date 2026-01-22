from __future__ import annotations

from abc import ABC, abstractmethod

from hearts_ai.core.cards import Card
from hearts_ai.core.state import TrickState


class Bot(ABC):
    @abstractmethod
    def choose_pass(self, hand: list[Card]) -> list[Card]:
        raise NotImplementedError

    @abstractmethod
    def choose_card(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
    ) -> Card:
        raise NotImplementedError

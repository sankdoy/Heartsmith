from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from hearts_ai.core.cards import Card, Suit


@dataclass
class TrickState:
    leader: int
    cards: list[tuple[int, Card]] = field(default_factory=list)

    @property
    def lead_suit(self) -> Suit | None:
        if not self.cards:
            return None
        return self.cards[0][1].suit

    def add_card(self, player_idx: int, card: Card) -> None:
        self.cards.append((player_idx, card))

    def is_complete(self) -> bool:
        return len(self.cards) == 4

    def winner(self) -> int:
        if not self.cards:
            raise ValueError("No cards in trick")
        lead = self.cards[0][1].suit
        lead_cards = [entry for entry in self.cards if entry[1].suit == lead]
        return max(lead_cards, key=lambda entry: entry[1].rank)[0]

    def points(self) -> int:
        from hearts_ai.core.scoring import card_points

        return sum(card_points(card) for _, card in self.cards)


@dataclass
class HandResult:
    points: list[int]
    qs_taken: list[bool]
    raw_points: list[int]
    hearts_taken: list[int]
    moon_shooter: int | None
    trick_history: list[TrickState] | None = None
    initial_hands: list[list[Card]] | None = None


@dataclass
class GameState:
    hands: list[list[Card]]
    hearts_broken: bool = False
    trick_index: int = 0
    leader: int = 0

    def hand_for(self, player_idx: int) -> list[Card]:
        return self.hands[player_idx]

    def remove_card(self, player_idx: int, card: Card) -> None:
        self.hands[player_idx].remove(card)

    def players_with_suit(self, suit: Suit) -> list[int]:
        players: list[int] = []
        for idx, hand in enumerate(self.hands):
            if any(card.suit == suit for card in hand):
                players.append(idx)
        return players

    def all_cards(self) -> Iterable[Card]:
        for hand in self.hands:
            for card in hand:
                yield card

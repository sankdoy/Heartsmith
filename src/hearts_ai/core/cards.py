from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from hearts_ai.util.rng import RNG, create_rng


class Suit(str, Enum):
    CLUBS = "C"
    DIAMONDS = "D"
    SPADES = "S"
    HEARTS = "H"


class Rank(int, Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.rank.name[0]}{self.suit.value}"

    def short(self) -> str:
        rank_map = {
            Rank.TEN: "T",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        r = rank_map.get(self.rank, str(int(self.rank)))
        return f"{r}{self.suit.value}"


class Deck:
    def __init__(self, rng: RNG | None = None) -> None:
        self._rng = rng or create_rng()
        self.cards = [Card(suit, rank) for suit in Suit for rank in Rank]

    def shuffle(self) -> None:
        self._rng.shuffle(self.cards)

    def deal(self, players: int = 4, cards_each: int = 13) -> list[list[Card]]:
        if players * cards_each > len(self.cards):
            raise ValueError("Not enough cards to deal")
        self.shuffle()
        hands = [[] for _ in range(players)]
        for idx, card in enumerate(self.cards[: players * cards_each]):
            hands[idx % players].append(card)
        for hand in hands:
            hand.sort(key=lambda c: (c.suit.value, c.rank))
        return hands


def find_card(hand: Iterable[Card], suit: Suit, rank: Rank) -> Card | None:
    for card in hand:
        if card.suit == suit and card.rank == rank:
            return card
    return None

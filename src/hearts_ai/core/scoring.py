from __future__ import annotations

from hearts_ai.core.cards import Card, Rank, Suit


def card_points(card: Card) -> int:
    if card.suit == Suit.HEARTS:
        return 1
    if card.suit == Suit.SPADES and card.rank == Rank.QUEEN:
        return 13
    return 0


def shoot_moon(points: list[int]) -> list[int]:
    if 26 in points and sum(points) == 26:
        moon_idx = points.index(26)
        return [0 if idx == moon_idx else 26 for idx in range(len(points))]
    return points

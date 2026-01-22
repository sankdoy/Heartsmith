from __future__ import annotations

from dataclasses import dataclass

from hearts_ai.bots.base import Bot
from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.scoring import card_points
from hearts_ai.core.state import TrickState
from hearts_ai.util.rng import RNG, create_rng


def _rank_value(card: Card) -> int:
    return int(card.rank)


@dataclass
class SafeBot(Bot):
    rng: RNG

    def __init__(self, rng: RNG | None = None) -> None:
        self.rng = rng or create_rng()
        self._qs_played = False
        self._last_hand_size = 0

    def choose_pass(self, hand: list[Card]) -> list[Card]:
        scores: list[tuple[Card, float]] = []
        suit_counts = _suit_counts(hand)
        for card in hand:
            score = 0.0
            if card.suit == Suit.SPADES and card.rank == Rank.QUEEN:
                score += 100.0
            elif card.suit == Suit.SPADES and card.rank == Rank.ACE:
                score += 90.0
            elif card.suit == Suit.SPADES and card.rank == Rank.KING:
                score += 80.0
            if card.suit == Suit.HEARTS:
                score += 50.0 + _rank_value(card) / 14.0 * 10.0
            elif card_points(card) == 0:
                score += _rank_value(card) / 14.0 * 5.0
            score += suit_counts[card.suit] * 0.5
            scores.append((card, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [card for card, _ in scores[:3]]

    def choose_card(
        self,
        hand: list[Card],
        legal_moves: list[Card],
        trick: TrickState,
        hearts_broken: bool,
        is_first_trick: bool,
    ) -> Card:
        self._maybe_update_hand(hand)
        if not legal_moves:
            raise ValueError("No legal moves")

        lead_suit = trick.lead_suit
        if lead_suit is None:
            chosen = self._lead_card(hand, legal_moves, hearts_broken)
            self.update_qs_seen(trick, chosen)
            return chosen

        follow = [card for card in legal_moves if card.suit == lead_suit]
        if follow:
            chosen = self._follow_card(trick, follow)
            self.update_qs_seen(trick, chosen)
            return chosen
        chosen = self._dump_card(legal_moves)
        self.update_qs_seen(trick, chosen)
        return chosen

    def _lead_card(self, hand: list[Card], legal_moves: list[Card], hearts_broken: bool) -> Card:
        non_hearts = [card for card in legal_moves if card.suit != Suit.HEARTS]
        if not hearts_broken and non_hearts:
            candidates = non_hearts
        else:
            candidates = legal_moves

        if not self._qs_played:
            non_spades = [card for card in candidates if card.suit != Suit.SPADES]
            if non_spades:
                candidates = non_spades

        return min(candidates, key=_rank_value)

    def _follow_card(self, trick: TrickState, follow: list[Card]) -> Card:
        lead_suit = trick.lead_suit
        highest = max((card.rank for _, card in trick.cards if card.suit == lead_suit), default=Rank.TWO)
        losing = [card for card in follow if card.rank < highest]
        if losing:
            return max(losing, key=_rank_value)
        return min(follow, key=_rank_value)

    def _dump_card(self, legal_moves: list[Card]) -> Card:
        for card in legal_moves:
            if card.suit == Suit.SPADES and card.rank == Rank.QUEEN:
                return card
        hearts = [card for card in legal_moves if card.suit == Suit.HEARTS]
        if hearts:
            return max(hearts, key=_rank_value)
        non_points = [card for card in legal_moves if card_points(card) == 0]
        if non_points:
            return max(non_points, key=_rank_value)
        return max(legal_moves, key=_rank_value)

    def _maybe_update_hand(self, hand: list[Card]) -> None:
        if len(hand) == 13 and self._last_hand_size != 13:
            self._qs_played = False
        self._last_hand_size = len(hand)

    def update_qs_seen(self, trick: TrickState, card: Card) -> None:
        if card.suit == Suit.SPADES and card.rank == Rank.QUEEN:
            self._qs_played = True
        if any(c.suit == Suit.SPADES and c.rank == Rank.QUEEN for _, c in trick.cards):
            self._qs_played = True


def _suit_counts(hand: list[Card]) -> dict[Suit, int]:
    counts = {Suit.CLUBS: 0, Suit.DIAMONDS: 0, Suit.SPADES: 0, Suit.HEARTS: 0}
    for card in hand:
        counts[card.suit] += 1
    return counts

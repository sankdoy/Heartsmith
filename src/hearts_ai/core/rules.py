from __future__ import annotations

from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.scoring import card_points
from hearts_ai.core.state import TrickState


def is_point_card(card: Card) -> bool:
    return card_points(card) > 0


def legal_moves(
    hand: list[Card],
    trick: TrickState,
    hearts_broken: bool,
    is_first_trick: bool,
) -> list[Card]:
    if not hand:
        return []

    lead_suit = trick.lead_suit
    if lead_suit is None:
        return _legal_lead(hand, hearts_broken, is_first_trick)

    follow = [card for card in hand if card.suit == lead_suit]
    if follow:
        return _filter_first_trick_points(follow, hand, is_first_trick)

    return _filter_first_trick_points(hand, hand, is_first_trick)


def _legal_lead(hand: list[Card], hearts_broken: bool, is_first_trick: bool) -> list[Card]:
    if is_first_trick:
        return [card for card in hand if card.suit == Suit.CLUBS and card.rank == Rank.TWO]

    if hearts_broken:
        return list(hand)

    non_hearts = [card for card in hand if card.suit != Suit.HEARTS]
    return non_hearts if non_hearts else list(hand)


def _filter_first_trick_points(
    candidates: list[Card],
    full_hand: list[Card],
    is_first_trick: bool,
) -> list[Card]:
    if not is_first_trick:
        return list(candidates)

    non_point_cards = [card for card in full_hand if not is_point_card(card)]
    if non_point_cards:
        return [card for card in candidates if not is_point_card(card)]

    return list(candidates)

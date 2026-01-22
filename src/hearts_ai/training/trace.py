from __future__ import annotations

import json
from pathlib import Path

from hearts_ai.bots.heuristic_bot import HeuristicBot
from hearts_ai.bots.safe_bot import SafeBot
from hearts_ai.core.cards import Card, Deck, Rank, Suit, find_card
from hearts_ai.core.passing import PassDirection, pass_direction
from hearts_ai.core.rules import legal_moves
from hearts_ai.core.scoring import card_points, shoot_moon
from hearts_ai.core.state import TrickState
from hearts_ai.training.params import ParameterSet
from hearts_ai.util.rng import RNG, create_rng


def generate_trace(
    params: ParameterSet,
    seed: int,
    hand_index: int,
    output_path: Path,
) -> None:
    rng: RNG = create_rng(seed)
    deck = Deck(rng)
    hands = deck.deal()
    direction = pass_direction(hand_index)
    bots = [HeuristicBot(params.copy(), rng, explain_enabled=False), SafeBot(rng), SafeBot(rng), SafeBot(rng)]

    if direction != PassDirection.KEEP:
        _apply_passing(hands, bots, direction)

    leader = _find_two_clubs_leader(hands)
    hearts_broken = False
    points = [0, 0, 0, 0]
    qs_taken = [False, False, False, False]
    trace = []

    for trick_index in range(13):
        trick = TrickState(leader=leader)
        for offset in range(4):
            player_idx = (leader + offset) % 4
            hand = hands[player_idx]
            legal = legal_moves(hand, trick, hearts_broken, trick_index == 0)
            if player_idx == 0:
                must_follow = trick.lead_suit is not None and any(
                    card.suit == trick.lead_suit for card in hand
                )
                info = bots[player_idx].choose_with_trace(
                    hand, legal, trick, hearts_broken, trick_index == 0, top_n=5
                )
                trace.append(
                    {
                        "trick_index": trick_index,
                        "lead_suit": trick.lead_suit.value if trick.lead_suit else None,
                        "hearts_broken": hearts_broken,
                        "hand_size": len(hand),
                        "hand": [card.short() for card in hand],
                        "must_follow": must_follow,
                        "points_on_table": _points_on_table_norm(trick.cards),
                        "trick_cards": [
                            {"player": player, "card": card.short()} for player, card in trick.cards
                        ],
                        "legal_moves": [card.short() for card in legal],
                        "chosen": info["chosen"],
                        "chosen_score": info.get("chosen_score"),
                        "candidates": [
                            {
                                "card": entry["card"],
                                "score": entry["score"],
                                "terms": entry["terms"],
                            }
                            for entry in info["candidates"]
                        ],
                    }
                )
                chosen_card = next(card for card in legal if card.short() == info["chosen"])
            else:
                chosen_card = bots[player_idx].choose_card(
                    hand, legal, trick, hearts_broken, trick_index == 0
                )
            trick.add_card(player_idx, chosen_card)
            hand.remove(chosen_card)
            if chosen_card.suit == Suit.HEARTS:
                hearts_broken = True
            if chosen_card.suit == Suit.SPADES and chosen_card.rank == Rank.QUEEN:
                qs_taken[player_idx] = True

        winner = trick.winner()
        points[winner] += trick.points()
        leader = winner

    raw_points = list(points)
    points = shoot_moon(points)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "hand_index": hand_index,
        "points": points,
        "raw_points": raw_points,
        "trace": trace,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_two_clubs_leader(hands: list[list[Card]]) -> int:
    for idx, hand in enumerate(hands):
        if find_card(hand, Suit.CLUBS, Rank.TWO):
            return idx
    return 0


def _apply_passing(hands: list[list[Card]], bots, direction: PassDirection) -> None:
    selections = []
    for idx, hand in enumerate(hands):
        picks = bots[idx].choose_pass(hand)
        for card in picks:
            hand.remove(card)
        selections.append(picks)

    for idx in range(4):
        if direction == PassDirection.LEFT:
            target = (idx + 1) % 4
        elif direction == PassDirection.RIGHT:
            target = (idx - 1) % 4
        elif direction == PassDirection.ACROSS:
            target = (idx + 2) % 4
        else:
            target = idx
        hands[target].extend(selections[idx])

    for hand in hands:
        hand.sort(key=lambda c: (c.suit.value, c.rank))


def _points_on_table_norm(cards) -> float:
    hearts = sum(1 for _, card in cards if card.suit == Suit.HEARTS)
    qs = any(card.suit == Suit.SPADES and card.rank == Rank.QUEEN for _, card in cards)
    return (hearts + (13 if qs else 0)) / 26.0

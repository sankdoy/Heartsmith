from __future__ import annotations

from dataclasses import dataclass

from hearts_ai.core.cards import Card, Deck, Rank, Suit, find_card
from hearts_ai.core.passing import PassDirection, pass_direction
from hearts_ai.core.rules import legal_moves
from hearts_ai.core.scoring import card_points, shoot_moon
from hearts_ai.core.state import HandResult, TrickState
from hearts_ai.util.rng import RNG, create_rng


@dataclass
class GameResult:
    hand_points: list[int]
    totals: list[int]


def play_hand(
    bots,
    rng: RNG | None = None,
    hand_index: int = 0,
    seed: int | None = None,
    capture_tricks: bool = False,
) -> HandResult:
    if len(bots) != 4:
        raise ValueError(f"Expected 4 bots, got {len(bots)}")
    rng = rng or create_rng(seed)
    deck = Deck(rng)
    hands = deck.deal()
    direction = pass_direction(hand_index)
    if direction != PassDirection.KEEP:
        _apply_passing(hands, bots, direction, rng)

    initial_hands = [hand.copy() for hand in hands] if capture_tricks else None

    leader = _find_two_clubs_leader(hands)
    hearts_broken = False
    points = [0, 0, 0, 0]
    hearts_taken = [0, 0, 0, 0]
    qs_taken = [False, False, False, False]

    trick_history = [] if capture_tricks else None
    for trick_index in range(13):
        trick = TrickState(leader=leader)
        for offset in range(4):
            player_idx = (leader + offset) % 4
            hand = hands[player_idx]
            legal = legal_moves(hand, trick, hearts_broken, trick_index == 0)
            card = bots[player_idx].choose_card(hand, legal, trick, hearts_broken, trick_index == 0)
            if card not in legal:
                card = legal[0]
            trick.add_card(player_idx, card)
            hand.remove(card)
            if card.suit == Suit.HEARTS:
                hearts_broken = True
            if card.suit == Suit.SPADES and card.rank == Rank.QUEEN:
                qs_taken[player_idx] = True

        winner = trick.winner()
        trick_points = trick.points()
        points[winner] += trick_points
        hearts_taken[winner] += sum(1 for _, card in trick.cards if card.suit == Suit.HEARTS)
        leader = winner
        if trick_history is not None:
            trick_history.append(trick)

    raw_points = list(points)
    moon_shooter = raw_points.index(26) if sum(raw_points) == 26 and 26 in raw_points else None
    points = shoot_moon(points)
    return HandResult(
        points=points,
        qs_taken=qs_taken,
        raw_points=raw_points,
        hearts_taken=hearts_taken,
        moon_shooter=moon_shooter,
        trick_history=trick_history,
        initial_hands=initial_hands,
    )


def play_game(
    bots,
    rng: RNG | None = None,
    target_score: int = 100,
    seed: int | None = None,
) -> GameResult:
    rng = rng or create_rng(seed)
    totals = [0, 0, 0, 0]
    hand_index = 0
    while True:
        hand = play_hand(bots, rng, hand_index)
        totals = [t + p for t, p in zip(totals, hand.points)]
        if should_end_game(totals, target_score):
            return GameResult(hand_points=hand.points, totals=totals)
        hand_index += 1


def should_end_game(totals: list[int], target_score: int = 100) -> bool:
    if max(totals) < target_score:
        return False
    min_score = min(totals)
    return totals.count(min_score) == 1


def _find_two_clubs_leader(hands: list[list[Card]]) -> int:
    for idx, hand in enumerate(hands):
        if find_card(hand, Suit.CLUBS, Rank.TWO):
            return idx
    return 0


def _apply_passing(hands: list[list[Card]], bots, direction: PassDirection, rng: RNG) -> None:
    selections = []
    for idx, hand in enumerate(hands):
        picks = bots[idx].choose_pass(hand)
        if len(picks) != 3:
            picks = rng.sample(hand, 3)
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

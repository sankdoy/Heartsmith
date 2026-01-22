import pytest

from hearts_ai.bots.random_bot import RandomBot
from hearts_ai.core.cards import Suit
from hearts_ai.core.game import play_hand
from hearts_ai.core.rules import legal_moves
from hearts_ai.core.scoring import shoot_moon
from hearts_ai.core.state import TrickState
from hearts_ai.util.rng import create_rng


@pytest.mark.slow
def test_invariants_soak():
    seeds = [1, 2, 3]
    hands_per_seed = 50
    for seed in seeds:
        rng = create_rng(seed)
        bots = [RandomBot(rng) for _ in range(4)]
        for hand_index in range(hands_per_seed):
            result = play_hand(bots, rng, hand_index, capture_tricks=True)
            assert result.initial_hands is not None
            assert result.trick_history is not None

            all_cards = [card for hand in result.initial_hands for card in hand]
            assert len(all_cards) == 52
            assert len(set(all_cards)) == 52

            assert len(result.trick_history) == 13
            for trick in result.trick_history:
                assert len(trick.cards) == 4

            assert sum(result.raw_points) == 26
            if result.moon_shooter is not None:
                expected = shoot_moon(result.raw_points.copy())
                assert result.points == expected

            hands = [hand.copy() for hand in result.initial_hands]
            hearts_broken = False
            for trick_index, trick in enumerate(result.trick_history):
                current = TrickState(leader=trick.leader)
                for player_idx, card in trick.cards:
                    legal = legal_moves(
                        hands[player_idx],
                        current,
                        hearts_broken=hearts_broken,
                        is_first_trick=trick_index == 0,
                    )
                    assert card in legal
                    current.add_card(player_idx, card)
                    hands[player_idx].remove(card)
                    if card.suit == Suit.HEARTS:
                        hearts_broken = True

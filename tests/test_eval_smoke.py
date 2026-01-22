import pytest

from hearts_ai.bots.random_bot import RandomBot
from hearts_ai.core.game import play_hand
from hearts_ai.training.eval import evaluate
from hearts_ai.training.params import ParameterSet
from hearts_ai.util.rng import create_rng


def test_evaluate_with_single_opponent_pool():
    params = ParameterSet()
    metrics = evaluate(params, params, seeds=[1], hands_per_seed=5, opponents=["SafeBot"])
    assert metrics.mean_penalty >= 0


def test_play_hand_requires_four_bots():
    rng = create_rng(1)
    bots = [RandomBot(rng) for _ in range(3)]
    with pytest.raises(ValueError, match="Expected 4 bots"):
        play_hand(bots, rng, hand_index=0)

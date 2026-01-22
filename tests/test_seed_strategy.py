from hearts_ai.core.cards import Deck
from hearts_ai.util.rng import create_rng


def test_training_seed_strategy_distinct_deals():
    base_seed = 42
    deals = []
    for idx in range(4):
        rng = create_rng(base_seed + idx)
        hand = Deck(rng).deal()
        deals.append(tuple(card.short() for card in hand[0]))
    assert len(set(deals)) == 4

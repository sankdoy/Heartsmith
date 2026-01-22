from hearts_ai.core.cards import Deck


def test_deck_deal_unique():
    deck = Deck()
    hands = deck.deal()
    all_cards = [card for hand in hands for card in hand]
    assert len(all_cards) == 52
    assert len(set(all_cards)) == 52

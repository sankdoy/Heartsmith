from hearts_ai.bots.safe_bot import SafeBot
from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.rules import legal_moves
from hearts_ai.core.state import TrickState


def test_safe_bot_passes_three_distinct_cards():
    hand = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.CLUBS, Rank.THREE),
        Card(Suit.DIAMONDS, Rank.FOUR),
        Card(Suit.DIAMONDS, Rank.FIVE),
        Card(Suit.SPADES, Rank.SIX),
        Card(Suit.SPADES, Rank.SEVEN),
        Card(Suit.HEARTS, Rank.EIGHT),
        Card(Suit.HEARTS, Rank.NINE),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.CLUBS, Rank.JACK),
        Card(Suit.DIAMONDS, Rank.QUEEN),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.SPADES, Rank.ACE),
    ]
    bot = SafeBot()
    picks = bot.choose_pass(hand)
    assert len(picks) == 3
    assert len(set(picks)) == 3


def test_safe_bot_choose_card_is_legal():
    hand = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.SPADES, Rank.QUEEN),
    ]
    trick = TrickState(leader=1)
    trick.add_card(1, Card(Suit.CLUBS, Rank.THREE))
    legal = legal_moves(hand, trick, hearts_broken=False, is_first_trick=False)
    bot = SafeBot()
    chosen = bot.choose_card(hand, legal, trick, hearts_broken=False, is_first_trick=False)
    assert chosen in legal

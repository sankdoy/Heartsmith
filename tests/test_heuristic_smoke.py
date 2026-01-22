from hearts_ai.bots.heuristic_bot import HeuristicBot
from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.rules import legal_moves
from hearts_ai.core.state import TrickState


def test_choose_pass_three_cards():
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
    bot = HeuristicBot()
    picks = bot.choose_pass(hand)
    assert len(picks) == 3
    assert len(set(picks)) == 3


def test_choose_play_is_legal():
    hand = [
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.SPADES, Rank.QUEEN),
    ]
    trick = TrickState(leader=1)
    trick.add_card(1, Card(Suit.CLUBS, Rank.THREE))
    legal = legal_moves(hand, trick, hearts_broken=False, is_first_trick=False)
    bot = HeuristicBot()
    chosen = bot.choose_card(hand, legal, trick, hearts_broken=False, is_first_trick=False)
    assert chosen in legal


def test_first_trick_no_point_dump_if_nonpoint_available():
    hand = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.THREE),
    ]
    trick = TrickState(leader=0)
    trick.add_card(0, Card(Suit.CLUBS, Rank.TWO))
    legal = legal_moves(hand, trick, hearts_broken=False, is_first_trick=True)
    bot = HeuristicBot()
    chosen = bot.choose_card(hand, legal, trick, hearts_broken=False, is_first_trick=True)
    assert chosen.suit != Suit.HEARTS

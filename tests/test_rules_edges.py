from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.game import should_end_game
from hearts_ai.core.passing import PassDirection, pass_direction
from hearts_ai.core.rules import legal_moves
from hearts_ai.core.scoring import shoot_moon
from hearts_ai.core.state import TrickState


def test_hearts_not_lead_until_broken_unless_only_hearts():
    hand = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.CLUBS, Rank.THREE)]
    trick = TrickState(leader=0)
    legal = legal_moves(hand, trick, hearts_broken=False, is_first_trick=False)
    assert all(card.suit != Suit.HEARTS for card in legal)

    only_hearts = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)]
    legal_only = legal_moves(only_hearts, trick, hearts_broken=False, is_first_trick=False)
    assert all(card.suit == Suit.HEARTS for card in legal_only)


def test_first_trick_must_lead_two_clubs():
    hand = [Card(Suit.CLUBS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE)]
    trick = TrickState(leader=0)
    legal = legal_moves(hand, trick, hearts_broken=False, is_first_trick=True)
    assert legal == [Card(Suit.CLUBS, Rank.TWO)]


def test_must_follow_suit_if_possible():
    hand = [Card(Suit.SPADES, Rank.THREE), Card(Suit.HEARTS, Rank.FOUR)]
    trick = TrickState(leader=1)
    trick.add_card(1, Card(Suit.SPADES, Rank.TWO))
    legal = legal_moves(hand, trick, hearts_broken=True, is_first_trick=False)
    assert legal == [Card(Suit.SPADES, Rank.THREE)]


def test_first_trick_void_in_clubs_no_point_dump_if_nonpoint_available():
    hand = [Card(Suit.HEARTS, Rank.TWO), Card(Suit.DIAMONDS, Rank.THREE)]
    trick = TrickState(leader=1)
    trick.add_card(1, Card(Suit.CLUBS, Rank.TWO))
    legal = legal_moves(hand, trick, hearts_broken=False, is_first_trick=True)
    assert Card(Suit.HEARTS, Rank.TWO) not in legal


def test_passing_cycle_correctness():
    assert pass_direction(0) == PassDirection.LEFT
    assert pass_direction(1) == PassDirection.RIGHT
    assert pass_direction(2) == PassDirection.ACROSS
    assert pass_direction(3) == PassDirection.KEEP
    assert pass_direction(4) == PassDirection.LEFT


def test_shoot_moon_transform():
    points = [0, 26, 0, 0]
    assert shoot_moon(points) == [26, 0, 26, 26]


def test_game_end_condition_unique_lowest():
    totals = [101, 100, 120, 110]
    assert should_end_game(totals) is True

    tied_lowest = [101, 100, 120, 100]
    assert should_end_game(tied_lowest) is False

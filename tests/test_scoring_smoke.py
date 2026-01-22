from hearts_ai.core.cards import Card, Rank, Suit
from hearts_ai.core.scoring import card_points, shoot_moon


def test_scoring_points():
    heart = Card(Suit.HEARTS, Rank.TWO)
    qs = Card(Suit.SPADES, Rank.QUEEN)
    assert card_points(heart) == 1
    assert card_points(qs) == 13


def test_shoot_moon_transform():
    points = [26, 0, 0, 0]
    assert shoot_moon(points) == [0, 26, 26, 26]

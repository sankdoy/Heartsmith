from hearts_ai.core.passing import PassDirection, pass_direction


def test_pass_cycle_order():
    assert pass_direction(0) == PassDirection.LEFT
    assert pass_direction(1) == PassDirection.RIGHT
    assert pass_direction(2) == PassDirection.ACROSS
    assert pass_direction(3) == PassDirection.KEEP
    assert pass_direction(4) == PassDirection.LEFT

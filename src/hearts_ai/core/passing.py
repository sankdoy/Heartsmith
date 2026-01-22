from __future__ import annotations

from enum import Enum


class PassDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    ACROSS = "across"
    KEEP = "keep"


PASS_CYCLE = [PassDirection.LEFT, PassDirection.RIGHT, PassDirection.ACROSS, PassDirection.KEEP]


def pass_direction(hand_index: int) -> PassDirection:
    return PASS_CYCLE[hand_index % len(PASS_CYCLE)]

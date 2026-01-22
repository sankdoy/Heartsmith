from __future__ import annotations

import random
from typing import Optional

RNG = random.Random


def create_rng(seed: Optional[int] = None) -> RNG:
    return random.Random(seed)

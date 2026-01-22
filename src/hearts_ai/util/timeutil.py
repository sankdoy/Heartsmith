from __future__ import annotations

from datetime import datetime


def format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")

from __future__ import annotations

import logging
import os
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from queue import Queue
from typing import Iterable


def setup_logging(
    queue: Queue,
    extra_handlers: Iterable[logging.Handler] | None = None,
    log_file: str | None = None,
) -> QueueListener:
    root = logging.getLogger()
    if not getattr(root, "_hearts_ai_configured", False):
        root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    handlers: list[logging.Handler] = []
    if os.getenv("HEARTS_AI_CONSOLE_LOG") == "1":
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        handlers.append(console)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    queue_handler = QueueHandler(queue)

    if not getattr(root, "_hearts_ai_configured", False):
        root.handlers.clear()
        root.addHandler(queue_handler)
        root._hearts_ai_configured = True

    if extra_handlers:
        handlers.extend(list(extra_handlers))

    listener = QueueListener(queue, *handlers)
    listener.start()
    return listener

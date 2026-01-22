import logging
import os
from queue import Queue

from hearts_ai.util.logging_setup import setup_logging


def _has_stream_handler() -> bool:
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return True
    return False


def test_logging_setup_no_console_by_default(monkeypatch):
    monkeypatch.delenv("HEARTS_AI_CONSOLE_LOG", raising=False)
    root = logging.getLogger()
    root.handlers.clear()
    if hasattr(root, "_hearts_ai_configured"):
        delattr(root, "_hearts_ai_configured")
    listener = setup_logging(Queue(), extra_handlers=None)
    listener.stop()
    assert _has_stream_handler() is False


def test_logging_setup_with_console(monkeypatch):
    monkeypatch.setenv("HEARTS_AI_CONSOLE_LOG", "1")
    root = logging.getLogger()
    root.handlers.clear()
    if hasattr(root, "_hearts_ai_configured"):
        delattr(root, "_hearts_ai_configured")
    listener = setup_logging(Queue(), extra_handlers=None)
    listener.stop()
    assert any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in listener.handlers
    )

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QTimer, Signal


@dataclass
class LogEntry:
    timestamp: float
    level: str
    source: str
    message: str


class QtLogHandler(logging.Handler, QObject):
    log_batch_received = Signal(list)

    def __init__(self) -> None:
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self._buffer: list[LogEntry] = []
        self._max_buffer = 10000
        self._max_per_flush = 50
        self._dropped = 0
        self._timer = QTimer()
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._flush)
        self._timer.start()

    def emit(self, record: logging.LogRecord) -> None:
        entry = LogEntry(
            timestamp=record.created or time.time(),
            level=record.levelname,
            source=record.name,
            message=record.getMessage(),
        )
        self._buffer.append(entry)
        overflow = len(self._buffer) - self._max_buffer
        if overflow > 0:
            self._buffer = self._buffer[overflow:]
            self._dropped += overflow

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self._dropped:
            self._buffer.insert(
                0,
                LogEntry(
                    timestamp=time.time(),
                    level="WARNING",
                    source="log_bridge",
                    message=f"Dropped {self._dropped} log lines due to overflow",
                ),
            )
            self._dropped = 0
        batch = self._buffer[: self._max_per_flush]
        self._buffer = self._buffer[self._max_per_flush :]
        self.log_batch_received.emit(batch)

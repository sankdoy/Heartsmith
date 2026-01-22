from __future__ import annotations

import threading


class TrainControl:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self._paused = False
        self._pause_cond = threading.Condition()

    def request_stop(self) -> None:
        self.stop_event.set()
        self.request_pause(False)

    def request_pause(self, paused: bool) -> None:
        with self._pause_cond:
            self._paused = paused
            self._pause_cond.notify_all()

    def check_stop(self) -> bool:
        return self.stop_event.is_set()

    def wait_if_paused(self) -> None:
        with self._pause_cond:
            while self._paused and not self.stop_event.is_set():
                self._pause_cond.wait(timeout=0.1)

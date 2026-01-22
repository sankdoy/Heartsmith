from __future__ import annotations

import sys
from queue import Queue

from PySide6.QtWidgets import QApplication

from hearts_ai.gui.main_window import MainWindow
from hearts_ai.gui.theme import THEME, build_stylesheet
from hearts_ai.gui.services.log_bridge import QtLogHandler
from hearts_ai.util.logging_setup import setup_logging


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(build_stylesheet(THEME))
    queue: Queue = Queue()
    qt_handler = QtLogHandler()
    listener = setup_logging(queue, extra_handlers=[qt_handler], log_file="logs/app.log")

    window = MainWindow(log_handler=qt_handler)
    window.show()

    exit_code = app.exec()
    listener.stop()
    sys.exit(exit_code)

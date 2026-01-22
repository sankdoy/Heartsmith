from __future__ import annotations

from collections import deque

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDialog

from hearts_ai.gui.theme import THEME


class MetricPlot(QWidget):
    def __init__(
        self,
        title: str,
        y_min: float | None = None,
        y_max: float | None = None,
        y_label: str | None = None,
        x_label: str = "tick",
    ) -> None:
        super().__init__()
        self._title = title
        self._series: dict[str, dict] = {}
        self._window_size: int | None = None
        self._smoothing_mode = "none"
        self._smoothing_value = 0.0
        self._markers: list[pg.InfiniteLine] = []
        self._baselines: dict[str, pg.InfiniteLine] = {}

        self._label = QLabel(title)
        self._plot = pg.PlotWidget()
        self._plot.setBackground(pg.mkColor(THEME.panel))
        self._plot.setAntialiasing(True)
        self._plot.showGrid(x=True, y=True, alpha=0.08)
        legend = self._plot.addLegend()
        if hasattr(legend, "setLabelTextColor"):
            legend.setLabelTextColor(THEME.text)
        if y_min is not None or y_max is not None:
            self._plot.setYRange(y_min if y_min is not None else 0, y_max if y_max is not None else 1)
        self._y_label = y_label or ""
        if y_label:
            self._plot.setLabel("left", y_label)
        self._plot.setLabel("bottom", x_label)
        for axis_name in ("bottom", "left"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(THEME.text_muted))
            axis.setTextPen(pg.mkPen(THEME.text_muted))
            axis.setStyle(tickFont=pg.QtGui.QFont("Arial", 10))
            axis.enableAutoSIPrefix(False)
        self._plot.setMinimumHeight(260)
        axis = self._plot.getAxis("bottom")
        axis.setTickSpacing(major=100, minor=20)
        if "rate" in self._y_label.lower():
            self._plot.setYRange(0, 1)
            self._plot.setAxisItems({"left": PercentAxisItem()})

        layout = QVBoxLayout()
        layout.addWidget(self._label)
        layout.addWidget(self._plot)
        self.setLayout(layout)

    def add_series(
        self,
        name: str,
        color: str,
        style: Qt.PenStyle = Qt.SolidLine,
        symbol: str | None = None,
    ) -> None:
        if name in self._series:
            return
        x = []
        y = []
        curve = self._plot.plot(
            pen=pg.mkPen(color=color, width=2, style=style),
            symbol=symbol,
            symbolSize=7,
            symbolBrush=pg.mkBrush(color),
            symbolPen=pg.mkPen(color),
            name=name,
        )
        self._series[name] = {"x": x, "y": y, "curve": curve}

    def add_point(self, series: str, x: float, y: float) -> None:
        if series not in self._series:
            self.add_series(series, "#2e2a52")
        entry = self._series[series]
        entry["x"].append(x)
        entry["y"].append(y)
        self._refresh_series(series)

    def add_event_marker(self, x: float) -> None:
        line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(color=THEME.border, style=Qt.DashLine))
        self._plot.addItem(line)
        self._markers.append(line)

    def set_baseline(self, name: str, y: float, color: str) -> None:
        if name in self._baselines:
            self._baselines[name].setPos(y)
            return
        line = pg.InfiniteLine(pos=y, angle=0, pen=pg.mkPen(color=color, style=Qt.DashLine))
        self._plot.addItem(line)
        self._baselines[name] = line

    def set_window(self, size: int | None) -> None:
        self._window_size = size
        self._refresh_all()

    def set_smoothing(self, mode: str, value: float) -> None:
        self._smoothing_mode = mode
        self._smoothing_value = value
        self._refresh_all()

    def _refresh_all(self) -> None:
        for name in self._series.keys():
            self._refresh_series(name)

    def _refresh_series(self, name: str) -> None:
        entry = self._series[name]
        xs = entry["x"]
        ys = entry["y"]
        if self._window_size and self._window_size > 0:
            xs = xs[-self._window_size :]
            ys = ys[-self._window_size :]

        if self._smoothing_mode == "ema" and self._smoothing_value > 0:
            ys = _ema(ys, self._smoothing_value)
        elif self._smoothing_mode == "rolling" and self._smoothing_value > 1:
            ys = _rolling_mean(ys, int(self._smoothing_value))

        entry["curve"].setData(xs, ys)

    def mouseDoubleClickEvent(self, event) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._title)
        layout = QVBoxLayout()
        plot = pg.PlotWidget()
        plot.setBackground(pg.mkColor(THEME.panel))
        plot.showGrid(x=True, y=True, alpha=0.15)
        legend = plot.addLegend()
        if hasattr(legend, "setLabelTextColor"):
            legend.setLabelTextColor(THEME.text)
        for axis_name in ("bottom", "left"):
            axis = plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(THEME.text_muted))
            axis.setTextPen(pg.mkPen(THEME.text_muted))
            axis.enableAutoSIPrefix(False)
        for name, entry in self._series.items():
            plot.plot(
                entry["x"],
                entry["y"],
                pen=entry["curve"].opts.get("pen"),
                symbol=entry["curve"].opts.get("symbol"),
                symbolSize=6,
                name=name,
            )
        layout.addWidget(plot)
        dialog.setLayout(layout)
        dialog.resize(720, 480)
        dialog.exec()
        super().mouseDoubleClickEvent(event)


class PercentAxisItem(pg.AxisItem):
    def __init__(self) -> None:
        super().__init__(orientation="left")

    def tickStrings(self, values, scale, spacing):
        return [f"{value * 100:.0f}%" for value in values]


def _ema(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


def _rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    out = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        out.append(sum(chunk) / len(chunk))
    return out

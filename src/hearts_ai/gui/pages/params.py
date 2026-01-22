from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QLabel,
    QLineEdit,
)

from hearts_ai.training.params import ParameterSet
from hearts_ai.gui.services.training_worker import TrainingWorker


class ParamsPage(QWidget):
    def __init__(self, store: ParameterSet, worker: TrainingWorker) -> None:
        super().__init__()
        self._store = store
        self._worker = worker
        self._loading = False

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search parameters")

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(["Name", "Group", "Value", "Min", "Max", "Locked"])
        self._table.horizontalHeader().setStretchLastSection(True)

        self._detail = QTextEdit()
        self._detail.setReadOnly(True)

        self._detail_label = QLabel("Details")

        left = QVBoxLayout()
        left.addWidget(self._search)
        left.addWidget(self._table)

        right = QVBoxLayout()
        right.addWidget(self._detail_label)
        right.addWidget(self._detail)

        layout = QHBoxLayout()
        layout.addLayout(left, 2)
        layout.addLayout(right, 1)
        self.setLayout(layout)

        self._table.itemSelectionChanged.connect(self._update_detail)
        self._table.itemChanged.connect(self._handle_item_change)
        self._search.textChanged.connect(self._populate)

        self._populate()

    def refresh(self) -> None:
        self._populate()

    def _populate(self) -> None:
        self._loading = True
        query = self._search.text().strip().lower()
        params = [p for p in self._store.all() if query in p.name.lower()]
        self._table.setRowCount(len(params))
        for row, param in enumerate(params):
            name_item = QTableWidgetItem(param.name)
            name_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            group_item = QTableWidgetItem(param.group)
            group_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            value_item = QTableWidgetItem(f"{param.value:.3f}")

            min_item = QTableWidgetItem(f"{param.min_value:.3f}")
            min_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            max_item = QTableWidgetItem(f"{param.max_value:.3f}")
            max_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)

            locked_item = QTableWidgetItem("")
            locked_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            locked_item.setCheckState(Qt.Checked if param.locked else Qt.Unchecked)

            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, group_item)
            self._table.setItem(row, 2, value_item)
            self._table.setItem(row, 3, min_item)
            self._table.setItem(row, 4, max_item)
            self._table.setItem(row, 5, locked_item)

        self._loading = False

    def _update_detail(self) -> None:
        items = self._table.selectedItems()
        if not items:
            return
        row = items[0].row()
        name = self._table.item(row, 0).text()
        param = self._store.get_param(name)
        doc = param.doc
        self._detail.setPlainText(
            f"What it controls:\n{doc.get('what','')}\n\n"
            f"When it matters:\n{doc.get('when','')}\n\n"
            f"If increased:\n{doc.get('increased','')}\n\n"
            f"Edge case:\n{doc.get('edge','')}"
        )

    def _handle_item_change(self, item: QTableWidgetItem) -> None:
        if self._loading:
            return
        name_item = self._table.item(item.row(), 0)
        if not name_item:
            return
        name = name_item.text()
        if item.column() == 2:
            try:
                value = float(item.text())
            except ValueError:
                return
            self._store.update(name, value)
            self._worker.update_param(name, value)
        elif item.column() == 5:
            locked = item.checkState() == Qt.Checked
            self._store.set_locked(name, locked)
            self._worker.update_locked(name, locked)

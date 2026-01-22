# Heartsmith — Hearts Bot Trainer (WIP)

Hearts bot trainer with a Mac-friendly GUI for experimentation and learning. This project is **not finished** and the training loop is **not improving reliably yet**.

## Status

- WIP: training quality is unstable and may regress.
- Expect breaking changes while the trainer and evaluation loop are tuned.

## What This Is

- A Hearts rules engine + simulation harness.
- A heuristic bot with tunable parameters.
- A GUI for training, evaluation, logs, and parameter editing.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
python -m hearts_ai.gui
```

## Tests

```bash
pytest -q
```

Slow tests:

```bash
pytest -q -m slow
```

## Development Notes

- Target: Python 3.12 + PySide6 + pyqtgraph.
- The GUI should stay responsive during training.
- Evaluation costs can dominate runtime; use short runs for iteration.

## Contributing

This is a work-in-progress. Issues and PRs are welcome, especially around:

- Evaluation stability and performance.
- Training acceptance criteria (avoid regressions).
- Better baseline bots and heuristics.
- GUI clarity and diagnostics.

## License

TBD. Choose a license before public release.

# Contributing to Heartsmith

Thanks for considering a contribution. This project is a work in progress and
expects breaking changes.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

## What Helps Most

- Evaluation stability and performance.
- Training acceptance criteria (avoid regressions).
- Baseline bots and heuristics.
- UI clarity and diagnostics.

## Pull Requests

- Keep PRs focused and small.
- Add or update tests when you change behavior.
- Mention any runtime or performance impacts.

## Reporting Issues

Include:

- Run settings (seed mode, eval config, preset).
- `run_meta.json`, `metrics_train.jsonl`, `metrics_eval.jsonl` if possible.
- Steps to reproduce.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from hearts_ai.util.version import get_version_string


def build_run_config(
    train_config: dict[str, Any],
    eval_config: dict[str, Any],
    opponent_pools: dict[str, Any],
    seed_mode: str,
    base_seed: int,
    latest_seed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "train_config": train_config,
        "seed_mode": seed_mode,
        "base_seed": base_seed,
        "latest_seed": latest_seed,
        "eval_config": eval_config,
        "opponent_pools": opponent_pools,
        "app_version": get_version_string(),
    }


def write_run_config(run_dir: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

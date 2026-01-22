from pathlib import Path

from hearts_ai.training.run_config import build_run_config, write_run_config


def test_write_run_config(tmp_path):
    config = build_run_config(
        train_config={"hands_per_tick": 100},
        eval_config={"eval_hands_per_seed": 50},
        opponent_pools={"train": {}, "eval": {}},
        seed_mode="cycle",
        base_seed=42,
        latest_seed={"seed": 43},
    )
    run_dir = tmp_path / "runs" / "run_test"
    write_run_config(run_dir, config)
    assert (run_dir / "run_config.json").exists()

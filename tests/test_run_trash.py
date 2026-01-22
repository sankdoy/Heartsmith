from pathlib import Path
import tempfile

from hearts_ai.gui.pages.runs import move_to_trash


def test_move_to_trash_moves_run_dir():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runs_dir = root / "runs"
        runs_dir.mkdir()
        run_dir = runs_dir / "run_20240101_000000"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text("{}", encoding="utf-8")
        trash_root = runs_dir / "_trash" / "batch"
        trash_root.mkdir(parents=True, exist_ok=True)

        moved = move_to_trash(run_dir, trash_root, runs_dir)

        assert not run_dir.exists()
        assert moved.exists()

import json

from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trainer import Trainer, TrainingConfig


def test_latest_params_written_and_loadable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    params = ParameterSet()
    config = TrainingConfig(hands_per_tick=1, seed=1, eval_interval=1000, holdout_seeds=[])
    trainer = Trainer(params, config)

    def on_metrics(_metrics):
        trainer.stop()

    trainer.run(on_metrics, lambda _p: None)

    latest_path = tmp_path / "runs" / "latest.json"
    assert latest_path.exists()
    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    params_path = tmp_path / payload["path"]
    assert params_path.exists()
    loaded = ParameterSet.from_json(params_path.read_text(encoding="utf-8"))
    assert loaded.all()

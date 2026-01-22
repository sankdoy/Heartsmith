from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trainer import Trainer, TrainingConfig


def test_trainer_run_one_tick_no_crash():
    params = ParameterSet()
    config = TrainingConfig(hands_per_tick=1, seed=123, eval_interval=10, holdout_seeds=[])
    trainer = Trainer(params, config)
    call_count = {"n": 0}

    def on_metrics(_metrics):
        call_count["n"] += 1
        if call_count["n"] >= 1:
            trainer.stop()

    def on_best(_params):
        pass

    trainer.run(on_metrics, on_best)
    assert call_count["n"] >= 1


def test_trainer_handles_immediate_stop():
    params = ParameterSet()
    config = TrainingConfig(hands_per_tick=1, seed=123, eval_interval=10, holdout_seeds=[])
    trainer = Trainer(params, config)
    trainer.stop()
    trainer.run(lambda _m: None, lambda _p: None)

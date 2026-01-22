import threading
import time

from hearts_ai.training.control import TrainControl
from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trainer import Trainer, TrainingConfig


def test_stop_exits_quickly():
    params = ParameterSet()
    control = TrainControl()
    config = TrainingConfig(hands_per_tick=2, seed=1, eval_interval=1000, holdout_seeds=[])
    trainer = Trainer(params, config, control=control)

    thread = threading.Thread(target=trainer.run, args=(lambda _m: None, lambda _p: None))
    thread.start()
    time.sleep(0.05)
    trainer.stop()
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_pause_freezes_progress():
    params = ParameterSet()
    control = TrainControl()
    config = TrainingConfig(hands_per_tick=2, seed=1, eval_interval=1000, holdout_seeds=[])
    trainer = Trainer(params, config, control=control)

    thread = threading.Thread(target=trainer.run, args=(lambda _m: None, lambda _p: None))
    thread.start()
    time.sleep(0.1)
    trainer.pause()
    hands_before = trainer.hands_done
    time.sleep(0.2)
    hands_after = trainer.hands_done
    assert hands_after == hands_before
    trainer.resume()
    time.sleep(0.1)
    assert trainer.hands_done >= hands_after
    trainer.stop()
    thread.join(timeout=1.0)

from hearts_ai.training.params import ParameterSet
from hearts_ai.training.trainer import Trainer, TrainingConfig


def test_adapt_hands_per_tick_clamped():
    config = TrainingConfig(hands_per_tick=1000, hands_per_second_target=1000, updates_per_sec=5.0)
    trainer = Trainer(ParameterSet(), config)

    trainer._adapt_hands_per_tick(elapsed=0.05)
    assert config.hands_per_tick >= 750

    config.hands_per_tick = 1000
    trainer._adapt_hands_per_tick(elapsed=20.0)
    assert config.hands_per_tick <= 1250


def test_simple_tick_cap():
    config = TrainingConfig(hands_per_tick=1000, simple_max_tick_seconds=0.25)
    trainer = Trainer(ParameterSet(), config)
    trainer._last_train_hands_per_s = 1000.0
    trainer._apply_simple_tick_cap()
    assert config.hands_per_tick <= 250

from types import SimpleNamespace

import hearts_ai.training.eval as eval_mod
from hearts_ai.training.params import ParameterSet


def test_eval_seat_mapping(monkeypatch):
    fake = SimpleNamespace(
        points=[10, 20, 30, 40],
        qs_taken=[False, False, False, False],
        hearts_taken=[0, 0, 0, 0],
        moon_shooter=None,
    )

    def fake_simulate_hand(_bots, _rng, _hand_index):
        return fake

    monkeypatch.setattr(eval_mod, "simulate_hand", fake_simulate_hand)
    params = ParameterSet()
    metrics = eval_mod.evaluate(params, params, seeds=[1], hands_per_seed=4, opponents=["SafeBot"])
    assert metrics.mean_penalty == 25.0

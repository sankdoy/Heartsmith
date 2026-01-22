from hearts_ai.training.eval import evaluate
from hearts_ai.training.metrics import format_optional_points
from hearts_ai.training.params import ParameterSet


def test_eval_mean_points_within_range():
    params = ParameterSet()
    metrics = evaluate(params, params, seeds=[1], hands_per_seed=5, opponents=["SafeBot"])
    assert 0.0 <= metrics.mean_penalty <= 26.0


def test_format_optional_points_none():
    assert format_optional_points(None) == "—"

from hearts_ai.training.autopolicy import AutoPolicy
from hearts_ai.training.metrics import EvalSnapshot, HoldoutSnapshot, MetricsSnapshot


def test_autopolicy_overfit_triggers_actions():
    policy = AutoPolicy()
    eval_snapshot = EvalSnapshot(
        iteration=10,
        mean_penalty=5.0,
        mean_penalty_se=0.1,
        win_rate=0.5,
        qs_rate=0.2,
        hearts_rate=0.1,
        moon_conceded_rate=0.0,
        seeds_count=5,
        hands_per_seed=200,
        opponent_breakdown={},
    )
    holdout = HoldoutSnapshot(mean_penalty=6.2, mean_penalty_se=0.1, seeds_count=5, hands_per_seed=50)
    result = policy.update(eval_snapshot, holdout, None)
    actions = {action.action for action in result.actions}
    assert "seed_mode" in actions
    assert "eval_hands_per_seed" in actions


def test_autopolicy_plateau_suggests_stop():
    policy = AutoPolicy(plateau_window=4, plateau_min_improvement=0.05)
    suggestion = None
    for idx, mean in enumerate([5.5, 5.48, 5.47, 5.46], start=1):
        eval_snapshot = EvalSnapshot(
            iteration=idx,
            mean_penalty=mean,
            mean_penalty_se=0.05,
            win_rate=0.5,
            qs_rate=0.2,
            hearts_rate=0.1,
            moon_conceded_rate=0.0,
            seeds_count=5,
            hands_per_seed=200,
            opponent_breakdown={},
        )
        result = policy.update(eval_snapshot, None, None)
        suggestion = result.suggestion
    assert suggestion is not None


def test_autopolicy_noisy_eval_increases_hands():
    policy = AutoPolicy(noisy_se_threshold=0.2, cooldown_evals=1)
    eval_snapshot = EvalSnapshot(
        iteration=5,
        mean_penalty=6.0,
        mean_penalty_se=0.4,
        win_rate=0.4,
        qs_rate=0.2,
        hearts_rate=0.1,
        moon_conceded_rate=0.0,
        seeds_count=10,
        hands_per_seed=150,
        opponent_breakdown={},
    )
    policy.set_run_length_mode("10 min")
    policy.update(eval_snapshot, None, None)
    result = policy.update(eval_snapshot, None, None)
    action = next((act for act in result.actions if act.action == "eval_hands_per_seed"), None)
    assert action is not None
    assert int(action.value) > eval_snapshot.hands_per_seed


def test_autopolicy_schedule_actions():
    policy = AutoPolicy()
    metrics = MetricsSnapshot(
        iteration=5,
        mean_penalty=6.0,
        win_rate=0.3,
        qs_rate=0.1,
        best_score=6.0,
        hands_done=200,
        hand_index=200,
    )
    result = policy.update(None, None, metrics)
    actions = {action.action for action in result.actions}
    assert "train_opponents" in actions


def test_autopolicy_respects_cap():
    policy = AutoPolicy(noisy_se_threshold=0.2, cooldown_evals=1)
    policy.set_run_length_mode("2 min")
    eval_snapshot = EvalSnapshot(
        iteration=3,
        mean_penalty=6.0,
        mean_penalty_se=0.5,
        win_rate=0.4,
        qs_rate=0.2,
        hearts_rate=0.1,
        moon_conceded_rate=0.0,
        seeds_count=10,
        hands_per_seed=280,
        opponent_breakdown={},
    )
    policy.update(eval_snapshot, None, None)
    result = policy.update(eval_snapshot, None, None)
    action = next((act for act in result.actions if act.action == "eval_hands_per_seed"), None)
    assert action is not None
    assert int(action.value) <= 300


def test_autopolicy_cooldown():
    policy = AutoPolicy(noisy_se_threshold=0.2, cooldown_evals=2)
    policy.set_run_length_mode("10 min")
    eval_snapshot = EvalSnapshot(
        iteration=1,
        mean_penalty=6.0,
        mean_penalty_se=0.5,
        win_rate=0.4,
        qs_rate=0.2,
        hearts_rate=0.1,
        moon_conceded_rate=0.0,
        seeds_count=10,
        hands_per_seed=200,
        opponent_breakdown={},
    )
    policy.update(eval_snapshot, None, None)
    first = policy.update(eval_snapshot, None, None)
    action = next((act for act in first.actions if act.action == "eval_hands_per_seed"), None)
    assert action is not None
    second = policy.update(eval_snapshot, None, None)
    third = policy.update(eval_snapshot, None, None)
    assert not any(act.action == "eval_hands_per_seed" for act in second.actions)
    assert not any(act.action == "eval_hands_per_seed" for act in third.actions)

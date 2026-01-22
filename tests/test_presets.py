from hearts_ai.training.presets import (
    get_presets,
    preset_to_values,
    seed_preset_to_list,
    apply_eval_budget,
    max_eval_hands_per_seed,
)


def test_preset_mapping_to_values():
    presets = get_presets()
    preset = presets[0]
    values = preset_to_values(preset)
    assert values["hands_per_tick"] == preset.hands_per_tick
    assert values["eval_hands_per_seed"] == preset.eval_hands_per_seed
    assert isinstance(values["eval_seeds"], list)
    assert len(values["eval_seeds"]) > 0


def test_seed_preset_list_sizes():
    assert len(seed_preset_to_list("quick")) == 5
    assert len(seed_preset_to_list("standard")) == 20
    assert len(seed_preset_to_list("thorough")) == 100


def test_apply_eval_budget_caps_total_hands():
    seed_preset, hands_per_seed, _ = apply_eval_budget("standard", 400, "2 min")
    assert hands_per_seed <= max_eval_hands_per_seed("2 min")


def test_eval_budget_for_run_length():
    assert max_eval_hands_per_seed("2 min") <= max_eval_hands_per_seed("10 min")

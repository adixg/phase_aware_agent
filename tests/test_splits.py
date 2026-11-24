"""The 16-week split and the rebuilt tool predictor's pre-registered grid."""

from analysis.dag import splits, tool_eval_16w, tool_gates_16w


def test_the_windows_are_in_time_order_and_never_overlap():
    windows = [splits.FIT_DAYS, splits.TUNE_DAYS, splits.VALIDATION_DAYS, splits.EXPOSED_DAYS, splits.HELDOUT_DAYS]
    flat = [d for w in windows for d in w]
    assert flat == sorted(flat) and len(flat) == len(set(flat))


def test_the_windows_cover_the_release_from_may_17_to_sep_5_without_a_gap():
    flat = [d for w in (splits.FIT_DAYS, splits.TUNE_DAYS, splits.VALIDATION_DAYS, splits.EXPOSED_DAYS, splits.HELDOUT_DAYS) for d in w]
    assert flat[0] == "2026-05-17" and flat[-1] == "2026-09-05" and len(flat) == 112


def test_the_window_sizes_are_the_ones_the_documentation_states():
    assert [len(w) for w in (splits.FIT_DAYS, splits.TUNE_DAYS, splits.VALIDATION_DAYS, splits.EXPOSED_DAYS, splits.HELDOUT_DAYS)] == [84, 7, 7, 10, 4]


def test_the_held_out_days_are_the_ones_the_builder_refuses_to_build():
    from analysis.dag.build import HELDOUT_FROM
    assert min(splits.HELDOUT_DAYS) == HELDOUT_FROM and tool_gates_16w.HELDOUT_DAYS == splits.HELDOUT_DAYS


def test_the_rebuilt_grid_fits_the_budget_and_every_chain_contains_the_baseline():
    assert len(tool_eval_16w.GRID) == tool_gates_16w.TUNING_BUDGET == 12
    assert all(chain[0] == "cat" for chain in tool_eval_16w.MODEL_CHAINS.values())


def test_every_decision_step_is_one_every_chain_can_act_at():
    assert min(tool_gates_16w.GATE_STEPS) >= 3       # the longest chain looks back three tools

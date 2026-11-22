"""The report's information ladder, on a hand-made cell."""

import pytest

from analysis.replay.report import ladder_rows


def cell(las, edf, llf, srpt, day="d1"):
    return {("las", "lru", day): las, ("edf", "lru", day): edf, ("llf_oracle", "lru", day): llf, ("srpt_oracle", "lru", day): srpt}


def test_ladder_separates_the_value_of_deadlines_from_the_value_of_the_future():
    rows = ladder_rows({(0.9, 1.0, 1_000_000.0): cell(0.80, 0.90, 0.91, 0.905)})
    r = rows[0]
    assert r["deadlines"] == pytest.approx(10.0) and r["future"] == pytest.approx(1.0) and r["total"] == pytest.approx(11.0)
    assert r["oracle"] == pytest.approx(91.0)                         # the better of the two oracles


def test_ladder_skips_unrealistic_memory_and_incomplete_cells():
    cells = {(0.9, 1.0, 0): cell(0.8, 0.9, 0.95, 0.95),                # unlimited memory: not a realistic cell
             (0.9, 1.0, 2_000_000.0): {("las", "lru", "d1"): 0.8}}      # no EDF or oracle result
    assert ladder_rows(cells) == []


def test_ladder_standard_error_comes_from_the_gap_between_days():
    c = {**cell(0.8, 0.90, 0.92, 0.92, "d1"), **cell(0.8, 0.90, 0.94, 0.94, "d2")}
    assert ladder_rows({(0.7, 0.5, 4_000_000.0): c})[0]["se"] == pytest.approx(1.0)   # gaps of 2 and 4 points

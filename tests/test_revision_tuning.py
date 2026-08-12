"""Protocol-level tests for the acceptance-critical revision controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_revision_tuning as revision  # noqa: E402


def test_all_seed_namespaces_are_pairwise_disjoint():
    pools = revision.seed_namespaces()
    revision.assert_seed_disjointness(pools)
    assert not (
        set(pools["fresh_interpolation"]) & set(pools["legacy_diagnostic"])
    )
    assert not (set(pools["nested_test"]) & set(pools["nested_selection"]))


def test_nested_grid_is_complete_and_identical_for_both_channels():
    grid = revision.nested_grid()
    assert len(grid) == (
        len(revision.NESTED_H)
        * len(revision.NESTED_DT)
        * len(revision.NESTED_MULTIPLIERS)
    )
    assert len(grid) == len(set(grid))
    assert (0.5, 0.5, 1.0) in grid
    assert revision.RIDGES[-4:] == (0.1, 1.0, 10.0, 100.0)


def test_config_selection_uses_only_supplied_calibration_rows():
    seeds = [11, 12, 13]
    configs = [(0.5, 0.5, 1.0), (1.0, 0.25, 2.0)]
    rows = []
    for config_index, (h, dt, multiplier) in enumerate(configs):
        for seed in seeds:
            rows.append(
                {
                    "method": "CD_paper",
                    "seed": seed,
                    "h": h,
                    "dt": dt,
                    "strength_multiplier": multiplier,
                    "ridge_validation_mc": {
                        f"{ridge:.12g}": 2.0 - config_index - ridge
                        for ridge in revision.RIDGES
                    },
                }
            )
    ranked = revision.rank_nested_configs(
        rows, ("CD_paper",), configs, seeds
    )
    assert ranked["CD_paper"][0]["config"] == list(configs[0])

    # A would-be test row cannot enter the API because its seed is not in the
    # supplied calibration seed set.
    contaminated = rows + [
        {
            "method": "CD_paper",
            "seed": 99,
            "h": configs[1][0],
            "dt": configs[1][1],
            "strength_multiplier": configs[1][2],
            "ridge_validation_mc": {
                f"{ridge:.12g}": 1e9 for ridge in revision.RIDGES
            },
        }
    ]
    repeated = revision.rank_nested_configs(
        contaminated, ("CD_paper",), configs, seeds
    )
    assert repeated["CD_paper"][0]["config"] == list(configs[0])
    assert not repeated["CD_paper"][0]["ridge_upper_boundary_unresolved"]


def test_ridge_upper_boundary_is_explicitly_flagged():
    seeds = [21, 22]
    config = (0.5, 0.5, 1.0)
    rows = [
        {
            "method": "B3_collective",
            "seed": seed,
            "h": config[0],
            "dt": config[1],
            "strength_multiplier": config[2],
            "ridge_validation_mc": {
                f"{ridge:.12g}": float(index)
                for index, ridge in enumerate(revision.RIDGES)
            },
        }
        for seed in seeds
    ]
    ranked = revision.rank_nested_configs(
        rows, ("B3_collective",), (config,), seeds
    )
    assert ranked["B3_collective"][0]["best_ridge"] == revision.RIDGES[-1]
    assert ranked["B3_collective"][0]["ridge_upper_boundary_unresolved"]


def test_curve_bracket_requires_both_sides_of_the_maximum():
    base = [
        {
            "method": "B3_collective",
            "mult": multiplier,
            "val_value": value,
        }
        for multiplier, value in ((1.0, 1.0), (2.0, 3.0), (4.0, 2.0))
    ]
    assert revision.curve_bracket(base, "B3_collective")["bracketed"]
    boundary = base[:2]
    assert not revision.curve_bracket(boundary, "B3_collective")["bracketed"]


def test_paired_interval_uses_within_seed_differences():
    candidate = np.array([3.0, 4.0, 5.0])
    reference = np.array([1.0, 2.0, 3.0])
    result = revision.paired_stats(candidate, reference)
    assert result["mean_difference"] == 2.0
    assert result["wins"] == 3
    assert result["n"] == 3

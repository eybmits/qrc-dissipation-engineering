"""Unit tests for the descriptive two-seed prescreen stability audit."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import audit_nested_prescreen_stability as audit  # noqa: E402


METHODS = ("local", "collective")
SEEDS = (11, 22)
RIDGES = (0.0, 1e-4)
CONFIGS = tuple((0.5, 1.0, float(index + 1)) for index in range(10))


def _synthetic_rows() -> list[dict]:
    rows = []
    for method in METHODS:
        method_offset = 0.01 if method == "collective" else 0.0
        for seed in SEEDS:
            for index, (h, dt, strength) in enumerate(CONFIGS):
                # The two seeds induce exactly reversed configuration rankings.
                score = (
                    10.0 - index if seed == SEEDS[0] else 1.0 + index
                ) + method_offset
                # Alternate which ridge is best to verify seed/config-specific
                # ridge selection rather than a pooled ridge choice.
                if (index + seed) % 2:
                    ridge_scores = {
                        "0": score - 0.25,
                        "0.0001": score,
                    }
                else:
                    ridge_scores = {
                        "0": score,
                        "0.0001": score - 0.25,
                    }
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "h": h,
                        "dt": dt,
                        "strength_multiplier": strength,
                        "ridge_validation_mc": ridge_scores,
                    }
                )
    return rows


def _payload(rows: list[dict]) -> dict:
    return audit.build_stability_payload(
        rows,
        methods=METHODS,
        configs=CONFIGS,
        seeds=SEEDS,
        ridges=RIDGES,
        frozen_selected={
            "local": CONFIGS[3],
            "collective": CONFIGS[3],
        },
        provenance={"synthetic": True},
    )


def test_seedwise_ranking_overlap_and_spearman_are_exact():
    payload = _payload(_synthetic_rows())
    for method in METHODS:
        result = payload["methods"][method]
        first = result["per_seed"][str(SEEDS[0])]
        second = result["per_seed"][str(SEEDS[1])]
        assert first["frozen_selected_config_rank"] == 4
        assert second["frozen_selected_config_rank"] == 7
        assert first["frozen_selected_config_in_top8"] is True
        assert second["frozen_selected_config_in_top8"] is True
        assert first["winner"]["config"] == list(CONFIGS[0])
        assert second["winner"]["config"] == list(CONFIGS[-1])
        assert result["top8_overlap"]["intersection_count"] == 6
        assert result["top8_overlap"]["union_count"] == 10
        assert result["top8_overlap"]["jaccard"] == pytest.approx(0.6)
        assert result["full_rank_spearman"] == pytest.approx(-1.0)


def test_best_ridge_is_chosen_separately_for_each_seed_and_config():
    rankings = audit.rank_seedwise(
        _synthetic_rows(),
        methods=METHODS,
        configs=CONFIGS,
        seeds=SEEDS,
        ridges=RIDGES,
    )
    for method in METHODS:
        for seed in SEEDS:
            by_config = {
                tuple(entry["config"]): entry
                for entry in rankings[method][seed]
            }
            for index, config in enumerate(CONFIGS):
                expected = 1e-4 if (index + seed) % 2 else 0.0
                assert by_config[config]["best_ridge"] == expected


def test_payload_and_report_are_deterministic_under_row_order_changes():
    rows = _synthetic_rows()
    expected = _payload(rows)
    random.Random(20260724).shuffle(rows)
    observed = _payload(rows)
    assert observed == expected
    assert audit.render_report(observed) == audit.render_report(expected)
    round_tripped = json.loads(json.dumps(observed, sort_keys=True))
    assert audit.render_report(round_tripped) == audit.render_report(observed)
    stored_hash = observed.pop("deterministic_payload_sha256")
    assert stored_hash == audit.sha256_json(observed)


def test_ranking_rejects_incomplete_or_duplicate_coverage():
    rows = _synthetic_rows()
    with pytest.raises(ValueError, match="coverage mismatch"):
        audit.rank_seedwise(
            rows[:-1],
            methods=METHODS,
            configs=CONFIGS,
            seeds=SEEDS,
            ridges=RIDGES,
        )
    with pytest.raises(ValueError, match="duplicate screen row"):
        audit.rank_seedwise(
            rows + [rows[0]],
            methods=METHODS,
            configs=CONFIGS,
            seeds=SEEDS,
            ridges=RIDGES,
        )


def test_tied_ridges_follow_declared_grid_order():
    rows = _synthetic_rows()
    target = rows[0]
    target["ridge_validation_mc"] = {"0": 3.0, "0.0001": 3.0}
    rankings = audit.rank_seedwise(
        rows,
        methods=METHODS,
        configs=CONFIGS,
        seeds=SEEDS,
        ridges=RIDGES,
    )
    match = next(
        entry
        for entry in rankings[target["method"]][target["seed"]]
        if tuple(entry["config"])
        == (
            target["h"],
            target["dt"],
            target["strength_multiplier"],
        )
    )
    assert match["best_ridge"] == RIDGES[0]

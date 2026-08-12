"""Tests for the focused parity and normalised-coupling revision controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_revision_controls as revision  # noqa: E402


def test_coupling_normalisations_are_anchored_at_n5():
    assert revision.coupling_multiplier(5, "variance") == pytest.approx(1.0)
    assert revision.coupling_multiplier(5, "kac") == pytest.approx(1.0)


def test_variance_and_kac_formulas():
    for n_qubits in range(4, 9):
        variance = revision.coupling_multiplier(n_qubits, "variance")
        kac = revision.coupling_multiplier(n_qubits, "kac")
        assert variance**2 * (n_qubits - 1) == pytest.approx(4.0)
        assert kac * (n_qubits - 1) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="unknown coupling scheme"):
        revision.coupling_multiplier(5, "unsupported")


def test_nested_base_draws_pair_every_n_and_scheme():
    seed = revision.fresh_seeds(revision.SCALING_SEED_NAMESPACE, 1)[0]
    full = revision.nested_base_couplings(seed, 8)
    for n_qubits in range(4, 9):
        base = revision.nested_base_couplings(seed, n_qubits)
        assert np.array_equal(base, full[:n_qubits, :n_qubits])
        variance, variance_meta = revision.scaled_couplings(
            seed, n_qubits, "variance"
        )
        kac, kac_meta = revision.scaled_couplings(seed, n_qubits, "kac")
        assert variance_meta["base_coupling_sha256"] == (
            kac_meta["base_coupling_sha256"]
        )
        assert np.allclose(
            variance,
            revision.coupling_multiplier(n_qubits, "variance") * base,
        )


def test_revision_seeds_are_fresh_and_namespaces_are_disjoint():
    parity = revision.fresh_seeds(revision.PARITY_SEED_NAMESPACE, 16)
    scaling = revision.fresh_seeds(revision.SCALING_SEED_NAMESPACE, 8)
    old = revision.legacy_seeds()
    assert not (set(parity) & old)
    assert not (set(scaling) & old)
    assert not (set(parity) & set(scaling))
    assert len(set(parity)) == len(parity)
    assert len(set(scaling)) == len(scaling)


def test_paper_parity_refit_is_overdetermined_and_config_is_explicit():
    preset = revision.PARITY_PRESETS["paper"]
    protocol = revision.parity_protocol(preset)
    assert preset.n_virtual == 15
    assert preset.refit_rows > preset.n_features
    assert protocol["refit_rows_exceed_features"] is True
    assert protocol["features_including_bias"] == 676
    assert protocol["refit_rows"] == 750
    assert protocol["disjoint_from_definitive_seed_pool"] is True
    assert len(protocol["methods"]) == 6
    assert protocol["feature_filter"]["threshold"] == revision.FEATURE_STD_TOL


def test_train_only_variance_filter_removes_roundoff_and_ignores_holdout():
    rng = np.random.default_rng(7)
    varying = rng.normal(size=20)
    x_train = np.column_stack(
        [
            varying,
            np.full(20, 1e-15) + 1e-16 * rng.normal(size=20),
            np.ones(20),
        ]
    )
    x_validation = np.column_stack(
        [
            rng.normal(size=8),
            rng.normal(size=8),  # large holdout variation must not rescue column
            np.ones(8),
        ]
    )
    x_test = np.column_stack(
        [rng.normal(size=9), rng.normal(size=9), np.ones(9)]
    )
    train, validation, test, metadata = revision.train_only_variance_filter(
        x_train, x_validation, x_test
    )
    assert train.shape == (20, 2)
    assert validation.shape == (8, 2)
    assert test.shape == (9, 2)
    assert metadata["retained_nonbias_indices"] == [0]
    assert metadata["dropped_nonbias_indices"] == [1]
    assert metadata["training_std_max_dropped"] < revision.FEATURE_STD_TOL
    assert np.array_equal(validation[:, 0], x_validation[:, 0])
    assert np.all(validation[:, -1] == 1.0)


def test_scaling_split_is_disjoint_and_targets_are_defined():
    preset = revision.SCALING_PRESETS["paper"]
    indices = revision._scaling_split_indices(preset)
    assert not set(indices["train"]) & set(indices["validation"])
    assert not set(indices["train"]) & set(indices["test"])
    assert not set(indices["validation"]) & set(indices["test"])
    assert indices["train"][0] >= max(preset.delays)
    assert indices["test"][-1] < (
        preset.train + preset.validation + preset.test
    )


def test_ridge_selection_uses_validation_and_refit_uses_more_rows():
    rng = np.random.default_rng(91)
    x_train = rng.normal(size=(40, 12))
    x_validation = rng.normal(size=(20, 12))
    x_test = rng.normal(size=(25, 12))
    coefficients = rng.normal(size=(12, 2))
    y_train = x_train @ coefficients + 0.05 * rng.normal(size=(40, 2))
    y_validation = (
        x_validation @ coefficients + 0.05 * rng.normal(size=(20, 2))
    )
    y_test = x_test @ coefficients + 0.05 * rng.normal(size=(25, 2))
    selected, totals, per_target = revision.select_ridge(
        x_train,
        y_train,
        x_validation,
        y_validation,
        revision.RIDGES,
        metric="capacity",
    )
    assert selected in revision.RIDGES
    assert len(totals) == len(revision.RIDGES)
    assert all(len(values) == 2 for values in per_target.values())
    total, by_target = revision.refit_and_test(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        selected,
        metric="capacity",
    )
    assert total == pytest.approx(sum(by_target))
    assert len(by_target) == 2


def test_smoke_protocols_validate_without_heavy_execution():
    parity = revision.parity_protocol(revision.PARITY_PRESETS["smoke"])
    scaling = revision.scaling_protocol(
        revision.SCALING_PRESETS["smoke"], ("variance", "kac")
    )
    assert parity["split_design"].startswith("independent RNG")
    assert scaling["anchor"].endswith("N=5")
    assert scaling["schemes"] == ["variance", "kac"]


def test_ridge_boundary_audit_distinguishes_zero_from_rising_upper_edge():
    scores = {f"{ridge:.12g}": 1.0 for ridge in revision.RIDGES}
    zero_row = {
        "method": "CD_paper",
        "seed": 1,
        "selected_ridge": 0.0,
        "validation_capacity_by_ridge": scores,
    }
    rising = scores.copy()
    rising[f"{revision.RIDGES[-2]:.12g}"] = 1.0
    rising[f"{revision.RIDGES[-1]:.12g}"] = 2.0
    upper_row = {
        "method": "B3_collective",
        "seed": 2,
        "selected_ridge": revision.RIDGES[-1],
        "validation_capacity_by_ridge": rising,
    }
    audit = revision.ridge_boundary_audit([zero_row, upper_row])
    assert audit["n_selected_zero"] == 1
    assert audit["n_selected_maximum"] == 1
    assert audit["n_unresolved_upper"] == 1
    assert audit["upper_boundary_is_bracketed"] is False


def test_scaling_ridge_boundary_audit_respects_metric_direction():
    capacity = {f"{ridge:.12g}": 1.0 for ridge in revision.RIDGES}
    nmse = capacity.copy()
    capacity[f"{revision.RIDGES[-2]:.12g}"] = 1.0
    capacity[f"{revision.RIDGES[-1]:.12g}"] = 2.0
    nmse[f"{revision.RIDGES[-2]:.12g}"] = 1.0
    nmse[f"{revision.RIDGES[-1]:.12g}"] = 0.5
    row = {
        "scheme": "variance",
        "n_qubits": 5,
        "method": "CD_paper",
        "seed": 3,
        "stm": {
            "selected_ridge": revision.RIDGES[-1],
            "validation_by_ridge": capacity,
        },
        "narma10": {
            "selected_ridge": revision.RIDGES[-1],
            "validation_by_ridge": nmse,
        },
    }
    audit = revision.scaling_ridge_boundary_audit([row])
    assert audit["n_selected_maximum"] == 2
    assert audit["n_unresolved_upper"] == 2
    assert {
        item["direction"] for item in audit["unresolved_upper"]
    } == {"maximize", "minimize"}


def test_production_scaling_invariant_audit_checks_paired_hashes():
    preset = revision.SCALING_PRESETS["paper"]
    protocol = revision.scaling_protocol(preset, ("variance",))
    rows = []
    for n_qubits in preset.n_values:
        for seed in protocol["seeds"]:
            _, metadata = revision.scaled_couplings(
                seed, n_qubits, "variance"
            )
            for method in revision.SCALING_METHODS:
                rows.append(
                    {
                        "scheme": "variance",
                        "n_qubits": n_qubits,
                        "method": method,
                        "seed": seed,
                        "input_sha256": f"input-{seed}",
                        **metadata,
                        "relative_budget_error": 0.0,
                        "backend": "exact_sparse_expm_multiply",
                    }
                )
    audit = revision.scaling_invariant_audit(
        rows, protocol, preset, ("variance",)
    )
    assert audit["production_contract_applies"] is True
    assert audit["all_passed"] is True
    rows[1]["input_sha256"] = "wrong"
    broken = revision.scaling_invariant_audit(
        rows, protocol, preset, ("variance",)
    )
    assert broken["all_passed"] is False
    assert broken["checks"]["paired_hashes_equal"] is False

"""Focused protocol tests for the prospective activity-matched response."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_activity_matched_response as response  # noqa: E402

from qrc.liouvillian import vec  # noqa: E402
from qrc import readout, tasks  # noqa: E402


def test_protocol_is_lean_fixed_and_staged():
    assert (
        response.PROTOCOL_VERSION
        == "activity-matched-response-v3-2026-07-25"
    )
    assert response.N_QUBITS == 5
    assert (response.H, response.DT) == (0.5, 0.5)
    assert (response.WASH, response.TRAIN, response.TEST) == (200, 600, 400)
    assert response.FIXED_RIDGE == 1e-8
    assert response.DELAYS == tuple(range(1, 21))
    assert (response.CAL_WASH, response.CAL_PREFIX, response.CAL_MEASURE) == (
        200,
        600,
        400,
    )
    assert response.BRANCHES["local"]["lower_rate"] == 0.05
    assert response.BRANCHES["local"]["anchor_rate"] == 0.25
    assert response.BRANCHES["local"]["upper_rate"] == 0.5
    assert response.BRANCHES["local"]["activity_orientation"] == "increasing"
    assert response.BRANCHES["collective"]["lower_rate"] == 4.0
    assert response.BRANCHES["collective"]["anchor_rate"] == 8.0
    assert response.BRANCHES["collective"]["upper_rate"] == 32.0
    assert (
        response.BRANCHES["collective"]["activity_orientation"]
        == "decreasing"
    )
    expected_rows = response.N_PILOT_SEEDS * sum(
        len(response.PILOT_RATE_GRIDS[design])
        for design in response.DESIGNS
    )
    assert response.build_pilot_protocol()["expected_pilot_rows"] == (
        expected_rows
    )


def test_new_seed_pools_are_pairwise_disjoint_from_canonical_prior_pools():
    ledger = response.seed_ledger()
    prior = set(ledger["prior_seeds"])
    pilot = set(ledger["pilot_seeds"])
    task = set(ledger["task_seeds"])
    assert len(pilot) == 8
    assert len(task) == 24
    assert not prior & pilot
    assert not prior & task
    assert not pilot & task
    assert ledger["pilot_task_overlap"] == []
    assert response.DEVELOPMENT_PILOT_NAMESPACE == 501
    assert response.DEVELOPMENT_UNUSED_TASK_NAMESPACE == 502
    assert response.V2_PILOT_NAMESPACE == 503
    assert response.V2_TASK_NAMESPACE == 504
    assert response.PILOT_NAMESPACE == 505
    assert response.TASK_NAMESPACE == 506
    assert ledger["pilot_namespace"] == 505
    assert ledger["task_namespace"] == 506

    prior_sources = ledger["prior_sources"]
    assert prior_sources["activity_v2_pilot"] == {
        "count": 8,
        "sha256": (
            "e7de08e3669e2b722fca0e8c98c0472222f2c48c8146f6d376500e66728831b8"
        ),
    }
    assert prior_sources["activity_v2_task"] == {
        "count": 24,
        "sha256": (
            "172abb78c16ac2200b7645c2460959fde17d575da89209b0635110ddb2c940ac"
        ),
    }


def test_manifest_discloses_the_single_response_blind_recovery():
    protocol = response.build_pilot_protocol()
    boundary = protocol["branch_development_and_recovery_boundary"]
    v1 = boundary["v1_development"]
    v2 = boundary["v2_failed_reachability"]
    v3 = boundary["v3_single_recovery"]

    assert v1["collective_branch"] == [2.0, 16.0]
    assert "low-rate collective turnover" in v1["observed_boundary"]
    assert v1["audit_report"].endswith(
        "activity_matched_response_development_audit.md"
    )
    assert not v1["constructed_or_scored_task_targets"]

    assert v2["collective_branch"] == [4.0, 16.0]
    assert v2["fresh_calibration_cells"] == 240
    assert v2["matched_cells"] == 235
    assert v2["censored_collective_cells"] == 5
    assert not v2["frozen_calibration_created"]
    assert v2["task_score_checkpoints_created"] == 0
    assert not v2["task_scores_computed_or_inspected"]
    assert v2["audit_report"].endswith(
        "activity_matched_response_v2_failure_audit.md"
    )

    assert v3["local_branch"] == [0.05, 0.5]
    assert v3["collective_branch"] == [4.0, 32.0]
    assert v3["collective_upper_bound_extensions_permitted"] == 1
    assert v3["pilot_namespace"] == 505
    assert v3["task_namespace"] == 506
    assert not v3["reuses_v2_pilot_or_task_seeds"]
    assert not v3["manual_target_adjustment_permitted"]
    assert v3["target_freeze_and_inference_rules_unchanged"]
    assert not v3["further_recovery_after_gate_failure_permitted"]


def test_augmented_integrator_counts_constant_identity_activity():
    rho = np.array([[0.75, 0.0], [0.0, 0.25]], dtype=complex)
    state = vec(rho)
    generator = sparse.csr_matrix((4, 4), dtype=complex)
    functional = response.activity_functional(2.5 * np.eye(2))
    evolved, count, residue = response.integrated_activity_step(
        generator, state, functional, 0.4
    )
    assert np.max(np.abs(evolved - state)) < 1e-14
    assert count == pytest.approx(1.0, abs=1e-13)
    assert residue < 1e-14


@pytest.mark.parametrize(
    ("orientation", "function", "expected"),
    [
        ("increasing", lambda rate: 1.0 + 2.0 * rate, 2.0),
        ("decreasing", lambda rate: 3.0 - rate, 2.0),
    ],
)
def test_branch_local_bisection_handles_both_orientations(
    orientation, function, expected
):
    result = response.bisect_activity_target(
        function,
        expected,
        0.25,
        1.0,
        orientation,
        relative_tolerance=1e-6,
        absolute_tolerance=1e-8,
        maximum_iterations=30,
    )
    assert result["status"] == "matched"
    assert result["matched_activity"] == pytest.approx(expected, rel=1e-6)
    assert result["monotonicity"]["passed"]


def test_unreachable_calibration_is_censored_not_extrapolated():
    result = response.bisect_activity_target(
        lambda rate: 1.0 + rate,
        target=5.0,
        lower_rate=0.1,
        upper_rate=0.5,
        orientation="increasing",
    )
    assert result["status"] == "censored_target_unreachable"
    assert "matched_rate" not in result


def _pilot_rows():
    rows = []
    for design in response.DESIGNS:
        orientation = response.BRANCHES[design]["activity_orientation"]
        for seed_index, seed in enumerate(response.seed_ledger()["pilot_seeds"]):
            for rate in response.PILOT_RATE_GRIDS[design]:
                if orientation == "increasing":
                    activity = 0.5 + 3.0 * rate + seed_index * 0.002
                else:
                    activity = 3.3 - 0.08 * rate + seed_index * 0.002
                rows.append(
                    {
                        "design": design,
                        "seed": seed,
                        "rate": rate,
                        "activity": activity,
                    }
                )
    return rows


def test_pilot_freeze_uses_common_range_and_geometric_targets():
    frozen = response.derive_frozen_targets(_pilot_rows(), "pilot-hash")
    targets = np.asarray(frozen["targets"])
    assert len(targets) == 5
    assert np.all(np.diff(targets) > 0)
    assert np.allclose(targets[1:] / targets[:-1], targets[1] / targets[0])
    common_low, common_high = frozen["common_activity_interval"]
    assert common_high / common_low >= 1.5
    assert common_low < targets[0] < targets[-1] < common_high
    assert not frozen["uses_supervised_task_information"]


def test_pilot_freeze_rejects_wrong_collective_orientation():
    rows = _pilot_rows()
    for row in rows:
        if row["design"] == "collective":
            row["activity"] = 1.0 + row["rate"]
    with pytest.raises(RuntimeError, match="not monotone"):
        response.derive_frozen_targets(rows, "pilot-hash")


def test_bonferroni_band_uses_predeclared_five_contrast_family():
    differences = np.linspace(0.8, 1.2, 24)
    result = response.bonferroni_paired_band(differences)
    expected_critical = response.student_t.ppf(0.995, 23)
    assert result["n"] == 24
    assert result["family_size"] == 5
    assert result["critical_value"] == pytest.approx(expected_critical)
    assert result["simultaneous_lower"] > 0


def test_task_and_calibration_streams_are_deterministic_and_independent():
    seed = response.seed_ledger()["task_seeds"][0]
    couplings_a, calibration_a, task_a = response._stream_material(seed)
    couplings_b, calibration_b, task_b = response._stream_material(seed)
    assert np.array_equal(couplings_a, couplings_b)
    assert np.array_equal(calibration_a, calibration_b)
    assert np.array_equal(task_a, task_b)
    assert len(calibration_a) == 1200
    assert len(task_a) == 1200
    assert response.array_sha256(calibration_a) != response.array_sha256(task_a)


def test_cached_affine_generator_matches_direct_reservoir_builder():
    seed = response.seed_ledger()["pilot_seeds"][0]
    couplings, _, _ = response._stream_material(seed)
    engine = response.AffineActivityEngine(couplings, "collective")
    rate = 3.25
    input_value = 0.37
    cached_base, _ = engine.affine_terms(rate)
    direct, _ = response.build_reservoir(couplings, "collective", rate)
    difference = cached_base + input_value * engine.drive - direct.liouvillian(
        input_value
    )
    assert (
        np.max(np.abs(difference.data)) if difference.nnz else 0.0
    ) < 1e-13


def test_cached_calibration_matches_stepwise_integrator(monkeypatch):
    monkeypatch.setattr(response, "CAL_WASH", 2)
    monkeypatch.setattr(response, "CAL_PREFIX", 3)
    monkeypatch.setattr(response, "CAL_MEASURE", 4)
    seed = response.seed_ledger()["pilot_seeds"][0]
    couplings, full_inputs, _ = response._stream_material(seed)
    inputs = full_inputs[:9]
    rate = 0.2

    cached = response.AffineActivityEngine(
        couplings, "local"
    ).calibration_activity(inputs, rate)

    reservoir, jumps = response.build_reservoir(couplings, "local", rate)
    rho = reservoir.initial_state()
    for value in inputs[:5]:
        rho = reservoir.step(rho, float(value))
    functional = response.activity_functional(
        response.jump_rate_operator(jumps)
    )
    state = vec(rho)
    total = 0.0
    for value in inputs[5:]:
        state, count, _ = response.integrated_activity_step(
            reservoir.liouvillian(float(value)),
            state,
            functional,
            response.DT,
        )
        total += count
    direct_activity = total / (4 * response.DT)
    assert cached["activity"] == pytest.approx(direct_activity, rel=2e-11)


def test_fused_task_trajectory_matches_stepwise_features_and_activity(
    monkeypatch,
):
    monkeypatch.setattr(response, "WASH", 2)
    monkeypatch.setattr(response, "TRAIN", 4)
    monkeypatch.setattr(response, "TEST", 3)
    seed = response.seed_ledger()["pilot_seeds"][1]
    couplings, _, task_inputs = response._stream_material(seed)
    inputs = task_inputs[:9]
    rate = 0.2

    fused_features, fused_activity = response.AffineActivityEngine(
        couplings, "local"
    ).task_trajectory(inputs, rate)

    reservoir, jumps = response.build_reservoir(couplings, "local", rate)
    observables = readout.pauli_observables(
        response.N_QUBITS, max_weight=2
    )
    matrices = np.stack([observable.matrix for observable in observables])
    direct_features = np.empty((7, len(observables)))
    rho = reservoir.initial_state()
    functional = response.activity_functional(
        response.jump_rate_operator(jumps)
    )
    total = 0.0
    for index, value in enumerate(inputs):
        if index < 6:
            rho = reservoir.step(rho, float(value))
        else:
            state, count, _ = response.integrated_activity_step(
                reservoir.liouvillian(float(value)),
                vec(rho),
                functional,
                response.DT,
            )
            rho = response.unvec(state, 2 ** response.N_QUBITS)
            total += count
        if index >= 2:
            direct_features[index - 2] = np.real(
                np.einsum("kij,ji->k", matrices, rho)
            )
    assert np.max(np.abs(fused_features - direct_features)) < 2e-11
    assert fused_activity["time_averaged_test_activity"] == pytest.approx(
        total / (3 * response.DT), rel=2e-11
    )


def test_multi_rhs_stm_matches_scalar_delay_loop(monkeypatch):
    monkeypatch.setattr(response, "WASH", 20)
    monkeypatch.setattr(response, "TRAIN", 30)
    monkeypatch.setattr(response, "TEST", 15)
    monkeypatch.setattr(response, "DELAYS", (1, 2, 3))
    rng = np.random.default_rng(91)
    features = rng.normal(size=(45, 8))
    inputs = tasks.stm_inputs(65, rng)

    total, capacities = response._stm_score(features, inputs)
    x_bias = readout.add_bias(features)
    expected = []
    for delay in response.DELAYS:
        target = tasks.delayed_target(inputs, delay)[20:]
        weights = readout.train_readout(
            x_bias[:30], target[:30], ridge=response.FIXED_RIDGE
        )
        prediction = readout.predict(x_bias[30:], weights)
        expected.append(readout.capacity(target[30:], prediction))
    assert capacities == pytest.approx(expected, rel=2e-11, abs=2e-12)
    assert total == pytest.approx(sum(expected), rel=2e-11, abs=2e-12)


def test_complete_negative_result_remains_complete_evidence():
    seeds = list(range(24))
    targets = [0.2, 0.25, 0.3, 0.35, 0.4]
    protocol_sha = "a" * 64
    frozen_sha = "b" * 64
    manifest = {
        "protocol_sha256": protocol_sha,
        "protocol": {
            "seed_ledger": {"task_seeds": seeds},
            "frozen_targets": {"targets": targets},
        },
    }
    cells = []
    rows = []
    for design in response.DESIGNS:
        for seed in seeds:
            coupling_hash = f"coupling-{seed}"
            calibration_hash = f"calibration-{seed}"
            task_hash = f"task-{seed}"
            for target_index, target in enumerate(targets):
                rate = (
                    0.1 + 0.02 * target_index
                    if design == "local"
                    else 12.0 - target_index
                )
                calibration_row_hash = (
                    f"calibration-row-{design}-{seed}-{target_index}"
                )
                cells.append(
                    {
                        "design": design,
                        "seed": seed,
                        "target_index": target_index,
                        "target_activity": target,
                        "matched_rate": rate,
                        "calibration_row_sha256": calibration_row_hash,
                        "couplings_sha256": coupling_hash,
                        "calibration_input_sha256": calibration_hash,
                    }
                )
                local_score = 8.0 + 0.01 * seed
                score = (
                    local_score
                    if design == "local"
                    else local_score
                    + (-0.2 if target_index == 0 else 0.5)
                )
                rows.append(
                    {
                        "design": design,
                        "seed": seed,
                        "target_index": target_index,
                        "target_activity": target,
                        "frozen_rate": rate,
                        "task_protocol_sha256": protocol_sha,
                        "frozen_calibration_sha256": frozen_sha,
                        "calibration_row_sha256": calibration_row_hash,
                        "couplings_sha256": coupling_hash,
                        "calibration_input_sha256": calibration_hash,
                        "task_input_sha256": task_hash,
                        "task_stream_is_independent_of_calibration_stream": True,
                        "ridge": response.FIXED_RIDGE,
                        "test_stm_capacity": score,
                        "test_capacity_by_delay": [score / 20.0] * 20,
                        "time_averaged_test_activity": target,
                        "maximum_trace_error": 0.0,
                        "maximum_activity_imaginary_residue": 0.0,
                        "minimum_test_interval_integrated_activity": 0.0,
                    }
                )
    aggregate = response.build_aggregate(
        rows,
        manifest,
        {"censored_cells": 0, "cells": cells},
        frozen_sha,
    )
    assert aggregate["status"] == "complete"
    assert aggregate["invariant_audit"]["passed"]
    assert not aggregate["claim_gates"][
        "simultaneous_stm_dominance_all_targets"
    ]
    assert not aggregate["claim_gates"][
        "activity_matched_dominance_claim_allowed"
    ]


def test_pilot_manifest_emits_exact_driver_source_snapshot(tmp_path):
    manifest, digest = response.ensure_pilot_manifest(tmp_path)
    snapshot = tmp_path / "source_snapshot" / "run_activity_matched_response.py"
    snapshot_manifest = tmp_path / "source_snapshot" / "manifest.json"
    assert snapshot.is_file()
    assert snapshot_manifest.is_file()
    assert response.file_sha256(snapshot) == manifest["protocol"][
        "scientific_sources_sha256"
    ]["experiments/run_activity_matched_response.py"]
    assert response.json.loads(snapshot_manifest.read_text())[
        "pilot_protocol_sha256"
    ] == digest

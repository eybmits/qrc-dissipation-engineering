"""Contract tests for the frozen Experiment-1 finite-size driver."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_experiment1_finite_size as scaling  # noqa: E402
import run_revision_controls as revision  # noqa: E402
from qrc import dissipators as dsp  # noqa: E402


def test_production_manifest_is_exact_fresh_and_stable():
    protocol = scaling.build_protocol()
    assert protocol["n_values"] == [4, 5, 6, 7, 8]
    assert protocol["methods"] == list(scaling.METHODS)
    assert protocol["n_lineages"] == 24
    assert protocol["n_jobs"] == 960
    assert protocol["seeds_are_fresh"] is True
    assert len(protocol["seeds"]) == len(set(protocol["seeds"])) == 24
    jobs = scaling.all_jobs(protocol)
    assert len(jobs) == len({job.key for job in jobs}) == 960
    assert scaling.protocol_sha256(protocol) == scaling.protocol_sha256(
        scaling.build_protocol()
    )
    paths_key = "experiments/_paths.py"
    assert paths_key in protocol["source_environment"]["files"]
    assert (
        protocol["source_environment"]["files"][paths_key]
        == scaling.file_sha256(scaling.ROOT / paths_key)
    )
    assert "src/qrc/liouvillian.py" in protocol["source_environment"]["files"]


def test_seed_namespace_is_disjoint_from_every_declared_sealed_pool():
    seeds = set(scaling.production_seeds())
    assert scaling.SEED_NAMESPACE == 2026072902
    assert (
        scaling.excluded_seed_pools()["superseded_v1_smoke_protocol"]
        == list(scaling.SUPERSEDED_V1_SEEDS)
    )
    assert 969779107 in scaling.SUPERSEDED_V1_SEEDS
    for pool in scaling.excluded_seed_pools().values():
        assert not seeds.intersection(pool)


def test_nested_variance_normalisation_and_jump_targets():
    seed = scaling.production_seeds(1)[0]
    full = revision.nested_base_couplings(seed, 8)
    expected_targets = {4: 32.0, 5: 80.0, 6: 192.0, 7: 448.0, 8: 1024.0}
    for n_qubits in scaling.N_VALUES:
        base = revision.nested_base_couplings(seed, n_qubits)
        couplings, metadata = revision.scaled_couplings(
            seed, n_qubits, scaling.COUPLING_SCHEME
        )
        multiplier = np.sqrt(4.0 / (n_qubits - 1))
        assert np.array_equal(base, full[:n_qubits, :n_qubits])
        assert metadata["coupling_multiplier"] == pytest.approx(multiplier)
        assert np.allclose(couplings, multiplier * base)
        target = dsp.jump_strength(dsp.local_loss(n_qubits, scaling.GAMMA))
        assert target == pytest.approx(expected_targets[n_qubits])


def test_readout_rule_and_split_match_experiment1():
    protocol = scaling.build_protocol()
    assert protocol["split"] == {
        "wash": 200,
        "primary_fit": 600,
        "validation_control_train": 450,
        "validation_control_validation": 150,
        "test": 400,
        "total_inputs": 1200,
        "test_is_untouched": True,
    }
    assert protocol["readout"]["primary"]["ridge"] == 1e-8
    assert protocol["readout"]["observable_counts"] == {
        "4": 30,
        "5": 45,
        "6": 63,
        "7": 84,
        "8": 108,
    }
    assert "62 two-sided exact" in protocol["prespecified_analysis"][
        "confirmatory_family"
    ]


def test_execution_filters_slice_but_do_not_change_protocol():
    protocol = scaling.build_protocol()
    selected = scaling.selected_jobs(
        protocol,
        n_values=(8,),
        methods=("CD_paper", "B3_collective"),
        seed_indices=(0,),
    )
    assert len(selected) == 2
    assert {job.n_qubits for job in selected} == {8}
    assert {job.method for job in selected} == {
        "CD_paper",
        "B3_collective",
    }
    assert {job.seed for job in selected} == {protocol["seeds"][0]}
    assert protocol["n_jobs"] == 960
    with pytest.raises(ValueError, match="outside"):
        scaling.selected_jobs(protocol, n_values=(9,))
    with pytest.raises(ValueError, match="outside"):
        scaling.selected_jobs(protocol, seed_indices=(24,))


def _synthetic_checkpoint(job: scaling.Job, protocol: dict) -> dict:
    couplings, coupling_meta = revision.scaled_couplings(
        job.seed, job.n_qubits, scaling.COUPLING_SCHEME
    )
    is_fn = job.method == "FN"
    jumps, target_strength, actual_strength = scaling._jump_family_for_job(
        job, couplings
    )
    inputs = scaling._input_sequence(job.seed)
    post_wash = inputs[scaling.WASH:]
    targets = {
        "stm": np.column_stack(
            [
                scaling.tasks.delayed_target(post_wash, delay)
                for delay in scaling.STM_DELAYS
            ]
        ),
        "narma10": scaling.tasks.narma_target(
            post_wash,
            order=scaling.NARMA_ORDER,
            input_scale=scaling.NARMA_INPUT_SCALE,
        )[:, None],
    }

    def result(task_name):
        n_targets = len(scaling.STM_DELAYS) if task_name == "stm" else 1
        metric = "capacity" if task_name == "stm" else "nmse"
        aggregate = float(n_targets) if task_name == "stm" else 1.0
        primary = {
            "metric": metric,
            "ridge": scaling.FIXED_RIDGE,
            "test": aggregate,
            "test_by_target": [1.0] * n_targets,
            "effective_fit_rows_by_target": (
                [
                    scaling.FIT_TOTAL - delay
                    for delay in scaling.STM_DELAYS
                ]
                if task_name == "stm"
                else [scaling.FIT_TOTAL - scaling.NARMA_ORDER]
            ),
            "test_rows": scaling.TEST,
        }
        control = {
            "metric": metric,
            "selected_ridge": scaling.FIXED_RIDGE,
            "selected_test": aggregate,
            "selected_test_by_target": [1.0] * n_targets,
            "fixed_ridge": scaling.FIXED_RIDGE,
            "fixed_test": aggregate,
            "fixed_test_by_target": [1.0] * n_targets,
        }
        if task_name == "stm":
            control["effective_train_rows_by_target"] = [
                scaling.TRAIN - delay for delay in scaling.STM_DELAYS
            ]
            control["effective_refit_rows_by_target"] = [
                scaling.FIT_TOTAL - delay for delay in scaling.STM_DELAYS
            ]
        return {
            "primary_fixed": primary,
            "validation_control": control,
        }

    raw_features = scaling.expected_observable_count(job.n_qubits)
    row = {
        "status": "complete",
        "protocol_version": scaling.PROTOCOL_VERSION,
        "protocol_sha256": scaling.protocol_sha256(protocol),
        "source_environment_sha256": protocol[
            "source_environment_sha256"
        ],
        "n_qubits": job.n_qubits,
        "method": job.method,
        "seed": job.seed,
        "h": scaling.H,
        "dt": scaling.DT,
        "wash": scaling.WASH,
        "fit": scaling.FIT_TOTAL,
        "test": scaling.TEST,
        "backend": (
            "exact_reset_unitary"
            if is_fn
            else "exact_sparse_expm_multiply"
        ),
        "n_observables": raw_features,
        "n_features_including_bias": raw_features + 1,
        "full_input_sha256": scaling.array_sha256(inputs),
        "post_wash_input_sha256": scaling.array_sha256(post_wash),
        "target_sha256": {
            name: scaling.array_sha256(value)
            for name, value in targets.items()
        },
        **coupling_meta,
        "target_jump_strength": target_strength,
        "actual_jump_strength": actual_strength,
        "relative_jump_budget_error": None if is_fn else 0.0,
        "jump_family_sha256": (
            None
            if is_fn
            else scaling.primary_readout.jump_family_sha256(jumps)
        ),
        "feature_guard": {
            "fit_on": "training rows only",
            "threshold": scaling.FEATURE_STD_TOL,
            "retained_nonbias_features": raw_features,
            "dropped_nonbias_features": 0,
        },
        "task_results": {
            task_name: result(task_name)
            for task_name in ("stm", "narma10")
        },
        "runtime_seconds": 0.1,
    }
    return scaling.seal_checkpoint(row)


def test_valid_checkpoint_skips_and_protocol_drift_is_fatal(tmp_path):
    protocol = scaling.build_protocol()
    job = scaling.all_jobs(protocol)[0]
    path = scaling.job_path(tmp_path, job)
    row = _synthetic_checkpoint(job, protocol)
    scaling.atomic_write_json(path, row)
    assert scaling._validate_checkpoint(path, job, protocol) == row
    result = scaling._run_and_write(job, protocol, str(tmp_path))
    assert result["status"] == "skip"

    changed = json.loads(json.dumps(row))
    changed["protocol_sha256"] = "wrong"
    changed = scaling.seal_checkpoint(changed)
    scaling.atomic_write_json(path, changed)
    with pytest.raises(scaling.CheckpointError, match="mismatched"):
        scaling._validate_checkpoint(path, job, protocol)


def test_protocol_must_be_explicitly_frozen_and_run_lock_is_exclusive(
    tmp_path,
):
    protocol = scaling.build_protocol()
    with pytest.raises(scaling.CheckpointError, match="freeze"):
        scaling.write_or_validate_protocol(tmp_path, protocol)
    scaling.write_or_validate_protocol(tmp_path, protocol, create=True)
    scaling.write_or_validate_protocol(tmp_path, protocol)
    with scaling.exclusive_run_lock(tmp_path, protocol):
        with pytest.raises(scaling.CheckpointError, match="another"):
            with scaling.exclusive_run_lock(tmp_path, protocol):
                pass


def test_dissipative_checkpoint_strength_and_hash_contract(tmp_path):
    protocol = scaling.build_protocol()
    job = next(
        job
        for job in scaling.all_jobs(protocol)
        if job.method == "A1_heterogeneous"
    )
    path = scaling.job_path(tmp_path, job)
    row = _synthetic_checkpoint(job, protocol)
    scaling.atomic_write_json(path, row)
    assert scaling._validate_checkpoint(path, job, protocol) == row

    changed = dict(row)
    changed["actual_jump_strength"] += 1.0
    changed = scaling.seal_checkpoint(changed)
    scaling.atomic_write_json(path, changed)
    with pytest.raises(scaling.CheckpointError, match="actual jump"):
        scaling._validate_checkpoint(path, job, protocol)

    changed = dict(row)
    changed["jump_family_sha256"] = "a" * 64
    changed = scaling.seal_checkpoint(changed)
    scaling.atomic_write_json(path, changed)
    with pytest.raises(scaling.CheckpointError, match="jump-family hash"):
        scaling._validate_checkpoint(path, job, protocol)


@pytest.mark.parametrize(
    ("section", "scalar"),
    (
        ("primary_fixed", "test"),
        ("validation_control", "selected_test"),
        ("validation_control", "fixed_test"),
    ),
)
def test_stm_totals_must_match_per_delay_scores(
    tmp_path, section, scalar
):
    protocol = scaling.build_protocol()
    job = scaling.all_jobs(protocol)[0]
    path = scaling.job_path(tmp_path, job)
    row = _synthetic_checkpoint(job, protocol)
    row["task_results"]["stm"][section][scalar] += 1.0
    row = scaling.seal_checkpoint(row)
    scaling.atomic_write_json(path, row)
    with pytest.raises(scaling.CheckpointError, match="inconsistent STM"):
        scaling._validate_checkpoint(path, job, protocol)


def test_corrupt_or_nonfinite_checkpoint_is_never_silently_overwritten(
    tmp_path,
):
    protocol = scaling.build_protocol()
    job = scaling.all_jobs(protocol)[0]
    path = scaling.job_path(tmp_path, job)
    path.parent.mkdir(parents=True)
    path.write_text("{truncated")
    with pytest.raises(scaling.CheckpointError, match="corrupt"):
        scaling._validate_checkpoint(path, job, protocol)

    path.unlink()
    row = _synthetic_checkpoint(job, protocol)
    row["task_results"]["stm"]["primary_fixed"]["test_by_target"] = [1.0]
    row = scaling.seal_checkpoint(row)
    path.write_text(json.dumps(row))
    with pytest.raises(scaling.CheckpointError, match="test_by_target"):
        scaling._validate_checkpoint(path, job, protocol)


def test_exact_sign_flip_uses_meet_in_the_middle_for_24_pairs():
    values = np.ones(24)
    assert scaling._sign_flip_pvalue(values, seed=1) == pytest.approx(
        2.0 / (1 << 24)
    )


def test_fixed_primary_score_preserves_target_specific_masks():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(scaling.FIT_TOTAL + scaling.TEST, 6))
    inputs = rng.uniform(size=scaling.FIT_TOTAL + scaling.TEST)
    targets = np.column_stack(
        [tasks for tasks in (
            np.concatenate((np.full(delay, np.nan), inputs[:-delay]))
            for delay in (1, 2)
        )]
    )
    result = scaling._primary_fixed_score(
        features, targets, metric="capacity"
    )
    assert result["effective_fit_rows_by_target"] == [599, 598]
    assert len(result["test_by_target"]) == 2
    assert np.isfinite(result["test"])


def test_aggregate_has_exact_60_cell_plus_endpoint_family(
    tmp_path, monkeypatch
):
    protocol = scaling.build_protocol()
    method_index = {
        method: index for index, method in enumerate(protocol["methods"])
    }
    seed_index = {
        int(seed): index for index, seed in enumerate(protocol["seeds"])
    }

    def fake_checkpoint(path, job, frozen):
        del path, frozen
        baseline_stm = 8.0 + 0.2 * job.n_qubits + 1e-3 * seed_index[job.seed]
        baseline_narma = 0.5 - 0.01 * job.n_qubits + 1e-5 * seed_index[job.seed]
        if job.method == scaling.REFERENCE_METHOD:
            stm = baseline_stm
            narma = baseline_narma
        else:
            scale = 0.02 * method_index[job.method]
            stm = baseline_stm + scale * (job.n_qubits - 3)
            narma = baseline_narma - 0.002 * method_index[job.method]
        return {
            "n_qubits": job.n_qubits,
            "method": job.method,
            "seed": job.seed,
            "runtime_seconds": 1.0,
            "task_results": {
                "stm": {
                    "primary_fixed": {"test": stm},
                    "validation_control": {
                        "selected_test": stm + 1e-4,
                    },
                },
                "narma10": {
                    "primary_fixed": {"test": narma},
                    "validation_control": {
                        "selected_test": narma - 1e-6,
                    },
                },
            },
        }

    monkeypatch.setattr(scaling, "_validate_checkpoint", fake_checkpoint)
    monkeypatch.setattr(
        scaling,
        "invariant_audit",
        lambda rows, frozen: {"passed": True, "errors": []},
    )
    scaling.write_or_validate_protocol(tmp_path, protocol, create=True)
    payload = scaling.aggregate(tmp_path, protocol)
    cells = payload["confirmatory_dissipative_cells"]
    assert payload["confirmatory_family_size"] == 62
    assert len(cells) == 60
    assert all("/FN" not in key for key in cells)
    assert all("holm_pvalue" in effect for effect in cells.values())
    endpoint = payload["finite_range_endpoint_contrast"]
    assert endpoint["mean"] > 0
    assert "holm_pvalue" in endpoint
    slope = payload["finite_range_slope_contrast"]
    assert slope["mean"] > 0
    assert "holm_pvalue" in slope

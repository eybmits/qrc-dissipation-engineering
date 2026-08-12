"""Tests for the prospective local-to-collective interpolation protocol."""

from __future__ import annotations

import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from qrc import dissipators as dsp
from qrc.liouvillian import dissipator_super

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_quantum_strengthening as strengthening  # noqa: E402


def _dissipative_super(jumps):
    first = np.asarray(jumps[0][0])
    total = np.zeros((first.size, first.size), dtype=complex)
    for operator, rate in jumps:
        total += rate * dissipator_super(operator)
    return total


def _synthetic_frozen(outdir: Path, preset: strengthening.Preset) -> dict:
    protocol = strengthening.protocol_dict(preset)
    protocol_hash = strengthening._sha256_json(protocol)
    source_hash = strengthening._sha256_json(protocol["source_environment"])
    rows = []
    predictions = {}
    for n_qubits in preset.n_qubits:
        summary = []
        for alpha in strengthening.ALPHAS:
            summary.append(
                {
                    "alpha": alpha,
                    "spectral_gap_mean": 1.0 - 0.8 * alpha,
                    "spectral_gap_se": 0.01,
                    "slow_mode_count_mean": alpha,
                    "slow_mode_count_se": 0.01,
                    "retained_mode_mass_mean": 1.0 + alpha,
                    "retained_mode_mass_se": 0.01,
                }
            )
            for seed in strengthening.deterministic_seeds(preset.n_seeds):
                target = float(2 ** (n_qubits - 1) * n_qubits)
                rows.append(
                    {
                        "N": n_qubits,
                        "alpha": alpha,
                        "seed": seed,
                        "jump_strength": target,
                        "target_strength": target,
                        "relative_budget_error": 0.0,
                        "unitality_defect": 1.0,
                        "spectral_gap": 1.0 - 0.8 * alpha,
                        "slow_mode_count": int(alpha == 1.0),
                        "retained_mode_mass": 1.0 + alpha,
                        "n_nonstationary_modes": 4,
                        "slow_decay_threshold": 0.4,
                        "max_positive_real_part": 0.0,
                        "runtime_s": 0.01,
                    }
                )
        predictions[str(n_qubits)] = {
            "diagnostic_summary": summary,
            "frozen_gap_rank_best_to_worst": list(
                reversed(strengthening.ALPHAS)
            ),
            "frozen_retained_mass_rank_best_to_worst": list(
                reversed(strengthening.ALPHAS)
            ),
            "diagnostic_selected_intermediate_alpha": 0.8,
            "primary_prediction": "synthetic",
            "secondary_prediction": "synthetic",
        }
    payload = {
        "artifact_type": "frozen_diagnostic_predictions",
        "status": "frozen_before_task_scores",
        "created_utc": "2026-07-23T00:00:00+00:00",
        "git_head": "test",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "source_environment_sha256": source_hash,
        "task_scores_present_at_freeze": False,
        "prediction_rule_was_fixed_before_scores": True,
        "git_dirty_status_at_freeze": [],
        "predictions_by_N": predictions,
        "success_criteria": {},
        "diagnostic_rows": rows,
    }
    path = strengthening._prediction_path(outdir)
    strengthening._atomic_write_json(path, payload)
    strengthening._prediction_seal_path(outdir).write_text(
        f"{strengthening._sha256_file(path)}  {path.name}\n"
    )
    return payload


def _synthetic_checkpoint(
    preset: strengthening.Preset,
    frozen: dict,
    frozen_hash: str,
    n_qubits: int,
    alpha: float,
    seed: int,
) -> dict:
    delay_values = [0.1 + 0.01 * alpha] * len(preset.delays)
    return {
        "N": n_qubits,
        "alpha": alpha,
        "seed": seed,
        "backend": "exact_sparse_expm_multiply",
        "jump_strength": 10.0,
        "target_strength": 10.0,
        "relative_budget_error": 0.0,
        "n_features": 3 * n_qubits + 3 * math.comb(n_qubits, 2),
        "selected_ridge": 1e-8,
        "validation_mc": float(sum(delay_values)),
        "test_mc": float(sum(delay_values) + alpha + (seed % 7) * 1e-4),
        "validation_capacity_by_delay": delay_values,
        "test_capacity_by_delay": delay_values,
        "validation_mc_by_ridge": {
            f"{ridge:.12g}": float(sum(delay_values) - ridge)
            for ridge in strengthening.RIDGES
        },
        "runtime_s": 0.01,
        "protocol_sha256": frozen["protocol_sha256"],
        "frozen_prediction_sha256": frozen_hash,
        "source_environment_sha256": frozen["source_environment_sha256"],
    }


def _write_complete_checkpoints(
    outdir: Path, preset: strengthening.Preset, frozen: dict
) -> None:
    frozen_hash = strengthening._sha256_file(
        strengthening._prediction_path(outdir)
    )
    strengthening._atomic_write_json(
        outdir / "task_stage_metadata.json",
        {
            "protocol_sha256": frozen["protocol_sha256"],
            "frozen_prediction_sha256": frozen_hash,
            "source_environment_sha256": frozen["source_environment_sha256"],
        },
    )
    for n_qubits in preset.n_qubits:
        for alpha in strengthening.ALPHAS:
            for seed in strengthening.deterministic_seeds(preset.n_seeds):
                row = _synthetic_checkpoint(
                    preset, frozen, frozen_hash, n_qubits, alpha, seed
                )
                strengthening._atomic_write_json(
                    strengthening._task_path(
                        outdir, n_qubits, alpha, seed
                    ),
                    row,
                )


def test_interpolation_preserves_budget_at_every_alpha():
    n_qubits = 3
    target = dsp.jump_strength(dsp.local_loss(n_qubits, 1.0))
    for alpha in strengthening.ALPHAS:
        jumps = strengthening.build_interpolated_jumps(
            n_qubits, alpha, target
        )
        assert np.isclose(dsp.jump_strength(jumps), target, rtol=1e-13)


def test_interpolation_is_direct_superoperator_convex_combination():
    n_qubits = 2
    target = dsp.jump_strength(dsp.local_loss(n_qubits, 1.0))
    local = dsp.normalize_jump_strength(
        dsp.local_loss(n_qubits, 1.0), target
    )
    collective = dsp.normalize_jump_strength(
        dsp.collective_loss(n_qubits, 1.0), target
    )
    local_super = _dissipative_super(local)
    collective_super = _dissipative_super(collective)
    for alpha in strengthening.ALPHAS:
        actual = _dissipative_super(
            strengthening.build_interpolated_jumps(
                n_qubits, alpha, target
            )
        )
        expected = (1.0 - alpha) * local_super + alpha * collective_super
        assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_interpolation_endpoints_reproduce_channel_superoperators():
    n_qubits = 2
    target = dsp.jump_strength(dsp.local_loss(n_qubits, 1.0))
    actual_local = _dissipative_super(
        strengthening.build_interpolated_jumps(n_qubits, 0.0, target)
    )
    actual_collective = _dissipative_super(
        strengthening.build_interpolated_jumps(n_qubits, 1.0, target)
    )
    expected_local = _dissipative_super(
        dsp.normalize_jump_strength(dsp.local_loss(n_qubits, 1.0), target)
    )
    expected_collective = _dissipative_super(
        dsp.normalize_jump_strength(
            dsp.collective_loss(n_qubits, 1.0), target
        )
    )
    assert np.allclose(actual_local, expected_local)
    assert np.allclose(actual_collective, expected_collective)


def test_train_validation_test_masks_are_disjoint_and_complete():
    preset = strengthening.PRESETS["smoke"]
    length = preset.train + preset.validation + preset.test
    masks = strengthening._split_masks(length, preset)
    assert sum(int(mask.sum()) for mask in masks.values()) == length
    assert not np.any(masks["train"] & masks["validation"])
    assert not np.any(masks["train"] & masks["test"])
    assert not np.any(masks["validation"] & masks["test"])


def test_test_targets_cannot_affect_ridge_selection():
    preset = strengthening.PRESETS["smoke"]
    rng = np.random.default_rng(9)
    inputs = rng.uniform(size=preset.total_len)
    features = rng.normal(
        size=(preset.train + preset.validation + preset.test, 8)
    )
    original = strengthening.held_out_stm_score(features, inputs, preset)
    mutated = inputs.copy()
    mutated[-(preset.test - max(preset.delays)) :] = rng.uniform(
        size=preset.test - max(preset.delays)
    )
    repeated = strengthening.held_out_stm_score(features, mutated, preset)
    assert repeated["selected_ridge"] == original["selected_ridge"]
    assert repeated["validation_mc_by_ridge"] == pytest.approx(
        original["validation_mc_by_ridge"]
    )


def test_slow_mode_threshold_is_fixed_by_task_horizon():
    eigenvalues = np.asarray([0.0, -0.05, -0.2, -1.0], dtype=complex)
    result = strengthening._spectrum_diagnostics(
        eigenvalues, dt=0.5, max_delay=20
    )
    assert np.isclose(result["slow_decay_threshold"], 0.1)
    assert result["slow_mode_count"] == 1
    assert np.isclose(result["spectral_gap"], 0.05)


def test_perfect_six_point_spearman_uses_exact_nonzero_p_value():
    result = strengthening._spearman_with_exact_p(
        [0, 1, 2, 3, 4, 5],
        [10, 11, 12, 13, 14, 15],
    )
    assert result["exact"] is True
    assert result["n_permutations"] == 720
    assert np.isclose(result["rho"], 1.0)
    assert np.isclose(result["p_two_sided"], 2 / 720)


def test_checkpoint_corruption_is_rejected_centrally(tmp_path):
    preset = strengthening.PRESETS["smoke"]
    frozen = _synthetic_frozen(tmp_path, preset)
    frozen_hash = strengthening._sha256_file(
        strengthening._prediction_path(tmp_path)
    )
    row = _synthetic_checkpoint(
        preset,
        frozen,
        frozen_hash,
        preset.n_qubits[0],
        strengthening.ALPHAS[0],
        strengthening.deterministic_seeds(preset.n_seeds)[0],
    )
    row["test_capacity_by_delay"] = [np.nan] * len(preset.delays)
    with pytest.raises(RuntimeError, match="not finite"):
        strengthening._validate_task_checkpoint(
            row,
            preset=preset,
            protocol_sha256=frozen["protocol_sha256"],
            frozen_sha256=frozen_hash,
            source_environment_sha256=frozen[
                "source_environment_sha256"
            ],
            n_qubits=preset.n_qubits[0],
            alpha=strengthening.ALPHAS[0],
            seed=strengthening.deterministic_seeds(preset.n_seeds)[0],
        )


def test_frozen_mutation_is_rejected_by_seal(tmp_path):
    preset = strengthening.PRESETS["smoke"]
    _synthetic_frozen(tmp_path, preset)
    frozen_path = strengthening._prediction_path(tmp_path)
    payload = json.loads(frozen_path.read_text())
    payload["predictions_by_N"][str(preset.n_qubits[0])][
        "diagnostic_selected_intermediate_alpha"
    ] = 0.6
    strengthening._atomic_write_json(frozen_path, payload)
    with pytest.raises(RuntimeError, match="changed after sealing"):
        strengthening._load_frozen(preset, tmp_path)


def test_missing_paired_checkpoint_is_rejected(tmp_path):
    preset = strengthening.PRESETS["smoke"]
    frozen = _synthetic_frozen(tmp_path, preset)
    _write_complete_checkpoints(tmp_path, preset, frozen)
    missing = strengthening._task_path(
        tmp_path,
        preset.n_qubits[0],
        strengthening.ALPHAS[-1],
        strengthening.deterministic_seeds(preset.n_seeds)[-1],
    )
    missing.unlink()
    with pytest.raises(RuntimeError, match="checkpoint.*missing"):
        strengthening._read_task_rows(
            preset,
            tmp_path,
            frozen,
            strengthening._sha256_file(
                strengthening._prediction_path(tmp_path)
            ),
        )


def test_archive_contains_complete_provenance_record(tmp_path):
    preset = strengthening.PRESETS["smoke"]
    outdir = tmp_path / "results"
    frozen = _synthetic_frozen(outdir, preset)
    _write_complete_checkpoints(outdir, preset, frozen)
    report = tmp_path / "report.md"
    archive = tmp_path / "record.zip"
    strengthening.archive_results(preset, outdir, report, archive)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        manifest_name = next(
            name for name in names if name.endswith("provenance_manifest.json")
        )
        manifest = json.loads(bundle.read(manifest_name))
    required_suffixes = {
        "experiments/run_quantum_strengthening.py",
        "tests/test_quantum_strengthening.py",
        "requirements.txt",
        "frozen_diagnostic_predictions.json",
        "quantum_strengthening_results.json",
        "report.md",
        "provenance_manifest.json",
    }
    for suffix in required_suffixes:
        assert any(name.endswith(suffix) for name in names), suffix
    assert manifest["source_environment"]["packages"]["numpy"]
    assert manifest["source_environment"]["packages"]["scipy"]
    assert "git_dirty_status" in manifest

"""Regression tests for the equal-total-shot measurement protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_measurement_full as measurement  # noqa: E402


def _random_states(n_qubits: int, n_states: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = 2**n_qubits
    states = []
    for _ in range(n_states):
        vector = rng.normal(size=d) + 1j * rng.normal(size=d)
        vector /= np.linalg.norm(vector)
        states.append(np.outer(vector, vector.conj()))
    return np.asarray(states)


def test_grouped_exact_features_match_observable_order():
    n_qubits = measurement.N_QUBITS
    states = _random_states(n_qubits, 3, seed=11)
    rotations, eigenvalues, pair_eigenvalues = measurement._rotations(n_qubits)
    probabilities = np.empty((len(states), 3, 2**n_qubits), dtype=float)
    for time, rho in enumerate(states):
        for axis_index, axis in enumerate(("x", "y", "z")):
            U = rotations[axis]
            probabilities[time, axis_index] = np.real(
                np.einsum("ai,ij,aj->a", U, rho, U.conj())
            )
    grouped = measurement.probabilities_to_features(
        probabilities, eigenvalues, pair_eigenvalues
    )
    observables = measurement.readout.pauli_observables(
        n_qubits, max_weight=2
    )
    direct = measurement.readout.features_from_states(states, observables)
    assert np.allclose(grouped, direct, rtol=1e-12, atol=1e-12)


def test_protocol_uses_identical_total_state_preparations():
    protocol = measurement.protocol_dict(
        channels=measurement.CHANNELS[:1],
        seeds=[1],
        independent_shots=[64, 256],
    )
    assert protocol["n_observables"] == 45
    for budget in protocol["finite_budgets"]:
        total = budget["total_shots_per_time_step"]
        assert 45 * budget["independent_shots_per_observable"] == total
        assert 3 * budget["grouped_shots_per_setting"] == total


def test_sampling_is_unbiased_to_monte_carlo_tolerance():
    exact = np.asarray([[0.2, -0.4, 0.9]])
    rng = np.random.default_rng(3)
    repeated = np.repeat(exact, 20000, axis=0)
    sampled = measurement.sample_independent(repeated, 128, rng)
    assert np.allclose(sampled.mean(axis=0), exact[0], atol=4e-3)


def test_ridge_selection_is_blind_to_test_features_and_targets():
    rng = np.random.default_rng(4)
    n_rows = measurement.TRAIN + measurement.VALIDATION + measurement.TEST
    inputs = rng.uniform(size=n_rows)
    features = rng.normal(size=(n_rows, 8))
    original = measurement.ridge_selected_stm(features, inputs)

    changed_features = features.copy()
    changed_inputs = inputs.copy()
    test_start = measurement.TRAIN + measurement.VALIDATION
    changed_features[test_start:] = rng.normal(
        size=changed_features[test_start:].shape
    )
    # Only mutate targets whose delayed source also lies in the test block.
    changed_inputs[test_start:] = rng.uniform(size=n_rows - test_start)
    changed = measurement.ridge_selected_stm(
        changed_features, changed_inputs
    )
    assert changed["selected_ridge"] == original["selected_ridge"]
    assert changed["validation_mc_by_ridge"] == original[
        "validation_mc_by_ridge"
    ]


def test_grouped_samples_have_the_expected_shape_and_range():
    n_qubits = measurement.N_QUBITS
    d = 2**n_qubits
    probabilities = np.full((7, 3, d), 1.0 / d)
    _, eigenvalues, pair_eigenvalues = measurement._rotations(n_qubits)
    features = measurement.sample_grouped(
        probabilities,
        shots_per_setting=960,
        rng=np.random.default_rng(8),
        eigenvalues=eigenvalues,
        pair_eigenvalues=pair_eigenvalues,
    )
    assert features.shape == (7, 45)
    assert np.all(features >= -1.0)
    assert np.all(features <= 1.0)


def test_infinite_ridge_candidate_matches_large_ridge_capacity_limit():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(180, 9))
    Y = rng.normal(size=(180, 4))
    X_test = rng.normal(size=(80, 9))
    limiting = measurement._fit_multioutput(
        X, Y, measurement.RIDGE_INFINITY
    )
    large_finite = measurement._fit_multioutput(X, Y, 1e10)
    capacity_limit = measurement._capacity_columns(Y[:80], X_test @ limiting)
    capacity_large = measurement._capacity_columns(
        Y[:80], X_test @ large_finite
    )
    assert np.allclose(capacity_limit, capacity_large, rtol=2e-7, atol=2e-9)

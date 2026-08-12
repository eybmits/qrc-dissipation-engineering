"""Physics and protocol-unit tests for the post-hoc driven-activity analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import run_primary_driven_activity as activity  # noqa: E402

from qrc import dissipators as dsp  # noqa: E402
from qrc.liouvillian import vec  # noqa: E402


def test_activity_functional_matches_direct_trace():
    rng = np.random.default_rng(41)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = raw @ raw.conj().T
    rho /= np.trace(rho)
    jumps = dsp.local_loss(2, 0.7) + dsp.dephasing(2, 0.2)
    rate_operator = activity.jump_rate_operator(jumps)
    direct = np.trace(rate_operator @ rho)
    from_vector = activity.activity_functional(rate_operator) @ vec(rho)
    assert abs(direct - from_vector) < 1e-13


def test_augmented_exponential_integrates_constant_identity_activity():
    # A trace-preserving generator with K = 1.75 I has constant activity 1.75.
    rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=complex)
    state = vec(rho)
    generator = sparse.csr_matrix((4, 4), dtype=complex)
    functional = activity.activity_functional(1.75 * np.eye(2))
    evolved, count, residue = activity.integrated_activity_step(
        generator, state, functional, dt=0.4
    )
    assert np.max(np.abs(evolved - state)) < 1e-14
    assert abs(count - 0.7) < 1e-13
    assert residue < 1e-14


def test_augmented_exponential_matches_dense_quadrature_identity():
    # Compare the block-exponential result with the analytic integral for a
    # small diagonal linear system. This tests non-constant activity.
    generator = np.diag([-0.3, -0.5]).astype(complex)
    state = np.array([0.4, 0.6], dtype=complex)
    functional = np.array([1.2, 0.7], dtype=complex)
    dt = 0.8
    evolved, count, residue = activity.integrated_activity_step(
        sparse.csr_matrix(generator), state, functional, dt
    )
    expected_state = expm(generator * dt) @ state
    expected_count = sum(
        functional[index]
        * state[index]
        * (np.exp(generator[index, index] * dt) - 1)
        / generator[index, index]
        for index in range(2)
    )
    assert np.max(np.abs(evolved - expected_state)) < 1e-13
    assert abs(count - expected_count.real) < 1e-13
    assert residue < 1e-14


def test_protocol_constants_preserve_primary_test_boundary():
    assert activity.WASH == 200
    assert activity.TRAIN == 600
    assert activity.TEST == 400
    assert activity.WASH + activity.TRAIN == 800
    assert activity.METHODS[0] == activity.REFERENCE_METHOD
    assert len(activity.METHODS) == 7
    assert len(activity.deterministic_seeds(32)) == 32

"""Mathematical regression checks for the driven-dephasing proof."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, null_space

from qrc import dissipators as dsp
from qrc.liouvillian import lindbladian, unvec, vec
from qrc.operators import pauli_op
from qrc.reservoirs import (
    ising_xx_hamiltonian,
    random_couplings,
    transverse_drive,
)


def _apply(superoperator: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return unvec(superoperator @ vec(matrix), matrix.shape[0])


def test_dephasing_dirichlet_identity_has_the_stated_factor():
    n_qubits = 2
    d = 2**n_qubits
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    X = raw - np.trace(raw) * np.eye(d) / d
    hermitian = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    H = hermitian + hermitian.conj().T
    jumps = [
        (pauli_op("z", 0, n_qubits), 0.37),
        (pauli_op("z", 1, n_qubits), 1.2),
    ]
    generator = lindbladian(H, jumps)
    lhs = 2.0 * np.real(np.vdot(X, _apply(generator, X)))
    rhs = -sum(
        rate
        * np.linalg.norm(operator @ X - X @ operator, ord="fro") ** 2
        for operator, rate in jumps
    )
    assert np.isclose(lhs, rhs, rtol=2e-13, atol=2e-13)


def test_diagonal_joint_commutant_is_only_the_scalars():
    n_qubits = 3
    d = 2**n_qubits
    rng = np.random.default_rng(13)
    couplings = random_couplings(n_qubits, 1.0, rng)
    h = 0.5
    s = 0.43
    H = (
        ising_xx_hamiltonian(couplings, h, n_qubits)
        + h * (1.0 + s) * transverse_drive(n_qubits)
    )
    # Column b is vec([H, |b><b|]); the kernel gives diagonal matrices
    # commuting with H.
    commutator_columns = []
    for basis in range(d):
        projector = np.zeros((d, d), dtype=complex)
        projector[basis, basis] = 1.0
        commutator_columns.append(vec(H @ projector - projector @ H))
    linear_map = np.column_stack(commutator_columns)
    singular_values = np.linalg.svd(linear_map, compute_uv=False)
    rank = int(np.sum(singular_values > 1e-10))
    assert rank == d - 1
    assert np.allclose(linear_map @ np.ones(d), 0.0, atol=1e-13)


def test_small_n_propagator_is_strictly_contracting_on_traceless_space():
    n_qubits = 2
    d = 2**n_qubits
    rng = np.random.default_rng(14)
    couplings = random_couplings(n_qubits, 1.0, rng)
    h = 0.5
    dt = 0.5
    target = dsp.jump_strength(dsp.local_loss(n_qubits, 1.0))
    jumps = dsp.normalize_jump_strength(
        dsp.dephasing(n_qubits, 1.0), target
    )
    identity_vector = vec(np.eye(d)) / np.sqrt(d)
    traceless_basis = null_space(identity_vector.conj()[None, :])
    singular_maxima = []
    for s in np.linspace(0.0, 1.0, 11):
        H = (
            ising_xx_hamiltonian(couplings, h, n_qubits)
            + h * (1.0 + s) * transverse_drive(n_qubits)
        )
        propagator = expm(lindbladian(H, jumps) * dt)
        restricted = traceless_basis.conj().T @ propagator @ traceless_basis
        singular_maxima.append(
            float(np.linalg.svd(restricted, compute_uv=False)[0])
        )
    assert max(singular_maxima) < 1.0 - 1e-10

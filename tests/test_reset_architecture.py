import numpy as np

from qrc import dissipators, readout
from qrc import reservoirs as res
from qrc.liouvillian import lindbladian, propagator
from qrc.operators import input_state, trace_out


def _random_density(d: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def test_zero_gamma_reset_lindblad_matches_fuji_nakajima_step_and_run():
    rng = np.random.default_rng(90210)
    n_qubits = 3
    J = res.random_couplings(n_qubits, 1.0, rng)
    inputs = rng.uniform(0.0, 1.0, size=16)
    observables = readout.pauli_observables(n_qubits, max_weight=2)

    unitary = res.FujiNakajimaReservoir(
        n_qubits,
        J,
        h=0.5,
        dt=0.5,
    )
    for family in ("local", "collective"):
        reset_lindblad = res.dissipative_input_reset(
            n_qubits,
            J,
            h=0.5,
            gamma=0.0,
            dt=0.5,
            jump_family=family,
        )
        rho_unitary = _random_density(2**n_qubits, rng)
        rho_lindblad = rho_unitary.copy()
        for input_value in inputs:
            rho_unitary = unitary.step(rho_unitary, float(input_value))
            rho_lindblad = reset_lindblad.step(
                rho_lindblad,
                float(input_value),
            )
            assert np.allclose(
                rho_lindblad,
                rho_unitary,
                atol=2e-13,
                rtol=2e-13,
            )

        features_unitary = unitary.run(
            inputs,
            observables,
            washout=3,
        )
        features_lindblad = reset_lindblad.run(
            inputs,
            observables,
            washout=3,
        )
        assert np.allclose(
            features_lindblad,
            features_unitary,
            atol=2e-13,
            rtol=2e-13,
        )


def test_strict_local_and_collective_structural_budgets_equal_80():
    n_qubits = 5
    gamma = 1.0
    local_budget = dissipators.jump_strength(
        dissipators.local_loss(n_qubits, gamma)
    )
    collective_budget = dissipators.jump_strength(
        dissipators.collective_loss(n_qubits, gamma)
    )
    assert local_budget == 80.0
    assert collective_budget == 80.0


def test_reset_lindblad_preserves_density_matrix_physicality():
    rng = np.random.default_rng(481516)
    n_qubits = 3
    J = res.random_couplings(n_qubits, 1.0, rng)
    inputs = rng.uniform(0.0, 1.0, size=20)

    for family in ("local", "collective"):
        reservoir = res.dissipative_input_reset(
            n_qubits,
            J,
            h=0.5,
            gamma=1.0,
            dt=0.5,
            jump_family=family,
        )
        rho = _random_density(2**n_qubits, rng)
        for input_value in inputs:
            rho = reservoir.step(rho, float(input_value))
            assert abs(np.trace(rho) - 1.0) < 2e-12
            assert np.max(np.abs(rho - rho.conj().T)) < 2e-12
            assert np.min(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))) > -2e-12


def test_reset_architecture_uses_repository_primitives_exactly():
    rng = np.random.default_rng(314159)
    n_qubits = 3
    J = res.random_couplings(n_qubits, 1.0, rng)
    H = res.ising_xx_hamiltonian(J, h=0.5, n_qubits=n_qubits)
    rho = _random_density(2**n_qubits, rng)
    input_value = 0.37

    expected_injection = np.kron(
        input_state(input_value),
        trace_out(rho, discard=[0], n_qubits=n_qubits),
    )
    assert np.array_equal(
        res.inject_input(rho, input_value, n_qubits),
        expected_injection,
    )

    for family, jumps in (
        ("local", dissipators.local_loss(n_qubits, 1.0)),
        ("collective", dissipators.collective_loss(n_qubits, 1.0)),
    ):
        reservoir = res.dissipative_input_reset(
            n_qubits,
            J,
            h=0.5,
            gamma=1.0,
            dt=0.5,
            jump_family=family,
        )
        expected_superoperator = lindbladian(H, jumps)
        expected_propagator = propagator(expected_superoperator, 0.5)
        assert np.array_equal(reservoir.base_super, expected_superoperator)
        assert np.array_equal(reservoir.P, expected_propagator)

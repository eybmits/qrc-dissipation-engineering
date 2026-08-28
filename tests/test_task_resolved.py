from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from qrc.liouvillian import commutator_super, dissipator_super, vec
from qrc.operators import SX, SZ, sminus
from qrc.readout import Observable
from qrc.task_resolved import (
    affine_binary_channel,
    build_kernel_library,
    delay_capacity,
    diagonal_linear_memory_capacity,
    first_order_kernels,
    input_affine_channel_expansion,
    interpolated_kossakowski,
    kernel_library_mixed_lags,
    kossakowski_dissipator_super,
    kossakowski_to_jumps,
    observable_readout_matrix,
    primitive_gap,
    product_capacity,
    project_psd_trace,
    second_order_cross_kernels,
    shot_covariance_independent,
)


def simple_expansion():
    h0 = 0.4 * SX + 0.2 * SZ
    h1 = 0.35 * SX
    jump = np.array([[0, 1], [0, 0]], complex)
    base = commutator_super(h0) + 0.15 * dissipator_super(jump)
    drive = commutator_super(h1)
    expansion = input_affine_channel_expansion(
        base,
        drive,
        0.5,
        0.3,
        [Observable("x", SX), Observable("z", SZ)],
    )
    return base, drive, expansion


def test_readout_matrix_matches_trace():
    rho = np.array([[0.7, 0.1j], [-0.1j, 0.3]], complex)
    readout = observable_readout_matrix(
        [Observable("x", SX), Observable("z", SZ)]
    )
    assert np.allclose(
        np.real(readout @ vec(rho)),
        [np.trace(SX @ rho).real, np.trace(SZ @ rho).real],
    )


def test_kossakowski_jump_decomposition_matches_direct():
    basis = [sminus(0, 2), sminus(1, 2)]
    matrix = np.array([[0.4, 0.12 + 0.05j], [0.12 - 0.05j, 0.2]])
    direct = kossakowski_dissipator_super(basis, matrix)
    via_jumps = sum(
        rate * dissipator_super(jump)
        for jump, rate in kossakowski_to_jumps(basis, matrix)
    )
    assert np.allclose(direct, via_jumps, atol=1e-11)


def test_frechet_derivative_matches_finite_difference():
    h0 = 0.4 * SX + 0.2 * SZ
    h1 = 0.3 * SZ
    jump = np.array([[0, 1], [0, 0]], complex)
    base = commutator_super(h0) + 0.1 * dissipator_super(jump)
    drive = commutator_super(h1)
    expansion = input_affine_channel_expansion(
        base,
        drive,
        0.7,
        0.2,
        [Observable("x", SX)],
    )
    epsilon = 1e-6
    finite = (
        expm((base + (0.2 + epsilon) * drive) * 0.7)
        - expm((base + (0.2 - epsilon) * drive) * 0.7)
    ) / (2 * epsilon)
    assert np.allclose(expansion.derivative, finite, atol=2e-8, rtol=2e-7)


def test_second_order_kernel_matches_sequence_finite_difference():
    base, drive, expansion = simple_expansion()
    kernels = second_order_cross_kernels(expansion, 4)
    delay_a, delay_b = 1, 4
    delta = 2e-4
    reference_channel = expansion.channel

    def output(sign_a, sign_b):
        state = expansion.fixed_point.copy()
        for delay in range(delay_b, -1, -1):
            perturbation = (
                sign_b if delay == delay_b else sign_a if delay == delay_a else 0
            )
            propagator = (
                expm((base + (0.3 + delta * perturbation) * drive) * 0.5)
                if perturbation
                else reference_channel
            )
            state = propagator @ state
        return float(np.real(expansion.readout[1] @ state))

    mixed = (
        output(1, 1)
        - output(1, -1)
        - output(-1, 1)
        + output(-1, -1)
    ) / (4 * delta * delta)
    assert np.isclose(mixed, kernels[1, delay_a, delay_b], atol=2e-5, rtol=2e-4)


def test_capacities_bounded():
    _, _, expansion = simple_expansion()
    library = build_kernel_library(expansion, 5, 0.02, ridge=1e-12)
    assert 0 <= delay_capacity(library, 2) <= 1 + 1e-7
    assert 0 <= product_capacity(library, 1, 3) <= 1 + 1e-7


def test_interpolation_preserves_trace_budget():
    for alpha in (0, 0.25, 0.75, 1):
        matrix = interpolated_kossakowski(4, 0.2, alpha)
        assert np.isclose(np.trace(matrix), 0.8)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_psd_projection():
    matrix = project_psd_trace(np.array([[0.8, 0.5j], [-0.5j, -0.2]]), 0.7)
    assert np.isclose(np.trace(matrix), 0.7)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_shot_covariance_scales_as_inverse_shots():
    rho = np.diag([0.8, 0.2])
    observables = [Observable("x", SX), Observable("z", SZ)]
    covariance_100 = shot_covariance_independent(rho, observables, 100)
    covariance_1000 = shot_covariance_independent(rho, observables, 1000)
    assert np.allclose(covariance_100, 10 * covariance_1000)


def test_mixed_lag_covariance_adds_psd_tail():
    _, _, expansion = simple_expansion()
    first = first_order_kernels(expansion, 30)
    second = second_order_cross_kernels(expansion, 6)
    mixed = kernel_library_mixed_lags(first, second, 0.02, target_max_delay=6)
    short = build_kernel_library(expansion, 6, 0.02)
    difference = mixed.feature_covariance - short.feature_covariance
    assert np.linalg.eigvalsh(0.5 * (difference + difference.T)).min() >= -1e-10


def test_primitive_gap_amplitude_damping():
    jump = np.array([[0, 1], [0, 0]], complex)
    diagnostic = primitive_gap(0.2 * dissipator_super(jump))
    assert diagnostic.stable
    assert diagnostic.stationary_multiplicity == 1
    assert np.isclose(diagnostic.gap, 0.1)


def test_affine_binary_channel_is_stochastic_and_has_claimed_bloch_map():
    channel = affine_binary_channel(0.7, 0.1)
    assert np.allclose(channel.sum(axis=0), 1)
    assert channel.min() >= 0
    probability = np.array([0.8, 0.2])
    z_value = probability[0] - probability[1]
    output = channel @ probability
    assert np.isclose(output[0] - output[1], 0.7 * z_value + 0.1)


def test_equal_budget_counterexample_has_incompatible_winners():
    balanced = np.exp(-np.array([0.4, 0.6]))
    heterogeneous = np.exp(-np.array([0.05, 0.95]))
    assert diagonal_linear_memory_capacity(
        balanced, 2
    ) > diagonal_linear_memory_capacity(heterogeneous, 2)
    assert diagonal_linear_memory_capacity(
        heterogeneous, 10
    ) > diagonal_linear_memory_capacity(balanced, 10)

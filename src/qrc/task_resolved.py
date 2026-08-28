"""Training-free Walsh-Volterra analysis and Kossakowski design for QRC.

The module uses column-stacked density matrices. For an input-affine channel
Phi_u = exp(dt (L0 + u L1)), it computes exact Frechet derivatives, temporal
response kernels, population linear-readout capacities, finite-shot covariance,
and physically valid Kossakowski coupling families.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.linalg import expm_frechet

from .liouvillian import commutator_super, unvec, vec

Array = np.ndarray


@dataclass(frozen=True)
class ChannelExpansion:
    channel: Array
    derivative: Array
    fixed_point: Array
    readout: Array
    reference_input: float
    dt: float


@dataclass(frozen=True)
class KernelLibrary:
    first: Array
    second_cross: Array
    feature_covariance: Array
    epsilon: float
    max_delay: int


@dataclass(frozen=True)
class GapDiagnostic:
    gap: float
    stationary_multiplicity: int
    stable: bool


def _validate_square(name: str, matrix: Array) -> Array:
    matrix = np.asarray(matrix, complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def affine_binary_channel(contraction: float, offset: float) -> Array:
    """Column-stochastic bit channel with z' = contraction*z + offset."""
    contraction = float(contraction)
    offset = float(offset)
    if not 0 <= contraction < 1:
        raise ValueError("contraction must lie in [0,1)")
    if abs(offset) > 1 - contraction + 1e-12:
        raise ValueError("offset is not stochastic")
    p00 = 0.5 * (1 + offset + contraction)
    p01 = 0.5 * (1 + offset - contraction)
    return np.array([[p00, p01], [1 - p00, 1 - p01]], float)


def _stable_inverse_quadratic(
    covariance: Array,
    vector: Array,
    ridge: float = 0.0,
) -> float:
    covariance = np.asarray(covariance, float)
    vector = np.asarray(vector, float)
    regularized = covariance + float(ridge) * np.eye(covariance.shape[0])
    value = float(vector @ np.linalg.pinv(regularized, rcond=1e-12) @ vector)
    return float(np.clip(value, 0.0, 1.0 + 1e-8))


def diagonal_linear_memory_capacity(
    decay_factors: Sequence[float],
    delay: int,
    write_vector: Sequence[float] | None = None,
    *,
    ridge: float = 0.0,
) -> float:
    """Exact capacity for x_t=diag(lambda)x_{t-1}+b u_t and iid unit inputs."""
    lambdas = np.asarray(decay_factors, float)
    if lambdas.ndim != 1 or lambdas.size == 0:
        raise ValueError("decay_factors must be a vector")
    if np.any(np.abs(lambdas) >= 1):
        raise ValueError("unstable decay factor")
    if delay < 0:
        raise ValueError("delay must be non-negative")
    write = np.ones_like(lambdas) if write_vector is None else np.asarray(write_vector, float)
    if write.shape != lambdas.shape:
        raise ValueError("write vector mismatch")
    gramian = np.outer(write, write) / (1 - np.outer(lambdas, lambdas))
    response = write * lambdas ** int(delay)
    return _stable_inverse_quadratic(gramian, response, ridge)


def observable_readout_matrix(observables: Sequence[object]) -> Array:
    if not observables:
        raise ValueError("at least one observable is required")
    rows = []
    for observable in observables:
        matrix = np.asarray(getattr(observable, "matrix", observable), complex)
        rows.append(vec(matrix.T))
    return np.stack(rows)


def fixed_point_of_channel(channel: Array, *, eigen_tol: float = 1e-7) -> Array:
    channel = _validate_square("channel", channel)
    d = int(round(np.sqrt(channel.shape[0])))
    if d * d != channel.shape[0]:
        raise ValueError("channel dimension is not d^2")
    values, vectors = np.linalg.eig(channel)
    idx = int(np.argmin(np.abs(values - 1)))
    if abs(values[idx] - 1) > eigen_tol:
        raise RuntimeError("unit eigenvalue not found")
    rho = unvec(vectors[:, idx], d)
    rho = 0.5 * (rho + rho.conj().T)
    trace = np.trace(rho)
    if abs(trace) < 1e-12:
        raise RuntimeError("fixed point has zero trace")
    rho /= trace
    vals, vecs = np.linalg.eigh(rho)
    if vals.min() < -1e-7:
        raise RuntimeError("fixed point is not PSD")
    vals = np.maximum(vals.real, 0)
    rho = (vecs * vals) @ vecs.conj().T
    return rho / np.trace(rho)


def input_affine_channel_expansion(
    base_super: Array,
    drive_super: Array,
    dt: float,
    reference_input: float,
    observables: Sequence[object],
) -> ChannelExpansion:
    base_super = _validate_square("base_super", base_super)
    drive_super = _validate_square("drive_super", drive_super)
    if base_super.shape != drive_super.shape:
        raise ValueError("shape mismatch")
    if dt <= 0:
        raise ValueError("dt must be positive")
    generator = (base_super + float(reference_input) * drive_super) * float(dt)
    channel, derivative = expm_frechet(
        generator,
        drive_super * float(dt),
        compute_expm=True,
    )
    rho = fixed_point_of_channel(channel)
    return ChannelExpansion(
        channel,
        derivative,
        vec(rho),
        observable_readout_matrix(observables),
        float(reference_input),
        float(dt),
    )


def first_order_kernels(expansion: ChannelExpansion, max_delay: int) -> Array:
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")
    A, D, R = expansion.channel, expansion.derivative, expansion.readout
    state = D @ expansion.fixed_point
    out = np.empty((R.shape[0], max_delay + 1), float)
    for delay in range(max_delay + 1):
        out[:, delay] = np.real_if_close(R @ state).real
        state = A @ state
    return out


def second_order_cross_kernels(
    expansion: ChannelExpansion,
    max_delay: int,
) -> Array:
    """Return q[:,a,b] = R A^a D A^(b-a-1) D rho* for 0 <= a < b."""
    n_features = expansion.readout.shape[0]
    q = np.zeros((n_features, max_delay + 1, max_delay + 1), float)
    if max_delay < 1:
        return q
    A, D, R = expansion.channel, expansion.derivative, expansion.readout
    b0 = D @ expansion.fixed_point
    first_states = np.empty((A.shape[0], max_delay), complex)
    state = b0.copy()
    for gap in range(max_delay):
        first_states[:, gap] = state
        state = A @ state
    propagated = D @ first_states
    for delay_a in range(max_delay):
        count = max_delay - delay_a
        values = np.real_if_close(R @ propagated[:, :count]).real
        for gap in range(count):
            q[:, delay_a, delay_a + gap + 1] = values[:, gap]
        if count > 1:
            propagated[:, : count - 1] = A @ propagated[:, : count - 1]
    return q


def shot_covariance_independent(
    fixed_point: Array,
    observables: Sequence[object],
    shots: float | Sequence[float],
) -> Array:
    rho = np.asarray(fixed_point, complex)
    means = np.array(
        [
            np.real(np.trace(rho @ np.asarray(getattr(o, "matrix", o), complex)))
            for o in observables
        ]
    )
    shot_vector = (
        np.full(len(observables), float(shots))
        if np.isscalar(shots)
        else np.asarray(shots, float)
    )
    if shot_vector.shape != (len(observables),) or np.any(shot_vector <= 0):
        raise ValueError("invalid shots")
    return np.diag(np.maximum(0, 1 - means * means) / shot_vector)


def shot_covariance_groups(
    fixed_point: Array,
    observables: Sequence[object],
    groups: Sequence[Sequence[int]],
    shots_per_group: float | Sequence[float],
    *,
    commute_tol: float = 1e-10,
) -> Array:
    rho = np.asarray(fixed_point, complex)
    mats = [np.asarray(getattr(o, "matrix", o), complex) for o in observables]
    means = np.array([np.real(np.trace(rho @ matrix)) for matrix in mats])
    group_shots = (
        np.full(len(groups), float(shots_per_group))
        if np.isscalar(shots_per_group)
        else np.asarray(shots_per_group, float)
    )
    covariance = np.zeros((len(mats), len(mats)))
    assigned: set[int] = set()
    for group_index, group in enumerate(groups):
        if group_shots[group_index] <= 0:
            raise ValueError("invalid shots")
        group = list(group)
        if assigned.intersection(group):
            raise ValueError("duplicate observable")
        assigned.update(group)
        for i in group:
            for j in group:
                if np.linalg.norm(mats[i] @ mats[j] - mats[j] @ mats[i], "fro") > commute_tol:
                    raise ValueError("noncommuting group")
                joint = np.real(np.trace(rho @ (mats[i] @ mats[j])))
                covariance[i, j] = (joint - means[i] * means[j]) / group_shots[group_index]
    if assigned != set(range(len(mats))):
        raise ValueError("groups must cover observables")
    return 0.5 * (covariance + covariance.T)


def kernel_library_from_kernels(
    first: Array,
    second_cross: Array,
    epsilon: float,
    *,
    shot_covariance: Array | None = None,
    ridge: float = 0.0,
) -> KernelLibrary:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    first = np.asarray(first, float)
    second_cross = np.asarray(second_cross, float)
    if (
        second_cross.ndim != 3
        or second_cross.shape[0] != first.shape[0]
        or second_cross.shape[1] != second_cross.shape[2]
        or first.shape[1] != second_cross.shape[1]
    ):
        raise ValueError("kernel mismatch")
    max_delay = second_cross.shape[1] - 1
    covariance = epsilon**2 * (first @ first.T)
    upper = np.triu_indices(max_delay + 1, k=1)
    flattened = second_cross[:, upper[0], upper[1]]
    if flattened.size:
        covariance += epsilon**4 * (flattened @ flattened.T)
    if shot_covariance is not None:
        covariance += np.asarray(shot_covariance, float)
    if ridge:
        covariance += float(ridge) * np.eye(covariance.shape[0])
    return KernelLibrary(first, second_cross, covariance, float(epsilon), max_delay)


def kernel_library_mixed_lags(
    first_covariance: Array,
    second_cross_covariance: Array,
    epsilon: float,
    *,
    target_max_delay: int | None = None,
    shot_covariance: Array | None = None,
    ridge: float = 0.0,
) -> KernelLibrary:
    first = np.asarray(first_covariance, float)
    second = np.asarray(second_cross_covariance, float)
    if (
        first.ndim != 2
        or second.ndim != 3
        or first.shape[0] != second.shape[0]
        or second.shape[1] != second.shape[2]
    ):
        raise ValueError("kernel mismatch")
    second_max = second.shape[1] - 1
    first_max = first.shape[1] - 1
    target = min(first_max, second_max) if target_max_delay is None else int(target_max_delay)
    if target < 0 or target > min(first_max, second_max):
        raise ValueError("target delay unavailable")
    covariance = epsilon**2 * (first @ first.T)
    upper = np.triu_indices(second_max + 1, k=1)
    flattened = second[:, upper[0], upper[1]]
    if flattened.size:
        covariance += epsilon**4 * (flattened @ flattened.T)
    if shot_covariance is not None:
        covariance += np.asarray(shot_covariance, float)
    if ridge:
        covariance += float(ridge) * np.eye(covariance.shape[0])
    return KernelLibrary(
        first[:, : target + 1],
        second[:, : target + 1, : target + 1],
        covariance,
        float(epsilon),
        target,
    )


def build_kernel_library(
    expansion: ChannelExpansion,
    max_delay: int,
    epsilon: float,
    *,
    shot_covariance: Array | None = None,
    ridge: float = 0.0,
) -> KernelLibrary:
    return kernel_library_from_kernels(
        first_order_kernels(expansion, max_delay),
        second_order_cross_kernels(expansion, max_delay),
        epsilon,
        shot_covariance=shot_covariance,
        ridge=ridge,
    )


def rescale_kernel_library(
    library: KernelLibrary,
    epsilon: float,
    *,
    shot_covariance: Array | None = None,
    ridge: float = 0.0,
) -> KernelLibrary:
    return kernel_library_from_kernels(
        library.first,
        library.second_cross,
        epsilon,
        shot_covariance=shot_covariance,
        ridge=ridge,
    )


def delay_capacity(library: KernelLibrary, delay: int, *, ridge: float = 0.0) -> float:
    if not 0 <= delay <= library.max_delay:
        raise ValueError("delay outside library")
    vector = library.epsilon * library.first[:, delay]
    return _stable_inverse_quadratic(library.feature_covariance, vector, ridge)


def product_capacity(
    library: KernelLibrary,
    delay_a: int,
    delay_b: int,
    *,
    ridge: float = 0.0,
) -> float:
    delay_a, delay_b = sorted((int(delay_a), int(delay_b)))
    if delay_a == delay_b or delay_a < 0 or delay_b > library.max_delay:
        raise ValueError("invalid product delays")
    vector = library.epsilon**2 * library.second_cross[:, delay_a, delay_b]
    return _stable_inverse_quadratic(library.feature_covariance, vector, ridge)


def primitive_gap(
    generator: Array,
    *,
    zero_tol: float = 1e-8,
    stability_tol: float = 1e-10,
) -> GapDiagnostic:
    eigenvalues = np.linalg.eigvals(_validate_square("generator", generator))
    zero = np.abs(eigenvalues) <= zero_tol
    multiplicity = int(zero.sum())
    nonzero = eigenvalues[~zero]
    if nonzero.size == 0:
        return GapDiagnostic(0, multiplicity, False)
    leading = float(nonzero.real.max())
    return GapDiagnostic(
        max(0, -leading),
        multiplicity,
        multiplicity == 1 and leading < -stability_tol,
    )


def kossakowski_dissipator_super(
    basis_jumps: Sequence[Array],
    matrix: Array,
) -> Array:
    if not basis_jumps:
        raise ValueError("empty basis")
    matrix = _validate_square("matrix", matrix)
    if (
        matrix.shape != (len(basis_jumps), len(basis_jumps))
        or not np.allclose(matrix, matrix.conj().T, atol=1e-10)
        or np.linalg.eigvalsh(matrix).min() < -1e-10
    ):
        raise ValueError("C must be Hermitian PSD")
    jumps = [np.asarray(jump, complex) for jump in basis_jumps]
    dimension = jumps[0].shape[0]
    identity = np.eye(dimension, dtype=complex)
    out = np.zeros((dimension * dimension, dimension * dimension), complex)
    for i, F_i in enumerate(jumps):
        for j, F_j in enumerate(jumps):
            coefficient = matrix[i, j]
            if abs(coefficient) < 1e-15:
                continue
            product = F_j.conj().T @ F_i
            out += coefficient * (
                np.kron(F_j.conj(), F_i)
                - 0.5 * np.kron(identity, product)
                - 0.5 * np.kron(product.T, identity)
            )
    return out


def kossakowski_to_jumps(
    basis_jumps: Sequence[Array],
    matrix: Array,
    *,
    eigen_tol: float = 1e-12,
) -> list[tuple[Array, float]]:
    matrix = _validate_square("matrix", matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
    if eigenvalues.min() < -1e-10:
        raise ValueError("C must be PSD")
    basis = [np.asarray(jump, complex) for jump in basis_jumps]
    result: list[tuple[Array, float]] = []
    for index, eigenvalue in enumerate(eigenvalues):
        if eigenvalue <= eigen_tol:
            continue
        jump = sum(
            (eigenvectors[i, index] * basis[i] for i in range(len(basis))),
            start=np.zeros_like(basis[0]),
        )
        result.append((np.sqrt(eigenvalue) * jump, 1.0))
    return result


def interpolated_kossakowski(
    n_sites: int,
    gamma: float,
    alpha: float,
    coefficients: Sequence[complex] | None = None,
) -> Array:
    if n_sites < 1 or gamma < 0:
        raise ValueError("invalid size/rate")
    lower = -1 / max(1, n_sites - 1)
    if not lower - 1e-12 <= alpha <= 1 + 1e-12:
        raise ValueError("alpha outside PSD interval")
    coefficients_array = (
        np.ones(n_sites, complex)
        if coefficients is None
        else np.asarray(coefficients, complex)
    )
    if coefficients_array.shape != (n_sites,) or np.vdot(coefficients_array, coefficients_array).real <= 0:
        raise ValueError("invalid coefficients")
    coefficients_array = coefficients_array * np.sqrt(
        n_sites / np.vdot(coefficients_array, coefficients_array).real
    )
    return float(gamma) * (
        (1 - float(alpha)) * np.eye(n_sites)
        + float(alpha) * np.outer(coefficients_array, coefficients_array.conj())
    )


def project_psd_trace(matrix: Array, trace_budget: float) -> Array:
    if trace_budget <= 0:
        raise ValueError("trace budget must be positive")
    matrix_array = np.asarray(matrix, complex)
    hermitian = 0.5 * (
        _validate_square("matrix", matrix_array) + matrix_array.conj().T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    sorted_values = np.sort(eigenvalues)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, len(sorted_values) + 1)
    condition = sorted_values - (cumulative - trace_budget) / indices > 0
    rho_index = np.nonzero(condition)[0][-1]
    threshold = (cumulative[rho_index] - trace_budget) / (rho_index + 1)
    projected = np.maximum(eigenvalues - threshold, 0)
    return (eigenvectors * projected) @ eigenvectors.conj().T


def input_affine_liouvillian_from_kossakowski(
    static_hamiltonian: Array,
    drive_hamiltonian: Array,
    basis_jumps: Sequence[Array],
    matrix: Array,
) -> tuple[Array, Array]:
    base = commutator_super(np.asarray(static_hamiltonian, complex))
    base += kossakowski_dissipator_super(basis_jumps, matrix)
    drive = commutator_super(np.asarray(drive_hamiltonian, complex))
    return base, drive

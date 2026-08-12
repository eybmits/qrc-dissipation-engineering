"""Rate profiles (Track A) and jump-operator families (Track B).

Track A keeps the jump operators fixed at local loss ``L_i = sigma^-_i`` and only
changes the per-qubit rates ``gamma_i``.  Track B changes the actual jump
operators.  All builders return a list of ``(L, rate)`` pairs consumable by
``LindbladReservoir.from_terms`` (or a ``gamma`` array for the CD factory).

Every comparison must control the *total* dissipative strength so that gains
cannot be attributed simply to more dissipation; see :func:`normalize_rates`.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .operators import SMINUS, SPLUS, sminus, splus, pauli_op, two_site_op


# ==========================================================================
# Track A - per-qubit rate profiles (jumps stay L_i = sigma^-_i)
# ==========================================================================

def normalize_rates(gammas: np.ndarray, target_mean: float) -> np.ndarray:
    """Rescale a rate profile so its mean equals ``target_mean`` (controls total
    dissipation when comparing heterogeneous vs uniform)."""
    gammas = np.asarray(gammas, float)
    m = gammas.mean()
    if m <= 0:
        raise ValueError("rate profile has non-positive mean")
    return gammas * (target_mean / m)


def uniform_rates(n: int, gamma: float) -> np.ndarray:
    return np.full(n, float(gamma))


def loguniform_rates(n: int, lo: float, hi: float, rng: np.random.Generator,
                     mean: Optional[float] = None) -> np.ndarray:
    """Random heterogeneous rates ``gamma_i ~ logU[lo, hi]`` (A1)."""
    g = np.exp(rng.uniform(np.log(lo), np.log(hi), size=n))
    return normalize_rates(g, mean) if mean is not None else g


def degree_rates(J: np.ndarray, scale: float = 1.0, eps: float = 1e-9,
                 mean: Optional[float] = None) -> np.ndarray:
    """Structured rate ``gamma_i ∝ deg(i)`` (number of nonzero couplings) (A2)."""
    deg = np.sum(np.abs(J) > eps, axis=1).astype(float)
    g = scale * (deg + eps)
    return normalize_rates(g, mean) if mean is not None else g


def coupling_rates(J: np.ndarray, scale: float = 1.0, eps: float = 1e-9,
                   mean: Optional[float] = None) -> np.ndarray:
    """Structured rate ``gamma_i ∝ sum_j |J_ij|`` (strongly coupled => faster
    forgetting) (A2)."""
    strength = np.sum(np.abs(J), axis=1)
    g = scale * (strength + eps)
    return normalize_rates(g, mean) if mean is not None else g


def inverse_coupling_rates(J: np.ndarray, eps: float = 1e-1,
                           mean: Optional[float] = None) -> np.ndarray:
    """Opposite profile ``gamma_i ∝ 1 / (eps + sum_j |J_ij|)`` (strongly coupled
    => slower forgetting) (A2)."""
    strength = np.sum(np.abs(J), axis=1)
    g = 1.0 / (eps + strength)
    return normalize_rates(g, mean) if mean is not None else g


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def rates_from_theta(theta: np.ndarray) -> np.ndarray:
    """A3: ``gamma_i = softplus(theta_i)`` (unconstrained optimisation param)."""
    return softplus(np.asarray(theta, float))


def adaptive_rates(s: float, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A4: input-adaptive ``gamma_i(s) = softplus(a_i s + b_i)``."""
    return softplus(np.asarray(a, float) * s + np.asarray(b, float))


# ==========================================================================
# Track B - jump-operator families (rates uniform unless noted)
# ==========================================================================

def jump_strength(jumps: List[Tuple[np.ndarray, float]]) -> float:
    """Total dissipative strength ``sum_k rate_k Tr(L_k^dag L_k)`` — the budget that
    must be held constant across Track-B families for a fair comparison."""
    return float(sum(rate * np.trace(np.asarray(L).conj().T @ np.asarray(L)).real
                     for L, rate in jumps))


def normalize_jump_strength(jumps: List[Tuple[np.ndarray, float]], target: float
                            ) -> List[Tuple[np.ndarray, float]]:
    """Rescale all rates so the family's total strength equals ``target``."""
    s = jump_strength(jumps)
    if s <= 0:
        raise ValueError("jump family has non-positive strength")
    f = target / s
    return [(L, rate * f) for L, rate in jumps]


def local_loss(n: int, gamma: float) -> List[Tuple[np.ndarray, float]]:
    """B0: amplitude damping ``L_i = sigma^-_i``."""
    return [(sminus(i, n), float(gamma)) for i in range(n)]


def dephasing(n: int, gamma: float) -> List[Tuple[np.ndarray, float]]:
    """B1: pure dephasing ``L_i = sigma^z_i`` (loses coherence, not energy)."""
    return [(pauli_op("z", i, n), float(gamma)) for i in range(n)]


def thermal(n: int, gamma_down: float, gamma_up: float
            ) -> List[Tuple[np.ndarray, float]]:
    """B2: fixed local gain/loss through lowering and raising jumps.

    The legacy function name is retained for compatibility with frozen result
    identifiers.  For a driven interacting Hamiltonian, the fixed ratio
    ``gamma_up/gamma_down`` does not in general impose detailed balance or
    guarantee relaxation to a Gibbs state.
    """
    jumps = [(sminus(i, n), float(gamma_down)) for i in range(n)]
    jumps += [(splus(i, n), float(gamma_up)) for i in range(n)]
    return jumps


def collective_loss(n: int, gamma: float, c: Optional[Sequence[complex]] = None
                    ) -> List[Tuple[np.ndarray, float]]:
    """B3: single collective jump ``L = sum_i c_i sigma^-_i``.

    ``c`` defaults to all-ones.  To keep the total dissipative strength
    comparable to ``n`` independent local-loss channels of rate ``gamma``, the
    possibly complex coefficient vector is normalised to
    ``sum_i |c_i|^2 = n`` (so ``Tr(L^dag L)`` matches the local-loss case).
    Relative phases are preserved.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    if c is None:
        c = np.ones(n, dtype=complex)
    c = np.asarray(c, dtype=complex)
    if c.shape != (n,):
        raise ValueError(f"collective coefficients must have shape ({n},)")
    if not np.all(np.isfinite(c)):
        raise ValueError("collective coefficients must be finite")
    norm_squared = float(np.vdot(c, c).real)
    if norm_squared <= 0.0:
        raise ValueError("collective coefficients must have positive norm")
    c = c * np.sqrt(n / norm_squared)   # sum_i |c_i|^2 = n
    d = 2 ** n
    L = np.zeros((d, d), dtype=complex)
    for i in range(n):
        L += c[i] * sminus(i, n)
    return [(L, float(gamma))]


def exchange(n: int, gamma: float, edges: Sequence[Tuple[int, int]]
             ) -> List[Tuple[np.ndarray, float]]:
    """B4: exchange dissipation ``L_ij = sigma^-_i sigma^+_j`` on selected edges
    (dissipative transport of excitations)."""
    return [(two_site_op(SMINUS, i, SPLUS, j, n), float(gamma)) for i, j in edges]


def pair_loss(n: int, gamma: float, edges: Sequence[Tuple[int, int]]
              ) -> List[Tuple[np.ndarray, float]]:
    """B5 (optional): pair loss ``L_ij = sigma^-_i sigma^-_j``."""
    return [(two_site_op(SMINUS, i, SMINUS, j, n), float(gamma)) for i, j in edges]


def graph_edges(J: np.ndarray, eps: float = 1e-9) -> List[Tuple[int, int]]:
    """Directed edges ``(i, j)`` for every nonzero coupling ``J_ij`` (i<j gives
    both orientations for exchange)."""
    n = J.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(J[i, j]) > eps:
                edges.append((i, j))
    return edges

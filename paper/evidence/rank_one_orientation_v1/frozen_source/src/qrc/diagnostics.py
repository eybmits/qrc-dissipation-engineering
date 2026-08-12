"""Diagnostic quantities for dissipative reservoirs (Phase 4).

These connect dynamical properties of the Liouvillian to task performance:
spectral gap, mixing time, trace-distance forgetting, stationary-state
separability, feature-matrix effective rank and condition number.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .liouvillian import steady_state


def liouvillian_spectrum(Lsuper: np.ndarray) -> np.ndarray:
    """Eigenvalues of the Liouvillian, sorted by descending real part."""
    w = np.linalg.eigvals(Lsuper)
    return w[np.argsort(-w.real)]


def spectral_gap(Lsuper: np.ndarray, zero_tol: float = 1e-9) -> float:
    """Asymptotic decay rate ``Delta_L = -Re(lambda_1)``.

    ``lambda_1`` is the eigenvalue with the largest real part among the
    non-zero modes (the zero mode is the stationary state).  Larger gap => faster
    relaxation / shorter memory.
    """
    w = liouvillian_spectrum(Lsuper)
    for lam in w:
        if abs(lam.real) > zero_tol or abs(lam.imag) > zero_tol:
            # skip the (near) zero stationary eigenvalue(s)
            if abs(lam) > zero_tol:
                return float(-lam.real)
    # all eigenvalues ~ 0
    return 0.0


def mixing_time(Lsuper: np.ndarray) -> float:
    """Mixing-time proxy ``t_mix ~ 1 / Delta_L``."""
    gap = spectral_gap(Lsuper)
    return float("inf") if gap <= 0 else 1.0 / gap


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance ``0.5 * ||rho - sigma||_1`` (sum of singular values)."""
    diff = rho - sigma
    sv = np.linalg.svd(diff, compute_uv=False)
    return float(0.5 * np.sum(sv))


def trace_distance_decay(
    reservoir,
    inputs: np.ndarray,
    rho_a: np.ndarray,
    rho_b: np.ndarray,
) -> np.ndarray:
    """Evolve two initial states under the *same* input sequence and return the
    trace distance ``||rho_t^a - rho_t^b||_1 / 2`` at every step (echo-state /
    fading-memory probe)."""
    ra, rb = rho_a.copy(), rho_b.copy()
    out = np.empty(len(inputs), dtype=float)
    for k, s in enumerate(inputs):
        ra = reservoir.step(ra, float(s))
        rb = reservoir.step(rb, float(s))
        out[k] = trace_distance(ra, rb)
    return out


def stationary_separability(reservoir, s_values: List[float]) -> np.ndarray:
    """Pairwise trace distance between Liouvillian stationary states for a set of
    constant inputs.  Returns a symmetric matrix.  Requires the reservoir to
    expose ``liouvillian(s)`` (Lindblad reservoirs)."""
    states = [steady_state(reservoir.liouvillian(float(s))) for s in s_values]
    n = len(states)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = trace_distance(states[i], states[j])
    return D


def unitality_defect(jumps) -> float:
    """`‖ Σ_k rate_k [L_k, L_k^dag] ‖_F` — the dissipator's deviation from unitality.

    A dissipator is *unital* (`D[I]=0`) iff this is zero, since
    `D[L](I) = L L^dag - L^dag L = [L, L^dag]`.  Unital dissipators (dephasing
    `σ^z`, exchange `σ^-σ^+`) relax to the maximally mixed state `I/2^N`, which no
    Hamiltonian input drive can shift — so they are **input-blind and useless for
    QRC**.  Non-unital dissipators (amplitude damping `σ^-`, collective `Σc_iσ^-`)
    have a non-trivial steady state and can encode the input.  This predicts
    QRC-usefulness from the jump operators alone, before any simulation.
    """
    if not jumps:
        return 0.0
    d = np.asarray(jumps[0][0]).shape[0]
    acc = np.zeros((d, d), dtype=complex)
    for L, rate in jumps:
        L = np.asarray(L, dtype=complex)
        acc += rate * (L @ L.conj().T - L.conj().T @ L)
    return float(np.linalg.norm(acc, "fro"))


def effective_rank(X: np.ndarray, center: bool = True) -> float:
    """Effective rank of a feature matrix from the entropy of its singular-value
    spectrum: ``exp(-sum p_i log p_i)`` with ``p_i = s_i / sum s_j``."""
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    sv = np.linalg.svd(Xc, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def condition_number(X: np.ndarray, center: bool = True) -> float:
    """Condition number (ratio of largest to smallest non-negligible singular
    value) of the feature matrix."""
    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    sv = np.linalg.svd(Xc, compute_uv=False)
    sv = sv[sv > 1e-12]
    if sv.size == 0:
        return float("inf")
    return float(sv[0] / sv[-1])


def summarize_features(X: np.ndarray) -> dict:
    """Bundle of feature-matrix diagnostics."""
    return {
        "n_features": int(X.shape[1]),
        "effective_rank": effective_rank(X),
        "condition_number": condition_number(X),
    }

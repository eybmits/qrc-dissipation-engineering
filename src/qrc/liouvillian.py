"""Vectorised Liouvillian and Lindblad superoperators.

Vectorisation convention
-------------------------
We use **column-stacking** ``vec(rho)`` (numpy ``order='F'``), for which

    vec(A rho B) = (B^T (x) A) vec(rho).

Hence, for a density matrix of dimension ``d`` the superoperators are ``d^2 x d^2``:

* Hamiltonian (commutator):
      -i[H, rho]  ->  -i (I (x) H - H^T (x) I)
* Lindblad dissipator for a jump operator ``L``:
      D[L] rho = L rho L^dag - 1/2 {L^dag L, rho}
      ->  conj(L) (x) L  -  1/2 (I (x) L^dag L)  -  1/2 ((L^dag L)^T (x) I)

The full Liouvillian is ``Lsuper = H_super + sum_k gamma_k D[L_k]_super`` and
the discrete-time propagator over a step ``dt`` is ``expm(Lsuper * dt)``.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply


def vec(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorisation."""
    return rho.reshape(-1, order="F")


def unvec(v: np.ndarray, d: int) -> np.ndarray:
    """Inverse of :func:`vec` for a ``d x d`` matrix."""
    return v.reshape((d, d), order="F")


def commutator_super(H: np.ndarray) -> np.ndarray:
    """Superoperator for ``-i[H, rho]`` (column-stacking convention)."""
    d = H.shape[0]
    Id = np.eye(d, dtype=complex)
    return -1j * (np.kron(Id, H) - np.kron(H.T, Id))


def dissipator_super(L: np.ndarray) -> np.ndarray:
    """Superoperator for the Lindblad dissipator ``D[L]`` (rate 1)."""
    d = L.shape[0]
    Id = np.eye(d, dtype=complex)
    LdL = L.conj().T @ L
    return (
        np.kron(L.conj(), L)
        - 0.5 * np.kron(Id, LdL)
        - 0.5 * np.kron(LdL.T, Id)
    )


def lindbladian(
    H: np.ndarray,
    jumps: Iterable[Tuple[np.ndarray, float]] = (),
) -> np.ndarray:
    """Full Liouvillian superoperator.

    Parameters
    ----------
    H : Hamiltonian (Hermitian ``d x d``).
    jumps : iterable of ``(L, rate)`` pairs; the dissipator contribution is
        ``rate * D[L]``.  The rate is folded in here (``L`` itself is the bare
        jump operator).
    """
    d = H.shape[0]
    Lsuper = commutator_super(H).astype(complex)
    for L, rate in jumps:
        if rate == 0:
            continue
        Lsuper = Lsuper + rate * dissipator_super(np.asarray(L, dtype=complex))
    return Lsuper


def propagator(Lsuper: np.ndarray, dt: float) -> np.ndarray:
    """Discrete-time propagator ``expm(Lsuper * dt)`` acting on ``vec(rho)``."""
    return expm(Lsuper * dt)


def evolve_step(Lsuper: np.ndarray, rho: np.ndarray, dt: float) -> np.ndarray:
    """Evolve one density matrix by ``dt`` using the action of the matrix exp.

    Uses :func:`scipy.sparse.linalg.expm_multiply` on the vectorised state,
    which avoids forming the (dense) full propagator and is the right tool when
    the Liouvillian changes every step (input-dependent Hamiltonian).
    """
    d = rho.shape[0]
    v = vec(rho)
    out = expm_multiply(Lsuper * dt, v)
    return unvec(out, d)


def apply_propagator(P: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply a precomputed propagator ``P`` to ``rho`` (returns a density matrix)."""
    d = rho.shape[0]
    return unvec(P @ vec(rho), d)


def steady_state(Lsuper: np.ndarray) -> np.ndarray:
    """Stationary state ``rho_ss`` solving ``Lsuper vec(rho_ss) = 0``.

    Returned normalised to unit trace and Hermitised.  Uses the eigenvector of
    ``Lsuper`` with eigenvalue closest to zero.
    """
    d = int(round(np.sqrt(Lsuper.shape[0])))
    w, v = np.linalg.eig(Lsuper)
    idx = int(np.argmin(np.abs(w)))
    rho = unvec(v[:, idx], d)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) < 1e-12:
        raise RuntimeError("steady state has ~zero trace; degenerate Liouvillian")
    return rho / tr

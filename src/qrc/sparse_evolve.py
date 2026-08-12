"""Sparse Lindblad superoperators and matrix-free reservoir evolution.

This module mirrors :mod:`qrc.liouvillian` exactly in column-stacking convention,
but returns CSR sparse matrices and evolves states with ``expm_multiply`` on the
sparse Liouvillian.  It is intentionally additive: the validated dense builders
and :class:`qrc.reservoirs.LindbladReservoir` are not modified.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .liouvillian import unvec, vec
from .readout import Observable
from .reservoirs import Reservoir


def _as_csr(A) -> sparse.csr_matrix:
    """Return ``A`` as a complex CSR matrix without forming any dense superop."""
    if sparse.issparse(A):
        return A.astype(complex).tocsr()
    return sparse.csr_matrix(np.asarray(A, dtype=complex))


def commutator_super(H) -> sparse.csr_matrix:
    """Sparse superoperator for ``-i[H, rho]``.

    Column-stacking convention:
    ``vec(A rho B) = (B^T kron A) vec(rho)``.
    """
    Hs = _as_csr(H)
    d = Hs.shape[0]
    Id = sparse.identity(d, dtype=complex, format="csr")
    out = -1j * (
        sparse.kron(Id, Hs, format="csr")
        - sparse.kron(Hs.T, Id, format="csr")
    )
    out.eliminate_zeros()
    return out.tocsr()


def dissipator_super(L) -> sparse.csr_matrix:
    """Sparse superoperator for the rate-1 Lindblad dissipator ``D[L]``."""
    Ls = _as_csr(L)
    d = Ls.shape[0]
    Id = sparse.identity(d, dtype=complex, format="csr")
    LdL = (Ls.conjugate().T @ Ls).tocsr()
    out = (
        sparse.kron(Ls.conjugate(), Ls, format="csr")
        - 0.5 * sparse.kron(Id, LdL, format="csr")
        - 0.5 * sparse.kron(LdL.T, Id, format="csr")
    )
    out.eliminate_zeros()
    return out.tocsr()


def lindbladian(
    H,
    jumps: Iterable[Tuple[np.ndarray, float]] = (),
) -> sparse.csr_matrix:
    """Sparse full Liouvillian ``-i[H,.] + sum_k rate_k D[L_k]``."""
    Lsuper = commutator_super(H)
    for L, rate in jumps:
        if rate == 0:
            continue
        Lsuper = Lsuper + float(rate) * dissipator_super(L)
    Lsuper.eliminate_zeros()
    return Lsuper.tocsr()


class SparseLindbladReservoir(Reservoir):
    """Input-affine Lindblad reservoir backed by sparse ``expm_multiply``.

    The Hamiltonian convention matches ``LindbladReservoir.from_terms``:
    ``H(s) = H_static + s * H_drive``.  No input quantisation or propagator cache
    is used; every step evolves with the true continuous input value.
    """

    def __init__(
        self,
        n_qubits: int,
        base_super,
        drive_super,
        dt: float,
    ):
        self.n_qubits = n_qubits
        self.base_super = _as_csr(base_super)
        self.drive_super = None if drive_super is None else _as_csr(drive_super)
        self.dt = float(dt)

    @classmethod
    def from_terms(
        cls,
        n_qubits: int,
        H_static,
        H_drive,
        jumps: Sequence[Tuple[np.ndarray, float]],
        dt: float,
    ) -> "SparseLindbladReservoir":
        base = lindbladian(H_static, jumps)
        drive = None if H_drive is None else commutator_super(H_drive)
        return cls(n_qubits, base, drive, dt)

    def liouvillian(self, s: float) -> sparse.csr_matrix:
        if self.drive_super is None or s == 0:
            return self.base_super
        out = self.base_super + float(s) * self.drive_super
        out.eliminate_zeros()
        return out.tocsr()

    def _evolve_vec(self, v: np.ndarray, s: float, dt: float) -> np.ndarray:
        return expm_multiply(self.liouvillian(s) * dt, v)

    def step(self, rho: np.ndarray, s: float) -> np.ndarray:
        d = rho.shape[0]
        out = self._evolve_vec(vec(rho), float(s), self.dt)
        return unvec(out, d)

    def substep_states(
        self,
        rho: np.ndarray,
        s: float,
        n_virtual: int,
    ) -> list[np.ndarray]:
        d = rho.shape[0]
        sub_dt = self.dt / n_virtual
        cur = vec(rho)
        states: list[np.ndarray] = []
        Ls = self.liouvillian(float(s))
        A = Ls * sub_dt
        for _ in range(n_virtual):
            cur = expm_multiply(A, cur)
            states.append(unvec(cur, d))
        return states


def run_sparse_lindblad_trajectory(
    n_qubits: int,
    H_static,
    H_drive,
    jumps: Sequence[Tuple[np.ndarray, float]],
    dt: float,
    inputs: Sequence[float],
    observables: Sequence[Observable],
    washout: int = 0,
    n_virtual: int = 1,
    rho0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run a sparse matrix-free Lindblad trajectory and return readout features."""
    reservoir = SparseLindbladReservoir.from_terms(
        n_qubits, H_static, H_drive, jumps, dt
    )
    return reservoir.run(inputs, observables, washout=washout,
                         n_virtual=n_virtual, rho0=rho0)

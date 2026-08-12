"""Pauli algebra, tensor embedding, Pauli strings and partial trace.

Conventions
-----------
* Qubits are indexed ``0 .. N-1``.  Qubit ``0`` is the *leftmost* factor in the
  Kronecker product, i.e. an operator acting on site ``i`` of an ``N``-qubit
  register is ``I^{(0)} (x) ... (x) op^{(i)} (x) ... (x) I^{(N-1)}``.
* The **input qubit is qubit 0** (this is where the input is injected in both
  baseline models).
* Computational basis ``|0> = (1, 0)^T``, ``|1> = (0, 1)^T``.
* Lowering operator ``sminus = |0><1|`` maps ``|1> -> |0>`` (energy loss toward
  ``|0>``); raising ``splus = |1><0|``.  Hence ``splus @ sminus = |1><1|`` is the
  number operator and the amplitude-damping channel relaxes toward ``|0>``.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterable, Sequence

import numpy as np

# --- single-qubit operators ------------------------------------------------

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SMINUS = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|, lowering (loss)
SPLUS = np.array([[0, 0], [1, 0]], dtype=complex)   # |1><0|, raising

PAULI = {"i": I2, "x": SX, "y": SY, "z": SZ}


def kron_list(ops: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product of a sequence of operators (left-to-right)."""
    return reduce(np.kron, ops)


def embed(op: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit ``op`` acting on ``site`` into an ``n_qubits`` register."""
    if not 0 <= site < n_qubits:
        raise ValueError(f"site {site} out of range for {n_qubits} qubits")
    factors = [I2] * n_qubits
    factors[site] = op
    return kron_list(factors)


def pauli_op(axis: str, site: int, n_qubits: int) -> np.ndarray:
    """Single-site Pauli operator ``sigma^axis_site`` on an ``n_qubits`` register."""
    return embed(PAULI[axis.lower()], site, n_qubits)


def two_site_pauli(axis: str, i: int, j: int, n_qubits: int) -> np.ndarray:
    """Two-site Pauli ``sigma^axis_i sigma^axis_j`` (same axis on both sites)."""
    if i == j:
        raise ValueError("two_site_pauli requires i != j")
    a = PAULI[axis.lower()]
    factors = [I2] * n_qubits
    factors[i] = a
    factors[j] = a
    return kron_list(factors)


def sminus(site: int, n_qubits: int) -> np.ndarray:
    return embed(SMINUS, site, n_qubits)


def splus(site: int, n_qubits: int) -> np.ndarray:
    return embed(SPLUS, site, n_qubits)


def two_site_op(opA: np.ndarray, i: int, opB: np.ndarray, j: int, n_qubits: int) -> np.ndarray:
    """Product ``opA_i opB_j`` of two single-site operators on different sites
    (used for exchange / pair jump operators, e.g. ``sigma^-_i sigma^+_j``)."""
    if i == j:
        raise ValueError("two_site_op requires i != j")
    factors = [I2] * n_qubits
    factors[i] = opA
    factors[j] = opB
    return kron_list(factors)


def pauli_string(axes: Iterable[str], n_qubits: int) -> np.ndarray:
    """Build a full Pauli string from an iterable of ``axes`` of length ``n_qubits``.

    Example: ``pauli_string(["x", "i", "z"], 3)`` -> sigma^x_0 sigma^z_2.
    """
    axes = list(axes)
    if len(axes) != n_qubits:
        raise ValueError("axes length must equal n_qubits")
    return kron_list([PAULI[a.lower()] for a in axes])


# --- input state -----------------------------------------------------------

def input_state(s: float) -> np.ndarray:
    """Single-qubit input density matrix |psi_s><psi_s| with
    ``|psi_s> = sqrt(1-s)|0> + sqrt(s)|1>`` (requires ``0 <= s <= 1``)."""
    if s < -1e-12 or s > 1 + 1e-12:
        raise ValueError(f"input s={s} must lie in [0, 1]")
    s = float(np.clip(s, 0.0, 1.0))
    psi = np.array([np.sqrt(1.0 - s), np.sqrt(s)], dtype=complex)
    return np.outer(psi, psi.conj())


# --- partial trace ---------------------------------------------------------

def partial_trace(rho: np.ndarray, keep: Sequence[int], n_qubits: int) -> np.ndarray:
    """Partial trace of ``rho`` keeping qubits in ``keep`` (order preserved).

    Parameters
    ----------
    rho : (2**n, 2**n) density matrix.
    keep : qubit indices to keep; the rest are traced out.
    n_qubits : total number of qubits.
    """
    keep = sorted(keep)
    traced = [q for q in range(n_qubits) if q not in keep]
    # reshape into rank-2N tensor: row indices then column indices
    tensor = rho.reshape([2] * (2 * n_qubits))
    # trace out each unwanted qubit by pairing its row and column axes
    # process from the highest index to keep axis bookkeeping simple
    for q in sorted(traced, reverse=True):
        tensor = np.trace(tensor, axis1=q, axis2=q + n_qubits)
        n_qubits -= 1
    d_keep = 2 ** len(keep)
    return tensor.reshape(d_keep, d_keep)


def trace_out(rho: np.ndarray, discard: Sequence[int], n_qubits: int) -> np.ndarray:
    """Convenience wrapper: trace out the qubits in ``discard``."""
    keep = [q for q in range(n_qubits) if q not in discard]
    return partial_trace(rho, keep, n_qubits)

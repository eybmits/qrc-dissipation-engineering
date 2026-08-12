"""Reservoir models.

* :class:`FujiNakajimaReservoir` - unitary, input-by-reset baseline.
* :class:`ResetLindbladReservoir` - input-by-reset followed by autonomous
  Lindblad evolution.
* :class:`LindbladReservoir`     - generic dissipative reservoir with an
  input-affine Liouvillian ``L(s) = base + s * drive``; basis for Track A/B.
* :func:`continuous_dissipation` - the paper's continuous-dissipation (CD) model.

Both reservoir families share the :meth:`Reservoir.run` interface, which streams
an input sequence and returns the readout feature matrix (with optional time
multiplexing into ``n_virtual`` virtual nodes per input step).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm

from .operators import (
    SMINUS,
    embed,
    input_state,
    pauli_op,
    sminus,
    trace_out,
    two_site_pauli,
)
from .liouvillian import (
    apply_propagator,
    commutator_super,
    dissipator_super,
    evolve_step,
    lindbladian,
    propagator,
    vec,
    unvec,
)
from .readout import Observable, features_from_states


# --- Hamiltonian building blocks ------------------------------------------

def random_couplings(n_qubits: int, J_s: float, rng: np.random.Generator) -> np.ndarray:
    """Symmetric coupling matrix with ``J_ij ~ U[-J_s, J_s]`` (zero diagonal)."""
    J = np.zeros((n_qubits, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            J[i, j] = J[j, i] = rng.uniform(-J_s, J_s)
    return J


def ising_xx_hamiltonian(J: np.ndarray, h: float, n_qubits: int) -> np.ndarray:
    """``H0 = sum_{i<j} J_ij sigma^x_i sigma^x_j + h sum_i sigma^z_i``."""
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if J[i, j] != 0:
                H += J[i, j] * two_site_pauli("x", i, j, n_qubits)
    for i in range(n_qubits):
        H += h * pauli_op("z", i, n_qubits)
    return H


def transverse_drive(n_qubits: int) -> np.ndarray:
    """``sum_i sigma^x_i`` (the operator multiplying the input drive)."""
    d = 2 ** n_qubits
    Hx = np.zeros((d, d), dtype=complex)
    for i in range(n_qubits):
        Hx += pauli_op("x", i, n_qubits)
    return Hx


# --- base class ------------------------------------------------------------

class Reservoir:
    n_qubits: int

    def initial_state(self) -> np.ndarray:
        d = 2 ** self.n_qubits
        rho = np.zeros((d, d), dtype=complex)
        rho[0, 0] = 1.0  # |0...0>
        return rho

    # subclasses implement these two:
    def step(self, rho: np.ndarray, s: float) -> np.ndarray:
        raise NotImplementedError

    def substep_states(self, rho: np.ndarray, s: float, n_virtual: int) -> List[np.ndarray]:
        """Return the reservoir state at each of ``n_virtual`` virtual nodes within
        one input step.  The last element is the state at the end of the step."""
        raise NotImplementedError

    def run(
        self,
        inputs: Sequence[float],
        observables: Sequence[Observable],
        washout: int = 0,
        n_virtual: int = 1,
        rho0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Stream ``inputs`` and return the feature matrix for steps after washout.

        Returned shape: ``(len(inputs) - washout, n_virtual * len(observables))``.
        Row ``t`` corresponds to input index ``washout + t``.
        """
        rho = self.initial_state() if rho0 is None else rho0.copy()
        # stack observables: Tr(rho O_k) = sum_ij rho_ij O_k,ji  (einsum, vectorised)
        Omats = np.stack([o.matrix for o in observables])  # (n_obs, d, d)
        n_obs = Omats.shape[0]
        T = len(inputs)
        feats = np.empty((T - washout, n_virtual * n_obs), dtype=float)
        for k, s in enumerate(inputs):
            states = self.substep_states(rho, float(s), n_virtual)
            rho = states[-1]
            if k >= washout:
                row = feats[k - washout]
                for v, st in enumerate(states):
                    row[v * n_obs:(v + 1) * n_obs] = np.real(
                        np.einsum("kij,ji->k", Omats, st)
                    )
        return feats


# --- Fuji-Nakajima reservoir ----------------------------------------------

def inject_input(
    rho: np.ndarray,
    s: float,
    n_qubits: int,
) -> np.ndarray:
    """Reset qubit 0 to ``|psi_s><psi_s|`` while retaining the other qubits.

    This is the input map shared by the unitary Fuji--Nakajima reservoir and
    the dissipative reset reservoir below.  Qubit 0 is the leftmost tensor
    factor, following :mod:`qrc.operators`.
    """
    rho_rest = trace_out(rho, discard=[0], n_qubits=n_qubits)
    return np.kron(input_state(s), rho_rest)


class FujiNakajimaReservoir(Reservoir):
    """Unitary baseline: input is injected by resetting qubit 0 to
    ``|psi_s> = sqrt(1-s)|0> + sqrt(s)|1>``, followed by unitary evolution
    ``U = exp(-i H dt)`` with the transverse-field Ising Hamiltonian."""

    def __init__(self, n_qubits: int, J: np.ndarray, h: float, dt: float):
        self.n_qubits = n_qubits
        self.J = J
        self.h = h
        self.dt = dt
        self.H = ising_xx_hamiltonian(J, h, n_qubits)
        self.U = expm(-1j * self.H * dt)
        self._U_sub_cache: dict[int, np.ndarray] = {1: self.U}

    def _U_sub(self, n_virtual: int) -> np.ndarray:
        if n_virtual not in self._U_sub_cache:
            self._U_sub_cache[n_virtual] = expm(-1j * self.H * (self.dt / n_virtual))
        return self._U_sub_cache[n_virtual]

    def _inject(self, rho: np.ndarray, s: float) -> np.ndarray:
        return inject_input(rho, s, self.n_qubits)

    def step(self, rho: np.ndarray, s: float) -> np.ndarray:
        new = self._inject(rho, s)
        return self.U @ new @ self.U.conj().T

    def substep_states(self, rho: np.ndarray, s: float, n_virtual: int) -> List[np.ndarray]:
        new = self._inject(rho, s)
        if n_virtual == 1:
            return [self.U @ new @ self.U.conj().T]
        Us = self._U_sub(n_virtual)
        Usd = Us.conj().T
        states = []
        cur = new
        for _ in range(n_virtual):
            cur = Us @ cur @ Usd
            states.append(cur)
        return states


# --- dissipative input-by-reset reservoir ---------------------------------

class ResetLindbladReservoir(Reservoir):
    """Input-by-reset followed by autonomous Lindblad evolution.

    Each input step first applies the same reset map as
    :class:`FujiNakajimaReservoir`,

    ``rho -> |psi_s><psi_s| (x) Tr_0(rho)``,

    and then evolves for ``dt`` under a static Hamiltonian and static jump
    operators.  The time-independent propagator is cached once, making this
    class suitable for continuous-valued reset inputs without quantisation.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Qubit 0 receives the input reset.
    H:
        Autonomous Hermitian Hamiltonian.
    jumps:
        Sequence of ``(L, rate)`` pairs in the repository's standard
        Lindblad convention.
    dt:
        Evolution time after every reset.
    """

    def __init__(
        self,
        n_qubits: int,
        H: np.ndarray,
        jumps: Sequence[Tuple[np.ndarray, float]],
        dt: float,
    ):
        self.n_qubits = int(n_qubits)
        self.H = np.asarray(H, dtype=complex)
        self.dt = float(dt)
        d = 2 ** self.n_qubits
        if self.H.shape != (d, d):
            raise ValueError(f"H must have shape {(d, d)}")
        if not np.allclose(self.H, self.H.conj().T, atol=1e-12, rtol=0.0):
            raise ValueError("H must be Hermitian")
        if self.dt <= 0:
            raise ValueError("dt must be positive")

        self.jumps: List[Tuple[np.ndarray, float]] = []
        for jump, rate in jumps:
            jump = np.asarray(jump, dtype=complex)
            rate = float(rate)
            if jump.shape != (d, d):
                raise ValueError(f"jump operator must have shape {(d, d)}")
            if rate < 0:
                raise ValueError("jump rates must be non-negative")
            self.jumps.append((jump, rate))

        self.base_super = lindbladian(self.H, self.jumps)
        self._prop_cache: dict[float, np.ndarray] = {}
        self.P = self._sub_propagator(self.dt)

    def _sub_propagator(self, sub_dt: float) -> np.ndarray:
        key = round(float(sub_dt), 12)
        P = self._prop_cache.get(key)
        if P is None:
            P = propagator(self.base_super, sub_dt)
            self._prop_cache[key] = P
        return P

    def _inject(self, rho: np.ndarray, s: float) -> np.ndarray:
        return inject_input(rho, s, self.n_qubits)

    def step(self, rho: np.ndarray, s: float) -> np.ndarray:
        return apply_propagator(self.P, self._inject(rho, s))

    def substep_states(
        self,
        rho: np.ndarray,
        s: float,
        n_virtual: int,
    ) -> List[np.ndarray]:
        if n_virtual < 1:
            raise ValueError("n_virtual must be at least 1")
        P = self._sub_propagator(self.dt / n_virtual)
        states = []
        cur = self._inject(rho, s)
        for _ in range(n_virtual):
            cur = apply_propagator(P, cur)
            states.append(cur)
        return states


def dissipative_input_reset(
    n_qubits: int,
    J: np.ndarray,
    h: float,
    gamma: float,
    dt: float,
    jump_family: str = "local",
    collective_coefficients: Optional[Sequence[float]] = None,
) -> ResetLindbladReservoir:
    """Build the reset-encoded local or collective relaxation reservoir.

    The coherent processor is the autonomous ``XX+Z`` Hamiltonian used by the
    Fuji--Nakajima baseline.  ``jump_family`` is either ``"local"`` for
    independent lowering operators or ``"collective"`` for one equal-phase
    (or explicitly weighted) collective lowering operator.  Both families use
    the existing jump builders and therefore share the repository's structural
    budget convention.
    """
    from .dissipators import collective_loss, local_loss

    family = jump_family.lower()
    if family == "local":
        if collective_coefficients is not None:
            raise ValueError(
                "collective_coefficients are only valid for collective loss"
            )
        jumps = local_loss(n_qubits, gamma)
    elif family == "collective":
        jumps = collective_loss(
            n_qubits,
            gamma,
            c=collective_coefficients,
        )
    else:
        raise ValueError("jump_family must be 'local' or 'collective'")

    H = ising_xx_hamiltonian(J, h, n_qubits)
    return ResetLindbladReservoir(n_qubits, H, jumps, dt)


# --- generic Lindblad reservoir -------------------------------------------

class LindbladReservoir(Reservoir):
    """Dissipative reservoir with input-affine Liouvillian
    ``L(s) = base_super + s * drive_super``.

    For continuous inputs, the action of the matrix exponential is computed per
    step (``expm_multiply``).  For discrete inputs (e.g. parity), enable
    ``cache_propagators`` to precompute and reuse the dense propagators.
    """

    def __init__(
        self,
        n_qubits: int,
        base_super: np.ndarray,
        drive_super: Optional[np.ndarray],
        dt: float,
        cache_propagators: bool = False,
        quantize: Optional[int] = None,
    ):
        self.n_qubits = n_qubits
        self.base_super = base_super
        self.drive_super = drive_super
        self.dt = dt
        # Quantising the input to ``quantize`` levels lets the dense-propagator
        # cache be reused across steps (huge speed-up for continuous inputs).
        # It snaps the *reservoir drive* to a grid; targets use the true input.
        self.quantize = quantize
        self.cache_propagators = cache_propagators or (quantize is not None)
        self._prop_cache: dict[Tuple[float, float], np.ndarray] = {}

    def _snap(self, s: float) -> float:
        if self.quantize is None:
            return s
        b = self.quantize - 1
        return round(s * b) / b

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_terms(
        cls,
        n_qubits: int,
        H_static: np.ndarray,
        H_drive: Optional[np.ndarray],
        jumps: Sequence[Tuple[np.ndarray, float]],
        dt: float,
        cache_propagators: bool = False,
        quantize: Optional[int] = None,
    ) -> "LindbladReservoir":
        base = commutator_super(H_static).astype(complex)
        for L, rate in jumps:
            if rate:
                base = base + rate * dissipator_super(np.asarray(L, dtype=complex))
        drive = None if H_drive is None else commutator_super(H_drive)
        return cls(n_qubits, base, drive, dt, cache_propagators, quantize)

    # -- dynamics ------------------------------------------------------------
    def liouvillian(self, s: float) -> np.ndarray:
        if self.drive_super is None or s == 0:
            return self.base_super
        return self.base_super + s * self.drive_super

    def _sub_propagator(self, s: float, sub_dt: float) -> np.ndarray:
        key = (round(float(s), 12), round(float(sub_dt), 12))
        P = self._prop_cache.get(key)
        if P is None:
            P = propagator(self.liouvillian(s), sub_dt)
            self._prop_cache[key] = P
        return P

    def step(self, rho: np.ndarray, s: float) -> np.ndarray:
        s = self._snap(s)
        if self.cache_propagators:
            return apply_propagator(self._sub_propagator(s, self.dt), rho)
        return evolve_step(self.liouvillian(s), rho, self.dt)

    def substep_states(self, rho: np.ndarray, s: float, n_virtual: int) -> List[np.ndarray]:
        s = self._snap(s)
        sub_dt = self.dt / n_virtual
        states = []
        cur = rho
        if self.cache_propagators:
            P = self._sub_propagator(s, sub_dt)
            for _ in range(n_virtual):
                cur = apply_propagator(P, cur)
                states.append(cur)
        else:
            L = self.liouvillian(s)
            for _ in range(n_virtual):
                cur = evolve_step(L, cur, sub_dt)
                states.append(cur)
        return states


# --- general reservoir with input-dependent jumps -------------------------

class GeneralLindbladReservoir(Reservoir):
    """Most general reservoir: the Hamiltonian and the jump operators may both
    depend on the input ``s``.

    Parameters
    ----------
    n_qubits : int
    hamiltonian_fn : callable ``s -> H`` (Hermitian).
    jump_fn : callable ``s -> list[(L, rate)]``.
    dt : float
    quantize : optional input quantisation (enables propagator caching).

    This backs Track-A4 (input-adaptive rates ``gamma_i(s)``) and any Track-B
    dissipator whose rate is input-modulated.  For purely static jumps prefer the
    faster affine :class:`LindbladReservoir`.
    """

    def __init__(self, n_qubits, hamiltonian_fn, jump_fn, dt,
                 quantize: Optional[int] = None):
        self.n_qubits = n_qubits
        self.hamiltonian_fn = hamiltonian_fn
        self.jump_fn = jump_fn
        self.dt = dt
        self.quantize = quantize
        self._prop_cache: dict = {}

    def _snap(self, s):
        if self.quantize is None:
            return s
        b = self.quantize - 1
        return round(s * b) / b

    def liouvillian(self, s):
        from .liouvillian import lindbladian
        return lindbladian(self.hamiltonian_fn(s), self.jump_fn(s))

    def _sub_propagator(self, s, sub_dt):
        key = (round(float(s), 12), round(float(sub_dt), 12))
        P = self._prop_cache.get(key)
        if P is None:
            P = propagator(self.liouvillian(s), sub_dt)
            self._prop_cache[key] = P
        return P

    def step(self, rho, s):
        s = self._snap(s)
        if self.quantize is not None:
            return apply_propagator(self._sub_propagator(s, self.dt), rho)
        return evolve_step(self.liouvillian(s), rho, self.dt)

    def substep_states(self, rho, s, n_virtual):
        s = self._snap(s)
        sub_dt = self.dt / n_virtual
        states, cur = [], rho
        if self.quantize is not None:
            P = self._sub_propagator(s, sub_dt)
            for _ in range(n_virtual):
                cur = apply_propagator(P, cur); states.append(cur)
        else:
            L = self.liouvillian(s)
            for _ in range(n_virtual):
                cur = evolve_step(L, cur, sub_dt); states.append(cur)
        return states


# --- continuous-dissipation (CD) factory ----------------------------------

def continuous_dissipation(
    n_qubits: int,
    J: np.ndarray,
    h: float,
    gamma,
    dt: float,
    cache_propagators: bool = False,
    quantize: Optional[int] = None,
) -> LindbladReservoir:
    """The paper's CD model.

    Hamiltonian ``H'_k = H0 + h (s_k + 1) sum_i sigma^x_i`` with
    ``H0 = sum_{i<j} J_ij sigma^x_i sigma^x_j + h sum_i sigma^z_i``, and local
    amplitude-damping jumps ``L_i = sigma^-_i`` with rate(s) ``gamma`` (scalar or
    per-qubit array).
    """
    H0 = ising_xx_hamiltonian(J, h, n_qubits)
    Hx = transverse_drive(n_qubits)
    H_static = H0 + h * Hx          # s-independent part (includes the +1 offset)
    H_drive = h * Hx                # multiplied by s_k
    if np.isscalar(gamma):
        rates = [float(gamma)] * n_qubits
    else:
        rates = list(np.asarray(gamma, dtype=float))
        if len(rates) != n_qubits:
            raise ValueError("gamma array must have length n_qubits")
    jumps = [(sminus(i, n_qubits), rates[i]) for i in range(n_qubits)]
    return LindbladReservoir.from_terms(
        n_qubits, H_static, H_drive, jumps, dt, cache_propagators, quantize
    )

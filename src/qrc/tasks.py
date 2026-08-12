"""Temporal benchmark tasks: STM, NARMA, parity, Mackey-Glass.

Each generator returns a driving-input sequence ``s_k`` that is valid as a
reservoir input (``s_k in [0, 1]``) together with the corresponding target(s).
Tasks whose target depends on a delay/order expose helper functions so the same
input sequence can be reused across delays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# --- Short-Term Memory (STM) ----------------------------------------------

def stm_inputs(T: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform i.i.d. inputs ``s_k ~ U[0, 1]``."""
    return rng.uniform(0.0, 1.0, size=T)


def delayed_target(inputs: np.ndarray, tau: int) -> np.ndarray:
    """Target ``y_k = s_{k-tau}`` (the first ``tau`` entries are NaN/undefined)."""
    y = np.full_like(inputs, np.nan, dtype=float)
    if tau == 0:
        return inputs.astype(float).copy()
    y[tau:] = inputs[:-tau]
    return y


# --- NARMA -----------------------------------------------------------------

def narma_inputs(T: int, rng: np.random.Generator) -> np.ndarray:
    """Driving inputs for NARMA, ``s_k ~ U[0, 1]`` (reservoir injection range)."""
    return rng.uniform(0.0, 1.0, size=T)


def narma_target(
    inputs: np.ndarray,
    order: int = 10,
    input_scale: float = 0.2,
    clip: float = 1e3,
) -> np.ndarray:
    """NARMA-``order`` target driven by ``input_scale * inputs``.

        y_k = 0.3 y_{k-1}
              + 0.05 y_{k-1} sum_{j=1}^{n} y_{k-j}
              + 1.5 s_{k-n} s_{k-1}
              + 0.1

    The recursion is run on the rescaled input ``hat_s = input_scale * s`` to keep
    it bounded (the paper rescales the input to avoid divergence).  ``input_scale``
    should be made smaller for larger ``order``.
    """
    n = order
    s = input_scale * np.asarray(inputs, dtype=float)
    T = len(s)
    y = np.zeros(T, dtype=float)
    diverged = False
    for k in range(n, T):
        window = np.sum(y[k - n:k])
        y[k] = (
            0.3 * y[k - 1]
            + 0.05 * y[k - 1] * window
            + 1.5 * s[k - n] * s[k - 1]
            + 0.1
        )
        if not np.isfinite(y[k]) or abs(y[k]) > clip:
            diverged = True
            y[k] = np.clip(np.nan_to_num(y[k]), -clip, clip)
    y[:n] = np.nan  # undefined warm-up region
    if diverged:
        import warnings

        warnings.warn(
            f"NARMA-{order} diverged (input_scale={input_scale}); values clipped. "
            "Reduce input_scale.",
            RuntimeWarning,
        )
    return y


# --- Parity check ----------------------------------------------------------

def parity_inputs(T: int, rng: np.random.Generator) -> np.ndarray:
    """Binary i.i.d. inputs ``s_k in {0, 1}``."""
    return rng.integers(0, 2, size=T).astype(float)


def parity_target(inputs: np.ndarray, tau: int) -> np.ndarray:
    """Target ``y_k = (sum_{j=1}^{tau} s_{k-j}) mod 2`` (first ``tau`` undefined)."""
    s = np.asarray(inputs, dtype=int)
    T = len(s)
    y = np.full(T, np.nan, dtype=float)
    for k in range(tau, T):
        y[k] = np.sum(s[k - tau:k]) % 2
    return y


# --- Mackey-Glass ----------------------------------------------------------

def mackey_glass_series(
    n_samples: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_exp: int = 10,
    dt: float = 1.0,
    sample_every: int = 1,
    x0: float = 1.2,
    discard: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a Mackey-Glass time series.

        dx/dt = beta x(t - tau) / (1 + x(t - tau)^n) - gamma x(t)

    Integrated with a simple Euler scheme at step ``dt``; the continuous delay
    ``tau`` is converted to ``tau/dt`` integration steps.  ``sample_every``
    integration steps are collapsed into one returned sample.  The first
    ``discard`` samples are dropped as transient.  Output is **not** rescaled
    here (use :func:`rescale_unit`).
    """
    tau_steps = int(round(tau / dt))
    total = (n_samples + discard) * sample_every + tau_steps + 10
    x = np.empty(total + tau_steps, dtype=float)
    # history: small fluctuations around x0
    if rng is not None:
        x[:tau_steps + 1] = x0 + 0.01 * rng.standard_normal(tau_steps + 1)
    else:
        x[:tau_steps + 1] = x0
    for t in range(tau_steps, total + tau_steps - 1):
        xtau = x[t - tau_steps]
        x[t + 1] = x[t] + dt * (beta * xtau / (1.0 + xtau ** n_exp) - gamma * x[t])
    series = x[tau_steps::sample_every]
    series = series[discard:discard + n_samples]
    return series


def rescale_unit(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Affine rescale ``x`` to the interval ``[lo, hi]``."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-12:
        return np.full_like(x, 0.5 * (lo + hi))
    return lo + (hi - lo) * (x - xmin) / (xmax - xmin)


# --- container -------------------------------------------------------------

@dataclass
class TaskData:
    name: str
    inputs: np.ndarray
    targets: Dict[int, np.ndarray] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

"""Observables, feature extraction, linear readout training and metrics.

Readout features are the ideal expectation values of Pauli strings of weight 1
and weight 2 (same axis on both sites), matching the paper:

    <sigma_i^a>,   <sigma_i^a sigma_j^a>,   a in {x, y, z}.

Only the *linear* output weights are trained, by least squares (SVD / ridge).
Training, validation and test splits are always kept separate by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .operators import pauli_op, two_site_pauli


@dataclass
class Observable:
    name: str
    matrix: np.ndarray


def pauli_observables(
    n_qubits: int,
    axes: Sequence[str] = ("x", "y", "z"),
    max_weight: int = 2,
) -> List[Observable]:
    """All weight-1 and (same-axis) weight-2 Pauli observables.

    For ``N`` qubits and 3 axes this yields ``3N`` weight-1 terms and
    ``3 * C(N,2)`` weight-2 terms (e.g. ``N=5`` -> 15 + 30 = 45 features).
    """
    obs: List[Observable] = []
    for a in axes:
        for i in range(n_qubits):
            obs.append(Observable(f"{a}{i}", pauli_op(a, i, n_qubits)))
    if max_weight >= 2:
        for a in axes:
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    obs.append(
                        Observable(f"{a}{i}{a}{j}", two_site_pauli(a, i, j, n_qubits))
                    )
    return obs


def expectation(rho: np.ndarray, O: np.ndarray) -> float:
    """Real part of ``Tr(rho O)`` (observables are Hermitian)."""
    return float(np.real(np.trace(rho @ O)))


def features_from_states(
    states: Sequence[np.ndarray], observables: Sequence[Observable]
) -> np.ndarray:
    """Feature matrix ``X`` of shape ``(len(states), len(observables))``."""
    mats = [o.matrix for o in observables]
    X = np.empty((len(states), len(mats)), dtype=float)
    for t, rho in enumerate(states):
        for m, O in enumerate(mats):
            X[t, m] = np.real(np.trace(rho @ O))
    return X


def add_bias(X: np.ndarray) -> np.ndarray:
    """Append a constant bias column of ones."""
    return np.hstack([X, np.ones((X.shape[0], 1))])


# --- linear readout --------------------------------------------------------

def train_readout(
    X: np.ndarray, y: np.ndarray, ridge: float = 0.0
) -> np.ndarray:
    """Train linear weights ``w`` mapping ``X -> y`` by (ridge) least squares.

    ``ridge=0`` uses the SVD-based minimum-norm least-squares solution
    (``numpy.linalg.lstsq``); ``ridge>0`` solves the regularised normal
    equations.  ``X`` is expected to already include a bias column if desired.
    """
    if ridge <= 0:
        w, *_ = np.linalg.lstsq(X, y, rcond=None)
        return w
    n_feat = X.shape[1]
    A = X.T @ X + ridge * np.eye(n_feat)
    b = X.T @ y
    return np.linalg.solve(A, b)


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


# --- metrics ---------------------------------------------------------------

def capacity(y: np.ndarray, yhat: np.ndarray) -> float:
    """Squared correlation capacity ``C = cov(y, yhat)^2 / (var(y) var(yhat))``.

    Lies in ``[0, 1]``; this is the memory-capacity metric used in the paper.
    """
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    yc = y - y.mean()
    yhc = yhat - yhat.mean()
    denom = np.sum(yc ** 2) * np.sum(yhc ** 2)
    if denom < 1e-30:
        return 0.0
    cov = np.sum(yc * yhc)
    return float(cov ** 2 / denom)


def mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2))


def nmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    var = np.var(y)
    if var < 1e-30:
        return float("inf")
    return float(np.mean((y - np.asarray(yhat, float)) ** 2) / var)


def nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(nmse(y, yhat)))


@dataclass
class FitResult:
    w: np.ndarray
    train_metric: float
    test_metric: float
    metric_name: str


def fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = "capacity",
    ridge: float = 0.0,
) -> FitResult:
    """Train on the train split, score on the test split with the named metric."""
    w = train_readout(X_train, y_train, ridge=ridge)
    metric_fn = {
        "capacity": capacity,
        "mse": mse,
        "nmse": nmse,
        "nrmse": nrmse,
    }[metric]
    tr = metric_fn(y_train, predict(X_train, w))
    te = metric_fn(y_test, predict(X_test, w))
    return FitResult(w=w, train_metric=tr, test_metric=te, metric_name=metric)

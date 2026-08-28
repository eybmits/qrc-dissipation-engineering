"""Certified training-free prescreening for finite QRC dissipator families.

The module turns approximate task scores into practitioner-facing guarantees.
It is deliberately architecture agnostic: the score may come from the
Walsh--Volterra construction in :mod:`qrc.task_resolved`, from another
response-kernel approximation, or from a calibrated surrogate.

For a finite feasible family, a uniform score error ``delta`` immediately gives
an at-most ``2*delta`` selection-regret bound.  It also produces safe elimination
rules and exact top-k certificates.  Additional helpers convert feature-level
and covariance-level approximation errors into a score error for the
ridge-regularized population score

    s_lambda(G, g) = g.T @ inv(G + lambda I) @ g.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class PrescreenCertificate:
    """Result of certified prescreening.

    Attributes
    ----------
    ordered_candidates:
        Candidate labels sorted by decreasing predicted score.
    shortlist:
        Candidates whose score intervals can still contain a true optimum.
    discarded:
        Candidates proven unable to be optimal under the supplied uniform
        error bound.
    predicted_winner:
        Candidate with the largest predicted score.
    uniform_error:
        Assumed deterministic bound ``|true_i - predicted_i| <= delta``.
    regret_bound:
        Guaranteed regret of selecting ``predicted_winner``; equals ``2 delta``.
    top1_exact:
        Whether the proxy margin proves that the predicted winner is the unique
        true winner.
    """

    ordered_candidates: tuple[Hashable, ...]
    shortlist: tuple[Hashable, ...]
    discarded: tuple[Hashable, ...]
    predicted_winner: Hashable
    uniform_error: float
    regret_bound: float
    top1_exact: bool


def _score_array(scores: Sequence[float]) -> Array:
    out = np.asarray(scores, dtype=float)
    if out.ndim != 1 or out.size == 0:
        raise ValueError("scores must be a non-empty vector")
    if not np.all(np.isfinite(out)):
        raise ValueError("scores contain non-finite values")
    return out


def rank_candidates(
    predicted_scores: Sequence[float],
    candidates: Sequence[Hashable] | None = None,
) -> tuple[Hashable, ...]:
    """Return candidate labels sorted by decreasing predicted score."""

    scores = _score_array(predicted_scores)
    labels: tuple[Hashable, ...]
    if candidates is None:
        labels = tuple(range(scores.size))
    else:
        labels = tuple(candidates)
        if len(labels) != scores.size:
            raise ValueError("candidate and score lengths differ")
        if len(set(labels)) != len(labels):
            raise ValueError("candidate labels must be unique")
    order = np.argsort(-scores, kind="stable")
    return tuple(labels[int(index)] for index in order)


def selected_regret_bound(uniform_error: float) -> float:
    """Worst-case regret of choosing the proxy winner: ``2*uniform_error``."""

    delta = float(uniform_error)
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("uniform_error must be finite and non-negative")
    return 2.0 * delta


def top1_is_certified(
    predicted_scores: Sequence[float],
    uniform_error: float,
) -> bool:
    """Return whether the predicted winner is provably the unique true winner."""

    scores = np.sort(_score_array(predicted_scores))[::-1]
    if scores.size == 1:
        return True
    return bool(scores[0] - scores[1] > selected_regret_bound(uniform_error))


def topk_set_is_certified(
    predicted_scores: Sequence[float],
    k: int,
    uniform_error: float,
) -> bool:
    """Return whether the predicted top-k set equals the true top-k set.

    A strict boundary gap larger than ``2 delta`` is sufficient.
    """

    scores = np.sort(_score_array(predicted_scores))[::-1]
    k = int(k)
    if not 1 <= k <= scores.size:
        raise ValueError("k outside candidate range")
    if k == scores.size:
        return True
    return bool(scores[k - 1] - scores[k] > selected_regret_bound(uniform_error))


def safe_shortlist_indices(
    predicted_scores: Sequence[float],
    uniform_error: float,
) -> tuple[int, ...]:
    """Return candidates that cannot be safely eliminated.

    Each true score lies in ``[prediction-delta, prediction+delta]``.  A
    candidate is discarded exactly when its upper endpoint is below the best
    lower endpoint.  Every true maximizer is guaranteed to remain.
    """

    scores = _score_array(predicted_scores)
    delta = float(uniform_error)
    if not np.isfinite(delta) or delta < 0:
        raise ValueError("uniform_error must be finite and non-negative")
    best_lower = float(np.max(scores - delta))
    keep = np.flatnonzero(scores + delta >= best_lower)
    return tuple(int(index) for index in keep)


def prescreen_candidates(
    predicted_scores: Sequence[float],
    candidates: Sequence[Hashable] | None = None,
    *,
    uniform_error: float,
) -> PrescreenCertificate:
    """Build a deterministic finite-family prescreening certificate."""

    scores = _score_array(predicted_scores)
    labels = tuple(range(scores.size)) if candidates is None else tuple(candidates)
    if len(labels) != scores.size or len(set(labels)) != len(labels):
        raise ValueError("invalid candidate labels")
    order = np.argsort(-scores, kind="stable")
    keep_indices = safe_shortlist_indices(scores, uniform_error)
    keep_set = set(keep_indices)
    shortlist = tuple(labels[index] for index in order if int(index) in keep_set)
    discarded = tuple(labels[index] for index in order if int(index) not in keep_set)
    return PrescreenCertificate(
        ordered_candidates=tuple(labels[int(index)] for index in order),
        shortlist=shortlist,
        discarded=discarded,
        predicted_winner=labels[int(order[0])],
        uniform_error=float(uniform_error),
        regret_bound=selected_regret_bound(uniform_error),
        top1_exact=top1_is_certified(scores, uniform_error),
    )


def regularized_score_error_bound(
    proxy_cross_covariance: Sequence[float],
    cross_covariance_error: float,
    covariance_error: float,
    ridge: float,
) -> float:
    r"""Bound the error in ``g^T (G+lambda I)^-1 g``.

    Let ``G`` and ``G_hat`` be positive semidefinite, let
    ``||G-G_hat||_2 <= eps_G`` and ``||g-g_hat||_2 <= eps_g``, and let
    ``lambda > 0``.  Then

    .. math::

       |s_\lambda(G,g)-s_\lambda(\hat G,\hat g)|
       \le
       \frac{\epsilon_g(2\|\hat g\|+\epsilon_g)}{\lambda}
       +
       \frac{\|\hat g\|^2\epsilon_G}{\lambda^2}.

    The expression is deterministic and directly usable as ``uniform_error``.
    """

    proxy = np.asarray(proxy_cross_covariance, dtype=float)
    if proxy.ndim != 1 or not np.all(np.isfinite(proxy)):
        raise ValueError("proxy_cross_covariance must be a finite vector")
    eps_g = float(cross_covariance_error)
    eps_G = float(covariance_error)
    lam = float(ridge)
    if eps_g < 0 or eps_G < 0 or not np.isfinite(eps_g + eps_G):
        raise ValueError("error bounds must be finite and non-negative")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("ridge must be positive")
    norm = float(np.linalg.norm(proxy, ord=2))
    return eps_g * (2.0 * norm + eps_g) / lam + norm * norm * eps_G / (lam * lam)


def feature_remainder_moment_bounds(
    proxy_feature_bound: float,
    feature_remainder_bound: float,
    *,
    shot_covariance_error: float = 0.0,
    target_bound: float = 1.0,
) -> tuple[float, float]:
    r"""Convert a uniform feature remainder into moment error bounds.

    For centered features ``x = x_hat + e`` with ``||x_hat|| <= B`` and
    ``||e|| <= r``, and a bounded target ``|y| <= Y``, this returns

    ``eps_g = Y r`` and ``eps_G = 2 B r + r^2 + eps_shot``.
    """

    B = float(proxy_feature_bound)
    r = float(feature_remainder_bound)
    eps_shot = float(shot_covariance_error)
    Y = float(target_bound)
    if min(B, r, eps_shot, Y) < 0 or not np.isfinite(B + r + eps_shot + Y):
        raise ValueError("bounds must be finite and non-negative")
    return Y * r, 2.0 * B * r + r * r + eps_shot


def geometric_lag_remainder_bound(
    contraction: float,
    max_delay: int,
    input_amplitude: float,
    first_order_prefactor: float,
    second_order_prefactor: float,
    *,
    higher_order_remainder: float = 0.0,
) -> float:
    r"""Bound lag truncation of first- and second-order response kernels.

    Assume ``||A|| <= eta < 1``, ``||h_d|| <= B1 eta^d`` and
    ``||q_{a,b}|| <= B2 eta^(b-1)`` for ``a<b``.  Truncating all histories older
    than ``L=max_delay`` yields

    .. math::

       r_{L,2} \le
       |\epsilon| B_1 \frac{\eta^{L+1}}{1-\eta}
       + |\epsilon|^2 B_2
         \frac{(L+1)\eta^L-L\eta^{L+1}}{(1-\eta)^2}
       + r_{\ge3}.
    """

    eta = float(contraction)
    L = int(max_delay)
    eps = abs(float(input_amplitude))
    B1 = float(first_order_prefactor)
    B2 = float(second_order_prefactor)
    high = float(higher_order_remainder)
    if not 0 <= eta < 1:
        raise ValueError("contraction must lie in [0,1)")
    if L < 0 or min(B1, B2, high) < 0:
        raise ValueError("invalid non-negative bound")
    first_tail = eps * B1 * eta ** (L + 1) / (1.0 - eta)
    second_sum = ((L + 1) * eta**L - L * eta ** (L + 1)) / (1.0 - eta) ** 2
    second_tail = eps * eps * B2 * second_sum
    return first_tail + second_tail + high


def empirical_regret(
    true_scores: Sequence[float],
    selected_index: int,
) -> float:
    """Return oracle minus selected score."""

    scores = _score_array(true_scores)
    index = int(selected_index)
    if not 0 <= index < scores.size:
        raise ValueError("selected_index outside range")
    return float(np.max(scores) - scores[index])


def shortlist_regret(
    true_scores: Sequence[float],
    shortlist_indices: Iterable[int],
) -> float:
    """Regret after fully training and choosing the best shortlist candidate."""

    scores = _score_array(true_scores)
    indices = np.asarray(tuple(int(i) for i in shortlist_indices), dtype=int)
    if indices.size == 0 or np.any(indices < 0) or np.any(indices >= scores.size):
        raise ValueError("invalid shortlist")
    return float(np.max(scores) - np.max(scores[indices]))

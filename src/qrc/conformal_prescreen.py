"""Task-conditional split-conformal candidate sets for QRC model selection.

Scope
-----
The QRC architecture, temporal task, readout, and finite implementable
candidate family are fixed. A training-free score vector is available for all
candidates. Previous exchangeable reservoir realizations for the same task
provide paired score and fully trained performance vectors.

For calibration realization r, define the oracle deficit

    d_r = max_i score_i^(r) - max_{i in Oracle_r} score_i^(r),

where Oracle_r is the set of candidates attaining the largest empirical fully
trained performance under the declared protocol. The split-conformal quantile
of these deficits defines a variable-size set

    Gamma_tau(score) = {i: score_i >= max_j score_j - tau}.

Under exchangeability, this set contains an empirical oracle candidate for a
new realization with finite-sample marginal probability at least the conformal
rank divided by n+1. The guarantee is task conditional and marginal over the
new reservoir realization; it is not a simultaneous guarantee across dependent
tasks on the same device.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ConformalCalibration:
    """Split-conformal threshold and exact finite-sample rank guarantee."""

    threshold: float
    miscoverage: float
    calibration_size: int
    order_rank: int
    guaranteed_coverage: float


@dataclass(frozen=True)
class CandidateSetEvaluation:
    """Oracle coverage and regret after fully evaluating the retained set."""

    contains_oracle: bool
    retained_count: int
    total_candidates: int
    set_regret: float
    selected_best_performance: float
    oracle_performance: float

    @property
    def training_reduction(self) -> float:
        """Fraction of full candidate evaluations avoided."""
        return 1.0 - self.retained_count / self.total_candidates


def _score_vector(name: str, values: Sequence[float]) -> Array:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def near_optimal_indices(
    predicted_scores: Sequence[float],
    tolerance: float,
    *,
    numerical_tolerance: float = 1e-14,
) -> tuple[int, ...]:
    """Return candidates within tolerance of the largest predicted score."""
    scores = _score_vector("predicted_scores", predicted_scores)
    tolerance = float(tolerance)
    if np.isposinf(tolerance):
        return tuple(range(scores.size))
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be non-negative")
    cutoff = float(scores.max() - tolerance - numerical_tolerance)
    return tuple(int(index) for index in np.flatnonzero(scores >= cutoff))


def oracle_deficit(
    predicted_scores: Sequence[float],
    empirical_performances: Sequence[float],
    *,
    tie_tolerance: float = 1e-12,
) -> float:
    """Smallest score tolerance whose set contains an empirical oracle."""
    predicted = _score_vector("predicted_scores", predicted_scores)
    performance = _score_vector("empirical_performances", empirical_performances)
    if predicted.shape != performance.shape:
        raise ValueError("score and performance vectors must have equal size")
    oracle = np.flatnonzero(
        performance >= performance.max() - float(tie_tolerance)
    )
    return float(predicted.max() - predicted[oracle].max())


def split_conformal_quantile(
    nonconformity_scores: Sequence[float],
    miscoverage: float,
) -> ConformalCalibration:
    r"""Return the exact split-conformal upper quantile.

    For n calibration values and target miscoverage alpha, the rank is
    ceil((n+1)*(1-alpha)). If the rank exceeds n, distribution-free validity
    requires retaining all candidates, represented by an infinite threshold.
    """
    values = _score_vector("nonconformity_scores", nonconformity_scores)
    if np.any(values < -1e-15):
        raise ValueError("nonconformity scores must be non-negative")
    alpha = float(miscoverage)
    if not 0.0 < alpha < 1.0:
        raise ValueError("miscoverage must lie strictly between zero and one")
    n = int(values.size)
    rank = int(ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        threshold = float("inf")
        guarantee = 1.0
    else:
        threshold = float(np.sort(values)[rank - 1])
        guarantee = rank / (n + 1)
    return ConformalCalibration(
        threshold=threshold,
        miscoverage=alpha,
        calibration_size=n,
        order_rank=rank,
        guaranteed_coverage=float(guarantee),
    )


def calibrate_oracle_deficit(
    calibration_predictions: Iterable[Sequence[float]],
    calibration_performances: Iterable[Sequence[float]],
    miscoverage: float,
) -> ConformalCalibration:
    """Calibrate a task-specific threshold from paired candidate vectors."""
    predictions = list(calibration_predictions)
    performances = list(calibration_performances)
    if len(predictions) != len(performances) or not predictions:
        raise ValueError("calibration collections must be non-empty and paired")
    deficits = [
        oracle_deficit(prediction, performance)
        for prediction, performance in zip(predictions, performances)
    ]
    return split_conformal_quantile(deficits, miscoverage)


def conformal_prescreen(
    predicted_scores: Sequence[float],
    calibration: ConformalCalibration | float,
) -> tuple[int, ...]:
    """Return the adaptive candidate set induced by the calibrated threshold."""
    threshold = (
        calibration.threshold
        if isinstance(calibration, ConformalCalibration)
        else float(calibration)
    )
    return near_optimal_indices(predicted_scores, threshold)


def evaluate_candidate_set(
    retained_indices: Sequence[int],
    empirical_performances: Sequence[float],
    *,
    tie_tolerance: float = 1e-12,
) -> CandidateSetEvaluation:
    """Evaluate oracle coverage and regret of a retained candidate set."""
    performance = _score_vector("empirical_performances", empirical_performances)
    retained = np.asarray(tuple(int(index) for index in retained_indices), dtype=int)
    if retained.ndim != 1 or retained.size == 0:
        raise ValueError("retained_indices must not be empty")
    if np.any(retained < 0) or np.any(retained >= performance.size):
        raise ValueError("retained candidate index outside range")
    retained = np.unique(retained)
    oracle_performance = float(performance.max())
    selected_performance = float(performance[retained].max())
    oracle = np.flatnonzero(
        performance >= oracle_performance - float(tie_tolerance)
    )
    contains = bool(np.intersect1d(retained, oracle).size)
    return CandidateSetEvaluation(
        contains_oracle=contains,
        retained_count=int(retained.size),
        total_candidates=int(performance.size),
        set_regret=max(0.0, oracle_performance - selected_performance),
        selected_best_performance=selected_performance,
        oracle_performance=oracle_performance,
    )

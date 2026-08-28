from __future__ import annotations

import numpy as np

from qrc.conformal_prescreen import (
    calibrate_oracle_deficit,
    conformal_prescreen,
    evaluate_candidate_set,
    near_optimal_indices,
    oracle_deficit,
    split_conformal_quantile,
)


def test_oracle_deficit_and_candidate_set():
    predicted = [0.9, 0.8, 0.7]
    empirical = [0.2, 1.0, 0.1]
    assert np.isclose(oracle_deficit(predicted, empirical), 0.1)
    assert set(near_optimal_indices(predicted, 0.1)) == {0, 1}


def test_split_conformal_rank_coverage_exhaustively():
    values = np.arange(11, dtype=float)
    alpha = 0.2
    covered = 0
    guarantees = []
    for held_out in range(values.size):
        calibration = split_conformal_quantile(np.delete(values, held_out), alpha)
        guarantees.append(calibration.guaranteed_coverage)
        covered += values[held_out] <= calibration.threshold
    assert covered / len(values) >= min(guarantees) - 1e-12


def test_infinite_threshold_when_rank_exceeds_calibration_size():
    calibration = split_conformal_quantile([0.0, 0.1], 0.1)
    assert np.isinf(calibration.threshold)
    assert conformal_prescreen([0.2, 0.1, 0.0], calibration) == (0, 1, 2)


def test_calibration_from_candidate_vectors_and_evaluation():
    predictions = [[0.8, 0.2], [0.6, 0.5], [0.3, 0.4], [0.7, 0.1]]
    performances = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    calibration = calibrate_oracle_deficit(predictions, performances, 0.25)
    retained = conformal_prescreen([0.55, 0.5], calibration)
    evaluation = evaluate_candidate_set(retained, [0.8, 0.7])
    assert evaluation.retained_count >= 1
    assert 0 <= evaluation.training_reduction <= 1
    assert evaluation.set_regret >= 0

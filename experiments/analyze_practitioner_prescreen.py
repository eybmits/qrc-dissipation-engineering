#!/usr/bin/env python3
"""Analyze task-conditional conformal dissipator prescreening.

For each fixed delayed-product task, calibration reservoir realizations provide
paired Walsh-Volterra score vectors and fully trained candidate performances.
The oracle-deficit conformal quantile defines a variable-size candidate set for
untouched reservoir realizations. Balanced calibration/test resplits can be
enumerated as a robustness audit; those resplits are not independent trials.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc.conformal_prescreen import (  # noqa: E402
    calibrate_oracle_deficit,
    conformal_prescreen,
    evaluate_candidate_set,
    oracle_deficit,
)
from qrc.prescreen import deterministic_prescreen  # noqa: E402


def parse_seed_set(text: str) -> set[int]:
    values: set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            values.update(range(int(left), int(right) + 1))
        else:
            values.add(int(item))
    if not values:
        raise argparse.ArgumentTypeError("seed set must not be empty")
    return values


def group(frame: pd.DataFrame, seed: int, task: str) -> pd.DataFrame:
    out = frame[(frame.seed == seed) & (frame.task == task)].sort_values("alpha")
    if out.empty:
        raise ValueError(f"missing seed={seed}, task={task}")
    return out.reset_index(drop=True)


def calibrate(frame, task, seeds, alpha):
    predictions, performances, deficits = [], [], []
    for seed in sorted(seeds):
        current = group(frame, seed, task)
        predicted = current.predicted_capacity.to_numpy(float)
        truth = current.empirical_capacity.to_numpy(float)
        predictions.append(predicted)
        performances.append(truth)
        deficits.append(
            {"seed": seed, "task": task, "oracle_deficit": oracle_deficit(predicted, truth)}
        )
    return calibrate_oracle_deficit(predictions, performances, alpha), deficits


def evaluate_split(frame, task, calibration_seeds, test_seeds, alpha):
    calibration, deficits = calibrate(frame, task, calibration_seeds, alpha)
    rows = []
    for seed in sorted(test_seeds):
        current = group(frame, seed, task)
        predicted = current.predicted_capacity.to_numpy(float)
        truth = current.empirical_capacity.to_numpy(float)
        alphas = current.alpha.to_numpy(float)
        retained = conformal_prescreen(predicted, calibration)
        result = evaluate_candidate_set(retained, truth)
        top1 = int(np.argmax(predicted))
        local = int(np.argmin(np.abs(alphas)))
        rows.append(
            {
                "seed": seed,
                "task": task,
                "threshold": calibration.threshold,
                "finite_sample_guaranteed_coverage": calibration.guaranteed_coverage,
                "retained_count": result.retained_count,
                "training_reduction": result.training_reduction,
                "contains_oracle": int(result.contains_oracle),
                "set_regret": result.set_regret,
                "top1_regret": float(truth.max() - truth[top1]),
                "top1_gain_vs_local": float(truth[top1] - truth[local]),
                "selected_alpha_top1": float(alphas[top1]),
                "oracle_alpha": float(alphas[int(np.argmax(truth))]),
                "retained_alphas": ";".join(str(float(alphas[index])) for index in retained),
            }
        )
    return calibration, deficits, rows


def task_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("task", as_index=False)
        .agg(
            heldout_cases=("seed", "size"),
            descriptive_oracle_coverage=("contains_oracle", "mean"),
            average_retained=("retained_count", "mean"),
            average_training_reduction=("training_reduction", "mean"),
            mean_set_regret=("set_regret", "mean"),
            maximum_set_regret=("set_regret", "max"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-seeds", type=parse_seed_set, default=parse_seed_set("0-8"))
    parser.add_argument("--test-seeds", type=parse_seed_set, default=parse_seed_set("9-11"))
    parser.add_argument("--miscoverage", type=float, default=0.1)
    parser.add_argument("--enumerate-balanced-splits", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, dtype={"task": str})
    required = {"seed", "task", "alpha", "predicted_capacity", "empirical_capacity"}
    if not required.issubset(frame.columns):
        raise ValueError(f"missing columns: {sorted(required - set(frame.columns))}")
    tasks = sorted(frame.task.unique())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    fixed_rows, deficits, thresholds = [], [], []
    for task in tasks:
        calibration, task_deficits, task_rows = evaluate_split(
            frame, task, args.calibration_seeds, args.test_seeds, args.miscoverage
        )
        deficits.extend(task_deficits)
        fixed_rows.extend(task_rows)
        thresholds.append(
            {
                "task": task,
                "threshold": calibration.threshold,
                "calibration_size": calibration.calibration_size,
                "order_rank": calibration.order_rank,
                "finite_sample_guaranteed_coverage": calibration.guaranteed_coverage,
            }
        )

    fixed = pd.DataFrame(fixed_rows)
    fixed.to_csv(output / "fixed_heldout_candidate_sets.csv", index=False)
    pd.DataFrame(deficits).to_csv(output / "fixed_calibration_oracle_deficits.csv", index=False)
    pd.DataFrame(thresholds).to_csv(output / "fixed_task_thresholds.csv", index=False)
    task_summary(fixed).to_csv(output / "fixed_task_summary.csv", index=False)

    heldout = frame[frame.seed.isin(args.test_seeds)]
    pearson = float(pearsonr(heldout.predicted_capacity, heldout.empirical_capacity).statistic)
    spearman = float(spearmanr(heldout.predicted_capacity, heldout.empirical_capacity).statistic)

    split_summary = None
    if args.enumerate_balanced_splits:
        seeds = sorted(int(seed) for seed in frame.seed.unique())
        calibration_size = len(args.calibration_seeds)
        split_rows = []
        for calibration_tuple in combinations(seeds, calibration_size):
            calibration_seeds = set(calibration_tuple)
            test_seeds = set(seeds) - calibration_seeds
            current = []
            for task in tasks:
                _, _, rows = evaluate_split(
                    frame, task, calibration_seeds, test_seeds, args.miscoverage
                )
                current.extend(rows)
            current_frame = pd.DataFrame(current)
            split_rows.append(
                {
                    "calibration_seeds": ",".join(map(str, sorted(calibration_seeds))),
                    "test_seeds": ",".join(map(str, sorted(test_seeds))),
                    "descriptive_oracle_coverage": current_frame.contains_oracle.mean(),
                    "training_reduction": current_frame.training_reduction.mean(),
                    "mean_set_regret": current_frame.set_regret.mean(),
                }
            )
        split_frame = pd.DataFrame(split_rows)
        split_frame.to_csv(output / "all_balanced_split_summary.csv", index=False)
        split_summary = {
            "balanced_splits": len(split_frame),
            "mean_descriptive_oracle_coverage": float(split_frame.descriptive_oracle_coverage.mean()),
            "mean_training_reduction": float(split_frame.training_reduction.mean()),
            "mean_set_regret": float(split_frame.mean_set_regret.mean()),
            "coverage_q025_q975": [float(x) for x in split_frame.descriptive_oracle_coverage.quantile([0.025, 0.975])],
            "training_reduction_q025_q975": [float(x) for x in split_frame.training_reduction.quantile([0.025, 0.975])],
            "note": "resplits of the same reservoir realizations; not independent experiments",
        }

    summary = {
        "claim": (
            "For a fixed task and finite feasible dissipator family, task-conditional "
            "split-conformal calibration returns a variable-size candidate set with "
            "finite-sample marginal coverage of the empirical fully trained oracle "
            "for a new exchangeable reservoir realization."
        ),
        "fixed_split": {
            "calibration_seeds": sorted(args.calibration_seeds),
            "test_seeds": sorted(args.test_seeds),
            "nominal_miscoverage": args.miscoverage,
            "finite_sample_guaranteed_coverage_per_task": float(
                min(row["finite_sample_guaranteed_coverage"] for row in thresholds)
            ),
            "descriptive_pooled_oracle_coverage": float(fixed.contains_oracle.mean()),
            "average_candidates_retained": float(fixed.retained_count.mean()),
            "average_full_training_reduction": float(fixed.training_reduction.mean()),
            "mean_set_regret": float(fixed.set_regret.mean()),
            "maximum_set_regret": float(fixed.set_regret.max()),
            "pooled_coverage_note": "descriptive only; tasks on one Hamiltonian are dependent",
        },
        "heldout_score_validation": {
            "candidate_rows": int(len(heldout)),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "mean_absolute_error": float(np.mean(np.abs(heldout.predicted_capacity - heldout.empirical_capacity))),
        },
        "all_balanced_splits": split_summary,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

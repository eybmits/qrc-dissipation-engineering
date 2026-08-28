#!/usr/bin/env python3
"""Evaluate the bounded QRC practitioner claim on exhaustive candidate sweeps.

Input rows must contain one predicted score and one fully trained empirical score
for every candidate within each architecture/task group.  The script measures:

* rank correlation and pairwise ordering accuracy;
* top-k oracle inclusion;
* empirical regret after training only the top-k shortlist;
* full-training runs saved;
* gains against local, most-collective, and random-candidate baselines;
* a conservative leave-one-seed-out interval shortlist.

It supports both the legacy validation schema and the canonical reproduction
schema produced by ``run_task_resolved_volterra.py``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    # 1.959963984540054 is Phi^{-1}(0.975).
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def bootstrap_mean_interval(values: Iterable[float], rng: np.random.Generator, draws: int = 20000) -> tuple[float, float]:
    array = np.asarray(tuple(values), float)
    if array.size == 0:
        return math.nan, math.nan
    indices = rng.integers(0, array.size, size=(draws, array.size))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pairwise_accuracy(predicted: np.ndarray, empirical: np.ndarray) -> float:
    correct = 0.0
    total = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            pred_sign = np.sign(predicted[i] - predicted[j])
            true_sign = np.sign(empirical[i] - empirical[j])
            correct += 1.0 if pred_sign == true_sign else 0.5 if pred_sign == 0 or true_sign == 0 else 0.0
            total += 1
    return correct / total if total else 1.0


def normalize_input(frame: pd.DataFrame, score_column: str | None, truth_column: str | None) -> tuple[pd.DataFrame, str, str, list[str]]:
    score = score_column or "predicted_capacity"
    if truth_column:
        truth = truth_column
    elif "empirical_capacity" in frame.columns:
        truth = "empirical_capacity"
    elif "empirical_test_corr2" in frame.columns:
        truth = "empirical_test_corr2"
    else:
        raise ValueError("could not infer empirical score column")
    required = {"seed", "task", "alpha", score, truth}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    group_columns = [column for column in ("n_qubits", "seed", "task") if column in frame.columns]
    frame = frame.copy()
    frame[score] = pd.to_numeric(frame[score], errors="raise")
    frame[truth] = pd.to_numeric(frame[truth], errors="raise")
    frame["alpha"] = pd.to_numeric(frame["alpha"], errors="raise")
    return frame, score, truth, group_columns


def group_metrics(group: pd.DataFrame, score: str, truth: str, group_columns: list[str]) -> dict:
    group = group.sort_values("alpha").reset_index(drop=True)
    predicted = group[score].to_numpy(float)
    empirical = group[truth].to_numpy(float)
    candidates = group["alpha"].to_numpy(float)
    order = np.argsort(-predicted, kind="stable")
    empirical_order = np.argsort(-empirical, kind="stable")
    oracle_index = int(empirical_order[0])
    proxy_index = int(order[0])
    output = {column: group.iloc[0][column] for column in group_columns}
    output.update(
        {
            "candidate_count": len(group),
            "predicted_best_alpha": candidates[proxy_index],
            "empirical_best_alpha": candidates[oracle_index],
            "oracle_score": empirical[oracle_index],
            "proxy_selected_score": empirical[proxy_index],
            "proxy_regret": empirical[oracle_index] - empirical[proxy_index],
            "local_score": empirical[int(np.argmin(candidates))],
            "most_collective_score": empirical[int(np.argmax(candidates))],
            "random_candidate_expected_score": float(empirical.mean()),
            "proxy_gain_vs_local": empirical[proxy_index] - empirical[int(np.argmin(candidates))],
            "proxy_gain_vs_most_collective": empirical[proxy_index] - empirical[int(np.argmax(candidates))],
            "proxy_gain_vs_random_expected": empirical[proxy_index] - float(empirical.mean()),
            "spearman": float(spearmanr(predicted, empirical).statistic),
            "kendall": float(kendalltau(predicted, empirical).statistic),
            "pairwise_accuracy": pairwise_accuracy(predicted, empirical),
            "max_absolute_score_error": float(np.max(np.abs(predicted - empirical))),
            "mean_absolute_score_error": float(np.mean(np.abs(predicted - empirical))),
        }
    )
    for k in range(1, len(group) + 1):
        shortlist = order[:k]
        best_shortlist_score = float(np.max(empirical[shortlist]))
        output[f"top{k}_contains_oracle"] = bool(oracle_index in shortlist)
        output[f"top{k}_regret"] = empirical[oracle_index] - best_shortlist_score
        output[f"top{k}_training_saving"] = 1.0 - k / len(group)
    return output


def conservative_loso_shortlists(frame: pd.DataFrame, score: str, truth: str, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for heldout_seed in sorted(frame["seed"].unique()):
        calibration = frame[frame["seed"] != heldout_seed]
        heldout = frame[frame["seed"] == heldout_seed]
        delta = float(np.max(np.abs(calibration[score] - calibration[truth])))
        heldout_group_columns = [column for column in group_columns if column != "seed"]
        for key, group in heldout.groupby(heldout_group_columns, sort=True):
            if not isinstance(key, tuple):
                key = (key,)
            predicted = group[score].to_numpy(float)
            empirical = group[truth].to_numpy(float)
            candidates = group["alpha"].to_numpy(float)
            best_lower = float(np.max(predicted - delta))
            keep = np.flatnonzero(predicted + delta >= best_lower)
            oracle = int(np.argmax(empirical))
            row = {
                "heldout_seed": heldout_seed,
                "uniform_error_from_other_seeds": delta,
                "candidate_count": len(group),
                "shortlist_size": len(keep),
                "training_saving": 1.0 - len(keep) / len(group),
                "contains_oracle": bool(oracle in keep),
                "shortlist_regret": float(np.max(empirical) - np.max(empirical[keep])),
                "shortlist_alphas": ";".join(f"{candidates[index]:g}" for index in keep),
            }
            row.update(dict(zip(heldout_group_columns, key)))
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(per_group: pd.DataFrame, calibrated: pd.DataFrame) -> dict:
    rng = np.random.default_rng(20260828)
    maximum_candidates = int(per_group["candidate_count"].max())
    topk = {}
    for k in range(1, maximum_candidates + 1):
        contains_column = f"top{k}_contains_oracle"
        if contains_column not in per_group:
            continue
        valid = per_group[per_group["candidate_count"] >= k]
        successes = int(valid[contains_column].sum())
        lower, upper = wilson_interval(successes, len(valid))
        regret_low, regret_high = bootstrap_mean_interval(valid[f"top{k}_regret"], rng)
        topk[str(k)] = {
            "groups": int(len(valid)),
            "oracle_inclusion_fraction": float(valid[contains_column].mean()),
            "oracle_inclusion_wilson_95": [lower, upper],
            "mean_regret": float(valid[f"top{k}_regret"].mean()),
            "mean_regret_bootstrap_95": [regret_low, regret_high],
            "max_regret": float(valid[f"top{k}_regret"].max()),
            "mean_full_training_runs_saved_fraction": float(valid[f"top{k}_training_saving"].mean()),
        }
    metrics = {}
    for column in (
        "spearman",
        "kendall",
        "pairwise_accuracy",
        "proxy_regret",
        "proxy_gain_vs_local",
        "proxy_gain_vs_most_collective",
        "proxy_gain_vs_random_expected",
        "mean_absolute_score_error",
        "max_absolute_score_error",
    ):
        low, high = bootstrap_mean_interval(per_group[column], rng)
        metrics[column] = {
            "mean": float(per_group[column].mean()),
            "median": float(per_group[column].median()),
            "min": float(per_group[column].min()),
            "max": float(per_group[column].max()),
            "mean_bootstrap_95": [low, high],
        }
    return {
        "bounded_claim": (
            "Within a fixed physically feasible candidate family, rank candidates "
            "from differentiated-channel task scores, fully train only a shortlist, "
            "and bound proxy-winner regret by twice a uniform score-error bound."
        ),
        "groups": int(len(per_group)),
        "candidate_count_values": sorted(int(value) for value in per_group["candidate_count"].unique()),
        "top_k": topk,
        "ranking_and_baselines": metrics,
        "leave_one_seed_out_uniform_interval_shortlist": {
            "groups": int(len(calibrated)),
            "oracle_inclusion_fraction": float(calibrated["contains_oracle"].mean()),
            "mean_shortlist_size": float(calibrated["shortlist_size"].mean()),
            "mean_training_saving_fraction": float(calibrated["training_saving"].mean()),
            "mean_regret": float(calibrated["shortlist_regret"].mean()),
            "max_regret": float(calibrated["shortlist_regret"].max()),
        },
    }


def make_plots(frame: pd.DataFrame, per_group: pd.DataFrame, summary: dict, score: str, truth: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(6.2, 5.6))
    plt.scatter(frame[score], frame[truth], alpha=0.7)
    limit = max(float(frame[score].max()), float(frame[truth].max()))
    plt.plot([0, limit], [0, limit], linestyle="--")
    plt.xlabel("Training-free predicted capacity")
    plt.ylabel("Fully trained held-out capacity")
    plt.title("Prescreen score versus exhaustive QRC training")
    plt.tight_layout()
    figure.savefig(output / "prediction_vs_training.png", dpi=220)
    plt.close(figure)

    ks = [int(key) for key in summary["top_k"]]
    savings = [summary["top_k"][str(k)]["mean_full_training_runs_saved_fraction"] for k in ks]
    regrets = [summary["top_k"][str(k)]["mean_regret"] for k in ks]
    figure = plt.figure(figsize=(6.6, 4.8))
    plt.plot(savings, regrets, marker="o")
    for k, x, y in zip(ks, savings, regrets):
        plt.annotate(f"top-{k}", (x, y), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Fraction of full candidate trainings saved")
    plt.ylabel("Mean regret after shortlist training")
    plt.title("Practitioner trade-off: training saved versus regret")
    plt.tight_layout()
    figure.savefig(output / "training_saving_vs_regret.png", dpi=220)
    plt.close(figure)

    figure = plt.figure(figsize=(6.6, 4.8))
    plt.hist(per_group["proxy_regret"], bins=min(10, max(4, len(per_group) // 2)))
    plt.xlabel("Oracle capacity minus prescreen-selected capacity")
    plt.ylabel("Architecture-task groups")
    plt.title("Regret of training only the predicted winner")
    plt.tight_layout()
    figure.savefig(output / "top1_regret_distribution.png", dpi=220)
    plt.close(figure)


def exact_linear_memory_capacity(decay_rates: Iterable[float], delay: int) -> float:
    """Exact delay capacity of fully observed independent affine binary modes.

    The nonzero input-write coefficients cancel under full readout, so unit
    coefficients are used without loss of generality.
    """

    gamma = np.asarray(tuple(decay_rates), float)
    if gamma.ndim != 1 or gamma.size == 0 or np.any(gamma <= 0):
        raise ValueError("decay_rates must be a positive vector")
    lam = np.exp(-gamma)
    cross = 1.0 / (1.0 - np.outer(lam, lam))
    target = lam ** int(delay)
    return float(target @ np.linalg.solve(cross, target))


def write_exact_counterexample(output: Path) -> None:
    rows = []
    environments = {
        "balanced": (0.4, 0.6),
        "heterogeneous": (0.05, 0.95),
    }
    for name, rates in environments.items():
        for delay in range(21):
            rows.append(
                {
                    "environment": name,
                    "gamma_1": rates[0],
                    "gamma_2": rates[1],
                    "total_decay_budget": sum(rates),
                    "delay": delay,
                    "capacity": exact_linear_memory_capacity(rates, delay),
                }
            )
    pd.DataFrame(rows).to_csv(
        output / "exact_equal_budget_counterexample.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-column")
    parser.add_argument("--truth-column")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    frame, score, truth, group_columns = normalize_input(frame, args.score_column, args.truth_column)
    rows = [group_metrics(group, score, truth, group_columns) for _, group in frame.groupby(group_columns, sort=True)]
    per_group = pd.DataFrame(rows)
    calibrated = conservative_loso_shortlists(frame, score, truth, group_columns)
    summary = summarize(per_group, calibrated)

    args.output.mkdir(parents=True, exist_ok=True)
    per_group.to_csv(args.output / "per_group_prescreening.csv", index=False)
    calibrated.to_csv(args.output / "loso_certified_shortlists.csv", index=False)
    (args.output / "prescreening_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_plots(frame, per_group, summary, score, truth, args.output)
    write_exact_counterexample(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate manuscript figures for certified dissipator prescreening."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def pipeline_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    boxes = [
        (0.25, 0.85, 2.2, 1.45, "Fixed QRC +\nfeasible dissipators"),
        (3.10, 0.85, 2.2, 1.45, "Differentiate the\nreference channel"),
        (5.95, 0.85, 2.2, 1.45, "Compute task-resolved\nWalsh--Volterra scores"),
        (8.80, 0.85, 2.2, 1.45, "Eliminate safely;\ntrain only survivors"),
    ]
    for x, y, width, height, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.08,rounding_size=0.08",
                linewidth=1.5, facecolor="white", edgecolor="black",
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=12)
    for index in range(3):
        x0 = boxes[index][0] + boxes[index][2] + 0.08
        x1 = boxes[index + 1][0] - 0.08
        ax.add_patch(
            FancyArrowPatch(
                (x0, 1.575), (x1, 1.575),
                arrowstyle="-|>", mutation_scale=16, linewidth=1.5,
            )
        )
    ax.text(
        6, 2.85,
        "Task specification enters the score; no task-specific readout training is used for ranking",
        ha="center", va="center", fontsize=12,
    )
    ax.text(9.9, 0.42, r"guarantee: regret $\leq 2\delta$", ha="center", va="center", fontsize=11)
    fig.tight_layout(pad=0.2)
    fig.savefig(output / "prescreen_pipeline.pdf", bbox_inches="tight")
    fig.savefig(output / "prescreen_pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def result_figures(results: Path, candidates: Path, output: Path) -> None:
    raw = pd.read_csv(candidates)
    groups = pd.read_csv(results / "per_group_prescreening.csv")
    summary = json.loads((results / "prescreening_summary.json").read_text(encoding="utf-8"))

    score = "predicted_capacity"
    truth = "empirical_capacity" if "empirical_capacity" in raw else "empirical_test_corr2"
    limit = max(float(raw[score].max()), float(raw[truth].max()))
    ks = sorted(int(key) for key in summary["top_k"])
    savings = [summary["top_k"][str(k)]["mean_full_training_runs_saved_fraction"] for k in ks]
    regrets = [summary["top_k"][str(k)]["mean_regret"] for k in ks]

    fig = plt.figure(figsize=(6.2, 5.6))
    plt.scatter(raw[score], raw[truth], alpha=0.72)
    plt.plot([0, limit], [0, limit], linestyle="--")
    plt.xlabel("Training-free predicted capacity")
    plt.ylabel("Fully trained held-out capacity")
    plt.title("Prescreen score versus exhaustive QRC training")
    plt.tight_layout()
    fig.savefig(output / "prediction_vs_training.pdf")
    fig.savefig(output / "prediction_vs_training.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.6, 4.8))
    plt.plot(savings, regrets, marker="o")
    for k, x, y in zip(ks, savings, regrets):
        plt.annotate(f"top-{k}", (x, y), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Fraction of full candidate trainings saved")
    plt.ylabel("Mean regret after shortlist training")
    plt.title("Training-saving versus regret")
    plt.tight_layout()
    fig.savefig(output / "training_saving_vs_regret.pdf")
    fig.savefig(output / "training_saving_vs_regret.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(6.6, 4.8))
    plt.hist(groups["proxy_regret"], bins=min(10, max(4, len(groups) // 2)))
    plt.xlabel("Oracle capacity minus selected capacity")
    plt.ylabel("Architecture-task groups")
    plt.title("Regret from training only the predicted winner")
    plt.tight_layout()
    fig.savefig(output / "top1_regret_distribution.pdf")
    fig.savefig(output / "top1_regret_distribution.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].scatter(raw[score], raw[truth], alpha=0.72)
    axes[0].plot([0, limit], [0, limit], linestyle="--")
    axes[0].set_xlabel("Predicted capacity")
    axes[0].set_ylabel("Fully trained test capacity")
    axes[0].set_title("(a) Candidate-level prediction")
    axes[1].plot(savings, regrets, marker="o")
    for k, x, y in zip(ks, savings, regrets):
        axes[1].annotate(f"top-{k}", (x, y), xytext=(4, 4), textcoords="offset points")
    axes[1].set_xlabel("Full candidate trainings saved")
    axes[1].set_ylabel("Mean shortlist regret")
    axes[1].set_title("(b) Practitioner trade-off")
    fig.tight_layout()
    fig.savefig(output / "prescreen_results_combined.pdf")
    fig.savefig(output / "prescreen_results_combined.png", dpi=220)
    plt.close(fig)

    counter = pd.read_csv(results / "exact_equal_budget_counterexample.csv")
    fig = plt.figure(figsize=(6.5, 4.4))
    for name, group in counter.groupby("environment"):
        plt.plot(group["delay"], group["capacity"], marker="o", markersize=3, label=name)
    plt.xlabel("Delay d")
    plt.ylabel("Exact linear memory capacity")
    plt.title("Equal decay budget, incompatible task optima")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output / "exact_counterexample.pdf")
    fig.savefig(output / "exact_counterexample.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/task_resolved_volterra/prescreening"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("results/task_resolved_volterra/prescreening_candidates.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pipeline_figure(args.output)
    result_figures(args.results, args.candidates, args.output)


if __name__ == "__main__":
    main()

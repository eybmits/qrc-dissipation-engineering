"""Selection-aware and multiplicity-aware inference for the final revision.

This module is deliberately separate from the frozen experiment drivers.  It
re-analyses their sealed per-reservoir outputs without changing any simulated
trajectory:

* the six-design strength comparison receives a reservoir-cluster bootstrap
  that repeats leave-one-reservoir-out strength selection in every resample;
* the measurement study receives one simultaneous family spanning every
  design pair, finite budget, and measurement model; and
* the channel-profile control receives all-pairs simultaneous intervals across
  the four primary 2x2 cells and its unequal-collective sanity rung.

Run from the repository root:

    PYTHONPATH=src:experiments python experiments/revision_inference.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TUNING = (
    ROOT
    / "results"
    / "revision_tuning"
    / "strength_extension"
    / "six_channel_aggregate.json"
)
DEFAULT_MEASUREMENT = (
    ROOT
    / "results"
    / "measurement_full_v3"
    / "measurement_full_aggregate.json"
)
DEFAULT_JOINT_GLOB = "G_joint__stm_N5_*.json"
DEFAULT_OUTPUT = (
    ROOT / "results" / "revision_tuning" / "critique_response_inference.json"
)

METHOD_ORDER = (
    "CD_paper",
    "A1_heterogeneous",
    "B2_thermal",
    "B3_collective",
    "B4_loss_exchange",
    "B5_pair",
)
JOINT_ORDER = (
    "G_local_uniform",
    "G_local_learned",
    "G_coll_uniform",
    "G_coll_learned",
    "G_coll_unequal",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_present(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    present = set(values)
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("paired sample must contain at least two finite values")
    return float(array.mean()), float(array.std(ddof=1) / math.sqrt(array.size))


def _pairwise_t_family(
    score_lookup: dict[tuple[str, str, int, str], float],
    methods: Sequence[str],
    models: Sequence[str],
    budgets: Sequence[str],
    seeds: Sequence[int],
    *,
    alpha: float,
) -> dict:
    pairs = list(itertools.combinations(methods, 2))
    family_size = len(models) * len(budgets) * len(pairs)
    if family_size <= 0:
        raise ValueError("empty simultaneous family")
    critical = float(
        stats.t.ppf(1.0 - alpha / (2.0 * family_size), len(seeds) - 1)
    )

    rows: list[dict] = []
    winners: list[dict] = []
    for model in models:
        for budget in budgets:
            means = {
                method: float(
                    np.mean(
                        [
                            score_lookup[(model, budget, seed, method)]
                            for seed in seeds
                        ]
                    )
                )
                for method in methods
            }
            for left, right in pairs:
                differences = [
                    score_lookup[(model, budget, seed, left)]
                    - score_lookup[(model, budget, seed, right)]
                    for seed in seeds
                ]
                mean, se = _mean_se(differences)
                rows.append(
                    {
                        "measurement_model": model,
                        "budget": budget,
                        "left": left,
                        "right": right,
                        "n": len(seeds),
                        "mean_difference": mean,
                        "se_difference": se,
                        "simultaneous_low": mean - critical * se,
                        "simultaneous_high": mean + critical * se,
                        "wins": int(sum(value > 0.0 for value in differences)),
                        "ties": int(sum(value == 0.0 for value in differences)),
                    }
                )

            winner = max(methods, key=means.__getitem__)
            lower_bounds = []
            for competitor in methods:
                if competitor == winner:
                    continue
                differences = [
                    score_lookup[(model, budget, seed, winner)]
                    - score_lookup[(model, budget, seed, competitor)]
                    for seed in seeds
                ]
                mean, se = _mean_se(differences)
                lower_bounds.append(mean - critical * se)
            winners.append(
                {
                    "measurement_model": model,
                    "budget": budget,
                    "mean_winner": winner,
                    "winner_mean": means[winner],
                    "simultaneously_above_every_competitor": bool(
                        min(lower_bounds) > 0.0
                    ),
                    "smallest_simultaneous_lower_bound": float(min(lower_bounds)),
                }
            )

    return {
        "family_size": family_size,
        "family_alpha": alpha,
        "critical_t": critical,
        "degrees_of_freedom": len(seeds) - 1,
        "pairwise_intervals": rows,
        "winner_assessment": winners,
    }


def measurement_simultaneous_inference(
    payload: dict,
    *,
    alpha: float = 0.05,
) -> dict:
    """Build all-pairs simultaneous bands over finite measurement budgets."""

    raw_rows = [
        row
        for row in payload["raw_rows"]
        if not bool(row["is_exact"])
    ]
    methods = _ordered_present(
        (str(row["channel"]) for row in raw_rows), METHOD_ORDER
    )
    models = sorted({str(row["measurement_model"]) for row in raw_rows})
    budgets = [
        str(value)
        for value in sorted(
            {int(row["total_shots_per_time_step"]) for row in raw_rows}
        )
    ]
    seeds = sorted({int(row["seed"]) for row in raw_rows})
    lookup = {
        (
            str(row["measurement_model"]),
            str(int(row["total_shots_per_time_step"])),
            int(row["seed"]),
            str(row["channel"]),
        ): float(row["test_mc"])
        for row in raw_rows
    }
    expected = len(methods) * len(models) * len(budgets) * len(seeds)
    if len(lookup) != expected:
        raise ValueError(
            f"incomplete measurement panel: {len(lookup)}/{expected} cells"
        )

    family = _pairwise_t_family(
        lookup,
        methods,
        models,
        budgets,
        seeds,
        alpha=alpha,
    )
    collective_local = []
    for row in family["pairwise_intervals"]:
        if {row["left"], row["right"]} != {"B3_collective", "CD_paper"}:
            continue
        oriented = dict(row)
        if row["left"] != "B3_collective":
            oriented["left"] = "B3_collective"
            oriented["right"] = "CD_paper"
            oriented["mean_difference"] = -row["mean_difference"]
            oriented["simultaneous_low"] = -row["simultaneous_high"]
            oriented["simultaneous_high"] = -row["simultaneous_low"]
            oriented["wins"] = len(seeds) - row["wins"] - row["ties"]
        collective_local.append(oriented)

    first_resolved = {}
    for model in models:
        ordered = sorted(
            (
                row
                for row in collective_local
                if row["measurement_model"] == model
            ),
            key=lambda row: int(row["budget"]),
        )
        first_resolved[model] = next(
            (
                row["budget"]
                for row in ordered
                if row["simultaneous_low"] > 0.0
            ),
            None,
        )

    return {
        "status": "complete",
        "estimand": (
            "paired held-out STM-capacity differences at each finite budget"
        ),
        "family_definition": (
            "all 15 design pairs x 7 finite budgets x 2 measurement models"
        ),
        "methods": methods,
        "models": models,
        "budgets": budgets,
        "seeds": seeds,
        **family,
        "collective_minus_local": collective_local,
        "first_familywise_resolved_collective_over_local": first_resolved,
        "interpretation_boundary": (
            "The first resolved point is a statement about the frozen fourfold "
            "grid, not an estimate of a continuous crossover budget or ratio. "
            "Mean-winner labels remain descriptive unless the winner is "
            "simultaneously above every competitor."
        ),
    }


def _strength_matrices(payload: dict) -> tuple[
    list[str],
    list[int],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    rows = payload["raw_rows"]
    methods = _ordered_present(
        (str(row["method"]) for row in rows), METHOD_ORDER
    )
    seeds = sorted({int(row["seed"]) for row in rows})
    seed_index = {seed: index for index, seed in enumerate(seeds)}
    multipliers: dict[str, np.ndarray] = {}
    validation: dict[str, np.ndarray] = {}
    test: dict[str, np.ndarray] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        grid = np.asarray(
            sorted({float(row["mult"]) for row in method_rows}), dtype=float
        )
        mult_index = {value: index for index, value in enumerate(grid)}
        val = np.full((len(seeds), len(grid)), np.nan, dtype=float)
        tst = np.full_like(val, np.nan)
        for row in method_rows:
            i = seed_index[int(row["seed"])]
            j = mult_index[float(row["mult"])]
            val[i, j] = float(row["val_value"])
            tst[i, j] = float(row["value"])
        if not np.all(np.isfinite(val)) or not np.all(np.isfinite(tst)):
            raise ValueError(f"incomplete strength matrix for {method}")
        multipliers[method] = grid
        validation[method] = val
        test[method] = tst
    return methods, seeds, multipliers, validation, test


def cluster_excluded_selected_means(
    counts: np.ndarray,
    methods: Sequence[str],
    validation: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Repeat LOSO selection while excluding every duplicate target cluster.

    ``counts`` is one paired reservoir-cluster bootstrap draw.  For a target
    reservoir with multiplicity ``c_i``, every copy of reservoir ``i`` is
    removed from its calibration mean; its held-out score receives weight
    ``c_i`` in the bootstrap estimand.
    """

    counts = np.asarray(counts, dtype=int)
    if counts.ndim != 1 or np.any(counts < 0) or counts.sum() < 2:
        raise ValueError("invalid cluster multiplicities")
    n_positions = int(counts.sum())
    active = np.flatnonzero(counts)
    if active.size < 2:
        raise ValueError("bootstrap draw contains only one source cluster")

    selected_means = np.empty(len(methods), dtype=float)
    selected_counts: dict[str, np.ndarray] = {}
    for method_index, method in enumerate(methods):
        val = validation[method]
        tst = test[method]
        if val.shape != tst.shape or val.shape[0] != counts.size:
            raise ValueError(f"matrix shape mismatch for {method}")
        weighted_validation = counts @ val
        score_sum = 0.0
        choices = np.zeros(val.shape[1], dtype=np.int64)
        for target in active:
            calibration_n = n_positions - int(counts[target])
            if calibration_n <= 0:
                raise ValueError("no independent calibration cluster remains")
            calibration_mean = (
                weighted_validation - counts[target] * val[target]
            ) / calibration_n
            # Multiplier columns are sorted ascending, so np.argmax implements
            # the frozen lower-multiplier tie break.
            chosen = int(np.argmax(calibration_mean))
            choices[chosen] += int(counts[target])
            score_sum += float(counts[target]) * float(tst[target, chosen])
        selected_means[method_index] = score_sum / n_positions
        selected_counts[method] = choices
    return selected_means, selected_counts


def tuning_selection_bootstrap(
    payload: dict,
    *,
    n_resamples: int = 50_000,
    seed: int = 20_260_724,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap the complete six-design LOSO strength-selection procedure."""

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    methods, seeds, multipliers, validation, test = _strength_matrices(payload)
    pairs = list(itertools.combinations(methods, 2))
    family_size = len(pairs)
    rng = np.random.default_rng(seed)
    boot_means = np.empty((n_resamples, len(methods)), dtype=float)
    selection_totals = {
        method: np.zeros(len(multipliers[method]), dtype=np.int64)
        for method in methods
    }

    completed = 0
    while completed < n_resamples:
        sampled = rng.integers(0, len(seeds), size=len(seeds))
        counts = np.bincount(sampled, minlength=len(seeds))
        if np.count_nonzero(counts) < 2:
            continue
        means, selections = cluster_excluded_selected_means(
            counts, methods, validation, test
        )
        boot_means[completed] = means
        for method in methods:
            selection_totals[method] += selections[method]
        completed += 1

    lower_q = alpha / (2.0 * family_size)
    upper_q = 1.0 - lower_q
    pairwise = []
    for left_index, right_index in itertools.combinations(
        range(len(methods)), 2
    ):
        differences = boot_means[:, left_index] - boot_means[:, right_index]
        pairwise.append(
            {
                "left": methods[left_index],
                "right": methods[right_index],
                "bootstrap_mean_difference": float(differences.mean()),
                "bonferroni_percentile_low": float(
                    np.quantile(differences, lower_q)
                ),
                "bonferroni_percentile_high": float(
                    np.quantile(differences, upper_q)
                ),
            }
        )

    collective_rows = []
    for row in pairwise:
        if "B3_collective" not in (row["left"], row["right"]):
            continue
        oriented = dict(row)
        if row["left"] != "B3_collective":
            oriented["left"] = "B3_collective"
            oriented["right"] = row["left"]
            oriented["bootstrap_mean_difference"] = -row[
                "bootstrap_mean_difference"
            ]
            oriented["bonferroni_percentile_low"] = -row[
                "bonferroni_percentile_high"
            ]
            oriented["bonferroni_percentile_high"] = -row[
                "bonferroni_percentile_low"
            ]
        collective_rows.append(oriented)

    observed_counts = {
        method: payload["methods"][method]["leave_one_seed_out"][
            "selection_counts"
        ]
        for method in methods
    }
    return {
        "status": "complete",
        "estimand": (
            "mean held-out STM-capacity difference after repeating the complete "
            "leave-one-reservoir-out strength-selection rule"
        ),
        "bootstrap_seed": seed,
        "n_resamples": n_resamples,
        "n_source_reservoirs": len(seeds),
        "paired_cluster_resampling": True,
        "duplicate_target_exclusion": (
            "all copies of a held-out source reservoir ID are excluded from "
            "that target's calibration mean"
        ),
        "family_definition": "all 15 design pairs",
        "family_size": family_size,
        "family_alpha": alpha,
        "percentile_quantiles": [lower_q, upper_q],
        "observed_loso_selection_counts": observed_counts,
        "bootstrap_selection_counts": {
            method: {
                f"{multiplier:.12g}": int(count)
                for multiplier, count in zip(
                    multipliers[method],
                    selection_totals[method],
                    strict=True,
                )
            }
            for method in methods
        },
        "pairwise_intervals": pairwise,
        "collective_minus_competitor": collective_rows,
        "interpretation_boundary": (
            "This is a selection-aware sensitivity analysis over the empirical "
            "reservoir distribution. It is not a substitute for a newly frozen "
            "calibration ensemble followed by fresh disjoint six-design tests."
        ),
    }


def joint_profile_simultaneous_inference(
    rows: Sequence[dict],
    *,
    alpha: float = 0.05,
) -> dict:
    """All-pairs intervals for the channel-profile cells and sanity rung."""

    present = {str(row["method"]) for row in rows}
    methods = [method for method in JOINT_ORDER if method in present]
    if len(methods) != len(JOINT_ORDER):
        raise ValueError(
            "missing channel-profile cells: "
            f"{set(JOINT_ORDER) - present}"
        )
    primary_rows = [row for row in rows if row["method"] in methods]
    seeds = sorted({int(row["seed"]) for row in primary_rows})
    lookup = {
        (int(row["seed"]), str(row["method"])): float(row["value"])
        for row in primary_rows
    }
    expected = len(methods) * len(seeds)
    if len(lookup) != expected:
        raise ValueError(f"incomplete joint profile panel: {len(lookup)}/{expected}")
    family_size = len(methods) * (len(methods) - 1) // 2
    critical = float(
        stats.t.ppf(1.0 - alpha / (2.0 * family_size), len(seeds) - 1)
    )
    pairwise = []
    for left, right in itertools.combinations(methods, 2):
        differences = [
            lookup[(seed, left)] - lookup[(seed, right)] for seed in seeds
        ]
        mean, se = _mean_se(differences)
        pairwise.append(
            {
                "left": left,
                "right": right,
                "n": len(seeds),
                "mean_difference": mean,
                "se_difference": se,
                "simultaneous_low": mean - critical * se,
                "simultaneous_high": mean + critical * se,
                "wins": int(sum(value > 0.0 for value in differences)),
                "ties": int(sum(value == 0.0 for value in differences)),
            }
        )
    return {
        "status": "complete",
        "family_definition": (
            "all 10 pairwise contrasts among the four primary 2x2 cells and "
            "the unequal-collective sanity rung"
        ),
        "family_size": family_size,
        "family_alpha": alpha,
        "critical_t": critical,
        "degrees_of_freedom": len(seeds) - 1,
        "methods": methods,
        "seeds": seeds,
        "pairwise_intervals": pairwise,
        "interpretation_boundary": (
            "The interaction fixes h, dt, mean strength, ridge protocol, and "
            "optimizer cap; it is not a full joint operating-point optimization."
        ),
    }


def build_payload(
    tuning_path: Path,
    measurement_path: Path,
    joint_paths: Sequence[Path],
    *,
    n_resamples: int,
    bootstrap_seed: int,
) -> dict:
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
    joint_rows = [
        json.loads(path.read_text(encoding="utf-8")) for path in joint_paths
    ]
    return {
        "artifact_type": "critique_response_inference",
        "status": "complete",
        "source_sha256": {
            "six_channel_aggregate": _sha256(tuning_path),
            "measurement_full_aggregate": _sha256(measurement_path),
            "joint_profile_rows": hashlib.sha256(
                "".join(_sha256(path) for path in joint_paths).encode("ascii")
            ).hexdigest(),
        },
        "tuning_selection_bootstrap": tuning_selection_bootstrap(
            tuning,
            n_resamples=n_resamples,
            seed=bootstrap_seed,
        ),
        "measurement_simultaneous_inference": measurement_simultaneous_inference(
            measurement
        ),
        "joint_profile_simultaneous_inference": (
            joint_profile_simultaneous_inference(joint_rows)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning", type=Path, default=DEFAULT_TUNING)
    parser.add_argument("--measurement", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument(
        "--joint-dir",
        type=Path,
        default=ROOT / "results" / "final_protocol",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=50_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joint_paths = sorted(args.joint_dir.glob(DEFAULT_JOINT_GLOB))
    payload = build_payload(
        args.tuning,
        args.measurement,
        joint_paths,
        n_resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    measurement = payload["measurement_simultaneous_inference"]
    tuning = payload["tuning_selection_bootstrap"]
    print(f"wrote {args.output}")
    print(
        "measurement family:",
        measurement["family_size"],
        "critical t:",
        f"{measurement['critical_t']:.6f}",
    )
    print(
        "first familywise collective/local points:",
        measurement["first_familywise_resolved_collective_over_local"],
    )
    print(
        "tuning bootstrap:",
        tuning["n_resamples"],
        "resamples; collective lower bounds:",
        {
            row["right"]: round(row["bonferroni_percentile_low"], 6)
            for row in tuning["collective_minus_competitor"]
        },
    )


if __name__ == "__main__":
    main()

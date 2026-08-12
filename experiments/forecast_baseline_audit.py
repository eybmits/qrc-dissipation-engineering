#!/usr/bin/env python3
"""Audit simple baselines for the sealed Mackey--Glass forecast task.

This analysis does not rerun any reservoir. It regenerates the deterministic
target sequence for each sealed seed, using the same random-number stream as
``R_mgfix``, and compares the stored 150-step closed-loop errors with:

* a training-mean prediction, fixed to the mean of the one-step training
  targets; and
* a last-value prediction, fixed to the final observed training value.

Run from the repository root:

    PYTHONPATH=src:experiments python experiments/forecast_baseline_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import _paths  # noqa: F401
import numpy as np
from scipy.stats import t as student_t

from qrc import reservoirs as res
from qrc import tasks
from run_final_scaling import TCFG, deterministic_seeds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "review_protocol"
DEFAULT_OUTPUT = ROOT / "results" / "forecast_baseline_audit" / "aggregate.json"
DEFAULT_REPORT = ROOT / "reports" / "forecast_baseline_audit.md"

N_QUBITS = 5
WASH = 200
TRAIN = 600
HORIZON = 150
MG_TAU = 17
MG_BETA = 0.2
MG_DECAY = 0.1
MG_EXPONENT = 10
MG_DT = 1.0
MG_SAMPLE_EVERY = 3
MG_INITIAL_VALUE = 1.2
MG_DISCARD = 500
METHODS = (
    "FN",
    "CD_paper",
    "B3_collective",
    "A1_heterogeneous",
    "B5_pair",
    "B2_thermal",
    "B4_loss_exchange",
    "B1_dephasing",
)
METHOD_LABELS = {
    "FN": "reset-encoded FN",
    "CD_paper": "uniform local",
    "B3_collective": "collective",
    "A1_heterogeneous": "unequal local",
    "B5_pair": "pair",
    "B2_thermal": "local gain/loss",
    "B4_loss_exchange": "exchange-assisted",
    "B1_dephasing": "dephasing",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _summary(values: Sequence[float]) -> dict[str, float | int | list[float]]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires at least two finite values")
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(array.size))
    half_width = float(
        student_t.ppf(0.975, array.size - 1) * standard_error
    )
    return {
        "n": int(array.size),
        "mean": mean,
        "standard_error": standard_error,
        "descriptive_95pct_t_interval": [
            mean - half_width,
            mean + half_width,
        ],
    }


def _paired_comparison(
    method_values: Sequence[float],
    baseline_values: Sequence[float],
) -> dict[str, object]:
    method = np.asarray(method_values, dtype=float)
    baseline = np.asarray(baseline_values, dtype=float)
    if method.shape != baseline.shape:
        raise ValueError("paired arrays must have the same shape")
    differences = method - baseline
    result: dict[str, object] = _summary(differences)
    interval = result["descriptive_95pct_t_interval"]
    assert isinstance(interval, list)
    result.update(
        {
            "difference_definition": "reservoir MSE minus baseline MSE",
            "negative_is_better_than_baseline": True,
            "wins": int(np.sum(differences < 0.0)),
            "ties": int(np.sum(differences == 0.0)),
            "losses": int(np.sum(differences > 0.0)),
            "interval_excludes_zero_in_better_direction": bool(
                float(interval[1]) < 0.0
            ),
            "interval_excludes_zero_in_worse_direction": bool(
                float(interval[0]) > 0.0
            ),
        }
    )
    return result


def load_sealed_scores(
    input_directory: Path,
) -> tuple[list[int], dict[str, dict[int, float]], dict[str, object]]:
    """Load and validate the complete 8-method by 64-seed ``R_mgfix`` grid."""
    expected_seeds = deterministic_seeds(64)
    expected_seed_set = set(expected_seeds)
    scores: dict[str, dict[int, float]] = {method: {} for method in METHODS}
    selected_files: list[tuple[str, bytes]] = []

    for path in sorted(input_directory.glob("R_mgfix__*.json")):
        data = path.read_bytes()
        row = json.loads(data)
        method = row.get("method")
        if method not in scores:
            continue
        seed = int(row["seed"])
        if (
            row.get("block") != "R_mgfix"
            or row.get("task") != "mg"
            or int(row.get("N", -1)) != N_QUBITS
        ):
            raise ValueError(f"unexpected forecast row contract: {path}")
        if seed in scores[method]:
            raise ValueError(f"duplicate method/seed row: {method}/{seed}")
        value = float(row["value"])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid forecast MSE: {path}")
        scores[method][seed] = value
        selected_files.append((path.relative_to(ROOT).as_posix(), data))

    for method in METHODS:
        observed = set(scores[method])
        if observed != expected_seed_set:
            missing = sorted(expected_seed_set - observed)
            extra = sorted(observed - expected_seed_set)
            raise ValueError(
                f"incomplete forecast grid for {method}: "
                f"missing={missing}, extra={extra}"
            )

    digest = hashlib.sha256()
    for relative, data in selected_files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    provenance = {
        "input_glob": "results/review_protocol/R_mgfix__*.json",
        "selected_file_count": len(selected_files),
        "selected_payload_sha256": digest.hexdigest(),
    }
    return expected_seeds, scores, provenance


def regenerate_simple_baselines(seed: int) -> dict[str, float]:
    """Regenerate target-only baselines on the exact sealed random stream."""
    if (
        HORIZON != TCFG.mg_horizon
        or MG_SAMPLE_EVERY != TCFG.mg_sample_every
    ):
        raise ValueError("forecast audit constants drifted from the sealed task")
    rng = np.random.default_rng(seed)

    # R_mgfix consumes the coupling draw before generating the target series.
    # Keeping that draw is necessary to reproduce each sealed target exactly.
    _ = res.random_couplings(N_QUBITS, 1.0, rng)

    n_samples = WASH + TRAIN + HORIZON + 5
    series = tasks.mackey_glass_series(
        n_samples,
        tau=MG_TAU,
        beta=MG_BETA,
        gamma=MG_DECAY,
        n_exp=MG_EXPONENT,
        dt=MG_DT,
        sample_every=MG_SAMPLE_EVERY,
        x0=MG_INITIAL_VALUE,
        discard=MG_DISCARD,
        rng=rng,
    )
    train_prefix = series[: WASH + TRAIN]
    minimum = float(np.min(train_prefix))
    maximum = float(np.max(train_prefix))
    series = (series - minimum) / max(maximum - minimum, 1e-12)

    # The stored reservoir fit uses X[:-1] against the 599 one-step targets
    # series[WASH + 1 : WASH + TRAIN].
    training_targets = series[WASH + 1 : WASH + TRAIN]
    truth = series[WASH + TRAIN : WASH + TRAIN + HORIZON]
    training_mean = float(np.mean(training_targets))
    last_value = float(series[WASH + TRAIN - 1])
    return {
        "training_mean_prediction": training_mean,
        "last_value_prediction": last_value,
        "training_mean_mse": float(np.mean((truth - training_mean) ** 2)),
        "last_value_mse": float(np.mean((truth - last_value) ** 2)),
    }


def build_audit(input_directory: Path = DEFAULT_INPUT) -> dict[str, object]:
    seeds, scores, source = load_sealed_scores(input_directory)
    baseline_rows = {
        seed: regenerate_simple_baselines(seed) for seed in seeds
    }
    training_mean_mse = [
        baseline_rows[seed]["training_mean_mse"] for seed in seeds
    ]
    last_value_mse = [baseline_rows[seed]["last_value_mse"] for seed in seeds]

    method_summaries: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    for method in METHODS:
        values = [scores[method][seed] for seed in seeds]
        method_summaries[method] = {
            "label": METHOD_LABELS[method],
            **_summary(values),
        }
        comparisons[method] = {
            "label": METHOD_LABELS[method],
            "versus_training_mean": _paired_comparison(
                values, training_mean_mse
            ),
            "versus_last_value": _paired_comparison(values, last_value_mse),
        }

    dephasing_difference = [
        scores["B1_dephasing"][seed]
        - baseline_rows[seed]["training_mean_mse"]
        for seed in seeds
    ]
    per_seed = [
        {
            "seed": seed,
            "baselines": baseline_rows[seed],
            "reservoir_mse": {
                method: scores[method][seed] for method in METHODS
            },
        }
        for seed in seeds
    ]
    result = {
        "artifact_type": "sealed_forecast_baseline_audit",
        "schema_version": 1,
        "status": "complete",
        "analysis_status": (
            "post-hoc benchmark context added after review; "
            "no reservoir rerun or model selection"
        ),
        "expected_checkpoints": len(METHODS) * len(seeds),
        "complete_checkpoints": len(METHODS) * len(seeds),
        "missing_checkpoints": [],
        "protocol": {
            "task": "Mackey-Glass 150-step closed-loop forecasting",
            "n_qubits": N_QUBITS,
            "washout_steps": WASH,
            "training_steps": TRAIN,
            "forecast_horizon": HORIZON,
            "mackey_glass": {
                "tau": MG_TAU,
                "beta": MG_BETA,
                "decay": MG_DECAY,
                "exponent": MG_EXPONENT,
                "integration_step": MG_DT,
                "sample_every": MG_SAMPLE_EVERY,
                "initial_value": MG_INITIAL_VALUE,
                "discarded_integration_samples": MG_DISCARD,
            },
            "target_scaling": (
                "minimum and maximum from the washout-plus-training prefix"
            ),
            "random_stream_alignment": (
                "consume the N=5 coupling draw before target generation"
            ),
            "training_mean_baseline": (
                "one constant prediction equal to the mean of the 599 "
                "one-step training targets"
            ),
            "last_value_baseline": (
                "one constant prediction equal to the final observed "
                "training value"
            ),
            "inference": "paired descriptive 95% Student-t intervals",
            "seed_count": len(seeds),
            "seeds": seeds,
        },
        "source": source,
        "baseline_summaries": {
            "training_mean": _summary(training_mean_mse),
            "last_value": _summary(last_value_mse),
        },
        "method_summaries": method_summaries,
        "paired_comparisons": comparisons,
        "validation": {
            "status": "complete",
            "complete_method_seed_grid": True,
            "dephasing_matches_training_mean_seedwise": bool(
                max(abs(value) for value in dephasing_difference) < 1e-12
            ),
            "maximum_dephasing_minus_training_mean_mse_absolute_error": max(
                abs(value) for value in dephasing_difference
            ),
        },
        "per_seed": per_seed,
    }
    result["deterministic_payload_sha256"] = _sha256(
        _canonical_json(result).encode("utf-8")
    )
    return result


def build_report(result: dict[str, object]) -> str:
    baselines = result["baseline_summaries"]
    methods = result["method_summaries"]
    comparisons = result["paired_comparisons"]
    assert isinstance(baselines, dict)
    assert isinstance(methods, dict)
    assert isinstance(comparisons, dict)

    def value(summary: object, key: str) -> float:
        assert isinstance(summary, dict)
        return float(summary[key])

    collective = comparisons["B3_collective"]
    assert isinstance(collective, dict)
    collective_mean = collective["versus_training_mean"]
    collective_last = collective["versus_last_value"]
    assert isinstance(collective_mean, dict)
    assert isinstance(collective_last, dict)
    mean_interval = collective_mean["descriptive_95pct_t_interval"]
    last_interval = collective_last["descriptive_95pct_t_interval"]
    assert isinstance(mean_interval, list)
    assert isinstance(last_interval, list)

    lines = [
        "# Forecast baseline audit",
        "",
        "## Scope",
        "",
        "This post-hoc check adds two simple reference predictions to the "
        "sealed 150-step closed-loop Mackey--Glass task. It regenerates only "
        "the deterministic target sequences; it does not rerun a reservoir "
        "or select any model setting.",
        "",
        "## Results",
        "",
        "| model or simple prediction | mean MSE | standard error |",
        "|---|---:|---:|",
        (
            "| training-mean prediction | "
            f"{value(baselines['training_mean'], 'mean'):.5f} | "
            f"{value(baselines['training_mean'], 'standard_error'):.5f} |"
        ),
        (
            "| last-value prediction | "
            f"{value(baselines['last_value'], 'mean'):.5f} | "
            f"{value(baselines['last_value'], 'standard_error'):.5f} |"
        ),
    ]
    for method in METHODS:
        summary = methods[method]
        assert isinstance(summary, dict)
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{value(summary, 'mean'):.5f} | "
            f"{value(summary, 'standard_error'):.5f} |"
        )
    lines.extend(
        [
            "",
            "Collective loss is worse than the training-mean prediction by "
            f"{value(collective_mean, 'mean'):.5f} MSE "
            f"(paired 95% interval [{float(mean_interval[0]):.5f}, "
            f"{float(mean_interval[1]):.5f}]). Its difference from the "
            "last-value prediction is "
            f"{value(collective_last, 'mean'):.5f} "
            f"([{float(last_interval[0]):.5f}, "
            f"{float(last_interval[1]):.5f}]), so that comparison is "
            "not resolved.",
            "",
            "Uniform local, unequal-local, pair, and local gain/loss each have "
            "lower errors than both simple predictions, with pointwise "
            "descriptive 95% intervals that exclude zero. "
            "Dephasing matches the training-mean prediction seed by seed to "
            "numerical precision. Several reservoir forecasts therefore "
            "outperform these two constant predictions, whereas the "
            "collective memory advantage does not transfer to this 150-step "
            "closed-loop task at the fixed-map operating point.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "PYTHONPATH=src:experiments python "
            "experiments/forecast_baseline_audit.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)

    result = build_audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_canonical_json(result), encoding="utf-8")
    args.report.write_text(build_report(result), encoding="utf-8")
    print(f"wrote {args.output.relative_to(ROOT)}")
    print(f"wrote {args.report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

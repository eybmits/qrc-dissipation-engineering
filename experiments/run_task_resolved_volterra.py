#!/usr/bin/env python3
"""Deterministic task-resolved Volterra reproduction entrypoint.

The smoke profile is suitable for CI. The validation profile uses the project
configuration: a matched-gap N=3 sweep, a fixed-trace N=4 replication, exact
switched-channel validation, and delayed-product tasks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc.operators import pauli_op, sminus, two_site_pauli
from qrc.readout import capacity, pauli_observables
from qrc.task_resolved import (
    build_kernel_library,
    input_affine_channel_expansion,
    input_affine_liouvillian_from_kossakowski,
    interpolated_kossakowski,
    primitive_gap,
    product_capacity,
)


@dataclass(frozen=True)
class Profile:
    n3_seeds: int
    n4_seeds: int
    exact_n3_seeds: int
    exact_n4_seeds: int
    alphas_n3: tuple[float, ...]
    alphas_n4: tuple[float, ...]
    max_delay_n3: int
    max_delay_n4: int
    train_length: int
    test_length: int
    washout: int


PROFILES = {
    "smoke": Profile(
        2, 0, 1, 0,
        (0.0, 0.5, 0.75), (),
        20, 0, 600, 600, 200,
    ),
    "validation": Profile(
        12, 4, 6, 2,
        (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90),
        (0.0, 0.30, 0.60, 0.85),
        50, 28, 3000, 3000, 500,
    ),
}

TASKS = (
    (1, 2),
    (1, 5),
    (1, 10),
    (5, 10),
    (5, 15),
    (10, 20),
    (15, 25),
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def random_processor(n_qubits: int, seed: int):
    rng = np.random.default_rng(seed)
    dimension = 2**n_qubits
    Hxx = np.zeros((dimension, dimension), complex)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            Hxx += rng.uniform(-1.0, 1.0) * two_site_pauli(
                "x", i, j, n_qubits
            )
    Hz = sum(
        (pauli_op("z", i, n_qubits) for i in range(n_qubits)),
        start=np.zeros((dimension, dimension), complex),
    )
    Hx = sum(
        (pauli_op("x", i, n_qubits) for i in range(n_qubits)),
        start=np.zeros((dimension, dimension), complex),
    )
    return Hxx + Hz + Hx, Hx


def make_generator(
    n_qubits: int,
    H_static: np.ndarray,
    H_drive: np.ndarray,
    alpha: float,
    gamma: float,
):
    matrix = interpolated_kossakowski(n_qubits, gamma, alpha)
    basis = [sminus(i, n_qubits) for i in range(n_qubits)]
    return input_affine_liouvillian_from_kossakowski(
        H_static,
        H_drive,
        basis,
        matrix,
    )


def gap_at(
    n_qubits: int,
    H_static: np.ndarray,
    H_drive: np.ndarray,
    alpha: float,
    gamma: float,
):
    base, drive = make_generator(
        n_qubits,
        H_static,
        H_drive,
        alpha,
        gamma,
    )
    return primitive_gap(base + 0.5 * drive)


def match_gamma(
    n_qubits: int,
    H_static: np.ndarray,
    H_drive: np.ndarray,
    alpha: float,
    target_gap: float,
):
    grid = np.geomspace(1e-3, 3.0, 24)
    points = []
    for gamma in grid:
        diagnostic = gap_at(
            n_qubits,
            H_static,
            H_drive,
            alpha,
            float(gamma),
        )
        if diagnostic.stable and diagnostic.gap > 0:
            points.append((float(gamma), diagnostic.gap))
    if not points:
        return math.nan, False

    for left, right in zip(points[:-1], points[1:]):
        if (left[1] - target_gap) * (right[1] - target_gap) > 0:
            continue

        def objective(log_gamma: float) -> float:
            diagnostic = gap_at(
                n_qubits,
                H_static,
                H_drive,
                alpha,
                math.exp(log_gamma),
            )
            if not diagnostic.stable:
                raise ValueError("unstable bracket")
            return diagnostic.gap - target_gap

        try:
            gamma = math.exp(
                brentq(
                    objective,
                    math.log(left[0]),
                    math.log(right[0]),
                    maxiter=50,
                )
            )
            achieved = gap_at(
                n_qubits,
                H_static,
                H_drive,
                alpha,
                gamma,
            )
            relative_error = abs(achieved.gap - target_gap) / target_gap
            return gamma, achieved.stable and relative_error <= 5e-3
        except (ValueError, RuntimeError):
            pass

    gamma, achieved_gap = min(
        points,
        key=lambda point: abs(point[1] - target_gap),
    )
    return gamma, abs(achieved_gap - target_gap) / target_gap <= 5e-3


def expansion_record(
    n_qubits: int,
    seed: int,
    alpha: float,
    gamma: float,
    max_delay: int,
):
    H_static, H_drive = random_processor(n_qubits, seed)
    base, drive = make_generator(
        n_qubits,
        H_static,
        H_drive,
        alpha,
        gamma,
    )
    expansion = input_affine_channel_expansion(
        base,
        drive,
        1.0,
        0.5,
        pauli_observables(n_qubits, max_weight=2),
    )
    return {
        "base": base,
        "drive": drive,
        "expansion": expansion,
        "library": build_kernel_library(
            expansion,
            max_delay,
            0.02,
            ridge=1e-13,
        ),
    }


def simulate(record: dict, total_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.choice(np.array([-1.0, 1.0]), size=total_steps)
    plus = expm(record["base"] + 0.52 * record["drive"])
    minus = expm(record["base"] + 0.48 * record["drive"])
    state = record["expansion"].fixed_point.copy()
    readout = record["expansion"].readout
    features = np.empty((total_steps, readout.shape[0]), float)
    for t, sign in enumerate(z):
        state = (plus if sign > 0 else minus) @ state
        features[t] = np.real_if_close(readout @ state).real
    return features, z


def fit_capacity(X_train, y_train, X_test, y_test):
    mean = X_train.mean(axis=0, keepdims=True)
    scale = X_train.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    Xtr = (X_train - mean) / scale
    Xte = (X_test - mean) / scale
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    weights = np.linalg.solve(
        Xtr.T @ Xtr + 1e-10 * np.eye(Xtr.shape[1]),
        Xtr.T @ y_train,
    )
    return capacity(y_test, Xte @ weights)


def run(profile: Profile, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    theory_rows = []
    exact_rows = []
    cache = {}

    for n_qubits, seed_count, alphas, max_delay, matched in (
        (3, profile.n3_seeds, profile.alphas_n3, profile.max_delay_n3, True),
        (4, profile.n4_seeds, profile.alphas_n4, profile.max_delay_n4, False),
    ):
        for seed in range(seed_count):
            H_static, H_drive = random_processor(n_qubits, seed)
            local_gap = gap_at(n_qubits, H_static, H_drive, 0.0, 0.1)
            for alpha in alphas:
                if matched:
                    gamma, feasible = match_gamma(
                        n_qubits,
                        H_static,
                        H_drive,
                        alpha,
                        local_gap.gap,
                    )
                    if not feasible:
                        continue
                else:
                    gamma = 0.1
                record = expansion_record(
                    n_qubits,
                    seed,
                    alpha,
                    gamma,
                    max_delay,
                )
                cache[(n_qubits, seed, alpha)] = record
                row = {
                    "n_qubits": n_qubits,
                    "seed": seed,
                    "alpha": alpha,
                    "gamma": gamma,
                }
                for task in TASKS:
                    if task[1] <= max_delay:
                        row[f"product_{task[0]}_{task[1]}"] = product_capacity(
                            record["library"],
                            *task,
                        )
                theory_rows.append(row)
    write_csv(output / "theory.csv", theory_rows)

    for n_qubits, exact_count, total_count, alphas, max_delay in (
        (
            3,
            profile.exact_n3_seeds,
            profile.n3_seeds,
            profile.alphas_n3,
            profile.max_delay_n3,
        ),
        (
            4,
            profile.exact_n4_seeds,
            profile.n4_seeds,
            profile.alphas_n4,
            profile.max_delay_n4,
        ),
    ):
        if exact_count == 0:
            continue
        total_steps = (
            profile.washout
            + max_delay
            + profile.train_length
            + profile.test_length
        )
        start = profile.washout + max_delay
        train_indices = np.arange(start, start + profile.train_length)
        test_indices = np.arange(start + profile.train_length, total_steps)
        for seed in range(max(0, total_count - exact_count), total_count):
            for alpha in alphas:
                record = cache.get((n_qubits, seed, alpha))
                if record is None:
                    continue
                features, z = simulate(
                    record,
                    total_steps,
                    100000 + n_qubits * 10000 + seed,
                )
                for task in TASKS:
                    if task[1] > max_delay:
                        continue
                    y_train = z[train_indices - task[0]] * z[train_indices - task[1]]
                    y_test = z[test_indices - task[0]] * z[test_indices - task[1]]
                    exact_rows.append({
                        "n_qubits": n_qubits,
                        "seed": seed,
                        "alpha": alpha,
                        "task": f"{task[0]},{task[1]}",
                        "predicted_capacity": product_capacity(
                            record["library"],
                            *task,
                        ),
                        "empirical_capacity": fit_capacity(
                            features[train_indices],
                            y_train,
                            features[test_indices],
                            y_test,
                        ),
                    })
    write_csv(output / "exact_validation.csv", exact_rows)

    predicted = np.array([
        float(row["predicted_capacity"])
        for row in exact_rows
    ])
    empirical = np.array([
        float(row["empirical_capacity"])
        for row in exact_rows
    ])
    summary = {
        "theory_rows": len(theory_rows),
        "exact_rows": len(exact_rows),
        "exact_pearson": (
            float(np.corrcoef(predicted, empirical)[0, 1])
            if len(predicted) > 1
            else None
        ),
        "exact_mae": (
            float(np.mean(np.abs(predicted - empirical)))
            if len(predicted)
            else None
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/task_resolved_volterra/reproduction",
    )
    args = parser.parse_args()
    run(PROFILES[args.profile], args.output)


if __name__ == "__main__":
    main()

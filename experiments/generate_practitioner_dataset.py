#!/usr/bin/env python3
"""Generate the exhaustive finite-family QRC prescreening data set.

For each random three-qubit XX+Z processor, seven local-to-collective
Kossakowski geometries are calibrated to the local candidate's constant-input
Liouvillian gap. A differentiated-channel Walsh-Volterra capacity is computed
without task training. Every candidate is then fully simulated and trained on
seven delayed-product tasks to define the paired empirical oracle.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Sequence

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

ALPHAS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
TASKS = ((1, 2), (1, 5), (1, 10), (5, 10), (5, 15), (10, 20), (15, 25))


def random_processor(n_qubits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    dimension = 2**n_qubits
    h_xx = np.zeros((dimension, dimension), complex)
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            h_xx += rng.uniform(-1.0, 1.0) * two_site_pauli("x", i, j, n_qubits)
    h_z = sum(
        (pauli_op("z", i, n_qubits) for i in range(n_qubits)),
        start=np.zeros((dimension, dimension), complex),
    )
    h_x = sum(
        (pauli_op("x", i, n_qubits) for i in range(n_qubits)),
        start=np.zeros((dimension, dimension), complex),
    )
    return h_xx + h_z + h_x, h_x


def make_generator(
    n_qubits: int,
    static_hamiltonian: np.ndarray,
    drive_hamiltonian: np.ndarray,
    alpha: float,
    gamma: float,
    coefficients: Sequence[complex] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = interpolated_kossakowski(
        n_qubits, gamma, alpha, coefficients
    )
    basis = [sminus(i, n_qubits) for i in range(n_qubits)]
    return input_affine_liouvillian_from_kossakowski(
        static_hamiltonian,
        drive_hamiltonian,
        basis,
        matrix,
    )


def gap_at(
    n_qubits: int,
    static_hamiltonian: np.ndarray,
    drive_hamiltonian: np.ndarray,
    alpha: float,
    gamma: float,
    reference_input: float,
):
    base, drive = make_generator(
        n_qubits,
        static_hamiltonian,
        drive_hamiltonian,
        alpha,
        gamma,
    )
    return primitive_gap(base + reference_input * drive)


def match_gamma(
    n_qubits: int,
    static_hamiltonian: np.ndarray,
    drive_hamiltonian: np.ndarray,
    alpha: float,
    target_gap: float,
    reference_input: float,
) -> tuple[float, float, float, bool]:
    scan = np.geomspace(1e-3, 3.0, 32)
    points: list[tuple[float, float]] = []
    for gamma in scan:
        diagnostic = gap_at(
            n_qubits,
            static_hamiltonian,
            drive_hamiltonian,
            alpha,
            float(gamma),
            reference_input,
        )
        if diagnostic.stable and diagnostic.gap > 0:
            points.append((float(gamma), float(diagnostic.gap)))
    if not points:
        return math.nan, math.nan, math.inf, False

    roots: list[float] = []
    for left, right in zip(points[:-1], points[1:]):
        left_value = left[1] - target_gap
        right_value = right[1] - target_gap
        if left_value == 0:
            roots.append(left[0])
            continue
        if left_value * right_value > 0:
            continue

        def objective(log_gamma: float) -> float:
            diagnostic = gap_at(
                n_qubits,
                static_hamiltonian,
                drive_hamiltonian,
                alpha,
                math.exp(log_gamma),
                reference_input,
            )
            if not diagnostic.stable:
                raise ValueError("non-primitive point inside bracket")
            return diagnostic.gap - target_gap

        try:
            roots.append(
                math.exp(
                    brentq(
                        objective,
                        math.log(left[0]),
                        math.log(right[0]),
                        maxiter=80,
                    )
                )
            )
        except (RuntimeError, ValueError):
            pass

    gamma = roots[0] if roots else min(
        points, key=lambda item: abs(item[1] - target_gap)
    )[0]
    achieved = gap_at(
        n_qubits,
        static_hamiltonian,
        drive_hamiltonian,
        alpha,
        gamma,
        reference_input,
    )
    relative_error = abs(float(achieved.gap) - target_gap) / max(target_gap, 1e-15)
    return gamma, float(achieved.gap), relative_error, bool(
        achieved.stable and relative_error <= 5e-3
    )


def fit_capacity(features_train, target_train, features_test, target_test) -> float:
    mean = features_train.mean(axis=0, keepdims=True)
    scale = features_train.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    train = (features_train - mean) / scale
    test = (features_test - mean) / scale
    train = np.hstack([train, np.ones((train.shape[0], 1))])
    test = np.hstack([test, np.ones((test.shape[0], 1))])
    weights = np.linalg.solve(
        train.T @ train + 1e-10 * np.eye(train.shape[1]),
        train.T @ target_train,
    )
    return capacity(target_test, test @ weights)


def simulate_features(base, drive, fixed_point, readout, signs, epsilon, reference_input):
    plus = expm(base + (reference_input + epsilon) * drive)
    minus = expm(base + (reference_input - epsilon) * drive)
    state = fixed_point.copy()
    features = np.empty((len(signs), readout.shape[0]), float)
    for index, sign in enumerate(signs):
        state = (plus if sign > 0 else minus) @ state
        features[index] = np.real_if_close(readout @ state).real
    return features


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows generated")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/task_resolved_volterra/prescreen_7candidate_all12.csv",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=12)
    parser.add_argument("--max-delay", type=int, default=50)
    parser.add_argument("--washout", type=int, default=500)
    parser.add_argument("--train", type=int, default=1500)
    parser.add_argument("--test", type=int, default=1500)
    parser.add_argument("--epsilon", type=float, default=0.02)
    args = parser.parse_args()

    n_qubits = 3
    reference_input = 0.5
    observables = pauli_observables(n_qubits, max_weight=2)
    total_steps = args.washout + args.max_delay + args.train + args.test
    train_start = args.washout + args.max_delay
    train_indices = np.arange(train_start, train_start + args.train)
    test_indices = np.arange(train_start + args.train, total_steps)

    rows: list[dict] = []
    calibration_rows: list[dict] = []
    for seed in range(args.seed_start, args.seed_stop):
        static_hamiltonian, drive_hamiltonian = random_processor(n_qubits, seed)
        local = gap_at(
            n_qubits,
            static_hamiltonian,
            drive_hamiltonian,
            0.0,
            0.1,
            reference_input,
        )
        if not local.stable:
            raise RuntimeError(f"local reference not primitive for seed {seed}")
        signs = np.random.default_rng(130000 + seed).choice(
            np.array([-1.0, 1.0]), size=total_steps
        )

        for alpha in ALPHAS:
            gamma, gap, gap_error, feasible = match_gamma(
                n_qubits,
                static_hamiltonian,
                drive_hamiltonian,
                alpha,
                float(local.gap),
                reference_input,
            )
            if not feasible:
                raise RuntimeError(
                    f"gap matching failed seed={seed} alpha={alpha}: {gap_error}"
                )
            base, drive = make_generator(
                n_qubits,
                static_hamiltonian,
                drive_hamiltonian,
                alpha,
                gamma,
            )
            expansion = input_affine_channel_expansion(
                base,
                drive,
                1.0,
                reference_input,
                observables,
            )
            library = build_kernel_library(
                expansion,
                args.max_delay,
                args.epsilon,
                ridge=1e-13,
            )
            features = simulate_features(
                base,
                drive,
                expansion.fixed_point,
                expansion.readout,
                signs,
                args.epsilon,
                reference_input,
            )
            for delay_a, delay_b in TASKS:
                target_train = signs[train_indices - delay_a] * signs[train_indices - delay_b]
                target_test = signs[test_indices - delay_a] * signs[test_indices - delay_b]
                rows.append(
                    {
                        "seed": seed,
                        "task": f"{delay_a},{delay_b}",
                        "alpha": alpha,
                        "gamma": gamma,
                        "target_gap": float(local.gap),
                        "achieved_gap": gap,
                        "relative_gap_error": gap_error,
                        "predicted_capacity": product_capacity(library, delay_a, delay_b),
                        "empirical_capacity": fit_capacity(
                            features[train_indices],
                            target_train,
                            features[test_indices],
                            target_test,
                        ),
                    }
                )
            calibration_rows.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "gamma": gamma,
                    "target_gap": float(local.gap),
                    "achieved_gap": gap,
                    "relative_gap_error": gap_error,
                }
            )
        print(f"completed seed {seed}", flush=True)

    write_rows(args.output, rows)
    write_rows(
        args.output.with_name(args.output.stem + "_gap_calibration.csv"),
        calibration_rows,
    )
    metadata = {
        "n_qubits": n_qubits,
        "seeds": [args.seed_start, args.seed_stop - 1],
        "candidate_alphas": list(ALPHAS),
        "tasks": [f"{a},{b}" for a, b in TASKS],
        "epsilon": args.epsilon,
        "max_delay": args.max_delay,
        "washout": args.washout,
        "train": args.train,
        "test": args.test,
        "same_input_within_seed": True,
        "gap_match_relative_tolerance": 0.005,
        "rows": len(rows),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

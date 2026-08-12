#!/usr/bin/env python3
"""Strict reset-architecture replication from the canonical ``qrc`` package.

The protocol is intentionally fixed:

* N=5, h=dt=0.5, gamma=1 and assigned jump weight B=80;
* washout/train/test = 800/600/400;
* STM delays 1..20 and NARMA-10;
* 45 Pauli features, bias, and ridge=1e-8;
* 16 paired local/collective reservoirs generated from master seed 2026080603.

By default the run also repeats both exact tasks from four initial states.  All
artifacts are portable, machine readable, source stamped, and covered by a
SHA-256 manifest.  Use ``--verify DIR`` to validate a completed result folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy
from scipy.stats import binomtest
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

import _paths
from qrc import dissipators, readout, reservoirs, tasks
from qrc.liouvillian import unvec, vec


@dataclass(frozen=True)
class Protocol:
    n_qubits: int = 5
    coupling_scale: float = 1.0
    h: float = 0.5
    dt: float = 0.5
    gamma: float = 1.0
    washout: int = 800
    train_len: int = 600
    test_len: int = 400
    stm_delay_start: int = 1
    stm_delay_stop: int = 20
    narma_order: int = 10
    narma_input_scale: float = 0.2
    ridge: float = 1e-8
    n_pairs: int = 16
    master_seed: int = 2026080603
    haar_seed_xor: int = 0x2468ACE0

    @property
    def total_len(self) -> int:
        return self.washout + self.train_len + self.test_len

    @property
    def stm_delays(self) -> tuple[int, ...]:
        return tuple(range(self.stm_delay_start, self.stm_delay_stop + 1))


PROTOCOL = Protocol()
MODELS = ("local", "collective")
INITIAL_STATES = ("ground", "excited", "mixed", "haar")
DEFAULT_OUT = Path(_paths.ROOT) / "results" / "reset_architecture_replication"
SOURCE_FILES = (
    "experiments/run_reset_architecture_strict.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/tasks.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _git_metadata(root: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                args,
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def protocol_record() -> dict:
    root = Path(_paths.ROOT)
    local_jumps = dissipators.local_loss(
        PROTOCOL.n_qubits,
        PROTOCOL.gamma,
    )
    collective_jumps = dissipators.collective_loss(
        PROTOCOL.n_qubits,
        PROTOCOL.gamma,
    )
    source_hashes = {
        relative: _sha256(root / relative)
        for relative in SOURCE_FILES
    }
    return {
        "experiment": "reset_encoded_local_vs_collective_strict_replication",
        "protocol": {
            **asdict(PROTOCOL),
            "total_len": PROTOCOL.total_len,
            "stm_delays": list(PROTOCOL.stm_delays),
            "models": list(MODELS),
            "initial_state_audit": list(INITIAL_STATES),
            "readout": {
                "observables": (
                    "all one-body and same-axis two-body Pauli expectations"
                ),
                "n_features": 45,
                "bias": True,
                "ridge": PROTOCOL.ridge,
            },
            "pairing": (
                "Hamiltonian, task input, target, and split are identical for "
                "local and collective conditions within each seed."
            ),
            "structural_budget": {
                "definition": "sum_k rate_k Tr(L_k^dagger L_k)",
                "local": dissipators.jump_strength(local_jumps),
                "collective": dissipators.jump_strength(collective_jumps),
            },
        },
        "ordered_seeds": ordered_seeds().tolist(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "repository": _git_metadata(root),
        "source_sha256": source_hashes,
    }


def ordered_seeds() -> np.ndarray:
    master = np.random.default_rng(PROTOCOL.master_seed)
    return master.integers(
        0,
        2**31 - 1,
        size=PROTOCOL.n_pairs,
        dtype=np.int64,
    )


def initial_states(seed: int) -> dict[str, np.ndarray]:
    d = 2 ** PROTOCOL.n_qubits
    ground = np.zeros((d, d), dtype=complex)
    ground[0, 0] = 1.0
    excited = np.zeros((d, d), dtype=complex)
    excited[-1, -1] = 1.0
    mixed = np.eye(d, dtype=complex) / d
    rng = np.random.default_rng(int(seed) ^ PROTOCOL.haar_seed_xor)
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    psi /= np.linalg.norm(psi)
    haar = np.outer(psi, psi.conj())
    return {
        "ground": ground,
        "excited": excited,
        "mixed": mixed,
        "haar": haar,
    }


def run_initial_state_ensemble(
    reservoir: reservoirs.ResetLindbladReservoir,
    inputs: np.ndarray,
    observable_matrices: np.ndarray,
    rho0: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, float]:
    """Run all initial states together through one static propagator."""
    names = tuple(rho0)
    states = [rho0[name].copy() for name in names]
    features = {
        name: np.empty(
            (PROTOCOL.train_len + PROTOCOL.test_len, len(observable_matrices)),
            dtype=float,
        )
        for name in names
    }
    worst_post_washout_feature_distance = 0.0
    trace_distance_at_washout = float("nan")

    for step, s in enumerate(inputs):
        injected = [
            reservoirs.inject_input(rho, float(s), PROTOCOL.n_qubits)
            for rho in states
        ]
        evolved = reservoir.P @ np.column_stack([vec(rho) for rho in injected])
        states = [unvec(evolved[:, i], reservoir.H.shape[0]) for i in range(len(names))]
        if step + 1 == PROTOCOL.washout:
            trace_distance_at_washout = max_pairwise_trace_distance(states)
        if step < PROTOCOL.washout:
            continue
        row = step - PROTOCOL.washout
        state_stack = np.stack(states)
        values = np.real(
            np.einsum("kij,sji->sk", observable_matrices, state_stack)
        )
        for i, name in enumerate(names):
            features[name][row] = values[i]
        for i, j in combinations(range(len(names)), 2):
            worst_post_washout_feature_distance = max(
                worst_post_washout_feature_distance,
                float(np.max(np.abs(values[i] - values[j]))),
            )

    return (
        features,
        dict(zip(names, states, strict=True)),
        worst_post_washout_feature_distance,
        trace_distance_at_washout,
    )


def _masks(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.zeros(len(target), dtype=bool)
    test = np.zeros(len(target), dtype=bool)
    train[: PROTOCOL.train_len] = True
    test[
        PROTOCOL.train_len : PROTOCOL.train_len + PROTOCOL.test_len
    ] = True
    valid = np.isfinite(target)
    return train & valid, test & valid


def evaluate_stm(
    features: np.ndarray,
    post_inputs: np.ndarray,
) -> tuple[float, np.ndarray]:
    X = readout.add_bias(features)
    capacities = []
    for delay in PROTOCOL.stm_delays:
        target = tasks.delayed_target(post_inputs, delay)
        train, test = _masks(target)
        weights = readout.train_readout(
            X[train],
            target[train],
            ridge=PROTOCOL.ridge,
        )
        capacities.append(
            readout.capacity(
                target[test],
                readout.predict(X[test], weights),
            )
        )
    lag_capacities = np.asarray(capacities)
    return float(lag_capacities.sum()), lag_capacities


def evaluate_narma(features: np.ndarray, post_inputs: np.ndarray) -> float:
    X = readout.add_bias(features)
    target = tasks.narma_target(
        post_inputs,
        order=PROTOCOL.narma_order,
        input_scale=PROTOCOL.narma_input_scale,
    )
    train, test = _masks(target)
    weights = readout.train_readout(
        X[train],
        target[train],
        ridge=PROTOCOL.ridge,
    )
    return readout.nmse(
        target[test],
        readout.predict(X[test], weights),
    )


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = 0.5 * ((a - b) + (a - b).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta))))


def max_pairwise_trace_distance(states: Iterable[np.ndarray]) -> float:
    state_list = list(states)
    return max(
        trace_distance(state_list[i], state_list[j])
        for i, j in combinations(range(len(state_list)), 2)
    )


def absolute_summary(values: np.ndarray) -> dict:
    n = len(values)
    standard_error = float(values.std(ddof=1) / math.sqrt(n))
    quantile = float(student_t.ppf(0.975, n - 1))
    mean = float(values.mean())
    return {
        "n": n,
        "mean": mean,
        "standard_deviation": float(values.std(ddof=1)),
        "standard_error": standard_error,
        "ci95": [
            mean - quantile * standard_error,
            mean + quantile * standard_error,
        ],
    }


def paired_summary(
    focal: np.ndarray,
    reference: np.ndarray,
    orientation: str,
) -> dict:
    difference = focal - reference
    n = len(difference)
    standard_error = float(difference.std(ddof=1) / math.sqrt(n))
    quantile = float(student_t.ppf(0.975, n - 1))
    mean = float(difference.mean())
    wins = int(np.sum(difference > 0.0))
    ties = int(np.sum(difference == 0.0))
    sign_n = n - ties
    return {
        "n": n,
        "orientation": orientation,
        "mean_difference": mean,
        "standard_deviation": float(difference.std(ddof=1)),
        "standard_error": standard_error,
        "ci95": [
            mean - quantile * standard_error,
            mean + quantile * standard_error,
        ],
        "wins": wins,
        "ties": ties,
        "paired_t_p_two_sided": float(ttest_rel(focal, reference).pvalue),
        "sign_test_p_two_sided": (
            float(binomtest(wins, sign_n, 0.5).pvalue)
            if sign_n
            else 1.0
        ),
    }


def _score_task(
    task_name: str,
    feature_map: dict[str, np.ndarray],
    post_inputs: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    scores = {}
    lag_map = {}
    for state_name, features in feature_map.items():
        if task_name == "stm":
            score, lags = evaluate_stm(features, post_inputs)
            scores[state_name] = score
            lag_map[state_name] = lags
        elif task_name == "narma10":
            scores[state_name] = evaluate_narma(features, post_inputs)
        else:
            raise ValueError(task_name)
    return scores, lag_map


def run(out_dir: Path, initial_state_audit: bool = True) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"{out_dir} is not empty; choose a new --out directory"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    record = protocol_record()
    _write_json(out_dir / "protocol.json", record)
    _write_json(
        out_dir / "ordered_seeds.json",
        {
            "master_seed": PROTOCOL.master_seed,
            "seeds": record["ordered_seeds"],
        },
    )

    observables = readout.pauli_observables(
        PROTOCOL.n_qubits,
        max_weight=2,
    )
    observable_matrices = np.stack([observable.matrix for observable in observables])
    if len(observables) != 45:
        raise RuntimeError(f"expected 45 Pauli features, found {len(observables)}")

    score_rows: list[dict] = []
    lag_rows: list[dict] = []
    audit_rows: list[dict] = []
    lag_values = {
        model: np.empty((PROTOCOL.n_pairs, len(PROTOCOL.stm_delays)))
        for model in MODELS
    }
    t0 = time.time()

    for pair, seed_value in enumerate(ordered_seeds()):
        seed = int(seed_value)
        rng = np.random.default_rng(seed)
        J = reservoirs.random_couplings(
            PROTOCOL.n_qubits,
            PROTOCOL.coupling_scale,
            rng,
        )
        task_inputs = {
            "stm": tasks.stm_inputs(PROTOCOL.total_len, rng),
            "narma10": tasks.narma_inputs(PROTOCOL.total_len, rng),
        }
        rho0_map = initial_states(seed)
        pair_scores: dict[str, dict[str, float]] = {
            task_name: {}
            for task_name in task_inputs
        }

        for model in MODELS:
            reservoir = reservoirs.dissipative_input_reset(
                PROTOCOL.n_qubits,
                J,
                PROTOCOL.h,
                PROTOCOL.gamma,
                PROTOCOL.dt,
                jump_family=model,
            )
            for task_name, inputs in task_inputs.items():
                post_inputs = inputs[PROTOCOL.washout :]
                if initial_state_audit:
                    (
                        feature_map,
                        final_states,
                        max_feature_distance,
                        trace_distance_at_washout,
                    ) = (
                        run_initial_state_ensemble(
                            reservoir,
                            inputs,
                            observable_matrices,
                            rho0_map,
                        )
                    )
                else:
                    feature_map = {
                        "ground": reservoir.run(
                            inputs,
                            observables,
                            washout=PROTOCOL.washout,
                        )
                    }
                    final_states = {}
                    max_feature_distance = float("nan")
                    trace_distance_at_washout = float("nan")

                state_scores, lag_map = _score_task(
                    task_name,
                    feature_map,
                    post_inputs,
                )
                pair_scores[task_name][model] = state_scores["ground"]

                if task_name == "stm":
                    for delay_index, delay in enumerate(PROTOCOL.stm_delays):
                        lag_values[model][pair, delay_index] = (
                            lag_map["ground"][delay_index]
                        )
                        lag_rows.append(
                            {
                                "pair": pair,
                                "seed": seed,
                                "delay": delay,
                                "model": model,
                                "capacity": float(
                                    lag_map["ground"][delay_index]
                                ),
                            }
                        )

                if initial_state_audit:
                    audit_rows.append(
                        {
                            "pair": pair,
                            "seed": seed,
                            "model": model,
                            "task": task_name,
                            **{
                                f"score_{state_name}": float(
                                    state_scores[state_name]
                                )
                                for state_name in INITIAL_STATES
                            },
                            "max_score_spread": float(
                                np.ptp(
                                    [
                                        state_scores[state_name]
                                        for state_name in INITIAL_STATES
                                    ]
                                )
                            ),
                            "max_post_washout_feature_distance": (
                                max_feature_distance
                            ),
                            "max_trace_distance_after_800_inputs": (
                                trace_distance_at_washout
                            ),
                            "final_max_trace_distance": (
                                max_pairwise_trace_distance(
                                    final_states.values()
                                )
                            ),
                        }
                    )

        score_rows.append(
            {
                "pair": pair,
                "seed": seed,
                "stm_local": pair_scores["stm"]["local"],
                "stm_collective": pair_scores["stm"]["collective"],
                "stm_collective_minus_local": (
                    pair_scores["stm"]["collective"]
                    - pair_scores["stm"]["local"]
                ),
                "narma10_nmse_local": pair_scores["narma10"]["local"],
                "narma10_nmse_collective": (
                    pair_scores["narma10"]["collective"]
                ),
                "narma10_local_minus_collective": (
                    pair_scores["narma10"]["local"]
                    - pair_scores["narma10"]["collective"]
                ),
            }
        )
        print(
            f"{pair + 1:02d}/{PROTOCOL.n_pairs} seed={seed} "
            f"STM L/C={pair_scores['stm']['local']:.6f}/"
            f"{pair_scores['stm']['collective']:.6f} "
            f"NARMA L/C={pair_scores['narma10']['local']:.6f}/"
            f"{pair_scores['narma10']['collective']:.6f} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    _write_csv(out_dir / "strict_washout_scores.csv", score_rows)
    _write_csv(out_dir / "strict_washout_lag_capacities.csv", lag_rows)
    if initial_state_audit:
        _write_csv(out_dir / "initial_state_audit.csv", audit_rows)

    stm_local = np.asarray([row["stm_local"] for row in score_rows])
    stm_collective = np.asarray(
        [row["stm_collective"] for row in score_rows]
    )
    narma_local = np.asarray(
        [row["narma10_nmse_local"] for row in score_rows]
    )
    narma_collective = np.asarray(
        [row["narma10_nmse_collective"] for row in score_rows]
    )
    np.savez_compressed(
        out_dir / "strict_washout_arrays.npz",
        seeds=ordered_seeds(),
        delays=np.asarray(PROTOCOL.stm_delays),
        stm_local=stm_local,
        stm_collective=stm_collective,
        narma_local=narma_local,
        narma_collective=narma_collective,
        lag_local=lag_values["local"],
        lag_collective=lag_values["collective"],
    )
    summary = {
        "protocol_file": "protocol.json",
        "score_file": "strict_washout_scores.csv",
        "lag_capacity_file": "strict_washout_lag_capacities.csv",
        "array_file": "strict_washout_arrays.npz",
        "runtime_seconds": time.time() - t0,
        "stm_absolute": {
            "local": absolute_summary(stm_local),
            "collective": absolute_summary(stm_collective),
        },
        "narma10_nmse_absolute": {
            "local": absolute_summary(narma_local),
            "collective": absolute_summary(narma_collective),
        },
        "stm_collective_vs_local": paired_summary(
            stm_collective,
            stm_local,
            "collective STM minus local STM; positive values favour collective",
        ),
        "narma10_local_minus_collective": paired_summary(
            narma_local,
            narma_collective,
            (
                "local NARMA-10 NMSE minus collective NARMA-10 NMSE; "
                "positive values favour collective"
            ),
        ),
        "initial_state_audit": (
            {
                "file": "initial_state_audit.csv",
                "n_rows": len(audit_rows),
                "worst_max_score_spread": max(
                    row["max_score_spread"]
                    for row in audit_rows
                ),
                "worst_post_washout_feature_distance": max(
                    row["max_post_washout_feature_distance"]
                    for row in audit_rows
                ),
                "worst_trace_distance_after_800_inputs": max(
                    row["max_trace_distance_after_800_inputs"]
                    for row in audit_rows
                ),
                "worst_final_trace_distance": max(
                    row["final_max_trace_distance"]
                    for row in audit_rows
                ),
            }
            if initial_state_audit
            else {"status": "not run"}
        ),
    }
    _write_json(out_dir / "strict_washout_summary.json", summary)
    write_checksums(out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote verified result package to {out_dir}")


def write_checksums(out_dir: Path) -> None:
    paths = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in paths]
    (out_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def verify(out_dir: Path) -> bool:
    manifest = out_dir / "SHA256SUMS.txt"
    if not manifest.is_file():
        print(f"missing checksum manifest: {manifest}", file=sys.stderr)
        return False
    ok = True
    manifest_names: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not relative or Path(relative).name != relative:
            print(f"invalid manifest line: {line}", file=sys.stderr)
            ok = False
            continue
        if relative in manifest_names:
            print(f"duplicate manifest entry: {relative}", file=sys.stderr)
            ok = False
            continue
        manifest_names.add(relative)
        path = out_dir / relative
        if not path.is_file():
            print(f"MISSING  {relative}", file=sys.stderr)
            ok = False
            continue
        actual = _sha256(path)
        matched = actual == expected
        print(f"{'OK' if matched else 'FAILED'}  {relative}")
        ok &= matched
    actual_names = {
        path.name
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    unexpected = actual_names - manifest_names
    omitted = manifest_names - actual_names
    if unexpected:
        print(
            "UNEXPECTED files not covered by manifest: "
            + ", ".join(sorted(unexpected)),
            file=sys.stderr,
        )
        ok = False
    if omitted:
        print(
            "MISSING manifest files: " + ", ".join(sorted(omitted)),
            file=sys.stderr,
        )
        ok = False
    required = {
        "ordered_seeds.json",
        "protocol.json",
        "strict_washout_arrays.npz",
        "strict_washout_lag_capacities.csv",
        "strict_washout_scores.csv",
        "strict_washout_summary.json",
    }
    missing_required = required - actual_names
    if missing_required:
        print(
            "MISSING required artifacts: "
            + ", ".join(sorted(missing_required)),
            file=sys.stderr,
        )
        ok = False
    summary_path = out_dir / "strict_washout_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            audit_file = summary["initial_state_audit"].get("file")
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            print(f"invalid summary metadata: {exc}", file=sys.stderr)
            ok = False
        else:
            if audit_file and audit_file not in actual_names:
                print(
                    f"MISSING declared initial-state artifact: {audit_file}",
                    file=sys.stderr,
                )
                ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"new output directory (default: {DEFAULT_OUT})",
    )
    action.add_argument(
        "--verify",
        type=Path,
        metavar="DIR",
        help="verify an existing result directory and exit",
    )
    parser.add_argument(
        "--skip-initial-state-audit",
        action="store_true",
        help="run only the strict ground-state protocol",
    )
    args = parser.parse_args()
    if args.verify is not None:
        raise SystemExit(0 if verify(args.verify.resolve()) else 1)
    run(
        args.out.resolve(),
        initial_state_audit=not args.skip_initial_state_audit,
    )


if __name__ == "__main__":
    main()

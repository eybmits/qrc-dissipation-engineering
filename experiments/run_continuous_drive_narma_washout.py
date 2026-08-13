#!/usr/bin/env python3
"""Paired strict-washout confirmation of continuous-drive NARMA-10.

This additive experiment closes the washout-specific evidence gap in the
principal local-versus-collective NARMA-10 comparison.  It reuses all 32
canonical primary lineages and evaluates exactly the same 200-input washout,
600-input training block, 400-input untouched test block, inputs, targets,
readout, and ridge rule as the principal protocol.  The strict condition adds
600 independently generated warm-up inputs *before* that unchanged canonical
1200-input block, so scoring after 800 total warm-up inputs begins at exactly
the same canonical input and target row as scoring after the original
200-input washout.

For both local and equal-phase collective relaxation, both washouts are scored
from the ground state on all 32 primary lineages.  The first eight ordered
lineages additionally repeat both washouts from the fully excited, maximally
mixed, and Haar-random initial states.
The canonical ground-state washout-200 scores must reproduce the sealed primary
archive before any aggregate is accepted.  The unique primary estimand is
local-minus-collective NARMA-10 NMSE after the strict 800-input washout.

The run is fixed at N=5, h=dt=0.5, gamma=1, B=80, 45 Pauli features plus bias,
ridge 1e-8, and 32 paired primary seeds.  No optional stopping, seed
replacement, or task-dependent tuning is permitted.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import _paths
import numpy as np
import scipy
from scipy.sparse.linalg import expm_multiply
from scipy.stats import binomtest
from scipy.stats import t as student_t
from scipy.stats import ttest_rel

from qrc import dissipators, readout, reservoirs, tasks
from qrc.liouvillian import unvec, vec
from qrc.sparse_evolve import SparseLindbladReservoir


@dataclass(frozen=True)
class Protocol:
    n_qubits: int = 5
    coupling_scale: float = 1.0
    h: float = 0.5
    dt: float = 0.5
    gamma: float = 1.0
    primary_washout: int = 200
    strict_washout: int = 800
    train_len: int = 600
    test_len: int = 400
    narma_order: int = 10
    narma_input_scale: float = 0.2
    ridge: float = 1e-8
    n_pairs: int = 32
    initial_state_audit_pairs: int = 8
    prefix_namespace: int = 2026081301
    haar_seed_xor: int = 0x2468ACE0
    baseline_tolerance: float = 1e-12

    @property
    def canonical_len(self) -> int:
        return self.primary_washout + self.train_len + self.test_len

    @property
    def strict_prefix_len(self) -> int:
        return self.strict_washout - self.primary_washout


PROTOCOL = Protocol()
MODELS = ("local", "collective")
INITIAL_STATES = ("ground", "excited", "mixed", "haar")
WASHOUTS = (PROTOCOL.primary_washout, PROTOCOL.strict_washout)
ROOT = Path(_paths.ROOT)
DEFAULT_OUT = ROOT / "results" / "continuous_drive_narma_washout_v1"
PRIMARY_SNAPSHOT = ROOT / "paper" / "data" / "experiment1_principal_summary.json"
SOURCE_FILES = (
    "experiments/_paths.py",
    "experiments/run_continuous_drive_narma_washout.py",
    "experiments/run_final_scaling.py",
    "src/qrc/__init__.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_metadata() -> dict:
    def run(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                args,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def load_primary_snapshot() -> dict:
    snapshot = json.loads(PRIMARY_SNAPSHOT.read_text(encoding="utf-8"))
    if snapshot.get("artifact_type") != "figure3_principal_absolute_summary":
        raise ValueError("unexpected principal-summary artifact type")
    if snapshot.get("status") != "complete" or snapshot.get("n_pairs") != 32:
        raise ValueError("principal summary is not the complete 32-pair record")
    seeds = [int(seed) for seed in snapshot.get("seeds", [])]
    from run_final_scaling import deterministic_seeds

    if seeds != sorted(deterministic_seeds(32)):
        raise ValueError("principal seed ledger differs from canonical driver")
    for model in MODELS:
        values = snapshot["values"]["narma10"][model]
        if len(values) != 32 or not np.isfinite(values).all():
            raise ValueError(f"invalid principal NARMA values for {model}")
    return snapshot


def protocol_payload() -> dict:
    snapshot = load_primary_snapshot()
    local = dissipators.local_loss(PROTOCOL.n_qubits, PROTOCOL.gamma)
    collective = dissipators.collective_loss(PROTOCOL.n_qubits, PROTOCOL.gamma)
    payload = {
        "artifact_type": "continuous_drive_narma_washout_protocol",
        "version": 2,
        "status": "frozen_before_scoring",
        "serialization_amendment": {
            "predecessor_protocol_sha256": (
                "8b30fa20e5fec3f46a0e08f0cde282044497d795528a545cdc571a495e02a881"
            ),
            "reason": (
                "The predecessor stopped at the first non-audited lineage "
                "because unavailable diagnostics were represented as NaN, "
                "which the strict JSON writer rejects."
            ),
            "state_when_identified": (
                "The eight prespecified initial-state-audit checkpoints had "
                "completed; no non-audit checkpoint had been written."
            ),
            "scientific_change": (
                "None: seeds, conditions, inputs, tasks, estimands, inference, "
                "and acceptance criteria are unchanged; unavailable "
                "diagnostics are serialized as null."
            ),
        },
        "protocol": {
            **asdict(PROTOCOL),
            "canonical_len": PROTOCOL.canonical_len,
            "strict_prefix_len": PROTOCOL.strict_prefix_len,
            "models": list(MODELS),
            "initial_states": list(INITIAL_STATES),
            "initial_state_audit_seed_rule": (
                "the first eight ordered primary lineages; ground-state scoring "
                "uses all 32 lineages"
            ),
            "washouts": list(WASHOUTS),
            "readout": {
                "features": "45 one-body and same-axis two-body Pauli expectations",
                "bias": True,
                "ridge": PROTOCOL.ridge,
            },
            "pairing": (
                "Within every seed, local and collective conditions share J, the "
                "strict prefix, canonical NARMA input, target, split, readout, "
                "and initial state."
            ),
            "strict_input_rule": (
                "Prepend 600 inputs drawn from a separate frozen namespace to "
                "the unchanged canonical 1200-input task sequence.  After an "
                "800-input total washout, the 600 training and 400 test rows are "
                "therefore identical to those after the canonical 200-input "
                "washout."
            ),
            "primary_estimand": (
                "ground-state local NARMA-10 NMSE minus collective NARMA-10 "
                "NMSE after washout 800"
            ),
            "secondary_estimands": [
                "the same favorable contrast after washout 200",
                "paired change of the favorable contrast from washout 200 to 800",
                "cross-initial-state score spreads at both washouts",
            ],
            "inference": (
                "paired two-sided 95% Student-t intervals, paired t tests, exact "
                "two-sided sign tests, and paired wins; no optional stopping"
            ),
            "structural_budget": {
                "definition": "sum_k rate_k Tr(L_k^dagger L_k)",
                "local": dissipators.jump_strength(local),
                "collective": dissipators.jump_strength(collective),
            },
        },
        "ordered_seeds": [int(seed) for seed in snapshot["seeds"]],
        "principal_snapshot": {
            "path": str(PRIMARY_SNAPSHOT.relative_to(ROOT)),
            "sha256": sha256_path(PRIMARY_SNAPSHOT),
            "source_archive": snapshot["source_archive"],
            "source_archive_sha256": snapshot["source_archive_sha256"],
            "narma10_values": snapshot["values"]["narma10"],
        },
        "source_sha256": {
            relative: sha256_path(ROOT / relative) for relative in SOURCE_FILES
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "repository": git_metadata(),
    }
    payload["protocol_sha256"] = canonical_json_sha256(payload)
    return payload


def verify_protocol(protocol: dict, check_live_sources: bool = True) -> None:
    stored = protocol.get("protocol_sha256")
    copy = dict(protocol)
    copy.pop("protocol_sha256", None)
    if stored != canonical_json_sha256(copy):
        raise ValueError("protocol self-hash mismatch")
    if protocol.get("artifact_type") != "continuous_drive_narma_washout_protocol":
        raise ValueError("unexpected protocol type")
    if protocol.get("ordered_seeds") != load_primary_snapshot()["seeds"]:
        raise ValueError("protocol seed ledger differs from primary snapshot")
    expected_snapshot_hash = protocol["principal_snapshot"]["sha256"]
    if sha256_path(PRIMARY_SNAPSHOT) != expected_snapshot_hash:
        raise ValueError("principal snapshot changed after protocol freeze")
    if check_live_sources:
        for relative, expected in protocol["source_sha256"].items():
            if sha256_path(ROOT / relative) != expected:
                raise ValueError(f"frozen source changed: {relative}")


def initial_states(seed: int) -> dict[str, np.ndarray]:
    dimension = 2 ** PROTOCOL.n_qubits
    ground = np.zeros((dimension, dimension), dtype=complex)
    ground[0, 0] = 1.0
    excited = np.zeros((dimension, dimension), dtype=complex)
    excited[-1, -1] = 1.0
    mixed = np.eye(dimension, dtype=complex) / dimension
    rng = np.random.default_rng(int(seed) ^ PROTOCOL.haar_seed_xor)
    psi = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    psi /= np.linalg.norm(psi)
    return {
        "ground": ground,
        "excited": excited,
        "mixed": mixed,
        "haar": np.outer(psi, psi.conj()),
    }


def problem(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    couplings = reservoirs.random_couplings(
        PROTOCOL.n_qubits,
        PROTOCOL.coupling_scale,
        rng,
    )
    canonical_inputs = tasks.narma_inputs(PROTOCOL.canonical_len, rng)
    prefix_rng = np.random.default_rng(
        np.random.SeedSequence([PROTOCOL.prefix_namespace, int(seed)])
    )
    strict_prefix = tasks.narma_inputs(PROTOCOL.strict_prefix_len, prefix_rng)
    return couplings, canonical_inputs, strict_prefix


def build_reservoir(model: str, couplings: np.ndarray) -> SparseLindbladReservoir:
    n = PROTOCOL.n_qubits
    h = PROTOCOL.h
    h_static = reservoirs.ising_xx_hamiltonian(couplings, h, n)
    drive = reservoirs.transverse_drive(n)
    target = dissipators.jump_strength(dissipators.local_loss(n, PROTOCOL.gamma))
    if model == "local":
        raw = dissipators.local_loss(n, PROTOCOL.gamma)
    elif model == "collective":
        raw = dissipators.collective_loss(n, PROTOCOL.gamma)
    else:
        raise ValueError(model)
    jumps = dissipators.normalize_jump_strength(raw, target)
    return SparseLindbladReservoir.from_terms(
        n,
        h_static + h * drive,
        h * drive,
        jumps,
        PROTOCOL.dt,
    )


def trace_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = 0.5 * ((a - b) + (a - b).conj().T)
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta))))


def max_pairwise_trace_distance(states: Iterable[np.ndarray]) -> float:
    values = list(states)
    return max(
        trace_distance(values[i], values[j])
        for i, j in combinations(range(len(values)), 2)
    )


def run_ensemble(
    reservoir: SparseLindbladReservoir,
    inputs: np.ndarray,
    washout: int,
    rho0: dict[str, np.ndarray],
    observable_matrices: np.ndarray,
) -> tuple[dict[str, np.ndarray], float, float, float]:
    names = tuple(rho0)
    states = [rho0[name].copy() for name in names]
    feature_maps = {
        name: np.empty(
            (PROTOCOL.train_len + PROTOCOL.test_len, len(observable_matrices)),
            dtype=float,
        )
        for name in names
    }
    trace_at_washout = float("nan")
    worst_feature_distance = 0.0
    worst_trace_after_washout = 0.0
    for step, value in enumerate(inputs):
        state_vectors = np.column_stack([vec(state) for state in states])
        evolved = expm_multiply(
            reservoir.liouvillian(float(value)) * reservoir.dt,
            state_vectors,
        )
        states = [
            unvec(evolved[:, index], 2 ** PROTOCOL.n_qubits)
            for index in range(len(names))
        ]
        if step + 1 == washout:
            trace_at_washout = max_pairwise_trace_distance(states)
        if step < washout:
            continue
        row = step - washout
        state_stack = np.stack(states)
        feature_values = np.real(
            np.einsum("kij,sji->sk", observable_matrices, state_stack)
        )
        for index, name in enumerate(names):
            feature_maps[name][row] = feature_values[index]
        worst_feature_distance = max(
            worst_feature_distance,
            max(
                float(np.max(np.abs(feature_values[i] - feature_values[j])))
                for i, j in combinations(range(len(names)), 2)
            ),
        )
        worst_trace_after_washout = max(
            worst_trace_after_washout,
            max_pairwise_trace_distance(states),
        )
    return (
        feature_maps,
        trace_at_washout,
        worst_feature_distance,
        worst_trace_after_washout,
    )


def score_narma(features: np.ndarray, post_inputs: np.ndarray) -> float:
    target = tasks.narma_target(
        post_inputs,
        order=PROTOCOL.narma_order,
        input_scale=PROTOCOL.narma_input_scale,
    )
    train = np.zeros(len(target), dtype=bool)
    test = np.zeros(len(target), dtype=bool)
    train[: PROTOCOL.train_len] = True
    test[PROTOCOL.train_len : PROTOCOL.train_len + PROTOCOL.test_len] = True
    valid = np.isfinite(target)
    train &= valid
    test &= valid
    matrix = readout.add_bias(features)
    weights = readout.train_readout(
        matrix[train],
        target[train],
        ridge=PROTOCOL.ridge,
    )
    return float(
        readout.nmse(target[test], readout.predict(matrix[test], weights))
    )


def checkpoint_path(out_dir: Path, seed_index: int, seed: int) -> Path:
    return out_dir / "jobs" / f"pair_{seed_index:02d}_seed_{seed}.json"


def run_seed(
    seed_index: int,
    seed: int,
    protocol_path: str,
    out_dir_string: str,
) -> dict:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    verify_protocol(protocol, check_live_sources=True)
    out_dir = Path(out_dir_string)
    checkpoint = checkpoint_path(out_dir, seed_index, seed)
    if checkpoint.is_file():
        old = json.loads(checkpoint.read_text(encoding="utf-8"))
        stored = old.get("checkpoint_sha256")
        copy = dict(old)
        copy.pop("checkpoint_sha256", None)
        if stored == canonical_json_sha256(copy):
            return {"status": "skip", "path": str(checkpoint), "seed": seed}

    couplings, canonical_inputs, strict_prefix = problem(seed)
    strict_inputs = np.concatenate([strict_prefix, canonical_inputs])
    canonical_post = canonical_inputs[PROTOCOL.primary_washout :]
    if not np.array_equal(
        strict_inputs[PROTOCOL.strict_washout :], canonical_post
    ):
        raise RuntimeError("strict scored input block differs from canonical block")
    target = tasks.narma_target(
        canonical_post,
        order=PROTOCOL.narma_order,
        input_scale=PROTOCOL.narma_input_scale,
    )
    observables = readout.pauli_observables(PROTOCOL.n_qubits, max_weight=2)
    observable_matrices = np.stack([item.matrix for item in observables])
    if observable_matrices.shape != (45, 32, 32):
        raise RuntimeError(f"unexpected observable shape {observable_matrices.shape}")
    rho0 = initial_states(seed)
    audited_states = (
        INITIAL_STATES
        if seed_index < PROTOCOL.initial_state_audit_pairs
        else ("ground",)
    )
    selected_rho0 = {state: rho0[state] for state in audited_states}
    results: dict[str, dict[str, dict]] = {}
    start = time.time()
    for washout in WASHOUTS:
        task_inputs = canonical_inputs if washout == 200 else strict_inputs
        results[str(washout)] = {}
        for model in MODELS:
            reservoir = build_reservoir(model, couplings)
            # Reproduce the sealed primary ground-state score with the same
            # single-RHS path. The multi-RHS evolution below is diagnostic:
            # Krylov norm estimation can change the final bits when the number
            # of right-hand sides changes, although the trajectories agree.
            ground_features = reservoir.run(
                task_inputs,
                observables,
                washout=washout,
                rho0=rho0["ground"],
            )
            ground_score = score_narma(ground_features, canonical_post)
            features = {"ground": ground_features}
            scores = {"ground": ground_score}
            trace_at = None
            feature_distance = None
            trace_after = None
            batch_ground_maximum_feature_difference = None
            batch_ground_score_difference = None
            audit_feature_sha256 = None
            if len(audited_states) > 1:
                (
                    audit_features,
                    trace_at,
                    feature_distance,
                    trace_after,
                ) = run_ensemble(
                    reservoir,
                    task_inputs,
                    washout,
                    selected_rho0,
                    observable_matrices,
                )
                audit_scores = {
                    state: score_narma(values, canonical_post)
                    for state, values in audit_features.items()
                }
                batch_ground_maximum_feature_difference = float(
                    np.max(np.abs(audit_features["ground"] - ground_features))
                )
                batch_ground_score_difference = float(
                    audit_scores["ground"] - ground_score
                )
                if batch_ground_maximum_feature_difference > 1e-12:
                    raise RuntimeError(
                        "single- and multi-RHS ground trajectories disagree "
                        f"by {batch_ground_maximum_feature_difference:.3e}"
                    )
                features.update(
                    {
                        state: values
                        for state, values in audit_features.items()
                        if state != "ground"
                    }
                )
                scores.update(
                    {
                        state: value
                        for state, value in audit_scores.items()
                        if state != "ground"
                    }
                )
                audit_feature_sha256 = {
                    state: array_sha256(values)
                    for state, values in audit_features.items()
                }
            results[str(washout)][model] = {
                "scores": scores,
                "maximum_score_spread": float(np.ptp(list(scores.values()))),
                "trace_distance_at_washout": trace_at,
                "maximum_post_washout_feature_distance": feature_distance,
                "maximum_post_washout_trace_distance": trace_after,
                "feature_sha256": {
                    state: array_sha256(values)
                    for state, values in features.items()
                },
                "multi_rhs_audit": {
                    "ground_maximum_feature_difference": (
                        batch_ground_maximum_feature_difference
                    ),
                    "ground_score_difference": batch_ground_score_difference,
                    "feature_sha256": audit_feature_sha256,
                },
            }
    baseline = protocol["principal_snapshot"]["narma10_values"]
    for model in MODELS:
        expected = float(baseline[model][seed_index])
        actual = float(results["200"][model]["scores"]["ground"])
        if abs(actual - expected) > PROTOCOL.baseline_tolerance:
            raise RuntimeError(
                f"seed {seed} {model} failed canonical baseline replay: "
                f"{actual} vs {expected}"
            )
    payload = {
        "artifact_type": "continuous_drive_narma_washout_pair",
        "protocol_sha256": protocol["protocol_sha256"],
        "seed_index": seed_index,
        "seed": seed,
        "pairing_sha256": {
            "couplings": array_sha256(couplings),
            "canonical_inputs": array_sha256(canonical_inputs),
            "strict_prefix": array_sha256(strict_prefix),
            "scored_inputs": array_sha256(canonical_post),
            "target": array_sha256(target),
        },
        "results": results,
        "baseline_replay_maximum_absolute_error": max(
            abs(
                float(results["200"][model]["scores"]["ground"])
                - float(baseline[model][seed_index])
            )
            for model in MODELS
        ),
        "runtime_seconds": time.time() - start,
        "audited_initial_states": list(audited_states),
    }
    payload["checkpoint_sha256"] = canonical_json_sha256(payload)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(f".tmp.{os.getpid()}")
    write_json(temporary, payload)
    os.replace(temporary, checkpoint)
    return {"status": "done", "path": str(checkpoint), "seed": seed}


def load_checkpoints(out_dir: Path, protocol: dict) -> list[dict]:
    rows = []
    seeds = protocol["ordered_seeds"]
    for seed_index, seed in enumerate(seeds):
        path = checkpoint_path(out_dir, seed_index, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        stored = row.get("checkpoint_sha256")
        copy = dict(row)
        copy.pop("checkpoint_sha256", None)
        if stored != canonical_json_sha256(copy):
            raise ValueError(f"checkpoint self-hash mismatch: {path.name}")
        if row.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError(f"checkpoint protocol mismatch: {path.name}")
        if row.get("seed_index") != seed_index or row.get("seed") != seed:
            raise ValueError(f"checkpoint seed mismatch: {path.name}")
        couplings, canonical_inputs, strict_prefix = problem(seed)
        post = canonical_inputs[PROTOCOL.primary_washout :]
        expected_pairing = {
            "couplings": array_sha256(couplings),
            "canonical_inputs": array_sha256(canonical_inputs),
            "strict_prefix": array_sha256(strict_prefix),
            "scored_inputs": array_sha256(post),
            "target": array_sha256(
                tasks.narma_target(
                    post,
                    order=PROTOCOL.narma_order,
                    input_scale=PROTOCOL.narma_input_scale,
                )
            ),
        }
        if row.get("pairing_sha256") != expected_pairing:
            raise ValueError(f"checkpoint pairing mismatch: {path.name}")
        baseline = protocol["principal_snapshot"]["narma10_values"]
        for model in MODELS:
            actual = row["results"]["200"][model]["scores"]["ground"]
            expected = baseline[model][seed_index]
            if abs(actual - expected) > PROTOCOL.baseline_tolerance:
                raise ValueError(f"baseline replay mismatch: {path.name} {model}")
        expected_states = (
            list(INITIAL_STATES)
            if seed_index < PROTOCOL.initial_state_audit_pairs
            else ["ground"]
        )
        if row.get("audited_initial_states") != expected_states:
            raise ValueError(f"initial-state audit membership mismatch: {path.name}")
        rows.append(row)
    return rows


def paired_summary(focal: np.ndarray, reference: np.ndarray, orientation: str) -> dict:
    difference = np.asarray(focal, float) - np.asarray(reference, float)
    n = len(difference)
    standard_error = float(difference.std(ddof=1) / math.sqrt(n))
    quantile = float(student_t.ppf(0.975, n - 1))
    mean = float(difference.mean())
    wins = int(np.sum(difference > 0.0))
    ties = int(np.sum(difference == 0.0))
    return {
        "n": n,
        "orientation": orientation,
        "values": difference.tolist(),
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
            float(binomtest(wins, n - ties, 0.5).pvalue) if n > ties else 1.0
        ),
    }


def absolute_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
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


def build_aggregate(out_dir: Path, protocol: dict, rows: list[dict]) -> dict:
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for washout in WASHOUTS:
        scores[str(washout)] = {}
        for model in MODELS:
            scores[str(washout)][model] = {}
            for state in INITIAL_STATES:
                scores[str(washout)][model][state] = np.asarray(
                    [
                        row["results"][str(washout)][model]["scores"][state]
                        for row in rows
                        if state in row["results"][str(washout)][model]["scores"]
                    ],
                    dtype=float,
                )
    effects = {}
    for washout in WASHOUTS:
        key = str(washout)
        effects[key] = {}
        for state in INITIAL_STATES:
            effects[key][state] = paired_summary(
                scores[key]["local"][state],
                scores[key]["collective"][state],
                "local NARMA-10 NMSE minus collective NARMA-10 NMSE",
            )
    ground_change = paired_summary(
        np.asarray(effects["800"]["ground"]["values"]),
        np.asarray(effects["200"]["ground"]["values"]),
        "strict-washout favorable effect minus primary-washout favorable effect",
    )
    summary_rows = []
    for washout in WASHOUTS:
        key = str(washout)
        for model in MODELS:
            for state in INITIAL_STATES:
                if not len(scores[key][model][state]):
                    continue
                item = absolute_summary(scores[key][model][state])
                summary_rows.append(
                    {
                        "washout": washout,
                        "model": model,
                        "initial_state": state,
                        "mean_nmse": item["mean"],
                        "standard_error": item["standard_error"],
                        "ci95_low": item["ci95"][0],
                        "ci95_high": item["ci95"][1],
                    }
                )
    per_pair_rows = []
    for index, row in enumerate(rows):
        per_pair_rows.append(
            {
                "seed_index": index,
                "seed": row["seed"],
                "local_w200": scores["200"]["local"]["ground"][index],
                "collective_w200": scores["200"]["collective"]["ground"][index],
                "effect_w200": effects["200"]["ground"]["values"][index],
                "local_w800": scores["800"]["local"]["ground"][index],
                "collective_w800": scores["800"]["collective"]["ground"][index],
                "effect_w800": effects["800"]["ground"]["values"][index],
                "effect_change_w800_minus_w200": ground_change["values"][index],
            }
        )
    convergence = {}
    for washout in WASHOUTS:
        key = str(washout)
        convergence[key] = {}
        for model in MODELS:
            cells = [
                row["results"][key][model]
                for row in rows[: PROTOCOL.initial_state_audit_pairs]
            ]
            convergence[key][model] = {
                "maximum_cross_initialization_score_spread": max(
                    cell["maximum_score_spread"] for cell in cells
                ),
                "mean_cross_initialization_score_spread": float(
                    np.mean([cell["maximum_score_spread"] for cell in cells])
                ),
                "maximum_trace_distance_at_washout": max(
                    cell["trace_distance_at_washout"] for cell in cells
                ),
                "maximum_post_washout_feature_distance": max(
                    cell["maximum_post_washout_feature_distance"] for cell in cells
                ),
                "maximum_post_washout_trace_distance": max(
                    cell["maximum_post_washout_trace_distance"] for cell in cells
                ),
            }
    aggregate = {
        "artifact_type": "continuous_drive_narma_washout_aggregate",
        "version": 1,
        "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "n_pairs": PROTOCOL.n_pairs,
        "ordered_seeds": protocol["ordered_seeds"],
        "baseline_replay_maximum_absolute_error": max(
            row["baseline_replay_maximum_absolute_error"] for row in rows
        ),
        "absolute_scores": {
            key: {
                model: {
                    state: absolute_summary(values)
                    for state, values in state_values.items()
                }
                for model, state_values in model_values.items()
            }
            for key, model_values in scores.items()
        },
        "favorable_effects": effects,
        "ground_effect_change_w800_minus_w200": ground_change,
        "initial_state_audit": convergence,
        "checkpoint_sha256": {
            checkpoint_path(out_dir, index, seed).name: row["checkpoint_sha256"]
            for index, (seed, row) in enumerate(zip(protocol["ordered_seeds"], rows, strict=True))
        },
        "claim_boundary": (
            "This experiment certifies the continuous-drive local-versus-collective "
            "NARMA-10 ordering under the tested strict washout.  It does not "
            "certify every jump-family ranking, encoding, task, system size, or "
            "operating point."
        ),
    }
    aggregate["aggregate_sha256"] = canonical_json_sha256(aggregate)
    write_csv(out_dir / "per_pair_scores.csv", per_pair_rows)
    write_csv(out_dir / "absolute_summary.csv", summary_rows)
    write_json(out_dir / "aggregate.json", aggregate)
    return aggregate


def write_checksums(out_dir: Path) -> None:
    paths = sorted(
        path for path in out_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    lines = [
        f"{sha256_path(path)}  {path.relative_to(out_dir).as_posix()}"
        for path in paths
    ]
    (out_dir / "SHA256SUMS.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def validate(out_dir: Path, check_live_sources: bool = True) -> dict:
    protocol_path = out_dir / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol, check_live_sources=check_live_sources)
    rows = load_checkpoints(out_dir, protocol)
    rebuilt = build_aggregate(out_dir, protocol, rows)
    aggregate_path = out_dir / "aggregate.json"
    stored = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if rebuilt != stored:
        raise ValueError("aggregate reconstruction mismatch")
    required = {
        "protocol.json",
        "aggregate.json",
        "per_pair_scores.csv",
        "absolute_summary.csv",
    }
    actual = {path.name for path in out_dir.iterdir() if path.is_file()}
    if not required <= actual:
        raise ValueError(f"missing required files: {sorted(required - actual)}")
    report = {
        "artifact_type": "continuous_drive_narma_washout_validation",
        "status": "pass",
        "protocol_sha256": protocol["protocol_sha256"],
        "aggregate_sha256": stored["aggregate_sha256"],
        "checkpoint_count": len(rows),
        "primary_effect": stored["favorable_effects"]["800"]["ground"],
        "effect_change": stored["ground_effect_change_w800_minus_w200"],
        "baseline_replay_maximum_absolute_error": stored[
            "baseline_replay_maximum_absolute_error"
        ],
        "initial_state_audit": stored["initial_state_audit"],
    }
    write_json(out_dir / "validation_report.json", report)
    write_checksums(out_dir)
    return report


def verify_checksums(out_dir: Path) -> None:
    manifest = out_dir / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    expected_names = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not relative:
            raise ValueError(f"malformed checksum row: {line}")
        path = out_dir / relative
        if not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"checksum mismatch: {relative}")
        expected_names.add(relative)
    actual_names = {
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    if expected_names != actual_names:
        raise ValueError("checksum membership mismatch")


def freeze(out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to freeze into nonempty {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jobs").mkdir()
    write_json(out_dir / "protocol.json", protocol_payload())
    print(f"Frozen protocol at {out_dir / 'protocol.json'}")


def run_all(out_dir: Path, workers: int) -> None:
    protocol_path = out_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    verify_protocol(protocol, check_live_sources=True)
    futures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, seed in enumerate(protocol["ordered_seeds"]):
            futures.append(
                pool.submit(
                    run_seed,
                    index,
                    int(seed),
                    str(protocol_path),
                    str(out_dir),
                )
            )
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            print(
                f"{completed:02d}/{len(futures)} seed={result['seed']} "
                f"status={result['status']}",
                flush=True,
            )


def smoke() -> None:
    temporary = ROOT / "results" / ".continuous_drive_narma_washout_smoke"
    if temporary.exists():
        shutil.rmtree(temporary)
    freeze(temporary)
    protocol_path = temporary / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    seed = int(protocol["ordered_seeds"][0])
    run_seed(0, seed, str(protocol_path), str(temporary))
    row = json.loads(checkpoint_path(temporary, 0, seed).read_text(encoding="utf-8"))
    if row["baseline_replay_maximum_absolute_error"] > PROTOCOL.baseline_tolerance:
        raise RuntimeError("smoke baseline replay failed")
    print(
        "Smoke PASS: maximum primary-score replay error "
        f"{row['baseline_replay_maximum_absolute_error']:.3e}"
    )
    shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("smoke", "freeze", "run", "aggregate", "verify", "all"),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.action == "smoke":
        smoke()
        return
    if args.action in {"freeze", "all"} and not args.out.exists():
        freeze(args.out)
    elif args.action == "freeze":
        freeze(args.out)
    if args.action in {"run", "all"}:
        run_all(args.out, args.workers)
    if args.action in {"aggregate", "all"}:
        protocol = json.loads((args.out / "protocol.json").read_text(encoding="utf-8"))
        verify_protocol(protocol, check_live_sources=True)
        rows = load_checkpoints(args.out, protocol)
        build_aggregate(args.out, protocol, rows)
        report = validate(args.out, check_live_sources=True)
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.action == "verify":
        verify_checksums(args.out)
        report = validate(args.out, check_live_sources=True)
        verify_checksums(args.out)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

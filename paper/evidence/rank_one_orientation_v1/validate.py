#!/usr/bin/env python3
"""Standalone validator for the sealed N=6 orientation intervention.

The validator reconstructs the stored aggregate statistics and tables from the
24 byte-preserved checkpoints.  It does not rerun the Lindblad trajectories;
the exact frozen source and environment required for that independent rerun are
included in the same directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr, t as student_t


ROOT = Path(__file__).resolve().parent
VERSION = "rank-one-orientation-v1-2026-08-12"
CONDITIONS = ("drive_orthogonal", "equal_phase")
SEEDS = (
    956087733,
    1375334633,
    707736772,
    1133846500,
    365211353,
    878523603,
    457552621,
    363662622,
    853972123,
    1403843447,
    151336801,
    1991628836,
    1627319819,
    336852480,
    1454963355,
    203675062,
    93339074,
    8147085,
    264759322,
    16866769,
    346211042,
    1665106229,
    1622806565,
    1222562911,
)
COEFFICIENTS = {
    "drive_orthogonal": np.asarray([1, 1, 1, -1, -1, -1], dtype=float),
    "equal_phase": np.ones(6, dtype=float),
}
EXPECTED_PROTOCOL_HASHES = {
    "5d4d8c53cea9dfabbc5a0416e19097ad63227a2ada59c34bc69fcf9b459bf7a4",
    "8fcbb1c2a6f22677b062cd36c5b24b5c8c0d7f098bc3adb6a67ef85771314d80",
}
EXPECTED_JUMP_HASHES = {
    "drive_orthogonal": "0b7c501fb88d59968692ac15a3680a182f6743cfa83064b7c5bce6637d851349",
    "equal_phase": "6aa82635a9e142fc2bee177a87deeeae4371d58a488e22af6410c08e4a6bf994",
}
METRICS = (
    "response_lag_centroid",
    "long_lag_energy_fraction",
    "feature_space_effective_rank",
    "leading_singular_energy_fraction",
)
TRACE_GATE = 1e-8
FEATURE_GATE = 2e-8
SEMANTIC_ATOL = 5e-12
CHECKSUM_NAME = "SHA256SUMS"
REPORT_NAME = "validation_report.json"


def canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_digest(array: np.ndarray) -> str:
    value = np.asarray(array)
    dtype = "<c16" if np.iscomplexobj(value) else "<f8"
    normalized = np.ascontiguousarray(value, dtype=dtype)
    payload = np.asarray(normalized.shape, dtype="<i8").tobytes()
    payload += normalized.tobytes()
    return sha256_bytes(payload)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def mean_ci(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array)))
    half = float(student_t.ppf(0.975, len(array) - 1) * standard_error)
    return {
        "n": len(array),
        "mean": mean,
        "standard_error": standard_error,
        "ci95": [mean - half, mean + half],
    }


def paired(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=float)
    out = mean_ci(array)
    wins = int(np.sum(array > 0))
    losses = int(np.sum(array < 0))
    ties = int(np.sum(array == 0))
    out.update(
        {
            "median": float(np.median(array)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "wins_positive": wins,
            "losses_negative": losses,
            "ties": ties,
            "exact_sign_test_p_two_sided": float(
                binomtest(wins, wins + losses, 0.5).pvalue
            )
            if wins + losses
            else 1.0,
            "cohens_dz": float(array.mean() / array.std(ddof=1)),
        }
    )
    return out


def same_semantics(expected: object, actual: object, path: str = "root") -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if expected != actual:
            raise ValueError(f"semantic mismatch at {path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(
            float(expected), float(actual), rel_tol=0.0, abs_tol=SEMANTIC_ATOL
        ):
            raise ValueError(f"numeric mismatch at {path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"list length mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            same_semantics(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(
                f"object keys mismatch at {path}: "
                f"missing={sorted(set(expected) - set(actual))}, "
                f"unexpected={sorted(set(actual) - set(expected))}"
            )
        for key in expected:
            same_semantics(expected[key], actual[key], f"{path}.{key}")
        return
    if expected != actual:
        raise ValueError(f"type/value mismatch at {path}: {expected!r} != {actual!r}")


def parse_ledger(root: Path) -> dict[str, str]:
    path = root / CHECKSUM_NAME
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("empty SHA256SUMS")
    records: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        pure = PurePosixPath(relative)
        if (
            separator != "  "
            or len(digest) != 64
            or digest.lower() != digest
            or pure.is_absolute()
            or ".." in pure.parts
            or relative != pure.as_posix()
            or relative == CHECKSUM_NAME
        ):
            raise ValueError(f"invalid checksum line: {line!r}")
        int(digest, 16)
        if relative in records:
            raise ValueError(f"duplicate checksum path: {relative}")
        records[relative] = digest
    if list(records) != sorted(records):
        raise ValueError("SHA256SUMS is not sorted")
    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_NAME
    }
    if set(records) != set(actual):
        raise ValueError(
            "checksum membership mismatch: "
            f"missing={sorted(set(records) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(records))}"
        )
    for relative, expected in records.items():
        observed = sha256_file(actual[relative])
        if observed != expected:
            raise ValueError(
                f"checksum mismatch for {relative}: {observed} != {expected}"
            )
    return records


def sminus(site: int, n_qubits: int) -> np.ndarray:
    identity = np.eye(2, dtype=complex)
    lowering = np.asarray([[0, 1], [0, 0]], dtype=complex)
    factors = [identity] * n_qubits
    factors[site] = lowering
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def channel_invariants() -> dict[str, dict]:
    uniform = np.ones(6, dtype=float)
    result = {}
    for condition, coefficients in COEFFICIENTS.items():
        gram = np.outer(coefficients, coefficients)
        coefficient_norm_squared = float(coefficients @ coefficients)
        jump = sum(
            (coefficient * sminus(site, 6) for site, coefficient in enumerate(coefficients)),
            np.zeros((64, 64), dtype=complex),
        )
        result[condition] = {
            "coefficient_norm_squared": coefficient_norm_squared,
            "coefficient_magnitudes": np.abs(coefficients).tolist(),
            # For C=c c^T these invariants are analytical.  Do not serialize
            # backend-dependent roundoff from numerical null eigenvalues.
            "kossakowski_rank": 1,
            "kossakowski_nonzero_eigenvalues": [coefficient_norm_squared],
            "kossakowski_trace": float(np.trace(gram)),
            "kossakowski_diagonal": np.diag(gram).tolist(),
            "kossakowski_kernel_dimension": 5,
            "physical_jump_kernel_dimension": int(64 - np.linalg.matrix_rank(jump)),
            "operator_weight_budget_B": float(
                np.trace(jump.conjugate().T @ jump).real
            ),
            "normalized_squared_overlap_with_uniform_drive": float(
                abs(uniform @ coefficients) ** 2
                / ((uniform @ uniform) * (coefficients @ coefficients))
            ),
            "jump_sha256": array_digest(jump),
        }
    return result


def stream_hashes(seed: int) -> dict[str, str]:
    coupling_stream, wash_stream, task_stream, _ = np.random.SeedSequence(
        [2026081201, seed]
    ).spawn(4)
    coupling_rng = np.random.default_rng(coupling_stream)
    wash_rng = np.random.default_rng(wash_stream)
    task_rng = np.random.default_rng(task_stream)
    couplings = np.zeros((6, 6), dtype=float)
    upper = np.triu_indices(6, 1)
    couplings[upper] = coupling_rng.uniform(-1, 1, len(upper[0]))
    couplings = math.sqrt(4.0 / 5.0) * (couplings + couplings.T)
    wash = wash_rng.uniform(0, 1, 1600)
    task = task_rng.uniform(0, 1, 1000)
    return {
        "coupling_sha256": array_digest(couplings),
        "wash_sha256": array_digest(wash),
        "task_sha256": array_digest(task),
    }


def validate_checkpoint(row: dict, index: int, invariants: dict) -> None:
    stored_payload = row.get("payload_sha256")
    unhashed = dict(row)
    unhashed.pop("payload_sha256", None)
    if stored_payload != sha256_bytes(canonical(unhashed)):
        raise ValueError(f"payload digest mismatch for seed index {index}")
    if (
        row.get("seed_index") != index
        or row.get("seed") != SEEDS[index]
        or row.get("version") != VERSION
        or row.get("protocol_sha256") not in EXPECTED_PROTOCOL_HASHES
        or row.get("full_four_state_audit") is not (index < 6)
    ):
        raise ValueError(f"checkpoint identity mismatch for seed index {index}")
    for key, expected in stream_hashes(SEEDS[index]).items():
        if row.get(key) != expected:
            raise ValueError(f"random-stream mismatch for seed index {index}: {key}")
    if set(row.get("conditions", {})) != set(CONDITIONS):
        raise ValueError(f"condition mismatch for seed index {index}")
    if row.get("convergence", {}).get("both_conditions_passed") is not True:
        raise ValueError(f"convergence gate failed for seed index {index}")
    if row["convergence"].get("selected_common_washout") != 800:
        raise ValueError(f"unexpected washout for seed index {index}")
    for condition in CONDITIONS:
        metadata = row["reservoirs"][condition]
        if (
            not math.isclose(metadata["budget"], 192.0, rel_tol=0, abs_tol=1e-12)
            or metadata["jump_sha256"] != EXPECTED_JUMP_HASHES[condition]
            or metadata["jump_sha256"] != invariants[condition]["jump_sha256"]
        ):
            raise ValueError(f"dissipator mismatch: seed {index}, {condition}")
        audits = row["convergence"]["audits"][condition]
        if set(audits) != {"800"}:
            raise ValueError(f"convergence checkpoint mismatch: seed {index}, {condition}")
        audit = audits["800"]
        expected_pairs = 6 if index < 6 else 1
        if audit.get("passed") is not True or len(audit.get("pairwise", [])) != expected_pairs:
            raise ValueError(f"initial-state audit scope mismatch: seed {index}, {condition}")
        maximum_trace = max(item["trace_distance"] for item in audit["pairwise"])
        maximum_feature = max(item["max_feature_distance"] for item in audit["pairwise"])
        if (
            not math.isclose(maximum_trace, audit["maximum_trace_distance"], rel_tol=0, abs_tol=1e-20)
            or not math.isclose(maximum_feature, audit["maximum_feature_distance"], rel_tol=0, abs_tol=1e-20)
            or maximum_trace > TRACE_GATE
            or maximum_feature > FEATURE_GATE
        ):
            raise ValueError(f"convergence metric mismatch: seed {index}, {condition}")
        condition_row = row["conditions"][condition]
        capacities = condition_row["stm"]["capacity_by_delay"]
        if (
            len(capacities) != 20
            or any(value < 0 or value > 1 + 1e-12 for value in capacities)
            or not math.isclose(
                sum(capacities),
                condition_row["stm"]["total_capacity"],
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"STM reconstruction failed: seed {index}, {condition}")
        kernel = condition_row["kernel"]
        energy = np.asarray(kernel["normalized_lag_energy"], dtype=float)
        if (
            energy.shape != (20,)
            or np.any(energy < 0)
            or not math.isclose(float(energy.sum()), 1.0, rel_tol=0, abs_tol=1e-12)
            or not math.isclose(
                float(np.arange(1, 21) @ energy),
                kernel["response_lag_centroid"],
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(energy[9:].sum()),
                kernel["long_lag_energy_fraction"],
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"response summary reconstruction failed: seed {index}, {condition}")
    effect = (
        row["conditions"]["equal_phase"]["stm"]["total_capacity"]
        - row["conditions"]["drive_orthogonal"]["stm"]["total_capacity"]
    )
    if not math.isclose(
        effect, row["stm_equal_minus_orthogonal"], rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError(f"stored paired effect mismatch for seed index {index}")


def load_checkpoints(root: Path, invariants: dict) -> list[dict]:
    paths = sorted((root / "checkpoints").glob("seed_*.json"))
    expected_names = [f"seed_{index:02d}.json" for index in range(24)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("checkpoint membership/order mismatch")
    rows = []
    for index, path in enumerate(paths):
        row = load_json(path)
        validate_checkpoint(row, index, invariants)
        rows.append(row)
    return rows


def reconstruct_tables(rows: list[dict]) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    absolute = {
        condition: {"stm": [], **{metric: [] for metric in METRICS}}
        for condition in CONDITIONS
    }
    per_seed: list[dict] = []
    lag_long: list[dict] = []
    kernel_long: list[dict] = []
    worst_trace = {condition: 0.0 for condition in CONDITIONS}
    worst_feature = {condition: 0.0 for condition in CONDITIONS}
    for row in rows:
        selected = str(row["convergence"]["selected_common_washout"])
        record = {
            "seed_index": row["seed_index"],
            "seed": row["seed"],
            "protocol_sha256": row["protocol_sha256"],
            "washout": int(selected),
            "convergence_passed": row["convergence"]["both_conditions_passed"],
            "runtime_seconds": row["runtime_seconds"],
        }
        for condition in CONDITIONS:
            result = row["conditions"][condition]
            stm = result["stm"]["total_capacity"]
            audit = row["convergence"]["audits"][condition][selected]
            record[f"{condition}_stm"] = stm
            record[f"{condition}_max_trace_distance"] = audit[
                "maximum_trace_distance"
            ]
            record[f"{condition}_max_feature_distance"] = audit[
                "maximum_feature_distance"
            ]
            worst_trace[condition] = max(
                worst_trace[condition], audit["maximum_trace_distance"]
            )
            worst_feature[condition] = max(
                worst_feature[condition], audit["maximum_feature_distance"]
            )
            absolute[condition]["stm"].append(stm)
            for metric in METRICS:
                value = result["kernel"][metric]
                absolute[condition][metric].append(value)
                record[f"{condition}_{metric}"] = value
            for delay, capacity in enumerate(result["stm"]["capacity_by_delay"], 1):
                lag_long.append(
                    {
                        "seed_index": row["seed_index"],
                        "seed": row["seed"],
                        "condition": condition,
                        "delay": delay,
                        "capacity": capacity,
                    }
                )
            for lag, energy in enumerate(
                result["kernel"]["normalized_lag_energy"], 1
            ):
                kernel_long.append(
                    {
                        "seed_index": row["seed_index"],
                        "seed": row["seed"],
                        "condition": condition,
                        "lag": lag,
                        "normalized_energy": energy,
                    }
                )
        record["stm_equal_minus_orthogonal"] = (
            record["equal_phase_stm"] - record["drive_orthogonal_stm"]
        )
        for metric in METRICS:
            record[f"{metric}_equal_minus_orthogonal"] = (
                record[f"equal_phase_{metric}"]
                - record[f"drive_orthogonal_{metric}"]
            )
        per_seed.append(record)

    paired_metrics = {
        metric: paired(
            np.asarray(absolute["equal_phase"][metric])
            - np.asarray(absolute["drive_orthogonal"][metric])
        )
        for metric in ("stm",) + METRICS
    }
    associations = {}
    stm_change = np.asarray(
        [record["stm_equal_minus_orthogonal"] for record in per_seed]
    )
    for metric in METRICS:
        change = np.asarray(
            [record[f"{metric}_equal_minus_orthogonal"] for record in per_seed]
        )
        pearson = pearsonr(stm_change, change)
        spearman = spearmanr(stm_change, change)
        associations[f"stm_change_vs_{metric}_change"] = {
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
        }

    lag_summary = []
    equal_capacities = np.asarray(
        [row["conditions"]["equal_phase"]["stm"]["capacity_by_delay"] for row in rows]
    )
    orthogonal_capacities = np.asarray(
        [
            row["conditions"]["drive_orthogonal"]["stm"]["capacity_by_delay"]
            for row in rows
        ]
    )
    for index in range(20):
        equal = mean_ci(equal_capacities[:, index])
        orthogonal = mean_ci(orthogonal_capacities[:, index])
        difference_values = equal_capacities[:, index] - orthogonal_capacities[:, index]
        difference = mean_ci(difference_values)
        lag_summary.append(
            {
                "delay": index + 1,
                "equal_phase_mean": equal["mean"],
                "equal_phase_ci95_low": equal["ci95"][0],
                "equal_phase_ci95_high": equal["ci95"][1],
                "drive_orthogonal_mean": orthogonal["mean"],
                "drive_orthogonal_ci95_low": orthogonal["ci95"][0],
                "drive_orthogonal_ci95_high": orthogonal["ci95"][1],
                "paired_difference_mean": difference["mean"],
                "paired_difference_ci95_low": difference["ci95"][0],
                "paired_difference_ci95_high": difference["ci95"][1],
                "paired_wins": int(np.sum(difference_values > 0)),
            }
        )

    summary = {
        "experiment": {
            "name": "rank-one dissipator-orientation intervention",
            "version": VERSION,
            "n_qubits": 6,
            "pair_count": 24,
            "channels": {
                condition: COEFFICIENTS[condition].astype(int).tolist()
                for condition in CONDITIONS
            },
            "changed_coordinate": "orientation of the rank-one lowering direction relative to the uniform input-drive direction",
            "matched_invariants": [
                "one lowering jump",
                "Kossakowski rank 1",
                "Kossakowski nonzero spectrum",
                "trace / operator-weight budget B=192",
                "coefficient magnitudes |c_i|=1",
                "sitewise Kossakowski diagonal",
                "five-dimensional kernel of the lowering block",
                "Hamiltonian, input stream, readout, data split, and ridge protocol within each pair",
            ],
        },
        "stm": {
            "equal_phase": mean_ci(absolute["equal_phase"]["stm"]),
            "drive_orthogonal": mean_ci(absolute["drive_orthogonal"]["stm"]),
            "paired_equal_minus_orthogonal": paired_metrics["stm"],
            "relative_gain_ratio_of_means_percent": float(
                100
                * (
                    np.mean(absolute["equal_phase"]["stm"])
                    / np.mean(absolute["drive_orthogonal"]["stm"])
                    - 1
                )
            ),
            "lag_resolved": lag_summary,
        },
        "kernel": {
            metric: {
                "equal_phase": mean_ci(absolute["equal_phase"][metric]),
                "drive_orthogonal": mean_ci(
                    absolute["drive_orthogonal"][metric]
                ),
                "paired_equal_minus_orthogonal": paired_metrics[metric],
            }
            for metric in METRICS
        },
        "association": associations,
        "validation": {
            "all_24_seed_jobs_present": True,
            "all_convergence_gates_passed": True,
            "all_selected_washouts": sorted(
                {record["washout"] for record in per_seed}
            ),
            "raw_payload_digests_verified": True,
            "protocol_hash_variants": sorted(
                {record["protocol_sha256"] for record in per_seed}
            ),
            "protocol_hash_note": "Two hashes occurred because the frozen protocol hash included raw floating-point null eigenvalues from np.linalg.eigvalsh; these vary at roundoff level across runner BLAS backends. Semantic invariants, jump hashes, budgets, version, seed identities, and every payload digest were independently verified.",
            "jump_sha256": EXPECTED_JUMP_HASHES,
            "worst_trace_distance": worst_trace,
            "worst_feature_distance": worst_feature,
        },
    }
    return summary, per_seed, lag_long, lag_summary, kernel_long


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_csv(expected: list[dict], path: Path) -> None:
    actual = read_csv(path)
    if len(expected) != len(actual):
        raise ValueError(f"CSV row-count mismatch: {path}")
    if expected and list(expected[0]) != list(actual[0]):
        raise ValueError(f"CSV column/order mismatch: {path}")
    for row_index, (left, right) in enumerate(zip(expected, actual)):
        for key, expected_value in left.items():
            actual_value = right[key]
            if isinstance(expected_value, bool):
                if actual_value != str(expected_value):
                    raise ValueError(f"CSV mismatch {path}:{row_index}:{key}")
            elif isinstance(expected_value, int):
                if int(actual_value) != expected_value:
                    raise ValueError(f"CSV mismatch {path}:{row_index}:{key}")
            elif isinstance(expected_value, float):
                if not math.isclose(
                    float(actual_value), expected_value, rel_tol=0, abs_tol=SEMANTIC_ATOL
                ):
                    raise ValueError(f"CSV mismatch {path}:{row_index}:{key}")
            elif actual_value != expected_value:
                raise ValueError(f"CSV mismatch {path}:{row_index}:{key}")


def build_report(root: Path) -> dict:
    protocol = load_json(root / "protocol.json")
    provenance = load_json(root / "provenance.json")
    environment = load_json(root / "environment.json")
    if protocol.get("input_and_randomization", {}).get("seeds_in_order") != list(SEEDS):
        raise ValueError("stable protocol seed ledger mismatch")
    if protocol.get("experiment_version") != VERSION:
        raise ValueError("stable protocol version mismatch")
    invariants = channel_invariants()
    for condition in CONDITIONS:
        observed = invariants[condition]
        if (
            observed["kossakowski_rank"] != 1
            or observed["kossakowski_kernel_dimension"] != 5
            or observed["physical_jump_kernel_dimension"] != 20
            or observed["kossakowski_nonzero_eigenvalues"] != [6.0]
            or observed["coefficient_magnitudes"] != [1.0] * 6
            or observed["kossakowski_diagonal"] != [1.0] * 6
            or not math.isclose(
                observed["operator_weight_budget_B"], 192.0, rel_tol=0, abs_tol=1e-12
            )
            or observed["jump_sha256"] != EXPECTED_JUMP_HASHES[condition]
        ):
            raise ValueError(f"stable channel invariant failed: {condition}")
    rows = load_checkpoints(root, invariants)
    if {row["protocol_sha256"] for row in rows} != EXPECTED_PROTOCOL_HASHES:
        raise ValueError("legacy protocol hash variants mismatch")
    summary, per_seed, lag_long, lag_summary, kernel_long = reconstruct_tables(rows)
    same_semantics(summary, load_json(root / "derived" / "summary.json"), "summary")
    compare_csv(per_seed, root / "derived" / "per_seed.csv")
    compare_csv(lag_long, root / "derived" / "lag_capacities_long.csv")
    compare_csv(lag_summary, root / "derived" / "lag_capacities_summary.csv")
    compare_csv(kernel_long, root / "derived" / "kernel_energy_long.csv")

    source_hashes = {
        "frozen_source/experiments/run_rank_one_orientation.py": provenance[
            "frozen_source"
        ]["runner_file_sha256"],
        "frozen_source/experiments/aggregate_rank_one_orientation_artifacts.py": provenance[
            "frozen_source"
        ]["aggregation_file_sha256"],
    }
    for relative, expected in source_hashes.items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"frozen source hash mismatch: {relative}")
    primary = summary["stm"]["paired_equal_minus_orthogonal"]
    if (
        primary["wins_positive"] != 24
        or primary["losses_negative"] != 0
        or primary["ci95"][0] <= 0
        or not all(row["convergence"]["both_conditions_passed"] for row in rows)
    ):
        raise ValueError("confirmatory decision gate failed")
    return {
        "schema_version": 1,
        "status": "validated",
        "experiment_version": VERSION,
        "pair_count": 24,
        "ground_mixed_audit_pair_count": 24,
        "additional_four_state_audit_pair_count": 6,
        "all_convergence_gates_passed": True,
        "primary_equal_minus_orthogonal": primary,
        "channel_invariants": invariants,
        "protocol_sha256": sha256_bytes(canonical(protocol)),
        "provenance_sha256": sha256_bytes(canonical(provenance)),
        "environment_sha256": sha256_bytes(canonical(environment)),
        "raw_checkpoint_sha256s": {
            path.name: sha256_file(path)
            for path in sorted((root / "checkpoints").glob("seed_*.json"))
        },
        "raw_checkpoint_set_sha256": sha256_bytes(
            canonical(
                {
                    path.name: sha256_file(path)
                    for path in sorted((root / "checkpoints").glob("seed_*.json"))
                }
            )
        ),
        "derived_files_reconstructed": [
            "derived/summary.json",
            "derived/per_seed.csv",
            "derived/lag_capacities_long.csv",
            "derived/lag_capacities_summary.csv",
            "derived/kernel_energy_long.csv",
        ],
        "claim_boundary": protocol["claim_boundary"],
    }


def seal(root: Path) -> None:
    report = build_report(root)
    atomic_text(
        root / REPORT_NAME,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != CHECKSUM_NAME
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files
    ]
    atomic_text(root / CHECKSUM_NAME, "\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--seal",
        action="store_true",
        help="regenerate the deterministic validation report and checksum ledger",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.seal:
        seal(root)
    records = parse_ledger(root)
    expected_report = build_report(root)
    same_semantics(
        expected_report,
        load_json(root / REPORT_NAME),
        "validation_report",
    )
    primary = expected_report["primary_equal_minus_orthogonal"]
    print(
        "Validated rank-one orientation evidence: "
        f"files={len(records)}, pairs=24, wins={primary['wins_positive']}/24, "
        f"mean={primary['mean']:.12f}, "
        f"ci95=[{primary['ci95'][0]:.12f}, {primary['ci95'][1]:.12f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

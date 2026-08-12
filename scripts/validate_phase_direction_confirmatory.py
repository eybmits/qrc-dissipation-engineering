#!/usr/bin/env python3
"""Post-run numerical replay validator for the phase-direction confirmation.

The frozen experiment uses a four-right-hand-side propagation for its
initial-state convergence audit and a one-right-hand-side propagation for task
scoring.  SciPy's ``expm_multiply`` is mathematically equivalent in those two
forms but not bitwise identical because its floating-point accumulation and
norm estimation differ.  The frozen driver therefore cannot legitimately
require equality of the two raw float64 hashes.

This validator leaves the frozen protocol, driver, checkpoints, and aggregate
untouched.  It reproduces the task-path hash exactly, retains the independently
replayed batch hash as a diagnostic, and checks numerical agreement for every
one of the 72 audit cells.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"

import numpy as np
from scipy import stats
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "src")]

import run_phase_direction_confirmatory as run  # noqa: E402
from qrc import readout  # noqa: E402
from qrc.liouvillian import unvec, vec  # noqa: E402


DEFAULT_OUTDIR = ROOT / "paper" / "evidence" / "phase_direction_confirmatory_v1"
VALIDATOR_VERSION = "phase-direction-numerical-replay-v1-2026-08-12"
FEATURE_ATOL = 1e-12
# Reuse the already frozen within-condition STM stability threshold.
STM_ATOL = run.SCORE_RANGE_GATE


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def replay_cell(seed: int, condition: str, outdir: Path) -> dict:
    protocol = run.load_protocol(outdir)
    job = run.Job(condition, seed)
    convergence_path = run.convergence_checkpoint_path(outdir, job)
    task_path = run.task_checkpoint_path(outdir, job)
    convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
    task = json.loads(task_path.read_text(encoding="utf-8"))

    if run.valid_checkpoint(
        convergence_path,
        job,
        protocol["protocol_sha256"],
        "phase_direction_convergence",
    ) is None:
        raise RuntimeError(f"invalid convergence checkpoint: {job.key}")
    if run.valid_checkpoint(
        task_path,
        job,
        protocol["protocol_sha256"],
        "phase_direction_task",
    ) is None:
        raise RuntimeError(f"invalid task checkpoint: {job.key}")

    couplings, inputs, targets = run.build_problem(seed)
    reservoir = run.build_reservoir(couplings, condition)
    observables = readout.pauli_observables(run.N_QUBITS, max_weight=2)
    observable_matrices = np.stack([observable.matrix for observable in observables])

    single_features = reservoir.run(inputs, observables, washout=run.WASHOUT)
    names, initial_states = run.density_states(seed)
    state_vectors = np.stack([vec(state) for state in initial_states], axis=1)
    batch_features = np.empty(
        (len(names), run.FIT_ROWS + run.TEST, len(observables)), dtype=float
    )
    trace_at_800 = None
    pauli_at_800 = None
    for input_index, input_value in enumerate(inputs):
        state_vectors = expm_multiply(
            reservoir.liouvillian(float(input_value)) * reservoir.dt,
            state_vectors,
        )
        step = input_index + 1
        if step == run.WASHOUT:
            trace_at_800, pauli_at_800, _, _ = run.state_distances(
                state_vectors, observable_matrices
            )
        if input_index >= run.WASHOUT:
            for state_index in range(state_vectors.shape[1]):
                state = unvec(state_vectors[:, state_index], 2 ** run.N_QUBITS)
                batch_features[state_index, input_index - run.WASHOUT] = np.real(
                    np.einsum("kij,ji->k", observable_matrices, state)
                )

    if trace_at_800 is None or pauli_at_800 is None:
        raise RuntimeError(f"missing step-800 replay metrics: {job.key}")
    batch_ground = batch_features[0]

    single_hash = run.array_sha256(single_features)
    batch_hash = run.array_sha256(batch_ground)
    stored_single_hash = task["feature_sha256"]
    stored_batch_hash = convergence["feature_sha256_by_initial_state"]["ground"]
    maximum_feature_difference = float(
        np.max(np.abs(single_features - batch_ground))
    )
    rms_feature_difference = float(
        np.sqrt(np.mean((single_features - batch_ground) ** 2))
    )
    single_stm = float(run.score_fixed_raw(single_features, targets)["capacity"])
    batch_scores = np.asarray(
        [run.score_fixed_raw(features, targets)["capacity"] for features in batch_features],
        dtype=float,
    )
    batch_stm = float(batch_scores[0])
    stm_difference = float(abs(single_stm - batch_stm))
    batch_score_range = float(np.max(batch_scores) - np.min(batch_scores))
    expected_pairing = run.pairing_hashes(seed, couplings, inputs, targets)
    expected_direction = run.direction_invariants(condition)
    gates = {
        "single_replay_hash_matches_task": single_hash == stored_single_hash,
        "feature_difference_within_1e-12": maximum_feature_difference <= FEATURE_ATOL,
        "stm_difference_within_frozen_1e-4_gate": stm_difference <= STM_ATOL,
        "single_replay_stm_matches_task": abs(
            single_stm - float(task["primary_fixed_ridge"]["capacity"])
        ) <= 1e-12,
        "task_pairing_matches_fresh_problem": task["pairing"] == expected_pairing,
        "convergence_pairing_matches_fresh_problem": (
            convergence["pairing"] == expected_pairing
        ),
        "task_direction_matches_fresh_direction": task["direction"] == expected_direction,
        "convergence_direction_matches_fresh_direction": (
            convergence["direction"] == expected_direction
        ),
        "replayed_trace_at_800_passes": trace_at_800 <= run.TRACE_GATE_AT_800,
        "replayed_pauli_at_800_passes": pauli_at_800 <= run.OBSERVABLE_GATE_AT_800,
        "replayed_stm_range_passes": batch_score_range <= run.SCORE_RANGE_GATE,
        "ground_state_is_first": names[0] == "ground",
    }
    return {
        "seed": int(seed),
        "condition": condition,
        "single_replay_sha256": single_hash,
        "batch_replay_sha256": batch_hash,
        "stored_batch_sha256": stored_batch_hash,
        "batch_replay_hash_matches_stored_diagnostic": batch_hash == stored_batch_hash,
        "maximum_feature_difference": maximum_feature_difference,
        "rms_feature_difference": rms_feature_difference,
        "single_rhs_stm": single_stm,
        "four_rhs_ground_stm": batch_stm,
        "absolute_stm_difference": stm_difference,
        "four_rhs_stm_by_initial_state": {
            name: float(score) for name, score in zip(names, batch_scores)
        },
        "four_rhs_stm_range": batch_score_range,
        "replayed_trace_distance_at_800": float(trace_at_800),
        "replayed_pauli_max_abs_at_800": float(pauli_at_800),
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def verify_convergence_summary(outdir: Path, protocol: dict) -> dict:
    stored = run.authenticated_convergence_summary(outdir, protocol)
    rows = []
    for job in run.convergence_jobs():
        row = run.valid_checkpoint(
            run.convergence_checkpoint_path(outdir, job),
            job,
            protocol["protocol_sha256"],
            "phase_direction_convergence",
        )
        if row is None:
            raise RuntimeError(f"missing convergence checkpoint: {job.key}")
        rows.append(row)
    failed = [
        {"seed": row["seed"], "condition": row["condition"], "gates": row["gates"]}
        for row in rows
        if not row["all_gates_passed"]
    ]
    expected = {
        "artifact_type": "phase_direction_convergence_summary",
        "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "n_expected": len(run.convergence_jobs()),
        "n_complete": len(rows),
        "all_gates_passed": not failed,
        "failed_jobs": failed,
        "worst_trace_distance_at_800": float(
            max(row["checkpoint_metrics"]["800"]["trace_distance"] for row in rows)
        ),
        "worst_pauli_max_abs_at_800": float(
            max(row["checkpoint_metrics"]["800"]["pauli_max_abs"] for row in rows)
        ),
        "worst_fixed_ridge_stm_range": float(
            max(row["fixed_ridge_stm_range"] for row in rows)
        ),
        "maximum_numerical_trace_error": float(
            max(row["maximum_numerical_trace_error"] for row in rows)
        ),
        "maximum_numerical_hermiticity_error": float(
            max(row["maximum_numerical_hermiticity_error"] for row in rows)
        ),
        "checkpoint_sha256s": {
            f"{row['condition']}__s{row['seed']}": row["checkpoint_sha256"]
            for row in rows
        },
    }
    expected["summary_sha256"] = run.sha256_json(expected)
    if run.canonical_json(expected) != run.canonical_json(stored):
        raise RuntimeError("convergence summary does not reconstruct from checkpoints")
    return stored


def verify_aggregate(
    outdir: Path, protocol: dict, convergence: dict
) -> tuple[dict, list[dict]]:
    required = [
        outdir / "protocol.json",
        outdir / "directions.json",
        outdir / "smoke.json",
        outdir / "convergence_summary.json",
        outdir / "aggregate.json",
        outdir / "phase_direction_confirmatory.png",
        outdir / "phase_direction_confirmatory.pdf",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    stored_ledger = json.loads((outdir / "directions.json").read_text(encoding="utf-8"))
    expected_ledger = run.direction_ledger()
    if run.canonical_json(stored_ledger) != run.canonical_json(expected_ledger):
        raise RuntimeError("direction ledger reconstruction mismatch")
    if stored_ledger["direction_ledger_sha256"] != protocol["directions_sha256"]:
        raise RuntimeError("direction ledger protocol binding mismatch")

    aggregate_path = outdir / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    unhashed = dict(aggregate)
    stored_hash = unhashed.pop("aggregate_sha256")
    if stored_hash != run.sha256_json(unhashed):
        raise RuntimeError("aggregate self-hash mismatch")

    rows = run._task_rows(outdir)
    pairing_by_seed: dict[int, set[str]] = {}
    expected_pairing_by_seed = {}
    for seed in run.CONFIRMATORY_SEEDS:
        couplings, inputs, targets = run.build_problem(seed)
        expected_pairing_by_seed[seed] = run.pairing_hashes(
            seed, couplings, inputs, targets
        )
    for row in rows:
        seed = int(row["seed"])
        condition = row["condition"]
        pairing_by_seed.setdefault(seed, set()).add(run.sha256_json(row["pairing"]))
        if row["pairing"] != expected_pairing_by_seed[seed]:
            raise RuntimeError(f"task pairing does not reconstruct: {condition} seed {seed}")
        if row["protocol_version"] != run.PROTOCOL_VERSION:
            raise RuntimeError("task protocol-version binding failed")
        if row["source_environment_sha256"] != protocol["source_environment_sha256"]:
            raise RuntimeError("task source-environment binding failed")
        if row["direction"] != run.direction_invariants(condition):
            raise RuntimeError(f"task direction does not reconstruct: {condition}")
        if row["feature_shape"] != [run.FIT_ROWS + run.TEST, 45]:
            raise RuntimeError(f"task feature shape mismatch: {condition} seed {seed}")
        if not np.isclose(
            sum(row["primary_fixed_ridge"]["delay_capacities"]),
            row["primary_fixed_ridge"]["capacity"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"task lag capacities do not reconstruct: {condition} seed {seed}")
    if set(pairing_by_seed) != set(run.CONFIRMATORY_SEEDS):
        raise RuntimeError("task seed ledger mismatch")
    if any(len(hashes) != 1 for hashes in pairing_by_seed.values()):
        raise RuntimeError("within-seed task pairing mismatch")
    by_condition = {
        condition: {
            "fixed": np.asarray(
                [
                    row["primary_fixed_ridge"]["capacity"]
                    for row in rows
                    if row["condition"] == condition
                ],
                dtype=float,
            ),
            "selected": np.asarray(
                [
                    row["validation_selected_sensitivity"]["selected_test"]
                    for row in rows
                    if row["condition"] == condition
                ],
                dtype=float,
            ),
        }
        for condition in run.CONDITIONS
    }
    expected_summaries = []
    for condition in run.CONDITIONS:
        fixed = by_condition[condition]["fixed"]
        selected_values = by_condition[condition]["selected"]
        expected_summaries.append(
            {
                **run.direction_invariants(condition),
                "n": run.N_SEEDS,
                "fixed_ridge_mean": float(np.mean(fixed)),
                "fixed_ridge_se": float(stats.sem(fixed)),
                "fixed_ridge_values": [float(value) for value in fixed],
                "validation_selected_mean": float(np.mean(selected_values)),
                "validation_selected_se": float(stats.sem(selected_values)),
                "validation_selected_values": [float(value) for value in selected_values],
            }
        )
    if run.canonical_json(expected_summaries) != run.canonical_json(
        aggregate["condition_summaries"]
    ):
        raise RuntimeError("condition summaries do not reconstruct")

    inference_rng = np.random.default_rng(run.INFERENCE_NAMESPACE)
    equal = by_condition[run.EQUAL_PHASE]["fixed"]
    primary_values = equal - by_condition[run.ORTHOGONAL_FOURIER]["fixed"]
    primary = run.paired_summary(primary_values)
    primary["estimand"] = "STM(path_f0)-STM(path_f1)"
    primary["monte_carlo_signflip_p_two_sided"] = run.monte_carlo_signflip_p(
        primary_values, inference_rng
    )
    primary["draws"] = run.SIGNFLIP_DRAWS

    zero_mean = np.mean(
        np.stack(
            [by_condition[condition]["fixed"] for condition in run.ZERO_OVERLAP_CONDITIONS]
        ),
        axis=0,
    )
    pooled_values = equal - zero_mean
    pooled = run.paired_summary(pooled_values)
    pooled["estimand"] = "STM(path_f0)-within-seed mean(five zero-overlap rays)"
    pooled["monte_carlo_signflip_p_two_sided"] = run.monte_carlo_signflip_p(
        pooled_values, inference_rng
    )
    pooled["draws"] = run.SIGNFLIP_DRAWS
    pooled["tested_by_fixed_sequence_gate"] = bool(
        primary["monte_carlo_signflip_p_two_sided"] <= 0.05
    )
    pooled["gatekeeping_rejects_at_0.05"] = bool(
        pooled["tested_by_fixed_sequence_gate"]
        and pooled["monte_carlo_signflip_p_two_sided"] <= 0.05
    )

    secondary_conditions = (
        "path_f025",
        "path_f05",
        "path_f075",
        "scrambled_r1",
        "scrambled_r2",
        "scrambled_r3",
        "scrambled_r4",
    )
    secondary = {}
    raw_p = {}
    for condition in secondary_conditions:
        values = equal - by_condition[condition]["fixed"]
        item = run.paired_summary(values, simultaneous_count=len(secondary_conditions))
        item["estimand"] = f"STM(path_f0)-STM({condition})"
        item["monte_carlo_signflip_p_two_sided"] = run.monte_carlo_signflip_p(
            values, inference_rng
        )
        item["draws"] = run.SIGNFLIP_DRAWS
        secondary[condition] = item
        raw_p[condition] = item["monte_carlo_signflip_p_two_sided"]
    for condition, value in run.holm_adjust(raw_p).items():
        secondary[condition]["holm_adjusted_signflip_p"] = float(value)

    selected_values = (
        by_condition[run.EQUAL_PHASE]["selected"]
        - by_condition[run.ORTHOGONAL_FOURIER]["selected"]
    )
    selected = run.paired_summary(selected_values)
    selected["estimand"] = "validation-selected STM(path_f0)-STM(path_f1)"
    ordered = run.ordered_path_diagnostic(
        np.column_stack([by_condition[condition]["fixed"] for condition in run.PHASE_PATH]),
        inference_rng,
    )

    expected_blocks = {
        "confirmatory_primary": primary,
        "gated_zero_overlap_generality": pooled,
        "secondary_contrasts": secondary,
        "validation_selected_sensitivity": selected,
        "ordered_path_diagnostic": ordered,
    }
    for key, expected in expected_blocks.items():
        if run.canonical_json(expected) != run.canonical_json(aggregate[key]):
            raise RuntimeError(f"reconstructed aggregate block mismatch: {key}")

    expected_checkpoint_hashes = {
        f"{row['condition']}__s{row['seed']}": row["checkpoint_sha256"] for row in rows
    }
    if expected_checkpoint_hashes != aggregate["task_checkpoint_sha256s"]:
        raise RuntimeError("aggregate task-checkpoint ledger mismatch")

    expected_top_level = {
        "artifact_type": "phase_direction_confirmatory_aggregate",
        "status": "complete",
        "protocol_version": run.PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_environment_sha256": protocol["source_environment_sha256"],
        "pilot_scores_included": False,
        "n_seeds": run.N_SEEDS,
        "n_conditions": len(run.CONDITIONS),
        "n_task_checkpoints": len(rows),
        "convergence_summary_sha256": convergence["summary_sha256"],
        "claim_boundary": protocol["claim_boundary"],
    }
    for key, expected in expected_top_level.items():
        if aggregate.get(key) != expected:
            raise RuntimeError(f"aggregate top-level mismatch: {key}")
    return aggregate, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    outdir = args.outdir.resolve()

    started = time.perf_counter()
    protocol = run.load_protocol(outdir)
    convergence = verify_convergence_summary(outdir, protocol)
    if convergence["all_gates_passed"] is not True:
        raise RuntimeError("frozen convergence gate failed")
    aggregate, rows = verify_aggregate(outdir, protocol, convergence)

    jobs = run.convergence_jobs()
    replay_rows = []
    if args.workers == 1:
        for job in jobs:
            replay_rows.append(replay_cell(job.seed, job.condition, outdir))
            print(f"{job.key} replayed", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {
                executor.submit(replay_cell, job.seed, job.condition, outdir): job
                for job in jobs
            }
            for future in as_completed(pending):
                job = pending[future]
                replay_rows.append(future.result())
                print(f"{job.key} replayed", flush=True)
    replay_rows.sort(key=lambda row: (row["seed"], run.CONDITIONS.index(row["condition"])))
    failed = [row for row in replay_rows if not row["all_gates_passed"]]
    if failed:
        raise RuntimeError(f"numerical replay failed for {len(failed)} audit cells")

    replay_audit = {
        "artifact_type": "phase_direction_numerical_replay_audit",
        "validator_version": VALIDATOR_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "feature_tolerance": FEATURE_ATOL,
        "stm_tolerance": STM_ATOL,
        "n_expected": len(jobs),
        "n_complete": len(replay_rows),
        "maximum_feature_difference": max(
            row["maximum_feature_difference"] for row in replay_rows
        ),
        "maximum_rms_feature_difference": max(
            row["rms_feature_difference"] for row in replay_rows
        ),
        "maximum_stm_difference": max(
            row["absolute_stm_difference"] for row in replay_rows
        ),
        "maximum_replayed_trace_distance_at_800": max(
            row["replayed_trace_distance_at_800"] for row in replay_rows
        ),
        "maximum_replayed_pauli_max_abs_at_800": max(
            row["replayed_pauli_max_abs_at_800"] for row in replay_rows
        ),
        "maximum_replayed_four_rhs_stm_range": max(
            row["four_rhs_stm_range"] for row in replay_rows
        ),
        "batch_hash_match_count_diagnostic": sum(
            row["batch_replay_hash_matches_stored_diagnostic"] for row in replay_rows
        ),
        "all_gates_passed": True,
        "rows": replay_rows,
    }
    replay_audit["replay_audit_sha256"] = run.sha256_json(replay_audit)
    run.atomic_json(outdir / "numerical_replay_audit.json", replay_audit)

    amendment = {
        "artifact_type": "phase_direction_validation_amendment",
        "validator_version": VALIDATOR_VERSION,
        "created_at_utc": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "reason": (
            "The frozen validator required bitwise equality between one-RHS task "
            "propagation and four-RHS convergence propagation. SciPy expm_multiply "
            "uses different floating-point accumulation and norm estimation in "
            "those forms, so raw float64 hashes are not expected to match."
        ),
        "remedy": (
            "The frozen source and all scientific checkpoints remain unchanged. "
            "The task backend must reproduce its archived hash; the four-state "
            "batch backend is replayed independently as a numerical diagnostic. "
            "All 72 audit cells must agree within 1e-12 in features and within "
            "the frozen 1e-4 STM stability threshold."
        ),
        "scientific_protocol_changed": False,
        "seeds_conditions_inference_or_scores_changed": False,
        "validator_source": "scripts/validate_phase_direction_confirmatory.py",
        "validator_source_sha256": run.file_sha256(Path(__file__).resolve()),
    }
    amendment["amendment_sha256"] = run.sha256_json(amendment)
    run.atomic_json(outdir / "validation_amendment.json", amendment)

    report = {
        "status": "validated_confirmatory_result_with_numerical_replay_amendment",
        "validated_at_utc": utc_now(),
        "protocol_sha256": protocol["protocol_sha256"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "convergence_summary_sha256": convergence["summary_sha256"],
        "n_task_checkpoints": len(rows),
        "n_convergence_checkpoints": len(jobs),
        "n_numerical_replays": len(replay_rows),
        "n_seeds": run.N_SEEDS,
        "n_conditions": len(run.CONDITIONS),
        "all_pairing_hashes_match": True,
        "all_convergence_gates_pass": True,
        "all_numerical_replay_gates_pass": True,
        "pilot_scores_included": False,
        "maximum_feature_difference": replay_audit["maximum_feature_difference"],
        "maximum_stm_difference": replay_audit["maximum_stm_difference"],
        "maximum_replayed_trace_distance_at_800": replay_audit[
            "maximum_replayed_trace_distance_at_800"
        ],
        "maximum_replayed_pauli_max_abs_at_800": replay_audit[
            "maximum_replayed_pauli_max_abs_at_800"
        ],
        "maximum_replayed_four_rhs_stm_range": replay_audit[
            "maximum_replayed_four_rhs_stm_range"
        ],
        "primary": aggregate["confirmatory_primary"],
        "gated_zero_overlap_generality": aggregate[
            "gated_zero_overlap_generality"
        ],
        "source_environment_sha256": protocol["source_environment_sha256"],
        "validator_source_sha256": amendment["validator_source_sha256"],
        "runtime_seconds": float(time.perf_counter() - started),
        "claim_boundary": protocol["claim_boundary"],
    }
    report["validation_report_sha256"] = run.sha256_json(report)
    run.atomic_json(outdir / "validation_report.json", report)

    checksum_files = sorted(
        path
        for path in outdir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_lines = [
        f"{run.file_sha256(path)}  {path.relative_to(outdir).as_posix()}"
        for path in checksum_files
    ]
    (outdir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

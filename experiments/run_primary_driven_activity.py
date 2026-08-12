"""Post-hoc realized jump activity on the sealed primary STM trajectories.

This analysis replays the 32 paired ``A_table`` STM trajectories from
``results/final_protocol_results.tar.gz``.  It does not fit, select, or score a
readout.  For each declared jump unraveling it evaluates

    A(t) = sum_k gamma_k Tr(L_k^dagger L_k rho(t))

and integrates it exactly (up to ``scipy.sparse.linalg.expm_multiply``
tolerance) through every one of the 400 held-out input intervals.  The reported
time average is

    A_bar = (400 * dt)^-1 sum_t integral_0^dt A_t(tau) d tau.

The test inputs, coupling draws, dissipators, initial state, washout, and
training prefix are exactly those of the sealed primary protocol.  Reusing the
reported test trajectories makes this a deterministic descriptive/post-hoc
sensitivity analysis, not a fresh confirmatory comparison and not an
activity-matched performance experiment.

Examples
--------
PYTHONPATH=src:experiments python experiments/run_primary_driven_activity.py run
PYTHONPATH=src:experiments python experiments/run_primary_driven_activity.py aggregate
PYTHONPATH=src:experiments python experiments/run_primary_driven_activity.py report
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import hashlib
import json
import math
import platform
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import _paths  # noqa: F401
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.stats import t as student_t

from _paths import REPORTS_DIR, RESULTS_DIR
from qrc import dissipators as dsp
from qrc import reservoirs as res
from qrc import tasks
from qrc.liouvillian import vec
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir
from run_final_scaling import build_jumps, deterministic_seeds


PROTOCOL_VERSION = "primary-driven-jump-activity-posthoc-v1-2026-07-25"
EXPECTED_BASELINE_ARCHIVE_SHA256 = (
    "e24df615f8762ba9aa950673b5d776eddbc186a9ad277086c2930cde0ea46948"
)
N_QUBITS = 5
H = 0.5
DT = 0.5
GAMMA = 1.0
WASH = 200
TRAIN = 600
TEST = 400
METHODS = (
    "CD_paper",
    "B3_collective",
    "A1_heterogeneous",
    "B5_pair",
    "B2_thermal",
    "B4_loss_exchange",
    "B1_dephasing",
)
METHOD_LABELS = {
    "CD_paper": "uniform local",
    "B3_collective": "collective",
    "A1_heterogeneous": "unequal local",
    "B5_pair": "pair loss",
    "B2_thermal": "local gain/loss",
    "B4_loss_exchange": "loss + exchange",
    "B1_dephasing": "dephasing",
}
REFERENCE_METHOD = "CD_paper"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "results" / "final_protocol_results.tar.gz"
DEFAULT_OUTDIR = Path(RESULTS_DIR) / "primary_driven_activity"
DEFAULT_REPORT = Path(REPORTS_DIR) / "primary_driven_activity_report.md"
SOURCE_FILES = (
    "experiments/run_primary_driven_activity.py",
    "experiments/run_final_scaling.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)
TRACE_TOL = 2e-9
IMAGINARY_TOL = 2e-9
NEGATIVE_COUNT_TOL = 2e-10


@dataclass(frozen=True)
class Job:
    method: str
    seed: int


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        canonical = np.ascontiguousarray(array, dtype="<c16")
    else:
        canonical = np.ascontiguousarray(array, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def jump_family_sha256(
    jumps: Sequence[tuple[np.ndarray, float]],
) -> str:
    payload = [
        {
            "operator_sha256": array_sha256(np.asarray(operator)),
            "rate": float(rate),
        }
        for operator, rate in jumps
    ]
    return sha256_json(payload)


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def baseline_member_name(job: Job) -> str:
    return (
        "final_protocol/"
        f"A_table__stm_N{N_QUBITS}_{job.method}_s{job.seed}"
        f"_h{H:g}_dt{DT:g}_L{WASH}-{TRAIN}-{TEST}.json"
    )


def _read_baseline_entries(archive: Path, jobs: Sequence[Job]) -> dict:
    if file_sha256(archive) != EXPECTED_BASELINE_ARCHIVE_SHA256:
        raise RuntimeError(
            "sealed primary archive SHA-256 does not match the frozen protocol"
        )
    entries = {}
    with tarfile.open(archive, "r:gz") as bundle:
        members = {member.name: member for member in bundle.getmembers()}
        for job in jobs:
            name = baseline_member_name(job)
            member = members.get(name)
            if member is None:
                raise RuntimeError(f"sealed primary checkpoint missing: {name}")
            handle = bundle.extractfile(member)
            if handle is None:
                raise RuntimeError(f"cannot read sealed checkpoint: {name}")
            row = json.load(handle)
            expected = {
                "block": "A_table",
                "N": N_QUBITS,
                "method": job.method,
                "task": "stm",
                "seed": job.seed,
                "h": H,
                "dt": DT,
                "wash": WASH,
                "train": TRAIN,
                "test": TEST,
                "backend": "sparse",
            }
            mismatches = {
                key: (row.get(key), value)
                for key, value in expected.items()
                if row.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    f"sealed checkpoint metadata mismatch for {name}: {mismatches}"
                )
            value = float(row["value"])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite sealed STM value: {name}")
            entries[f"{job.method}/{job.seed}"] = {
                "member": name,
                "checkpoint_sha256": hashlib.sha256(
                    canonical_json(row).encode("utf-8")
                ).hexdigest(),
                "stm_capacity": value,
            }
    return entries


def build_protocol(archive: Path = DEFAULT_ARCHIVE) -> dict:
    seeds = deterministic_seeds(32)
    jobs = [Job(method, seed) for method in METHODS for seed in seeds]
    source_environment = {
        relative: file_sha256(ROOT / relative) for relative in SOURCE_FILES
    }
    entries = _read_baseline_entries(archive, jobs)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "analysis_status": (
            "deterministic descriptive post-hoc reuse of sealed primary test "
            "trajectories"
        ),
        "definition": {
            "instantaneous_activity": (
                "sum_k gamma_k Tr(L_k^dagger L_k rho(t))"
            ),
            "reported_activity": (
                "(TEST*dt)^-1 sum_test_intervals integral_0^dt "
                "instantaneous_activity(tau) d tau"
            ),
            "interpretation": (
                "expected jump count per unit physical simulation time under "
                "the declared jump unraveling"
            ),
            "integration": (
                "augmented sparse matrix exponential action for each held-out "
                "piecewise-constant driven interval"
            ),
        },
        "n_qubits": N_QUBITS,
        "h": H,
        "dt": DT,
        "gamma": GAMMA,
        "split": {
            "wash": WASH,
            "train": TRAIN,
            "test": TEST,
            "test_start_input_index": WASH + TRAIN,
            "total_inputs": WASH + TRAIN + TEST,
            "activity_uses_only_test_intervals": True,
        },
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "reference_method": REFERENCE_METHOD,
        "seeds": seeds,
        "n_jobs": len(jobs),
        "baseline_archive": {
            "path": str(archive.relative_to(ROOT)),
            "sha256": EXPECTED_BASELINE_ARCHIVE_SHA256,
            "entries": entries,
            "entries_sha256": sha256_json(entries),
        },
        "boundary_guard": {
            "uses_supervised_targets": False,
            "fits_or_selects_readout": False,
            "reuses_reported_test_block": True,
            "fresh_confirmatory_ensemble": False,
        },
        "source_environment": source_environment,
        "source_environment_sha256": sha256_json(source_environment),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    return protocol


def protocol_sha256(protocol: dict) -> str:
    return sha256_json(protocol)


def jump_rate_operator(
    jumps: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    if not jumps:
        raise ValueError("at least one jump is required")
    d = np.asarray(jumps[0][0]).shape[0]
    operator = np.zeros((d, d), dtype=complex)
    for jump, rate in jumps:
        matrix = np.asarray(jump, dtype=complex)
        operator += float(rate) * (matrix.conj().T @ matrix)
    return operator


def activity_functional(rate_operator: np.ndarray) -> np.ndarray:
    """Row ``q`` satisfying ``q @ vec(rho) == Tr(K rho)``."""
    return vec(np.asarray(rate_operator).T)


def integrated_activity_step(
    generator: sparse.spmatrix,
    state_vector: np.ndarray,
    rate_functional: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    """Evolve one interval and return state, expected count, imaginary residue."""
    generator = sparse.csr_matrix(generator, dtype=complex)
    n = generator.shape[0]
    if generator.shape != (n, n) or state_vector.shape != (n,):
        raise ValueError("generator/state dimensions do not agree")
    functional = np.asarray(rate_functional, dtype=complex)
    if functional.shape != (n,):
        raise ValueError("activity functional dimension does not agree")

    augmented = sparse.bmat(
        [
            [generator, sparse.csr_matrix((n, 1), dtype=complex)],
            [
                sparse.csr_matrix(functional.reshape(1, n)),
                sparse.csr_matrix((1, 1), dtype=complex),
            ],
        ],
        format="csr",
    )
    initial = np.empty(n + 1, dtype=complex)
    initial[:n] = state_vector
    initial[n] = 0.0
    evolved = expm_multiply(augmented * float(dt), initial)
    residue = float(abs(np.imag(evolved[n])))
    count = float(np.real(evolved[n]))
    if count < -NEGATIVE_COUNT_TOL:
        raise RuntimeError(f"negative integrated jump activity: {count}")
    return evolved[:n], max(count, 0.0), residue


def _trace_from_vector(state_vector: np.ndarray, dimension: int) -> complex:
    identity_functional = activity_functional(np.eye(dimension, dtype=complex))
    return complex(identity_functional @ state_vector)


def _construct_job(job: Job):
    problem_rng = np.random.default_rng(job.seed)
    couplings = res.random_couplings(N_QUBITS, 1.0, problem_rng)
    inputs = tasks.stm_inputs(WASH + TRAIN + TEST, problem_rng)
    method_rng = np.random.default_rng(job.seed + 1)
    target_strength = dsp.jump_strength(dsp.local_loss(N_QUBITS, GAMMA))
    jumps = build_jumps(
        job.method,
        couplings,
        N_QUBITS,
        target_strength,
        method_rng,
    )
    h0 = ising_xx_hamiltonian(couplings, H, N_QUBITS)
    hx = transverse_drive(N_QUBITS)
    reservoir = SparseLindbladReservoir.from_terms(
        N_QUBITS,
        h0 + H * hx,
        H * hx,
        jumps,
        DT,
    )
    return couplings, inputs, jumps, target_strength, reservoir


def run_job(job: Job, protocol: dict) -> dict:
    started = time.perf_counter()
    key = f"{job.method}/{job.seed}"
    baseline = protocol["baseline_archive"]["entries"].get(key)
    if baseline is None:
        raise RuntimeError(f"job is absent from frozen primary entries: {key}")
    couplings, inputs, jumps, target_strength, reservoir = _construct_job(job)
    actual_strength = dsp.jump_strength(jumps)
    if abs(actual_strength - target_strength) > 1e-10 * target_strength:
        raise RuntimeError("jump-strength invariant failed")

    rho = reservoir.initial_state()
    for input_value in inputs[: WASH + TRAIN]:
        rho = reservoir.step(rho, float(input_value))

    rate_operator = jump_rate_operator(jumps)
    functional = activity_functional(rate_operator)
    state_vector = vec(rho)
    counts = []
    max_trace_error = abs(
        _trace_from_vector(state_vector, rate_operator.shape[0]) - 1.0
    )
    max_imaginary_residue = 0.0
    for input_value in inputs[WASH + TRAIN :]:
        state_vector, count, residue = integrated_activity_step(
            reservoir.liouvillian(float(input_value)),
            state_vector,
            functional,
            DT,
        )
        counts.append(count)
        trace_error = abs(
            _trace_from_vector(state_vector, rate_operator.shape[0]) - 1.0
        )
        max_trace_error = max(max_trace_error, trace_error)
        max_imaginary_residue = max(max_imaginary_residue, residue)

    if len(counts) != TEST:
        raise RuntimeError("test-interval coverage invariant failed")
    if max_trace_error > TRACE_TOL:
        raise RuntimeError(f"trace drift exceeded tolerance: {max_trace_error}")
    if max_imaginary_residue > IMAGINARY_TOL:
        raise RuntimeError(
            f"activity imaginary residue exceeded tolerance: "
            f"{max_imaginary_residue}"
        )

    counts_array = np.asarray(counts)
    time_averaged_activity = float(np.sum(counts_array) / (TEST * DT))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol["source_environment_sha256"],
        "analysis_status": "descriptive_posthoc_test_trajectory_reuse",
        "method": job.method,
        "method_label": METHOD_LABELS[job.method],
        "seed": job.seed,
        "n_qubits": N_QUBITS,
        "h": H,
        "dt": DT,
        "wash_intervals": WASH,
        "training_intervals": TRAIN,
        "test_intervals": TEST,
        "baseline_checkpoint_member": baseline["member"],
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "sealed_stm_capacity": baseline["stm_capacity"],
        "coupling_sha256": array_sha256(couplings),
        "full_input_sha256": array_sha256(inputs),
        "test_input_sha256": array_sha256(inputs[WASH + TRAIN :]),
        "jump_family_sha256": jump_family_sha256(jumps),
        "target_frobenius_jump_strength": float(target_strength),
        "actual_frobenius_jump_strength": float(actual_strength),
        "relative_frobenius_strength_error": float(
            abs(actual_strength - target_strength) / target_strength
        ),
        "time_averaged_jump_activity": time_averaged_activity,
        "total_expected_jumps_over_test": float(np.sum(counts_array)),
        "mean_expected_jumps_per_input_interval": float(np.mean(counts_array)),
        "minimum_interval_averaged_activity": float(np.min(counts_array) / DT),
        "maximum_interval_averaged_activity": float(np.max(counts_array) / DT),
        "interval_averaged_activity_std": float(
            np.std(counts_array / DT, ddof=1)
        ),
        "maximum_trace_error": float(max_trace_error),
        "maximum_activity_imaginary_residue": float(max_imaginary_residue),
        "minimum_integrated_interval_activity": float(np.min(counts_array)),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def job_path(outdir: Path, job: Job) -> Path:
    return outdir / "checkpoints" / f"{job.method}__seed_{job.seed}.json"


def protocol_path(outdir: Path) -> Path:
    return outdir / "protocol.json"


def aggregate_path(outdir: Path) -> Path:
    return outdir / "aggregate.json"


def _valid_checkpoint(path: Path, job: Job, protocol: dict) -> dict | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    expected = {
        "method": job.method,
        "seed": job.seed,
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol["source_environment_sha256"],
    }
    if all(row.get(key) == value for key, value in expected.items()):
        return row
    raise RuntimeError(f"stale or mismatched checkpoint: {path}")


def _run_and_write(job: Job, protocol: dict, outdir_text: str) -> str:
    outdir = Path(outdir_text)
    path = job_path(outdir, job)
    existing = _valid_checkpoint(path, job, protocol)
    if existing is not None:
        return f"skip {job.method} seed={job.seed}"
    row = run_job(job, protocol)
    atomic_write_json(path, row)
    return (
        f"done {job.method} seed={job.seed} "
        f"activity={row['time_averaged_jump_activity']:.9f} "
        f"runtime={row['runtime_seconds']:.2f}s"
    )


def run_jobs(outdir: Path, protocol: dict, workers: int) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1, 8]")
    jobs = [
        Job(method, seed)
        for method in protocol["methods"]
        for seed in protocol["seeds"]
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(protocol_path(outdir), protocol)
    if workers <= 1:
        for job in jobs:
            print(_run_and_write(job, protocol, str(outdir)), flush=True)
        return
    try:
        executor = ProcessPoolExecutor(max_workers=workers)
    except (OSError, PermissionError) as exc:
        print(
            "process pool unavailable; using bounded thread fallback "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )
        executor = ThreadPoolExecutor(max_workers=workers)
    with executor as pool:
        futures = {
            pool.submit(_run_and_write, job, protocol, str(outdir)): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(
                    f"activity job failed for {job.method}/{job.seed}"
                ) from exc


def _load_rows(outdir: Path, protocol: dict) -> list[dict]:
    rows = []
    missing = []
    for method in protocol["methods"]:
        for seed in protocol["seeds"]:
            job = Job(method, seed)
            row = _valid_checkpoint(job_path(outdir, job), job, protocol)
            if row is None:
                missing.append(f"{method}/{seed}")
            else:
                rows.append(row)
    if missing:
        raise RuntimeError(
            f"incomplete activity result set: {len(missing)} missing; "
            f"first={missing[:5]}"
        )
    return rows


def _mean_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    critical = float(student_t.ppf(0.975, len(values) - 1))
    return mean, se, mean - critical * se, mean + critical * se


def build_aggregate(rows: Sequence[dict], protocol: dict) -> dict:
    expected_jobs = len(protocol["methods"]) * len(protocol["seeds"])
    keys = {(row["method"], int(row["seed"])) for row in rows}
    expected_keys = {
        (method, seed)
        for method in protocol["methods"]
        for seed in protocol["seeds"]
    }
    errors = []
    if len(rows) != expected_jobs or keys != expected_keys:
        errors.append("coverage or unique method/seed key invariant failed")

    max_budget_error = max(
        float(row["relative_frobenius_strength_error"]) for row in rows
    )
    max_trace_error = max(float(row["maximum_trace_error"]) for row in rows)
    max_imaginary = max(
        float(row["maximum_activity_imaginary_residue"]) for row in rows
    )
    min_count = min(
        float(row["minimum_integrated_interval_activity"]) for row in rows
    )
    if max_budget_error > 1e-10:
        errors.append("Frobenius jump-strength invariant failed")
    if max_trace_error > TRACE_TOL:
        errors.append("trace invariant failed")
    if max_imaginary > IMAGINARY_TOL:
        errors.append("activity imaginary-residue invariant failed")
    if min_count < -NEGATIVE_COUNT_TOL:
        errors.append("non-negativity invariant failed")

    by_method = {}
    arrays = {}
    for method in protocol["methods"]:
        ordered = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: protocol["seeds"].index(int(row["seed"])),
        )
        values = np.asarray(
            [row["time_averaged_jump_activity"] for row in ordered], dtype=float
        )
        arrays[method] = values
        mean, se, lower, upper = _mean_interval(values)
        by_method[method] = {
            "label": METHOD_LABELS[method],
            "n": len(values),
            "mean_time_averaged_activity": mean,
            "standard_error": se,
            "descriptive_95pct_t_interval": [lower, upper],
            "minimum_seed_activity": float(np.min(values)),
            "maximum_seed_activity": float(np.max(values)),
            "mean_total_expected_jumps_over_test": float(mean * TEST * DT),
        }

    local = arrays[REFERENCE_METHOD]
    paired_vs_local = {}
    for method in protocol["methods"]:
        values = arrays[method]
        differences = values - local
        ratios = values / local
        mean_diff, se_diff, lower_diff, upper_diff = _mean_interval(differences)
        mean_ratio, se_ratio, lower_ratio, upper_ratio = _mean_interval(ratios)
        paired_vs_local[method] = {
            "label": METHOD_LABELS[method],
            "mean_activity_difference": mean_diff,
            "difference_standard_error": se_diff,
            "descriptive_difference_95pct_t_interval": [
                lower_diff,
                upper_diff,
            ],
            "ratio_of_method_mean_to_local_mean": float(
                np.mean(values) / np.mean(local)
            ),
            "mean_paired_activity_ratio": mean_ratio,
            "paired_ratio_standard_error": se_ratio,
            "descriptive_paired_ratio_95pct_t_interval": [
                lower_ratio,
                upper_ratio,
            ],
            "higher_activity_count": int(np.sum(differences > 0)),
            "equal_activity_count": int(np.sum(differences == 0)),
            "lower_activity_count": int(np.sum(differences < 0)),
        }

    aggregate = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol["source_environment_sha256"],
        "status": "complete" if not errors else "invalid",
        "analysis_status": (
            "descriptive_posthoc_reuse_of_sealed_primary_test_trajectories"
        ),
        "n_jobs": len(rows),
        "n_seeds": len(protocol["seeds"]),
        "method_summaries": by_method,
        "paired_vs_uniform_local": paired_vs_local,
        "invariant_audit": {
            "passed": not errors,
            "errors": errors,
            "expected_jobs": expected_jobs,
            "observed_jobs": len(rows),
            "maximum_relative_frobenius_strength_error": max_budget_error,
            "maximum_trace_error": max_trace_error,
            "maximum_activity_imaginary_residue": max_imaginary,
            "minimum_integrated_interval_activity": min_count,
            "all_rows_use_400_test_intervals": all(
                int(row["test_intervals"]) == TEST for row in rows
            ),
            "all_rows_link_sealed_checkpoint": all(
                row["baseline_checkpoint_member"]
                == protocol["baseline_archive"]["entries"][
                    f"{row['method']}/{row['seed']}"
                ]["member"]
                for row in rows
            ),
        },
        "runtime": {
            "sum_job_seconds": float(
                sum(float(row["runtime_seconds"]) for row in rows)
            ),
            "maximum_job_seconds": float(
                max(float(row["runtime_seconds"]) for row in rows)
            ),
        },
        "limitations": [
            (
                "The analysis reuses the reported held-out STM test trajectories "
                "and is descriptive/post-hoc, not a fresh confirmatory test."
            ),
            (
                "It characterizes activity at fixed Frobenius normalization; "
                "it does not rerun task performance after matching activity."
            ),
            (
                "Jump activity is tied to the declared gauge-fixed jump "
                "unraveling and is not a generator-invariant cost."
            ),
            (
                "The density-matrix expectation is the mean jump count of that "
                "unraveling; no stochastic trajectories are sampled."
            ),
            (
                "It is neither dissipated energy nor entropy production; those "
                "require an explicit bath/thermodynamic model."
            ),
        ],
    }
    return aggregate


def aggregate_results(outdir: Path, protocol: dict) -> dict:
    rows = _load_rows(outdir, protocol)
    aggregate = build_aggregate(rows, protocol)
    if aggregate["status"] != "complete":
        raise RuntimeError(
            f"activity aggregate failed: {aggregate['invariant_audit']['errors']}"
        )
    atomic_write_json(aggregate_path(outdir), aggregate)
    return aggregate


def render_report(aggregate: dict, protocol: dict) -> str:
    lines = [
        "# Realized jump activity on the primary driven test trajectories",
        "",
        "**Status:** deterministic descriptive/post-hoc sensitivity analysis.",
        "",
        "This analysis replays the sealed 32-seed primary STM ensemble at "
        "$N=5$ and the fixed Frobenius jump-strength normalization. It uses "
        "the original 200 washout, 600 training, and 400 held-out test input "
        "intervals. No target, readout fit, validation choice, or task score "
        "enters the activity calculation.",
        "",
        "For the declared jump unraveling, the instantaneous expected jump rate is",
        "",
        "$$A(t)=\\sum_k\\gamma_k\\,\\mathrm{Tr}"
        "(L_k^\\dagger L_k\\rho(t)).$$",
        "",
        "The reported value integrates this rate through each continuously "
        "evolved, piecewise-constant input interval and divides the total "
        "expected count by the held-out physical duration "
        f"$400\\times{DT:g}={TEST * DT:g}$.",
        "",
        "| design | mean activity | descriptive 95% t interval | "
        "ratio of means vs local | paired higher/lower count |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in protocol["methods"]:
        summary = aggregate["method_summaries"][method]
        paired = aggregate["paired_vs_uniform_local"][method]
        lower, upper = summary["descriptive_95pct_t_interval"]
        lines.append(
            f"| {summary['label']} | "
            f"{summary['mean_time_averaged_activity']:.9f} | "
            f"[{lower:.9f}, {upper:.9f}] | "
            f"{paired['ratio_of_method_mean_to_local_mean']:.6f} | "
            f"{paired['higher_activity_count']}/"
            f"{paired['lower_activity_count']} |"
        )

    collective = aggregate["paired_vs_uniform_local"]["B3_collective"]
    dlow, dhigh = collective["descriptive_difference_95pct_t_interval"]
    lines.extend(
        [
            "",
            "## Headline local--collective comparison",
            "",
            "At the same Frobenius normalization, collective loss has a mean "
            f"realized activity ratio of "
            f"{collective['ratio_of_method_mean_to_local_mean']:.6f} relative "
            "to uniform local damping. The mean paired activity difference is "
            f"{collective['mean_activity_difference']:.9f}, with a descriptive "
            f"95% t interval [{dlow:.9f}, {dhigh:.9f}] and "
            f"{collective['higher_activity_count']}/"
            f"{protocol['n_jobs'] // len(protocol['methods'])} instances above "
            "local.",
            "",
            "This quantifies whether the fixed Frobenius convention realizes "
            "the same environmental activity on the actual driven ensemble. "
            "It does **not** establish collective performance superiority after "
            "activity matching; that would require a separate, frozen "
            "rate-matching and fresh-test protocol.",
            "",
            "## Reproducibility and numerical audit",
            "",
            f"- Sealed baseline archive SHA-256: "
            f"`{protocol['baseline_archive']['sha256']}`.",
            f"- Protocol SHA-256: `{aggregate['protocol_sha256']}`.",
            f"- Source-environment SHA-256: "
            f"`{aggregate['source_environment_sha256']}`.",
            f"- Coverage: {aggregate['n_jobs']}/{protocol['n_jobs']} jobs.",
            f"- Maximum relative Frobenius-strength error: "
            f"{aggregate['invariant_audit']['maximum_relative_frobenius_strength_error']:.3e}.",
            f"- Maximum trace error: "
            f"{aggregate['invariant_audit']['maximum_trace_error']:.3e}.",
            f"- Maximum imaginary integration residue: "
            f"{aggregate['invariant_audit']['maximum_activity_imaginary_residue']:.3e}.",
            f"- Minimum interval-integrated activity: "
            f"{aggregate['invariant_audit']['minimum_integrated_interval_activity']:.3e}.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in aggregate["limitations"])
    lines.extend(
        [
            "",
            "Raw per-method/per-seed checkpoints and the machine-readable "
            "aggregate are stored in `results/primary_driven_activity/`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: Path, aggregate: dict, protocol: dict) -> None:
    atomic_write_text(report, render_report(aggregate, protocol))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run", "aggregate", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    protocol = build_protocol(args.archive.resolve())
    if args.command in ("run", "all"):
        run_jobs(args.outdir, protocol, args.workers)
    if args.command in ("aggregate", "all"):
        aggregate = aggregate_results(args.outdir, protocol)
    else:
        path = aggregate_path(args.outdir)
        if not path.exists():
            raise FileNotFoundError(f"aggregate missing: {path}")
        aggregate = json.loads(path.read_text())
        if aggregate.get("protocol_sha256") != protocol_sha256(protocol):
            raise RuntimeError("aggregate does not match the current frozen protocol")
    if args.command in ("report", "all"):
        write_report(args.report, aggregate, protocol)

    print(f"protocol: {protocol_path(args.outdir)}")
    if args.command in ("aggregate", "all", "report"):
        print(f"aggregate: {aggregate_path(args.outdir)}")
    if args.command in ("report", "all"):
        print(f"report: {args.report}")


if __name__ == "__main__":
    main()

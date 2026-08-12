"""Prospective activity-matched local-versus-collective STM response study.

The experiment is deliberately staged so that supervised task scores cannot
influence the operational matching convention:

1. ``pilot`` measures jump activity only on fixed local and collective rate
   branches using pilot-only reservoirs and label-free i.i.d. inputs.
2. ``freeze-targets`` checks every pilot curve for its prespecified monotone
   orientation and freezes five absolute activity targets in the common range.
3. ``calibrate`` matches every fresh reservoir/design/target cell on an
   independent label-free input stream.  Unreachable, non-monotone, or
   unconverged cells are recorded as censored.
4. ``freeze-calibration`` requires all 240 cells to be uncensored and within
   0.5% of target, then freezes their rates and hashes before any task score.
5. ``score`` resets every reservoir and evaluates STM on an independent input
   stream at the frozen rate.  It also integrates held-out test activity.
6. ``aggregate`` applies paired two-sided Bonferroni-t bands across exactly
   five STM and five test-activity contrasts.

No command before ``score`` constructs a target or fits a readout.  The v3
rate branches are fixed before its new pilot: local ``[0.05, 0.5]``
(containing the prior validation point 0.25) and collective ``[4, 32]``
(containing 8).  This is the single permitted response-blind recovery:

* the v1 activity-only development audit on collective ``[2,16]`` crossed a
  low-rate turnover and produced no target freeze, fresh calibration, readout
  fit, or task score; and
* the disjoint v2 attempt on collective ``[4,16]`` completed its label-free
  fresh calibration with five unreachable collective cells, so its mandatory
  calibration freeze failed and no task score was computed.

V3 doubles only the failed collective upper bound, uses new disjoint pilot and
task namespaces, and reruns the unchanged target-freeze and inference rules.
No second extension or manual target adjustment is permitted.  The preserved
audits are ``reports/activity_matched_response_development_audit.md`` and
``reports/activity_matched_response_v2_failure_audit.md``; the recovery rule
is ``reports/activity_matched_response_v2_recovery_plan.md``.

Examples
--------
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py pilot
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py freeze-targets
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py calibrate
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py freeze-calibration
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py score
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py aggregate
PYTHONPATH=src:experiments python experiments/run_activity_matched_response.py report
"""

from __future__ import annotations

import os

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import _paths  # noqa: F401
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import expm_multiply
from scipy.stats import t as student_t

from _paths import REPORTS_DIR, RESULTS_DIR
from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.liouvillian import unvec, vec
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import (
    SparseLindbladReservoir,
    commutator_super as sparse_commutator_super,
    dissipator_super as sparse_dissipator_super,
)


PROTOCOL_VERSION = "activity-matched-response-v3-2026-07-25"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = Path(RESULTS_DIR) / "activity_matched_response"
DEFAULT_REPORT = Path(REPORTS_DIR) / "activity_matched_response_report.md"

N_QUBITS = 5
H = 0.5
DT = 0.5
DESIGNS = ("local", "collective")
REFERENCE_DESIGN = "local"
BRANCHES = {
    "local": {
        "lower_rate": 0.05,
        "anchor_rate": 0.25,
        "upper_rate": 0.5,
        "activity_orientation": "increasing",
    },
    "collective": {
        "lower_rate": 4.0,
        "anchor_rate": 8.0,
        "upper_rate": 32.0,
        "activity_orientation": "decreasing",
    },
}
PILOT_RATE_GRIDS = {
    design: tuple(
        sorted(
            {
                *(
                    float(f"{value:.12g}")
                    for value in np.geomspace(
                        branch["lower_rate"], branch["upper_rate"], 9
                    )
                ),
                branch["anchor_rate"],
            }
        )
    )
    for design, branch in BRANCHES.items()
}

N_PILOT_SEEDS = 8
N_TASK_SEEDS = 24
N_ACTIVITY_TARGETS = 5
DEVELOPMENT_PILOT_NAMESPACE = 501
DEVELOPMENT_UNUSED_TASK_NAMESPACE = 502
V2_PILOT_NAMESPACE = 503
V2_TASK_NAMESPACE = 504
PILOT_NAMESPACE = 505
TASK_NAMESPACE = 506
CAL_WASH = 200
CAL_PREFIX = 600
CAL_MEASURE = 400
WASH = 200
TRAIN = 600
TEST = 400
DELAYS = tuple(range(1, 21))
FIXED_RIDGE = 1e-8

TARGET_LOG_INSET_FRACTION = 0.05
MIN_COMMON_ACTIVITY_SPAN_RATIO = 1.5
CALIBRATION_RELATIVE_TOL = 0.005
CALIBRATION_ABSOLUTE_TOL = 1e-5
BISECTION_MAX_ITERATIONS = 18
MONOTONIC_ABSOLUTE_TOL = 2e-7
MONOTONIC_RELATIVE_TOL = 2e-5
TRACE_TOL = 2e-9
IMAGINARY_TOL = 2e-9
NEGATIVE_COUNT_TOL = 2e-10
TEST_ACTIVITY_EQUIVALENCE_MARGIN = 0.05
SIMULTANEOUS_ALPHA = 0.05

SOURCE_FILES = (
    "experiments/run_activity_matched_response.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)


@dataclass(frozen=True)
class PilotJob:
    design: str
    seed: int
    rate: float


@dataclass(frozen=True)
class PilotBundleJob:
    design: str
    seed: int


@dataclass(frozen=True)
class CalibrationJob:
    design: str
    seed: int
    target_index: int
    target: float


@dataclass(frozen=True)
class CalibrationBundleJob:
    design: str
    seed: int


@dataclass(frozen=True)
class ScoreJob:
    design: str
    seed: int
    target_index: int
    target: float
    rate: float
    calibration_sha256: str


@dataclass(frozen=True)
class ScoreBundleJob:
    design: str
    seed: int


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


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
    dtype = "<c16" if np.iscomplexobj(array) else "<f8"
    canonical = np.ascontiguousarray(array, dtype=dtype)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def source_hashes() -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"scientific sources are missing: {missing}")
    return {name: file_sha256(ROOT / name) for name in SOURCE_FILES}


def _fresh_pool(
    namespace: int, count: int, excluded: set[int]
) -> list[int]:
    rng = np.random.default_rng(
        np.random.SeedSequence([2026, 7, 25, namespace])
    )
    values: list[int] = []
    used = set(excluded)
    while len(values) < count:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in used:
            values.append(candidate)
            used.add(candidate)
    return values


def _revision_tuning_pool(
    namespace: int, count: int, excluded: set[int]
) -> list[int]:
    rng = np.random.default_rng(
        np.random.SeedSequence([2026, 7, 23, namespace])
    )
    values: list[int] = []
    used = set(excluded)
    while len(values) < count:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in used:
            values.append(candidate)
            used.add(candidate)
    return values


def _revision_control_pool(namespace: int, count: int) -> list[int]:
    legacy = set(
        map(
            int,
            np.random.default_rng(2024).integers(0, 2**31 - 1, 64),
        )
    )
    rng = np.random.default_rng(namespace)
    values: list[int] = []
    while len(values) < count:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in legacy and candidate not in values:
            values.append(candidate)
    return values


def seed_ledger() -> dict:
    """Enumerate canonical prior pools and create disjoint pilot/task pools."""
    legacy = list(
        map(
            int,
            np.random.default_rng(2024).integers(0, 2**31 - 1, 256),
        )
    )
    excluded = set(legacy)
    screen = _revision_tuning_pool(101, 2, excluded)
    excluded.update(screen)
    selection = _revision_tuning_pool(102, 12, excluded)
    excluded.update(selection)
    nested_test = _revision_tuning_pool(103, 24, excluded)
    excluded.update(nested_test)
    interpolation = _revision_tuning_pool(104, 24, excluded)
    excluded.update(interpolation)
    nested_extension = _revision_tuning_pool(301, 24, excluded)
    excluded.update(nested_extension)
    parity = _revision_control_pool(2026072301, 16)
    normalized_scaling = _revision_control_pool(2026072302, 8)
    excluded.update(parity)
    excluded.update(normalized_scaling)
    development_pilot = _fresh_pool(
        DEVELOPMENT_PILOT_NAMESPACE, N_PILOT_SEEDS, excluded
    )
    excluded.update(development_pilot)
    development_unused_task = _fresh_pool(
        DEVELOPMENT_UNUSED_TASK_NAMESPACE, N_TASK_SEEDS, excluded
    )
    excluded.update(development_unused_task)
    v2_pilot = _fresh_pool(V2_PILOT_NAMESPACE, N_PILOT_SEEDS, excluded)
    excluded.update(v2_pilot)
    v2_task = _fresh_pool(V2_TASK_NAMESPACE, N_TASK_SEEDS, excluded)
    excluded.update(v2_task)

    prior_sources = {
        "definitive_2024_pool_256": legacy,
        "revision_nested_screen": screen,
        "revision_nested_selection": selection,
        "revision_nested_test": nested_test,
        "revision_fresh_interpolation": interpolation,
        "revision_nested_extension_test": nested_extension,
        "revision_parity_control": parity,
        "revision_normalized_scaling": normalized_scaling,
        "activity_branch_development_pilot": development_pilot,
        "activity_branch_development_unused_task_pool": (
            development_unused_task
        ),
        "activity_v2_pilot": v2_pilot,
        "activity_v2_task": v2_task,
    }
    pilot = _fresh_pool(PILOT_NAMESPACE, N_PILOT_SEEDS, excluded)
    task = _fresh_pool(TASK_NAMESPACE, N_TASK_SEEDS, excluded | set(pilot))
    if set(pilot) & excluded or set(task) & (excluded | set(pilot)):
        raise RuntimeError("new seed pools overlap a canonical prior pool")
    return {
        "prior_sources": {
            name: {
                "count": len(values),
                "sha256": sha256_json(values),
            }
            for name, values in prior_sources.items()
        },
        "prior_seed_count": len(excluded),
        "prior_seeds_sha256": sha256_json(sorted(excluded)),
        "prior_seeds": sorted(excluded),
        "pilot_namespace": PILOT_NAMESPACE,
        "pilot_seeds": pilot,
        "task_namespace": TASK_NAMESPACE,
        "task_seeds": task,
        "pilot_task_overlap": sorted(set(pilot) & set(task)),
        "pilot_prior_overlap": sorted(set(pilot) & excluded),
        "task_prior_overlap": sorted(set(task) & excluded),
    }


def build_pilot_protocol() -> dict:
    ledger = seed_ledger()
    hashes = source_hashes()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "activity_only_pilot",
        "status": "must_be_frozen_before_pilot_rows",
        "scientific_sources_sha256": hashes,
        "scientific_sources_combined_sha256": sha256_json(hashes),
        "source_snapshot_contract": {
            "manifest": (
                "results/activity_matched_response/"
                "source_snapshot/manifest.json"
            ),
            "driver_source": "experiments/run_activity_matched_response.py",
            "driver_snapshot": (
                "results/activity_matched_response/source_snapshot/"
                "run_activity_matched_response.py"
            ),
            "driver_sha256": hashes[
                "experiments/run_activity_matched_response.py"
            ],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "seed_ledger": ledger,
        "random_stream_namespace_contract": {
            "root": "SeedSequence([2026,7,25,reservoir_seed])",
            "spawn_0": "coupling draw",
            "spawn_1": "label-free calibration input",
            "spawn_2": "independent STM task input",
            "all_three_arrays_hash_linked_in_fresh_rows": True,
        },
        "branch_development_and_recovery_boundary": {
            "v1_development": {
                "audit_report": (
                    "reports/"
                    "activity_matched_response_development_audit.md"
                ),
                "artifact": (
                    "results/"
                    "activity_matched_response_exploratory_branch_audit"
                ),
                "collective_branch": [2.0, 16.0],
                "observed_boundary": (
                    "the activity-only development pilot crossed a "
                    "low-rate collective turnover"
                ),
                "created_frozen_targets": False,
                "created_fresh_calibration": False,
                "constructed_or_scored_task_targets": False,
            },
            "v2_failed_reachability": {
                "protocol_version": (
                    "activity-matched-response-v2-2026-07-25"
                ),
                "audit_report": (
                    "reports/"
                    "activity_matched_response_v2_failure_audit.md"
                ),
                "artifact": (
                    "results/"
                    "activity_matched_response_failed_v2_"
                    "reachability_audit"
                ),
                "local_branch": [0.05, 0.5],
                "collective_branch": [4.0, 16.0],
                "fresh_calibration_cells": 240,
                "matched_cells": 235,
                "censored_collective_cells": 5,
                "frozen_calibration_created": False,
                "task_score_checkpoints_created": 0,
                "task_scores_computed_or_inspected": False,
            },
            "v3_single_recovery": {
                "recovery_plan": (
                    "reports/"
                    "activity_matched_response_v2_recovery_plan.md"
                ),
                "local_branch": [0.05, 0.5],
                "collective_branch": [4.0, 32.0],
                "collective_upper_bound_extensions_permitted": 1,
                "manual_target_adjustment_permitted": False,
                "reuses_v2_pilot_or_task_seeds": False,
                "pilot_namespace": PILOT_NAMESPACE,
                "task_namespace": TASK_NAMESPACE,
                "pilot_and_task_namespaces_are_new": True,
                "target_freeze_and_inference_rules_unchanged": True,
                "further_recovery_after_gate_failure_permitted": False,
            },
        },
        "physics": {
            "N": N_QUBITS,
            "h": H,
            "dt": DT,
            "designs": list(DESIGNS),
            "branches": BRANCHES,
            "pilot_rate_grids": {
                design: list(values)
                for design, values in PILOT_RATE_GRIDS.items()
            },
        },
        "calibration_input": {
            "distribution": "iid Uniform[0,1]",
            "labels_or_task_targets_used": False,
            "wash_intervals": CAL_WASH,
            "unsupervised_prefix_intervals": CAL_PREFIX,
            "measured_intervals": CAL_MEASURE,
            "activity_definition": (
                "(CAL_MEASURE*dt)^-1 sum measured intervals "
                "integral Tr[K rho(t)] dt, K=sum_k rate_k L_k^dagger L_k"
            ),
            "continuous_interval_integration": "augmented expm_multiply",
        },
        "target_freeze_rule": {
            "required_monotone_orientations": {
                design: BRANCHES[design]["activity_orientation"]
                for design in DESIGNS
            },
            "common_interval": (
                "intersection of endpoint-reachable intervals over every "
                "pilot seed and both designs"
            ),
            "minimum_common_span_ratio": MIN_COMMON_ACTIVITY_SPAN_RATIO,
            "log_inset_fraction_each_side": TARGET_LOG_INSET_FRACTION,
            "target_spacing": "five geometric targets in inset interval",
            "n_targets": N_ACTIVITY_TARGETS,
        },
        "expected_pilot_rows": int(
            sum(
                len(ledger["pilot_seeds"]) * len(PILOT_RATE_GRIDS[design])
                for design in DESIGNS
            )
        ),
        "supervised_boundary": {
            "constructs_task_targets": False,
            "fits_readout": False,
            "scores_task": False,
        },
    }


def pilot_manifest_path(outdir: Path) -> Path:
    return outdir / "pilot_manifest.json"


def frozen_targets_path(outdir: Path) -> Path:
    return outdir / "frozen_targets.json"


def task_manifest_path(outdir: Path) -> Path:
    return outdir / "task_manifest.json"


def frozen_calibration_path(outdir: Path) -> Path:
    return outdir / "frozen_calibration.json"


def aggregate_path(outdir: Path) -> Path:
    return outdir / "aggregate.json"


def source_snapshot_manifest_path(outdir: Path) -> Path:
    return outdir / "source_snapshot" / "manifest.json"


def ensure_source_snapshot(
    outdir: Path,
    pilot_protocol_sha256: str,
    hashes: dict[str, str],
) -> dict:
    snapshot_dir = outdir / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments" / "run_activity_matched_response.py"
    snapshot = snapshot_dir / "run_activity_matched_response.py"
    expected_driver_sha = hashes[
        "experiments/run_activity_matched_response.py"
    ]
    if snapshot.exists():
        if file_sha256(snapshot) != expected_driver_sha:
            raise RuntimeError("activity driver source snapshot drift")
    else:
        shutil.copyfile(source, snapshot)
    payload = {
        "artifact_type": "activity_matched_source_snapshot",
        "pilot_protocol_sha256": pilot_protocol_sha256,
        "source_path": "experiments/run_activity_matched_response.py",
        "snapshot_path": (
            "results/activity_matched_response/source_snapshot/"
            "run_activity_matched_response.py"
        ),
        "sha256": expected_driver_sha,
        "all_scientific_source_hashes": hashes,
        "all_scientific_source_hashes_sha256": sha256_json(hashes),
    }
    path = source_snapshot_manifest_path(outdir)
    if path.exists():
        old = json.loads(path.read_text())
        if canonical_json(old) != canonical_json(payload):
            raise RuntimeError("activity source-snapshot manifest drift")
    else:
        atomic_json(path, payload)
    return payload


def _rate_tag(rate: float) -> str:
    return f"{rate:.12g}".replace(".", "p").replace("-", "m")


def pilot_checkpoint_path(outdir: Path, job: PilotJob) -> Path:
    return (
        outdir
        / "pilot"
        / "checkpoints"
        / f"{job.design}__seed_{job.seed}__rate_{_rate_tag(job.rate)}.json"
    )


def calibration_checkpoint_path(
    outdir: Path, job: CalibrationJob
) -> Path:
    return (
        outdir
        / "calibration"
        / "checkpoints"
        / (
            f"{job.design}__seed_{job.seed}"
            f"__target_{job.target_index:02d}.json"
        )
    )


def score_checkpoint_path(outdir: Path, job: ScoreJob) -> Path:
    return (
        outdir
        / "score"
        / "checkpoints"
        / (
            f"{job.design}__seed_{job.seed}"
            f"__target_{job.target_index:02d}.json"
        )
    )


def ensure_pilot_manifest(outdir: Path) -> tuple[dict, str]:
    protocol = build_pilot_protocol()
    digest = sha256_json(protocol)
    payload = {
        "artifact_type": "activity_matched_pilot_manifest",
        "manifest_status": "frozen_before_pilot_rows",
        "protocol": protocol,
        "protocol_sha256": digest,
    }
    path = pilot_manifest_path(outdir)
    if path.exists():
        old = json.loads(path.read_text())
        if canonical_json(old) != canonical_json(payload):
            raise RuntimeError("pilot manifest drift")
    else:
        rows = outdir / "pilot" / "checkpoints"
        if rows.exists() and any(rows.glob("*.json")):
            raise RuntimeError("pilot rows exist before the pilot manifest")
        atomic_json(path, payload)
    ensure_source_snapshot(
        outdir, digest, protocol["scientific_sources_sha256"]
    )
    return payload, digest


def _stream_material(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    children = np.random.SeedSequence([2026, 7, 25, int(seed)]).spawn(3)
    coupling_rng = np.random.default_rng(children[0])
    calibration_rng = np.random.default_rng(children[1])
    task_rng = np.random.default_rng(children[2])
    couplings = res.random_couplings(N_QUBITS, 1.0, coupling_rng)
    calibration_inputs = tasks.stm_inputs(
        CAL_WASH + CAL_PREFIX + CAL_MEASURE, calibration_rng
    )
    task_inputs = tasks.stm_inputs(WASH + TRAIN + TEST, task_rng)
    return couplings, calibration_inputs, task_inputs


def build_jumps(design: str, rate: float):
    if rate <= 0:
        raise ValueError("rate must be positive")
    if design == "local":
        return dsp.local_loss(N_QUBITS, rate)
    if design == "collective":
        return dsp.collective_loss(N_QUBITS, rate)
    raise ValueError(f"unknown design: {design}")


def build_reservoir(
    couplings: np.ndarray, design: str, rate: float
) -> tuple[SparseLindbladReservoir, list[tuple[np.ndarray, float]]]:
    jumps = build_jumps(design, rate)
    h0 = ising_xx_hamiltonian(couplings, H, N_QUBITS)
    hx = transverse_drive(N_QUBITS)
    reservoir = SparseLindbladReservoir.from_terms(
        N_QUBITS, h0 + H * hx, H * hx, jumps, DT
    )
    return reservoir, jumps


class AffineActivityEngine:
    """Cache the Hamiltonian, drive, and unit-rate dissipative affine pieces."""

    def __init__(self, couplings: np.ndarray, design: str):
        self.design = design
        h0 = ising_xx_hamiltonian(couplings, H, N_QUBITS)
        hx = transverse_drive(N_QUBITS)
        self.hamiltonian_base = sparse_commutator_super(h0 + H * hx)
        self.drive = sparse_commutator_super(H * hx)
        self.unit_jumps = build_jumps(design, 1.0)
        self.dissipation = sparse.csr_matrix(
            self.hamiltonian_base.shape, dtype=complex
        )
        for jump, unit_rate in self.unit_jumps:
            self.dissipation = self.dissipation + float(
                unit_rate
            ) * sparse_dissipator_super(jump)
        self.dissipation.eliminate_zeros()
        self.dissipation = self.dissipation.tocsr()
        self.unit_rate_operator = jump_rate_operator(self.unit_jumps)
        self.unit_functional = activity_functional(self.unit_rate_operator)
        self.dimension = self.unit_rate_operator.shape[0]
        self.super_dimension = self.dimension**2
        self.unit_jump_strength = float(dsp.jump_strength(self.unit_jumps))
        self._zero_column = sparse.csr_matrix(
            (self.super_dimension, 1), dtype=complex
        )
        self._zero_scalar = sparse.csr_matrix((1, 1), dtype=complex)
        self.augmented_drive = sparse.bmat(
            [
                [self.drive, self._zero_column],
                [
                    sparse.csr_matrix(
                        (1, self.super_dimension), dtype=complex
                    ),
                    self._zero_scalar,
                ],
            ],
            format="csr",
        )

    def initial_state_vector(self) -> np.ndarray:
        rho = np.zeros((self.dimension, self.dimension), dtype=complex)
        rho[0, 0] = 1.0
        return vec(rho)

    def affine_terms(
        self, rate: float
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        base = (
            self.hamiltonian_base + float(rate) * self.dissipation
        ).tocsr()
        augmented_base = sparse.bmat(
            [
                [base, self._zero_column],
                [
                    sparse.csr_matrix(
                        (float(rate) * self.unit_functional).reshape(
                            1, self.super_dimension
                        )
                    ),
                    self._zero_scalar,
                ],
            ],
            format="csr",
        )
        return base, augmented_base

    def calibration_activity(
        self, inputs: np.ndarray, rate: float
    ) -> dict:
        base, augmented_base = self.affine_terms(rate)
        state = self.initial_state_vector()
        maximum_trace_error = 0.0
        measured_start = CAL_WASH + CAL_PREFIX
        for value in inputs[:measured_start]:
            state = expm_multiply(
                (base + float(value) * self.drive) * DT, state
            )
            maximum_trace_error = max(
                maximum_trace_error,
                abs(_trace_from_vector(state, self.dimension) - 1.0),
            )
        augmented_state = np.empty(self.super_dimension + 1, dtype=complex)
        augmented_state[:-1] = state
        augmented_state[-1] = 0.0
        counts: list[float] = []
        previous_count = 0.0
        maximum_imaginary = 0.0
        for value in inputs[measured_start:]:
            augmented_state = expm_multiply(
                (
                    augmented_base
                    + float(value) * self.augmented_drive
                )
                * DT,
                augmented_state,
            )
            cumulative = float(np.real(augmented_state[-1]))
            count = cumulative - previous_count
            if count < -NEGATIVE_COUNT_TOL:
                raise RuntimeError(
                    f"negative integrated calibration activity: {count}"
                )
            counts.append(max(count, 0.0))
            previous_count = cumulative
            maximum_imaginary = max(
                maximum_imaginary, float(abs(np.imag(augmented_state[-1])))
            )
            maximum_trace_error = max(
                maximum_trace_error,
                abs(
                    _trace_from_vector(
                        augmented_state[:-1], self.dimension
                    )
                    - 1.0
                ),
            )
        if len(counts) != CAL_MEASURE:
            raise RuntimeError("calibration interval coverage failed")
        if maximum_trace_error > TRACE_TOL:
            raise RuntimeError("calibration trace tolerance failed")
        if maximum_imaginary > IMAGINARY_TOL:
            raise RuntimeError("calibration imaginary tolerance failed")
        counts_array = np.asarray(counts, dtype=float)
        return {
            "activity": float(
                np.sum(counts_array) / (CAL_MEASURE * DT)
            ),
            "total_expected_jumps": float(np.sum(counts_array)),
            "maximum_trace_error": float(maximum_trace_error),
            "maximum_activity_imaginary_residue": float(maximum_imaginary),
            "minimum_interval_integrated_activity": float(
                np.min(counts_array)
            ),
            "jump_strength": float(rate * self.unit_jump_strength),
        }

    def task_trajectory(
        self, inputs: np.ndarray, rate: float
    ) -> tuple[np.ndarray, dict]:
        base, augmented_base = self.affine_terms(rate)
        observables = readout.pauli_observables(
            N_QUBITS, max_weight=2
        )
        observable_matrices = np.stack(
            [item.matrix for item in observables]
        )
        features = np.empty(
            (TRAIN + TEST, len(observables)), dtype=float
        )
        state = self.initial_state_vector()
        maximum_trace_error = 0.0
        for index, value in enumerate(inputs[: WASH + TRAIN]):
            state = expm_multiply(
                (base + float(value) * self.drive) * DT, state
            )
            maximum_trace_error = max(
                maximum_trace_error,
                abs(_trace_from_vector(state, self.dimension) - 1.0),
            )
            if index >= WASH:
                rho = unvec(state, self.dimension)
                features[index - WASH] = np.real(
                    np.einsum("kij,ji->k", observable_matrices, rho)
                )

        augmented_state = np.empty(self.super_dimension + 1, dtype=complex)
        augmented_state[:-1] = state
        augmented_state[-1] = 0.0
        counts: list[float] = []
        previous_count = 0.0
        maximum_imaginary = 0.0
        for offset, value in enumerate(inputs[WASH + TRAIN :]):
            augmented_state = expm_multiply(
                (
                    augmented_base
                    + float(value) * self.augmented_drive
                )
                * DT,
                augmented_state,
            )
            cumulative = float(np.real(augmented_state[-1]))
            count = cumulative - previous_count
            if count < -NEGATIVE_COUNT_TOL:
                raise RuntimeError(
                    f"negative integrated task activity: {count}"
                )
            counts.append(max(count, 0.0))
            previous_count = cumulative
            maximum_imaginary = max(
                maximum_imaginary, float(abs(np.imag(augmented_state[-1])))
            )
            maximum_trace_error = max(
                maximum_trace_error,
                abs(
                    _trace_from_vector(
                        augmented_state[:-1], self.dimension
                    )
                    - 1.0
                ),
            )
            rho = unvec(augmented_state[:-1], self.dimension)
            features[TRAIN + offset] = np.real(
                np.einsum("kij,ji->k", observable_matrices, rho)
            )
        if len(counts) != TEST:
            raise RuntimeError("test activity coverage failed")
        if maximum_trace_error > TRACE_TOL:
            raise RuntimeError("task trajectory trace tolerance failed")
        if maximum_imaginary > IMAGINARY_TOL:
            raise RuntimeError("task activity imaginary tolerance failed")
        counts_array = np.asarray(counts, dtype=float)
        return features, {
            "time_averaged_test_activity": float(
                np.sum(counts_array) / (TEST * DT)
            ),
            "total_expected_test_jumps": float(np.sum(counts_array)),
            "maximum_trace_error": float(maximum_trace_error),
            "maximum_activity_imaginary_residue": float(maximum_imaginary),
            "minimum_test_interval_integrated_activity": float(
                np.min(counts_array)
            ),
        }


def jump_rate_operator(
    jumps: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray:
    if not jumps:
        raise ValueError("at least one jump is required")
    dimension = np.asarray(jumps[0][0]).shape[0]
    operator = np.zeros((dimension, dimension), dtype=complex)
    for jump, rate in jumps:
        matrix = np.asarray(jump, dtype=complex)
        operator += float(rate) * matrix.conj().T @ matrix
    return operator


def activity_functional(rate_operator: np.ndarray) -> np.ndarray:
    return vec(np.asarray(rate_operator, dtype=complex).T)


def integrated_activity_step(
    generator: sparse.spmatrix,
    state_vector: np.ndarray,
    rate_functional: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float, float]:
    generator = sparse.csr_matrix(generator, dtype=complex)
    size = generator.shape[0]
    if generator.shape != (size, size) or state_vector.shape != (size,):
        raise ValueError("generator and state dimensions disagree")
    functional = np.asarray(rate_functional, dtype=complex)
    if functional.shape != (size,):
        raise ValueError("activity functional dimension disagrees")
    augmented = sparse.bmat(
        [
            [generator, sparse.csr_matrix((size, 1), dtype=complex)],
            [
                sparse.csr_matrix(functional.reshape(1, size)),
                sparse.csr_matrix((1, 1), dtype=complex),
            ],
        ],
        format="csr",
    )
    initial = np.empty(size + 1, dtype=complex)
    initial[:-1] = state_vector
    initial[-1] = 0.0
    evolved = expm_multiply(augmented * float(dt), initial)
    residue = float(abs(np.imag(evolved[-1])))
    count = float(np.real(evolved[-1]))
    if count < -NEGATIVE_COUNT_TOL:
        raise RuntimeError(f"negative integrated activity: {count}")
    return evolved[:-1], max(count, 0.0), residue


def _trace_from_vector(state: np.ndarray, dimension: int) -> complex:
    return complex(activity_functional(np.eye(dimension)) @ state)


def calibration_activity(
    couplings: np.ndarray,
    inputs: np.ndarray,
    design: str,
    rate: float,
) -> dict:
    return AffineActivityEngine(couplings, design).calibration_activity(
        inputs, rate
    )


def run_pilot_job(job: PilotJob, protocol_sha256: str) -> dict:
    started = time.perf_counter()
    couplings, calibration_inputs, _ = _stream_material(job.seed)
    measurement = AffineActivityEngine(
        couplings, job.design
    ).calibration_activity(calibration_inputs, job.rate)
    return {
        "artifact_type": "activity_only_pilot_row",
        "protocol_version": PROTOCOL_VERSION,
        "pilot_protocol_sha256": protocol_sha256,
        "design": job.design,
        "seed": job.seed,
        "rate": job.rate,
        "branch": BRANCHES[job.design],
        "couplings_sha256": array_sha256(couplings),
        "calibration_input_sha256": array_sha256(calibration_inputs),
        **measurement,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _valid_pilot_checkpoint(
    path: Path, job: PilotJob, protocol_sha256: str
) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    expected = {
        "design": job.design,
        "seed": job.seed,
        "rate": job.rate,
        "pilot_protocol_sha256": protocol_sha256,
    }
    if all(row.get(key) == value for key, value in expected.items()):
        return row
    raise RuntimeError(f"stale or mismatched pilot checkpoint: {path}")


def _pilot_bundle_worker(
    bundle: PilotBundleJob, protocol_sha256: str, outdir_text: str
) -> str:
    outdir = Path(outdir_text)
    jobs = [
        PilotJob(bundle.design, bundle.seed, rate)
        for rate in PILOT_RATE_GRIDS[bundle.design]
    ]
    missing = [
        job
        for job in jobs
        if _valid_pilot_checkpoint(
            pilot_checkpoint_path(outdir, job), job, protocol_sha256
        )
        is None
    ]
    if not missing:
        return f"skip pilot bundle {bundle.design}/{bundle.seed}"
    couplings, calibration_inputs, _ = _stream_material(bundle.seed)
    engine = AffineActivityEngine(couplings, bundle.design)
    for job in missing:
        started = time.perf_counter()
        measurement = engine.calibration_activity(
            calibration_inputs, job.rate
        )
        row = {
            "artifact_type": "activity_only_pilot_row",
            "protocol_version": PROTOCOL_VERSION,
            "pilot_protocol_sha256": protocol_sha256,
            "design": job.design,
            "seed": job.seed,
            "rate": job.rate,
            "branch": BRANCHES[job.design],
            "couplings_sha256": array_sha256(couplings),
            "calibration_input_sha256": array_sha256(calibration_inputs),
            **measurement,
            "runtime_seconds": float(time.perf_counter() - started),
        }
        atomic_json(pilot_checkpoint_path(outdir, job), row)
    return f"done pilot bundle {bundle.design}/{bundle.seed}"


def _run_parallel(
    worker: Callable[..., str],
    jobs: Sequence[object],
    common_args: tuple,
    workers: int,
) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must lie in [1,8]")
    if workers == 1:
        for job in jobs:
            print(worker(job, *common_args), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker, job, *common_args): job for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                raise RuntimeError(f"worker failed for {job}") from exc


def run_pilot(outdir: Path, workers: int) -> None:
    manifest, digest = ensure_pilot_manifest(outdir)
    seeds = manifest["protocol"]["seed_ledger"]["pilot_seeds"]
    jobs = [
        PilotBundleJob(design, seed)
        for design in DESIGNS
        for seed in seeds
    ]
    _run_parallel(
        _pilot_bundle_worker, jobs, (digest, str(outdir)), workers
    )


def load_pilot_rows(
    outdir: Path, manifest: dict, protocol_sha256: str
) -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    seeds = manifest["protocol"]["seed_ledger"]["pilot_seeds"]
    for design in DESIGNS:
        for seed in seeds:
            for rate in PILOT_RATE_GRIDS[design]:
                job = PilotJob(design, seed, rate)
                row = _valid_pilot_checkpoint(
                    pilot_checkpoint_path(outdir, job), job, protocol_sha256
                )
                if row is None:
                    missing.append(f"{design}/{seed}/{rate:g}")
                else:
                    rows.append(row)
    if missing:
        raise RuntimeError(
            f"pilot incomplete: {len(missing)} rows missing; first={missing[:5]}"
        )
    return rows


def _monotonicity_audit(
    rates: np.ndarray, activities: np.ndarray, orientation: str
) -> dict:
    order = np.argsort(rates)
    rates = np.asarray(rates, dtype=float)[order]
    activities = np.asarray(activities, dtype=float)[order]
    sign = 1.0 if orientation == "increasing" else -1.0
    signed_differences = sign * np.diff(activities)
    tolerance = MONOTONIC_ABSOLUTE_TOL + MONOTONIC_RELATIVE_TOL * float(
        np.max(np.abs(activities))
    )
    passed = bool(np.all(signed_differences > -tolerance))
    return {
        "passed": passed,
        "orientation": orientation,
        "minimum_oriented_increment": float(np.min(signed_differences)),
        "tolerance": float(tolerance),
        "rates": rates.tolist(),
        "activities": activities.tolist(),
    }


def derive_frozen_targets(
    rows: Sequence[dict], pilot_protocol_sha256: str
) -> dict:
    curves: list[dict] = []
    reachable: list[tuple[float, float]] = []
    identities = set()
    for design in DESIGNS:
        seeds = sorted(
            {int(row["seed"]) for row in rows if row["design"] == design}
        )
        for seed in seeds:
            group = [
                row
                for row in rows
                if row["design"] == design and int(row["seed"]) == seed
            ]
            rates = np.asarray([row["rate"] for row in group], dtype=float)
            activities = np.asarray(
                [row["activity"] for row in group], dtype=float
            )
            if len(group) != len(PILOT_RATE_GRIDS[design]):
                raise RuntimeError(f"incomplete pilot curve {design}/{seed}")
            audit = _monotonicity_audit(
                rates,
                activities,
                BRANCHES[design]["activity_orientation"],
            )
            if not audit["passed"]:
                raise RuntimeError(
                    f"pilot branch is not monotone: {design}/{seed}"
                )
            low = float(min(audit["activities"][0], audit["activities"][-1]))
            high = float(max(audit["activities"][0], audit["activities"][-1]))
            if low <= 0 or high <= low:
                raise RuntimeError(f"invalid pilot reachability {design}/{seed}")
            reachable.append((low, high))
            identities.add((design, seed))
            curves.append(
                {
                    "design": design,
                    "seed": seed,
                    "reachable_interval": [low, high],
                    "monotonicity": audit,
                }
            )
    expected_identities = {
        (design, seed)
        for design in DESIGNS
        for seed in seed_ledger()["pilot_seeds"]
    }
    if identities != expected_identities:
        raise RuntimeError("pilot curve identity coverage failed")
    common_low = max(low for low, _ in reachable)
    common_high = min(high for _, high in reachable)
    if common_high <= common_low:
        raise RuntimeError("pilot curves have no common activity interval")
    span_ratio = common_high / common_low
    if span_ratio < MIN_COMMON_ACTIVITY_SPAN_RATIO:
        raise RuntimeError(
            f"common activity span {span_ratio:.6g} is below "
            f"{MIN_COMMON_ACTIVITY_SPAN_RATIO:g}"
        )
    log_low = math.log(common_low)
    log_high = math.log(common_high)
    log_span = log_high - log_low
    target_low = math.exp(
        log_low + TARGET_LOG_INSET_FRACTION * log_span
    )
    target_high = math.exp(
        log_high - TARGET_LOG_INSET_FRACTION * log_span
    )
    targets = np.geomspace(
        target_low, target_high, N_ACTIVITY_TARGETS
    ).tolist()
    pilot_row_index = sorted(
        {
            (
                row["design"],
                int(row["seed"]),
                float(row["rate"]),
                sha256_json(row),
            )
            for row in rows
        }
    )
    return {
        "artifact_type": "frozen_activity_targets",
        "freeze_status": "frozen_before_fresh_calibration_or_task_scores",
        "pilot_protocol_sha256": pilot_protocol_sha256,
        "pilot_rows_sha256": sha256_json(pilot_row_index),
        "uses_supervised_task_information": False,
        "common_activity_interval": [common_low, common_high],
        "common_activity_span_ratio": span_ratio,
        "target_interval_after_log_inset": [target_low, target_high],
        "targets": targets,
        "n_targets": len(targets),
        "pilot_curve_audits": curves,
    }


def freeze_targets(outdir: Path) -> dict:
    manifest, digest = ensure_pilot_manifest(outdir)
    rows = load_pilot_rows(outdir, manifest, digest)
    payload = derive_frozen_targets(rows, digest)
    task_artifacts = [
        task_manifest_path(outdir),
        frozen_calibration_path(outdir),
        outdir / "calibration" / "checkpoints",
        outdir / "score" / "checkpoints",
    ]
    path = frozen_targets_path(outdir)
    if path.exists():
        old = json.loads(path.read_text())
        if canonical_json(old) != canonical_json(payload):
            raise RuntimeError("frozen activity targets drift")
    else:
        if any(
            artifact.exists()
            and (
                artifact.is_file()
                or any(artifact.glob("*.json"))
            )
            for artifact in task_artifacts
        ):
            raise RuntimeError("fresh artifacts exist before target freeze")
        atomic_json(path, payload)
    return payload


def load_frozen_targets(outdir: Path) -> dict:
    path = frozen_targets_path(outdir)
    if not path.is_file():
        raise RuntimeError("activity targets are not frozen")
    payload = json.loads(path.read_text())
    manifest, digest = ensure_pilot_manifest(outdir)
    if payload.get("pilot_protocol_sha256") != digest:
        raise RuntimeError("frozen targets do not match pilot protocol")
    if len(payload.get("targets", [])) != N_ACTIVITY_TARGETS:
        raise RuntimeError("frozen target count changed")
    if (
        manifest["protocol"]["scientific_sources_sha256"]
        != source_hashes()
    ):
        raise RuntimeError("scientific source drift after pilot freeze")
    return payload


def build_task_protocol(outdir: Path, frozen: dict) -> dict:
    ledger = seed_ledger()
    hashes = source_hashes()
    targets = list(map(float, frozen["targets"]))
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "fresh_calibration_then_frozen_task_scoring",
        "status": "must_be_frozen_before_fresh_calibration_rows",
        "scientific_sources_sha256": hashes,
        "scientific_sources_combined_sha256": sha256_json(hashes),
        "frozen_targets": {
            "path": str(frozen_targets_path(outdir).relative_to(ROOT)),
            "sha256": file_sha256(frozen_targets_path(outdir)),
            "pilot_protocol_sha256": frozen["pilot_protocol_sha256"],
            "targets": targets,
        },
        "seed_ledger": ledger,
        "fresh_boundary": {
            "task_seed_count": len(ledger["task_seeds"]),
            "task_seeds": ledger["task_seeds"],
            "task_prior_overlap": ledger["task_prior_overlap"],
            "task_pilot_overlap": ledger["pilot_task_overlap"],
            "same_couplings_and_streams_across_designs_and_targets": True,
            "calibration_and_task_streams_independent": True,
        },
        "calibration": {
            "branches": BRANCHES,
            "input_split": {
                "wash": CAL_WASH,
                "unsupervised_prefix": CAL_PREFIX,
                "measured_activity": CAL_MEASURE,
            },
            "target_count": len(targets),
            "expected_cells": len(ledger["task_seeds"])
            * len(DESIGNS)
            * len(targets),
            "bisection_scale": "geometric rate midpoint",
            "maximum_iterations": BISECTION_MAX_ITERATIONS,
            "relative_match_tolerance": CALIBRATION_RELATIVE_TOL,
            "absolute_match_tolerance": CALIBRATION_ABSOLUTE_TOL,
            "censored_statuses": [
                "censored_target_unreachable",
                "censored_branch_nonmonotone",
                "censored_nonconvergence",
            ],
            "freeze_gate": (
                "all cells uncensored and relative target error <=0.5%; "
                "all rates/activities/hashes frozen before task scoring"
            ),
        },
        "task": {
            "task_name": "STM",
            "N": N_QUBITS,
            "h": H,
            "dt": DT,
            "wash": WASH,
            "train": TRAIN,
            "test": TEST,
            "delays": list(DELAYS),
            "ridge": FIXED_RIDGE,
            "ridge_selection": "none; fixed before all fresh task scores",
            "test_evaluations_per_cell": 1,
            "test_activity": (
                "integrated descriptively through all held-out intervals"
            ),
        },
        "inference": {
            "unit": "paired reservoir seed",
            "family_size": N_ACTIVITY_TARGETS,
            "method": (
                "two-sided Bonferroni-t simultaneous 95% bands over exactly "
                "five prespecified contrasts"
            ),
            "critical_quantile": (
                "t_(23, 1-0.05/(2*5)) = t_(23,0.995)"
            ),
            "dominance_gate": (
                "zero censored cells and all five STM lower bounds >0"
            ),
            "test_activity_equivalence_gate": (
                "all simultaneous bands for (collective-local)/target lie "
                "inside [-0.05,0.05]"
            ),
        },
    }


def ensure_task_manifest(outdir: Path) -> tuple[dict, str, dict]:
    frozen = load_frozen_targets(outdir)
    protocol = build_task_protocol(outdir, frozen)
    digest = sha256_json(protocol)
    payload = {
        "artifact_type": "activity_matched_task_manifest",
        "manifest_status": "frozen_before_fresh_calibration_rows",
        "protocol": protocol,
        "protocol_sha256": digest,
    }
    path = task_manifest_path(outdir)
    if path.exists():
        old = json.loads(path.read_text())
        if canonical_json(old) != canonical_json(payload):
            raise RuntimeError("task manifest drift")
    else:
        calibration_dir = outdir / "calibration" / "checkpoints"
        score_dir = outdir / "score" / "checkpoints"
        if (
            calibration_dir.exists()
            and any(calibration_dir.glob("*.json"))
        ) or (score_dir.exists() and any(score_dir.glob("*.json"))):
            raise RuntimeError("fresh rows exist before task manifest")
        atomic_json(path, payload)
    return payload, digest, frozen


def _orientation_sign(orientation: str) -> float:
    if orientation == "increasing":
        return 1.0
    if orientation == "decreasing":
        return -1.0
    raise ValueError(f"unknown orientation: {orientation}")


def bisect_activity_target(
    evaluator: Callable[[float], float],
    target: float,
    lower_rate: float,
    upper_rate: float,
    orientation: str,
    *,
    relative_tolerance: float = CALIBRATION_RELATIVE_TOL,
    absolute_tolerance: float = CALIBRATION_ABSOLUTE_TOL,
    maximum_iterations: int = BISECTION_MAX_ITERATIONS,
) -> dict:
    """Match one target inside one predeclared monotone rate branch."""
    cache: dict[float, float] = {}

    def evaluate(rate: float) -> float:
        key = float(rate)
        if key not in cache:
            value = float(evaluator(key))
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("activity evaluator returned an invalid value")
            cache[key] = value
        return cache[key]

    low_rate = float(lower_rate)
    high_rate = float(upper_rate)
    low_activity = evaluate(low_rate)
    high_activity = evaluate(high_rate)
    sign = _orientation_sign(orientation)
    monotonic_tolerance = MONOTONIC_ABSOLUTE_TOL + MONOTONIC_RELATIVE_TOL * max(
        abs(low_activity), abs(high_activity)
    )
    if sign * (high_activity - low_activity) < -monotonic_tolerance:
        return {
            "status": "censored_branch_nonmonotone",
            "evaluations": [
                {"rate": rate, "activity": activity}
                for rate, activity in sorted(cache.items())
            ],
        }
    reach_low = min(low_activity, high_activity)
    reach_high = max(low_activity, high_activity)
    bracket_tolerance = max(
        absolute_tolerance, relative_tolerance * abs(target)
    )
    if target < reach_low - bracket_tolerance or target > reach_high + bracket_tolerance:
        return {
            "status": "censored_target_unreachable",
            "reachable_interval": [reach_low, reach_high],
            "evaluations": [
                {"rate": rate, "activity": activity}
                for rate, activity in sorted(cache.items())
            ],
        }

    best_rate, best_activity = min(
        cache.items(), key=lambda item: abs(item[1] - target)
    )
    for iteration in range(maximum_iterations + 1):
        tolerance = max(absolute_tolerance, relative_tolerance * abs(target))
        if abs(best_activity - target) <= tolerance:
            ordered = sorted(cache.items())
            audit = _monotonicity_audit(
                np.asarray([item[0] for item in ordered]),
                np.asarray([item[1] for item in ordered]),
                orientation,
            )
            if not audit["passed"]:
                return {
                    "status": "censored_branch_nonmonotone",
                    "evaluations": [
                        {"rate": rate, "activity": activity}
                        for rate, activity in ordered
                    ],
                    "monotonicity": audit,
                }
            return {
                "status": "matched",
                "matched_rate": float(best_rate),
                "matched_activity": float(best_activity),
                "absolute_error": float(abs(best_activity - target)),
                "relative_error": float(abs(best_activity - target) / target),
                "iterations": iteration,
                "reachable_interval": [reach_low, reach_high],
                "evaluations": [
                    {"rate": rate, "activity": activity}
                    for rate, activity in ordered
                ],
                "monotonicity": audit,
            }
        middle_rate = math.sqrt(low_rate * high_rate)
        middle_activity = evaluate(middle_rate)
        if abs(middle_activity - target) < abs(best_activity - target):
            best_rate, best_activity = middle_rate, middle_activity
        oriented_middle = sign * middle_activity
        oriented_target = sign * target
        if oriented_middle < oriented_target:
            low_rate = middle_rate
        else:
            high_rate = middle_rate
    return {
        "status": "censored_nonconvergence",
        "matched_rate": float(best_rate),
        "matched_activity": float(best_activity),
        "absolute_error": float(abs(best_activity - target)),
        "relative_error": float(abs(best_activity - target) / target),
        "evaluations": [
            {"rate": rate, "activity": activity}
            for rate, activity in sorted(cache.items())
        ],
    }


def run_calibration_job(
    job: CalibrationJob, task_protocol_sha256: str
) -> dict:
    started = time.perf_counter()
    couplings, calibration_inputs, _ = _stream_material(job.seed)
    engine = AffineActivityEngine(couplings, job.design)
    measurements: dict[float, dict] = {}

    def evaluator(rate: float) -> float:
        measurement = engine.calibration_activity(calibration_inputs, rate)
        measurements[float(rate)] = measurement
        return float(measurement["activity"])

    branch = BRANCHES[job.design]
    solution = bisect_activity_target(
        evaluator,
        job.target,
        branch["lower_rate"],
        branch["upper_rate"],
        branch["activity_orientation"],
    )
    evaluations = []
    for entry in solution.get("evaluations", []):
        rate = float(entry["rate"])
        measurement = measurements[rate]
        evaluations.append(
            {
                **entry,
                "jump_strength": measurement["jump_strength"],
                "maximum_trace_error": measurement["maximum_trace_error"],
                "maximum_activity_imaginary_residue": measurement[
                    "maximum_activity_imaginary_residue"
                ],
            }
        )
    solution["evaluations"] = evaluations
    return {
        "artifact_type": "fresh_activity_calibration_row",
        "protocol_version": PROTOCOL_VERSION,
        "task_protocol_sha256": task_protocol_sha256,
        "design": job.design,
        "seed": job.seed,
        "target_index": job.target_index,
        "target_activity": job.target,
        "branch": branch,
        "couplings_sha256": array_sha256(couplings),
        "calibration_input_sha256": array_sha256(calibration_inputs),
        "calibration_uses_task_targets_or_scores": False,
        **solution,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _valid_calibration_checkpoint(
    path: Path, job: CalibrationJob, task_protocol_sha256: str
) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    expected = {
        "design": job.design,
        "seed": job.seed,
        "target_index": job.target_index,
        "target_activity": job.target,
        "task_protocol_sha256": task_protocol_sha256,
    }
    if all(row.get(key) == value for key, value in expected.items()):
        return row
    raise RuntimeError(f"stale calibration checkpoint: {path}")


def _calibration_bundle_worker(
    bundle: CalibrationBundleJob,
    task_protocol_sha256: str,
    targets: Sequence[float],
    outdir_text: str,
) -> str:
    outdir = Path(outdir_text)
    jobs = [
        CalibrationJob(
            bundle.design, bundle.seed, index, float(target)
        )
        for index, target in enumerate(targets)
    ]
    missing = [
        job
        for job in jobs
        if _valid_calibration_checkpoint(
            calibration_checkpoint_path(outdir, job),
            job,
            task_protocol_sha256,
        )
        is None
    ]
    if not missing:
        return f"skip calibration bundle {bundle.design}/{bundle.seed}"

    couplings, calibration_inputs, _ = _stream_material(bundle.seed)
    engine = AffineActivityEngine(couplings, bundle.design)
    measurements: dict[float, dict] = {}

    def evaluator(rate: float) -> float:
        key = float(rate)
        if key not in measurements:
            measurements[key] = engine.calibration_activity(
                calibration_inputs, key
            )
        return float(measurements[key]["activity"])

    branch = BRANCHES[bundle.design]
    for job in missing:
        started = time.perf_counter()
        solution = bisect_activity_target(
            evaluator,
            job.target,
            branch["lower_rate"],
            branch["upper_rate"],
            branch["activity_orientation"],
        )
        evaluations = []
        for entry in solution.get("evaluations", []):
            measurement = measurements[float(entry["rate"])]
            evaluations.append(
                {
                    **entry,
                    "jump_strength": measurement["jump_strength"],
                    "maximum_trace_error": measurement[
                        "maximum_trace_error"
                    ],
                    "maximum_activity_imaginary_residue": measurement[
                        "maximum_activity_imaginary_residue"
                    ],
                }
            )
        solution["evaluations"] = evaluations
        row = {
            "artifact_type": "fresh_activity_calibration_row",
            "protocol_version": PROTOCOL_VERSION,
            "task_protocol_sha256": task_protocol_sha256,
            "design": job.design,
            "seed": job.seed,
            "target_index": job.target_index,
            "target_activity": job.target,
            "branch": branch,
            "couplings_sha256": array_sha256(couplings),
            "calibration_input_sha256": array_sha256(calibration_inputs),
            "calibration_uses_task_targets_or_scores": False,
            **solution,
            "runtime_seconds": float(time.perf_counter() - started),
        }
        atomic_json(calibration_checkpoint_path(outdir, job), row)
    return f"done calibration bundle {bundle.design}/{bundle.seed}"


def calibration_jobs(manifest: dict) -> list[CalibrationJob]:
    seeds = manifest["protocol"]["seed_ledger"]["task_seeds"]
    targets = manifest["protocol"]["frozen_targets"]["targets"]
    return [
        CalibrationJob(design, seed, index, float(target))
        for design in DESIGNS
        for seed in seeds
        for index, target in enumerate(targets)
    ]


def run_calibration(outdir: Path, workers: int) -> None:
    manifest, digest, _ = ensure_task_manifest(outdir)
    score_dir = outdir / "score" / "checkpoints"
    if score_dir.exists() and any(score_dir.glob("*.json")):
        raise RuntimeError("task scores exist before calibration is frozen")
    targets = manifest["protocol"]["frozen_targets"]["targets"]
    bundles = [
        CalibrationBundleJob(design, seed)
        for design in DESIGNS
        for seed in manifest["protocol"]["seed_ledger"]["task_seeds"]
    ]
    _run_parallel(
        _calibration_bundle_worker,
        bundles,
        (digest, targets, str(outdir)),
        workers,
    )


def load_calibration_rows(
    outdir: Path, manifest: dict, protocol_sha256: str
) -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    for job in calibration_jobs(manifest):
        row = _valid_calibration_checkpoint(
            calibration_checkpoint_path(outdir, job),
            job,
            protocol_sha256,
        )
        if row is None:
            missing.append(f"{job.design}/{job.seed}/{job.target_index}")
        else:
            rows.append(row)
    if missing:
        raise RuntimeError(
            f"calibration incomplete: {len(missing)} missing; first={missing[:5]}"
        )
    return rows


def freeze_calibration(outdir: Path) -> dict:
    manifest, digest, frozen_targets = ensure_task_manifest(outdir)
    rows = load_calibration_rows(outdir, manifest, digest)
    expected = manifest["protocol"]["calibration"]["expected_cells"]
    errors: list[str] = []
    if len(rows) != expected:
        errors.append(f"coverage {len(rows)}/{expected}")
    censored = [row for row in rows if row.get("status") != "matched"]
    if censored:
        errors.append(f"{len(censored)} calibration cells censored")
    excessive = [
        row
        for row in rows
        if row.get("status") == "matched"
        and float(row["relative_error"]) > CALIBRATION_RELATIVE_TOL
    ]
    if excessive:
        errors.append(f"{len(excessive)} cells exceed 0.5% match error")

    targets = list(map(float, frozen_targets["targets"]))
    expected_keys = {
        (design, int(seed), target_index)
        for design in DESIGNS
        for seed in manifest["protocol"]["seed_ledger"]["task_seeds"]
        for target_index in range(len(targets))
    }
    observed_keys = {
        (str(row["design"]), int(row["seed"]), int(row["target_index"]))
        for row in rows
    }
    if observed_keys != expected_keys or len(rows) != len(expected_keys):
        errors.append("calibration identity coverage or uniqueness failed")

    for seed in manifest["protocol"]["seed_ledger"]["task_seeds"]:
        seed_rows = [row for row in rows if int(row["seed"]) == int(seed)]
        coupling_hashes = {row["couplings_sha256"] for row in seed_rows}
        input_hashes = {
            row["calibration_input_sha256"] for row in seed_rows
        }
        if len(coupling_hashes) != 1:
            errors.append(f"coupling pairing failed for seed {seed}")
        if len(input_hashes) != 1:
            errors.append(f"calibration-input pairing failed for seed {seed}")

    for row in rows:
        if row.get("status") != "matched":
            continue
        branch = BRANCHES[str(row["design"])]
        rate = float(row["matched_rate"])
        target = float(row["target_activity"])
        activity = float(row["matched_activity"])
        derived_error = abs(activity - target) / target
        if not (
            float(branch["lower_rate"])
            <= rate
            <= float(branch["upper_rate"])
        ):
            errors.append(
                f"matched rate outside frozen branch "
                f"{row['design']}/{row['seed']}/{row['target_index']}"
            )
        if not math.isclose(
            derived_error,
            float(row["relative_error"]),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            errors.append(
                f"stored calibration error drift "
                f"{row['design']}/{row['seed']}/{row['target_index']}"
            )

    for design in DESIGNS:
        orientation = BRANCHES[design]["activity_orientation"]
        sign = _orientation_sign(orientation)
        for seed in manifest["protocol"]["seed_ledger"]["task_seeds"]:
            group = sorted(
                (
                    row
                    for row in rows
                    if row["design"] == design and int(row["seed"]) == seed
                ),
                key=lambda row: int(row["target_index"]),
            )
            if len(group) != len(targets) or any(
                row.get("status") != "matched" for row in group
            ):
                continue
            target_values = np.asarray(
                [row["target_activity"] for row in group], dtype=float
            )
            rate_values = np.asarray(
                [row["matched_rate"] for row in group], dtype=float
            )
            if not np.all(np.diff(target_values) > 0):
                errors.append(f"target order failed {design}/{seed}")
            if np.any(sign * np.diff(rate_values) < -1e-12):
                errors.append(f"matched-rate order failed {design}/{seed}")

    payload = {
        "artifact_type": "frozen_fresh_activity_calibration",
        "freeze_status": "frozen_before_any_task_score",
        "task_protocol_sha256": digest,
        "frozen_targets_sha256": file_sha256(frozen_targets_path(outdir)),
        "expected_cells": expected,
        "observed_cells": len(rows),
        "censored_cells": len(censored),
        "maximum_relative_match_error": (
            max(
                (
                    float(row["relative_error"])
                    for row in rows
                    if row.get("status") == "matched"
                ),
                default=float("inf"),
            )
        ),
        "gate_passed": not errors,
        "gate_errors": errors,
        "cells": [
            {
                "design": row["design"],
                "seed": int(row["seed"]),
                "target_index": int(row["target_index"]),
                "target_activity": float(row["target_activity"]),
                "matched_rate": (
                    float(row["matched_rate"])
                    if row.get("status") == "matched"
                    else None
                ),
                "matched_activity": (
                    float(row["matched_activity"])
                    if row.get("status") == "matched"
                    else None
                ),
                "relative_error": (
                    float(row["relative_error"])
                    if row.get("status") == "matched"
                    else None
                ),
                "calibration_row_sha256": file_sha256(
                    calibration_checkpoint_path(
                        outdir,
                        CalibrationJob(
                            row["design"],
                            int(row["seed"]),
                            int(row["target_index"]),
                            float(row["target_activity"]),
                        ),
                    )
                ),
                "calibration_row_payload_sha256": sha256_json(row),
                "couplings_sha256": row["couplings_sha256"],
                "calibration_input_sha256": row[
                    "calibration_input_sha256"
                ],
                "status": row["status"],
            }
            for row in sorted(
                rows,
                key=lambda row: (
                    row["design"],
                    int(row["seed"]),
                    int(row["target_index"]),
                ),
            )
        ],
    }
    if errors:
        raise RuntimeError(
            "fresh calibration cannot be frozen: " + "; ".join(errors)
        )
    score_dir = outdir / "score" / "checkpoints"
    path = frozen_calibration_path(outdir)
    if path.exists():
        old = json.loads(path.read_text())
        if canonical_json(old) != canonical_json(payload):
            raise RuntimeError("frozen calibration drift")
    else:
        if score_dir.exists() and any(score_dir.glob("*.json")):
            raise RuntimeError("task scores exist before calibration freeze")
        atomic_json(path, payload)
    return payload


def load_frozen_calibration(outdir: Path) -> dict:
    path = frozen_calibration_path(outdir)
    if not path.is_file():
        raise RuntimeError("fresh calibration has not been frozen")
    payload = json.loads(path.read_text())
    manifest, digest, _ = ensure_task_manifest(outdir)
    if payload.get("task_protocol_sha256") != digest:
        raise RuntimeError("frozen calibration protocol mismatch")
    if not payload.get("gate_passed") or payload.get("censored_cells") != 0:
        raise RuntimeError("frozen calibration did not pass its gate")
    if len(payload.get("cells", [])) != manifest["protocol"]["calibration"][
        "expected_cells"
    ]:
        raise RuntimeError("frozen calibration coverage changed")
    return payload


def _task_trajectory(
    couplings: np.ndarray,
    inputs: np.ndarray,
    design: str,
    rate: float,
) -> tuple[np.ndarray, dict]:
    return AffineActivityEngine(couplings, design).task_trajectory(
        inputs, rate
    )


def _stm_score(features: np.ndarray, inputs: np.ndarray) -> tuple[float, list[float]]:
    if features.shape[0] != TRAIN + TEST:
        raise ValueError("feature split has wrong length")
    x_bias = readout.add_bias(features)
    train_targets: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    for delay in DELAYS:
        target = tasks.delayed_target(inputs, delay)[WASH:]
        train_target = target[:TRAIN]
        test_target = target[TRAIN:]
        if not np.all(np.isfinite(train_target)) or not np.all(
            np.isfinite(test_target)
        ):
            raise RuntimeError("STM target contains undefined values")
        train_targets.append(train_target)
        test_targets.append(test_target)
    train_matrix = np.column_stack(train_targets)
    gram = x_bias[:TRAIN].T @ x_bias[:TRAIN]
    rhs = x_bias[:TRAIN].T @ train_matrix
    weights = np.linalg.solve(
        gram + FIXED_RIDGE * np.eye(gram.shape[0]), rhs
    )
    predictions = x_bias[TRAIN:] @ weights
    capacities = [
        readout.capacity(target, predictions[:, index])
        for index, target in enumerate(test_targets)
    ]
    return float(np.sum(capacities)), capacities


def score_jobs(frozen_calibration: dict) -> list[ScoreJob]:
    return [
        ScoreJob(
            cell["design"],
            int(cell["seed"]),
            int(cell["target_index"]),
            float(cell["target_activity"]),
            float(cell["matched_rate"]),
            cell["calibration_row_sha256"],
        )
        for cell in frozen_calibration["cells"]
    ]


def run_score_job(
    job: ScoreJob,
    task_protocol_sha256: str,
    frozen_calibration_sha256: str,
) -> dict:
    started = time.perf_counter()
    couplings, calibration_inputs, task_inputs = _stream_material(job.seed)
    features, activity = AffineActivityEngine(
        couplings, job.design
    ).task_trajectory(task_inputs, job.rate)
    score, by_delay = _stm_score(features, task_inputs)
    return {
        "artifact_type": "fresh_activity_matched_stm_row",
        "protocol_version": PROTOCOL_VERSION,
        "task_protocol_sha256": task_protocol_sha256,
        "frozen_calibration_sha256": frozen_calibration_sha256,
        "calibration_row_sha256": job.calibration_sha256,
        "design": job.design,
        "seed": job.seed,
        "target_index": job.target_index,
        "target_activity": job.target,
        "frozen_rate": job.rate,
        "couplings_sha256": array_sha256(couplings),
        "calibration_input_sha256": array_sha256(calibration_inputs),
        "task_input_sha256": array_sha256(task_inputs),
        "task_stream_is_independent_of_calibration_stream": (
            array_sha256(calibration_inputs) != array_sha256(task_inputs)
        ),
        "ridge": FIXED_RIDGE,
        "test_stm_capacity": score,
        "test_capacity_by_delay": by_delay,
        **activity,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _valid_score_checkpoint(
    path: Path,
    job: ScoreJob,
    task_protocol_sha256: str,
    frozen_calibration_sha256: str,
) -> dict | None:
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    expected = {
        "design": job.design,
        "seed": job.seed,
        "target_index": job.target_index,
        "target_activity": job.target,
        "frozen_rate": job.rate,
        "task_protocol_sha256": task_protocol_sha256,
        "frozen_calibration_sha256": frozen_calibration_sha256,
        "calibration_row_sha256": job.calibration_sha256,
    }
    if all(row.get(key) == value for key, value in expected.items()):
        return row
    raise RuntimeError(f"stale task-score checkpoint: {path}")


def _score_bundle_worker(
    bundle: ScoreBundleJob,
    cells: Sequence[dict],
    task_protocol_sha256: str,
    frozen_calibration_sha256: str,
    outdir_text: str,
) -> str:
    outdir = Path(outdir_text)
    jobs = [
        ScoreJob(
            cell["design"],
            int(cell["seed"]),
            int(cell["target_index"]),
            float(cell["target_activity"]),
            float(cell["matched_rate"]),
            cell["calibration_row_sha256"],
        )
        for cell in cells
        if cell["design"] == bundle.design
        and int(cell["seed"]) == bundle.seed
    ]
    missing = [
        job
        for job in jobs
        if _valid_score_checkpoint(
            score_checkpoint_path(outdir, job),
            job,
            task_protocol_sha256,
            frozen_calibration_sha256,
        )
        is None
    ]
    if not missing:
        return f"skip score bundle {bundle.design}/{bundle.seed}"
    couplings, calibration_inputs, task_inputs = _stream_material(bundle.seed)
    engine = AffineActivityEngine(couplings, bundle.design)
    coupling_hash = array_sha256(couplings)
    calibration_hash = array_sha256(calibration_inputs)
    task_hash = array_sha256(task_inputs)
    for job in missing:
        started = time.perf_counter()
        features, activity = engine.task_trajectory(task_inputs, job.rate)
        score, by_delay = _stm_score(features, task_inputs)
        row = {
            "artifact_type": "fresh_activity_matched_stm_row",
            "protocol_version": PROTOCOL_VERSION,
            "task_protocol_sha256": task_protocol_sha256,
            "frozen_calibration_sha256": frozen_calibration_sha256,
            "calibration_row_sha256": job.calibration_sha256,
            "design": job.design,
            "seed": job.seed,
            "target_index": job.target_index,
            "target_activity": job.target,
            "frozen_rate": job.rate,
            "couplings_sha256": coupling_hash,
            "calibration_input_sha256": calibration_hash,
            "task_input_sha256": task_hash,
            "task_stream_is_independent_of_calibration_stream": (
                calibration_hash != task_hash
            ),
            "ridge": FIXED_RIDGE,
            "test_stm_capacity": score,
            "test_capacity_by_delay": by_delay,
            **activity,
            "runtime_seconds": float(time.perf_counter() - started),
        }
        atomic_json(score_checkpoint_path(outdir, job), row)
    return f"done score bundle {bundle.design}/{bundle.seed}"


def run_scores(outdir: Path, workers: int) -> None:
    manifest, digest, _ = ensure_task_manifest(outdir)
    frozen = load_frozen_calibration(outdir)
    frozen_digest = file_sha256(frozen_calibration_path(outdir))
    bundles = [
        ScoreBundleJob(design, seed)
        for design in DESIGNS
        for seed in manifest["protocol"]["seed_ledger"]["task_seeds"]
    ]
    _run_parallel(
        _score_bundle_worker,
        bundles,
        (frozen["cells"], digest, frozen_digest, str(outdir)),
        workers,
    )


def load_score_rows(
    outdir: Path,
    manifest: dict,
    protocol_sha256: str,
    frozen: dict,
) -> list[dict]:
    frozen_digest = file_sha256(frozen_calibration_path(outdir))
    rows: list[dict] = []
    missing: list[str] = []
    for job in score_jobs(frozen):
        row = _valid_score_checkpoint(
            score_checkpoint_path(outdir, job),
            job,
            protocol_sha256,
            frozen_digest,
        )
        if row is None:
            missing.append(f"{job.design}/{job.seed}/{job.target_index}")
        else:
            rows.append(row)
    expected = manifest["protocol"]["calibration"]["expected_cells"]
    if missing or len(rows) != expected:
        raise RuntimeError(
            f"task score coverage {len(rows)}/{expected}; "
            f"first missing={missing[:5]}"
        )
    return rows


def bonferroni_paired_band(
    differences: Sequence[float],
    *,
    family_size: int = N_ACTIVITY_TARGETS,
    alpha: float = SIMULTANEOUS_ALPHA,
) -> dict:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a finite vector of length >=2")
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / math.sqrt(len(values)))
    critical = float(
        student_t.ppf(1.0 - alpha / (2.0 * family_size), len(values) - 1)
    )
    nonzero = values[~np.isclose(values, 0.0, atol=1e-14)]
    wins = int(np.sum(nonzero > 0))
    return {
        "n": int(len(values)),
        "mean": mean,
        "standard_error": standard_error,
        "critical_value": critical,
        "simultaneous_lower": float(mean - critical * standard_error),
        "simultaneous_upper": float(mean + critical * standard_error),
        "family_size": family_size,
        "familywise_alpha": alpha,
        "wins": wins,
        "ties": int(len(values) - len(nonzero)),
        "losses": int(np.sum(nonzero < 0)),
        "paired_values": values.tolist(),
    }


def build_aggregate(
    rows: Sequence[dict],
    manifest: dict,
    frozen_calibration: dict,
    frozen_calibration_file_sha256: str,
) -> dict:
    seeds = list(map(int, manifest["protocol"]["seed_ledger"]["task_seeds"]))
    targets = list(
        map(float, manifest["protocol"]["frozen_targets"]["targets"])
    )
    expected_keys = {
        (design, seed, index)
        for design in DESIGNS
        for seed in seeds
        for index in range(len(targets))
    }
    keys = {
        (row["design"], int(row["seed"]), int(row["target_index"]))
        for row in rows
    }
    errors: list[str] = []
    if keys != expected_keys or len(rows) != len(expected_keys):
        errors.append("score coverage or uniqueness failed")
    if frozen_calibration.get("censored_cells") != 0:
        errors.append("fresh calibration includes censored cells")
    if any(
        not bool(row["task_stream_is_independent_of_calibration_stream"])
        for row in rows
    ):
        errors.append("calibration/task input separation failed")

    frozen_cells = {
        (
            str(cell["design"]),
            int(cell["seed"]),
            int(cell["target_index"]),
        ): cell
        for cell in frozen_calibration.get("cells", [])
    }
    if set(frozen_cells) != expected_keys or len(frozen_cells) != len(
        expected_keys
    ):
        errors.append("frozen-calibration cell coverage failed")
    for row in rows:
        key = (
            str(row["design"]),
            int(row["seed"]),
            int(row["target_index"]),
        )
        cell = frozen_cells.get(key)
        if cell is None:
            continue
        if (
            row.get("task_protocol_sha256") != manifest["protocol_sha256"]
            or row.get("frozen_calibration_sha256")
            != frozen_calibration_file_sha256
            or row.get("calibration_row_sha256")
            != cell.get("calibration_row_sha256")
            or not math.isclose(
                float(row["target_activity"]),
                float(cell["target_activity"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            )
            or not math.isclose(
                float(row["frozen_rate"]),
                float(cell["matched_rate"]),
                rel_tol=0.0,
                abs_tol=1e-14,
            )
            or row.get("couplings_sha256") != cell.get("couplings_sha256")
            or row.get("calibration_input_sha256")
            != cell.get("calibration_input_sha256")
        ):
            errors.append(
                f"frozen-cell linkage failed "
                f"{row['design']}/{row['seed']}/{row['target_index']}"
            )

    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        coupling_hashes = {row["couplings_sha256"] for row in seed_rows}
        calibration_hashes = {
            row["calibration_input_sha256"] for row in seed_rows
        }
        task_hashes = {row["task_input_sha256"] for row in seed_rows}
        if len(coupling_hashes) != 1:
            errors.append(f"score coupling pairing failed for seed {seed}")
        if len(calibration_hashes) != 1:
            errors.append(
                f"score calibration-input pairing failed for seed {seed}"
            )
        if len(task_hashes) != 1:
            errors.append(f"score task-input pairing failed for seed {seed}")
        if calibration_hashes & task_hashes:
            errors.append(f"calibration/task stream reuse for seed {seed}")

    fixed_ridge_all_rows = all(
        math.isclose(
            float(row["ridge"]), FIXED_RIDGE, rel_tol=0.0, abs_tol=1e-18
        )
        for row in rows
    )
    if not fixed_ridge_all_rows:
        errors.append("fixed-ridge invariant failed")
    delay_score_sums_match = all(
        isinstance(row.get("test_capacity_by_delay"), list)
        and len(row["test_capacity_by_delay"]) == len(DELAYS)
        and math.isclose(
            math.fsum(float(value) for value in row["test_capacity_by_delay"]),
            float(row["test_stm_capacity"]),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for row in rows
    )
    if not delay_score_sums_match:
        errors.append("delay-capacity sum invariant failed")

    max_trace = max(float(row["maximum_trace_error"]) for row in rows)
    max_imaginary = max(
        float(row["maximum_activity_imaginary_residue"]) for row in rows
    )
    min_count = min(
        float(row["minimum_test_interval_integrated_activity"])
        for row in rows
    )
    if max_trace > TRACE_TOL:
        errors.append("trace invariant failed")
    if max_imaginary > IMAGINARY_TOL:
        errors.append("activity imaginary-residue invariant failed")
    if min_count < -NEGATIVE_COUNT_TOL:
        errors.append("activity non-negativity invariant failed")

    target_results: list[dict] = []
    for index, target in enumerate(targets):
        by_design: dict[str, dict[int, dict]] = {}
        for design in DESIGNS:
            group = {
                int(row["seed"]): row
                for row in rows
                if row["design"] == design
                and int(row["target_index"]) == index
            }
            if set(group) != set(seeds):
                errors.append(f"incomplete target {index}/{design}")
            by_design[design] = group
        local_scores = np.asarray(
            [by_design["local"][seed]["test_stm_capacity"] for seed in seeds]
        )
        collective_scores = np.asarray(
            [
                by_design["collective"][seed]["test_stm_capacity"]
                for seed in seeds
            ]
        )
        local_activity = np.asarray(
            [
                by_design["local"][seed]["time_averaged_test_activity"]
                for seed in seeds
            ]
        )
        collective_activity = np.asarray(
            [
                by_design["collective"][seed]["time_averaged_test_activity"]
                for seed in seeds
            ]
        )
        stm_band = bonferroni_paired_band(
            collective_scores - local_scores
        )
        activity_band = bonferroni_paired_band(
            (collective_activity - local_activity) / target
        )
        target_results.append(
            {
                "target_index": index,
                "target_activity": target,
                "local_stm_mean": float(np.mean(local_scores)),
                "collective_stm_mean": float(np.mean(collective_scores)),
                "stm_collective_minus_local": stm_band,
                "local_test_activity_mean": float(np.mean(local_activity)),
                "collective_test_activity_mean": float(
                    np.mean(collective_activity)
                ),
                "relative_test_activity_collective_minus_local": activity_band,
                "stm_dominance_at_target": (
                    stm_band["simultaneous_lower"] > 0
                ),
                "test_activity_equivalent_at_target": (
                    activity_band["simultaneous_lower"]
                    >= -TEST_ACTIVITY_EQUIVALENCE_MARGIN
                    and activity_band["simultaneous_upper"]
                    <= TEST_ACTIVITY_EQUIVALENCE_MARGIN
                ),
            }
        )

    no_censoring = frozen_calibration.get("censored_cells") == 0
    all_stm = all(
        item["stm_dominance_at_target"] for item in target_results
    )
    all_activity = all(
        item["test_activity_equivalent_at_target"] for item in target_results
    )
    aggregate = {
        "artifact_type": "activity_matched_response_aggregate",
        "protocol_version": PROTOCOL_VERSION,
        "task_protocol_sha256": manifest["protocol_sha256"],
        "frozen_calibration_sha256": frozen_calibration_file_sha256,
        "status": "complete" if not errors else "invalid",
        "n_rows": len(rows),
        "n_seeds": len(seeds),
        "n_targets": len(targets),
        "target_results": target_results,
        "claim_gates": {
            "zero_censored_fresh_cells": no_censoring,
            "simultaneous_stm_dominance_all_targets": all_stm,
            "simultaneous_test_activity_equivalence_all_targets": all_activity,
            "range_wide_dominance_supported": all_stm,
            "task_activity_equivalence_supported": all_activity,
            "activity_matched_dominance_claim_allowed": (
                not errors and no_censoring and all_stm and all_activity
            ),
            "failure_rule": (
                "the range-wide dominance claim fails if any calibration cell "
                "is censored, any STM simultaneous lower bound is <=0, or any "
                "test-activity relative band leaves [-0.05,0.05]"
            ),
        },
        "invariant_audit": {
            "passed": not errors,
            "errors": errors,
            "expected_rows": len(expected_keys),
            "observed_rows": len(rows),
            "maximum_trace_error": max_trace,
            "maximum_activity_imaginary_residue": max_imaginary,
            "minimum_test_interval_integrated_activity": min_count,
            "fixed_ridge_all_rows": fixed_ridge_all_rows,
            "delay_capacity_sums_match": delay_score_sums_match,
        },
        "limitations": [
            (
                "Matching is to expected jump activity under the declared "
                "gauge-fixed unraveling, not dissipated energy, entropy "
                "production, bath power, or hardware cost."
            ),
            (
                "The result compares one prespecified weak local branch with "
                "one prespecified strong collective branch at N=5."
            ),
            (
                "Task-test activity is an independent generalization check; "
                "the rates were chosen only from label-free calibration inputs."
            ),
        ],
    }
    return aggregate


def aggregate_results(outdir: Path) -> dict:
    manifest, digest, _ = ensure_task_manifest(outdir)
    frozen = load_frozen_calibration(outdir)
    rows = load_score_rows(outdir, manifest, digest, frozen)
    frozen_file_sha = file_sha256(frozen_calibration_path(outdir))
    aggregate = build_aggregate(
        rows, manifest, frozen, frozen_file_sha
    )
    if aggregate["status"] != "complete":
        raise RuntimeError(
            f"aggregate invariants failed: "
            f"{aggregate['invariant_audit']['errors']}"
        )
    atomic_json(aggregate_path(outdir), aggregate)
    return aggregate


def render_report(aggregate: dict, manifest: dict, frozen: dict) -> str:
    lines = [
        "# Prospective activity-matched STM response",
        "",
        f"**Status:** {aggregate['status']}.",
        "",
        "Eight pilot-only reservoirs fixed five activity targets before any "
        "fresh task score. All fresh scalar rates were calibrated on independent "
        "label-free inputs and hash-frozen before the STM trajectories.",
        "",
        "| target $\\mathcal J$ | local STM | collective STM | paired gain "
        "(simultaneous 95% band) | relative test-activity difference "
        "(simultaneous 95% band) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate["target_results"]:
        stm = item["stm_collective_minus_local"]
        activity = item["relative_test_activity_collective_minus_local"]
        lines.append(
            f"| {item['target_activity']:.8f} | "
            f"{item['local_stm_mean']:.6f} | "
            f"{item['collective_stm_mean']:.6f} | "
            f"{stm['mean']:.6f} "
            f"[{stm['simultaneous_lower']:.6f}, "
            f"{stm['simultaneous_upper']:.6f}] | "
            f"{activity['mean']:.4%} "
            f"[{activity['simultaneous_lower']:.4%}, "
            f"{activity['simultaneous_upper']:.4%}] |"
        )
    gates = aggregate["claim_gates"]
    lines.extend(
        [
            "",
            "## Prespecified claim gates",
            "",
            f"- Zero censored calibration cells: "
            f"`{gates['zero_censored_fresh_cells']}`.",
            f"- Collective STM lower band above zero at all five targets: "
            f"`{gates['simultaneous_stm_dominance_all_targets']}`.",
            f"- Test-trajectory activity bands within ±5% at all targets: "
            f"`{gates['simultaneous_test_activity_equivalence_all_targets']}`.",
            f"- Range-wide activity-matched dominance claim allowed: "
            f"`{gates['activity_matched_dominance_claim_allowed']}`.",
            "",
            "The statistical unit is the paired reservoir seed. Bands are "
            "two-sided Bonferroni-t simultaneous 95% intervals over exactly "
            "five prespecified target contrasts.",
            "",
            "## Provenance",
            "",
            f"- Task protocol SHA-256: `{aggregate['task_protocol_sha256']}`.",
            f"- Frozen calibration SHA-256: "
            f"`{aggregate['frozen_calibration_sha256']}`.",
            f"- Fresh coverage: {aggregate['n_rows']}/"
            f"{manifest['protocol']['calibration']['expected_cells']} rows.",
            f"- Maximum calibration relative error: "
            f"{frozen['maximum_relative_match_error']:.3e}.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in aggregate["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_report(outdir: Path, report: Path) -> None:
    manifest, _, _ = ensure_task_manifest(outdir)
    frozen = load_frozen_calibration(outdir)
    path = aggregate_path(outdir)
    if not path.is_file():
        raise RuntimeError("aggregate is missing")
    aggregate = json.loads(path.read_text())
    if aggregate.get("task_protocol_sha256") != manifest["protocol_sha256"]:
        raise RuntimeError("aggregate/task protocol mismatch")
    atomic_text(report, render_report(aggregate, manifest, frozen))


def validate_artifacts(outdir: Path) -> dict:
    manifest, digest, _ = ensure_task_manifest(outdir)
    frozen = load_frozen_calibration(outdir)
    rows = load_score_rows(outdir, manifest, digest, frozen)
    frozen_file_sha = file_sha256(frozen_calibration_path(outdir))
    rebuilt = build_aggregate(rows, manifest, frozen, frozen_file_sha)
    path = aggregate_path(outdir)
    if path.is_file():
        saved = json.loads(path.read_text())
        if canonical_json(saved) != canonical_json(rebuilt):
            raise RuntimeError("saved aggregate does not reproduce")
    return {
        "status": rebuilt["status"],
        "task_protocol_sha256": digest,
        "frozen_calibration_sha256": frozen_file_sha,
        "score_rows": len(rows),
        "claim_allowed": rebuilt["claim_gates"][
            "activity_matched_dominance_claim_allowed"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "pilot",
            "freeze-targets",
            "calibrate",
            "freeze-calibration",
            "score",
            "aggregate",
            "report",
            "validate",
            "all",
        ),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    outdir = args.outdir.resolve()

    if args.command in ("pilot", "all"):
        run_pilot(outdir, args.workers)
    if args.command in ("freeze-targets", "all"):
        freeze_targets(outdir)
    if args.command in ("calibrate", "all"):
        run_calibration(outdir, args.workers)
    if args.command in ("freeze-calibration", "all"):
        freeze_calibration(outdir)
    if args.command in ("score", "all"):
        run_scores(outdir, args.workers)
    if args.command in ("aggregate", "all"):
        aggregate_results(outdir)
    if args.command in ("report", "all"):
        write_report(outdir, args.report)
    if args.command == "validate":
        print(json.dumps(validate_artifacts(outdir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

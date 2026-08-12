"""Camera-ready primary-task readout-regularisation control.

This additive experiment repeats the seven dissipative N=5 STM and NARMA-10
comparisons from the definitive protocol while isolating one issue: readout
regularisation.  Each method/seed trajectory is computed once and both tasks
are scored with a validation-selected ridge and the historical fixed ridge of
1e-8.

The train/validation/test split is frozen before any fitting.  A near-constant
feature guard is fitted on the raw training rows only, then applied unchanged
to validation and test rows.  The selected ridge is chosen on validation
NMSE, refitted on train+validation, and evaluated once on the untouched test
block.  Atomic per-job JSON files make the 224-trajectory run restart-safe.

Examples
--------
python experiments/run_revision_primary_regularization.py all --workers 4
python experiments/run_revision_primary_regularization.py aggregate
python experiments/run_revision_primary_regularization.py report
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
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import _paths  # noqa: F401
import numpy as np
import scipy
from scipy.stats import t as student_t

from _paths import REPORTS_DIR, RESULTS_DIR
from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir
from run_final_scaling import build_jumps, deterministic_seeds
from run_revision_controls import (
    _svd_ridge_predict,
    refit_and_test,
    select_ridge,
    train_only_variance_filter,
)


PROTOCOL_VERSION = "revision-primary-readout-regularization-v1-2026-07-24"
N_QUBITS = 5
H = 0.5
DT = 0.5
GAMMA = 1.0
WASH = 200
TRAIN = 450
VALIDATION = 150
TEST = 400
NARMA_ORDER = 10
INPUT_SCALE = 0.2
STM_DELAYS = tuple(range(1, 21))
FIXED_RIDGE = 1e-8
FEATURE_STD_TOL = 1e-12
RIDGES = (
    0.0,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)
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
TASKS = ("stm", "narma")
TASK_METRICS = {"stm": "capacity", "narma": "nmse"}
DEFAULT_OUTDIR = Path(RESULTS_DIR) / "revision_primary_regularization"
DEFAULT_REPORT = Path(REPORTS_DIR) / "revision_primary_regularization_report.md"
ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "results" / "final_protocol"
BASELINE_ATOL = 1e-9
BASELINE_RTOL = 1e-9
BASELINE_GUARD_EXCEPTION_METHODS = ("B1_dephasing",)
SOURCE_FILES = (
    "experiments/run_revision_primary_regularization.py",
    "experiments/run_final_scaling.py",
    "experiments/run_revision_controls.py",
    "src/qrc/dissipators.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)


@dataclass(frozen=True)
class Job:
    method: str
    seed: int

    @property
    def key(self) -> tuple[str, int]:
        return self.method, self.seed


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
    elif np.issubdtype(array.dtype, np.integer):
        canonical = np.ascontiguousarray(array, dtype="<i8")
    elif np.issubdtype(array.dtype, np.bool_):
        canonical = np.ascontiguousarray(array, dtype=np.uint8)
    else:
        canonical = np.ascontiguousarray(array, dtype="<f8")
    header = canonical_json(
        {"shape": list(canonical.shape), "dtype": canonical.dtype.str}
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + canonical.tobytes()).hexdigest()


def jump_family_sha256(jumps: Sequence[tuple[np.ndarray, float]]) -> str:
    parts = [
        {
            "operator_sha256": array_sha256(operator),
            "rate": float(rate),
        }
        for operator, rate in jumps
    ]
    return sha256_json(parts)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def source_environment_manifest() -> dict:
    files = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"protocol source file missing: {path}")
        files[relative] = file_sha256(path)
    return {
        "files": files,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def baseline_checkpoint_path(task_name: str, method: str, seed: int) -> Path:
    return BASELINE_DIR / (
        f"A_table__{task_name}_N5_{method}_s{seed}"
        "_h0.5_dt0.5_L200-600-400.json"
    )


def baseline_reference_manifest(
    methods: Sequence[str],
    seeds: Sequence[int],
) -> dict:
    entries = {}
    for task_name in TASKS:
        for method in methods:
            for seed in seeds:
                path = baseline_checkpoint_path(task_name, method, int(seed))
                if not path.is_file():
                    raise FileNotFoundError(
                        f"sealed primary baseline checkpoint missing: {path}"
                    )
                payload = json.loads(path.read_text())
                expected_identity = {
                    "block": "A_table",
                    "N": N_QUBITS,
                    "method": method,
                    "task": task_name,
                    "seed": int(seed),
                    "wash": WASH,
                    "train": TRAIN + VALIDATION,
                    "test": TEST,
                }
                for field, expected in expected_identity.items():
                    if payload.get(field) != expected:
                        raise RuntimeError(
                            f"sealed baseline identity mismatch at {path}: {field}"
                        )
                key = f"{task_name}/{method}/{int(seed)}"
                entries[key] = {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                    "value": float(payload["value"]),
                }
    return {
        "group": "results/final_protocol/A_table",
        "historical_ridge": FIXED_RIDGE,
        "historical_split": {
            "wash": WASH,
            "train": TRAIN + VALIDATION,
            "test": TEST,
        },
        "absolute_tolerance": BASELINE_ATOL,
        "relative_tolerance": BASELINE_RTOL,
        "guard_exception_methods": list(BASELINE_GUARD_EXCEPTION_METHODS),
        "guard_exception_reason": (
            "train-only near-constant filtering intentionally removes "
            "dephasing roundoff features amplified by the historical fit"
        ),
        "entries": entries,
        "entries_sha256": sha256_json(entries),
    }


def build_protocol(
    methods: Sequence[str] = METHODS,
    seeds: Sequence[int] | None = None,
) -> dict:
    methods = tuple(methods)
    seeds = tuple(deterministic_seeds(32) if seeds is None else seeds)
    if len(methods) != len(set(methods)) or not set(methods).issubset(METHODS):
        raise ValueError("methods must be unique members of the predeclared set")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    source = source_environment_manifest()
    baseline = baseline_reference_manifest(methods, seeds)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Does validation-selected readout regularisation change the primary "
            "seven-dissipator STM and NARMA-10 comparison?"
        ),
        "methods": list(methods),
        "method_labels": {method: METHOD_LABELS[method] for method in methods},
        "reference_method": REFERENCE_METHOD,
        "seeds": [int(seed) for seed in seeds],
        "n_jobs": len(methods) * len(seeds),
        "reservoir": {
            "n_qubits": N_QUBITS,
            "h": H,
            "dt": DT,
            "backend": "SparseLindbladReservoir exact expm_multiply",
            "observables": "all weight-1 and same-axis weight-2 Pauli strings",
            "n_raw_features": 45,
        },
        "dissipation_budget": {
            "definition": "sum_k rate_k Tr(L_k^dagger L_k)",
            "reference": "uniform local loss at gamma=1",
            "gamma": GAMMA,
            "matching": "primary Frobenius budget",
        },
        "tasks": {
            "stm": {
                "name": "short-term memory",
                "delays": list(STM_DELAYS),
                "metric": "summed capacity across delays (higher is better)",
                "target_specific_training_masks": (
                    "delay d uses all finite rows, i.e. 450-d training rows"
                ),
            },
            "narma": {
                "name": "NARMA-10",
                "order": NARMA_ORDER,
                "input_scale": INPUT_SCALE,
                "metric": "NMSE (lower is better)",
                "training_warmup_rows": NARMA_ORDER,
            },
            "shared_input_distribution": "iid Uniform[0,1]",
            "shared_trajectory": True,
        },
        "split": {
            "wash": WASH,
            "train": TRAIN,
            "validation": VALIDATION,
            "test": TEST,
            "total_inputs": WASH + TRAIN + VALIDATION + TEST,
            "post_wash_rows": TRAIN + VALIDATION + TEST,
            "expected_effective_train_rows": {
                "stm_by_delay": [TRAIN - delay for delay in STM_DELAYS],
                "narma": TRAIN - NARMA_ORDER,
            },
            "expected_effective_refit_rows": {
                "stm_by_delay": [
                    TRAIN + VALIDATION - delay for delay in STM_DELAYS
                ],
                "narma": TRAIN + VALIDATION - NARMA_ORDER,
            },
            "expected_effective_validation_rows": VALIDATION,
            "expected_effective_test_rows": TEST,
            "test_untouched_until_final_scoring": True,
        },
        "readout": {
            "ridge_grid": list(RIDGES),
            "selection_metric": {
                "stm": "summed validation capacity (higher is better)",
                "narma": "validation NMSE (lower is better)",
            },
            "tie_break": "stronger ridge within absolute tolerance 1e-12",
            "refit": "train plus validation at the selected ridge",
            "fixed_sensitivity_ridge": FIXED_RIDGE,
            "fixed_sensitivity_refit": "train plus validation",
            "feature_guard_fit_on": "raw training rows only",
            "feature_guard_std_threshold": FEATURE_STD_TOL,
            "bias": "appended after reservoir features and always retained",
        },
        "pairing": {
            "coupling_and_input_rng": "np.random.default_rng(seed)",
            "method_rng": "np.random.default_rng(seed + 1)",
            "same_couplings_inputs_targets_and_split_within_seed": True,
        },
        "source_environment": source,
        "source_environment_sha256": sha256_json(source),
        "baseline_reproduction": baseline,
    }
    return protocol


def protocol_sha256(protocol: dict) -> str:
    return sha256_json(protocol)


def protocol_path(outdir: Path) -> Path:
    return outdir / "protocol.json"


def aggregate_path(outdir: Path) -> Path:
    return outdir / "aggregate.json"


def job_path(outdir: Path, job: Job) -> Path:
    return outdir / "jobs" / f"{job.method}__s{job.seed}.json"


def write_or_validate_protocol(outdir: Path, protocol: dict) -> None:
    path = protocol_path(outdir)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != protocol:
            raise RuntimeError(
                f"protocol/source drift at {path}; use a new output directory"
            )
        return
    atomic_write_json(path, protocol)


def train_only_feature_guard(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    threshold: float = FEATURE_STD_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply the existing audited guard after appending an explicit bias."""
    return train_only_variance_filter(
        readout.add_bias(np.asarray(x_train, dtype=float)),
        readout.add_bias(np.asarray(x_validation, dtype=float)),
        readout.add_bias(np.asarray(x_test, dtype=float)),
        threshold=threshold,
    )


def select_and_refit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metric: str,
    ridges: Sequence[float] = RIDGES,
    fixed_ridge: float = FIXED_RIDGE,
) -> dict:
    if metric not in ("capacity", "nmse"):
        raise ValueError(f"unsupported primary-task metric {metric!r}")
    y_train = np.asarray(y_train, dtype=float)
    y_validation = np.asarray(y_validation, dtype=float)
    y_test = np.asarray(y_test, dtype=float)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
        y_validation = y_validation[:, None]
        y_test = y_test[:, None]
    selected, validation_totals, validation_by_target = select_ridge(
        x_train,
        y_train,
        x_validation,
        y_validation,
        ridges,
        metric=metric,
    )
    selected_test, selected_by_target = refit_and_test(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        selected,
        metric=metric,
    )
    # Keep the fixed-ridge sensitivity numerically identical to the sealed
    # primary implementation, which used the normal-equation readout helper.
    x_refit = np.vstack([x_train, x_validation])
    y_refit = np.vstack([y_train, y_validation])
    fixed_weights = readout.train_readout(x_refit, y_refit, ridge=fixed_ridge)
    fixed_prediction = readout.predict(x_test, fixed_weights)
    if metric == "capacity":
        fixed_by_target = [
            readout.capacity(y_test[:, column], fixed_prediction[:, column])
            for column in range(y_test.shape[1])
        ]
        fixed_test = float(sum(fixed_by_target))
    else:
        fixed_by_target = [
            readout.nmse(y_test[:, column], fixed_prediction[:, column])
            for column in range(y_test.shape[1])
        ]
        fixed_test = float(np.mean(fixed_by_target))
    selected_minus_fixed = float(selected_test - fixed_test)
    improvement = (
        selected_minus_fixed if metric == "capacity" else -selected_minus_fixed
    )
    return {
        "metric": metric,
        "selected_ridge": float(selected),
        "validation_by_ridge": {
            str(key): float(value) for key, value in validation_totals.items()
        },
        "validation_by_target_and_ridge": validation_by_target,
        "selected_test": float(selected_test),
        "selected_test_by_target": [float(value) for value in selected_by_target],
        "fixed_ridge": float(fixed_ridge),
        "fixed_test": float(fixed_test),
        "fixed_test_by_target": [float(value) for value in fixed_by_target],
        "selected_minus_fixed": selected_minus_fixed,
        "selection_improvement": float(improvement),
    }


def select_and_refit_stm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ridges: Sequence[float] = RIDGES,
    fixed_ridge: float = FIXED_RIDGE,
) -> dict:
    """Select one STM ridge while preserving per-delay finite-row fits.

    Delay ``d`` uses all ``TRAIN-d`` valid training targets.  Validation and
    test are fully defined for every delay.  Validation capacities are summed
    across delays to select one design/seed ridge; each delay is then refitted
    on its own valid training rows plus all validation rows.
    """
    matrices = tuple(
        np.asarray(value, dtype=float)
        for value in (x_train, y_train, x_validation, y_validation, x_test, y_test)
    )
    x_train, y_train, x_validation, y_validation, x_test, y_test = matrices
    if y_train.shape[1] != len(STM_DELAYS):
        raise ValueError("STM target columns do not match the predeclared delays")
    ridge_keys = [f"{float(ridge):.12g}" for ridge in ridges]
    validation_by_ridge = {key: 0.0 for key in ridge_keys}
    validation_by_target_and_ridge = {key: [] for key in ridge_keys}
    train_rows_by_delay = []
    refit_rows_by_delay = []
    per_delay_training = []
    for column, delay in enumerate(STM_DELAYS):
        train_valid = np.isfinite(y_train[:, column])
        if int(np.sum(train_valid)) != TRAIN - delay:
            raise RuntimeError(f"STM delay {delay} training-row invariant failed")
        if not np.all(np.isfinite(y_validation[:, column])):
            raise RuntimeError(f"STM delay {delay} validation target is undefined")
        if not np.all(np.isfinite(y_test[:, column])):
            raise RuntimeError(f"STM delay {delay} test target is undefined")
        x_delay = x_train[train_valid]
        y_delay = y_train[train_valid, column]
        predictions = _svd_ridge_predict(
            x_delay, y_delay, x_validation, ridges
        )
        for ridge, key in zip(ridges, ridge_keys):
            prediction = predictions[float(ridge)][:, 0]
            score = readout.capacity(y_validation[:, column], prediction)
            validation_by_ridge[key] += float(score)
            validation_by_target_and_ridge[key].append(float(score))
        train_rows_by_delay.append(int(np.sum(train_valid)))
        refit_rows_by_delay.append(int(np.sum(train_valid) + len(y_validation)))
        per_delay_training.append((x_delay, y_delay))

    best = max(validation_by_ridge.values())
    candidates = [
        float(key)
        for key, value in validation_by_ridge.items()
        if np.isclose(value, best, rtol=0.0, atol=1e-12)
    ]
    selected = max(candidates)
    selected_by_target = []
    fixed_by_target = []
    for column, (x_delay, y_delay) in enumerate(per_delay_training):
        x_refit = np.vstack([x_delay, x_validation])
        y_refit = np.concatenate([y_delay, y_validation[:, column]])
        selected_prediction = _svd_ridge_predict(
            x_refit, y_refit, x_test, (selected,)
        )[selected][:, 0]
        selected_by_target.append(
            readout.capacity(y_test[:, column], selected_prediction)
        )
        fixed_weights = readout.train_readout(
            x_refit, y_refit, ridge=fixed_ridge
        )
        fixed_by_target.append(
            readout.capacity(
                y_test[:, column], readout.predict(x_test, fixed_weights)
            )
        )
    selected_test = float(sum(selected_by_target))
    fixed_test = float(sum(fixed_by_target))
    return {
        "metric": "capacity",
        "selected_ridge": float(selected),
        "validation_by_ridge": validation_by_ridge,
        "validation_by_target_and_ridge": validation_by_target_and_ridge,
        "selected_test": selected_test,
        "selected_test_by_target": [float(value) for value in selected_by_target],
        "fixed_ridge": float(fixed_ridge),
        "fixed_test": fixed_test,
        "fixed_test_by_target": [float(value) for value in fixed_by_target],
        "selected_minus_fixed": float(selected_test - fixed_test),
        "selection_improvement": float(selected_test - fixed_test),
        "effective_train_rows_by_target": train_rows_by_delay,
        "effective_refit_rows_by_target": refit_rows_by_delay,
    }


def split_hash(
    inputs: np.ndarray,
    target: np.ndarray,
    start: int,
    stop: int,
) -> str:
    return sha256_json(
        {
            "post_wash_input_sha256": array_sha256(inputs[start:stop]),
            "target_sha256": array_sha256(target[start:stop]),
            "start": int(start),
            "stop": int(stop),
        }
    )


def run_job(job: Job, protocol: dict) -> dict:
    started = time.perf_counter()
    if job.method not in protocol["methods"] or job.seed not in protocol["seeds"]:
        raise ValueError(f"job is outside protocol: {job}")

    problem_rng = np.random.default_rng(job.seed)
    couplings = res.random_couplings(N_QUBITS, 1.0, problem_rng)
    inputs = tasks.stm_inputs(
        WASH + TRAIN + VALIDATION + TEST,
        problem_rng,
    )
    post_wash_inputs = inputs[WASH:]
    targets = {
        "stm": np.column_stack(
            [
                tasks.delayed_target(post_wash_inputs, delay)
                for delay in STM_DELAYS
            ]
        ),
        "narma": tasks.narma_target(
            post_wash_inputs,
            order=NARMA_ORDER,
            input_scale=INPUT_SCALE,
        )[:, None],
    }

    method_rng = np.random.default_rng(job.seed + 1)
    target_strength = dsp.jump_strength(dsp.local_loss(N_QUBITS, GAMMA))
    jumps = build_jumps(
        job.method,
        couplings,
        N_QUBITS,
        target_strength,
        method_rng,
    )
    actual_strength = dsp.jump_strength(jumps)
    h0 = ising_xx_hamiltonian(couplings, H, N_QUBITS)
    hx = transverse_drive(N_QUBITS)
    reservoir = SparseLindbladReservoir.from_terms(
        N_QUBITS,
        h0 + H * hx,
        H * hx,
        jumps,
        DT,
    )
    observables = readout.pauli_observables(N_QUBITS, max_weight=2)
    features = reservoir.run(inputs, observables, washout=WASH)
    if features.shape != (TRAIN + VALIDATION + TEST, 45):
        raise RuntimeError(f"unexpected feature shape {features.shape}")

    train_stop = TRAIN
    validation_stop = TRAIN + VALIDATION
    raw_train = features[:train_stop]
    raw_validation = features[train_stop:validation_stop]
    raw_test = features[validation_stop:]
    guarded_train, guarded_validation, guarded_test, feature_meta = (
        train_only_feature_guard(raw_train, raw_validation, raw_test)
    )

    task_results = {}
    effective_rows = {}
    for task_name in TASKS:
        target = targets[task_name]
        y_train = target[:train_stop]
        y_validation = target[train_stop:validation_stop]
        y_test = target[validation_stop:]
        if task_name == "stm":
            result = select_and_refit_stm(
                guarded_train,
                y_train,
                guarded_validation,
                y_validation,
                guarded_test,
                y_test,
            )
            effective_rows[task_name] = {
                "train_by_target": result["effective_train_rows_by_target"],
                "validation": VALIDATION,
                "test": TEST,
                "refit_by_target": result["effective_refit_rows_by_target"],
            }
        else:
            train_valid = np.all(np.isfinite(y_train), axis=1)
            validation_valid = np.all(np.isfinite(y_validation), axis=1)
            test_valid = np.all(np.isfinite(y_test), axis=1)
            counts = {
                "train": int(np.sum(train_valid)),
                "validation": int(np.sum(validation_valid)),
                "test": int(np.sum(test_valid)),
            }
            if (
                counts["train"] != TRAIN - NARMA_ORDER
                or counts["validation"] != VALIDATION
                or counts["test"] != TEST
            ):
                raise RuntimeError(
                    "narma frozen split/effective-row invariant failed"
                )
            counts["refit"] = counts["train"] + counts["validation"]
            effective_rows[task_name] = counts
            result = select_and_refit(
                guarded_train[train_valid],
                y_train[train_valid],
                guarded_validation[validation_valid],
                y_validation[validation_valid],
                guarded_test[test_valid],
                y_test[test_valid],
                metric=TASK_METRICS[task_name],
            )
        if not all(
            math.isfinite(result[key])
            for key in ("selected_test", "fixed_test", "selection_improvement")
        ):
            raise RuntimeError(f"non-finite {task_name} test score")
        task_results[task_name] = result

    budget_error = float(actual_strength - target_strength)
    combined_targets = np.column_stack([targets["stm"], targets["narma"]])
    row = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol["source_environment_sha256"],
        "method": job.method,
        "method_label": METHOD_LABELS[job.method],
        "seed": int(job.seed),
        "n_qubits": N_QUBITS,
        "h": H,
        "dt": DT,
        "backend": "SparseLindbladReservoir exact expm_multiply",
        "wash_rows": WASH,
        "raw_train_rows": TRAIN,
        "raw_validation_rows": VALIDATION,
        "raw_test_rows": TEST,
        "effective_rows": effective_rows,
        "raw_feature_count": int(features.shape[1]),
        "feature_guard": feature_meta,
        "coupling_sha256": array_sha256(couplings),
        "full_input_sha256": array_sha256(inputs),
        "post_wash_input_sha256": array_sha256(post_wash_inputs),
        "target_sha256": {
            task_name: array_sha256(targets[task_name]) for task_name in TASKS
        },
        "train_split_sha256": split_hash(
            post_wash_inputs, combined_targets, 0, train_stop
        ),
        "validation_split_sha256": split_hash(
            post_wash_inputs, combined_targets, train_stop, validation_stop
        ),
        "test_split_sha256": split_hash(
            post_wash_inputs, combined_targets, validation_stop, len(combined_targets)
        ),
        "feature_mask_sha256": sha256_json(
            feature_meta["retained_nonbias_indices"]
        ),
        "jump_family_sha256": jump_family_sha256(jumps),
        "target_jump_strength": float(target_strength),
        "actual_jump_strength": float(actual_strength),
        "jump_strength_error": budget_error,
        "runtime_seconds": float(time.perf_counter() - started),
        "task_results": task_results,
    }
    return row


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


def _run_and_write(job: Job, protocol: dict, outdir_text: str) -> tuple[str, float]:
    outdir = Path(outdir_text)
    path = job_path(outdir, job)
    existing = _valid_checkpoint(path, job, protocol)
    if existing is not None:
        return f"skip {job.method} seed={job.seed}", float(
            existing["runtime_seconds"]
        )
    row = run_job(job, protocol)
    atomic_write_json(path, row)
    return f"done {job.method} seed={job.seed}", float(row["runtime_seconds"])


def all_jobs(protocol: dict) -> list[Job]:
    return [
        Job(method, int(seed))
        for method in protocol["methods"]
        for seed in protocol["seeds"]
    ]


def run_jobs(outdir: Path, protocol: dict, workers: int) -> None:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1, 8]")
    write_or_validate_protocol(outdir, protocol)
    pending = [
        job
        for job in all_jobs(protocol)
        if _valid_checkpoint(job_path(outdir, job), job, protocol) is None
    ]
    total = protocol["n_jobs"]
    completed = total - len(pending)
    print(f"{completed}/{total} validated checkpoints; {len(pending)} pending")
    if not pending:
        return
    if workers == 1:
        for job in pending:
            message, runtime = _run_and_write(job, protocol, str(outdir))
            completed += 1
            print(f"[{completed}/{total}] {message} ({runtime:.2f}s)", flush=True)
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
    with executor:
        futures = {
            executor.submit(_run_and_write, job, protocol, str(outdir)): job
            for job in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                message, runtime = future.result()
            except Exception as exc:
                for other in futures:
                    other.cancel()
                raise RuntimeError(
                    f"job failed for {job.method} seed={job.seed}"
                ) from exc
            completed += 1
            print(f"[{completed}/{total}] {message} ({runtime:.2f}s)", flush=True)


def load_rows(outdir: Path, protocol: dict, require_complete: bool = True) -> list[dict]:
    rows = []
    missing = []
    for job in all_jobs(protocol):
        row = _valid_checkpoint(job_path(outdir, job), job, protocol)
        if row is None:
            missing.append(job)
        else:
            rows.append(row)
    if require_complete and missing:
        preview = ", ".join(f"{job.method}/s{job.seed}" for job in missing[:5])
        raise RuntimeError(f"{len(missing)} checkpoints missing: {preview}")
    return rows


def invariant_audit(rows: Sequence[dict], protocol: dict) -> dict:
    errors: list[str] = []
    expected_keys = {
        (method, int(seed))
        for method in protocol["methods"]
        for seed in protocol["seeds"]
    }
    observed_keys = [(row.get("method"), row.get("seed")) for row in rows]
    if len(observed_keys) != len(set(observed_keys)):
        errors.append("duplicate method/seed rows")
    missing = sorted(expected_keys - set(observed_keys))
    extra = sorted(set(observed_keys) - expected_keys)
    if missing:
        errors.append(f"missing {len(missing)} method/seed rows")
    if extra:
        errors.append(f"unexpected {len(extra)} method/seed rows")

    psha = protocol_sha256(protocol)
    source_sha = protocol["source_environment_sha256"]
    expected_row_counts = {
        "wash_rows": WASH,
        "raw_train_rows": TRAIN,
        "raw_validation_rows": VALIDATION,
        "raw_test_rows": TEST,
        "raw_feature_count": 45,
    }
    expected_effective_rows = {
        "stm": {
            "train_by_target": [TRAIN - delay for delay in STM_DELAYS],
            "validation": VALIDATION,
            "test": TEST,
            "refit_by_target": [
                TRAIN + VALIDATION - delay for delay in STM_DELAYS
            ],
        },
        "narma": {
            "train": TRAIN - NARMA_ORDER,
            "validation": VALIDATION,
            "test": TEST,
            "refit": TRAIN - NARMA_ORDER + VALIDATION,
        },
    }
    ridge_set = set(float(value) for value in RIDGES)
    for row in rows:
        key = (row.get("method"), row.get("seed"))
        if row.get("protocol_sha256") != psha:
            errors.append(f"{key}: protocol hash mismatch")
        if row.get("source_environment_sha256") != source_sha:
            errors.append(f"{key}: source hash mismatch")
        if row.get("backend") != "SparseLindbladReservoir exact expm_multiply":
            errors.append(f"{key}: backend mismatch")
        for field, expected in expected_row_counts.items():
            if row.get(field) != expected:
                errors.append(f"{key}: {field} mismatch")
        if row.get("effective_rows") != expected_effective_rows:
            errors.append(f"{key}: task-specific effective rows mismatch")
        task_results = row.get("task_results", {})
        if set(task_results) != set(TASKS):
            errors.append(f"{key}: task-result coverage mismatch")
        for task_name in TASKS:
            task_result = task_results.get(task_name, {})
            selected = task_result.get("selected_ridge")
            if selected not in ridge_set:
                errors.append(f"{key}/{task_name}: selected ridge outside grid")
            if task_result.get("fixed_ridge") != FIXED_RIDGE:
                errors.append(f"{key}/{task_name}: fixed ridge mismatch")
            if task_result.get("metric") != TASK_METRICS[task_name]:
                errors.append(f"{key}/{task_name}: metric mismatch")
            expected_targets = len(STM_DELAYS) if task_name == "stm" else 1
            for field in ("selected_test_by_target", "fixed_test_by_target"):
                if len(task_result.get(field, [])) != expected_targets:
                    errors.append(f"{key}/{task_name}: {field} length mismatch")
            if task_name == "stm":
                if task_result.get("effective_train_rows_by_target") != [
                    TRAIN - delay for delay in STM_DELAYS
                ]:
                    errors.append(f"{key}/stm: per-delay train rows mismatch")
                if task_result.get("effective_refit_rows_by_target") != [
                    TRAIN + VALIDATION - delay for delay in STM_DELAYS
                ]:
                    errors.append(f"{key}/stm: per-delay refit rows mismatch")
            for field in ("selected_test", "fixed_test", "selection_improvement"):
                if not math.isfinite(float(task_result.get(field, math.nan))):
                    errors.append(f"{key}/{task_name}: non-finite {field}")
        if abs(float(row.get("jump_strength_error", math.inf))) > 1e-10:
            errors.append(f"{key}: dissipative budget mismatch")
        guard = row.get("feature_guard", {})
        if guard.get("fit_on") != "training rows only":
            errors.append(f"{key}: feature guard is not train-only")
        if guard.get("threshold") != FEATURE_STD_TOL:
            errors.append(f"{key}: feature threshold mismatch")
        if (
            guard.get("retained_nonbias_features", -1)
            + guard.get("dropped_nonbias_features", -1)
            != 45
        ):
            errors.append(f"{key}: feature accounting mismatch")
    pairing_fields = (
        "coupling_sha256",
        "full_input_sha256",
        "post_wash_input_sha256",
        "target_sha256",
        "train_split_sha256",
        "validation_split_sha256",
        "test_split_sha256",
    )
    rows_by_seed: dict[int, list[dict]] = {}
    for row in rows:
        rows_by_seed.setdefault(int(row["seed"]), []).append(row)
    for seed, seed_rows in sorted(rows_by_seed.items()):
        if len(seed_rows) != len(protocol["methods"]):
            continue
        for field in pairing_fields:
            if len({canonical_json(row[field]) for row in seed_rows}) != 1:
                errors.append(f"seed {seed}: unpaired {field}")

    return {
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "expected_jobs": len(expected_keys),
        "observed_jobs": len(rows),
        "pairing_fields": list(pairing_fields),
    }


def ridge_boundary_audit(rows: Sequence[dict]) -> dict:
    maximum = max(RIDGES)
    previous = sorted(RIDGES)[-2]
    at_maximum = []
    unresolved = []
    for row in rows:
        for task_name in TASKS:
            result = row["task_results"][task_name]
            if float(result["selected_ridge"]) != maximum:
                continue
            identity = {
                "task": task_name,
                "method": row["method"],
                "seed": int(row["seed"]),
            }
            at_maximum.append(identity)
            scores = result["validation_by_ridge"]
            maximum_score = float(scores[f"{maximum:.12g}"])
            previous_score = float(scores[f"{previous:.12g}"])
            still_improving = (
                maximum_score > previous_score + 1e-12
                if TASK_METRICS[task_name] == "capacity"
                else maximum_score < previous_score - 1e-12
            )
            if still_improving:
                unresolved.append(
                    {
                        **identity,
                        "previous_ridge": previous,
                        "previous_score": previous_score,
                        "maximum_ridge": maximum,
                        "maximum_score": maximum_score,
                    }
                )
    at_maximum.sort(key=lambda item: (item["task"], item["method"], item["seed"]))
    unresolved.sort(key=lambda item: (item["task"], item["method"], item["seed"]))
    return {
        "passed": not unresolved,
        "maximum_predeclared_ridge": maximum,
        "selected_at_maximum_count": len(at_maximum),
        "selected_at_maximum_by_task": {
            task_name: int(sum(item["task"] == task_name for item in at_maximum))
            for task_name in TASKS
        },
        "selected_at_maximum": at_maximum,
        "unresolved_upper_boundary_count": len(unresolved),
        "unresolved_upper_boundary": unresolved,
        "criterion": (
            "no task/design/seed selected at the maximum may still improve "
            "relative to the preceding ridge"
        ),
    }


def feature_guard_audit(rows: Sequence[dict]) -> dict:
    by_method = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        dropped = [
            int(row["feature_guard"]["dropped_nonbias_features"])
            for row in method_rows
        ]
        retained = [
            int(row["feature_guard"]["retained_nonbias_features"])
            for row in method_rows
        ]
        by_method[method] = {
            "jobs": len(method_rows),
            "jobs_with_drops": int(sum(value > 0 for value in dropped)),
            "dropped_min": min(dropped),
            "dropped_max": max(dropped),
            "dropped_mean": float(np.mean(dropped)),
            "retained_min": min(retained),
            "retained_max": max(retained),
        }
    return {
        "passed": all(
            row["feature_guard"]["fit_on"] == "training rows only"
            and row["feature_guard"]["retained_features_including_bias"]
            == row["feature_guard"]["retained_nonbias_features"] + 1
            for row in rows
        ),
        "threshold": FEATURE_STD_TOL,
        "by_method": by_method,
    }


def baseline_reproduction_audit(rows: Sequence[dict], protocol: dict) -> dict:
    """Compare fixed-1e-8 scores with the sealed primary A-table checkpoints."""
    manifest = protocol["baseline_reproduction"]
    atol = float(manifest["absolute_tolerance"])
    rtol = float(manifest["relative_tolerance"])
    exceptions = set(manifest["guard_exception_methods"])
    active_comparisons = []
    guarded_exceptions = []
    violations = []
    by_task_differences = {task_name: [] for task_name in TASKS}
    for row in sorted(rows, key=lambda item: (item["method"], int(item["seed"]))):
        method = row["method"]
        seed = int(row["seed"])
        dropped = int(row["feature_guard"]["dropped_nonbias_features"])
        if method not in exceptions and dropped:
            violations.append(
                {
                    "task": "feature_guard",
                    "method": method,
                    "seed": seed,
                    "reason": f"active baseline row dropped {dropped} features",
                }
            )
        for task_name in TASKS:
            key = f"{task_name}/{method}/{seed}"
            entry = manifest["entries"].get(key)
            if entry is None:
                violations.append(
                    {
                        "task": task_name,
                        "method": method,
                        "seed": seed,
                        "reason": "sealed baseline entry missing from protocol",
                    }
                )
                continue
            path = ROOT / entry["path"]
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                violations.append(
                    {
                        "task": task_name,
                        "method": method,
                        "seed": seed,
                        "reason": "sealed baseline file/hash mismatch",
                    }
                )
                continue
            expected = float(entry["value"])
            observed = float(row["task_results"][task_name]["fixed_test"])
            difference = observed - expected
            comparison = {
                "task": task_name,
                "method": method,
                "seed": seed,
                "observed_fixed": observed,
                "sealed_primary": expected,
                "difference": difference,
                "absolute_difference": abs(difference),
            }
            if method in exceptions:
                guarded_exceptions.append(
                    {
                        **comparison,
                        "reason": manifest["guard_exception_reason"],
                    }
                )
                continue
            tolerance = atol + rtol * abs(expected)
            comparison["tolerance"] = tolerance
            comparison["within_tolerance"] = abs(difference) <= tolerance
            active_comparisons.append(comparison)
            by_task_differences[task_name].append(abs(difference))
            if not comparison["within_tolerance"]:
                violations.append(
                    {**comparison, "reason": "fixed score drifted from baseline"}
                )
    return {
        "passed": not violations,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "active_comparison_count": len(active_comparisons),
        "guarded_exception_count": len(guarded_exceptions),
        "guard_exception_methods": sorted(exceptions),
        "maximum_active_absolute_difference_by_task": {
            task_name: (
                float(max(differences)) if differences else None
            )
            for task_name, differences in by_task_differences.items()
        },
        "violations": violations,
        "active_comparisons": active_comparisons,
        "guarded_exceptions": guarded_exceptions,
    }


def mean_se_ci(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sample")
    mean = float(np.mean(array))
    if array.size == 1:
        se = 0.0
        low = high = mean
    else:
        se = float(np.std(array, ddof=1) / np.sqrt(array.size))
        half = float(student_t.ppf(0.975, array.size - 1) * se)
        low, high = mean - half, mean + half
    return {
        "n": int(array.size),
        "mean": mean,
        "se": se,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def build_aggregate(rows: Sequence[dict], protocol: dict) -> dict:
    rows = sorted(rows, key=lambda row: (row["method"], int(row["seed"])))
    invariants = invariant_audit(rows, protocol)
    boundary = ridge_boundary_audit(rows)
    guard = feature_guard_audit(rows)
    reproduction = baseline_reproduction_audit(rows, protocol)

    by_method_rows = {
        method: [row for row in rows if row["method"] == method]
        for method in protocol["methods"]
    }
    reference = {
        int(row["seed"]): row for row in by_method_rows[REFERENCE_METHOD]
    }
    task_summaries = {}
    for task_name in TASKS:
        metric = TASK_METRICS[task_name]
        higher_is_better = metric == "capacity"
        method_summaries = {}
        for method, method_rows in by_method_rows.items():
            ridge_counts = {
                f"{ridge:.12g}": int(
                    sum(
                        float(
                            row["task_results"][task_name]["selected_ridge"]
                        )
                        == ridge
                        for row in method_rows
                    )
                )
                for ridge in RIDGES
            }
            results = [row["task_results"][task_name] for row in method_rows]
            method_summaries[method] = {
                "label": METHOD_LABELS[method],
                "selected_test": mean_se_ci(
                    result["selected_test"] for result in results
                ),
                "fixed_test": mean_se_ci(
                    result["fixed_test"] for result in results
                ),
                "selection_improvement": mean_se_ci(
                    result["selection_improvement"] for result in results
                ),
                "selection_better_than_fixed_count": int(
                    sum(result["selection_improvement"] > 0 for result in results)
                ),
                "selection_equal_to_fixed_count": int(
                    sum(
                        np.isclose(
                            result["selection_improvement"],
                            0.0,
                            rtol=0.0,
                            atol=1e-15,
                        )
                        for result in results
                    )
                ),
                "selected_ridge_counts": ridge_counts,
            }

        paired_vs_local = {}
        for method, method_rows in by_method_rows.items():
            if method == REFERENCE_METHOD:
                continue
            differences = []
            for row in method_rows:
                local_value = float(
                    reference[int(row["seed"])]["task_results"][task_name][
                        "selected_test"
                    ]
                )
                method_value = float(
                    row["task_results"][task_name]["selected_test"]
                )
                differences.append(
                    method_value - local_value
                    if higher_is_better
                    else local_value - method_value
                )
            paired_vs_local[method] = {
                "label": METHOD_LABELS[method],
                "method_advantage_over_local": mean_se_ci(differences),
                "method_better_count": int(sum(value > 0 for value in differences)),
                "ties": int(sum(value == 0 for value in differences)),
                "interpretation": (
                    "positive values favour this method over uniform local"
                ),
            }

        ranking = [
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "mean_selected_test": summary["selected_test"]["mean"],
            }
            for method, summary in method_summaries.items()
        ]
        ranking.sort(
            key=lambda item: (
                -item["mean_selected_test"]
                if higher_is_better
                else item["mean_selected_test"],
                item["method"],
            )
        )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank

        winner_counts = {method: 0 for method in protocol["methods"]}
        for seed in protocol["seeds"]:
            seed_rows = [row for row in rows if int(row["seed"]) == int(seed)]
            winner = sorted(
                seed_rows,
                key=lambda row: (
                    -row["task_results"][task_name]["selected_test"]
                    if higher_is_better
                    else row["task_results"][task_name]["selected_test"],
                    row["method"],
                ),
            )[0]
            winner_counts[winner["method"]] += 1
        task_summaries[task_name] = {
            "metric": metric,
            "direction": "higher is better" if higher_is_better else "lower is better",
            "ranking": ranking,
            "method_summaries": method_summaries,
            "paired_vs_uniform_local": paired_vs_local,
            "per_seed_winner_counts": winner_counts,
        }

    runtimes = [float(row["runtime_seconds"]) for row in rows]
    aggregate = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol["source_environment_sha256"],
        "n_jobs": len(rows),
        "status": (
            "complete"
            if (
                invariants["passed"]
                and boundary["passed"]
                and guard["passed"]
                and reproduction["passed"]
            )
            else "invalid"
        ),
        "invariant_audit": invariants,
        "ridge_boundary_audit": boundary,
        "feature_guard_audit": guard,
        "baseline_reproduction_audit": reproduction,
        "task_summaries": task_summaries,
        "runtime": {
            "sum_job_seconds": float(sum(runtimes)),
            "mean_job_seconds": float(np.mean(runtimes)),
            "max_job_seconds": float(max(runtimes)),
        },
        "rows": rows,
    }
    return aggregate


def aggregate_results(outdir: Path, protocol: dict) -> dict:
    rows = load_rows(outdir, protocol, require_complete=True)
    aggregate = build_aggregate(rows, protocol)
    if aggregate["status"] != "complete":
        problems = aggregate["invariant_audit"]["errors"]
        boundary = aggregate["ridge_boundary_audit"]
        reproduction = aggregate["baseline_reproduction_audit"]
        raise RuntimeError(
            "control failed validation: "
            + "; ".join(
                problems
                + (
                    [
                        f"{boundary['unresolved_upper_boundary_count']} fits "
                        "remain unresolved at the upper ridge boundary"
                    ]
                    if not boundary["passed"]
                    else []
                )
                + (
                    [
                        f"{len(reproduction['violations'])} fixed-score "
                        "baseline-reproduction violations"
                    ]
                    if not reproduction["passed"]
                    else []
                )
            )
        )
    atomic_write_json(aggregate_path(outdir), aggregate)
    return aggregate


def _fmt_ci(summary: dict) -> str:
    return (
        f"{summary['mean']:.6g} "
        f"[95% CI {summary['ci95_low']:.6g}, {summary['ci95_high']:.6g}]"
    )


def render_report(aggregate: dict, protocol: dict) -> str:
    task_sections = []
    for task_name in TASKS:
        task = aggregate["task_summaries"][task_name]
        task_label = "STM capacity (delays 1–20)" if task_name == "stm" else "NARMA-10 NMSE"
        ranking_rows = []
        for row in task["ranking"]:
            summary = task["method_summaries"][row["method"]]
            ranking_rows.append(
                "| {rank} | {label} (`{method}`) | {selected} | {fixed} | "
                "{improvement} |".format(
                    rank=row["rank"],
                    label=row["label"],
                    method=row["method"],
                    selected=_fmt_ci(summary["selected_test"]),
                    fixed=_fmt_ci(summary["fixed_test"]),
                    improvement=_fmt_ci(summary["selection_improvement"]),
                )
            )
        effect_rows = []
        for method in protocol["methods"]:
            if method == REFERENCE_METHOD:
                continue
            effect = task["paired_vs_uniform_local"][method]
            effect_rows.append(
                f"| {effect['label']} (`{method}`) | "
                f"{_fmt_ci(effect['method_advantage_over_local'])} | "
                f"{effect['method_better_count']}/{len(protocol['seeds'])} |"
            )
        task_sections.append(
            f"""## {task_label}

Metric direction: **{task['direction']}**.  The “selection improvement” column
is signed so positive always means validation-selected ridge improved on the
fixed-1e-8 result.

| Rank | Dissipator | selected-ridge test | fixed-1e-8 test | selection improvement |
|---:|---|---:|---:|---:|
{chr(10).join(ranking_rows)}

### Paired effects versus uniform local loss

The paired advantage is signed so positive always favours the listed method.

| Dissipator | paired method advantage | seed wins |
|---|---:|---:|
{chr(10).join(effect_rows)}

Per-seed winner counts:
`{json.dumps(task['per_seed_winner_counts'], sort_keys=True)}`.
"""
        )
    guard_rows = []
    for method in protocol["methods"]:
        item = aggregate["feature_guard_audit"]["by_method"][method]
        guard_rows.append(
            f"| {METHOD_LABELS[method]} | {item['jobs_with_drops']}/{item['jobs']} | "
            f"{item['dropped_min']}–{item['dropped_max']} |"
        )
    runtime = aggregate["runtime"]
    return f"""# Primary STM/NARMA readout-regularisation control

## Result

The isolated control completed all {aggregate['n_jobs']} paired jobs
({len(protocol['methods'])} dissipators × {len(protocol['seeds'])} seeds), with
STM and NARMA-10 scored from every shared trajectory.  Every score uses the
frozen 200 wash / 450 train / 150 validation / 400 untouched test protocol.
Ridge selection uses validation only; both selected and fixed-1e-8 fits are
refitted on train+validation.  STM delay `d` retains its own `450-d` training
rows and `600-d` refit rows, exactly preserving the primary per-delay fits.

{chr(10).join(task_sections)}

## Readout and integrity diagnostics

- Upper ridge boundary bracketed: **{aggregate['ridge_boundary_audit']['passed']}**
  ({aggregate['ridge_boundary_audit']['selected_at_maximum_count']}
  task/design/seed fits at ridge {max(RIDGES):g};
  `{json.dumps(aggregate['ridge_boundary_audit']['selected_at_maximum_by_task'], sort_keys=True)}`;
  {aggregate['ridge_boundary_audit']['unresolved_upper_boundary_count']}
  still improving at the boundary).
- Pairing, split, source, budget, and completeness audit:
  **{aggregate['invariant_audit']['passed']}**.
- Fixed-1e-8 reproduction of the sealed primary table:
  **{aggregate['baseline_reproduction_audit']['passed']}**
  ({aggregate['baseline_reproduction_audit']['active_comparison_count']} active
  task/design/seed comparisons; maximum absolute differences
  `{json.dumps(aggregate['baseline_reproduction_audit']['maximum_active_absolute_difference_by_task'], sort_keys=True)}`;
  {aggregate['baseline_reproduction_audit']['guarded_exception_count']} guarded
  dephasing comparisons reported separately).

| Dissipator | jobs with guarded features | dropped-feature range |
|---|---:|---:|
{chr(10).join(guard_rows)}

The guard threshold was {FEATURE_STD_TOL:g}, estimated on raw training rows
only; the bias was always retained.

## Runtime and provenance

- Sum of job runtimes: {runtime['sum_job_seconds']:.1f} s
- Mean / maximum job runtime: {runtime['mean_job_seconds']:.2f} /
  {runtime['max_job_seconds']:.2f} s
- Protocol SHA-256: `{aggregate['protocol_sha256']}`
- Source-environment SHA-256: `{aggregate['source_environment_sha256']}`

Raw per-job checkpoints, protocol, and the machine-readable aggregate are in
`results/revision_primary_regularization/`.
"""


def write_report(report_path: Path, aggregate: dict, protocol: dict) -> None:
    atomic_write_text(report_path, render_report(aggregate, protocol))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("run", "aggregate", "report", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first seed for local and dephasing in an isolated subdirectory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.smoke:
        methods = ("CD_paper", "B1_dephasing")
        seeds = deterministic_seeds(32)[:1]
        outdir = args.outdir / "smoke"
        report_path = args.report.with_name(args.report.stem + "_smoke.md")
    else:
        methods = METHODS
        seeds = deterministic_seeds(32)
        outdir = args.outdir
        report_path = args.report
    protocol = build_protocol(methods, seeds)
    write_or_validate_protocol(outdir, protocol)
    if args.command in ("run", "all"):
        run_jobs(outdir, protocol, args.workers)
    if args.command in ("aggregate", "all"):
        aggregate = aggregate_results(outdir, protocol)
    elif args.command == "report":
        path = aggregate_path(outdir)
        if not path.exists():
            raise FileNotFoundError(f"aggregate missing: {path}")
        aggregate = json.loads(path.read_text())
    else:
        aggregate = None
    if args.command in ("report", "all"):
        assert aggregate is not None
        write_report(report_path, aggregate, protocol)
        print(f"report: {report_path}")
    if args.command in ("aggregate", "all"):
        print(f"aggregate: {aggregate_path(outdir)}")


if __name__ == "__main__":
    main()

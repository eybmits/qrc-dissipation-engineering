"""Frozen all-family finite-size extension of Experiment 1.

This driver evaluates the eight architectures from Experiment 1 at
``N = 4, ..., 8`` on one shared continuous-input trajectory per
``(N, architecture, lineage)``.  Both STM and NARMA-10 are scored from that
trajectory.  The protocol preserves the primary Experiment-1 task definition:

* 200 washout, 600 fitting, and 400 untouched test inputs;
* fixed ridge ``1e-8`` as the primary score;
* the existing 450/150 split of the 600 fitting rows as a validation-selected
  readout control, with no additional reservoir evolution;
* all weight-one and same-axis weight-two Pauli observables;
* fixed within-size Frobenius jump weight.

For the finite-size comparison, each lineage starts from one N=8 coupling draw.
Leading principal submatrices pair N=4,...,8, and the multiplier
``sqrt(4/(N-1))`` fixes expected per-site interaction variance while remaining
exactly one at N=5.

Every expensive trajectory is written atomically to one JSON checkpoint.
Existing checkpoints are skipped only after their identity, protocol hash,
source hash, task scores, and physical invariants validate.  A failed job is
recorded separately and is retried on the next invocation.

Examples
--------
Freeze and inspect the 960-job production manifest::

    python experiments/run_experiment1_finite_size.py freeze
    python experiments/run_experiment1_finite_size.py status

Run one N=8 lineage as a timing/implementation check, then resume everything::

    python experiments/run_experiment1_finite_size.py run \
        --n-values 8 --seed-indices 0 --workers 1
    python experiments/run_experiment1_finite_size.py run --workers 4

Aggregate only after all checkpoints are complete::

    python experiments/run_experiment1_finite_size.py aggregate
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
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import _paths  # noqa: F401
import numpy as np
import scipy
from scipy.stats import t as student_t

from _paths import RESULTS_DIR
from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir
from run_final_scaling import build_jumps
import run_revision_controls as revision
import run_revision_primary_regularization as primary_readout


PROTOCOL_VERSION = "experiment1-finite-size-v2-2026-07-29"
SEED_NAMESPACE = 2026072902
SUPERSEDED_V1_SEEDS = (
    969779107,
    466586310,
    652303082,
    1806021257,
    934059971,
    1852093697,
    69698774,
    261879255,
    723856339,
    1782130787,
    1005234531,
    866066252,
    1241087361,
    195665586,
    1266633660,
    417477370,
    1202721483,
    515208926,
    410577769,
    252433623,
    1827080111,
    594649085,
    446873694,
    1041783474,
)
N_VALUES = (4, 5, 6, 7, 8)
N_LINEAGES = 24
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
    "B3_collective": "collective loss",
    "A1_heterogeneous": "unequal local loss",
    "B5_pair": "pair loss",
    "B2_thermal": "local gain/loss",
    "B4_loss_exchange": "exchange-assisted loss",
    "B1_dephasing": "dephasing",
}
REFERENCE_METHOD = "CD_paper"
COLLECTIVE_METHOD = "B3_collective"
DISSIPATIVE_METHODS = tuple(method for method in METHODS if method != "FN")

H = 0.5
DT = 0.5
GAMMA = 1.0
WASH = 200
TRAIN = 450
VALIDATION = 150
TEST = 400
FIT_TOTAL = TRAIN + VALIDATION
TOTAL_INPUTS = WASH + FIT_TOTAL + TEST
STM_DELAYS = tuple(range(1, 21))
NARMA_ORDER = 10
NARMA_INPUT_SCALE = 0.2
FIXED_RIDGE = 1e-8
RIDGES = primary_readout.RIDGES
FEATURE_STD_TOL = primary_readout.FEATURE_STD_TOL
SIGN_FLIP_DRAWS = 100_000
BOOTSTRAP_DRAWS = 20_000
COUPLING_SCHEME = "variance"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = Path(RESULTS_DIR) / "experiment1_finite_size_v2"
SOURCE_FILES = (
    "experiments/_paths.py",
    "experiments/run_experiment1_finite_size.py",
    "experiments/run_final_scaling.py",
    "experiments/run_revision_controls.py",
    "experiments/run_revision_primary_regularization.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)


@dataclass(frozen=True)
class Job:
    n_qubits: int
    method: str
    seed: int

    @property
    def key(self) -> tuple[int, str, int]:
        return self.n_qubits, self.method, self.seed


class CheckpointError(RuntimeError):
    """Raised when an existing checkpoint cannot be trusted or overwritten."""


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
    return primary_readout.array_sha256(np.asarray(value))


def checkpoint_payload_sha256(payload: dict) -> str:
    unsigned = dict(payload)
    unsigned.pop("checkpoint_payload_sha256", None)
    return sha256_json(unsigned)


def seal_checkpoint(payload: dict) -> dict:
    sealed = dict(payload)
    sealed["checkpoint_payload_sha256"] = checkpoint_payload_sha256(sealed)
    return sealed


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
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


def excluded_seed_pools() -> dict[str, list[int]]:
    """Seed pools reserved or outcome-observed before this protocol was frozen."""
    return {
        "definitive_experiment1": sorted(revision.legacy_seeds()),
        "revision_normalized_scaling": revision.fresh_seeds(
            revision.SCALING_SEED_NAMESPACE, 8
        ),
        "revision_parity": revision.fresh_seeds(
            revision.PARITY_SEED_NAMESPACE, 16
        ),
        "superseded_v1_smoke_protocol": list(SUPERSEDED_V1_SEEDS),
    }


def production_seeds(n_lineages: int = N_LINEAGES) -> list[int]:
    if n_lineages < 1:
        raise ValueError("n_lineages must be positive")
    excluded = {
        seed
        for pool in excluded_seed_pools().values()
        for seed in pool
    }
    rng = np.random.default_rng(SEED_NAMESPACE)
    seeds: list[int] = []
    while len(seeds) < n_lineages:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in excluded and candidate not in seeds:
            seeds.append(candidate)
    return seeds


def expected_observable_count(n_qubits: int) -> int:
    return 3 * n_qubits + 3 * math.comb(n_qubits, 2)


def build_protocol() -> dict:
    seeds = production_seeds()
    source = source_environment_manifest()
    excluded = excluded_seed_pools()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Do the all-family STM and NARMA-10 performance profiles from "
            "Experiment 1 persist or change over the tested finite range N=4,...,8?"
        ),
        "claim_boundary": (
            "A paired finite-size comparison over N=4,...,8 under the declared "
            "normalisation; not an asymptotic scaling law, hardware-scaling "
            "claim, or universal Hamiltonian-independent ordering."
        ),
        "n_values": list(N_VALUES),
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "reference_method": REFERENCE_METHOD,
        "seeds": seeds,
        "n_lineages": N_LINEAGES,
        "seed_namespace": SEED_NAMESPACE,
        "excluded_seed_pools": excluded,
        "excluded_seed_pools_sha256": sha256_json(excluded),
        "seeds_are_fresh": not bool(
            set(seeds)
            & {
                seed
                for pool in excluded.values()
                for seed in pool
            }
        ),
        "n_jobs": len(N_VALUES) * len(METHODS) * len(seeds),
        "pairing": {
            "couplings": (
                "one N=8 U[-1,1] draw per lineage; leading principal "
                "submatrices for N=4,...,8"
            ),
            "inputs": (
                "one iid Uniform[0,1] sequence per lineage, reused across "
                "all N and architectures"
            ),
            "targets_and_splits": "identical within each lineage and N",
            "method_randomness": (
                "deterministic SHA-256-derived method seed; unequal-local "
                "rates are regenerated consistently for each N"
            ),
        },
        "hamiltonian": {
            "coupling_distribution": "symmetric complete graph, J_ij~U[-1,1]",
            "normalisation": "sqrt(4/(N-1))",
            "normalisation_purpose": (
                "fix expected per-site interaction variance and equal one at N=5"
            ),
            "h": H,
            "dt": DT,
            "initial_state": "|0...0><0...0|",
        },
        "dissipation": {
            "budget": "sum_k rate_k Tr(L_k^dagger L_k)",
            "reference": "unit-rate uniform local loss separately at each N",
            "matching": "all seven dissipative designs match within each N",
            "cross_n_note": (
                "the numerical budget follows the N-qubit unit-rate local "
                "reference and is not one N-independent scalar"
            ),
            "fn_boundary": "FN is reset-unitary and has no jump budget",
        },
        "tasks": {
            "shared_trajectory": True,
            "input_distribution": "iid Uniform[0,1]",
            "stm": {
                "delays": list(STM_DELAYS),
                "metric": "summed squared-correlation capacity; higher is better",
            },
            "narma10": {
                "order": NARMA_ORDER,
                "input_scale": NARMA_INPUT_SCALE,
                "metric": "NMSE; lower is better",
            },
        },
        "split": {
            "wash": WASH,
            "primary_fit": FIT_TOTAL,
            "validation_control_train": TRAIN,
            "validation_control_validation": VALIDATION,
            "test": TEST,
            "total_inputs": TOTAL_INPUTS,
            "test_is_untouched": True,
        },
        "readout": {
            "rule": (
                "all weight-one and same-axis weight-two Pauli expectations "
                "plus a fitted bias"
            ),
            "observable_counts": {
                str(n_qubits): expected_observable_count(n_qubits)
                for n_qubits in N_VALUES
            },
            "primary": {
                "ridge": FIXED_RIDGE,
                "fit_rows": FIT_TOTAL,
                "matches_experiment1": True,
            },
            "validation_control": {
                "ridge_grid": list(RIDGES),
                "feature_guard_fit_on": "first 450 fitting rows only",
                "feature_guard_std_threshold": FEATURE_STD_TOL,
                "selection": "450 train / 150 validation",
                "refit": "train plus validation before the untouched test",
            },
        },
        "backend": {
            "dissipative": "exact sparse expm_multiply; continuous inputs",
            "fn": "exact reset-unitary propagation",
            "input_quantisation": False,
        },
        "prespecified_analysis": {
            "primary_estimand": (
                "paired collective-minus-uniform-local fixed-ridge STM "
                "difference at each N"
            ),
            "finite_range_contrast": (
                "within-lineage collective-versus-local relative STM effect "
                "at N=8 minus the corresponding effect at N=4"
            ),
            "finite_range_slope": (
                "within-lineage OLS slope of the collective-versus-local "
                "relative STM effect over all five tested N values"
            ),
            "confirmatory_family": (
                "six non-reference dissipative-versus-local contrasts across "
                "five N and two tasks, plus the finite-range endpoint and slope "
                "contrasts; 62 two-sided exact paired sign-flip tests with Holm "
                "FWER control"
            ),
            "fn_reference": "descriptive and outside the dissipative test family",
            "validation_selected_scores": "readout robustness control, not a new task",
            "winner_language": (
                "largest mean is descriptive unless simultaneously separated "
                "from every competitor"
            ),
        },
        "source_environment": source,
        "source_environment_sha256": sha256_json(source),
    }
    if protocol["n_jobs"] != 960:
        raise RuntimeError("production manifest must contain exactly 960 jobs")
    if not protocol["seeds_are_fresh"]:
        raise RuntimeError("production seeds overlap a sealed result pool")
    return protocol


def protocol_sha256(protocol: dict) -> str:
    return sha256_json(protocol)


def protocol_path(outdir: Path) -> Path:
    return outdir / "protocol.json"


def write_or_validate_protocol(
    outdir: Path, protocol: dict, *, create: bool = False
) -> None:
    path = protocol_path(outdir)
    if path.exists():
        try:
            stored = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"corrupt frozen protocol: {path}") from exc
        if stored != protocol:
            raise CheckpointError(
                f"protocol or source drift at {path}; use a new output directory"
            )
        return
    if not create:
        raise CheckpointError(
            f"frozen protocol missing at {path}; run the freeze command first"
        )
    outdir.mkdir(parents=True, exist_ok=True)
    lock_path = outdir / ".protocol-freeze.lock"
    try:
        descriptor = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
    except FileExistsError as exc:
        raise CheckpointError(
            f"another process is freezing the protocol: {lock_path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(f"pid={os.getpid()}\n")
        if path.exists():
            stored = json.loads(path.read_text())
            if stored != protocol:
                raise CheckpointError(
                    f"protocol or source drift at {path}"
                )
        else:
            atomic_write_json(path, protocol)
    finally:
        lock_path.unlink(missing_ok=True)


@contextmanager
def exclusive_run_lock(outdir: Path, protocol: dict):
    """Prevent two local production runners from duplicating expensive jobs."""
    lock_path = outdir / ".run.lock"
    identity = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "protocol_sha256": protocol_sha256(protocol),
        "started_unix_time": time.time(),
    }
    while True:
        try:
            descriptor = os.open(
                lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
            )
            break
        except FileExistsError as exc:
            try:
                existing = json.loads(lock_path.read_text())
            except (OSError, json.JSONDecodeError):
                raise CheckpointError(f"corrupt run lock: {lock_path}") from exc
            same_host = existing.get("hostname") == socket.gethostname()
            pid = existing.get("pid")
            alive = False
            if same_host and isinstance(pid, int):
                try:
                    os.kill(pid, 0)
                    alive = True
                except ProcessLookupError:
                    alive = False
                except PermissionError:
                    alive = True
            if alive or not same_host:
                raise CheckpointError(
                    f"another production runner owns {lock_path}: {existing}"
                ) from exc
            lock_path.unlink()
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(identity, handle, sort_keys=True)
            handle.write("\n")
        yield
    finally:
        try:
            stored = json.loads(lock_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored = None
        if stored == identity:
            lock_path.unlink(missing_ok=True)


def job_path(outdir: Path, job: Job) -> Path:
    return (
        outdir
        / "checkpoints"
        / f"N{job.n_qubits}"
        / f"{job.method}__s{job.seed}.json"
    )


def failure_path(outdir: Path, job: Job) -> Path:
    return (
        outdir
        / "failures"
        / f"N{job.n_qubits}"
        / f"{job.method}__s{job.seed}.json"
    )


def all_jobs(protocol: dict) -> list[Job]:
    """Order cheap sizes first while completing every paired lineage together."""
    return [
        Job(n_qubits, method, int(seed))
        for n_qubits in protocol["n_values"]
        for seed in protocol["seeds"]
        for method in protocol["methods"]
    ]


def selected_jobs(
    protocol: dict,
    n_values: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    seed_indices: Sequence[int] | None = None,
) -> list[Job]:
    n_filter = set(protocol["n_values"] if n_values is None else n_values)
    method_filter = set(protocol["methods"] if methods is None else methods)
    if not n_filter.issubset(set(protocol["n_values"])):
        raise ValueError("N filter contains a value outside the frozen protocol")
    if not method_filter.issubset(set(protocol["methods"])):
        raise ValueError("method filter contains a value outside the frozen protocol")
    if seed_indices is None:
        seed_filter = set(protocol["seeds"])
    else:
        if any(index < 0 or index >= len(protocol["seeds"]) for index in seed_indices):
            raise ValueError("seed index lies outside the frozen protocol")
        seed_filter = {protocol["seeds"][index] for index in seed_indices}
    return [
        job
        for job in all_jobs(protocol)
        if job.n_qubits in n_filter
        and job.method in method_filter
        and job.seed in seed_filter
    ]


def _input_sequence(seed: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A2B3C4D]))
    return tasks.stm_inputs(TOTAL_INPUTS, rng)


def _method_seed(seed: int, method: str) -> int:
    digest = hashlib.sha256(f"{seed}:{method}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _jump_family_for_job(
    job: Job, couplings: np.ndarray
) -> tuple[list[tuple[np.ndarray, float]] | None, float, float | None]:
    target_strength = float(
        dsp.jump_strength(dsp.local_loss(job.n_qubits, GAMMA))
    )
    if job.method == "FN":
        return None, target_strength, None
    method_rng = np.random.default_rng(_method_seed(job.seed, job.method))
    jumps = build_jumps(
        job.method,
        couplings,
        job.n_qubits,
        target_strength,
        method_rng,
    )
    actual_strength = float(dsp.jump_strength(jumps))
    if not np.isclose(
        actual_strength, target_strength, rtol=1e-10, atol=1e-12
    ):
        raise RuntimeError(
            f"jump budget mismatch for {job}: "
            f"{actual_strength} != {target_strength}"
        )
    return jumps, target_strength, actual_strength


def _build_reservoir(
    job: Job, couplings: np.ndarray
) -> tuple[object, list[tuple[np.ndarray, float]] | None, float, float | None]:
    jumps, target_strength, actual_strength = _jump_family_for_job(
        job, couplings
    )
    if job.method == "FN":
        reservoir = res.FujiNakajimaReservoir(
            job.n_qubits, couplings, H, DT
        )
        return reservoir, jumps, target_strength, actual_strength
    h0 = ising_xx_hamiltonian(couplings, H, job.n_qubits)
    hx = transverse_drive(job.n_qubits)
    reservoir = SparseLindbladReservoir.from_terms(
        job.n_qubits, h0 + H * hx, H * hx, jumps, DT
    )
    return reservoir, jumps, target_strength, actual_strength


def _primary_fixed_score(
    features: np.ndarray, targets: np.ndarray, metric: str
) -> dict:
    x_fit = readout.add_bias(features[:FIT_TOTAL])
    x_test = readout.add_bias(features[FIT_TOTAL:])
    target = np.asarray(targets, dtype=float)
    if target.ndim == 1:
        target = target[:, None]
    scores = []
    effective_fit_rows = []
    for column in range(target.shape[1]):
        y_fit = target[:FIT_TOTAL, column]
        y_test = target[FIT_TOTAL:, column]
        fit_valid = np.isfinite(y_fit)
        test_valid = np.isfinite(y_test)
        if not np.all(test_valid):
            raise RuntimeError("undefined target survived into the primary test")
        weights = readout.train_readout(
            x_fit[fit_valid], y_fit[fit_valid], ridge=FIXED_RIDGE
        )
        prediction = readout.predict(x_test[test_valid], weights)
        if metric == "capacity":
            score = readout.capacity(y_test[test_valid], prediction)
        elif metric == "nmse":
            score = readout.nmse(y_test[test_valid], prediction)
        else:
            raise ValueError(f"unsupported metric {metric!r}")
        scores.append(float(score))
        effective_fit_rows.append(int(np.sum(fit_valid)))
    total = float(sum(scores) if metric == "capacity" else np.mean(scores))
    return {
        "metric": metric,
        "ridge": FIXED_RIDGE,
        "test": total,
        "test_by_target": scores,
        "effective_fit_rows_by_target": effective_fit_rows,
        "test_rows": TEST,
    }


def _validation_control(
    features: np.ndarray,
    targets: dict[str, np.ndarray],
) -> tuple[dict[str, dict], dict]:
    raw_train = features[:TRAIN]
    raw_validation = features[TRAIN:FIT_TOTAL]
    raw_test = features[FIT_TOTAL:]
    guarded_train, guarded_validation, guarded_test, feature_guard = (
        primary_readout.train_only_feature_guard(
            raw_train, raw_validation, raw_test
        )
    )
    stm = primary_readout.select_and_refit_stm(
        guarded_train,
        targets["stm"][:TRAIN],
        guarded_validation,
        targets["stm"][TRAIN:FIT_TOTAL],
        guarded_test,
        targets["stm"][FIT_TOTAL:],
    )
    narma_target = targets["narma10"]
    narma_train = narma_target[:TRAIN]
    narma_validation = narma_target[TRAIN:FIT_TOTAL]
    narma_test = narma_target[FIT_TOTAL:]
    train_valid = np.all(np.isfinite(narma_train), axis=1)
    validation_valid = np.all(np.isfinite(narma_validation), axis=1)
    test_valid = np.all(np.isfinite(narma_test), axis=1)
    narma = primary_readout.select_and_refit(
        guarded_train[train_valid],
        narma_train[train_valid],
        guarded_validation[validation_valid],
        narma_validation[validation_valid],
        guarded_test[test_valid],
        narma_test[test_valid],
        metric="nmse",
    )
    return {"stm": stm, "narma10": narma}, feature_guard


def run_job(job: Job, protocol: dict) -> dict:
    started = time.perf_counter()
    if job not in all_jobs(protocol):
        raise ValueError(f"job is outside the frozen protocol: {job}")
    couplings, coupling_meta = revision.scaled_couplings(
        job.seed, job.n_qubits, COUPLING_SCHEME
    )
    inputs = _input_sequence(job.seed)
    post_wash = inputs[WASH:]
    targets = {
        "stm": np.column_stack(
            [
                tasks.delayed_target(post_wash, delay)
                for delay in STM_DELAYS
            ]
        ),
        "narma10": tasks.narma_target(
            post_wash,
            order=NARMA_ORDER,
            input_scale=NARMA_INPUT_SCALE,
        )[:, None],
    }
    reservoir, jumps, target_strength, actual_strength = _build_reservoir(
        job, couplings
    )
    observables = readout.pauli_observables(job.n_qubits, max_weight=2)
    features = reservoir.run(inputs, observables, washout=WASH)
    expected_shape = (
        FIT_TOTAL + TEST,
        expected_observable_count(job.n_qubits),
    )
    if features.shape != expected_shape:
        raise RuntimeError(
            f"unexpected feature shape for {job}: "
            f"{features.shape} != {expected_shape}"
        )
    primary_scores = {
        "stm": _primary_fixed_score(
            features, targets["stm"], metric="capacity"
        ),
        "narma10": _primary_fixed_score(
            features, targets["narma10"], metric="nmse"
        ),
    }
    validation_scores, feature_guard = _validation_control(features, targets)
    for task_name in ("stm", "narma10"):
        required = (
            primary_scores[task_name]["test"],
            validation_scores[task_name]["selected_test"],
            validation_scores[task_name]["fixed_test"],
        )
        if not all(math.isfinite(float(value)) for value in required):
            raise RuntimeError(f"non-finite {task_name} score for {job}")
    budget_error = (
        None
        if actual_strength is None
        else abs(actual_strength - target_strength) / target_strength
    )
    task_results = {
        task_name: {
            "primary_fixed": primary_scores[task_name],
            "validation_control": validation_scores[task_name],
        }
        for task_name in ("stm", "narma10")
    }
    row = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol[
            "source_environment_sha256"
        ],
        **asdict(job),
        "method_label": METHOD_LABELS[job.method],
        "h": H,
        "dt": DT,
        "wash": WASH,
        "fit": FIT_TOTAL,
        "test": TEST,
        "backend": (
            "exact_reset_unitary"
            if job.method == "FN"
            else "exact_sparse_expm_multiply"
        ),
        "n_observables": len(observables),
        "n_features_including_bias": len(observables) + 1,
        "full_input_sha256": array_sha256(inputs),
        "post_wash_input_sha256": array_sha256(post_wash),
        "target_sha256": {
            name: array_sha256(value) for name, value in targets.items()
        },
        **coupling_meta,
        "coupling_scheme": COUPLING_SCHEME,
        "jump_family_sha256": (
            None
            if jumps is None
            else primary_readout.jump_family_sha256(jumps)
        ),
        "target_jump_strength": target_strength,
        "actual_jump_strength": actual_strength,
        "relative_jump_budget_error": budget_error,
        "feature_guard": feature_guard,
        "task_results": task_results,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return seal_checkpoint(row)


def _validate_checkpoint(
    path: Path, job: Job, protocol: dict
) -> dict | None:
    if not path.exists():
        return None
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"corrupt checkpoint: {path}") from exc
    stored_digest = row.get("checkpoint_payload_sha256")
    if (
        not isinstance(stored_digest, str)
        or stored_digest != checkpoint_payload_sha256(row)
    ):
        raise CheckpointError(
            f"checkpoint payload digest mismatch: {path}"
        )
    expected = {
        "status": "complete",
        "protocol_sha256": protocol_sha256(protocol),
        "source_environment_sha256": protocol[
            "source_environment_sha256"
        ],
        "n_qubits": job.n_qubits,
        "method": job.method,
        "seed": job.seed,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if row.get(field) != value
    ]
    if mismatches:
        raise CheckpointError(
            f"stale or mismatched checkpoint {path}: {mismatches}"
        )
    for task_name in ("stm", "narma10"):
        result = row.get("task_results", {}).get(task_name, {})
        primary_result = result.get("primary_fixed", {})
        control_result = result.get("validation_control", {})
        values = (
            primary_result.get("test"),
            control_result.get("selected_test"),
            control_result.get("fixed_test"),
        )
        if any(value is None or not math.isfinite(float(value)) for value in values):
            raise CheckpointError(
                f"invalid {task_name} scores in checkpoint: {path}"
            )
        expected_metric = "capacity" if task_name == "stm" else "nmse"
        expected_targets = len(STM_DELAYS) if task_name == "stm" else 1
        if (
            primary_result.get("metric") != expected_metric
            or primary_result.get("ridge") != FIXED_RIDGE
            or primary_result.get("test_rows") != TEST
        ):
            raise CheckpointError(
                f"invalid {task_name} primary readout schema: {path}"
            )
        if (
            control_result.get("metric") != expected_metric
            or control_result.get("selected_ridge") not in RIDGES
            or control_result.get("fixed_ridge") != FIXED_RIDGE
        ):
            raise CheckpointError(
                f"invalid {task_name} validation readout schema: {path}"
            )
        vector_fields = (
            (primary_result, "test_by_target"),
            (control_result, "selected_test_by_target"),
            (control_result, "fixed_test_by_target"),
        )
        for container, field in vector_fields:
            vector = container.get(field)
            if (
                not isinstance(vector, list)
                or len(vector) != expected_targets
                or not all(math.isfinite(float(value)) for value in vector)
            ):
                raise CheckpointError(
                    f"invalid {task_name} {field}: {path}"
                )
        scalar_vector_fields = (
            (primary_result, "test", "test_by_target"),
            (
                control_result,
                "selected_test",
                "selected_test_by_target",
            ),
            (control_result, "fixed_test", "fixed_test_by_target"),
        )
        for container, scalar_field, vector_field in scalar_vector_fields:
            vector = container[vector_field]
            aggregate = (
                float(sum(vector))
                if task_name == "stm"
                else float(np.mean(vector))
            )
            if not np.isclose(
                float(container[scalar_field]),
                aggregate,
                rtol=1e-12,
                atol=1e-12,
            ):
                raise CheckpointError(
                    f"inconsistent {task_name.upper()} "
                    f"{scalar_field}: {path}"
                )
        expected_fit_rows = (
            [FIT_TOTAL - delay for delay in STM_DELAYS]
            if task_name == "stm"
            else [FIT_TOTAL - NARMA_ORDER]
        )
        if (
            primary_result.get("effective_fit_rows_by_target")
            != expected_fit_rows
        ):
            raise CheckpointError(
                f"invalid {task_name} effective fit rows: {path}"
            )
        if task_name == "stm":
            if control_result.get("effective_train_rows_by_target") != [
                TRAIN - delay for delay in STM_DELAYS
            ] or control_result.get("effective_refit_rows_by_target") != [
                FIT_TOTAL - delay for delay in STM_DELAYS
            ]:
                raise CheckpointError(
                    f"invalid STM validation row counts: {path}"
                )
    if row.get("n_observables") != expected_observable_count(job.n_qubits):
        raise CheckpointError(f"observable-count mismatch in checkpoint: {path}")
    if (
        row.get("n_features_including_bias")
        != expected_observable_count(job.n_qubits) + 1
    ):
        raise CheckpointError(f"feature-count mismatch in checkpoint: {path}")
    expected_scalars = {
        "h": H,
        "dt": DT,
        "wash": WASH,
        "fit": FIT_TOTAL,
        "test": TEST,
    }
    if any(row.get(field) != value for field, value in expected_scalars.items()):
        raise CheckpointError(f"task-configuration mismatch: {path}")
    runtime = row.get("runtime_seconds")
    if runtime is None or not math.isfinite(float(runtime)) or float(runtime) < 0:
        raise CheckpointError(f"invalid runtime in checkpoint: {path}")
    expected_backend = (
        "exact_reset_unitary"
        if job.method == "FN"
        else "exact_sparse_expm_multiply"
    )
    if row.get("backend") != expected_backend:
        raise CheckpointError(f"backend mismatch in checkpoint: {path}")
    expected_inputs = _input_sequence(job.seed)
    if row.get("full_input_sha256") != array_sha256(expected_inputs):
        raise CheckpointError(f"input hash mismatch in checkpoint: {path}")
    post_wash = expected_inputs[WASH:]
    if row.get("post_wash_input_sha256") != array_sha256(post_wash):
        raise CheckpointError(
            f"post-wash input hash mismatch in checkpoint: {path}"
        )
    expected_targets = {
        "stm": np.column_stack(
            [
                tasks.delayed_target(post_wash, delay)
                for delay in STM_DELAYS
            ]
        ),
        "narma10": tasks.narma_target(
            post_wash,
            order=NARMA_ORDER,
            input_scale=NARMA_INPUT_SCALE,
        )[:, None],
    }
    expected_target_hashes = {
        name: array_sha256(value) for name, value in expected_targets.items()
    }
    if row.get("target_sha256") != expected_target_hashes:
        raise CheckpointError(f"target hash mismatch in checkpoint: {path}")
    couplings, coupling_meta = revision.scaled_couplings(
        job.seed, job.n_qubits, COUPLING_SCHEME
    )
    for field in (
        "base_coupling_sha256",
        "scaled_coupling_sha256",
        "coupling_multiplier",
    ):
        actual = row.get(field)
        expected_value = coupling_meta[field]
        if isinstance(expected_value, float):
            matches = actual is not None and np.isclose(
                float(actual), expected_value, rtol=0.0, atol=1e-15
            )
        else:
            matches = actual == expected_value
        if not matches:
            raise CheckpointError(
                f"{field} mismatch in checkpoint: {path}"
            )
    try:
        expected_jumps, target_strength, expected_actual_strength = (
            _jump_family_for_job(job, couplings)
        )
    except (RuntimeError, ValueError) as exc:
        raise CheckpointError(
            f"could not regenerate jump family for checkpoint: {path}"
        ) from exc
    if not np.isclose(
        float(row.get("target_jump_strength", math.nan)),
        target_strength,
        rtol=0.0,
        atol=1e-12,
    ):
        raise CheckpointError(
            f"target jump strength mismatch in checkpoint: {path}"
        )
    guard = row.get("feature_guard", {})
    if (
        guard.get("fit_on") != "training rows only"
        or guard.get("threshold") != FEATURE_STD_TOL
        or guard.get("retained_nonbias_features", -1)
        + guard.get("dropped_nonbias_features", -1)
        != expected_observable_count(job.n_qubits)
    ):
        raise CheckpointError(f"feature-guard mismatch in checkpoint: {path}")
    if job.method in DISSIPATIVE_METHODS:
        error = row.get("relative_jump_budget_error")
        if error is None or float(error) > 1e-10:
            raise CheckpointError(f"jump-budget mismatch in checkpoint: {path}")
        if not np.isclose(
            float(row.get("actual_jump_strength", math.nan)),
            float(expected_actual_strength),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise CheckpointError(
                f"actual jump strength mismatch in checkpoint: {path}"
            )
        jump_hash = row.get("jump_family_sha256")
        expected_jump_hash = primary_readout.jump_family_sha256(
            expected_jumps
        )
        if jump_hash != expected_jump_hash:
            raise CheckpointError(
                f"jump-family hash mismatch in checkpoint: {path}"
            )
    elif (
        row.get("actual_jump_strength") is not None
        or row.get("relative_jump_budget_error") is not None
        or row.get("jump_family_sha256") is not None
    ):
        raise CheckpointError(f"FN checkpoint unexpectedly has jumps: {path}")
    return row


def _record_failure(
    outdir: Path, job: Job, protocol: dict, exc: BaseException
) -> dict:
    payload = {
        "status": "error",
        "protocol_sha256": protocol_sha256(protocol),
        **asdict(job),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "recorded_unix_time": time.time(),
    }
    atomic_write_json(failure_path(outdir, job), payload)
    return payload


def _run_and_write(job: Job, protocol: dict, outdir_text: str) -> dict:
    outdir = Path(outdir_text)
    path = job_path(outdir, job)
    existing = _validate_checkpoint(path, job, protocol)
    if existing is not None:
        return {
            "status": "skip",
            **asdict(job),
            "runtime_seconds": existing["runtime_seconds"],
        }
    try:
        row = run_job(job, protocol)
        atomic_write_json(path, row)
        return {
            "status": "done",
            **asdict(job),
            "runtime_seconds": row["runtime_seconds"],
        }
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001
        failure = _record_failure(outdir, job, protocol, exc)
        return {
            "status": "error",
            **asdict(job),
            "runtime_seconds": 0.0,
            "error": f"{failure['error_type']}: {failure['error']}",
        }


def status_payload(outdir: Path, protocol: dict) -> dict:
    complete = []
    pending = []
    runtime = 0.0
    by_n = {}
    for n_qubits in protocol["n_values"]:
        by_n[str(n_qubits)] = {"complete": 0, "expected": 0}
    for job in all_jobs(protocol):
        by_n[str(job.n_qubits)]["expected"] += 1
        row = _validate_checkpoint(job_path(outdir, job), job, protocol)
        if row is None:
            pending.append(job)
        else:
            complete.append(job)
            runtime += float(row["runtime_seconds"])
            by_n[str(job.n_qubits)]["complete"] += 1
    payload = {
        "protocol_sha256": protocol_sha256(protocol),
        "expected": len(complete) + len(pending),
        "complete": len(complete),
        "pending": len(pending),
        "recorded_job_hours": runtime / 3600.0,
        "by_n": by_n,
        "complete_fraction": len(complete) / (len(complete) + len(pending)),
    }
    atomic_write_json(outdir / "status.json", payload)
    return payload


def _progress(result: dict, completed: int, total: int) -> str:
    runtime = float(result.get("runtime_seconds", 0.0))
    error = "" if "error" not in result else f" ! {result['error']}"
    return (
        f"[{completed}/{total}] {result['status']:5s} "
        f"N={result['n_qubits']} {result['method']} s={result['seed']} "
        f"{runtime:.1f}s{error}"
    )


def run_jobs(
    outdir: Path,
    protocol: dict,
    jobs: Sequence[Job],
    workers: int,
) -> dict:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be in [1,8]")
    write_or_validate_protocol(outdir, protocol)
    with exclusive_run_lock(outdir, protocol):
        return _run_jobs_locked(outdir, protocol, jobs, workers)


def _run_jobs_locked(
    outdir: Path,
    protocol: dict,
    jobs: Sequence[Job],
    workers: int,
) -> dict:
    pending = [
        job
        for job in jobs
        if _validate_checkpoint(job_path(outdir, job), job, protocol) is None
    ]
    print(
        f"{len(jobs) - len(pending)}/{len(jobs)} selected checkpoints complete; "
        f"{len(pending)} pending; workers={workers}; "
        f"protocol={protocol_sha256(protocol)[:12]}",
        flush=True,
    )
    completed = len(jobs) - len(pending)
    errors = []
    if workers == 1:
        for job in pending:
            result = _run_and_write(job, protocol, str(outdir))
            completed += 1
            print(_progress(result, completed, len(jobs)), flush=True)
            if result["status"] == "error":
                errors.append(result)
    elif pending:
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
                executor.submit(
                    _run_and_write, job, protocol, str(outdir)
                ): job
                for job in pending
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except CheckpointError:
                    for other in futures:
                        other.cancel()
                    raise
                except Exception as exc:  # noqa: BLE001
                    failure = _record_failure(outdir, job, protocol, exc)
                    result = {
                        "status": "error",
                        **asdict(job),
                        "runtime_seconds": 0.0,
                        "error": (
                            f"{failure['error_type']}: {failure['error']}"
                        ),
                    }
                completed += 1
                print(_progress(result, completed, len(jobs)), flush=True)
                if result["status"] == "error":
                    errors.append(result)
    status = status_payload(outdir, protocol)
    if errors:
        print(
            f"{len(errors)} selected jobs failed; completed checkpoints were "
            "preserved and failed jobs will be retried on restart",
            flush=True,
        )
    return {**status, "selected_errors": len(errors)}


def _paired_summary(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("paired summary requires at least two finite values")
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    critical = float(student_t.ppf(0.975, len(array) - 1))
    return {
        "n": len(array),
        "mean": mean,
        "se": se,
        "ci95": [mean - critical * se, mean + critical * se],
        "positive": int(np.sum(array > 0)),
        "negative": int(np.sum(array < 0)),
        "zero": int(np.sum(array == 0)),
    }


def _bootstrap_summary(values: Sequence[float], seed: int) -> dict:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(array), size=(BOOTSTRAP_DRAWS, len(array))
    )
    means = np.mean(array[indices], axis=1)
    return {
        "n": len(array),
        "mean": float(np.mean(array)),
        "ci95_percentile": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def _sign_flip_pvalue(values: Sequence[float], seed: int) -> float:
    array = np.asarray(values, dtype=float)
    n = len(array)
    if n == 0 or np.allclose(array, 0.0):
        return 1.0
    observed_sum = abs(float(np.sum(array)))
    tolerance = 1e-12
    if n <= 32:
        def signed_sums(values: np.ndarray) -> np.ndarray:
            sums = np.array([0.0])
            for value in values:
                sums = np.concatenate((sums + value, sums - value))
            return sums

        split = n // 2
        left = signed_sums(array[:split])
        right = np.sort(signed_sums(array[split:]))
        threshold = observed_sum - tolerance * n
        if threshold <= 0:
            return 1.0
        below = np.searchsorted(
            right, -threshold - left, side="right"
        )
        above = len(right) - np.searchsorted(
            right, threshold - left, side="left"
        )
        exceed = int(
            np.sum(below, dtype=np.int64)
            + np.sum(above, dtype=np.int64)
        )
        return float(exceed / (1 << n))
    observed = observed_sum / n
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = SIGN_FLIP_DRAWS
    batch = 10_000
    while remaining:
        size = min(batch, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        permuted = np.abs(np.mean(signs * array[None, :], axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
        remaining -= size
    return float((extreme + 1) / (SIGN_FLIP_DRAWS + 1))


def _holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (m - rank) * float(pvalues[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _score(row: dict, task_name: str, readout_name: str) -> float:
    result = row["task_results"][task_name]
    if readout_name == "primary_fixed":
        return float(result["primary_fixed"]["test"])
    if readout_name == "validation_selected":
        return float(result["validation_control"]["selected_test"])
    raise ValueError(f"unknown readout result {readout_name!r}")


def invariant_audit(rows: Sequence[dict], protocol: dict) -> dict:
    errors = []
    expected = {job.key for job in all_jobs(protocol)}
    observed = [
        (row.get("n_qubits"), row.get("method"), row.get("seed"))
        for row in rows
    ]
    if len(observed) != len(set(observed)):
        errors.append("duplicate checkpoint identities")
    if set(observed) != expected:
        errors.append("checkpoint identity set does not match the manifest")
    indexed = {
        (row["n_qubits"], row["method"], row["seed"]): row for row in rows
    }
    for seed in protocol["seeds"]:
        expected_input_hash = None
        full_base = revision.nested_base_couplings(seed, 8)
        for n_qubits in protocol["n_values"]:
            paired = [
                indexed[(n_qubits, method, seed)]
                for method in protocol["methods"]
            ]
            input_hashes = {row["full_input_sha256"] for row in paired}
            coupling_hashes = {
                row["scaled_coupling_sha256"] for row in paired
            }
            target_hashes = {
                canonical_json(row["target_sha256"]) for row in paired
            }
            if len(input_hashes) != 1:
                errors.append(f"input pairing failed for N={n_qubits}, seed={seed}")
            if len(coupling_hashes) != 1:
                errors.append(
                    f"coupling pairing failed for N={n_qubits}, seed={seed}"
                )
            if len(target_hashes) != 1:
                errors.append(
                    f"target pairing failed for N={n_qubits}, seed={seed}"
                )
            current_input_hash = next(iter(input_hashes))
            if expected_input_hash is None:
                expected_input_hash = current_input_hash
            elif current_input_hash != expected_input_hash:
                errors.append(f"cross-N input pairing failed for seed={seed}")
            base = revision.nested_base_couplings(seed, n_qubits)
            if not np.array_equal(base, full_base[:n_qubits, :n_qubits]):
                errors.append(f"nested coupling invariant failed for seed={seed}")
            multiplier = revision.coupling_multiplier(
                n_qubits, COUPLING_SCHEME
            )
            for row in paired:
                if not np.isclose(
                    row["coupling_multiplier"],
                    multiplier,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    errors.append(
                        f"coupling multiplier mismatch for {row['method']}, "
                        f"N={n_qubits}, seed={seed}"
                    )
                if (
                    row["n_observables"]
                    != expected_observable_count(n_qubits)
                ):
                    errors.append(
                        f"readout feature mismatch for {row['method']}, "
                        f"N={n_qubits}, seed={seed}"
                    )
                if row["method"] in DISSIPATIVE_METHODS and float(
                    row["relative_jump_budget_error"]
                ) > 1e-10:
                    errors.append(
                        f"jump budget mismatch for {row['method']}, "
                        f"N={n_qubits}, seed={seed}"
                    )
    return {"passed": not errors, "errors": errors}


def aggregate(outdir: Path, protocol: dict) -> dict:
    write_or_validate_protocol(outdir, protocol)
    rows = []
    missing = []
    for job in all_jobs(protocol):
        row = _validate_checkpoint(job_path(outdir, job), job, protocol)
        if row is None:
            missing.append(asdict(job))
        else:
            rows.append(row)
    if missing:
        raise RuntimeError(
            f"cannot aggregate: {len(missing)} of {protocol['n_jobs']} "
            "checkpoints are still missing"
        )
    audit = invariant_audit(rows, protocol)
    if not audit["passed"]:
        raise RuntimeError(
            f"complete checkpoint set failed invariants: {audit['errors'][:5]}"
        )
    indexed = {
        (row["n_qubits"], row["method"], row["seed"]): row for row in rows
    }
    summaries = {}
    secondary_pvalues = {}
    secondary_effects = {}
    readout_names = ("primary_fixed", "validation_selected")
    for readout_name in readout_names:
        readout_summary = {}
        for n_qubits in protocol["n_values"]:
            n_summary = {}
            for task_name in ("stm", "narma10"):
                task_summary = {}
                local = np.asarray(
                    [
                        _score(
                            indexed[
                                (n_qubits, REFERENCE_METHOD, int(seed))
                            ],
                            task_name,
                            readout_name,
                        )
                        for seed in protocol["seeds"]
                    ]
                )
                for method in protocol["methods"]:
                    values = np.asarray(
                        [
                            _score(
                                indexed[(n_qubits, method, int(seed))],
                                task_name,
                                readout_name,
                            )
                            for seed in protocol["seeds"]
                        ]
                    )
                    oriented = (
                        values - local
                        if task_name == "stm"
                        else local - values
                    )
                    method_summary = {
                        "n": len(values),
                        "mean": float(np.mean(values)),
                        "se": float(
                            np.std(values, ddof=1) / np.sqrt(len(values))
                        ),
                    }
                    if method != REFERENCE_METHOD:
                        effect = _paired_summary(oriented)
                        relative = oriented / local
                        effect["relative"] = _bootstrap_summary(
                            relative,
                            seed=80_000
                            + 1000 * n_qubits
                            + 100 * list(protocol["methods"]).index(method)
                            + (0 if task_name == "stm" else 1),
                        )
                        key = (
                            f"{readout_name}/N{n_qubits}/"
                            f"{task_name}/{method}"
                        )
                        pvalue = _sign_flip_pvalue(
                            oriented,
                            seed=70_000
                            + 1000 * n_qubits
                            + 100 * list(protocol["methods"]).index(method)
                            + (0 if task_name == "stm" else 1),
                        )
                        effect["sign_flip_pvalue"] = pvalue
                        method_summary["versus_local"] = effect
                        if (
                            readout_name == "primary_fixed"
                            and method in DISSIPATIVE_METHODS
                        ):
                            secondary_pvalues[key] = pvalue
                            secondary_effects[key] = effect
                    task_summary[method] = method_summary
                n_summary[task_name] = task_summary
            readout_summary[str(n_qubits)] = n_summary
        summaries[readout_name] = readout_summary

    endpoint_relative_effects = []
    finite_range_slopes = []
    for seed in protocol["seeds"]:
        relative_by_n = []
        for n_qubits in protocol["n_values"]:
            local = _score(
                indexed[
                    (n_qubits, REFERENCE_METHOD, int(seed))
                ],
                "stm",
                "primary_fixed",
            )
            collective = _score(
                indexed[
                    (n_qubits, COLLECTIVE_METHOD, int(seed))
                ],
                "stm",
                "primary_fixed",
            )
            relative_by_n.append((collective - local) / local)
        endpoint_relative_effects.append(
            float(relative_by_n[-1] - relative_by_n[0])
        )
        finite_range_slopes.append(
            float(
                np.polyfit(
                    np.asarray(protocol["n_values"], dtype=float),
                    np.asarray(relative_by_n, dtype=float),
                    deg=1,
                )[0]
            )
        )
    endpoint_key = "collective_relative_stm_N8_minus_N4"
    endpoint_effect = _paired_summary(endpoint_relative_effects)
    endpoint_effect["sign_flip_pvalue"] = _sign_flip_pvalue(
        endpoint_relative_effects, seed=91_001
    )
    endpoint_effect["bootstrap"] = _bootstrap_summary(
        endpoint_relative_effects, seed=91_002
    )
    slope_key = "collective_relative_stm_slope_over_N4_to_N8"
    slope_effect = _paired_summary(finite_range_slopes)
    slope_effect["sign_flip_pvalue"] = _sign_flip_pvalue(
        finite_range_slopes, seed=91_003
    )
    slope_effect["bootstrap"] = _bootstrap_summary(
        finite_range_slopes, seed=91_004
    )
    confirmatory_pvalues = {
        **secondary_pvalues,
        endpoint_key: endpoint_effect["sign_flip_pvalue"],
        slope_key: slope_effect["sign_flip_pvalue"],
    }
    expected_cell_keys = {
        f"primary_fixed/N{n_qubits}/{task_name}/{method}"
        for n_qubits in protocol["n_values"]
        for task_name in ("stm", "narma10")
        for method in DISSIPATIVE_METHODS
        if method != REFERENCE_METHOD
    }
    expected_confirmatory_keys = expected_cell_keys | {
        endpoint_key,
        slope_key,
    }
    if set(confirmatory_pvalues) != expected_confirmatory_keys:
        missing = sorted(expected_confirmatory_keys - set(confirmatory_pvalues))
        extra = sorted(set(confirmatory_pvalues) - expected_confirmatory_keys)
        raise RuntimeError(
            "confirmatory family does not match the frozen 62-test contract: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    confirmatory_adjusted = _holm_adjust(confirmatory_pvalues)
    for key, adjusted in confirmatory_adjusted.items():
        if key == endpoint_key:
            endpoint_effect["holm_pvalue"] = adjusted
            continue
        if key == slope_key:
            slope_effect["holm_pvalue"] = adjusted
            continue
        secondary_effects[key]["holm_pvalue"] = adjusted

    payload = {
        "status": "complete",
        "protocol_sha256": protocol_sha256(protocol),
        "expected_checkpoints": protocol["n_jobs"],
        "complete_checkpoints": len(rows),
        "invariant_audit": audit,
        "confirmatory_family_size": len(confirmatory_pvalues),
        "confirmatory_dissipative_cells": secondary_effects,
        "finite_range_endpoint_contrast": endpoint_effect,
        "finite_range_slope_contrast": slope_effect,
        "fn_is_descriptive": True,
        "summary": summaries,
        "recorded_job_hours": sum(
            float(row["runtime_seconds"]) for row in rows
        )
        / 3600.0,
    }
    atomic_write_json(outdir / "aggregate.json", payload)
    return payload


def _parse_csv(text: str | None, cast=str) -> list | None:
    if text is None:
        return None
    values = [part.strip() for part in text.split(",") if part.strip()]
    return [cast(value) for value in values]


def _print_status(payload: dict) -> None:
    print(
        f"{payload['complete']}/{payload['expected']} complete "
        f"({100 * payload['complete_fraction']:.1f}%); "
        f"{payload['pending']} pending; "
        f"{payload['recorded_job_hours']:.2f} recorded job-hours"
    )
    for n_qubits, values in payload["by_n"].items():
        print(
            f"N={n_qubits}: {values['complete']}/{values['expected']}",
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("status")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=4)
    run_parser.add_argument("--n-values")
    run_parser.add_argument("--methods")
    run_parser.add_argument(
        "--seed-indices",
        help="zero-based indices into the frozen 24-lineage list",
    )
    run_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser("aggregate")
    subparsers.add_parser("validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = build_protocol()
    outdir = args.outdir.resolve()
    if args.command == "freeze":
        write_or_validate_protocol(outdir, protocol, create=True)
        print(
            f"frozen {protocol['n_jobs']} jobs at {outdir}; "
            f"protocol={protocol_sha256(protocol)}"
        )
        return 0
    if args.command == "status":
        write_or_validate_protocol(outdir, protocol)
        _print_status(status_payload(outdir, protocol))
        return 0
    if args.command == "run":
        jobs = selected_jobs(
            protocol,
            n_values=_parse_csv(args.n_values, int),
            methods=_parse_csv(args.methods),
            seed_indices=_parse_csv(args.seed_indices, int),
        )
        if args.dry_run:
            print(
                f"selected {len(jobs)}/{protocol['n_jobs']} jobs; "
                f"protocol={protocol_sha256(protocol)}"
            )
            return 0
        result = run_jobs(outdir, protocol, jobs, args.workers)
        _print_status(result)
        return 1 if result["selected_errors"] else 0
    if args.command == "aggregate":
        payload = aggregate(outdir, protocol)
        print(
            f"aggregate complete: {payload['complete_checkpoints']} checkpoints, "
            f"{payload['recorded_job_hours']:.2f} job-hours"
        )
        return 0
    if args.command == "validate":
        write_or_validate_protocol(outdir, protocol)
        status = status_payload(outdir, protocol)
        if status["complete"] != protocol["n_jobs"]:
            _print_status(status)
            return 1
        payload = aggregate(outdir, protocol)
        print(
            f"all invariants pass; aggregate status={payload['status']}"
        )
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Confirmatory rank-one phase-direction intervention.

This experiment changes only the coefficient direction of one collective
lowering jump while holding its Kossakowski rank, spectrum, diagonal, trace,
coefficient magnitudes, and assigned jump-strength budget fixed.  The task
protocol matches the manuscript's principal continuously driven STM
comparison, except that the strict 800-input washout is used throughout.

The protocol is fail-closed and is frozen before any cross-condition task
contrast is opened:

1. ``freeze`` writes the exact seed, direction, source, and inference ledger;
2. ``convergence`` audits four initial states for every direction on the first
   eight fresh reservoirs and must pass before task scoring is allowed;
3. ``run`` evaluates all 32 paired reservoirs with atomic checkpoints;
4. ``aggregate`` and ``validate`` reconstruct every reported result.

Examples
--------
PYTHONPATH=src:experiments python experiments/run_phase_direction_confirmatory.py smoke
PYTHONPATH=src:experiments python experiments/run_phase_direction_confirmatory.py freeze
PYTHONPATH=src:experiments python experiments/run_phase_direction_confirmatory.py convergence --workers 4
PYTHONPATH=src:experiments python experiments/run_phase_direction_confirmatory.py run --workers 4
PYTHONPATH=src:experiments python experiments/run_phase_direction_confirmatory.py aggregate
PYTHONPATH=src:experiments python experiments/run_phase_direction_confirmatory.py validate
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

for _variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qrc_phase_direction_mplconfig")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/qrc_phase_direction_cache")

import _paths  # noqa: E402,F401
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

from qrc import dissipators as dsp  # noqa: E402
from qrc import liouvillian as dense_liouvillian  # noqa: E402
from qrc import readout, reservoirs as res, sparse_evolve, tasks  # noqa: E402
from qrc.liouvillian import unvec, vec  # noqa: E402
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive  # noqa: E402
from qrc.sparse_evolve import SparseLindbladReservoir  # noqa: E402
from run_revision_primary_regularization import (  # noqa: E402
    RIDGES,
    select_and_refit_stm,
    train_only_feature_guard,
)


PROTOCOL_VERSION = "phase-direction-confirmatory-v1-2026-08-12"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "paper" / "evidence" / "phase_direction_confirmatory_v1"

N_QUBITS = 5
H = 0.5
DT = 0.5
GAMMA = 1.0
WASHOUT = 800
TRAIN = 450
VALIDATION = 150
FIT_ROWS = TRAIN + VALIDATION
TEST = 400
TOTAL_INPUTS = WASHOUT + FIT_ROWS + TEST
STM_DELAYS = tuple(range(1, 21))
FIXED_RIDGE = 1e-8
FEATURE_STD_TOL = 1e-12
TARGET_BUDGET = float(N_QUBITS * 2 ** (N_QUBITS - 1))

SEED_NAMESPACE = 2026081202
DIRECTION_NAMESPACE = 2026081203
INFERENCE_NAMESPACE = 2026081204
HAAR_NAMESPACE = 2026081205
N_SEEDS = 32
N_AUDIT_SEEDS = 8
SIGNFLIP_DRAWS = 100_000
ORDERED_PATH_DRAWS = 100_000

CONFIRMATORY_SEEDS = (
    653405976, 1031846597, 552412115, 580193326,
    812996642, 96539450, 740577026, 395071372,
    1088019782, 1947384789, 1238786023, 1603309630,
    345831413, 1378099110, 271924869, 246873821,
    272816610, 952697876, 1565274822, 697368240,
    1620152868, 250081386, 1519772029, 1266500966,
    1226991834, 1000639827, 1685804651, 823320203,
    1825046501, 394437236, 1768248831, 181572750,
)
PILOT_SEEDS = (
    956087733, 1375334633, 707736772, 1133846500,
    365211353, 878523603, 457552621, 363662622,
)

PHASE_PATH = {
    "path_f0": 0.0,
    "path_f025": 0.25,
    "path_f05": 0.5,
    "path_f075": 0.75,
    "path_f1": 1.0,
}
# Drawn once from PCG64 namespace DIRECTION_NAMESPACE before task scoring,
# conditional only on c_0=1 and exclusion of affine Fourier permutations.
SCRAMBLED_EXPONENTS = {
    "scrambled_r1": (0, 2, 3, 1, 4),
    "scrambled_r2": (0, 1, 3, 4, 2),
    "scrambled_r3": (0, 2, 4, 3, 1),
    "scrambled_r4": (0, 3, 4, 1, 2),
}
CONDITIONS = tuple(PHASE_PATH) + tuple(SCRAMBLED_EXPONENTS)
EQUAL_PHASE = "path_f0"
ORTHOGONAL_FOURIER = "path_f1"
ZERO_OVERLAP_CONDITIONS = (ORTHOGONAL_FOURIER,) + tuple(SCRAMBLED_EXPONENTS)
AUDIT_SEEDS = CONFIRMATORY_SEEDS[:N_AUDIT_SEEDS]

TRACE_GATE_AT_800 = 1e-8
OBSERVABLE_GATE_AT_800 = 1e-8
SCORE_RANGE_GATE = 1e-4
CHECKPOINT_STEPS = (0, 50, 100, 200, 400, 800, 1100, 1200)
CONVERGENCE_CURVE_STOP = 1200

SOURCE_FILES = (
    "experiments/_paths.py",
    "experiments/run_phase_direction_confirmatory.py",
    "experiments/run_final_scaling.py",
    "experiments/run_revision_primary_regularization.py",
    "experiments/run_revision_controls.py",
    "src/qrc/__init__.py",
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
    condition: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.condition}__s{self.seed}"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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


def atomic_json(path: Path, payload: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def expected_seeds() -> tuple[int, ...]:
    generated = tuple(
        int(value)
        for value in np.random.default_rng(SEED_NAMESPACE).integers(
            0, 2**31 - 1, size=N_SEEDS
        )
    )
    if generated != CONFIRMATORY_SEEDS:
        raise RuntimeError("confirmatory seed ledger drifted from its namespace")
    if len(set(generated)) != N_SEEDS or set(generated).intersection(PILOT_SEEDS):
        raise RuntimeError("confirmatory seeds duplicate or overlap the pilot")
    return generated


def _collect_declared_seeds(value: object, key: str = "") -> set[int]:
    found: set[int] = set()
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            found.update(_collect_declared_seeds(child_value, str(child_key)))
    elif isinstance(value, list):
        if "seed" in key.lower():
            found.update(int(item) for item in value if isinstance(item, int) and not isinstance(item, bool))
        else:
            for item in value:
                found.update(_collect_declared_seeds(item, key))
    elif "seed" in key.lower() and isinstance(value, int) and not isinstance(value, bool):
        found.add(int(value))
    return found


def seed_freshness_audit() -> dict:
    manifest_path = ROOT / "paper" / "data" / "reproducibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = _collect_declared_seeds(manifest)
    confirmation = set(expected_seeds())
    principal = set(
        int(value)
        for value in np.random.default_rng(2024).integers(0, 2**31 - 1, size=64)
    )
    collisions = sorted(confirmation.intersection(declared.union(PILOT_SEEDS)))
    principal_collisions = sorted(confirmation.intersection(principal))
    if collisions or principal_collisions:
        raise RuntimeError(f"confirmatory seeds overlap prior ledgers: {collisions}")
    return {
        "manifest_sha256_at_freeze": file_sha256(manifest_path),
        "declared_seed_count_checked": len(declared),
        "pilot_seed_count_checked": len(PILOT_SEEDS),
        "principal_seed_count_checked": len(principal),
        "collisions": collisions,
        "principal_collisions": principal_collisions,
    }


def source_environment() -> dict:
    files = {}
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source missing: {path}")
        files[relative] = file_sha256(path)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    return {
        "files": files,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "git_head_at_freeze": head,
        "thread_limits": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
    }


def coefficients(condition: str) -> np.ndarray:
    if condition in PHASE_PATH:
        fraction = PHASE_PATH[condition]
        result = np.exp(
            1j * 2.0 * np.pi * fraction * np.arange(N_QUBITS) / N_QUBITS
        )
    elif condition in SCRAMBLED_EXPONENTS:
        exponents = np.asarray(SCRAMBLED_EXPONENTS[condition], dtype=float)
        result = np.exp(1j * 2.0 * np.pi * exponents / N_QUBITS)
    else:
        raise ValueError(f"unknown condition {condition!r}")
    result = np.asarray(result, dtype=complex)
    result *= np.exp(-1j * np.angle(result[0]))
    if not np.allclose(result[0], 1.0, rtol=0.0, atol=1e-14):
        raise RuntimeError("global phase canonicalization failed")
    return result


def expected_scrambled_exponents() -> dict[str, tuple[int, ...]]:
    affine = {
        tuple(int((slope * index) % N_QUBITS) for index in range(N_QUBITS))
        for slope in range(1, N_QUBITS)
    }
    candidates = sorted(
        (0,) + permutation
        for permutation in itertools.permutations(range(1, N_QUBITS))
        if (0,) + permutation not in affine
    )
    rng = np.random.default_rng(DIRECTION_NAMESPACE)
    selected = [candidates[int(index)] for index in rng.choice(len(candidates), 4, replace=False)]
    expected = {
        f"scrambled_r{index + 1}": tuple(value)
        for index, value in enumerate(selected)
    }
    if expected != SCRAMBLED_EXPONENTS:
        raise RuntimeError("scrambled direction ledger drifted from its namespace")
    return expected


def direction_invariants(condition: str) -> dict:
    c = coefficients(condition)
    gamma = np.outer(c, c.conjugate())
    eigenvalues = np.linalg.eigvalsh(gamma)
    uniform = np.ones(N_QUBITS, dtype=complex)
    overlap = abs(np.vdot(uniform, c)) ** 2 / (
        float(np.vdot(uniform, uniform).real) * float(np.vdot(c, c).real)
    )
    return {
        "condition": condition,
        "coefficients_real": [float(value.real) for value in c],
        "coefficients_imag": [float(value.imag) for value in c],
        "coefficient_sha256": array_sha256(c),
        "uniform_direction_overlap": float(overlap),
        "magnitude_ipr": float(np.sum(np.abs(c) ** 4) / np.sum(np.abs(c) ** 2) ** 2),
        "kossakowski_trace": float(np.trace(gamma).real),
        "kossakowski_diagonal": [float(value.real) for value in np.diag(gamma)],
        "kossakowski_rank": int(np.linalg.matrix_rank(gamma, tol=1e-10)),
        "kossakowski_eigenvalues": [float(value) for value in eigenvalues],
        "coefficient_magnitudes": [float(value) for value in np.abs(c)],
    }


def direction_ledger() -> dict:
    expected_scrambled_exponents()
    payload = {
        "direction_namespace": DIRECTION_NAMESPACE,
        "generation_rule": (
            "five fixed phase-gradient points plus four frozen non-affine "
            "permutations of the fifth roots of unity; c0 is gauge-fixed to one"
        ),
        "conditions": [direction_invariants(condition) for condition in CONDITIONS],
    }
    payload["direction_ledger_sha256"] = sha256_json(payload)
    return payload


def jumps_for_condition(condition: str) -> list[tuple[np.ndarray, float]]:
    jumps = dsp.collective_loss(N_QUBITS, GAMMA, c=coefficients(condition))
    budget = dsp.jump_strength(jumps)
    if not np.isclose(budget, TARGET_BUDGET, rtol=1e-12, atol=1e-12):
        raise RuntimeError(f"{condition} budget {budget} != {TARGET_BUDGET}")
    return jumps


def protocol_payload() -> tuple[dict, dict]:
    seeds = expected_seeds()
    ledger = direction_ledger()
    environment = source_environment()
    smoke_result = smoke()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "confirmatory_frozen_before_scoring",
        "frozen_at_utc": utc_now(),
        "question": (
            "Does rotating an otherwise matched rank-one collective lowering "
            "channel away from the equal-phase direction reduce held-out STM?"
        ),
        "claim_boundary": (
            "The direction of this matched rank-one channel affects memory in "
            "the tested driven-spin reservoirs; no unique or universal mechanism "
            "and no globally optimal direction are claimed."
        ),
        "pilot_use": "pilot seeds and pilot scores are excluded and never pooled",
        "seed_freshness_audit": seed_freshness_audit(),
        "seeds": list(seeds),
        "seed_namespace": SEED_NAMESPACE,
        "n_seeds": N_SEEDS,
        "audit_seeds": list(AUDIT_SEEDS),
        "directions_sha256": ledger["direction_ledger_sha256"],
        "conditions": list(CONDITIONS),
        "phase_path": PHASE_PATH,
        "scrambled_exponents": {
            key: list(value) for key, value in SCRAMBLED_EXPONENTS.items()
        },
        "primary_condition": EQUAL_PHASE,
        "primary_reference": ORTHOGONAL_FOURIER,
        "zero_overlap_conditions": list(ZERO_OVERLAP_CONDITIONS),
        "reservoir": {
            "n_qubits": N_QUBITS,
            "h": H,
            "dt": DT,
            "couplings": "J_ij iid Uniform[-1,1] for i<j",
            "input": "iid Uniform[0,1] from the paired problem RNG",
            "initial_state_for_task": "|0...0>",
            "features": "45 weight-one/two same-axis Pauli expectations plus bias",
            "backend": "SparseLindbladReservoir exact expm_multiply",
        },
        "task": {
            "name": "STM",
            "delays": list(STM_DELAYS),
            "washout": WASHOUT,
            "fit_rows": FIT_ROWS,
            "test_rows": TEST,
            "primary_readout": {
                "ridge": FIXED_RIDGE,
                "fit": "all 600 fitting rows, per-delay finite-row rule",
            },
            "sensitivity_readout": {
                "train": TRAIN,
                "validation": VALIDATION,
                "ridge_grid": list(RIDGES),
                "selection": "maximize summed validation capacity; stronger ridge wins ties",
                "refit": "per-delay train plus all validation rows",
            },
        },
        "held_fixed": [
            "Hamiltonian", "input stream", "target", "split", "readout observables",
            "Kossakowski rank one", "Kossakowski diagonal one at every site",
            "Kossakowski spectrum (5,0,...,0)", "Kossakowski trace five",
            "coefficient magnitudes one", "magnitude IPR one fifth",
            "assigned jump-strength budget B=80",
        ],
        "convergence_gate": {
            "states": ["ground", "excited", "maximally_mixed", "haar_pure"],
            "seeds": list(AUDIT_SEEDS),
            "conditions": list(CONDITIONS),
            "trace_distance_at_800_max": TRACE_GATE_AT_800,
            "pauli_max_abs_at_800_max": OBSERVABLE_GATE_AT_800,
            "fixed_ridge_score_range_max": SCORE_RANGE_GATE,
            "gate_must_pass_before_task_scoring": True,
        },
        "inference": {
            "single_primary_estimand": "STM(path_f0)-STM(path_f1)",
            "primary_interval": "paired two-sided 95% Student-t interval",
            "primary_test": f"{SIGNFLIP_DRAWS} fixed-seed Monte Carlo paired sign flips",
            "gated_key_secondary": (
                "STM(path_f0)-within-seed mean(path_f1,scrambled_r1,...,scrambled_r4); "
                "tested at alpha=.05 only if the primary passes"
            ),
            "adjusted_secondary_contrasts": [
                "path_f0-path_f025", "path_f0-path_f05", "path_f0-path_f075",
                "path_f0-scrambled_r1", "path_f0-scrambled_r2",
                "path_f0-scrambled_r3", "path_f0-scrambled_r4",
            ],
            "adjusted_secondary_multiplicity": (
                "Holm sign-flip tests and Bonferroni-simultaneous 95% t intervals across seven"
            ),
            "ordered_path_diagnostic": (
                f"mean within-seed Spearman rho with {ORDERED_PATH_DRAWS} "
                "repeated-measures label permutations"
            ),
            "inference_namespace": INFERENCE_NAMESPACE,
            "no_optional_stopping_or_seed_replacement": True,
        },
        "source_environment": environment,
        "source_environment_sha256": sha256_json(environment),
        "pre_score_smoke": smoke_result,
        "pre_score_smoke_sha256": sha256_json(smoke_result),
    }
    protocol["protocol_sha256"] = sha256_json(protocol)
    return protocol, ledger


def protocol_path(outdir: Path) -> Path:
    return outdir / "protocol.json"


def load_protocol(outdir: Path, *, verify_sources: bool = True) -> dict:
    path = protocol_path(outdir)
    if not path.is_file():
        raise FileNotFoundError(f"frozen protocol missing: {path}")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = protocol.get("protocol_sha256")
    unhashed = dict(protocol)
    unhashed.pop("protocol_sha256", None)
    if stored_hash != sha256_json(unhashed):
        raise RuntimeError("protocol self-hash failed")
    smoke_path = outdir / "smoke.json"
    if not smoke_path.is_file():
        raise FileNotFoundError(f"archived pre-score smoke gate missing: {smoke_path}")
    stored_smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        stored_smoke != protocol.get("pre_score_smoke")
        or sha256_json(stored_smoke) != protocol.get("pre_score_smoke_sha256")
        or stored_smoke.get("status") != "passed"
    ):
        raise RuntimeError("archived pre-score smoke gate failed")
    if verify_sources:
        for relative, expected in protocol["source_environment"]["files"].items():
            if file_sha256(ROOT / relative) != expected:
                raise RuntimeError(f"frozen source drift: {relative}")
    return protocol


def freeze(outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    path = protocol_path(outdir)
    if path.exists():
        protocol = load_protocol(outdir)
        ledger = json.loads((outdir / "directions.json").read_text(encoding="utf-8"))
        if ledger.get("direction_ledger_sha256") != protocol["directions_sha256"]:
            raise RuntimeError("stored direction ledger does not match protocol")
        return protocol
    protocol, ledger = protocol_payload()
    atomic_json(outdir / "directions.json", ledger)
    atomic_json(outdir / "smoke.json", protocol["pre_score_smoke"])
    atomic_json(path, protocol)
    return protocol


def build_problem(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    couplings = res.random_couplings(N_QUBITS, 1.0, rng)
    inputs = tasks.stm_inputs(TOTAL_INPUTS, rng)
    post = inputs[WASHOUT:]
    targets = np.column_stack(
        [tasks.delayed_target(post, delay) for delay in STM_DELAYS]
    )
    return couplings, inputs, targets


def build_reservoir(couplings: np.ndarray, condition: str) -> SparseLindbladReservoir:
    h0 = ising_xx_hamiltonian(couplings, H, N_QUBITS)
    hx = transverse_drive(N_QUBITS)
    return SparseLindbladReservoir.from_terms(
        N_QUBITS, h0 + H * hx, H * hx, jumps_for_condition(condition), DT
    )


def score_fixed_raw(features: np.ndarray, targets: np.ndarray) -> dict:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if features.shape != (FIT_ROWS + TEST, 45):
        raise RuntimeError(f"unexpected feature shape {features.shape}")
    x = readout.add_bias(features)
    capacities = []
    train_rows = []
    for column, delay in enumerate(STM_DELAYS):
        y = targets[:, column]
        training_mask = np.zeros(len(y), dtype=bool)
        training_mask[:FIT_ROWS] = True
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[FIT_ROWS:] = True
        training_mask &= np.isfinite(y)
        test_mask &= np.isfinite(y)
        if int(np.sum(training_mask)) != FIT_ROWS - delay or int(np.sum(test_mask)) != TEST:
            raise RuntimeError(f"delay {delay} finite-row invariant failed")
        weights = readout.train_readout(
            x[training_mask], y[training_mask], ridge=FIXED_RIDGE
        )
        prediction = readout.predict(x[test_mask], weights)
        capacities.append(float(readout.capacity(y[test_mask], prediction)))
        train_rows.append(int(np.sum(training_mask)))
    total = float(sum(capacities))
    if not math.isfinite(total) or not 0.0 <= total <= len(STM_DELAYS) + 1e-10:
        raise RuntimeError(f"invalid STM capacity {total}")
    return {
        "capacity": total,
        "delay_capacities": capacities,
        "ridge": FIXED_RIDGE,
        "effective_train_rows_by_delay": train_rows,
        "test_rows": TEST,
    }


def score_validation_sensitivity(features: np.ndarray, targets: np.ndarray) -> tuple[dict, dict]:
    raw_train = features[:TRAIN]
    raw_validation = features[TRAIN:FIT_ROWS]
    raw_test = features[FIT_ROWS:]
    guarded_train, guarded_validation, guarded_test, metadata = train_only_feature_guard(
        raw_train, raw_validation, raw_test, threshold=FEATURE_STD_TOL
    )
    result = select_and_refit_stm(
        guarded_train,
        targets[:TRAIN],
        guarded_validation,
        targets[TRAIN:FIT_ROWS],
        guarded_test,
        targets[FIT_ROWS:],
        ridges=RIDGES,
        fixed_ridge=FIXED_RIDGE,
    )
    return result, metadata


def pairing_hashes(seed: int, couplings: np.ndarray, inputs: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "seed": int(seed),
        "couplings_sha256": array_sha256(couplings),
        "full_input_sha256": array_sha256(inputs),
        "washout_prefix_sha256": array_sha256(inputs[:WASHOUT]),
        "post_wash_input_sha256": array_sha256(inputs[WASHOUT:]),
        "target_sha256": array_sha256(targets),
        "split_sha256": sha256_json(
            {"washout": WASHOUT, "train": TRAIN, "validation": VALIDATION, "test": TEST}
        ),
    }


def checkpoint_payload_hash(payload: Mapping) -> str:
    stripped = dict(payload)
    stripped.pop("checkpoint_sha256", None)
    return sha256_json(stripped)


def task_checkpoint_path(outdir: Path, job: Job) -> Path:
    return outdir / "task_checkpoints" / f"{job.key}.json"


def convergence_checkpoint_path(outdir: Path, job: Job) -> Path:
    return outdir / "convergence_checkpoints" / f"{job.key}.json"


def valid_checkpoint(path: Path, job: Job, protocol_sha256: str, artifact: str) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("artifact_type") != artifact
        or payload.get("status") != "complete"
        or payload.get("seed") != job.seed
        or payload.get("condition") != job.condition
        or payload.get("protocol_sha256") != protocol_sha256
        or payload.get("checkpoint_sha256") != checkpoint_payload_hash(payload)
    ):
        raise RuntimeError(f"invalid existing checkpoint: {path}")
    return payload


def run_task_job(job: Job, outdir: Path) -> dict:
    protocol = load_protocol(outdir)
    if job.seed not in protocol["seeds"] or job.condition not in protocol["conditions"]:
        raise ValueError(f"job outside frozen protocol: {job}")
    gate = authenticated_convergence_summary(outdir, protocol)
    if not gate["all_gates_passed"]:
        raise RuntimeError("task scores remain locked until convergence gates pass")
    path = task_checkpoint_path(outdir, job)
    stored = valid_checkpoint(path, job, protocol["protocol_sha256"], "phase_direction_task")
    if stored is not None:
        return stored

    started = time.perf_counter()
    couplings, inputs, targets = build_problem(job.seed)
    reservoir = build_reservoir(couplings, job.condition)
    observables = readout.pauli_observables(N_QUBITS, max_weight=2)
    features = reservoir.run(inputs, observables, washout=WASHOUT)
    if not np.all(np.isfinite(features)):
        raise RuntimeError("non-finite task features")
    primary = score_fixed_raw(features, targets)
    sensitivity, feature_guard = score_validation_sensitivity(features, targets)
    payload = {
        "artifact_type": "phase_direction_task",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_environment_sha256": protocol["source_environment_sha256"],
        "seed": int(job.seed),
        "condition": job.condition,
        "direction": direction_invariants(job.condition),
        "pairing": pairing_hashes(job.seed, couplings, inputs, targets),
        "feature_sha256": array_sha256(features),
        "feature_shape": list(features.shape),
        "primary_fixed_ridge": primary,
        "validation_selected_sensitivity": sensitivity,
        "feature_guard": feature_guard,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload["checkpoint_sha256"] = checkpoint_payload_hash(payload)
    atomic_json(path, payload)
    return payload


def density_states(seed: int) -> tuple[list[str], list[np.ndarray]]:
    dimension = 2**N_QUBITS
    ground = np.zeros((dimension, dimension), dtype=complex)
    ground[0, 0] = 1.0
    excited = np.zeros_like(ground)
    excited[-1, -1] = 1.0
    mixed = np.eye(dimension, dtype=complex) / dimension
    rng = np.random.default_rng(np.random.SeedSequence([HAAR_NAMESPACE, int(seed)]))
    vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    haar = np.outer(vector, vector.conjugate())
    return ["ground", "excited", "maximally_mixed", "haar_pure"], [ground, excited, mixed, haar]


def state_matrices(state_vectors: np.ndarray) -> list[np.ndarray]:
    dimension = 2**N_QUBITS
    return [unvec(state_vectors[:, index], dimension) for index in range(state_vectors.shape[1])]


def state_distances(state_vectors: np.ndarray, observable_matrices: np.ndarray) -> tuple[float, float, float, float]:
    states = state_matrices(state_vectors)
    trace_distances = []
    features = []
    for state in states:
        hermitian = 0.5 * (state + state.conjugate().T)
        features.append(np.real(np.einsum("kij,ji->k", observable_matrices, hermitian)))
    for left, right in itertools.combinations(range(len(states)), 2):
        delta = 0.5 * (
            states[left] - states[right]
            + (states[left] - states[right]).conjugate().T
        )
        trace_distances.append(float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(delta)))))
    feature_array = np.asarray(features)
    observable_max_abs = max(
        float(np.max(np.abs(feature_array[left] - feature_array[right])))
        for left, right in itertools.combinations(range(len(states)), 2)
    )
    trace_error = max(float(abs(np.trace(state) - 1.0)) for state in states)
    hermiticity_error = max(
        float(np.max(np.abs(state - state.conjugate().T))) for state in states
    )
    return max(trace_distances), observable_max_abs, trace_error, hermiticity_error


def run_convergence_job(job: Job, outdir: Path) -> dict:
    protocol = load_protocol(outdir)
    if job.seed not in protocol["audit_seeds"] or job.condition not in protocol["conditions"]:
        raise ValueError(f"convergence job outside protocol: {job}")
    path = convergence_checkpoint_path(outdir, job)
    stored = valid_checkpoint(path, job, protocol["protocol_sha256"], "phase_direction_convergence")
    if stored is not None:
        return stored

    started = time.perf_counter()
    couplings, inputs, targets = build_problem(job.seed)
    reservoir = build_reservoir(couplings, job.condition)
    observables = readout.pauli_observables(N_QUBITS, max_weight=2)
    observable_matrices = np.stack([observable.matrix for observable in observables])
    names, states = density_states(job.seed)
    state_vectors = np.stack([vec(state) for state in states], axis=1)
    features = np.empty((len(names), FIT_ROWS + TEST, len(observables)), dtype=float)
    trace_curve = []
    observable_curve = []
    numerical_trace_error = 0.0
    numerical_hermiticity_error = 0.0
    trace_distance, observable_max_abs, trace_error, hermiticity_error = state_distances(
        state_vectors, observable_matrices
    )
    trace_curve.append(trace_distance)
    observable_curve.append(observable_max_abs)
    numerical_trace_error = max(numerical_trace_error, trace_error)
    numerical_hermiticity_error = max(numerical_hermiticity_error, hermiticity_error)
    for index, input_value in enumerate(inputs):
        state_vectors = expm_multiply(
            reservoir.liouvillian(float(input_value)) * reservoir.dt,
            state_vectors,
        )
        step = index + 1
        if step <= CONVERGENCE_CURVE_STOP:
            trace_distance, observable_max_abs, trace_error, hermiticity_error = state_distances(
                state_vectors, observable_matrices
            )
            trace_curve.append(trace_distance)
            observable_curve.append(observable_max_abs)
            numerical_trace_error = max(numerical_trace_error, trace_error)
            numerical_hermiticity_error = max(numerical_hermiticity_error, hermiticity_error)
        if index >= WASHOUT:
            for state_index, state in enumerate(state_matrices(state_vectors)):
                features[state_index, index - WASHOUT] = np.real(
                    np.einsum("kij,ji->k", observable_matrices, state)
                )
    scores = [score_fixed_raw(features[index], targets)["capacity"] for index in range(len(names))]
    score_range = float(max(scores) - min(scores))
    trace_at_800 = float(trace_curve[800])
    observable_at_800 = float(observable_curve[800])
    gates = {
        "trace_distance_at_800": trace_at_800 <= TRACE_GATE_AT_800,
        "pauli_max_abs_at_800": observable_at_800 <= OBSERVABLE_GATE_AT_800,
        "fixed_ridge_score_range": score_range <= SCORE_RANGE_GATE,
    }
    payload = {
        "artifact_type": "phase_direction_convergence",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_environment_sha256": protocol["source_environment_sha256"],
        "seed": int(job.seed),
        "condition": job.condition,
        "direction": direction_invariants(job.condition),
        "pairing": pairing_hashes(job.seed, couplings, inputs, targets),
        "initial_states": names,
        "trace_distance_by_step_0_to_1200": trace_curve,
        "pauli_max_abs_by_step_0_to_1200": observable_curve,
        "checkpoint_metrics": {
            str(step): {
                "trace_distance": float(trace_curve[step]),
                "pauli_max_abs": float(observable_curve[step]),
            }
            for step in CHECKPOINT_STEPS
        },
        "fixed_ridge_stm_range": score_range,
        "feature_sha256_by_initial_state": {
            name: array_sha256(features[index]) for index, name in enumerate(names)
        },
        "maximum_numerical_trace_error": numerical_trace_error,
        "maximum_numerical_hermiticity_error": numerical_hermiticity_error,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload["checkpoint_sha256"] = checkpoint_payload_hash(payload)
    atomic_json(path, payload)
    return payload


def smoke() -> dict:
    expected_seeds()
    invariants = [direction_invariants(condition) for condition in CONDITIONS]
    for row in invariants:
        if row["kossakowski_rank"] != 1:
            raise RuntimeError(f"rank gate failed for {row['condition']}")
        if not np.isclose(row["kossakowski_trace"], 5.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"trace gate failed for {row['condition']}")
        if not np.allclose(row["kossakowski_diagonal"], 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"diagonal gate failed for {row['condition']}")
        if not np.allclose(row["coefficient_magnitudes"], 1.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"magnitude gate failed for {row['condition']}")
        expected_spectrum = np.asarray([0.0, 0.0, 0.0, 0.0, 5.0])
        if not np.allclose(row["kossakowski_eigenvalues"], expected_spectrum, atol=1e-12):
            raise RuntimeError(f"spectrum gate failed for {row['condition']}")
        if not np.isclose(dsp.jump_strength(jumps_for_condition(row["condition"])), 80.0):
            raise RuntimeError(f"budget gate failed for {row['condition']}")

    equal_jump = jumps_for_condition(EQUAL_PHASE)[0][0]
    repository_equal = dsp.collective_loss(N_QUBITS, GAMMA)[0][0]
    equal_builder_error = float(np.max(np.abs(equal_jump - repository_equal)))
    global_jump = dsp.collective_loss(
        N_QUBITS,
        GAMMA,
        c=np.exp(1j * 0.731) * coefficients(ORTHOGONAL_FOURIER),
    )[0][0]
    global_phase_error = float(
        np.max(
            np.abs(
                sparse_evolve.dissipator_super(global_jump).toarray()
                - sparse_evolve.dissipator_super(
                    jumps_for_condition(ORTHOGONAL_FOURIER)[0][0]
                ).toarray()
            )
        )
    )
    dense_sparse_error = float(
        np.max(
            np.abs(
                dense_liouvillian.dissipator_super(equal_jump)
                - sparse_evolve.dissipator_super(equal_jump).toarray()
            )
        )
    )
    if max(equal_builder_error, global_phase_error, dense_sparse_error) > 1e-12:
        raise RuntimeError("backend/global-phase smoke gate failed")
    return {
        "status": "passed",
        "n_conditions": len(CONDITIONS),
        "n_seeds": N_SEEDS,
        "equal_builder_max_abs": equal_builder_error,
        "global_phase_dissipator_max_abs": global_phase_error,
        "dense_sparse_dissipator_max_abs": dense_sparse_error,
    }


def _run_parallel(
    jobs: Sequence[Job],
    worker,
    outdir: Path,
    workers: int,
) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    results = []
    if workers == 1:
        for job in jobs:
            row = worker(job, outdir)
            results.append(row)
            print(f"{job.key} complete", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(worker, job, outdir): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            row = future.result()
            results.append(row)
            print(f"{job.key} complete", flush=True)
    return results


def convergence_jobs() -> list[Job]:
    return [Job(condition, seed) for seed in AUDIT_SEEDS for condition in CONDITIONS]


def task_jobs() -> list[Job]:
    return [Job(condition, seed) for seed in CONFIRMATORY_SEEDS for condition in CONDITIONS]


def convergence_summary(outdir: Path) -> dict:
    protocol = load_protocol(outdir)
    rows = []
    for job in convergence_jobs():
        row = valid_checkpoint(
            convergence_checkpoint_path(outdir, job),
            job,
            protocol["protocol_sha256"],
            "phase_direction_convergence",
        )
        if row is None:
            raise RuntimeError(f"missing convergence checkpoint {job.key}")
        rows.append(row)
    failed = [
        {"seed": row["seed"], "condition": row["condition"], "gates": row["gates"]}
        for row in rows
        if not row["all_gates_passed"]
    ]
    payload = {
        "artifact_type": "phase_direction_convergence_summary",
        "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "n_expected": len(convergence_jobs()),
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
    payload["summary_sha256"] = sha256_json(payload)
    atomic_json(outdir / "convergence_summary.json", payload)
    if failed:
        raise RuntimeError(f"convergence gates failed for {len(failed)} jobs")
    return payload


def authenticated_convergence_summary(outdir: Path, protocol: dict | None = None) -> dict:
    if protocol is None:
        protocol = load_protocol(outdir)
    path = outdir / "convergence_summary.json"
    if not path.is_file():
        raise FileNotFoundError("convergence summary is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.get("summary_sha256")
    unhashed = dict(payload)
    unhashed.pop("summary_sha256", None)
    expected_keys = {job.key for job in convergence_jobs()}
    if (
        payload.get("artifact_type") != "phase_direction_convergence_summary"
        or payload.get("status") != "complete"
        or payload.get("protocol_sha256") != protocol["protocol_sha256"]
        or stored_hash != sha256_json(unhashed)
        or payload.get("n_expected") != len(expected_keys)
        or payload.get("n_complete") != len(expected_keys)
        or set(payload.get("checkpoint_sha256s", {})) != expected_keys
        or payload.get("failed_jobs") != []
        or payload.get("all_gates_passed") is not True
    ):
        raise RuntimeError("convergence summary authentication failed")
    return payload


def paired_summary(values: np.ndarray, *, simultaneous_count: int = 1) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    point_critical = float(stats.t.ppf(0.975, n - 1))
    simultaneous_critical = float(
        stats.t.ppf(1.0 - 0.05 / (2.0 * simultaneous_count), n - 1)
    )
    return {
        "n": n,
        "values": [float(value) for value in values],
        "mean": mean,
        "se": se,
        "ci95_student_t": [mean - point_critical * se, mean + point_critical * se],
        "ci95_bonferroni_simultaneous": [
            mean - simultaneous_critical * se,
            mean + simultaneous_critical * se,
        ],
        "wins": int(np.sum(values > 0.0)),
        "ties": int(np.sum(values == 0.0)),
        "losses": int(np.sum(values < 0.0)),
        "paired_t_p_two_sided": float(stats.ttest_1samp(values, 0.0).pvalue),
        "sign_test_p_two_sided": float(
            stats.binomtest(int(np.sum(values > 0.0)), n - int(np.sum(values == 0.0)), 0.5).pvalue
        ),
    }


def monte_carlo_signflip_p(values: np.ndarray, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(values)))
    exceed = 0
    completed = 0
    batch = 5000
    while completed < SIGNFLIP_DRAWS:
        count = min(batch, SIGNFLIP_DRAWS - completed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(count, len(values)))
        statistics = np.abs(np.mean(signs * values[None, :], axis=1))
        exceed += int(np.sum(statistics >= observed - 1e-14))
        completed += count
    return float((exceed + 1) / (SIGNFLIP_DRAWS + 1))


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for index, key in enumerate(ordered):
        candidate = min(1.0, (m - index) * float(pvalues[key]))
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def ordered_path_diagnostic(matrix: np.ndarray, rng: np.random.Generator) -> dict:
    matrix = np.asarray(matrix, dtype=float)
    overlaps = np.asarray(
        [direction_invariants(condition)["uniform_direction_overlap"] for condition in PHASE_PATH]
    )
    observed_rhos = np.asarray(
        [stats.spearmanr(overlaps, row).statistic for row in matrix], dtype=float
    )
    observed = float(np.mean(observed_rhos))
    exceed = 0
    completed = 0
    batch_size = 1000
    centered_overlap = stats.rankdata(overlaps) - 3.0
    overlap_norm = float(np.sqrt(np.sum(centered_overlap**2)))
    score_ranks = np.asarray([stats.rankdata(row) - 3.0 for row in matrix])
    score_norms = np.sqrt(np.sum(score_ranks**2, axis=1))
    while completed < ORDERED_PATH_DRAWS:
        count = min(batch_size, ORDERED_PATH_DRAWS - completed)
        random_order = np.argsort(rng.random((count, matrix.shape[0], matrix.shape[1])), axis=2)
        permuted = np.take_along_axis(score_ranks[None, :, :], random_order, axis=2)
        rhos = np.sum(permuted * centered_overlap[None, None, :], axis=2) / (
            score_norms[None, :] * overlap_norm
        )
        statistics = np.mean(rhos, axis=1)
        exceed += int(np.sum(np.abs(statistics) >= abs(observed) - 1e-14))
        completed += count
    mean = float(np.mean(observed_rhos))
    se = float(stats.sem(observed_rhos))
    critical = float(stats.t.ppf(0.975, len(observed_rhos) - 1))
    return {
        "estimand": "mean within-seed Spearman rho(overlap, STM) along five-point path",
        "seed_rhos": [float(value) for value in observed_rhos],
        "mean_rho": mean,
        "ci95_student_t": [mean - critical * se, mean + critical * se],
        "monte_carlo_label_permutation_p_two_sided": float(
            (exceed + 1) / (ORDERED_PATH_DRAWS + 1)
        ),
        "draws": ORDERED_PATH_DRAWS,
    }


def _task_rows(outdir: Path) -> list[dict]:
    protocol = load_protocol(outdir)
    rows = []
    for job in task_jobs():
        row = valid_checkpoint(
            task_checkpoint_path(outdir, job),
            job,
            protocol["protocol_sha256"],
            "phase_direction_task",
        )
        if row is None:
            raise RuntimeError(f"missing task checkpoint {job.key}")
        rows.append(row)
    return rows


def build_aggregate(outdir: Path) -> dict:
    protocol = load_protocol(outdir)
    convergence = json.loads((outdir / "convergence_summary.json").read_text(encoding="utf-8"))
    if not convergence.get("all_gates_passed"):
        raise RuntimeError("cannot aggregate task scores after a failed convergence gate")
    rows = _task_rows(outdir)
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
        for condition in CONDITIONS
    }
    if any(len(values["fixed"]) != N_SEEDS for values in by_condition.values()):
        raise RuntimeError("condition sample-size invariant failed")
    summaries = []
    for condition in CONDITIONS:
        fixed = by_condition[condition]["fixed"]
        selected = by_condition[condition]["selected"]
        summaries.append(
            {
                **direction_invariants(condition),
                "n": N_SEEDS,
                "fixed_ridge_mean": float(np.mean(fixed)),
                "fixed_ridge_se": float(stats.sem(fixed)),
                "fixed_ridge_values": [float(value) for value in fixed],
                "validation_selected_mean": float(np.mean(selected)),
                "validation_selected_se": float(stats.sem(selected)),
                "validation_selected_values": [float(value) for value in selected],
            }
        )

    inference_rng = np.random.default_rng(INFERENCE_NAMESPACE)
    primary_values = by_condition[EQUAL_PHASE]["fixed"] - by_condition[ORTHOGONAL_FOURIER]["fixed"]
    primary = paired_summary(primary_values)
    primary["estimand"] = "STM(path_f0)-STM(path_f1)"
    primary["monte_carlo_signflip_p_two_sided"] = monte_carlo_signflip_p(
        primary_values, inference_rng
    )
    primary["draws"] = SIGNFLIP_DRAWS

    zero_mean = np.mean(
        np.stack([by_condition[condition]["fixed"] for condition in ZERO_OVERLAP_CONDITIONS]),
        axis=0,
    )
    pooled_values = by_condition[EQUAL_PHASE]["fixed"] - zero_mean
    pooled = paired_summary(pooled_values)
    pooled["estimand"] = "STM(path_f0)-within-seed mean(five zero-overlap rays)"
    pooled["monte_carlo_signflip_p_two_sided"] = monte_carlo_signflip_p(
        pooled_values, inference_rng
    )
    pooled["draws"] = SIGNFLIP_DRAWS
    pooled["tested_by_fixed_sequence_gate"] = bool(
        primary["monte_carlo_signflip_p_two_sided"] <= 0.05
    )
    pooled["gatekeeping_rejects_at_0.05"] = bool(
        pooled["tested_by_fixed_sequence_gate"]
        and pooled["monte_carlo_signflip_p_two_sided"] <= 0.05
    )

    secondary_conditions = (
        "path_f025", "path_f05", "path_f075",
        "scrambled_r1", "scrambled_r2", "scrambled_r3", "scrambled_r4",
    )
    secondary = {}
    raw_p = {}
    for condition in secondary_conditions:
        values = by_condition[EQUAL_PHASE]["fixed"] - by_condition[condition]["fixed"]
        item = paired_summary(values, simultaneous_count=len(secondary_conditions))
        item["estimand"] = f"STM(path_f0)-STM({condition})"
        item["monte_carlo_signflip_p_two_sided"] = monte_carlo_signflip_p(
            values, inference_rng
        )
        item["draws"] = SIGNFLIP_DRAWS
        secondary[condition] = item
        raw_p[condition] = item["monte_carlo_signflip_p_two_sided"]
    adjusted = holm_adjust(raw_p)
    for condition, value in adjusted.items():
        secondary[condition]["holm_adjusted_signflip_p"] = float(value)

    selected_primary_values = (
        by_condition[EQUAL_PHASE]["selected"] - by_condition[ORTHOGONAL_FOURIER]["selected"]
    )
    selected_primary = paired_summary(selected_primary_values)
    selected_primary["estimand"] = "validation-selected STM(path_f0)-STM(path_f1)"

    path_matrix = np.column_stack(
        [by_condition[condition]["fixed"] for condition in PHASE_PATH]
    )
    ordered = ordered_path_diagnostic(path_matrix, inference_rng)
    payload = {
        "artifact_type": "phase_direction_confirmatory_aggregate",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol["protocol_sha256"],
        "source_environment_sha256": protocol["source_environment_sha256"],
        "pilot_scores_included": False,
        "n_seeds": N_SEEDS,
        "n_conditions": len(CONDITIONS),
        "n_task_checkpoints": len(rows),
        "condition_summaries": summaries,
        "confirmatory_primary": primary,
        "gated_zero_overlap_generality": pooled,
        "secondary_contrasts": secondary,
        "validation_selected_sensitivity": selected_primary,
        "ordered_path_diagnostic": ordered,
        "convergence_summary_sha256": convergence["summary_sha256"],
        "task_checkpoint_sha256s": {
            f"{row['condition']}__s{row['seed']}": row["checkpoint_sha256"]
            for row in rows
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    payload["aggregate_sha256"] = sha256_json(payload)
    atomic_json(outdir / "aggregate.json", payload)
    return payload


def make_plot(outdir: Path) -> tuple[Path, Path]:
    aggregate = json.loads((outdir / "aggregate.json").read_text(encoding="utf-8"))
    summaries = {row["condition"]: row for row in aggregate["condition_summaries"]}
    purple = "#762a83"
    orange = "#e67e22"
    gray = "#9e9e9e"
    fig, axes = plt.subplots(1, 2, figsize=(6.6929, 2.55))

    path_conditions = list(PHASE_PATH)
    fractions = np.asarray([PHASE_PATH[key] for key in path_conditions])
    matrix = np.column_stack(
        [np.asarray(summaries[key]["fixed_ridge_values"], dtype=float) for key in path_conditions]
    )
    for row in matrix:
        axes[0].plot(fractions, row, color=purple, alpha=0.10, linewidth=0.7)
    means = np.mean(matrix, axis=0)
    errors = stats.sem(matrix, axis=0) * stats.t.ppf(0.975, N_SEEDS - 1)
    axes[0].errorbar(
        fractions, means, yerr=errors, color=purple, marker="o", linewidth=2.0,
        capsize=3, label="mean and 95% interval",
    )
    axes[0].set_xlabel(r"phase-gradient fraction $f$")
    axes[0].set_ylabel("STM capacity")
    axes[0].set_title("(a) Prespecified phase path", loc="left")
    axes[0].set_xticks(fractions)
    axes[0].grid(True, color="#d9dfe5", linewidth=0.6, alpha=0.8)

    zero_conditions = list(ZERO_OVERLAP_CONDITIONS)
    equal = np.asarray(summaries[EQUAL_PHASE]["fixed_ridge_values"], dtype=float)
    differences = np.column_stack(
        [equal - np.asarray(summaries[key]["fixed_ridge_values"], dtype=float) for key in zero_conditions]
    )
    pooled = np.mean(differences, axis=1)
    all_values = np.column_stack([differences, pooled])
    x = np.arange(all_values.shape[1])
    for column in range(all_values.shape[1]):
        jitter = np.linspace(-0.10, 0.10, N_SEEDS)
        axes[1].scatter(
            np.full(N_SEEDS, x[column]) + jitter,
            all_values[:, column],
            s=7,
            color=gray if column < len(zero_conditions) else purple,
            alpha=0.30,
            linewidths=0,
        )
    effect_means = np.mean(all_values, axis=0)
    effect_errors = stats.sem(all_values, axis=0) * stats.t.ppf(0.975, N_SEEDS - 1)
    axes[1].errorbar(
        x[:-1], effect_means[:-1], yerr=effect_errors[:-1], fmt="o",
        color=gray, markeredgecolor="#606060", capsize=3,
    )
    axes[1].errorbar(
        x[-1], effect_means[-1], yerr=effect_errors[-1], fmt="o",
        color=purple, markersize=7, capsize=3,
    )
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
    axes[1].set_xticks(x, ["DFT", "A", "B", "C", "D", "mean"])
    axes[1].set_xlabel("zero-overlap direction")
    axes[1].set_ylabel("STM difference (equal phase - control)")
    axes[1].set_title("(b) Direction-control effects", loc="left")
    axes[1].grid(True, axis="y", color="#d9dfe5", linewidth=0.6, alpha=0.8)
    fig.tight_layout(w_pad=2.0)
    png = outdir / "phase_direction_confirmatory.png"
    pdf = outdir / "phase_direction_confirmatory.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def validate(outdir: Path) -> dict:
    protocol = load_protocol(outdir)
    ledger = json.loads((outdir / "directions.json").read_text(encoding="utf-8"))
    if ledger.get("direction_ledger_sha256") != protocol["directions_sha256"]:
        raise RuntimeError("direction ledger hash mismatch")
    convergence = convergence_summary(outdir)
    if not convergence["all_gates_passed"]:
        raise RuntimeError("convergence gate is not satisfied")
    rows = _task_rows(outdir)
    pairing_by_seed: dict[int, set[str]] = {}
    for row in rows:
        pairing_by_seed.setdefault(int(row["seed"]), set()).add(
            sha256_json(row["pairing"])
        )
        invariant = row["direction"]
        if invariant["kossakowski_rank"] != 1:
            raise RuntimeError("rank invariant failed in a task row")
        if not np.allclose(invariant["kossakowski_diagonal"], 1.0, atol=1e-12):
            raise RuntimeError("diagonal invariant failed in a task row")
        if not np.allclose(invariant["coefficient_magnitudes"], 1.0, atol=1e-12):
            raise RuntimeError("magnitude invariant failed in a task row")
        if not np.isclose(sum(row["primary_fixed_ridge"]["delay_capacities"]), row["primary_fixed_ridge"]["capacity"], atol=1e-12):
            raise RuntimeError("lag capacities do not reconstruct STM")
    if any(len(hashes) != 1 for hashes in pairing_by_seed.values()):
        raise RuntimeError("condition pairing hash mismatch")
    if set(pairing_by_seed) != set(CONFIRMATORY_SEEDS):
        raise RuntimeError("task seed ledger mismatch")

    aggregate = build_aggregate(outdir)
    make_plot(outdir)
    if aggregate.get("pilot_scores_included") is not False:
        raise RuntimeError("pilot exclusion invariant failed")
    audit_ground_hashes = {
        (job.seed, job.condition): json.loads(
            convergence_checkpoint_path(outdir, job).read_text(encoding="utf-8")
        )["feature_sha256_by_initial_state"]["ground"]
        for job in convergence_jobs()
    }
    for row in rows:
        key = (int(row["seed"]), row["condition"])
        if key in audit_ground_hashes and row["feature_sha256"] != audit_ground_hashes[key]:
            raise RuntimeError(f"ground trajectory replay mismatch for {key}")

    required = [
        outdir / "protocol.json", outdir / "directions.json", outdir / "smoke.json",
        outdir / "convergence_summary.json", outdir / "aggregate.json",
        outdir / "phase_direction_confirmatory.png",
        outdir / "phase_direction_confirmatory.pdf",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    files = sorted(
        [path for path in outdir.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "validation_report.json"}],
        key=lambda path: path.relative_to(outdir).as_posix(),
    )
    checksums = {path.relative_to(outdir).as_posix(): file_sha256(path) for path in files}
    report = {
        "status": "validated_confirmatory_result",
        "protocol_sha256": protocol["protocol_sha256"],
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "n_task_checkpoints": len(rows),
        "n_convergence_checkpoints": len(convergence_jobs()),
        "n_seeds": N_SEEDS,
        "n_conditions": len(CONDITIONS),
        "all_pairing_hashes_match": True,
        "all_convergence_gates_pass": True,
        "pilot_scores_included": False,
        "primary": aggregate["confirmatory_primary"],
        "gated_zero_overlap_generality": aggregate["gated_zero_overlap_generality"],
        "source_environment_sha256": protocol["source_environment_sha256"],
        "file_count_excluding_validation_files": len(files),
        "claim_boundary": protocol["claim_boundary"],
    }
    report["validation_report_sha256"] = sha256_json(report)
    atomic_json(outdir / "validation_report.json", report)
    checksum_lines = [f"{digest}  {relative}" for relative, digest in checksums.items()]
    checksum_lines.append(
        f"{file_sha256(outdir / 'validation_report.json')}  validation_report.json"
    )
    (outdir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("smoke", "freeze", "convergence", "run", "aggregate", "validate", "all"),
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    outdir = args.outdir.resolve()

    if args.command == "smoke":
        print(json.dumps(smoke(), indent=2, sort_keys=True))
        return
    protocol = freeze(outdir)
    if args.command == "freeze":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return
    if args.command in ("convergence", "all"):
        _run_parallel(convergence_jobs(), run_convergence_job, outdir, args.workers)
        summary = convergence_summary(outdir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.command == "convergence":
            return
    if args.command in ("run", "all"):
        convergence_summary(outdir)
        _run_parallel(task_jobs(), run_task_job, outdir, args.workers)
        if args.command == "run":
            return
    if args.command in ("aggregate", "all"):
        aggregate = build_aggregate(outdir)
        make_plot(outdir)
        print(json.dumps(aggregate["confirmatory_primary"], indent=2, sort_keys=True))
        if args.command == "aggregate":
            return
    if args.command in ("validate", "all"):
        if not (outdir / "aggregate.json").is_file():
            build_aggregate(outdir)
        if not (outdir / "phase_direction_confirmatory.png").is_file():
            make_plot(outdir)
        report = validate(outdir)
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

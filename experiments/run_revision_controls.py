"""Focused robustness controls requested for the Quantum revision.

This additive driver does not alter the canonical protocol or paper.  It closes
two specific validation gaps:

1. Parity at N=5 for the six active dissipative channels, using 15 virtual
   nodes, independent train/validation/test input sequences, validation-selected
   ridge regularisation, and a train+validation refit with more rows than
   features.  The historical fixed ridge (1e-8) is scored on the same held-out
   test rows.
2. Local-versus-collective STM/NARMA scaling at N=4,...,8 after normalising the
   coupling scale.  The variance-normalised factor sqrt(4/(N-1)) and optional
   Kac factor 4/(N-1) are both anchored to one at N=5.  Every method and
   normalisation uses the same nested base coupling draw and input sequence.

Every expensive trajectory is an atomic JSON checkpoint.  Re-running a command
skips validated checkpoints.  Heavy work is guarded by ``__main__``.

Examples
--------
PYTHONPATH=../src python run_revision_controls.py parity --workers 4
PYTHONPATH=../src python run_revision_controls.py scaling --workers 4 \
    --schemes variance
PYTHONPATH=../src python run_revision_controls.py report
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
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import _paths  # noqa: F401
import numpy as np

from _paths import REPORTS_DIR, RESULTS_DIR
from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir
from run_final_scaling import build_jumps


PROTOCOL_VERSION = "revision-controls-v1-2026-07-23"
GAMMA = 1.0
H = 0.5
DT = 0.5
FIXED_RIDGE = 1e-8
FEATURE_STD_TOL = 1e-12
# Logarithmic ridge grid plus the explicit unregularised/minimum-norm endpoint.
# Including zero prevents a smallest-positive-grid boundary from masquerading as
# an optimum in the overparameterised parity validation split.
RIDGES = (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2)
PARITY_METHODS = (
    "CD_paper",
    "A1_heterogeneous",
    "B2_thermal",
    "B3_collective",
    "B4_loss_exchange",
    "B5_pair",
)
PARITY_REFERENCE_METHODS = ("FN", "B1_dephasing")
SCALING_METHODS = ("CD_paper", "B3_collective")
COUPLING_SCHEMES = ("variance", "kac")
METHOD_LABELS = {
    "FN": "reset-unitary reference",
    "CD_paper": "uniform local",
    "A1_heterogeneous": "unequal local",
    "B2_thermal": "local gain/loss",
    "B3_collective": "collective",
    "B4_loss_exchange": "loss + exchange",
    "B5_pair": "pair loss",
    "B1_dephasing": "dephasing",
}
MAX_BASE_N = 8
PARITY_SEED_NAMESPACE = 2026072301
SCALING_SEED_NAMESPACE = 2026072302
OLD_PROTOCOL_SEED = 2024
BOOTSTRAP_DRAWS = 20_000

REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_DIR = Path(RESULTS_DIR) / "revision_parity_control"
SCALING_DIR = Path(RESULTS_DIR) / "revision_normalized_scaling"
REPORT_PATH = Path(REPORTS_DIR) / "revision_controls_report.md"


@dataclass(frozen=True)
class ParityPreset:
    name: str
    n_seeds: int
    wash: int
    train: int
    validation: int
    test: int
    delays: tuple[int, ...]
    n_virtual: int = 15
    h: float = H
    dt: float = DT

    @property
    def n_features(self) -> int:
        # 45 observables * 15 virtual nodes + a fitted bias.
        return 45 * self.n_virtual + 1

    @property
    def refit_rows(self) -> int:
        # Split construction removes max(delays) before returning usable rows.
        return self.train + self.validation


@dataclass(frozen=True)
class ScalingPreset:
    name: str
    n_seeds: int
    n_values: tuple[int, ...]
    wash: int
    train: int
    validation: int
    test: int
    delays: tuple[int, ...]
    h: float = H
    dt: float = DT

    @property
    def total_len(self) -> int:
        return self.wash + self.train + self.validation + self.test


PARITY_PRESETS = {
    "smoke": ParityPreset(
        "smoke", n_seeds=2, wash=10, train=70, validation=30, test=25,
        delays=(1, 2), n_virtual=2,
    ),
    "paper": ParityPreset(
        "paper", n_seeds=16, wash=100, train=500, validation=250, test=400,
        delays=tuple(range(1, 8)), n_virtual=15,
    ),
}
SCALING_PRESETS = {
    "smoke": ScalingPreset(
        "smoke", n_seeds=2, n_values=(4, 5), wash=10, train=30,
        validation=20, test=25, delays=(1, 2),
    ),
    "paper": ScalingPreset(
        "paper", n_seeds=8, n_values=(4, 5, 6, 7, 8), wash=100, train=300,
        validation=150, test=250, delays=tuple(range(1, 21)),
    ),
}


@dataclass(frozen=True)
class ParityJob:
    method: str
    seed: int


@dataclass(frozen=True)
class ScalingJob:
    scheme: str
    n_qubits: int
    method: str
    seed: int


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def legacy_seeds(n: int = 64) -> set[int]:
    """Seeds used by the definitive 2024-namespace protocol."""
    return {
        int(value)
        for value in np.random.default_rng(OLD_PROTOCOL_SEED).integers(
            0, 2**31 - 1, n
        )
    }


def fresh_seeds(namespace: int, n: int) -> list[int]:
    """Return deterministic seeds disjoint from the definitive protocol.

    The explicit rejection loop is part of the protocol: an accidental collision
    with the earlier 64-seed pool cannot silently turn this into a reanalysis.
    """
    old = legacy_seeds()
    rng = np.random.default_rng(namespace)
    selected: list[int] = []
    while len(selected) < n:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in old and candidate not in selected:
            selected.append(candidate)
    return selected


def _method_seed(seed: int, method: str) -> int:
    digest = hashlib.sha256(f"{seed}:{method}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def coupling_multiplier(n_qubits: int, scheme: str) -> float:
    """Coupling multiplier anchored to the original N=5 scale.

    ``variance`` fixes the expected per-site interaction variance:
    (N-1) Var(multiplier * J_ij) = 4 Var(J_ij).
    ``kac`` fixes the extensive row-sum scale using the standard 1/(N-1)
    prescription, again normalised to one at N=5.
    """
    if n_qubits < 2:
        raise ValueError("n_qubits must be at least two")
    if scheme == "variance":
        return math.sqrt(4.0 / (n_qubits - 1))
    if scheme == "kac":
        return 4.0 / (n_qubits - 1)
    if scheme == "raw":
        return 1.0
    raise ValueError(f"unknown coupling scheme {scheme!r}")


def nested_base_couplings(seed: int, n_qubits: int) -> np.ndarray:
    """Nested U[-1,1] base draw: every N is a leading principal submatrix."""
    if not 2 <= n_qubits <= MAX_BASE_N:
        raise ValueError(f"n_qubits must lie in [2, {MAX_BASE_N}]")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xC011EC7]))
    upper = np.zeros((MAX_BASE_N, MAX_BASE_N), dtype=float)
    indices = np.triu_indices(MAX_BASE_N, k=1)
    upper[indices] = rng.uniform(-1.0, 1.0, size=len(indices[0]))
    full = upper + upper.T
    return full[:n_qubits, :n_qubits].copy()


def scaled_couplings(seed: int, n_qubits: int, scheme: str) -> tuple[np.ndarray, dict]:
    base = nested_base_couplings(seed, n_qubits)
    multiplier = coupling_multiplier(n_qubits, scheme)
    scaled = multiplier * base
    return scaled, {
        "base_coupling_sha256": _array_sha256(base),
        "scaled_coupling_sha256": _array_sha256(scaled),
        "coupling_multiplier": multiplier,
        "base_upper_rms": _upper_rms(base),
        "scaled_upper_rms": _upper_rms(scaled),
        "base_spectral_norm": float(np.linalg.norm(base, ord=2)),
        "scaled_spectral_norm": float(np.linalg.norm(scaled, ord=2)),
    }


def _upper_rms(matrix: np.ndarray) -> float:
    values = matrix[np.triu_indices(matrix.shape[0], k=1)]
    return float(np.sqrt(np.mean(values**2)))


def _build_exact_reservoir(
    method: str,
    couplings: np.ndarray,
    n_qubits: int,
    seed: int,
    h: float,
    dt: float,
) -> tuple[SparseLindbladReservoir, float, float]:
    target_strength = dsp.jump_strength(dsp.local_loss(n_qubits, GAMMA))
    rng = np.random.default_rng(_method_seed(seed, method))
    jumps = build_jumps(method, couplings, n_qubits, target_strength, rng)
    actual_strength = dsp.jump_strength(jumps)
    if not np.isclose(actual_strength, target_strength, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            f"matched jump budget failed for {method}: "
            f"{actual_strength} != {target_strength}"
        )
    h0 = ising_xx_hamiltonian(couplings, h, n_qubits)
    hx = transverse_drive(n_qubits)
    reservoir = SparseLindbladReservoir.from_terms(
        n_qubits, h0 + h * hx, h * hx, jumps, dt
    )
    return reservoir, float(actual_strength), float(target_strength)


def _svd_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    ridges: Sequence[float],
) -> dict[float, np.ndarray]:
    """Fit all ridge values from one thin SVD.

    ``y_train`` is two-dimensional so all parity delays share the same
    decomposition and row support.
    """
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    u, singular, vt = np.linalg.svd(x_train, full_matrices=False)
    uy = u.T @ y_train
    tol = np.finfo(float).eps * max(x_train.shape) * singular[0]
    predictions: dict[float, np.ndarray] = {}
    for ridge in ridges:
        ridge = float(ridge)
        if ridge < 0:
            raise ValueError("ridge values must be nonnegative")
        if ridge == 0:
            factors = np.divide(
                1.0,
                singular,
                out=np.zeros_like(singular),
                where=singular > tol,
            )
        else:
            factors = singular / (singular**2 + ridge)
        weights = vt.T @ (factors[:, None] * uy)
        predictions[ridge] = x_eval @ weights
    return predictions


def _capacity_columns(y: np.ndarray, yhat: np.ndarray) -> list[float]:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return [readout.capacity(y[:, col], yhat[:, col]) for col in range(y.shape[1])]


def _nmse_columns(y: np.ndarray, yhat: np.ndarray) -> list[float]:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return [readout.nmse(y[:, col], yhat[:, col]) for col in range(y.shape[1])]


def select_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    ridges: Sequence[float],
    metric: str,
) -> tuple[float, dict[str, float], dict[str, list[float]]]:
    """Select ridge using validation only; prefer the stronger ridge on a tie."""
    predictions = _svd_ridge_predict(x_train, y_train, x_validation, ridges)
    totals: dict[str, float] = {}
    by_target: dict[str, list[float]] = {}
    for ridge in ridges:
        key = f"{float(ridge):.12g}"
        if metric == "capacity":
            scores = _capacity_columns(y_validation, predictions[float(ridge)])
            total = float(sum(scores))
        elif metric == "nmse":
            scores = _nmse_columns(y_validation, predictions[float(ridge)])
            total = float(np.mean(scores))
        else:
            raise ValueError(f"unknown metric {metric!r}")
        totals[key] = total
        by_target[key] = scores
    if metric == "capacity":
        best_value = max(totals.values())
        candidates = [
            float(key) for key, value in totals.items()
            if np.isclose(value, best_value, rtol=0.0, atol=1e-12)
        ]
    else:
        best_value = min(totals.values())
        candidates = [
            float(key) for key, value in totals.items()
            if np.isclose(value, best_value, rtol=0.0, atol=1e-12)
        ]
    return max(candidates), totals, by_target


def refit_and_test(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ridge: float,
    metric: str,
) -> tuple[float, list[float]]:
    x_refit = np.vstack([x_train, x_validation])
    y_refit = np.vstack([y_train, y_validation])
    predictions = _svd_ridge_predict(x_refit, y_refit, x_test, (ridge,))[ridge]
    if metric == "capacity":
        scores = _capacity_columns(y_test, predictions)
        return float(sum(scores)), scores
    if metric == "nmse":
        scores = _nmse_columns(y_test, predictions)
        return float(np.mean(scores)), scores
    raise ValueError(f"unknown metric {metric!r}")


def _parity_problem(seed: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    children = np.random.SeedSequence([seed, 0xA11CE]).spawn(4)
    coupling_rng = np.random.default_rng(children[0])
    couplings = res.random_couplings(5, 1.0, coupling_rng)
    return couplings, {
        "train": np.random.default_rng(children[1]),
        "validation": np.random.default_rng(children[2]),
        "test": np.random.default_rng(children[3]),
    }


def _parity_split(
    reservoir: SparseLindbladReservoir,
    observables,
    rng: np.random.Generator,
    n_rows: int,
    preset: ParityPreset,
) -> tuple[np.ndarray, np.ndarray, str]:
    max_delay = max(preset.delays)
    inputs = tasks.parity_inputs(preset.wash + max_delay + n_rows, rng)
    features = reservoir.run(
        inputs, observables, washout=preset.wash, n_virtual=preset.n_virtual
    )
    post = inputs[preset.wash:]
    targets = np.column_stack(
        [tasks.parity_target(post, delay) for delay in preset.delays]
    )
    features = features[max_delay:]
    targets = targets[max_delay:]
    if features.shape[0] != n_rows or targets.shape[0] != n_rows:
        raise RuntimeError("parity split row-count invariant failed")
    if not np.all(np.isfinite(targets)):
        raise RuntimeError("parity split contains undefined targets")
    return readout.add_bias(features), targets, _array_sha256(inputs)


def train_only_variance_filter(
    x_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    threshold: float = FEATURE_STD_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Drop numerically constant non-bias features using training rows only.

    All parity design matrices already contain the bias in their last column.
    Validation and test rows never influence the mask.
    """
    if threshold <= 0:
        raise ValueError("feature-variance threshold must be positive")
    matrices = [
        np.asarray(matrix, dtype=float)
        for matrix in (x_train, x_validation, x_test)
    ]
    n_columns = matrices[0].shape[1]
    if n_columns < 2 or any(matrix.ndim != 2 for matrix in matrices):
        raise ValueError("feature matrices must be two-dimensional with a bias")
    if any(matrix.shape[1] != n_columns for matrix in matrices):
        raise ValueError("train/validation/test feature counts differ")
    if not np.allclose(matrices[0][:, -1], 1.0):
        raise ValueError("last training column must be the constant bias")
    train_std = np.std(matrices[0][:, :-1], axis=0)
    retained_nonbias = train_std > threshold
    retained = np.append(retained_nonbias, True)
    filtered = tuple(matrix[:, retained] for matrix in matrices)
    dropped_indices = np.flatnonzero(~retained_nonbias).astype(int).tolist()
    retained_indices = np.flatnonzero(retained_nonbias).astype(int).tolist()
    metadata = {
        "fit_on": "training rows only",
        "threshold": float(threshold),
        "raw_nonbias_features": int(n_columns - 1),
        "retained_nonbias_features": int(np.sum(retained_nonbias)),
        "dropped_nonbias_features": int(np.sum(~retained_nonbias)),
        "retained_features_including_bias": int(np.sum(retained)),
        "retained_nonbias_indices": retained_indices,
        "dropped_nonbias_indices": dropped_indices,
        "training_std_min": float(np.min(train_std)),
        "training_std_max": float(np.max(train_std)),
        "training_std_min_retained": (
            None
            if not np.any(retained_nonbias)
            else float(np.min(train_std[retained_nonbias]))
        ),
        "training_std_max_dropped": (
            None
            if np.all(retained_nonbias)
            else float(np.max(train_std[~retained_nonbias]))
        ),
    }
    return filtered[0], filtered[1], filtered[2], metadata


def parity_protocol(preset: ParityPreset) -> dict:
    if preset.name == "paper" and preset.n_virtual != 15:
        raise ValueError("paper parity control requires 15 virtual nodes")
    if preset.refit_rows <= preset.n_features:
        raise ValueError(
            f"refit requires rows > features, got {preset.refit_rows} <= "
            f"{preset.n_features}"
        )
    seeds = fresh_seeds(PARITY_SEED_NAMESPACE, preset.n_seeds)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "control": "parity_regularisation",
        "preset": asdict(preset),
        "seeds": seeds,
        "fresh_seed_namespace": PARITY_SEED_NAMESPACE,
        "disjoint_from_definitive_seed_pool": not bool(set(seeds) & legacy_seeds()),
        "methods": list(PARITY_METHODS),
        "matched_jump_budget": "local-loss Frobenius strength at gamma=1",
        "split_design": (
            "independent RNG streams and independent reservoir resets for "
            "train, validation, and test"
        ),
        "common_delay_support": True,
        "ridge_grid": list(RIDGES),
        "ridge_grid_description": (
            "unregularised minimum-norm endpoint plus log grid 1e-14,...,1e2"
        ),
        "ridge_selection": (
            "maximize summed validation parity capacity; refit once on "
            "train+validation; evaluate test once"
        ),
        "feature_filter": {
            "fit_on": "training rows only",
            "rule": (
                "retain non-bias columns with training standard deviation "
                f"> {FEATURE_STD_TOL:g}; always retain bias"
            ),
            "threshold": FEATURE_STD_TOL,
            "rationale": (
                "threshold is four orders above the observed exact-propagation "
                "roundoff floor for stationary dephasing features and ten "
                "orders below the weakest active local-loss feature"
            ),
        },
        "fixed_ridge_comparator": FIXED_RIDGE,
        "refit_rows": preset.refit_rows,
        "features_including_bias": preset.n_features,
        "refit_rows_exceed_features": preset.refit_rows > preset.n_features,
        "backend": "exact sparse expm_multiply, no input quantisation",
    }


def parity_reference_protocol(preset: ParityPreset) -> dict:
    """Hash-linked protocol for the two non-active reference rows."""
    active = parity_protocol(preset)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "control": "parity_regularisation_reference_rows",
        "active_protocol_sha256": _sha256_json(active),
        "preset": asdict(preset),
        "seeds": active["seeds"],
        "fresh_seed_namespace": PARITY_SEED_NAMESPACE,
        "methods": list(PARITY_REFERENCE_METHODS),
        "split_design": active["split_design"],
        "common_delay_support": active["common_delay_support"],
        "ridge_grid": list(RIDGES),
        "ridge_grid_description": active["ridge_grid_description"],
        "ridge_selection": active["ridge_selection"],
        "feature_filter": active["feature_filter"],
        "fixed_ridge_comparator": FIXED_RIDGE,
        "refit_rows": preset.refit_rows,
        "features_including_bias": preset.n_features,
        "refit_rows_exceed_features": preset.refit_rows > preset.n_features,
        "reference_boundary": (
            "FN is a reset-unitary model rather than a Lindblad channel; "
            "dephasing is the unital negative control"
        ),
        "backend": (
            "exact unitary propagation for FN; exact sparse expm_multiply for "
            "dephasing; no input quantisation"
        ),
    }


def scaling_protocol(preset: ScalingPreset, schemes: Sequence[str]) -> dict:
    schemes = tuple(schemes)
    if not schemes or any(scheme not in COUPLING_SCHEMES for scheme in schemes):
        raise ValueError(f"schemes must be drawn from {COUPLING_SCHEMES}")
    seeds = fresh_seeds(SCALING_SEED_NAMESPACE, preset.n_seeds)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "control": "normalised_coupling_scaling",
        "preset": asdict(preset),
        "seeds": seeds,
        "fresh_seed_namespace": SCALING_SEED_NAMESPACE,
        "disjoint_from_definitive_seed_pool": not bool(set(seeds) & legacy_seeds()),
        "schemes": list(schemes),
        "normalisation_formulas": {
            "variance": "sqrt(4/(N-1))",
            "kac": "4/(N-1)",
        },
        "anchor": "both multipliers equal one at N=5",
        "coupling_pairing": (
            "leading principal submatrices of one N=8 U[-1,1] draw per seed; "
            "identical base draw for both methods and normalisations"
        ),
        "methods": list(SCALING_METHODS),
        "tasks": ["stm", "narma10"],
        "task_reuse": "STM and NARMA are scored from the same U[0,1] trajectory",
        "ridge_grid": list(RIDGES),
        "ridge_grid_description": (
            "unregularised minimum-norm endpoint plus log grid 1e-14,...,1e2"
        ),
        "fixed_ridge_comparator": FIXED_RIDGE,
        "backend": "exact sparse expm_multiply, no input quantisation",
    }


def _parity_checkpoint_path(outdir: Path, preset: ParityPreset, job: ParityJob) -> Path:
    return outdir / f"{preset.name}__N5_{job.method}_s{job.seed}.json"


def _parity_reference_checkpoint_path(
    outdir: Path, preset: ParityPreset, job: ParityJob
) -> Path:
    return outdir / f"{preset.name}_reference__N5_{job.method}_s{job.seed}.json"


def _scaling_checkpoint_path(
    outdir: Path, preset: ScalingPreset, job: ScalingJob
) -> Path:
    return outdir / (
        f"{preset.name}__{job.scheme}_N{job.n_qubits}_"
        f"{job.method}_s{job.seed}.json"
    )


def _valid_checkpoint(path: Path, protocol_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("protocol_sha256") != protocol_sha256:
        return False
    return payload.get("status") == "complete"


def run_parity_job(
    job: ParityJob,
    preset: ParityPreset,
    outdir_s: str,
    protocol_sha256: str,
) -> dict:
    outdir = Path(outdir_s)
    path = _parity_checkpoint_path(outdir, preset, job)
    if _valid_checkpoint(path, protocol_sha256):
        return {"status": "skip", **asdict(job), "path": str(path)}
    t0 = time.time()
    try:
        couplings, split_rngs = _parity_problem(job.seed)
        reservoir, strength, target = _build_exact_reservoir(
            job.method, couplings, 5, job.seed, preset.h, preset.dt
        )
        observables = readout.pauli_observables(5, max_weight=2)
        x_train, y_train, train_hash = _parity_split(
            reservoir, observables, split_rngs["train"], preset.train, preset
        )
        x_validation, y_validation, validation_hash = _parity_split(
            reservoir,
            observables,
            split_rngs["validation"],
            preset.validation,
            preset,
        )
        x_test, y_test, test_hash = _parity_split(
            reservoir, observables, split_rngs["test"], preset.test, preset
        )
        raw_n_features = x_train.shape[1]
        x_train, x_validation, x_test, feature_filter = (
            train_only_variance_filter(x_train, x_validation, x_test)
        )
        selected, validation_totals, validation_by_delay = select_ridge(
            x_train,
            y_train,
            x_validation,
            y_validation,
            RIDGES,
            metric="capacity",
        )
        selected_total, selected_by_delay = refit_and_test(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            selected,
            metric="capacity",
        )
        fixed_total, fixed_by_delay = refit_and_test(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            FIXED_RIDGE,
            metric="capacity",
        )
        payload = {
            "status": "complete",
            "control": "parity_regularisation",
            "N": 5,
            **asdict(job),
            "h": preset.h,
            "dt": preset.dt,
            "n_virtual": preset.n_virtual,
            "delays": list(preset.delays),
            "n_observables": len(observables),
            "n_features_without_bias": raw_n_features - 1,
            "n_features_including_bias": raw_n_features,
            "fit_features_without_bias": x_train.shape[1] - 1,
            "fit_features_including_bias": x_train.shape[1],
            "feature_filter": feature_filter,
            "train_rows": x_train.shape[0],
            "validation_rows": x_validation.shape[0],
            "test_rows": x_test.shape[0],
            "refit_rows": x_train.shape[0] + x_validation.shape[0],
            "refit_rows_exceed_features": (
                x_train.shape[0] + x_validation.shape[0] > x_train.shape[1]
            ),
            "selected_ridge": selected,
            "validation_capacity_by_ridge": validation_totals,
            "validation_capacity_by_delay_and_ridge": validation_by_delay,
            "selected_test_capacity": selected_total,
            "selected_test_capacity_by_delay": selected_by_delay,
            "fixed_ridge": FIXED_RIDGE,
            "fixed_test_capacity": fixed_total,
            "fixed_test_capacity_by_delay": fixed_by_delay,
            "selected_minus_fixed": selected_total - fixed_total,
            "split_input_sha256": {
                "train": train_hash,
                "validation": validation_hash,
                "test": test_hash,
            },
            "split_hashes_are_distinct": len(
                {train_hash, validation_hash, test_hash}
            ) == 3,
            "coupling_sha256": _array_sha256(couplings),
            "jump_strength": strength,
            "target_jump_strength": target,
            "relative_budget_error": abs(strength - target) / target,
            "backend": "exact_sparse_expm_multiply",
            "protocol_sha256": protocol_sha256,
            "runtime_s": time.time() - t0,
        }
        _atomic_write_json(path, payload)
        return {
            "status": "done",
            **asdict(job),
            "value": selected_total,
            "ridge": selected,
            "runtime_s": payload["runtime_s"],
            "path": str(path),
        }
    except Exception as exc:  # keep the batch alive; do not seal an error checkpoint
        return {
            "status": "error",
            **asdict(job),
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.time() - t0,
            "path": str(path),
        }


def run_parity_reference_job(
    job: ParityJob,
    preset: ParityPreset,
    outdir_s: str,
    protocol_sha256: str,
) -> dict:
    """Score FN and dephasing with the exact corrected parity protocol."""
    if job.method not in PARITY_REFERENCE_METHODS:
        raise ValueError(f"reference method must be in {PARITY_REFERENCE_METHODS}")
    outdir = Path(outdir_s)
    path = _parity_reference_checkpoint_path(outdir, preset, job)
    if _valid_checkpoint(path, protocol_sha256):
        return {"status": "skip", **asdict(job), "path": str(path)}
    t0 = time.time()
    try:
        couplings, split_rngs = _parity_problem(job.seed)
        if job.method == "FN":
            reservoir = res.FujiNakajimaReservoir(
                5, couplings, preset.h, preset.dt
            )
            strength = None
            target = None
            relative_budget_error = None
            backend = "exact_unitary_reset"
        else:
            reservoir, strength, target = _build_exact_reservoir(
                job.method, couplings, 5, job.seed, preset.h, preset.dt
            )
            relative_budget_error = abs(strength - target) / target
            backend = "exact_sparse_expm_multiply"
        observables = readout.pauli_observables(5, max_weight=2)
        x_train, y_train, train_hash = _parity_split(
            reservoir, observables, split_rngs["train"], preset.train, preset
        )
        x_validation, y_validation, validation_hash = _parity_split(
            reservoir,
            observables,
            split_rngs["validation"],
            preset.validation,
            preset,
        )
        x_test, y_test, test_hash = _parity_split(
            reservoir, observables, split_rngs["test"], preset.test, preset
        )
        raw_n_features = x_train.shape[1]
        x_train, x_validation, x_test, feature_filter = (
            train_only_variance_filter(x_train, x_validation, x_test)
        )
        selected, validation_totals, validation_by_delay = select_ridge(
            x_train,
            y_train,
            x_validation,
            y_validation,
            RIDGES,
            metric="capacity",
        )
        selected_total, selected_by_delay = refit_and_test(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            selected,
            metric="capacity",
        )
        fixed_total, fixed_by_delay = refit_and_test(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            FIXED_RIDGE,
            metric="capacity",
        )
        payload = {
            "status": "complete",
            "control": "parity_regularisation_reference_rows",
            "N": 5,
            **asdict(job),
            "h": preset.h,
            "dt": preset.dt,
            "n_virtual": preset.n_virtual,
            "delays": list(preset.delays),
            "n_observables": len(observables),
            "n_features_without_bias": raw_n_features - 1,
            "n_features_including_bias": raw_n_features,
            "fit_features_without_bias": x_train.shape[1] - 1,
            "fit_features_including_bias": x_train.shape[1],
            "feature_filter": feature_filter,
            "train_rows": x_train.shape[0],
            "validation_rows": x_validation.shape[0],
            "test_rows": x_test.shape[0],
            "refit_rows": x_train.shape[0] + x_validation.shape[0],
            "refit_rows_exceed_features": (
                x_train.shape[0] + x_validation.shape[0] > x_train.shape[1]
            ),
            "selected_ridge": selected,
            "validation_capacity_by_ridge": validation_totals,
            "validation_capacity_by_delay_and_ridge": validation_by_delay,
            "selected_test_capacity": selected_total,
            "selected_test_capacity_by_delay": selected_by_delay,
            "fixed_ridge": FIXED_RIDGE,
            "fixed_test_capacity": fixed_total,
            "fixed_test_capacity_by_delay": fixed_by_delay,
            "selected_minus_fixed": selected_total - fixed_total,
            "split_input_sha256": {
                "train": train_hash,
                "validation": validation_hash,
                "test": test_hash,
            },
            "split_hashes_are_distinct": len(
                {train_hash, validation_hash, test_hash}
            ) == 3,
            "coupling_sha256": _array_sha256(couplings),
            "jump_strength": strength,
            "target_jump_strength": target,
            "relative_budget_error": relative_budget_error,
            "backend": backend,
            "protocol_sha256": protocol_sha256,
            "runtime_s": time.time() - t0,
        }
        _atomic_write_json(path, payload)
        return {
            "status": "done",
            **asdict(job),
            "value": selected_total,
            "ridge": selected,
            "runtime_s": payload["runtime_s"],
            "path": str(path),
        }
    except Exception as exc:
        return {
            "status": "error",
            **asdict(job),
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.time() - t0,
            "path": str(path),
        }


def _scaling_inputs(seed: int, total_len: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x51A11]))
    return tasks.stm_inputs(total_len, rng)


def _scaling_split_indices(preset: ScalingPreset) -> dict[str, np.ndarray]:
    max_delay = max(max(preset.delays), 10)
    n_post = preset.train + preset.validation + preset.test
    indices = np.arange(n_post)
    return {
        "train": indices[(indices >= max_delay) & (indices < preset.train)],
        "validation": indices[
            (indices >= preset.train)
            & (indices < preset.train + preset.validation)
        ],
        "test": indices[
            (indices >= preset.train + preset.validation)
            & (indices < n_post)
        ],
    }


def _held_out_score(
    x: np.ndarray,
    y: np.ndarray,
    indices: dict[str, np.ndarray],
    metric: str,
) -> dict:
    xb = readout.add_bias(x)
    selected, validation_totals, validation_by_target = select_ridge(
        xb[indices["train"]],
        y[indices["train"]],
        xb[indices["validation"]],
        y[indices["validation"]],
        RIDGES,
        metric=metric,
    )
    selected_total, selected_by_target = refit_and_test(
        xb[indices["train"]],
        y[indices["train"]],
        xb[indices["validation"]],
        y[indices["validation"]],
        xb[indices["test"]],
        y[indices["test"]],
        selected,
        metric=metric,
    )
    fixed_total, fixed_by_target = refit_and_test(
        xb[indices["train"]],
        y[indices["train"]],
        xb[indices["validation"]],
        y[indices["validation"]],
        xb[indices["test"]],
        y[indices["test"]],
        FIXED_RIDGE,
        metric=metric,
    )
    return {
        "selected_ridge": selected,
        "validation_by_ridge": validation_totals,
        "validation_by_target_and_ridge": validation_by_target,
        "selected_test": selected_total,
        "selected_test_by_target": selected_by_target,
        "fixed_ridge": FIXED_RIDGE,
        "fixed_test": fixed_total,
        "fixed_test_by_target": fixed_by_target,
    }


def run_scaling_job(
    job: ScalingJob,
    preset: ScalingPreset,
    outdir_s: str,
    protocol_sha256: str,
) -> dict:
    outdir = Path(outdir_s)
    path = _scaling_checkpoint_path(outdir, preset, job)
    if _valid_checkpoint(path, protocol_sha256):
        return {"status": "skip", **asdict(job), "path": str(path)}
    t0 = time.time()
    try:
        couplings, coupling_meta = scaled_couplings(
            job.seed, job.n_qubits, job.scheme
        )
        reservoir, strength, target = _build_exact_reservoir(
            job.method,
            couplings,
            job.n_qubits,
            job.seed,
            preset.h,
            preset.dt,
        )
        observables = readout.pauli_observables(job.n_qubits, max_weight=2)
        inputs = _scaling_inputs(job.seed, preset.total_len)
        x = reservoir.run(inputs, observables, washout=preset.wash)
        post = inputs[preset.wash:]
        indices = _scaling_split_indices(preset)
        stm_targets = np.column_stack(
            [tasks.delayed_target(post, delay) for delay in preset.delays]
        )
        narma_targets = tasks.narma_target(
            post, order=10, input_scale=0.2
        )[:, None]
        if not np.all(np.isfinite(stm_targets[indices["train"]])):
            raise RuntimeError("undefined STM training target survived split")
        if not np.all(np.isfinite(narma_targets[indices["train"]])):
            raise RuntimeError("undefined NARMA training target survived split")
        stm = _held_out_score(x, stm_targets, indices, metric="capacity")
        narma = _held_out_score(x, narma_targets, indices, metric="nmse")
        payload = {
            "status": "complete",
            "control": "normalised_coupling_scaling",
            **asdict(job),
            "h": preset.h,
            "dt": preset.dt,
            "wash": preset.wash,
            "train": preset.train,
            "validation": preset.validation,
            "test": preset.test,
            "delays": list(preset.delays),
            "n_observables": len(observables),
            "n_features_including_bias": x.shape[1] + 1,
            "usable_rows": {
                name: len(rows) for name, rows in indices.items()
            },
            "input_sha256": _array_sha256(inputs),
            **coupling_meta,
            "jump_strength": strength,
            "target_jump_strength": target,
            "relative_budget_error": abs(strength - target) / target,
            "stm": stm,
            "narma10": narma,
            "backend": "exact_sparse_expm_multiply",
            "protocol_sha256": protocol_sha256,
            "runtime_s": time.time() - t0,
        }
        _atomic_write_json(path, payload)
        return {
            "status": "done",
            **asdict(job),
            "value": stm["selected_test"],
            "runtime_s": payload["runtime_s"],
            "path": str(path),
        }
    except Exception as exc:
        return {
            "status": "error",
            **asdict(job),
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.time() - t0,
            "path": str(path),
        }


def _run_parallel(function, jobs: Sequence, args: tuple, workers: int) -> None:
    if workers <= 1:
        for job in jobs:
            result = function(job, *args)
            print(_progress(result), flush=True)
        return

    def consume(pool) -> None:
        futures = {pool.submit(function, job, *args): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "status": "error",
                    **asdict(job),
                    "error": f"worker died: {type(exc).__name__}: {exc}",
                }
            print(_progress(result), flush=True)

    # Some managed environments disallow the semaphore sysconf queried by
    # ProcessPoolExecutor.  Threads are a bounded, checkpoint-safe fallback; the
    # BLAS thread caps above prevent multiplicative oversubscription.
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool)
    except (PermissionError, OSError) as exc:
        print(
            f"process pool unavailable ({type(exc).__name__}: {exc}); "
            f"falling back to {workers} threads",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            consume(pool)


def _progress(result: dict) -> str:
    identity = " ".join(
        f"{key}={result[key]}"
        for key in ("scheme", "n_qubits", "method", "seed")
        if key in result
    )
    runtime = (
        "" if result.get("runtime_s") is None
        else f" {float(result['runtime_s']):.1f}s"
    )
    value = (
        "" if result.get("value") is None
        else f" value={float(result['value']):.5g}"
    )
    error = "" if "error" not in result else f" ! {result['error']}"
    return f"{result['status']:5s} {identity}{value}{runtime}{error}"


def _paired_bootstrap(values: Sequence[float], seed: int) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired bootstrap needs at least two finite differences")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "se": float(values.std(ddof=1) / np.sqrt(len(values))),
        "ci95_percentile": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "positive_pairs": int(np.sum(values > 0)),
        "negative_pairs": int(np.sum(values < 0)),
        "zero_pairs": int(np.sum(values == 0)),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def ridge_boundary_audit(rows: Sequence[dict]) -> dict:
    """Audit whether validation supports the upper edge of the ridge grid.

    Zero is an intentional endpoint (the minimum-norm unregularised fit), so a
    zero selection is reported but is not an unbracketed positive-ridge optimum.
    A selection at the largest ridge is unresolved only when validation still
    improves from the preceding grid point.
    """
    maximum = max(RIDGES)
    previous = sorted(RIDGES)[-2]
    selected_zero = []
    selected_maximum = []
    unresolved_upper = []
    for row in rows:
        identity = {"method": row["method"], "seed": row["seed"]}
        selected = float(row["selected_ridge"])
        scores = row["validation_capacity_by_ridge"]
        if selected == 0.0:
            selected_zero.append(identity)
        if selected == maximum:
            selected_maximum.append(identity)
            maximum_score = float(scores[f"{maximum:.12g}"])
            previous_score = float(scores[f"{previous:.12g}"])
            if maximum_score > previous_score + 1e-12:
                unresolved_upper.append(
                    {
                        **identity,
                        "previous_ridge": previous,
                        "previous_score": previous_score,
                        "maximum_ridge": maximum,
                        "maximum_score": maximum_score,
                    }
                )
    return {
        "grid_minimum": min(RIDGES),
        "grid_maximum": maximum,
        "zero_is_explicit_unregularised_endpoint": True,
        "n_selected_zero": len(selected_zero),
        "selected_zero": selected_zero,
        "n_selected_maximum": len(selected_maximum),
        "selected_maximum": selected_maximum,
        "n_unresolved_upper": len(unresolved_upper),
        "unresolved_upper": unresolved_upper,
        "upper_boundary_is_bracketed": not unresolved_upper,
    }


def scaling_ridge_boundary_audit(rows: Sequence[dict]) -> dict:
    """Audit upper-grid selections for both scaling metrics."""
    maximum = max(RIDGES)
    previous = sorted(RIDGES)[-2]
    selected_zero = []
    selected_maximum = []
    unresolved_upper = []
    for row in rows:
        for task_name, direction in (("stm", "maximize"), ("narma10", "minimize")):
            result = row[task_name]
            identity = {
                "scheme": row["scheme"],
                "N": row["n_qubits"],
                "method": row["method"],
                "seed": row["seed"],
                "task": task_name,
            }
            selected = float(result["selected_ridge"])
            if selected == 0.0:
                selected_zero.append(identity)
            if selected != maximum:
                continue
            selected_maximum.append(identity)
            scores = result["validation_by_ridge"]
            maximum_score = float(scores[f"{maximum:.12g}"])
            previous_score = float(scores[f"{previous:.12g}"])
            still_improving = (
                maximum_score > previous_score + 1e-12
                if direction == "maximize"
                else maximum_score < previous_score - 1e-12
            )
            if still_improving:
                unresolved_upper.append(
                    {
                        **identity,
                        "direction": direction,
                        "previous_ridge": previous,
                        "previous_score": previous_score,
                        "maximum_ridge": maximum,
                        "maximum_score": maximum_score,
                    }
                )
    return {
        "grid_minimum": min(RIDGES),
        "grid_maximum": maximum,
        "zero_is_explicit_unregularised_endpoint": True,
        "n_selected_zero": len(selected_zero),
        "selected_zero": selected_zero,
        "n_selected_maximum": len(selected_maximum),
        "selected_maximum": selected_maximum,
        "n_unresolved_upper": len(unresolved_upper),
        "unresolved_upper": unresolved_upper,
        "upper_boundary_is_bracketed": not unresolved_upper,
    }


def feature_filter_audit(rows: Sequence[dict]) -> dict:
    """Summarise the observed signal/roundoff separation across raw rows."""
    retained_candidates = []
    dropped_candidates = []
    rows_with_drops = 0
    dropped_by_method: dict[str, int] = {}
    retained_counts = []
    for row in rows:
        metadata = row.get("feature_filter")
        if not metadata:
            continue
        identity = {"method": row["method"], "seed": row["seed"]}
        retained_counts.append(metadata["retained_nonbias_features"])
        if metadata["dropped_nonbias_features"]:
            rows_with_drops += 1
            dropped_by_method[row["method"]] = (
                dropped_by_method.get(row["method"], 0)
                + metadata["dropped_nonbias_features"]
            )
        if metadata["training_std_min_retained"] is not None:
            retained_candidates.append(
                (
                    float(metadata["training_std_min_retained"]),
                    identity,
                )
            )
        if metadata["training_std_max_dropped"] is not None:
            dropped_candidates.append(
                (
                    float(metadata["training_std_max_dropped"]),
                    identity,
                )
            )
    minimum_retained = min(retained_candidates, default=None, key=lambda x: x[0])
    maximum_dropped = max(dropped_candidates, default=None, key=lambda x: x[0])
    return {
        "threshold": FEATURE_STD_TOL,
        "n_rows": len(rows),
        "rows_with_dropped_features": rows_with_drops,
        "dropped_feature_count_by_method": dropped_by_method,
        "minimum_retained_nonbias_features": (
            None if not retained_counts else int(min(retained_counts))
        ),
        "maximum_retained_nonbias_features": (
            None if not retained_counts else int(max(retained_counts))
        ),
        "global_minimum_retained_training_std": (
            None if minimum_retained is None else minimum_retained[0]
        ),
        "global_minimum_retained_identity": (
            None if minimum_retained is None else minimum_retained[1]
        ),
        "global_maximum_dropped_training_std": (
            None if maximum_dropped is None else maximum_dropped[0]
        ),
        "global_maximum_dropped_identity": (
            None if maximum_dropped is None else maximum_dropped[1]
        ),
    }


def aggregate_parity(preset: ParityPreset, outdir: Path = PARITY_DIR) -> dict:
    protocol = parity_protocol(preset)
    protocol_hash = _sha256_json(protocol)
    rows = []
    missing = []
    for method in PARITY_METHODS:
        for seed in protocol["seeds"]:
            path = _parity_checkpoint_path(outdir, preset, ParityJob(method, seed))
            if not _valid_checkpoint(path, protocol_hash):
                missing.append(path.name)
                continue
            rows.append(json.loads(path.read_text()))
    by_method = {}
    local = {
        row["seed"]: row for row in rows if row["method"] == "CD_paper"
    }
    for index, method in enumerate(PARITY_METHODS):
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["seed"],
        )
        selected = np.asarray(
            [row["selected_test_capacity"] for row in method_rows], dtype=float
        )
        fixed = np.asarray(
            [row["fixed_test_capacity"] for row in method_rows], dtype=float
        )
        selected_ridges = [
            float(row["selected_ridge"]) for row in method_rows
        ]
        paired_local = [
            row["selected_test_capacity"]
            - local[row["seed"]]["selected_test_capacity"]
            for row in method_rows
            if row["seed"] in local
        ]
        summary = {
            "n_complete": len(method_rows),
            "selected_test_mean": (
                None if not len(selected) else float(selected.mean())
            ),
            "selected_test_se": (
                None if len(selected) < 2
                else float(selected.std(ddof=1) / np.sqrt(len(selected)))
            ),
            "fixed_test_mean": None if not len(fixed) else float(fixed.mean()),
            "selected_ridges": selected_ridges,
            "selected_ridge_counts": {
                f"{ridge:.12g}": selected_ridges.count(ridge)
                for ridge in RIDGES
            },
        }
        if len(method_rows) >= 2:
            summary["selected_minus_fixed_paired"] = _paired_bootstrap(
                selected - fixed, 10_000 + index
            )
        if len(paired_local) >= 2:
            summary["selected_minus_local_paired"] = _paired_bootstrap(
                paired_local, 20_000 + index
            )
        by_method[method] = summary

    ranking = sorted(
        (
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "mean": summary["selected_test_mean"],
                "n": summary["n_complete"],
            }
            for method, summary in by_method.items()
            if summary["selected_test_mean"] is not None
        ),
        key=lambda item: item["mean"],
        reverse=True,
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    rows_by_method_seed = {
        (row["method"], row["seed"]): row for row in rows
    }
    winner_counts = {method: 0 for method in PARITY_METHODS}
    complete_seed_ranks = 0
    for seed in protocol["seeds"]:
        candidates = [
            rows_by_method_seed.get((method, seed)) for method in PARITY_METHODS
        ]
        if all(candidate is not None for candidate in candidates):
            complete_seed_ranks += 1
            winner = max(
                candidates, key=lambda row: row["selected_test_capacity"]
            )
            winner_counts[winner["method"]] += 1

    pairwise = {}
    pair_index = 0
    for left_index, left in enumerate(PARITY_METHODS):
        for right in PARITY_METHODS[left_index + 1:]:
            differences = [
                rows_by_method_seed[(left, seed)]["selected_test_capacity"]
                - rows_by_method_seed[(right, seed)]["selected_test_capacity"]
                for seed in protocol["seeds"]
                if (
                    (left, seed) in rows_by_method_seed
                    and (right, seed) in rows_by_method_seed
                )
            ]
            key = f"{left}_minus_{right}"
            pairwise[key] = (
                None
                if len(differences) < 2
                else _paired_bootstrap(differences, 70_000 + pair_index)
            )
            pair_index += 1

    positive_specialists = []
    for method in PARITY_METHODS:
        if method == "CD_paper":
            continue
        comparison = by_method[method].get("selected_minus_local_paired")
        if comparison and comparison["ci95_percentile"][0] > 0:
            positive_specialists.append(method)

    payload = {
        "control": "parity_regularisation",
        "status": "complete" if not missing else "partial",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "expected_checkpoints": len(PARITY_METHODS) * preset.n_seeds,
        "complete_checkpoints": len(rows),
        "missing_checkpoints": missing,
        "ridge_boundary_audit": ridge_boundary_audit(rows),
        "feature_filter_audit": feature_filter_audit(rows),
        "summary_by_method": by_method,
        "channel_ranking": ranking,
        "per_seed_winner_counts": {
            "n_complete_seed_rankings": complete_seed_ranks,
            "counts": winner_counts,
        },
        "pairwise_selected_capacity_differences": pairwise,
        "positive_parity_specialists_vs_local": positive_specialists,
        "raw_rows": rows,
    }
    if not missing and not payload["ridge_boundary_audit"][
        "upper_boundary_is_bracketed"
    ]:
        payload["status"] = "unresolved_upper_ridge_boundary"
    _atomic_write_json(outdir / f"{preset.name}_aggregate.json", payload)
    return payload


def aggregate_parity_references(
    preset: ParityPreset, outdir: Path = PARITY_DIR
) -> dict:
    protocol = parity_reference_protocol(preset)
    protocol_hash = _sha256_json(protocol)
    active_protocol_hash = protocol["active_protocol_sha256"]
    rows = []
    missing = []
    for method in PARITY_REFERENCE_METHODS:
        for seed in protocol["seeds"]:
            job = ParityJob(method, seed)
            path = _parity_reference_checkpoint_path(outdir, preset, job)
            if not _valid_checkpoint(path, protocol_hash):
                missing.append(path.name)
                continue
            rows.append(json.loads(path.read_text()))
    local = {}
    for seed in protocol["seeds"]:
        path = _parity_checkpoint_path(
            outdir, preset, ParityJob("CD_paper", seed)
        )
        if _valid_checkpoint(path, active_protocol_hash):
            local[seed] = json.loads(path.read_text())
    summary_by_method = {}
    for index, method in enumerate(PARITY_REFERENCE_METHODS):
        method_rows = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["seed"],
        )
        selected = np.asarray(
            [row["selected_test_capacity"] for row in method_rows]
        )
        fixed = np.asarray([row["fixed_test_capacity"] for row in method_rows])
        summary = {
            "n_complete": len(method_rows),
            "selected_test_mean": (
                None if not len(selected) else float(selected.mean())
            ),
            "selected_test_se": (
                None if len(selected) < 2
                else float(selected.std(ddof=1) / np.sqrt(len(selected)))
            ),
            "fixed_test_mean": None if not len(fixed) else float(fixed.mean()),
            "selected_ridge_counts": {
                f"{ridge:.12g}": sum(
                    float(row["selected_ridge"]) == ridge for row in method_rows
                )
                for ridge in RIDGES
            },
        }
        if len(selected) >= 2:
            summary["selected_minus_fixed_paired"] = _paired_bootstrap(
                selected - fixed, 80_000 + index
            )
        differences = [
            row["selected_test_capacity"]
            - local[row["seed"]]["selected_test_capacity"]
            for row in method_rows
            if row["seed"] in local
        ]
        if len(differences) >= 2:
            summary["selected_minus_local_paired"] = _paired_bootstrap(
                differences, 81_000 + index
            )
        summary_by_method[method] = summary
    boundary = ridge_boundary_audit(rows)
    payload = {
        "control": "parity_regularisation_reference_rows",
        "status": "complete" if not missing else "partial",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "active_protocol_sha256": active_protocol_hash,
        "expected_checkpoints": len(PARITY_REFERENCE_METHODS) * preset.n_seeds,
        "complete_checkpoints": len(rows),
        "missing_checkpoints": missing,
        "ridge_boundary_audit": boundary,
        "feature_filter_audit": feature_filter_audit(rows),
        "summary_by_method": summary_by_method,
        "raw_rows": rows,
    }
    if not missing and not boundary["upper_boundary_is_bracketed"]:
        payload["status"] = "unresolved_upper_ridge_boundary"
    _atomic_write_json(
        outdir / f"{preset.name}_reference_aggregate.json", payload
    )
    if not missing and not boundary["upper_boundary_is_bracketed"]:
        raise RuntimeError(
            "reference parity ridge selection is unbracketed at the upper edge"
        )
    return payload


def _scaling_row_index(rows: Iterable[dict]) -> dict[tuple, dict]:
    return {
        (row["scheme"], row["n_qubits"], row["method"], row["seed"]): row
        for row in rows
    }


def scaling_invariant_audit(
    rows: Sequence[dict],
    protocol: dict,
    preset: ScalingPreset,
    schemes: Sequence[str],
) -> dict:
    """Machine-readable paired-design and physics audit for scaling rows."""
    indexed = _scaling_row_index(rows)
    duplicate_free = len(indexed) == len(rows)
    pairing_violations = []
    multiplier_violations = []
    budget_violations = []
    backend_violations = []
    max_budget_error = 0.0
    for row in rows:
        expected_multiplier = coupling_multiplier(
            row["n_qubits"], row["scheme"]
        )
        if not np.isclose(
            row["coupling_multiplier"],
            expected_multiplier,
            rtol=0.0,
            atol=1e-15,
        ):
            multiplier_violations.append(
                {
                    "scheme": row["scheme"],
                    "N": row["n_qubits"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "actual": row["coupling_multiplier"],
                    "expected": expected_multiplier,
                }
            )
        error = float(row["relative_budget_error"])
        max_budget_error = max(max_budget_error, error)
        if error > 1e-10:
            budget_violations.append(
                {
                    "scheme": row["scheme"],
                    "N": row["n_qubits"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "relative_budget_error": error,
                }
            )
        if row["backend"] != "exact_sparse_expm_multiply":
            backend_violations.append(
                {
                    "scheme": row["scheme"],
                    "N": row["n_qubits"],
                    "method": row["method"],
                    "seed": row["seed"],
                    "backend": row["backend"],
                }
            )
    for scheme in schemes:
        for n_qubits in preset.n_values:
            for seed in protocol["seeds"]:
                key_local = (scheme, n_qubits, "CD_paper", seed)
                key_collective = (
                    scheme,
                    n_qubits,
                    "B3_collective",
                    seed,
                )
                if key_local not in indexed or key_collective not in indexed:
                    continue
                local = indexed[key_local]
                collective = indexed[key_collective]
                paired_fields = (
                    "input_sha256",
                    "base_coupling_sha256",
                    "scaled_coupling_sha256",
                )
                unequal = [
                    field
                    for field in paired_fields
                    if local[field] != collective[field]
                ]
                if unequal:
                    pairing_violations.append(
                        {
                            "scheme": scheme,
                            "N": n_qubits,
                            "seed": seed,
                            "unequal_fields": unequal,
                        }
                    )
    anchor_violations = [
        {
            "scheme": row["scheme"],
            "method": row["method"],
            "seed": row["seed"],
            "multiplier": row["coupling_multiplier"],
        }
        for row in rows
        if row["n_qubits"] == 5
        and not np.isclose(
            row["coupling_multiplier"], 1.0, rtol=0.0, atol=1e-15
        )
    ]
    production = preset.name == "paper"
    checks = {
        "protocol_variance_only": (
            not production or list(schemes) == ["variance"]
        ),
        "n_values_are_4_through_8": (
            not production or tuple(preset.n_values) == (4, 5, 6, 7, 8)
        ),
        "seed_count_is_8": not production or preset.n_seeds == 8,
        "fresh_seeds_disjoint": protocol["disjoint_from_definitive_seed_pool"],
        "both_methods_declared": tuple(protocol["methods"]) == SCALING_METHODS,
        "row_identities_unique": duplicate_free,
        "paired_hashes_equal": not pairing_violations,
        "multipliers_match_formula": not multiplier_violations,
        "n5_anchor_exact": not anchor_violations,
        "jump_budget_within_1e-10": not budget_violations,
        "exact_backend_only": not backend_violations,
    }
    return {
        "production_contract_applies": production,
        "checks": checks,
        "all_passed": all(checks.values()),
        "max_relative_jump_budget_error": max_budget_error,
        "pairing_violations": pairing_violations,
        "multiplier_violations": multiplier_violations,
        "anchor_violations": anchor_violations,
        "budget_violations": budget_violations,
        "backend_violations": backend_violations,
    }


def aggregate_scaling(
    preset: ScalingPreset,
    schemes: Sequence[str],
    outdir: Path = SCALING_DIR,
) -> dict:
    protocol = scaling_protocol(preset, schemes)
    protocol_hash = _sha256_json(protocol)
    rows = []
    missing = []
    for scheme in schemes:
        for n_qubits in preset.n_values:
            for method in SCALING_METHODS:
                for seed in protocol["seeds"]:
                    job = ScalingJob(scheme, n_qubits, method, seed)
                    path = _scaling_checkpoint_path(outdir, preset, job)
                    if not _valid_checkpoint(path, protocol_hash):
                        missing.append(path.name)
                        continue
                    rows.append(json.loads(path.read_text()))
    indexed = _scaling_row_index(rows)
    summaries = {}
    for scheme_index, scheme in enumerate(schemes):
        scheme_summary = {}
        per_seed_relative: dict[int, dict[int, float]] = {}
        for n_qubits in preset.n_values:
            paired = [
                (
                    indexed[(scheme, n_qubits, "CD_paper", seed)],
                    indexed[(scheme, n_qubits, "B3_collective", seed)],
                )
                for seed in protocol["seeds"]
                if (
                    (scheme, n_qubits, "CD_paper", seed) in indexed
                    and (scheme, n_qubits, "B3_collective", seed) in indexed
                )
            ]
            n_summary = {"n_pairs": len(paired)}
            if len(paired) >= 2:
                local_stm = np.asarray(
                    [pair[0]["stm"]["selected_test"] for pair in paired]
                )
                collective_stm = np.asarray(
                    [pair[1]["stm"]["selected_test"] for pair in paired]
                )
                relative_stm = (collective_stm - local_stm) / local_stm
                local_narma = np.asarray(
                    [pair[0]["narma10"]["selected_test"] for pair in paired]
                )
                collective_narma = np.asarray(
                    [pair[1]["narma10"]["selected_test"] for pair in paired]
                )
                relative_narma = (local_narma - collective_narma) / local_narma
                n_summary.update(
                    {
                        "local_stm_mean": float(local_stm.mean()),
                        "collective_stm_mean": float(collective_stm.mean()),
                        "stm_collective_minus_local": _paired_bootstrap(
                            collective_stm - local_stm,
                            30_000 + 100 * scheme_index + n_qubits,
                        ),
                        "stm_relative_advantage": _paired_bootstrap(
                            relative_stm,
                            40_000 + 100 * scheme_index + n_qubits,
                        ),
                        "local_narma_nmse_mean": float(local_narma.mean()),
                        "collective_narma_nmse_mean": float(
                            collective_narma.mean()
                        ),
                        "narma_relative_improvement": _paired_bootstrap(
                            relative_narma,
                            50_000 + 100 * scheme_index + n_qubits,
                        ),
                    }
                )
                for (local_row, _), relative in zip(paired, relative_stm):
                    per_seed_relative.setdefault(local_row["seed"], {})[
                        n_qubits
                    ] = float(relative)
            scheme_summary[str(n_qubits)] = n_summary
        slopes = []
        for seed, by_n in per_seed_relative.items():
            if all(n_qubits in by_n for n_qubits in preset.n_values):
                x = np.asarray(preset.n_values, dtype=float)
                y = np.asarray([by_n[n_qubits] for n_qubits in preset.n_values])
                slopes.append(float(np.polyfit(x, y, deg=1)[0]))
        if len(slopes) >= 2:
            scheme_summary["relative_advantage_slope_per_qubit"] = (
                _paired_bootstrap(
                    slopes, 60_000 + scheme_index
                )
            )
        else:
            scheme_summary["relative_advantage_slope_per_qubit"] = {
                "n": len(slopes),
                "status": "insufficient complete seed curves",
            }
        summaries[scheme] = scheme_summary
    payload = {
        "control": "normalised_coupling_scaling",
        "status": "complete" if not missing else "partial",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "expected_checkpoints": (
            len(schemes)
            * len(preset.n_values)
            * len(SCALING_METHODS)
            * preset.n_seeds
        ),
        "complete_checkpoints": len(rows),
        "missing_checkpoints": missing,
        "ridge_boundary_audit": scaling_ridge_boundary_audit(rows),
        "invariant_audit": scaling_invariant_audit(
            rows, protocol, preset, schemes
        ),
        "summary_by_scheme": summaries,
        "raw_rows": rows,
    }
    unresolved = payload["ridge_boundary_audit"]["n_unresolved_upper"]
    invariant_failure = not payload["invariant_audit"]["all_passed"]
    if not missing and unresolved:
        payload["status"] = "unresolved_upper_ridge_boundary"
    if not missing and invariant_failure:
        payload["status"] = "failed_invariant_audit"
    suffix = "-".join(schemes)
    _atomic_write_json(outdir / f"{preset.name}_{suffix}_aggregate.json", payload)
    if not missing and unresolved:
        raise RuntimeError(
            f"{unresolved} scaling ridge selections still improve at the "
            f"upper grid edge {max(RIDGES):g}; extend RIDGES and rerun before "
            "using this aggregate"
        )
    if not missing and invariant_failure:
        failed = [
            name
            for name, passed in payload["invariant_audit"]["checks"].items()
            if not passed
        ]
        raise RuntimeError(
            f"complete scaling aggregate failed invariants: {failed}"
        )
    return payload


def write_report(
    parity: dict,
    scaling: dict,
    references: dict | None = None,
    path: Path = REPORT_PATH,
) -> None:
    lines = [
        "# Focused revision controls",
        "",
        f"Protocol: `{PROTOCOL_VERSION}`. This report is generated from atomic raw "
        "checkpoints; it does not replace the canonical paper protocol.",
        "",
        "## 1. Parity regularisation control",
        "",
        f"Completion: **{parity['complete_checkpoints']}/"
        f"{parity['expected_checkpoints']}** checkpoints "
        f"({parity['status']}). The refit uses "
        f"{parity['protocol']['refit_rows']} rows for "
        f"{parity['protocol']['features_including_bias']} fitted coefficients.",
        "",
        "Ridge-boundary audit: "
        f"zero selected in {parity['ridge_boundary_audit']['n_selected_zero']} "
        "rows (legitimate unregularised endpoint); maximum ridge selected in "
        f"{parity['ridge_boundary_audit']['n_selected_maximum']} rows; "
        f"unresolved rising upper edges = "
        f"{parity['ridge_boundary_audit']['n_unresolved_upper']}.",
        "",
        "Train-only feature filter: "
        f"threshold={parity['feature_filter_audit']['threshold']:.1e}; "
        f"global minimum retained active-feature std="
        f"{_fmt_sci(parity['feature_filter_audit']['global_minimum_retained_training_std'])}; "
        f"global maximum dropped std="
        f"{_fmt_sci(parity['feature_filter_audit']['global_maximum_dropped_training_std'])}.",
        "",
        "| rank | channel | n | selected capacity | fixed-1e-8 capacity | "
        "selected - fixed, 95% CI | selected - local, 95% CI |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    rank_by_method = {
        item["method"]: item["rank"] for item in parity["channel_ranking"]
    }
    for method in PARITY_METHODS:
        summary = parity["summary_by_method"][method]
        sf = summary.get("selected_minus_fixed_paired")
        sl = summary.get("selected_minus_local_paired")
        lines.append(
            "| {rank} | {label} | {n} | {selected} | {fixed} | {sf} | {sl} |".format(
                rank=rank_by_method.get(method, "—"),
                label=METHOD_LABELS[method],
                n=summary["n_complete"],
                selected=_fmt(summary["selected_test_mean"]),
                fixed=_fmt(summary["fixed_test_mean"]),
                sf=_fmt_ci(sf),
                sl=_fmt_ci(sl),
            )
        )
    lines.extend(
        [
            "",
            "Per-seed winner counts across complete six-channel rankings: "
            + ", ".join(
                f"{METHOD_LABELS[method]}="
                f"{parity['per_seed_winner_counts']['counts'][method]}"
                for method in PARITY_METHODS
            )
            + f" (n={parity['per_seed_winner_counts']['n_complete_seed_rankings']}).",
            "",
            "Channels with a positive paired lower 95% bound versus uniform "
            "local loss: "
            + (
                ", ".join(
                    METHOD_LABELS[method]
                    for method in parity[
                        "positive_parity_specialists_vs_local"
                    ]
                )
                or "none"
            )
            + ".",
            "",
            "Interpretation rule: use the selected-ridge column for channel "
            "comparisons. The selected-minus-fixed interval tests whether the "
            "historical ridge materially affected each paired seed.",
            "",
        ]
    )
    if references is not None:
        lines.extend(
            [
                "### Corrected parity reference rows",
                "",
                f"Completion: **{references['complete_checkpoints']}/"
                f"{references['expected_checkpoints']}**. Global maximum "
                "dropped training-feature std="
                f"{_fmt_sci(references['feature_filter_audit']['global_maximum_dropped_training_std'])}.",
                "",
                "| reference | n | selected capacity | SE | selected - local, 95% CI |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method in PARITY_REFERENCE_METHODS:
            summary = references["summary_by_method"][method]
            lines.append(
                "| {label} | {n} | {mean} | {se} | {difference} |".format(
                    label=METHOD_LABELS[method],
                    n=summary["n_complete"],
                    mean=_fmt(summary["selected_test_mean"]),
                    se=_fmt(summary["selected_test_se"]),
                    difference=_fmt_ci(
                        summary.get("selected_minus_local_paired")
                    ),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## 2. Normalised-coupling scaling",
            "",
            f"Completion: **{scaling['complete_checkpoints']}/"
            f"{scaling['expected_checkpoints']}** checkpoints "
            f"({scaling['status']}). Positive STM percentages favour collective "
            "loss; positive NARMA percentages mean lower collective NMSE.",
            "",
            "Scaling ridge-boundary audit: "
            f"zero selected in "
            f"{scaling['ridge_boundary_audit']['n_selected_zero']} fits; "
            f"maximum ridge selected in "
            f"{scaling['ridge_boundary_audit']['n_selected_maximum']} fits; "
            f"unresolved rising upper edges = "
            f"{scaling['ridge_boundary_audit']['n_unresolved_upper']}.",
            "",
            "Scaling invariant audit: "
            f"all passed = {scaling['invariant_audit']['all_passed']}; "
            "maximum relative jump-budget error = "
            f"{scaling['invariant_audit']['max_relative_jump_budget_error']:.3e}.",
            "",
        ]
    )
    for scheme in scaling["protocol"]["schemes"]:
        lines.extend(
            [
                f"### {scheme} normalisation",
                "",
                "| N | pairs | local STM | collective STM | collective - local, "
                "95% CI | relative STM advantage, 95% CI | relative NARMA "
                "improvement, 95% CI |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        summary = scaling["summary_by_scheme"][scheme]
        for n_qubits in scaling["protocol"]["preset"]["n_values"]:
            row = summary[str(n_qubits)]
            lines.append(
                "| {n} | {pairs} | {local} | {collective} | {gap} | {stm} | {narma} |".format(
                    n=n_qubits,
                    pairs=row["n_pairs"],
                    local=_fmt(row.get("local_stm_mean")),
                    collective=_fmt(row.get("collective_stm_mean")),
                    gap=_fmt_ci(row.get("stm_collective_minus_local")),
                    stm=_fmt_ci(row.get("stm_relative_advantage"), percent=True),
                    narma=_fmt_ci(
                        row.get("narma_relative_improvement"), percent=True
                    ),
                )
            )
        lines.extend(
            [
                "",
                "Seed-level linear slope of relative STM advantage per added "
                f"qubit: {_fmt_ci(summary.get('relative_advantage_slope_per_qubit'), percent=True)}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            "- **Fixed-N claim:** N=5 is an exact anchor because both coupling "
            "multipliers equal one there. Its paired result is an independent "
            "fresh-seed replication of the structural comparison.",
            "- **Trend claim:** growth with N is supported only when the paired "
            "seed-level slope interval excludes zero under a normalised coupling "
            "scheme. A positive point estimate alone is not described as a "
            "confirmed growth law.",
            "- All seeds are disjoint from the 64-seed definitive-protocol pool, "
            "and every raw row contains coupling/input hashes.",
            "",
        ]
    )
    if parity["missing_checkpoints"] or scaling["missing_checkpoints"]:
        lines.extend(
            [
                "## Incomplete computation",
                "",
                "This report is preliminary because checkpoints are missing. "
                "Re-run the corresponding command; completed trajectories will be "
                "skipped.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _fmt(value) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _fmt_sci(value) -> str:
    return "—" if value is None else f"{float(value):.3e}"


def _fmt_ci(summary: dict | None, percent: bool = False) -> str:
    if not summary or summary.get("mean") is None:
        return "—"
    scale = 100.0 if percent else 1.0
    lo, hi = summary["ci95_percentile"]
    suffix = "%" if percent else ""
    return (
        f"{scale * summary['mean']:+.2f}{suffix} "
        f"[{scale * lo:+.2f}, {scale * hi:+.2f}]{suffix}"
    )


def _parse_csv(value: str | None, cast=str):
    if value is None:
        return None
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _run_parity(args) -> None:
    preset = PARITY_PRESETS[args.preset]
    if args.n_seeds is not None:
        preset = replace(preset, n_seeds=args.n_seeds)
    protocol = parity_protocol(preset)
    protocol_hash = _sha256_json(protocol)
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        PARITY_DIR / f"{preset.name}_protocol.json",
        {"protocol": protocol, "protocol_sha256": protocol_hash},
    )
    methods = _parse_csv(args.methods) or PARITY_METHODS
    if any(method not in PARITY_METHODS for method in methods):
        raise ValueError(f"methods must be drawn from {PARITY_METHODS}")
    jobs = [
        ParityJob(method, seed)
        for method in methods
        for seed in protocol["seeds"]
    ]
    print(
        f"parity jobs={len(jobs)} workers={args.workers} "
        f"protocol={protocol_hash[:12]}",
        flush=True,
    )
    _run_parallel(
        run_parity_job,
        jobs,
        (preset, str(PARITY_DIR), protocol_hash),
        args.workers,
    )
    aggregate = aggregate_parity(preset)
    print(
        f"parity checkpoints={aggregate['complete_checkpoints']}/"
        f"{aggregate['expected_checkpoints']}",
        flush=True,
    )


def _run_scaling(args) -> None:
    preset = SCALING_PRESETS[args.preset]
    if args.n_seeds is not None:
        preset = replace(preset, n_seeds=args.n_seeds)
    schemes = _parse_csv(args.schemes) or ("variance",)
    protocol = scaling_protocol(preset, schemes)
    protocol_hash = _sha256_json(protocol)
    SCALING_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "-".join(schemes)
    _atomic_write_json(
        SCALING_DIR / f"{preset.name}_{suffix}_protocol.json",
        {"protocol": protocol, "protocol_sha256": protocol_hash},
    )
    n_values = _parse_csv(args.n_values, int) or preset.n_values
    if any(value not in preset.n_values for value in n_values):
        raise ValueError(f"N filter must be drawn from {preset.n_values}")
    methods = _parse_csv(args.methods) or SCALING_METHODS
    if any(method not in SCALING_METHODS for method in methods):
        raise ValueError(f"methods must be drawn from {SCALING_METHODS}")
    jobs = [
        ScalingJob(scheme, n_qubits, method, seed)
        for n_qubits in n_values
        for scheme in schemes
        for method in methods
        for seed in protocol["seeds"]
    ]
    print(
        f"scaling jobs={len(jobs)} workers={args.workers} "
        f"protocol={protocol_hash[:12]}",
        flush=True,
    )
    _run_parallel(
        run_scaling_job,
        jobs,
        (preset, str(SCALING_DIR), protocol_hash),
        args.workers,
    )
    aggregate = aggregate_scaling(preset, schemes)
    print(
        f"scaling checkpoints={aggregate['complete_checkpoints']}/"
        f"{aggregate['expected_checkpoints']}",
        flush=True,
    )


def _run_parity_references(args) -> None:
    preset = PARITY_PRESETS[args.preset]
    if args.n_seeds is not None:
        preset = replace(preset, n_seeds=args.n_seeds)
    protocol = parity_reference_protocol(preset)
    protocol_hash = _sha256_json(protocol)
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        PARITY_DIR / f"{preset.name}_reference_protocol.json",
        {"protocol": protocol, "protocol_sha256": protocol_hash},
    )
    jobs = [
        ParityJob(method, seed)
        for method in PARITY_REFERENCE_METHODS
        for seed in protocol["seeds"]
    ]
    print(
        f"parity-reference jobs={len(jobs)} workers={args.workers} "
        f"protocol={protocol_hash[:12]}",
        flush=True,
    )
    _run_parallel(
        run_parity_reference_job,
        jobs,
        (preset, str(PARITY_DIR), protocol_hash),
        args.workers,
    )
    aggregate = aggregate_parity_references(preset)
    print(
        f"parity-reference checkpoints={aggregate['complete_checkpoints']}/"
        f"{aggregate['expected_checkpoints']}",
        flush=True,
    )


def _run_report(args) -> None:
    parity_preset = PARITY_PRESETS[args.parity_preset]
    scaling_preset = SCALING_PRESETS[args.scaling_preset]
    if args.parity_n_seeds is not None:
        parity_preset = replace(parity_preset, n_seeds=args.parity_n_seeds)
    if args.scaling_n_seeds is not None:
        scaling_preset = replace(scaling_preset, n_seeds=args.scaling_n_seeds)
    schemes = _parse_csv(args.schemes) or ("variance",)
    parity = aggregate_parity(parity_preset)
    references = aggregate_parity_references(parity_preset)
    scaling = aggregate_scaling(scaling_preset, schemes)
    write_report(parity, scaling, references=references)
    print(f"wrote {REPORT_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--preset", choices=PARITY_PRESETS, default="paper")
    parity_parser.add_argument("--workers", type=int, default=4)
    parity_parser.add_argument("--n-seeds", type=int)
    parity_parser.add_argument("--methods")
    parity_parser.set_defaults(function=_run_parity)

    reference_parser = subparsers.add_parser("parity-references")
    reference_parser.add_argument(
        "--preset", choices=PARITY_PRESETS, default="paper"
    )
    reference_parser.add_argument("--workers", type=int, default=2)
    reference_parser.add_argument("--n-seeds", type=int)
    reference_parser.set_defaults(function=_run_parity_references)

    scaling_parser = subparsers.add_parser("scaling")
    scaling_parser.add_argument("--preset", choices=SCALING_PRESETS, default="paper")
    scaling_parser.add_argument("--workers", type=int, default=4)
    scaling_parser.add_argument("--n-seeds", type=int)
    scaling_parser.add_argument(
        "--schemes", default="variance", help="variance and/or kac"
    )
    scaling_parser.add_argument("--n-values", help="optional N filter")
    scaling_parser.add_argument("--methods")
    scaling_parser.set_defaults(function=_run_scaling)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument(
        "--parity-preset", choices=PARITY_PRESETS, default="paper"
    )
    report_parser.add_argument(
        "--scaling-preset", choices=SCALING_PRESETS, default="paper"
    )
    report_parser.add_argument("--parity-n-seeds", type=int)
    report_parser.add_argument("--scaling-n-seeds", type=int)
    report_parser.add_argument("--schemes", default="variance")
    report_parser.set_defaults(function=_run_report)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

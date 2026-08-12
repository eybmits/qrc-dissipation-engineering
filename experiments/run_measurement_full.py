"""Full finite-measurement comparison with operational total-shot accounting.

This protocol addresses two limitations of the earlier measurement checks:

* every one of the six active channels is evaluated at every budget with a
  validation-selected ridge parameter; and
* the independent-observable and grouped X/Y/Z models use the same total
  number of state preparations per reservoir time step.

For N=5 the readout contains 45 Pauli observables.  At a finite total budget B,
the independent model assigns B/45 shots to each observable, while the grouped
model assigns B/3 shots to each of the three product-basis settings.  The
default grid is parameterised by q=B/45, so the grouped setting receives 15q
shots and both models use exactly B=45q state preparations.

Each (channel, problem seed) is one atomic, restart-safe JSON checkpoint.  The
Hamiltonian, input, exact trajectory, and target are shared by both measurement
models and all budgets within a checkpoint.  Measurement sampling has
deterministic, disjoint RNG streams.  Ridge is selected using only a validation
window, the readout is retrained on train+validation, and the test block is
evaluated once.

Run the paper protocol:

    cd experiments
    PYTHONPATH=../src python run_measurement_full.py --workers 8

Re-running the command skips valid checkpoints and rebuilds the aggregate.
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
import copy
import hashlib
import json
import math
import platform
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
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
from run_final_scaling import build_jumps, deterministic_seeds


PROTOCOL_VERSION = "measurement-full-v3-2026-07-23"
N_QUBITS = 5
H = 0.5
DT = 0.5
GAMMA = 1.0
WASH = 200
TRAIN = 480
VALIDATION = 120
TEST = 400
DELAYS = tuple(range(1, 21))
CHANNELS = (
    "CD_paper",
    "B3_collective",
    "A1_heterogeneous",
    "B2_thermal",
    "B4_loss_exchange",
    "B5_pair",
)
CHANNEL_LABELS = {
    "CD_paper": "uniform local",
    "B3_collective": "collective",
    "A1_heterogeneous": "unequal local",
    "B2_thermal": "local gain/loss",
    "B4_loss_exchange": "loss plus exchange",
    "B5_pair": "pair loss",
}
# q is the independent-model shots per observable.  Total state preparations
# per time step are 45q in both models; the grouped model uses 15q per setting.
INDEPENDENT_SHOTS = (64, 256, 1024, 4096, 16384, 65536, 262144)
RIDGES = (
    0.0,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
)
RIDGE_INFINITY = "infinity_limit"
RIDGE_CHOICES = RIDGES + (RIDGE_INFINITY,)
RNG_NAMESPACE = 20260723


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(_json_bytes(payload) + b"\n")
    os.replace(tmp, path)


def _source_environment(root: Path) -> dict:
    source_paths = (
        Path(__file__).resolve(),
        root / "experiments" / "run_final_scaling.py",
        root / "src" / "qrc" / "dissipators.py",
        root / "src" / "qrc" / "readout.py",
        root / "src" / "qrc" / "reservoirs.py",
        root / "src" / "qrc" / "sparse_evolve.py",
        root / "src" / "qrc" / "tasks.py",
    )
    return {
        "git_head": _git_head(root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "files_sha256": {
            str(path.relative_to(root)): _sha256_file(path) for path in source_paths
        },
    }


def protocol_dict(
    *,
    channels: Sequence[str] = CHANNELS,
    seeds: Sequence[int] | None = None,
    independent_shots: Sequence[int] = INDEPENDENT_SHOTS,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    if seeds is None:
        seeds = deterministic_seeds(20)
    n_observables = 3 * N_QUBITS + 3 * math.comb(N_QUBITS, 2)
    budgets = [
        {
            "total_shots_per_time_step": n_observables * int(q),
            "independent_shots_per_observable": int(q),
            "grouped_shots_per_setting": n_observables * int(q) // 3,
        }
        for q in independent_shots
    ]
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "task": "short-term memory",
        "N": N_QUBITS,
        "h": H,
        "dt": DT,
        "gamma": GAMMA,
        "wash": WASH,
        "train": TRAIN,
        "validation": VALIDATION,
        "test": TEST,
        "delays": list(DELAYS),
        "channels": list(channels),
        "channel_labels": {
            channel: CHANNEL_LABELS[channel] for channel in channels
        },
        "seeds": [int(seed) for seed in seeds],
        "ridge_grid": list(RIDGE_CHOICES),
        "ridge_infinity_definition": (
            "analytic lambda-to-infinity correlation limit: W proportional "
            "to X^T Y; its arbitrary overall scale does not change squared "
            "Pearson-correlation capacity"
        ),
        "ridge_selection": (
            "maximize summed validation STM capacity after fitting on train; "
            "retrain on train+validation; evaluate the untouched test once"
        ),
        "delay_alignment": (
            "all delays use the same rows; the first max(delays) post-wash "
            "rows are excluded from readout fitting"
        ),
        "n_observables": n_observables,
        "measurement_models": {
            "independent": (
                "each Pauli observable is estimated by an independent exact "
                "binomial sample mean"
            ),
            "grouped": (
                "three X/Y/Z product-basis settings; joint multinomial "
                "bitstrings estimate all weight-one and same-axis weight-two "
                "observables with their sampling covariance"
            ),
        },
        "finite_budgets": budgets,
        "exact_endpoint": True,
        "total_shot_accounting": (
            "independent: 45 observables times q; grouped: 3 settings times "
            "15q; both equal 45q state preparations per reservoir time step"
        ),
        "paired_design": (
            "same Hamiltonian, input, targets, and split across channels within "
            "a seed; within each channel the exact trajectory is shared by both "
            "measurement models and all budgets"
        ),
        "problem_rng": (
            "numpy.random.default_rng(seed); couplings drawn before STM input"
        ),
        "measurement_rng": (
            "numpy SeedSequence([20260723, seed, channel_index, "
            "model_index, shots_per_observable])"
        ),
        "Hamiltonian": (
            "sum_ij J_ij X_i X_j + h sum_i Z_i + h(1+s) sum_i X_i"
        ),
        "couplings": "J_ij iid Uniform[-1,1] on the complete graph",
        "matched_dissipation": (
            "sum_k rate_k Tr(L_k^dag L_k) equals uniform local loss at gamma=1"
        ),
        "readout": (
            "weight-one X/Y/Z plus same-axis weight-two XX/YY/ZZ, linear "
            "ridge readout with bias"
        ),
        "source_environment": _source_environment(root),
    }
    return protocol


def _rotations(n_qubits: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    s_dagger = np.array([[1, 0], [0, -1j]], dtype=complex)
    one_qubit = {
        "x": hadamard,
        "y": hadamard @ s_dagger,
        "z": np.eye(2, dtype=complex),
    }

    def kron_power(matrix: np.ndarray) -> np.ndarray:
        out = np.ones((1, 1), dtype=complex)
        for _ in range(n_qubits):
            out = np.kron(out, matrix)
        return out

    rotations = {axis: kron_power(matrix) for axis, matrix in one_qubit.items()}
    d = 2**n_qubits
    bits = np.asarray(
        [
            [(basis >> (n_qubits - 1 - site)) & 1 for site in range(n_qubits)]
            for basis in range(d)
        ],
        dtype=int,
    )
    eigenvalues = 1 - 2 * bits
    pairs = [
        (i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
    ]
    pair_eigenvalues = np.column_stack(
        [eigenvalues[:, i] * eigenvalues[:, j] for i, j in pairs]
    )
    return rotations, eigenvalues, pair_eigenvalues


def probabilities_to_features(
    probabilities: np.ndarray,
    eigenvalues: np.ndarray,
    pair_eigenvalues: np.ndarray,
) -> np.ndarray:
    """Convert exact or sampled X/Y/Z outcome frequencies to Pauli features."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (time, 3, outcomes)")
    weight_one = [
        probabilities[:, axis, :] @ eigenvalues for axis in range(3)
    ]
    weight_two = [
        probabilities[:, axis, :] @ pair_eigenvalues for axis in range(3)
    ]
    return np.concatenate(weight_one + weight_two, axis=1)


def sample_independent(
    exact_features: np.ndarray, shots_per_observable: int, rng: np.random.Generator
) -> np.ndarray:
    """Independent exact binomial estimates for Pauli observables."""
    if shots_per_observable <= 0:
        raise ValueError("shots_per_observable must be positive")
    probabilities = np.clip((np.asarray(exact_features) + 1.0) / 2.0, 0.0, 1.0)
    counts = rng.binomial(shots_per_observable, probabilities)
    return 2.0 * counts / shots_per_observable - 1.0


def sample_grouped(
    outcome_probabilities: np.ndarray,
    shots_per_setting: int,
    rng: np.random.Generator,
    eigenvalues: np.ndarray,
    pair_eigenvalues: np.ndarray,
) -> np.ndarray:
    """Joint multinomial X/Y/Z sampling, preserving within-setting covariance."""
    if shots_per_setting <= 0:
        raise ValueError("shots_per_setting must be positive")
    sampled = np.empty_like(outcome_probabilities, dtype=float)
    for axis in range(3):
        counts = rng.multinomial(
            shots_per_setting, outcome_probabilities[:, axis, :]
        )
        sampled[:, axis, :] = counts / shots_per_setting
    return probabilities_to_features(sampled, eigenvalues, pair_eigenvalues)


def _capacity_columns(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    target_centered = target - target.mean(axis=0, keepdims=True)
    prediction_centered = prediction - prediction.mean(axis=0, keepdims=True)
    numerator = np.sum(target_centered * prediction_centered, axis=0) ** 2
    denominator = np.sum(target_centered**2, axis=0) * np.sum(
        prediction_centered**2, axis=0
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator >= 1e-30,
    )


def _ridge_key(ridge: float | str) -> str:
    if ridge == RIDGE_INFINITY:
        return RIDGE_INFINITY
    return f"{float(ridge):.12g}"


def _ridge_equal(left: float | str, right: float | str) -> bool:
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    return math.isclose(float(left), float(right))


def _fit_multioutput(
    X: np.ndarray, Y: np.ndarray, ridge: float | str
) -> np.ndarray:
    if ridge == RIDGE_INFINITY:
        # (X^T X + lambda I)^(-1) X^T Y is proportional to X^T Y as
        # lambda -> infinity.  Capacity is invariant to a nonzero scalar
        # rescaling of each predicted target, so normalise columns only for
        # numerical stability.
        weights = X.T @ Y
        norms = np.linalg.norm(weights, axis=0, keepdims=True)
        return np.divide(
            weights,
            norms,
            out=np.zeros_like(weights),
            where=norms > 1e-30,
        )
    if ridge <= 0.0:
        weights, *_ = np.linalg.lstsq(X, Y, rcond=None)
        return weights
    gram = X.T @ X
    rhs = X.T @ Y
    return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), rhs)


def ridge_selected_stm(features: np.ndarray, post_inputs: np.ndarray) -> dict:
    """Validation-select one ridge for all delays, then score the held-out test."""
    features = np.asarray(features, dtype=float)
    post_inputs = np.asarray(post_inputs, dtype=float)
    expected_rows = TRAIN + VALIDATION + TEST
    if features.shape[0] != expected_rows or post_inputs.shape != (expected_rows,):
        raise ValueError(
            f"expected {expected_rows} feature/input rows, got "
            f"{features.shape[0]} and {post_inputs.shape}"
        )
    Xb = readout.add_bias(features)
    targets = np.column_stack(
        [tasks.delayed_target(post_inputs, delay) for delay in DELAYS]
    )
    start = max(DELAYS)
    train_slice = slice(start, TRAIN)
    validation_slice = slice(TRAIN, TRAIN + VALIDATION)
    fit_slice = slice(start, TRAIN + VALIDATION)
    test_slice = slice(TRAIN + VALIDATION, expected_rows)
    if not np.all(np.isfinite(targets[start:])):
        raise RuntimeError("non-finite aligned delayed target")

    validation_by_ridge: dict[str, float] = {}
    validation_delay_by_ridge: dict[str, list[float]] = {}
    for ridge in RIDGE_CHOICES:
        weights = _fit_multioutput(
            Xb[train_slice], targets[train_slice], ridge
        )
        predicted = Xb[validation_slice] @ weights
        by_delay = _capacity_columns(
            targets[validation_slice], predicted
        )
        key = _ridge_key(ridge)
        validation_by_ridge[key] = float(np.sum(by_delay))
        validation_delay_by_ridge[key] = [float(x) for x in by_delay]

    # RIDGE_CHOICES is ordered from least to most regularised, ending in the
    # analytic infinity limit.  max keeps the first entry on an exact tie, so
    # the tie break is deterministic and test-blind.
    selected_ridge = max(
        RIDGE_CHOICES, key=lambda ridge: validation_by_ridge[_ridge_key(ridge)]
    )
    selected_key = _ridge_key(selected_ridge)
    weights = _fit_multioutput(
        Xb[fit_slice], targets[fit_slice], selected_ridge
    )
    predicted = Xb[test_slice] @ weights
    test_by_delay = _capacity_columns(targets[test_slice], predicted)
    return {
        "selected_ridge": selected_ridge,
        "validation_mc": validation_by_ridge[selected_key],
        "test_mc": float(np.sum(test_by_delay)),
        "validation_capacity_by_delay": validation_delay_by_ridge[selected_key],
        "test_capacity_by_delay": [float(x) for x in test_by_delay],
        "validation_mc_by_ridge": validation_by_ridge,
    }


def _measurement_rng(
    seed: int,
    channel_index: int,
    model_index: int,
    shots_per_observable: int,
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [
            RNG_NAMESPACE,
            int(seed),
            int(channel_index),
            int(model_index),
            int(shots_per_observable),
        ]
    )
    return np.random.default_rng(sequence)


def _trajectory(
    channel: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    problem_rng = np.random.default_rng(seed)
    couplings = res.random_couplings(N_QUBITS, 1.0, problem_rng)
    inputs = tasks.stm_inputs(WASH + TRAIN + VALIDATION + TEST, problem_rng)
    method_rng = np.random.default_rng(seed + 1)
    target_strength = dsp.jump_strength(dsp.local_loss(N_QUBITS, GAMMA))
    jumps = build_jumps(
        channel, couplings, N_QUBITS, target_strength, method_rng
    )
    jump_strength = dsp.jump_strength(jumps)
    H0 = ising_xx_hamiltonian(couplings, H, N_QUBITS)
    Hx = transverse_drive(N_QUBITS)
    reservoir = SparseLindbladReservoir.from_terms(
        N_QUBITS, H0 + H * Hx, H * Hx, jumps, DT
    )
    observables = readout.pauli_observables(N_QUBITS, max_weight=2)
    observable_matrices = np.stack([item.matrix for item in observables])
    rotations, eigenvalues, pair_eigenvalues = _rotations(N_QUBITS)

    n_rows = TRAIN + VALIDATION + TEST
    d = 2**N_QUBITS
    exact_features = np.empty((n_rows, len(observables)), dtype=float)
    outcome_probabilities = np.empty((n_rows, 3, d), dtype=float)
    rho = reservoir.initial_state()
    max_trace_error = 0.0
    min_raw_probability = 1.0
    row = 0
    for index, value in enumerate(inputs):
        rho = reservoir.step(rho, float(value))
        if index < WASH:
            continue
        rho_measure = 0.5 * (rho + rho.conj().T)
        trace = float(np.real(np.trace(rho_measure)))
        max_trace_error = max(max_trace_error, abs(trace - 1.0))
        if abs(trace) < 1e-14:
            raise RuntimeError("trajectory produced a zero-trace state")
        rho_measure = rho_measure / trace
        exact_features[row] = np.real(
            np.einsum("kij,ji->k", observable_matrices, rho_measure)
        )
        for axis_index, axis in enumerate(("x", "y", "z")):
            U = rotations[axis]
            probabilities = np.real(
                np.einsum("ai,ij,aj->a", U, rho_measure, U.conj())
            )
            min_raw_probability = min(
                min_raw_probability, float(np.min(probabilities))
            )
            probabilities = np.clip(probabilities, 0.0, None)
            total = float(probabilities.sum())
            if total <= 0.0:
                raise RuntimeError("measurement probabilities have zero mass")
            outcome_probabilities[row, axis_index] = probabilities / total
        row += 1
    if row != n_rows:
        raise RuntimeError(f"trajectory produced {row}/{n_rows} rows")

    grouped_exact = probabilities_to_features(
        outcome_probabilities, eigenvalues, pair_eigenvalues
    )
    exact_group_error = float(np.max(np.abs(grouped_exact - exact_features)))
    if exact_group_error > 2e-10:
        raise RuntimeError(
            f"grouped/exact feature ordering mismatch: {exact_group_error:.3e}"
        )
    diagnostics = {
        "target_jump_strength": float(target_strength),
        "jump_strength": float(jump_strength),
        "relative_jump_strength_error": float(
            abs(jump_strength - target_strength) / target_strength
        ),
        "max_trace_error": max_trace_error,
        "min_raw_measurement_probability": min_raw_probability,
        "max_grouped_exact_feature_error": exact_group_error,
        "n_features": int(exact_features.shape[1]),
        "n_rows": int(exact_features.shape[0]),
    }
    return exact_features, outcome_probabilities, inputs[WASH:], diagnostics


def _checkpoint_path(outdir: Path, channel: str, seed: int) -> Path:
    return outdir / "jobs" / f"{channel}__s{seed}.json"


def _load_valid_checkpoint(path: Path, protocol_sha256: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid checkpoint {path}: {exc}") from exc
    if payload.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(
            f"checkpoint protocol mismatch at {path}; use a new output directory"
        )
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete checkpoint at {path}")
    return payload


def run_one(
    channel: str, seed: int, protocol: dict, outdir_string: str
) -> dict:
    started = time.perf_counter()
    outdir = Path(outdir_string)
    protocol_sha256 = _sha256_json(protocol)
    output = _checkpoint_path(outdir, channel, seed)
    old = _load_valid_checkpoint(output, protocol_sha256)
    if old is not None:
        return {
            "status": "skip",
            "channel": channel,
            "seed": seed,
            "runtime_s": old.get("runtime_s", 0.0),
            "path": str(output),
        }

    exact_features, probabilities, post_inputs, diagnostics = _trajectory(
        channel, seed
    )
    _, eigenvalues, pair_eigenvalues = _rotations(N_QUBITS)
    exact_score = ridge_selected_stm(exact_features, post_inputs)
    rows: list[dict] = []
    for model in ("independent", "grouped"):
        rows.append(
            {
                "measurement_model": model,
                "is_exact": True,
                "total_shots_per_time_step": None,
                "shots_per_observable": None,
                "shots_per_setting": None,
                **copy.deepcopy(exact_score),
            }
        )

    channel_index = list(protocol["channels"]).index(channel)
    for budget in protocol["finite_budgets"]:
        total_shots = int(budget["total_shots_per_time_step"])
        independent_shots = int(
            budget["independent_shots_per_observable"]
        )
        grouped_shots = int(budget["grouped_shots_per_setting"])
        if 45 * independent_shots != total_shots:
            raise RuntimeError("invalid independent total-shot accounting")
        if 3 * grouped_shots != total_shots:
            raise RuntimeError("invalid grouped total-shot accounting")

        independent_rng = _measurement_rng(
            seed, channel_index, 0, independent_shots
        )
        independent_features = sample_independent(
            exact_features, independent_shots, independent_rng
        )
        independent_score = ridge_selected_stm(
            independent_features, post_inputs
        )
        rows.append(
            {
                "measurement_model": "independent",
                "is_exact": False,
                "total_shots_per_time_step": total_shots,
                "shots_per_observable": independent_shots,
                "shots_per_setting": None,
                **independent_score,
            }
        )

        grouped_rng = _measurement_rng(
            seed, channel_index, 1, independent_shots
        )
        grouped_features = sample_grouped(
            probabilities,
            grouped_shots,
            grouped_rng,
            eigenvalues,
            pair_eigenvalues,
        )
        grouped_score = ridge_selected_stm(grouped_features, post_inputs)
        rows.append(
            {
                "measurement_model": "grouped",
                "is_exact": False,
                "total_shots_per_time_step": total_shots,
                "shots_per_observable": None,
                "shots_per_setting": grouped_shots,
                **grouped_score,
            }
        )

    payload = {
        "artifact_type": "measurement_full_checkpoint",
        "status": "complete",
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha256,
        "channel": channel,
        "channel_label": CHANNEL_LABELS[channel],
        "seed": int(seed),
        "diagnostics": diagnostics,
        "scores": rows,
        "runtime_s": float(time.perf_counter() - started),
    }
    _atomic_write_json(output, payload)
    return {
        "status": "done",
        "channel": channel,
        "seed": seed,
        "runtime_s": payload["runtime_s"],
        "path": str(output),
    }


def run_manifest(protocol: dict, outdir: Path, workers: int) -> list[dict]:
    protocol_sha256 = _sha256_json(protocol)
    protocol_path = outdir / "protocol.json"
    if protocol_path.exists():
        try:
            existing_protocol = json.loads(protocol_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid existing protocol file {protocol_path}: {exc}"
            ) from exc
        if existing_protocol.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError(
                f"protocol mismatch at {protocol_path}; use a new output directory"
            )
    else:
        _atomic_write_json(
            protocol_path,
            {
                "artifact_type": "measurement_full_protocol",
                "created_utc": _utc_now(),
                "protocol_sha256": protocol_sha256,
                "protocol": protocol,
            },
        )
    jobs = [
        (channel, int(seed))
        for channel in protocol["channels"]
        for seed in protocol["seeds"]
    ]
    outputs: list[dict] = []
    if workers == 1:
        for index, (channel, seed) in enumerate(jobs, 1):
            result = run_one(channel, seed, protocol, str(outdir))
            outputs.append(result)
            print(
                f"[{index}/{len(jobs)}] {result['status']} "
                f"{channel} seed={seed} ({result['runtime_s']:.1f}s)",
                flush=True,
            )
        return outputs

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_one, channel, seed, protocol, str(outdir)): (
                channel,
                seed,
            )
            for channel, seed in jobs
        }
        for index, future in enumerate(as_completed(future_map), 1):
            channel, seed = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {
                    "status": "error",
                    "channel": channel,
                    "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_s": 0.0,
                }
            outputs.append(result)
            detail = result.get("error", f"{result['runtime_s']:.1f}s")
            print(
                f"[{index}/{len(jobs)}] {result['status']} "
                f"{channel} seed={seed} ({detail})",
                flush=True,
            )
    errors = [row for row in outputs if row["status"] == "error"]
    if errors:
        raise RuntimeError(f"{len(errors)} jobs failed; first error: {errors[0]}")
    return outputs


def _mean_ci(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=float)
    n = len(array)
    mean = float(np.mean(array))
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "se": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    se = float(np.std(array, ddof=1) / np.sqrt(n))
    half = float(student_t.ppf(0.975, n - 1) * se)
    return {
        "n": n,
        "mean": mean,
        "se": se,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def _budget_key(row: dict) -> str:
    if row["is_exact"]:
        return "exact"
    return str(int(row["total_shots_per_time_step"]))


def collect_rows(protocol: dict, outdir: Path) -> list[dict]:
    protocol_sha256 = _sha256_json(protocol)
    rows: list[dict] = []
    missing: list[str] = []
    for channel in protocol["channels"]:
        for seed in protocol["seeds"]:
            path = _checkpoint_path(outdir, channel, int(seed))
            checkpoint = _load_valid_checkpoint(path, protocol_sha256)
            if checkpoint is None:
                missing.append(str(path))
                continue
            for score in checkpoint["scores"]:
                rows.append(
                    {
                        "channel": channel,
                        "channel_label": CHANNEL_LABELS[channel],
                        "seed": int(seed),
                        **score,
                    }
                )
    if missing:
        raise RuntimeError(
            f"missing {len(missing)} checkpoints; first missing: {missing[0]}"
        )
    return rows


def validate_rows(protocol: dict, rows: Sequence[dict]) -> dict:
    expected_per_job = 2 * (len(protocol["finite_budgets"]) + 1)
    expected_rows = (
        len(protocol["channels"]) * len(protocol["seeds"]) * expected_per_job
    )
    issues: list[str] = []
    if len(rows) != expected_rows:
        issues.append(f"row count {len(rows)} != {expected_rows}")
    seen: set[tuple] = set()
    for row in rows:
        identity = (
            row["channel"],
            int(row["seed"]),
            row["measurement_model"],
            _budget_key(row),
        )
        if identity in seen:
            issues.append(f"duplicate row {identity}")
        seen.add(identity)
        if row["measurement_model"] not in ("independent", "grouped"):
            issues.append(f"invalid model {identity}")
        if not 0.0 <= float(row["test_mc"]) <= len(DELAYS) + 1e-9:
            issues.append(f"invalid test MC {identity}: {row['test_mc']}")
        if not any(
            _ridge_equal(row["selected_ridge"], ridge)
            for ridge in RIDGE_CHOICES
        ):
            issues.append(f"invalid selected ridge {identity}")
        if len(row["test_capacity_by_delay"]) != len(DELAYS):
            issues.append(f"invalid delay vector {identity}")
        if not row["is_exact"]:
            total = int(row["total_shots_per_time_step"])
            if row["measurement_model"] == "independent":
                if 45 * int(row["shots_per_observable"]) != total:
                    issues.append(f"independent shot mismatch {identity}")
            else:
                if 3 * int(row["shots_per_setting"]) != total:
                    issues.append(f"grouped shot mismatch {identity}")

    lookup = {
        (
            row["channel"],
            int(row["seed"]),
            row["measurement_model"],
            _budget_key(row),
        ): row
        for row in rows
    }
    for channel in protocol["channels"]:
        for seed in protocol["seeds"]:
            independent = lookup[(channel, int(seed), "independent", "exact")]
            grouped = lookup[(channel, int(seed), "grouped", "exact")]
            if independent["test_mc"] != grouped["test_mc"]:
                issues.append(f"exact endpoints differ for {channel} seed={seed}")
            if independent["selected_ridge"] != grouped["selected_ridge"]:
                issues.append(f"exact ridges differ for {channel} seed={seed}")
    if issues:
        raise RuntimeError(
            f"measurement validation failed with {len(issues)} issues: {issues[:5]}"
        )
    return {
        "status": "complete",
        "n_rows": len(rows),
        "n_jobs": len(protocol["channels"]) * len(protocol["seeds"]),
        "n_seeds": len(protocol["seeds"]),
        "n_channels": len(protocol["channels"]),
        "n_models": 2,
        "n_finite_budgets": len(protocol["finite_budgets"]),
        "issues": [],
    }


def aggregate(protocol: dict, outdir: Path) -> dict:
    rows = collect_rows(protocol, outdir)
    validation = validate_rows(protocol, rows)
    models = ("independent", "grouped")
    budget_keys = [
        str(item["total_shots_per_time_step"])
        for item in protocol["finite_budgets"]
    ] + ["exact"]

    summaries: list[dict] = []
    paired_vs_local: list[dict] = []
    ridge_counts: list[dict] = []
    winners: list[dict] = []
    lookup: dict[tuple, dict] = {
        (
            row["channel"],
            int(row["seed"]),
            row["measurement_model"],
            _budget_key(row),
        ): row
        for row in rows
    }
    for model in models:
        for budget_key in budget_keys:
            channel_means: dict[str, float] = {}
            for channel in protocol["channels"]:
                group = [
                    lookup[(channel, int(seed), model, budget_key)]
                    for seed in protocol["seeds"]
                ]
                stats = _mean_ci(row["test_mc"] for row in group)
                channel_means[channel] = float(stats["mean"])
                summaries.append(
                    {
                        "measurement_model": model,
                        "budget": budget_key,
                        "is_exact": budget_key == "exact",
                        "total_shots_per_time_step": (
                            None if budget_key == "exact" else int(budget_key)
                        ),
                        "channel": channel,
                        "channel_label": CHANNEL_LABELS[channel],
                        **stats,
                    }
                )
                counts = {
                    _ridge_key(ridge): sum(
                        _ridge_equal(row["selected_ridge"], ridge)
                        for row in group
                    )
                    for ridge in RIDGE_CHOICES
                }
                ridge_counts.append(
                    {
                        "measurement_model": model,
                        "budget": budget_key,
                        "channel": channel,
                        "counts": counts,
                    }
                )
                local = [
                    lookup[("CD_paper", int(seed), model, budget_key)]
                    for seed in protocol["seeds"]
                ]
                differences = [
                    float(row["test_mc"] - reference["test_mc"])
                    for row, reference in zip(group, local, strict=True)
                ]
                paired_vs_local.append(
                    {
                        "measurement_model": model,
                        "budget": budget_key,
                        "is_exact": budget_key == "exact",
                        "total_shots_per_time_step": (
                            None if budget_key == "exact" else int(budget_key)
                        ),
                        "channel": channel,
                        "channel_label": CHANNEL_LABELS[channel],
                        "reference_channel": "CD_paper",
                        "wins": int(sum(value > 0.0 for value in differences)),
                        "ties": int(sum(value == 0.0 for value in differences)),
                        **_mean_ci(differences),
                    }
                )
            winner = max(
                protocol["channels"], key=lambda channel: channel_means[channel]
            )
            winners.append(
                {
                    "measurement_model": model,
                    "budget": budget_key,
                    "is_exact": budget_key == "exact",
                    "total_shots_per_time_step": (
                        None if budget_key == "exact" else int(budget_key)
                    ),
                    "winner": winner,
                    "winner_label": CHANNEL_LABELS[winner],
                    "winner_mean_test_mc": channel_means[winner],
                }
            )

    grouped_minus_independent: list[dict] = []
    for budget_key in budget_keys:
        for channel in protocol["channels"]:
            differences = [
                lookup[(channel, int(seed), "grouped", budget_key)]["test_mc"]
                - lookup[
                    (channel, int(seed), "independent", budget_key)
                ]["test_mc"]
                for seed in protocol["seeds"]
            ]
            grouped_minus_independent.append(
                {
                    "budget": budget_key,
                    "is_exact": budget_key == "exact",
                    "total_shots_per_time_step": (
                        None if budget_key == "exact" else int(budget_key)
                    ),
                    "channel": channel,
                    "channel_label": CHANNEL_LABELS[channel],
                    "wins": int(sum(value > 0.0 for value in differences)),
                    "ties": int(sum(value == 0.0 for value in differences)),
                    **_mean_ci(differences),
                }
            )

    payload = {
        "artifact_type": "measurement_full_aggregate",
        "created_utc": _utc_now(),
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": _sha256_json(protocol),
        "validation": validation,
        "summaries": summaries,
        "paired_vs_local": paired_vs_local,
        "grouped_minus_independent": grouped_minus_independent,
        "ridge_selection_counts": ridge_counts,
        "winners": winners,
        "raw_rows": rows,
        "interpretation_boundary": (
            "The protocol compares two idealised estimators at equal state-"
            "preparation count. It does not include basis-change, reset, "
            "detector, or wall-clock costs."
        ),
    }
    _atomic_write_json(outdir / "measurement_full_aggregate.json", payload)
    return payload


def print_summary(payload: dict) -> None:
    summaries = {
        (
            row["measurement_model"],
            row["budget"],
            row["channel"],
        ): row
        for row in payload["summaries"]
    }
    paired = {
        (
            row["measurement_model"],
            row["budget"],
            row["channel"],
        ): row
        for row in payload["paired_vs_local"]
    }
    winners = {
        (row["measurement_model"], row["budget"]): row
        for row in payload["winners"]
    }
    budget_keys = sorted(
        {
            row["budget"]
            for row in payload["summaries"]
            if row["budget"] != "exact"
        },
        key=int,
    ) + ["exact"]
    n_seeds = payload["validation"]["n_seeds"]
    print(
        f"\nEqual-total-shot STM comparison (mean test MC; n={n_seeds}):"
    )
    for model in ("independent", "grouped"):
        print(f"\n{model.upper()}")
        print(
            f"{'total shots':>12} {'local':>8} {'collective':>11} "
            f"{'coll-local [95% CI]':>27} {'winner':>18}"
        )
        for budget in budget_keys:
            local = summaries[(model, budget, "CD_paper")]["mean"]
            collective = summaries[(model, budget, "B3_collective")]["mean"]
            delta = paired[(model, budget, "B3_collective")]
            label = "exact" if budget == "exact" else budget
            if delta["ci95_low"] is None:
                interval = f"{delta['mean']:+.2f} [n<2]"
            else:
                interval = (
                    f"{delta['mean']:+.2f} "
                    f"[{delta['ci95_low']:+.2f},{delta['ci95_high']:+.2f}]"
                )
            winner = winners[(model, budget)]["winner_label"]
            print(
                f"{label:>12} {local:8.2f} {collective:11.2f} "
                f"{interval:>27} {winner:>18}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(RESULTS_DIR) / "measurement_full_v3",
    )
    parser.add_argument(
        "--mode", choices=("run", "aggregate", "validate", "all"), default="all"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one seed, two channels, two finite budgets; uses a separate outdir",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    if args.smoke:
        channels = CHANNELS[:2]
        seeds = deterministic_seeds(1)
        budgets = INDEPENDENT_SHOTS[:2]
        if args.outdir == Path(RESULTS_DIR) / "measurement_full_v3":
            args.outdir = Path(RESULTS_DIR) / "measurement_full_v3_smoke"
    else:
        channels = CHANNELS
        seeds = deterministic_seeds(20)
        budgets = INDEPENDENT_SHOTS
    protocol = protocol_dict(
        channels=channels, seeds=seeds, independent_shots=budgets
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("run", "all"):
        run_manifest(protocol, args.outdir, args.workers)
    if args.mode in ("aggregate", "validate", "all"):
        rows = collect_rows(protocol, args.outdir)
        validation = validate_rows(protocol, rows)
        print(json.dumps(validation, indent=2))
    if args.mode in ("aggregate", "all"):
        payload = aggregate(protocol, args.outdir)
        print_summary(payload)


if __name__ == "__main__":
    main()

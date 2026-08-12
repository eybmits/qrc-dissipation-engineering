"""Full-input stationary-spectrum diagnostic for the primary collective loss.

This additive, deterministic driver audits the constant-input Liouvillian of
the primary N=5 collective-loss reservoir at the paper operating point.  It
does not modify or reinterpret any sealed result group.

The protocol is intentionally explicit:

* the 21-point input grid covers s=0,...,1 in increments of 0.05;
* the ten Hamiltonian seeds are the existing R_wash control seeds;
* h=dt=0.5 and the collective jump uses the matched local-loss Frobenius budget;
* every 1024-dimensional Liouvillian is diagonalised densely;
* six predeclared cases receive independent sparse shift-invert checks.

The diagnostic establishes stationary-mode uniqueness and positive sampled
constant-input gaps only on the declared finite grid.  It is not a proof of a
continuum bound and does not prove contraction for arbitrary switched inputs.

Examples
--------
PYTHONPATH=src:experiments python \
    experiments/run_collective_loss_full_input_diagnostic.py freeze
PYTHONPATH=src:experiments python \
    experiments/run_collective_loss_full_input_diagnostic.py run --workers 4
PYTHONPATH=src:experiments python \
    experiments/run_collective_loss_full_input_diagnostic.py verify
PYTHONPATH=src:experiments python \
    experiments/run_collective_loss_full_input_diagnostic.py run \
    --workers 4 --recompute
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
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import _paths  # noqa: F401
import numpy as np
import scipy
from scipy import linalg
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from _paths import REPORTS_DIR, RESULTS_DIR
from qrc import dissipators as dsp
from qrc import liouvillian as dense_lio
from qrc import reservoirs as res
from qrc import sparse_evolve as sparse_lio
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive


PROTOCOL_VERSION = "collective-loss-full-input-v1-2026-07-24"
N_QUBITS = 5
H = 0.5
DT = 0.5
GAMMA = 1.0
S_GRID = tuple(float(value) for value in np.linspace(0.0, 1.0, 21))
# First ten deterministic seeds in the definitive 2024 namespace.  These are
# exactly the seeds used by the existing R_wash initial-state controls.
WASHOUT_CONTROL_SEEDS = (
    518677875,
    1451336746,
    198305900,
    460255569,
    682143856,
    664543175,
    1951374833,
    1716840368,
    1965675192,
    2138468722,
)
SPARSE_CROSSCHECK_CASES = (
    (518677875, 0.0),
    (518677875, 0.5),
    (518677875, 1.0),
    (682143856, 0.5),
    (2138468722, 0.0),
    (2138468722, 1.0),
)

STATIONARY_ABS_TOL = 1e-8
POSITIVE_REAL_TOL = 1e-9
MIN_GAP_TOL = 1e-10
RELATIVE_RESIDUAL_TOL = 1e-11
TRACE_PRESERVATION_TOL = 1e-13
SPARSE_MATRIX_TOL = 1e-13
SPARSE_EIGENVALUE_TOL = 2e-6
SPARSE_RELATIVE_RESIDUAL_TOL = 2e-9
SPARSE_SHIFT = complex(1e-7, 1e-7)
SPARSE_K_NEAR_ZERO = 6
SPARSE_K_GAP = 2
SPARSE_SOLVER_TOL = 1e-11
SPARSE_MAXITER = 50_000
SPARSE_NCV = 36

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = Path(RESULTS_DIR) / "collective_loss_full_input_diagnostic"
PROTOCOL_PATH = RESULT_DIR / "protocol.json"
RAW_PATH = RESULT_DIR / "raw_spectrum.json"
AGGREGATE_PATH = RESULT_DIR / "aggregate.json"
REPORT_PATH = Path(REPORTS_DIR) / "collective_loss_full_input_diagnostic.md"
SOURCE_FILES = (
    "experiments/run_collective_loss_full_input_diagnostic.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
)

CAVEAT = (
    "Finite-grid constant-input diagnostic only: positive sampled gaps and a "
    "unique sampled stationary mode do not prove a continuum lower bound or "
    "contraction under arbitrary switched-input sequences."
)


def _canonical_json(value: object) -> str:
    """Canonical JSON used for all scientific-payload hashes."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required scientific source is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _seed_source_check() -> bool:
    generated = np.random.default_rng(2024).integers(
        0,
        2**31 - 1,
        len(WASHOUT_CONTROL_SEEDS),
    )
    return tuple(int(value) for value in generated) == WASHOUT_CONTROL_SEEDS


def protocol_dict() -> dict:
    """Return the complete predeclared scientific protocol."""
    if not _seed_source_check():
        raise RuntimeError("hard-coded washout-control seeds drifted from namespace")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "diagnostic_scope": "primary_N5_collective_loss_constant_input_spectrum",
        "n_qubits": N_QUBITS,
        "hilbert_dimension": 2**N_QUBITS,
        "liouvillian_dimension": 4**N_QUBITS,
        "h": H,
        "dt": DT,
        "gamma_reference": GAMMA,
        "hamiltonian": (
            "sum_{i<j} J_ij X_i X_j + h sum_i Z_i + "
            "h(1+s) sum_i X_i"
        ),
        "coupling_distribution": "independent J_ij ~ Uniform[-1,1], i<j",
        "coupling_rng": "numpy.default_rng(seed), identical to R_wash",
        "dissipator": "single collective jump sum_i sigma_i^-",
        "jump_budget": (
            "sum_k Tr(L_k^dagger L_k) matched to unit-rate local loss"
        ),
        "s_grid": list(S_GRID),
        "s_grid_count": len(S_GRID),
        "seeds": list(WASHOUT_CONTROL_SEEDS),
        "seed_count": len(WASHOUT_CONTROL_SEEDS),
        "seed_source": (
            "first ten numpy.default_rng(2024) integer seeds; identical to "
            "existing R_wash controls"
        ),
        "dense_solver": {
            "implementation": "scipy.linalg.eig",
            "spectrum": "all eigenvalues and right eigenvectors",
            "check_finite": True,
            "stationary_abs_tolerance": STATIONARY_ABS_TOL,
            "positive_real_tolerance": POSITIVE_REAL_TOL,
            "minimum_gap_tolerance": MIN_GAP_TOL,
            "relative_residual_tolerance": RELATIVE_RESIDUAL_TOL,
            "trace_preservation_tolerance": TRACE_PRESERVATION_TOL,
        },
        "sparse_crosscheck": {
            "cases": [
                {"seed": int(seed), "s": float(s_value)}
                for seed, s_value in SPARSE_CROSSCHECK_CASES
            ],
            "implementation": "scipy.sparse.linalg.eigs shift-invert",
            "near_zero_modes": SPARSE_K_NEAR_ZERO,
            "gap_target_modes": SPARSE_K_GAP,
            "shift_real": SPARSE_SHIFT.real,
            "shift_imag": SPARSE_SHIFT.imag,
            "solver_tolerance": SPARSE_SOLVER_TOL,
            "maximum_iterations": SPARSE_MAXITER,
            "ncv": SPARSE_NCV,
            "matrix_tolerance": SPARSE_MATRIX_TOL,
            "eigenvalue_tolerance": SPARSE_EIGENVALUE_TOL,
            "relative_residual_tolerance": SPARSE_RELATIVE_RESIDUAL_TOL,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "scientific_sources_sha256": _source_hashes(),
        "interpretation_caveat": CAVEAT,
    }


def freeze_protocol() -> tuple[dict, str]:
    """Create the protocol before rows, or fail if the frozen file drifted."""
    protocol = protocol_dict()
    payload = {
        "artifact_type": "collective_loss_full_input_protocol",
        "status": "frozen_before_diagnostic_rows",
        "protocol": protocol,
        "protocol_sha256": _sha256_json(protocol),
    }
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"frozen protocol drift: {PROTOCOL_PATH}")
    else:
        if RAW_PATH.exists() or AGGREGATE_PATH.exists():
            raise RuntimeError("diagnostic rows exist without a frozen protocol")
        _atomic_write_json(PROTOCOL_PATH, payload)
    return payload, _sha256_file(PROTOCOL_PATH)


def _complex_pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def _spectrum_order(values: np.ndarray) -> np.ndarray:
    """Stable scientific ordering: descending real part, then |imag| and imag."""
    return np.lexsort((values.imag, np.abs(values.imag), -values.real))


def _relative_residuals(
    matrix: np.ndarray,
    values: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, float]:
    matrix_scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    vector_norms = np.linalg.norm(vectors, axis=0)
    if np.any(vector_norms == 0.0):
        raise RuntimeError("dense eigensolver returned a zero eigenvector")
    residual = matrix @ vectors - vectors * values[np.newaxis, :]
    relative = np.linalg.norm(residual, axis=0) / (matrix_scale * vector_norms)
    return relative, matrix_scale


def _trace_preservation_residual(matrix: np.ndarray, dimension: int) -> float:
    trace_functional = dense_lio.vec(np.eye(dimension, dtype=complex)).conj()
    scale = (
        max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
        * float(np.linalg.norm(trace_functional))
    )
    return float(np.linalg.norm(trace_functional @ matrix) / scale)


def _spectrum_sha256(values: np.ndarray) -> str:
    packed = np.empty((len(values), 2), dtype="<f8")
    packed[:, 0] = values.real
    packed[:, 1] = values.imag
    return hashlib.sha256(packed.tobytes()).hexdigest()


def analyze_dense_liouvillian(matrix: np.ndarray) -> tuple[dict, np.ndarray]:
    """Diagonalise and validate one full dense constant-input Liouvillian."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Liouvillian must be square")
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("Liouvillian contains non-finite entries")

    values, vectors = linalg.eig(
        matrix,
        left=False,
        right=True,
        overwrite_a=False,
        check_finite=True,
    )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise RuntimeError("dense eigensolver returned non-finite values")

    residuals, matrix_scale = _relative_residuals(matrix, values, vectors)
    order = _spectrum_order(values)
    values = values[order]
    residuals = residuals[order]
    stationary_mask = np.abs(values) <= STATIONARY_ABS_TOL
    nonstationary_indices = np.flatnonzero(~stationary_mask)
    if not len(nonstationary_indices):
        raise RuntimeError("no nonstationary Liouvillian mode was found")
    gap_index = int(nonstationary_indices[0])
    gap_eigenvalue = values[gap_index]
    gap = float(-gap_eigenvalue.real)
    spectral_abscissa = float(np.max(values.real))
    max_residual = float(np.max(residuals))

    summary = {
        "eigenvalue_count": int(len(values)),
        "stationary_mode_count": int(np.count_nonzero(stationary_mask)),
        "stationary_eigenvalues": [
            _complex_pair(value) for value in values[stationary_mask]
        ],
        "stationary_abs_max": (
            float(np.max(np.abs(values[stationary_mask])))
            if np.any(stationary_mask)
            else None
        ),
        "first_nonstationary_eigenvalue": _complex_pair(gap_eigenvalue),
        "first_nonstationary_decay_gap": gap,
        "first_nonstationary_relative_residual": float(residuals[gap_index]),
        "spectral_abscissa": spectral_abscissa,
        "positive_real_part_leakage": float(max(0.0, spectral_abscissa)),
        "max_all_mode_relative_residual": max_residual,
        "matrix_infinity_norm": matrix_scale,
        "spectrum_sha256": _spectrum_sha256(values),
        "spectrum": [_complex_pair(value) for value in values],
    }
    return summary, values


def _deterministic_v0(size: int) -> np.ndarray:
    real = np.linspace(1.0, 2.0, size, dtype=float)
    imag = np.linspace(2.0, 1.0, size, dtype=float)
    value = real + 1j * imag
    return value / np.linalg.norm(value)


def _sparse_eigs_checked(
    matrix,
    *,
    k: int,
    sigma: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        values, vectors = eigs(
            matrix,
            k=k,
            sigma=sigma,
            which="LM",
            v0=_deterministic_v0(matrix.shape[0]),
            tol=SPARSE_SOLVER_TOL,
            maxiter=SPARSE_MAXITER,
            ncv=SPARSE_NCV,
        )
    except ArpackNoConvergence as error:
        converged = 0 if error.eigenvalues is None else len(error.eigenvalues)
        raise RuntimeError(
            f"sparse eigensolver did not converge: {converged}/{k} modes"
        ) from error
    if len(values) != k:
        raise RuntimeError(f"sparse eigensolver returned {len(values)}/{k} modes")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
        raise RuntimeError("sparse eigensolver returned non-finite values")
    matrix_scale = max(
        1.0,
        float(np.max(np.asarray(np.abs(matrix).sum(axis=1)).ravel())),
    )
    vector_norms = np.linalg.norm(vectors, axis=0)
    residual = matrix @ vectors - vectors * values[np.newaxis, :]
    relative = np.linalg.norm(residual, axis=0) / (matrix_scale * vector_norms)
    return values, vectors, relative


def sparse_crosscheck(
    dense_matrix: np.ndarray,
    sparse_matrix,
    dense_values: np.ndarray,
    dense_gap_eigenvalue: complex,
) -> dict:
    """Cross-check representative dense results with sparse shift-invert."""
    difference = sparse_matrix.toarray() - dense_matrix
    matrix_difference = float(np.max(np.abs(difference)))

    near_values, _, near_residuals = _sparse_eigs_checked(
        sparse_matrix,
        k=SPARSE_K_NEAR_ZERO,
        sigma=SPARSE_SHIFT,
    )
    costs = np.abs(near_values[:, np.newaxis] - dense_values[np.newaxis, :])
    near_rows, dense_columns = linear_sum_assignment(costs)
    if len(near_rows) != SPARSE_K_NEAR_ZERO:
        raise RuntimeError("sparse/dense near-zero assignment was incomplete")
    near_differences = costs[near_rows, dense_columns]
    stationary_index = int(np.argmin(np.abs(near_values)))

    gap_sigma = dense_gap_eigenvalue + SPARSE_SHIFT
    gap_values, _, gap_residuals = _sparse_eigs_checked(
        sparse_matrix,
        k=SPARSE_K_GAP,
        sigma=gap_sigma,
    )
    gap_index = int(np.argmin(np.abs(gap_values - dense_gap_eigenvalue)))
    gap_difference = float(abs(gap_values[gap_index] - dense_gap_eigenvalue))

    result = {
        "dense_sparse_matrix_max_abs_difference": matrix_difference,
        "near_zero_sparse_eigenvalues": [
            _complex_pair(value)
            for value in near_values[np.argsort(np.abs(near_values))]
        ],
        "near_zero_sparse_stationary_abs": float(abs(near_values[stationary_index])),
        "near_zero_sparse_stationary_count": int(
            np.count_nonzero(np.abs(near_values) <= STATIONARY_ABS_TOL)
        ),
        "near_zero_max_dense_eigenvalue_abs_difference": float(
            np.max(near_differences)
        ),
        "near_zero_max_relative_residual": float(np.max(near_residuals)),
        "targeted_gap_sparse_eigenvalue": _complex_pair(gap_values[gap_index]),
        "targeted_gap_dense_eigenvalue": _complex_pair(dense_gap_eigenvalue),
        "targeted_gap_eigenvalue_abs_difference": gap_difference,
        "targeted_gap_relative_residual": float(gap_residuals[gap_index]),
    }
    _validate_sparse_crosscheck(result)
    return result


def _validate_sparse_crosscheck(result: dict) -> None:
    if result["dense_sparse_matrix_max_abs_difference"] > SPARSE_MATRIX_TOL:
        raise RuntimeError("dense/sparse Liouvillian matrices disagree")
    if result["near_zero_sparse_stationary_count"] != 1:
        raise RuntimeError("sparse cross-check did not find one stationary mode")
    if (
        result["near_zero_max_dense_eigenvalue_abs_difference"]
        > SPARSE_EIGENVALUE_TOL
    ):
        raise RuntimeError("near-zero sparse/dense eigenvalues disagree")
    if result["targeted_gap_eigenvalue_abs_difference"] > SPARSE_EIGENVALUE_TOL:
        raise RuntimeError("targeted sparse/dense gap eigenvalues disagree")
    if (
        result["near_zero_max_relative_residual"]
        > SPARSE_RELATIVE_RESIDUAL_TOL
        or result["targeted_gap_relative_residual"]
        > SPARSE_RELATIVE_RESIDUAL_TOL
    ):
        raise RuntimeError("sparse eigensolver residual exceeds tolerance")


def _build_seed_terms(seed: int) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    couplings = res.random_couplings(N_QUBITS, 1.0, rng)
    drive_direction = transverse_drive(N_QUBITS)
    static_hamiltonian = (
        ising_xx_hamiltonian(couplings, H, N_QUBITS) + H * drive_direction
    )
    drive_hamiltonian = H * drive_direction
    target = dsp.jump_strength(dsp.local_loss(N_QUBITS, GAMMA))
    jumps = dsp.normalize_jump_strength(
        dsp.collective_loss(N_QUBITS, GAMMA),
        target,
    )
    return couplings, jumps, static_hamiltonian, drive_hamiltonian


def _validate_dense_row(row: dict) -> None:
    if row["eigenvalue_count"] != 4**N_QUBITS:
        raise RuntimeError("dense solver did not return the full spectrum")
    if row["stationary_mode_count"] != 1:
        raise RuntimeError(
            f"stationary-mode multiplicity is {row['stationary_mode_count']}"
        )
    if not math.isfinite(row["first_nonstationary_decay_gap"]):
        raise RuntimeError("non-finite first nonstationary gap")
    if row["first_nonstationary_decay_gap"] <= MIN_GAP_TOL:
        raise RuntimeError("first nonstationary gap is not strictly positive")
    if row["positive_real_part_leakage"] > POSITIVE_REAL_TOL:
        raise RuntimeError("positive-real-part leakage exceeds tolerance")
    if row["max_all_mode_relative_residual"] > RELATIVE_RESIDUAL_TOL:
        raise RuntimeError("dense eigenpair residual exceeds tolerance")
    if row["trace_preservation_relative_residual"] > TRACE_PRESERVATION_TOL:
        raise RuntimeError("trace-preservation residual exceeds tolerance")
    if not math.isfinite(row["relative_jump_budget_error"]):
        raise RuntimeError("non-finite jump-budget error")
    if row["relative_jump_budget_error"] > 1e-14:
        raise RuntimeError("collective jump does not match the declared budget")


def _compute_seed(seed: int) -> list[dict]:
    couplings, jumps, static_hamiltonian, drive_hamiltonian = _build_seed_terms(seed)
    target = float(dsp.jump_strength(dsp.local_loss(N_QUBITS, GAMMA)))
    actual = float(dsp.jump_strength(jumps))
    coupling_hash = _sha256_array(couplings)

    dense_base = dense_lio.lindbladian(static_hamiltonian, jumps)
    dense_drive = dense_lio.commutator_super(drive_hamiltonian)
    sparse_base = sparse_lio.lindbladian(static_hamiltonian, jumps)
    sparse_drive = sparse_lio.commutator_super(drive_hamiltonian)
    dimension = 2**N_QUBITS

    rows: list[dict] = []
    for s_index, s_value in enumerate(S_GRID):
        dense_matrix = dense_base + s_value * dense_drive
        summary, dense_values = analyze_dense_liouvillian(dense_matrix)
        row = {
            "seed": int(seed),
            "s_index": int(s_index),
            "s": float(s_value),
            "coupling_sha256": coupling_hash,
            "target_jump_strength": target,
            "actual_jump_strength": actual,
            "relative_jump_budget_error": float(abs(actual - target) / target),
            "trace_preservation_relative_residual": (
                _trace_preservation_residual(dense_matrix, dimension)
            ),
            **summary,
        }
        if (seed, s_value) in SPARSE_CROSSCHECK_CASES:
            sparse_matrix = sparse_base + s_value * sparse_drive
            row["sparse_crosscheck"] = sparse_crosscheck(
                dense_matrix,
                sparse_matrix,
                dense_values,
                complex(*row["first_nonstationary_eigenvalue"]),
            )
        else:
            row["sparse_crosscheck"] = None
        _validate_dense_row(row)
        rows.append(row)
    return rows


def compute_raw_payload(protocol_sha256: str, workers: int) -> dict:
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        nested_rows = [_compute_seed(seed) for seed in WASHOUT_CONTROL_SEEDS]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            nested_rows = list(pool.map(_compute_seed, WASHOUT_CONTROL_SEEDS))
    rows = [row for seed_rows in nested_rows for row in seed_rows]
    rows.sort(key=lambda row: (WASHOUT_CONTROL_SEEDS.index(row["seed"]), row["s_index"]))
    return {
        "artifact_type": "collective_loss_full_input_raw_spectrum",
        "protocol_sha256": protocol_sha256,
        "row_count": len(rows),
        "rows": rows,
    }


def _extreme_record(row: dict) -> dict:
    return {
        "seed": int(row["seed"]),
        "s": float(row["s"]),
        "gap": float(row["first_nonstationary_decay_gap"]),
        "eigenvalue": list(row["first_nonstationary_eigenvalue"]),
    }


def build_aggregate(
    raw: dict,
    *,
    protocol_file_sha256: str,
    raw_file_sha256: str,
) -> dict:
    rows = raw["rows"]
    expected = {
        (seed, s_index)
        for seed in WASHOUT_CONTROL_SEEDS
        for s_index in range(len(S_GRID))
    }
    observed = {(row["seed"], row["s_index"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError("raw spectrum does not cover the declared seed/input grid")
    for row in rows:
        if row["s"] != S_GRID[row["s_index"]]:
            raise RuntimeError("raw s value does not match its declared grid index")
        _validate_dense_row(row)

    gaps = np.asarray(
        [row["first_nonstationary_decay_gap"] for row in rows],
        dtype=float,
    )
    min_row = rows[int(np.argmin(gaps))]
    max_row = rows[int(np.argmax(gaps))]
    per_seed: list[dict] = []
    for seed in WASHOUT_CONTROL_SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        seed_gaps = np.asarray(
            [row["first_nonstationary_decay_gap"] for row in selected],
            dtype=float,
        )
        seed_min = selected[int(np.argmin(seed_gaps))]
        seed_max = selected[int(np.argmax(seed_gaps))]
        per_seed.append(
            {
                "seed": seed,
                "coupling_sha256": selected[0]["coupling_sha256"],
                "unique_stationary_mode_rows": int(
                    sum(row["stationary_mode_count"] == 1 for row in selected)
                ),
                "grid_rows": len(selected),
                "minimum_gap": _extreme_record(seed_min),
                "maximum_gap": _extreme_record(seed_max),
                "mean_gap": float(np.mean(seed_gaps)),
                "maximum_positive_real_part_leakage": float(
                    max(row["positive_real_part_leakage"] for row in selected)
                ),
                "maximum_dense_relative_residual": float(
                    max(row["max_all_mode_relative_residual"] for row in selected)
                ),
            }
        )

    crosschecks = [
        {
            "seed": row["seed"],
            "s": row["s"],
            **row["sparse_crosscheck"],
        }
        for row in rows
        if row["sparse_crosscheck"] is not None
    ]
    if len(crosschecks) != len(SPARSE_CROSSCHECK_CASES):
        raise RuntimeError("sparse cross-check grid is incomplete")

    aggregate = {
        "artifact_type": "collective_loss_full_input_aggregate",
        "status": "complete",
        "all_declared_checks_passed": True,
        "protocol_sha256": raw["protocol_sha256"],
        "protocol_file_sha256": protocol_file_sha256,
        "raw_payload_sha256": _sha256_json(raw),
        "raw_file_sha256": raw_file_sha256,
        "row_count": len(rows),
        "seed_count": len(WASHOUT_CONTROL_SEEDS),
        "s_grid_count": len(S_GRID),
        "full_grid_complete": True,
        "all_rows_have_unique_stationary_mode": all(
            row["stationary_mode_count"] == 1 for row in rows
        ),
        "unique_stationary_mode_rows": int(
            sum(row["stationary_mode_count"] == 1 for row in rows)
        ),
        "minimum_sampled_gap": _extreme_record(min_row),
        "maximum_sampled_gap": _extreme_record(max_row),
        "mean_sampled_gap": float(np.mean(gaps)),
        "maximum_positive_real_part_leakage": float(
            max(row["positive_real_part_leakage"] for row in rows)
        ),
        "maximum_dense_relative_residual": float(
            max(row["max_all_mode_relative_residual"] for row in rows)
        ),
        "maximum_trace_preservation_relative_residual": float(
            max(row["trace_preservation_relative_residual"] for row in rows)
        ),
        "maximum_relative_jump_budget_error": float(
            max(row["relative_jump_budget_error"] for row in rows)
        ),
        "sparse_crosscheck_count": len(crosschecks),
        "maximum_sparse_matrix_abs_difference": float(
            max(
                item["dense_sparse_matrix_max_abs_difference"]
                for item in crosschecks
            )
        ),
        "maximum_sparse_dense_eigenvalue_abs_difference": float(
            max(
                max(
                    item["near_zero_max_dense_eigenvalue_abs_difference"],
                    item["targeted_gap_eigenvalue_abs_difference"],
                )
                for item in crosschecks
            )
        ),
        "maximum_sparse_relative_residual": float(
            max(
                max(
                    item["near_zero_max_relative_residual"],
                    item["targeted_gap_relative_residual"],
                )
                for item in crosschecks
            )
        ),
        "per_seed": per_seed,
        "sparse_crosschecks": crosschecks,
        "interpretation_caveat": CAVEAT,
    }
    _validate_aggregate(aggregate)
    return aggregate


def _validate_aggregate(aggregate: dict) -> None:
    if aggregate["row_count"] != len(WASHOUT_CONTROL_SEEDS) * len(S_GRID):
        raise RuntimeError("aggregate row count is incomplete")
    if not aggregate["all_rows_have_unique_stationary_mode"]:
        raise RuntimeError("not every sampled Liouvillian has one stationary mode")
    if aggregate["minimum_sampled_gap"]["gap"] <= MIN_GAP_TOL:
        raise RuntimeError("aggregate minimum sampled gap is not positive")
    if aggregate["maximum_positive_real_part_leakage"] > POSITIVE_REAL_TOL:
        raise RuntimeError("aggregate positive-real leakage exceeds tolerance")
    if aggregate["maximum_dense_relative_residual"] > RELATIVE_RESIDUAL_TOL:
        raise RuntimeError("aggregate dense residual exceeds tolerance")
    if (
        aggregate["maximum_trace_preservation_relative_residual"]
        > TRACE_PRESERVATION_TOL
    ):
        raise RuntimeError("aggregate trace residual exceeds tolerance")
    if aggregate["maximum_sparse_matrix_abs_difference"] > SPARSE_MATRIX_TOL:
        raise RuntimeError("aggregate dense/sparse matrix mismatch")
    if (
        aggregate["maximum_sparse_dense_eigenvalue_abs_difference"]
        > SPARSE_EIGENVALUE_TOL
    ):
        raise RuntimeError("aggregate dense/sparse eigenvalue mismatch")
    if (
        aggregate["maximum_sparse_relative_residual"]
        > SPARSE_RELATIVE_RESIDUAL_TOL
    ):
        raise RuntimeError("aggregate sparse residual exceeds tolerance")


def render_report(aggregate: dict) -> str:
    minimum = aggregate["minimum_sampled_gap"]
    maximum = aggregate["maximum_sampled_gap"]
    lines = [
        "# Collective-loss full-input stationary-spectrum diagnostic",
        "",
        "**Status: PASS.** All declared numerical and integrity checks passed.",
        "",
        "## Scope and protocol",
        "",
        (
            "This standalone additive diagnostic evaluates the primary "
            f"$N={N_QUBITS}$ collective-loss Liouvillian at "
            f"$h=\\Delta t={H:g}$ with the matched local-loss Frobenius budget. "
            f"It uses {len(WASHOUT_CONTROL_SEEDS)} fixed Hamiltonian instances "
            "from the existing washout-control seed set and "
            f"{len(S_GRID)} constant inputs $s=0,0.05,\\ldots,1$."
        ),
        "",
        (
            f"For every one of the {aggregate['row_count']} seed/input cases, "
            "the complete 1,024-eigenvalue dense spectrum and right "
            "eigenvectors were computed. Six predeclared cases were also "
            "checked with an independently assembled sparse Liouvillian and "
            "shift-invert eigensolves near zero and the first decay mode."
        ),
        "",
        f"**Interpretation boundary.** {CAVEAT}",
        "",
        "## Aggregate result",
        "",
        (
            f"- Unique stationary mode: "
            f"{aggregate['unique_stationary_mode_rows']}/"
            f"{aggregate['row_count']} sampled Liouvillians."
        ),
        (
            f"- Minimum sampled decay gap: {minimum['gap']:.12g} "
            f"(seed {minimum['seed']}, $s={minimum['s']:.2f}$)."
        ),
        (
            f"- Maximum sampled decay gap: {maximum['gap']:.12g} "
            f"(seed {maximum['seed']}, $s={maximum['s']:.2f}$)."
        ),
        f"- Mean sampled decay gap: {aggregate['mean_sampled_gap']:.12g}.",
        (
            "- Maximum positive-real-part leakage: "
            f"{aggregate['maximum_positive_real_part_leakage']:.3e}."
        ),
        (
            "- Maximum dense all-mode relative eigenpair residual: "
            f"{aggregate['maximum_dense_relative_residual']:.3e}."
        ),
        (
            "- Maximum trace-preservation relative residual: "
            f"{aggregate['maximum_trace_preservation_relative_residual']:.3e}."
        ),
        (
            "- Maximum relative jump-budget error: "
            f"{aggregate['maximum_relative_jump_budget_error']:.3e}."
        ),
        "",
        "## Per-seed sampled interval",
        "",
        "| seed | unique/grid | min gap (s) | max gap (s) | mean gap |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate["per_seed"]:
        lines.append(
            f"| {item['seed']} | "
            f"{item['unique_stationary_mode_rows']}/{item['grid_rows']} | "
            f"{item['minimum_gap']['gap']:.9g} "
            f"({item['minimum_gap']['s']:.2f}) | "
            f"{item['maximum_gap']['gap']:.9g} "
            f"({item['maximum_gap']['s']:.2f}) | "
            f"{item['mean_gap']:.9g} |"
        )
    lines.extend(
        [
            "",
            "## Dense/sparse cross-checks",
            "",
            (
                f"- Cases checked: {aggregate['sparse_crosscheck_count']}; "
                "all found exactly one stationary mode among the near-zero "
                "shift-invert modes."
            ),
            (
                "- Maximum dense/sparse Liouvillian entry difference: "
                f"{aggregate['maximum_sparse_matrix_abs_difference']:.3e}."
            ),
            (
                "- Maximum matched dense/sparse eigenvalue difference: "
                f"{aggregate['maximum_sparse_dense_eigenvalue_abs_difference']:.3e}."
            ),
            (
                "- Maximum sparse relative eigenpair residual: "
                f"{aggregate['maximum_sparse_relative_residual']:.3e}."
            ),
            "",
            "## Integrity and reproduction",
            "",
            f"- Protocol SHA-256: `{aggregate['protocol_sha256']}`",
            f"- Protocol-file SHA-256: `{aggregate['protocol_file_sha256']}`",
            f"- Raw scientific-payload SHA-256: `{aggregate['raw_payload_sha256']}`",
            f"- Raw-file SHA-256: `{aggregate['raw_file_sha256']}`",
            "",
            "```bash",
            (
                "PYTHONPATH=src:experiments python "
                "experiments/run_collective_loss_full_input_diagnostic.py verify"
            ),
            (
                "PYTHONPATH=src:experiments python "
                "experiments/run_collective_loss_full_input_diagnostic.py "
                "run --workers 4 --recompute"
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def verify_artifacts() -> dict:
    for path in (PROTOCOL_PATH, RAW_PATH, AGGREGATE_PATH, REPORT_PATH):
        if not path.is_file():
            raise RuntimeError(f"required diagnostic artifact is missing: {path}")

    frozen = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    expected_protocol = protocol_dict()
    if frozen.get("protocol") != expected_protocol:
        raise RuntimeError("frozen protocol does not match current scientific sources")
    expected_protocol_hash = _sha256_json(expected_protocol)
    if frozen.get("protocol_sha256") != expected_protocol_hash:
        raise RuntimeError("frozen protocol hash is invalid")

    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if raw.get("protocol_sha256") != expected_protocol_hash:
        raise RuntimeError("raw spectrum references the wrong protocol")
    expected_aggregate = build_aggregate(
        raw,
        protocol_file_sha256=_sha256_file(PROTOCOL_PATH),
        raw_file_sha256=_sha256_file(RAW_PATH),
    )
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    if aggregate != expected_aggregate:
        raise RuntimeError("aggregate does not match raw spectrum")
    expected_report = render_report(aggregate)
    if REPORT_PATH.read_text(encoding="utf-8") != expected_report:
        raise RuntimeError("report does not match verified aggregate")
    return aggregate


def run_diagnostic(workers: int, recompute: bool) -> tuple[dict, bool]:
    frozen, protocol_file_hash = freeze_protocol()
    if RAW_PATH.exists() and not recompute:
        return verify_artifacts(), False

    prior_raw = (
        json.loads(RAW_PATH.read_text(encoding="utf-8"))
        if RAW_PATH.exists()
        else None
    )
    raw = compute_raw_payload(frozen["protocol_sha256"], workers)
    if prior_raw is not None and _canonical_json(prior_raw) != _canonical_json(raw):
        raise RuntimeError("deterministic recomputation disagrees with stored raw data")
    if prior_raw is None:
        _atomic_write_json(RAW_PATH, raw)
    raw_file_hash = _sha256_file(RAW_PATH)
    aggregate = build_aggregate(
        raw,
        protocol_file_sha256=protocol_file_hash,
        raw_file_sha256=raw_file_hash,
    )
    _atomic_write_json(AGGREGATE_PATH, aggregate)
    _atomic_write_text(REPORT_PATH, render_report(aggregate))
    verify_artifacts()
    return aggregate, prior_raw is not None


def _print_summary(aggregate: dict, wall_runtime: float | None = None) -> None:
    minimum = aggregate["minimum_sampled_gap"]
    maximum = aggregate["maximum_sampled_gap"]
    print(
        f"PASS rows={aggregate['row_count']} "
        f"unique={aggregate['unique_stationary_mode_rows']} "
        f"min_gap={minimum['gap']:.17g}@seed{minimum['seed']},s={minimum['s']:.2f} "
        f"max_gap={maximum['gap']:.17g}@seed{maximum['seed']},s={maximum['s']:.2f} "
        f"max_leakage={aggregate['maximum_positive_real_part_leakage']:.3e} "
        f"max_dense_residual={aggregate['maximum_dense_relative_residual']:.3e}"
    )
    if wall_runtime is not None:
        print(f"wall_runtime_s={wall_runtime:.6f}")
    print(f"raw_file_sha256={aggregate['raw_file_sha256']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze", help="freeze/validate the declared protocol")
    run_parser = subparsers.add_parser("run", help="compute or verify the diagnostic")
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument(
        "--recompute",
        action="store_true",
        help="recompute all rows and fail unless they exactly match stored data",
    )
    subparsers.add_parser("verify", help="verify hashes, grid, results, and report")
    args = parser.parse_args(argv)

    if args.command == "freeze":
        payload, file_hash = freeze_protocol()
        print(f"protocol_sha256={payload['protocol_sha256']}")
        print(f"protocol_file_sha256={file_hash}")
        return 0
    if args.command == "verify":
        start = time.perf_counter()
        aggregate = verify_artifacts()
        _print_summary(aggregate, time.perf_counter() - start)
        return 0
    if args.command == "run":
        start = time.perf_counter()
        aggregate, deterministic_match = run_diagnostic(
            args.workers,
            args.recompute,
        )
        _print_summary(aggregate, time.perf_counter() - start)
        if args.recompute:
            print(f"deterministic_recompute_match={str(deterministic_match).lower()}")
        return 0
    raise RuntimeError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

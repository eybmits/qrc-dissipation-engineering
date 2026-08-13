"""Prospective out-of-family validation for jump-family prediction.

This script tests a continuous, matched-budget interpolation that was not part of
the original channel benchmark:

    D_alpha = (1-alpha) D_local + alpha D_collective,  alpha in [0, 1].

The workflow is deliberately split into three commands:

1. ``freeze`` computes *static* Liouvillian diagnostics and writes a frozen
   prediction artifact.  It never evaluates a task score.
2. ``score`` refuses to run without that artifact, then evaluates held-out STM
   using paired Hamiltonian/input seeds and explicit train/validation/test splits.
3. ``report`` aggregates the raw checkpoints and evaluates the frozen criteria.
4. ``archive`` validates and packages source, checkpoints, results, report, and
   a machine-readable provenance manifest.

This is an internal prospective freeze (diagnostics precede scores and are
hash-linked), not a claim of an externally registered study.  Heavy work is
guarded under ``if __name__ == "__main__"``.
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
import itertools
import json
import math
import platform
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import _paths  # noqa: F401
import numpy as np
import scipy
from scipy import stats

from _paths import REPORTS_DIR, RESULTS_DIR
from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.reservoirs import (
    LindbladReservoir,
    ising_xx_hamiltonian,
    transverse_drive,
)
from qrc.sparse_evolve import SparseLindbladReservoir


PROTOCOL_VERSION = "quantum-strengthening-v2-2026-07-23"
GAMMA = 1.0
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
RIDGES = (0.0, 1e-10, 1e-8, 1e-6, 1e-4)
DEFAULT_INPUT = 0.5
ZERO_TOL = 1e-9
BUDGET_RTOL = 1e-10
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "experiments/run_quantum_strengthening.py",
    "tests/test_quantum_strengthening.py",
    "requirements.txt",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)


@dataclass(frozen=True)
class Preset:
    name: str
    n_qubits: tuple[int, ...]
    n_seeds: int
    wash: int
    train: int
    validation: int
    test: int
    delays: tuple[int, ...]
    h: float = 0.5
    dt: float = 0.5

    @property
    def total_len(self) -> int:
        return self.wash + self.train + self.validation + self.test


PRESETS = {
    "smoke": Preset(
        "smoke", (3,), 2, wash=20, train=50, validation=30, test=40,
        delays=tuple(range(1, 6)),
    ),
    "validation": Preset(
        "validation", (4,), 12, wash=100, train=300, validation=150, test=250,
        delays=tuple(range(1, 16)),
    ),
    "paper": Preset(
        "paper", (4, 5), 20, wash=200, train=600, validation=300, test=400,
        delays=tuple(range(1, 21)),
    ),
}


def deterministic_seeds(n: int) -> list[int]:
    """Nested seed pool shared with the definitive protocol."""
    return [
        int(x)
        for x in np.random.default_rng(2024).integers(0, 2**31 - 1, n)
    ]


def build_interpolated_jumps(
    n_qubits: int,
    alpha: float,
    target_strength: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Build the local-to-collective interpolation at exactly matched budget.

    The endpoints reproduce the existing local and collective families.  At an
    intermediate ``alpha``, fractions ``1-alpha`` and ``alpha`` of the same
    Frobenius jump-strength budget are assigned to the two channels.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if target_strength is None:
        target_strength = dsp.jump_strength(dsp.local_loss(n_qubits, GAMMA))
    if target_strength <= 0:
        raise ValueError("target_strength must be positive")

    jumps: list[tuple[np.ndarray, float]] = []
    if alpha < 1.0:
        jumps.extend(
            dsp.normalize_jump_strength(
                dsp.local_loss(n_qubits, GAMMA),
                (1.0 - alpha) * target_strength,
            )
        )
    if alpha > 0.0:
        jumps.extend(
            dsp.normalize_jump_strength(
                dsp.collective_loss(n_qubits, GAMMA),
                alpha * target_strength,
            )
        )
    return jumps


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty_status() -> list[str]:
    """Return porcelain-v1 lines so provenance records uncommitted inputs."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ["git-status-unavailable"]
    return [line for line in result.stdout.splitlines() if line]


def source_environment_manifest() -> dict:
    """Hash every scientific source input and identify the numerical runtime."""
    files: dict[str, str] = {}
    missing: list[str] = []
    for relative in SOURCE_FILES:
        path = REPO_ROOT / relative
        if path.is_file():
            files[relative] = _sha256_file(path)
        else:
            missing.append(relative)
    if missing:
        raise RuntimeError(f"provenance source files missing: {missing}")
    return {
        "files_sha256": files,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            # Record the runtime identity without leaking a machine-local path.
            "executable": Path(sys.executable).name,
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "platform": platform.platform(),
    }


def protocol_dict(preset: Preset) -> dict:
    """The immutable protocol payload hashed into every stage."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "preset": preset.name,
        "n_qubits": list(preset.n_qubits),
        "seeds": deterministic_seeds(preset.n_seeds),
        "alphas": list(ALPHAS),
        "matched_budget": "sum_k rate_k Tr(L_k^dag L_k) equals local loss at gamma=1",
        "interpolation": "(1-alpha) D_local + alpha D_collective",
        "couplings": {
            "topology": "complete undirected graph without self-loops",
            "distribution": (
                "independent upper-triangle J_ij ~ Uniform[-1,1], reflected "
                "symmetrically; diagonal exactly zero"
            ),
            "rng": "numpy.random.default_rng(seed)",
        },
        "input": {
            "distribution": "independent s_k ~ Uniform[0,1]",
            "rng": (
                "same seeded Generator used after drawing the Hamiltonian "
                "couplings, identically for every alpha in a paired seed"
            ),
        },
        "initial_state": "|0...0><0...0|",
        "hamiltonian": (
            "H(s)=sum_{i<j} J_ij X_i X_j + h sum_i Z_i "
            "+ h(1+s) sum_i X_i"
        ),
        "drive_construction": (
            "H_static=ising_xx_hamiltonian(J,h,N)+h*sum_i X_i; "
            "H_drive=h*sum_i X_i"
        ),
        "h": preset.h,
        "dt": preset.dt,
        "diagnostic_input": DEFAULT_INPUT,
        "zero_tolerance": ZERO_TOL,
        "slow_mode_definition": (
            "nonstationary Liouvillian mode retaining at least exp(-1) amplitude "
            "over max(delays) input steps"
        ),
        "task": "short-term memory",
        "observables": (
            "all weight-1 X,Y,Z and same-axis weight-2 XX,YY,ZZ Pauli "
            "observables; one sample per input step; linear readout with bias"
        ),
        "feature_count": "3*N + 3*binomial(N,2)",
        "capacity": (
            "per-delay squared Pearson correlation between target and prediction; "
            "STM MC is the sum over predefined delays"
        ),
        "backend": (
            "exact sparse Lindblad propagation using scipy.sparse.linalg."
            "expm_multiply; dense numpy.linalg.eigvals for frozen diagnostics"
        ),
        "wash": preset.wash,
        "train": preset.train,
        "validation": preset.validation,
        "test": preset.test,
        "delays": list(preset.delays),
        "ridge_grid": list(RIDGES),
        "ridge_selection": (
            "maximize summed validation capacity after training on train; "
            "retrain on train+validation; evaluate test once"
        ),
        "paired_design": (
            "same Hamiltonian and input sequence for every alpha within N and seed"
        ),
        "budget_relative_tolerance": BUDGET_RTOL,
        "source_environment": source_environment_manifest(),
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if len(array) < 2:
        return mean, 0.0
    return mean, float(np.std(array, ddof=1) / np.sqrt(len(array)))


def _alpha_tag(alpha: float) -> str:
    return f"a{int(round(100 * alpha)):03d}"


def _prediction_path(outdir: Path) -> Path:
    return outdir / "frozen_diagnostic_predictions.json"


def _prediction_seal_path(outdir: Path) -> Path:
    return outdir / "frozen_diagnostic_predictions.sha256"


def _task_dir(outdir: Path) -> Path:
    return outdir / "task_jobs"


def _task_path(outdir: Path, n_qubits: int, alpha: float, seed: int) -> Path:
    return _task_dir(outdir) / (
        f"stm_N{n_qubits}_{_alpha_tag(alpha)}_seed{seed}.json"
    )


def _require_finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    return number


def _validate_task_checkpoint(
    row: dict,
    *,
    preset: Preset,
    protocol_sha256: str,
    frozen_sha256: str,
    source_environment_sha256: str,
    n_qubits: int,
    alpha: float,
    seed: int,
) -> None:
    """Strict, central validation used for both reuse and aggregation."""
    identity = f"N={n_qubits}, alpha={alpha}, seed={seed}"
    expected_scalars = {
        "protocol_sha256": protocol_sha256,
        "frozen_prediction_sha256": frozen_sha256,
        "source_environment_sha256": source_environment_sha256,
        "backend": "exact_sparse_expm_multiply",
    }
    for key, expected in expected_scalars.items():
        if row.get(key) != expected:
            raise RuntimeError(f"{identity}: invalid {key}")
    if type(row.get("N")) is not int or row["N"] != n_qubits:
        raise RuntimeError(f"{identity}: invalid N")
    if type(row.get("seed")) is not int or row["seed"] != seed:
        raise RuntimeError(f"{identity}: invalid seed")
    if not math.isclose(
        _require_finite(row.get("alpha"), f"{identity} alpha"),
        alpha,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"{identity}: invalid alpha")

    target = _require_finite(row.get("target_strength"), f"{identity} target")
    actual = _require_finite(row.get("jump_strength"), f"{identity} budget")
    relative = _require_finite(
        row.get("relative_budget_error"), f"{identity} budget error"
    )
    if target <= 0.0 or actual <= 0.0:
        raise RuntimeError(f"{identity}: non-positive dissipative budget")
    computed = abs(actual - target) / target
    if relative < 0.0 or relative > BUDGET_RTOL:
        raise RuntimeError(f"{identity}: budget error exceeds tolerance")
    if not math.isclose(relative, computed, rel_tol=1e-7, abs_tol=1e-15):
        raise RuntimeError(f"{identity}: inconsistent budget error")

    expected_features = 3 * n_qubits + 3 * math.comb(n_qubits, 2)
    if type(row.get("n_features")) is not int or row["n_features"] != expected_features:
        raise RuntimeError(f"{identity}: invalid observable feature count")
    selected_ridge = _require_finite(
        row.get("selected_ridge"), f"{identity} selected ridge"
    )
    if not any(math.isclose(selected_ridge, ridge) for ridge in RIDGES):
        raise RuntimeError(f"{identity}: selected ridge is outside frozen grid")
    for key in ("validation_mc", "test_mc", "runtime_s"):
        value = _require_finite(row.get(key), f"{identity} {key}")
        if key != "runtime_s" and value < 0.0:
            raise RuntimeError(f"{identity}: negative {key}")

    for key in ("validation_capacity_by_delay", "test_capacity_by_delay"):
        values = row.get(key)
        if not isinstance(values, list) or len(values) != len(preset.delays):
            raise RuntimeError(f"{identity}: invalid {key} delay count")
        for index, value in enumerate(values):
            _require_finite(value, f"{identity} {key}[{index}]")
    ridge_scores = row.get("validation_mc_by_ridge")
    expected_ridge_keys = {f"{ridge:.12g}" for ridge in RIDGES}
    if not isinstance(ridge_scores, dict) or set(ridge_scores) != expected_ridge_keys:
        raise RuntimeError(f"{identity}: invalid validation ridge grid")
    for key, value in ridge_scores.items():
        _require_finite(value, f"{identity} ridge score {key}")


def _executor(workers: int):
    """Prefer process isolation, with a thread fallback for restricted runtimes.

    Some managed macOS sandboxes deny ``sysconf(SC_SEM_NSEMS_MAX)`` while
    constructing ``ProcessPoolExecutor``.  NumPy/SciPy release the GIL for the
    expensive eigensolver and sparse-exponential kernels, so a bounded thread
    pool remains a useful and scientifically identical fallback.
    """
    try:
        return ProcessPoolExecutor(max_workers=workers), "process"
    except (OSError, PermissionError) as exc:
        print(
            f"process executor unavailable ({type(exc).__name__}: {exc}); "
            "using bounded threads",
            flush=True,
        )
        return ThreadPoolExecutor(max_workers=workers), "thread"


def _spectrum_diagnostics(
    eigenvalues: np.ndarray,
    dt: float,
    max_delay: int,
) -> dict:
    """Summarize decay modes once the full dense spectrum has been computed."""
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    nonstationary = eigenvalues[np.abs(eigenvalues) > ZERO_TOL]
    if nonstationary.size == 0:
        return {
            "spectral_gap": 0.0,
            "slow_mode_count": 0,
            "retained_mode_mass": 0.0,
            "n_nonstationary_modes": 0,
            "slow_decay_threshold": 1.0 / (dt * max_delay),
            "max_positive_real_part": 0.0,
        }

    # Lindbladians have Re(lambda)<=0; clip tiny numerical positivity only when
    # converting to a decay rate, while recording it as a numerical audit.
    decay_rates = np.maximum(-nonstationary.real, 0.0)
    horizon = float(dt * max_delay)
    threshold = 1.0 / horizon
    return {
        "spectral_gap": float(np.min(decay_rates)),
        "slow_mode_count": int(np.count_nonzero(decay_rates <= threshold)),
        "retained_mode_mass": float(np.sum(np.exp(-decay_rates * horizon))),
        "n_nonstationary_modes": int(nonstationary.size),
        "slow_decay_threshold": float(threshold),
        "max_positive_real_part": float(
            max(0.0, np.max(nonstationary.real))
        ),
    }


def diagnostic_job(args: tuple[int, float, int, Preset]) -> dict:
    """Static diagnostics only.  No input sequence or task target is constructed."""
    n_qubits, alpha, seed, preset = args
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    J = res.random_couplings(n_qubits, 1.0, rng)
    target = dsp.jump_strength(dsp.local_loss(n_qubits, GAMMA))
    jumps = build_interpolated_jumps(n_qubits, alpha, target)
    H0 = ising_xx_hamiltonian(J, preset.h, n_qubits)
    Hx = transverse_drive(n_qubits)
    reservoir = LindbladReservoir.from_terms(
        n_qubits,
        H0 + preset.h * Hx,
        preset.h * Hx,
        jumps,
        preset.dt,
        cache_propagators=False,
        quantize=None,
    )
    eigenvalues = np.linalg.eigvals(reservoir.liouvillian(DEFAULT_INPUT))
    spectral = _spectrum_diagnostics(
        eigenvalues, preset.dt, max(preset.delays)
    )
    return {
        "N": n_qubits,
        "alpha": alpha,
        "seed": seed,
        "jump_strength": dsp.jump_strength(jumps),
        "target_strength": target,
        "relative_budget_error": abs(dsp.jump_strength(jumps) - target) / target,
        "unitality_defect": float(
            np.linalg.norm(
                sum(
                    rate
                    * (
                        L @ L.conj().T
                        - L.conj().T @ L
                    )
                    for L, rate in jumps
                ),
                ord="fro",
            )
        ),
        **spectral,
        "runtime_s": time.perf_counter() - started,
    }


def _aggregate_diagnostics(rows: Sequence[dict], preset: Preset) -> dict:
    by_n: dict[str, dict] = {}
    for n_qubits in preset.n_qubits:
        alpha_stats: list[dict] = []
        for alpha in ALPHAS:
            group = [
                row
                for row in rows
                if row["N"] == n_qubits and math.isclose(row["alpha"], alpha)
            ]
            if len(group) != preset.n_seeds:
                raise RuntimeError(
                    f"incomplete diagnostics for N={n_qubits}, alpha={alpha}: "
                    f"{len(group)}/{preset.n_seeds}"
                )
            gap_mean, gap_se = _mean_se([row["spectral_gap"] for row in group])
            count_mean, count_se = _mean_se(
                [row["slow_mode_count"] for row in group]
            )
            mass_mean, mass_se = _mean_se(
                [row["retained_mode_mass"] for row in group]
            )
            alpha_stats.append(
                {
                    "alpha": alpha,
                    "spectral_gap_mean": gap_mean,
                    "spectral_gap_se": gap_se,
                    "slow_mode_count_mean": count_mean,
                    "slow_mode_count_se": count_se,
                    "retained_mode_mass_mean": mass_mean,
                    "retained_mode_mass_se": mass_se,
                }
            )

        gap_rank = [
            item["alpha"]
            for item in sorted(
                alpha_stats,
                key=lambda item: (item["spectral_gap_mean"], -item["alpha"]),
            )
        ]
        mass_rank = [
            item["alpha"]
            for item in sorted(
                alpha_stats,
                key=lambda item: (-item["retained_mode_mass_mean"], -item["alpha"]),
            )
        ]
        intermediate = [
            item for item in alpha_stats if 0.0 < item["alpha"] < 1.0
        ]
        selected = min(
            intermediate,
            key=lambda item: (
                item["spectral_gap_mean"],
                -item["retained_mode_mass_mean"],
            ),
        )["alpha"]
        by_n[str(n_qubits)] = {
            "diagnostic_summary": alpha_stats,
            "frozen_gap_rank_best_to_worst": gap_rank,
            "frozen_retained_mass_rank_best_to_worst": mass_rank,
            "diagnostic_selected_intermediate_alpha": selected,
            "primary_prediction": (
                "mean held-out STM capacity ranks inversely with the frozen "
                "mean spectral gap across alpha"
            ),
            "secondary_prediction": (
                "mean held-out STM capacity ranks with retained-mode mass"
            ),
        }
    return by_n


def freeze_predictions(preset: Preset, outdir: Path, workers: int) -> Path:
    """Compute diagnostics, freeze predictions, and write no task scores."""
    protocol = protocol_dict(preset)
    fingerprint = _sha256_json(protocol)
    path = _prediction_path(outdir)
    if path.exists():
        _load_frozen(preset, outdir)
        print(f"prediction artifact already frozen: {path}")
        return path
    if _task_dir(outdir).exists() and any(_task_dir(outdir).glob("*.json")):
        raise RuntimeError("task scores already exist; refusing a retrospective freeze")

    jobs = [
        (n_qubits, alpha, seed, preset)
        for n_qubits in preset.n_qubits
        for alpha in ALPHAS
        for seed in deterministic_seeds(preset.n_seeds)
    ]
    rows: list[dict] = []
    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            row = diagnostic_job(job)
            rows.append(row)
            print(
                f"diagnostic {index}/{len(jobs)} N={row['N']} "
                f"alpha={row['alpha']:.1f} seed={row['seed']} "
                f"gap={row['spectral_gap']:.6g}",
                flush=True,
            )
    else:
        executor, executor_kind = _executor(workers)
        print(f"diagnostic executor={executor_kind} workers={workers}", flush=True)
        with executor as pool:
            futures = {pool.submit(diagnostic_job, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows.append(row)
                print(
                    f"diagnostic {index}/{len(jobs)} N={row['N']} "
                    f"alpha={row['alpha']:.1f} seed={row['seed']} "
                    f"gap={row['spectral_gap']:.6g}",
                    flush=True,
                )
    rows.sort(key=lambda row: (row["N"], row["alpha"], row["seed"]))
    predictions = _aggregate_diagnostics(rows, preset)
    payload = {
        "artifact_type": "frozen_diagnostic_predictions",
        "status": "frozen_before_task_scores",
        "created_utc": _utc_now(),
        "git_head": _git_head(),
        "protocol": protocol,
        "protocol_sha256": fingerprint,
        "source_environment_sha256": _sha256_json(
            protocol["source_environment"]
        ),
        "task_scores_present_at_freeze": False,
        "prediction_rule_was_fixed_before_scores": True,
        "predictions_by_N": predictions,
        "success_criteria": {
            "primary_rank_validation": (
                "within each N, Spearman rho between -frozen mean gap and mean "
                "held-out test STM across the six alpha values is at least 0.8"
            ),
            "selected_candidate_validation": (
                "the diagnostic-selected intermediate alpha has positive paired "
                "mean test-STM advantage over alpha=0 and a 95% paired t interval "
                "whose lower bound exceeds zero"
            ),
            "interpretation": (
                "passing supports prospective validity within this interpolation "
                "and operating point; it does not prove global "
                "optimality or Hamiltonian/task universality"
            ),
        },
        "git_dirty_status_at_freeze": _git_dirty_status(),
        "diagnostic_rows": rows,
    }
    _atomic_write_json(path, payload)
    frozen_hash = _sha256_file(path)
    _prediction_seal_path(outdir).write_text(f"{frozen_hash}  {path.name}\n")
    print(f"FROZEN {path} sha256={frozen_hash}", flush=True)
    return path


def _split_masks(length: int, preset: Preset) -> dict[str, np.ndarray]:
    expected = preset.train + preset.validation + preset.test
    if length != expected:
        raise ValueError(f"expected {expected} post-wash rows, got {length}")
    train = np.zeros(length, dtype=bool)
    validation = np.zeros(length, dtype=bool)
    test = np.zeros(length, dtype=bool)
    train[: preset.train] = True
    validation[preset.train : preset.train + preset.validation] = True
    test[preset.train + preset.validation :] = True
    return {"train": train, "validation": validation, "test": test}


def held_out_stm_score(
    features: np.ndarray,
    full_inputs: np.ndarray,
    preset: Preset,
) -> dict:
    """Select one ridge on validation, then evaluate the untouched test split."""
    post_inputs = np.asarray(full_inputs[preset.wash :], dtype=float)
    masks = _split_masks(len(post_inputs), preset)
    Xb = readout.add_bias(np.asarray(features, dtype=float))

    validation_by_ridge: dict[str, float] = {}
    validation_delay_by_ridge: dict[str, list[float]] = {}
    for ridge in RIDGES:
        delay_scores: list[float] = []
        for delay in preset.delays:
            y = tasks.delayed_target(full_inputs, delay)[preset.wash :]
            train_mask = masks["train"] & np.isfinite(y)
            validation_mask = masks["validation"] & np.isfinite(y)
            weights = readout.train_readout(
                Xb[train_mask], y[train_mask], ridge=ridge
            )
            delay_scores.append(
                readout.capacity(
                    y[validation_mask],
                    readout.predict(Xb[validation_mask], weights),
                )
            )
        key = f"{ridge:.12g}"
        validation_by_ridge[key] = float(np.sum(delay_scores))
        validation_delay_by_ridge[key] = delay_scores

    # RIDGES is ordered from least to most regularized, providing a deterministic
    # tie break without consulting the test split.
    chosen_ridge = max(
        RIDGES,
        key=lambda ridge: validation_by_ridge[f"{ridge:.12g}"],
    )
    chosen_key = f"{chosen_ridge:.12g}"
    test_delay_scores: list[float] = []
    train_validation_mask = masks["train"] | masks["validation"]
    for delay in preset.delays:
        y = tasks.delayed_target(full_inputs, delay)[preset.wash :]
        fit_mask = train_validation_mask & np.isfinite(y)
        test_mask = masks["test"] & np.isfinite(y)
        weights = readout.train_readout(
            Xb[fit_mask], y[fit_mask], ridge=chosen_ridge
        )
        test_delay_scores.append(
            readout.capacity(
                y[test_mask], readout.predict(Xb[test_mask], weights)
            )
        )

    return {
        "selected_ridge": chosen_ridge,
        "validation_mc": validation_by_ridge[chosen_key],
        "test_mc": float(np.sum(test_delay_scores)),
        "validation_capacity_by_delay": validation_delay_by_ridge[chosen_key],
        "test_capacity_by_delay": test_delay_scores,
        "validation_mc_by_ridge": validation_by_ridge,
    }


def task_job(args: tuple[int, float, int, Preset]) -> dict:
    """One paired held-out STM evaluation for a fixed N, alpha, and seed."""
    n_qubits, alpha, seed, preset = args
    started = time.perf_counter()
    problem_rng = np.random.default_rng(seed)
    J = res.random_couplings(n_qubits, 1.0, problem_rng)
    inputs = tasks.stm_inputs(preset.total_len, problem_rng)

    target = dsp.jump_strength(dsp.local_loss(n_qubits, GAMMA))
    jumps = build_interpolated_jumps(n_qubits, alpha, target)
    H0 = ising_xx_hamiltonian(J, preset.h, n_qubits)
    Hx = transverse_drive(n_qubits)
    reservoir = SparseLindbladReservoir.from_terms(
        n_qubits,
        H0 + preset.h * Hx,
        preset.h * Hx,
        jumps,
        preset.dt,
    )
    observables = readout.pauli_observables(n_qubits, max_weight=2)
    features = reservoir.run(inputs, observables, washout=preset.wash)
    scores = held_out_stm_score(features, inputs, preset)
    return {
        "N": n_qubits,
        "alpha": alpha,
        "seed": seed,
        "backend": "exact_sparse_expm_multiply",
        "jump_strength": dsp.jump_strength(jumps),
        "target_strength": target,
        "relative_budget_error": abs(dsp.jump_strength(jumps) - target) / target,
        "n_features": int(features.shape[1]),
        **scores,
        "runtime_s": time.perf_counter() - started,
    }


def _load_frozen(preset: Preset, outdir: Path) -> tuple[Path, dict]:
    path = _prediction_path(outdir)
    if not path.exists():
        raise RuntimeError(
            f"missing {path}; run the freeze command before any task scoring"
        )
    frozen = json.loads(path.read_text())
    expected_protocol = protocol_dict(preset)
    expected = _sha256_json(expected_protocol)
    if frozen.get("protocol") != expected_protocol:
        raise RuntimeError("frozen protocol payload does not match this source/runtime")
    if frozen.get("protocol_sha256") != expected:
        raise RuntimeError("frozen prediction artifact does not match this preset")
    if _sha256_json(frozen["protocol"]) != frozen["protocol_sha256"]:
        raise RuntimeError("frozen protocol payload/hash mismatch")
    expected_source = _sha256_json(expected_protocol["source_environment"])
    if frozen.get("source_environment_sha256") != expected_source:
        raise RuntimeError("frozen source/environment hash mismatch")
    if frozen.get("status") != "frozen_before_task_scores":
        raise RuntimeError("prediction artifact is not marked as prospectively frozen")
    seal = _prediction_seal_path(outdir)
    if not seal.exists():
        raise RuntimeError("frozen prediction seal is missing")
    seal_fields = seal.read_text().strip().split()
    if len(seal_fields) != 2 or seal_fields[1] != path.name:
        raise RuntimeError("frozen prediction seal is malformed")
    if seal_fields[0] != _sha256_file(path):
        raise RuntimeError("frozen prediction artifact changed after sealing")

    expected_keys = {
        (n_qubits, alpha, seed)
        for n_qubits in preset.n_qubits
        for alpha in ALPHAS
        for seed in deterministic_seeds(preset.n_seeds)
    }
    observed_keys: set[tuple[int, float, int]] = set()
    rows = frozen.get("diagnostic_rows")
    if not isinstance(rows, list):
        raise RuntimeError("frozen diagnostic rows are missing")
    for row in rows:
        try:
            raw_n = row["N"]
            raw_alpha = row["alpha"]
            raw_seed = row["seed"]
        except KeyError as exc:
            raise RuntimeError("malformed frozen diagnostic identity") from exc
        if type(raw_n) is not int or type(raw_seed) is not int:
            raise RuntimeError("frozen diagnostic N/seed must be integers")
        alpha_value = _require_finite(raw_alpha, "frozen diagnostic alpha")
        if not any(
            math.isclose(alpha_value, alpha, rel_tol=0.0, abs_tol=1e-12)
            for alpha in ALPHAS
        ):
            raise RuntimeError("frozen diagnostic alpha is outside protocol")
        key = (raw_n, alpha_value, raw_seed)
        if key in observed_keys:
            raise RuntimeError(f"duplicate frozen diagnostic row {key}")
        observed_keys.add(key)
        for field in (
            "jump_strength",
            "target_strength",
            "relative_budget_error",
            "unitality_defect",
            "spectral_gap",
            "slow_mode_count",
            "retained_mode_mass",
            "n_nonstationary_modes",
            "slow_decay_threshold",
            "max_positive_real_part",
            "runtime_s",
        ):
            _require_finite(row.get(field), f"frozen diagnostic {key} {field}")
        if float(row["relative_budget_error"]) > BUDGET_RTOL:
            raise RuntimeError(f"frozen diagnostic {key} violates budget")
    if observed_keys != expected_keys:
        raise RuntimeError("frozen diagnostic grid is incomplete or unexpected")
    return path, frozen


def score_tasks(preset: Preset, outdir: Path, workers: int) -> Path:
    """Run/reuse checkpointed task jobs only after validating the frozen artifact."""
    frozen_path, frozen = _load_frozen(preset, outdir)
    frozen_hash = _sha256_file(frozen_path)
    source_hash = frozen["source_environment_sha256"]
    metadata_path = outdir / "task_stage_metadata.json"
    if not metadata_path.exists():
        _atomic_write_json(
            metadata_path,
            {
                "artifact_type": "task_stage_metadata",
                "created_utc": _utc_now(),
                "git_head": _git_head(),
                "protocol_sha256": frozen["protocol_sha256"],
                "source_environment_sha256": source_hash,
                "frozen_prediction_file": frozen_path.name,
                "frozen_prediction_sha256": frozen_hash,
                "frozen_created_utc": frozen["created_utc"],
            },
        )
    else:
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("protocol_sha256") != frozen["protocol_sha256"]:
            raise RuntimeError("task metadata protocol hash mismatch")
        if metadata.get("frozen_prediction_sha256") != frozen_hash:
            raise RuntimeError(
                "frozen prediction file changed after task scoring began"
            )
        if metadata.get("source_environment_sha256") != source_hash:
            raise RuntimeError("task metadata source/environment hash mismatch")

    jobs = [
        (n_qubits, alpha, seed, preset)
        for n_qubits in preset.n_qubits
        for alpha in ALPHAS
        for seed in deterministic_seeds(preset.n_seeds)
    ]
    pending: list[tuple[int, float, int, Preset]] = []
    for job in jobs:
        n_qubits, alpha, seed, _ = job
        path = _task_path(outdir, n_qubits, alpha, seed)
        if path.exists():
            old = json.loads(path.read_text())
            _validate_task_checkpoint(
                old,
                preset=preset,
                protocol_sha256=frozen["protocol_sha256"],
                frozen_sha256=frozen_hash,
                source_environment_sha256=source_hash,
                n_qubits=n_qubits,
                alpha=alpha,
                seed=seed,
            )
            continue
        pending.append(job)

    print(
        f"task jobs total={len(jobs)} complete={len(jobs)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )

    def save_result(result: dict) -> None:
        result["protocol_sha256"] = frozen["protocol_sha256"]
        result["frozen_prediction_sha256"] = frozen_hash
        result["source_environment_sha256"] = source_hash
        _validate_task_checkpoint(
            result,
            preset=preset,
            protocol_sha256=frozen["protocol_sha256"],
            frozen_sha256=frozen_hash,
            source_environment_sha256=source_hash,
            n_qubits=int(result["N"]),
            alpha=float(result["alpha"]),
            seed=int(result["seed"]),
        )
        path = _task_path(outdir, result["N"], result["alpha"], result["seed"])
        _atomic_write_json(path, result)

    if workers <= 1:
        for index, job in enumerate(pending, 1):
            result = task_job(job)
            save_result(result)
            print(
                f"score {index}/{len(pending)} N={result['N']} "
                f"alpha={result['alpha']:.1f} seed={result['seed']} "
                f"test_MC={result['test_mc']:.6g} "
                f"{result['runtime_s']:.2f}s",
                flush=True,
            )
    else:
        executor, executor_kind = _executor(workers)
        print(f"task executor={executor_kind} workers={workers}", flush=True)
        with executor as pool:
            futures = {pool.submit(task_job, job): job for job in pending}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                save_result(result)
                print(
                    f"score {index}/{len(pending)} N={result['N']} "
                    f"alpha={result['alpha']:.1f} seed={result['seed']} "
                    f"test_MC={result['test_mc']:.6g} "
                    f"{result['runtime_s']:.2f}s",
                    flush=True,
                )

    aggregate_path = aggregate_results(preset, outdir)
    print(f"SCORED {aggregate_path}", flush=True)
    return aggregate_path


def _paired_interval(values: Sequence[float], confidence: float = 0.95) -> dict:
    array = np.asarray(values, dtype=float)
    mean, se = _mean_se(array)
    if len(array) < 2 or se == 0.0:
        low = high = mean
        t_stat = float("inf") if mean != 0 else 0.0
    else:
        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(array) - 1))
        low, high = mean - critical * se, mean + critical * se
        t_stat = mean / se
    return {
        "n": int(len(array)),
        "mean": mean,
        "se": se,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_t_stat": float(t_stat),
    }


def _paired_signflip_p(values: Sequence[float]) -> dict:
    """Two-sided paired sign-flip test, exact for up to 22 pairs."""
    array = np.asarray(values, dtype=float)
    n = len(array)
    observed = abs(float(np.mean(array)))
    if n == 0:
        return {"p_two_sided": float("nan"), "exact": False, "n_draws": 0}
    if n <= 22:
        total = 1 << n
        extreme = 0
        chunk = 16384
        bit_positions = np.arange(n, dtype=np.uint64)
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            numbers = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = 1.0 - 2.0 * ((numbers >> bit_positions) & 1).astype(float)
            permuted = np.abs(signs @ array / n)
            extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
        return {
            "p_two_sided": float(extreme / total),
            "exact": True,
            "n_draws": total,
        }

    rng = np.random.default_rng(20240723)
    draws = 1_000_000
    extreme = 0
    chunk = 10000
    for _ in range(draws // chunk):
        signs = rng.choice((-1.0, 1.0), size=(chunk, n))
        permuted = np.abs(signs @ array / n)
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return {
        "p_two_sided": float((extreme + 1) / (draws + 1)),
        "exact": False,
        "n_draws": draws,
    }


def _spearman_with_exact_p(
    predictor: Sequence[float],
    outcome: Sequence[float],
) -> dict:
    """Spearman rho with an exact two-sided permutation p for small grids."""
    predictor_array = np.asarray(predictor, dtype=float)
    outcome_array = np.asarray(outcome, dtype=float)
    observed = float(stats.spearmanr(predictor_array, outcome_array).statistic)
    n = len(outcome_array)
    if n <= 9:
        extreme = 0
        total = math.factorial(n)
        for permutation in itertools.permutations(outcome_array.tolist()):
            rho = float(
                stats.spearmanr(predictor_array, np.asarray(permutation)).statistic
            )
            extreme += int(abs(rho) >= abs(observed) - 1e-12)
        return {
            "rho": observed,
            "p_two_sided": float(extreme / total),
            "exact": True,
            "n_permutations": total,
        }
    asymptotic = stats.spearmanr(predictor_array, outcome_array)
    return {
        "rho": observed,
        "p_two_sided": float(asymptotic.pvalue),
        "exact": False,
        "n_permutations": 0,
    }


def _read_task_rows(
    preset: Preset,
    outdir: Path,
    frozen: dict,
    frozen_hash: str,
) -> list[dict]:
    rows: list[dict] = []
    missing: list[str] = []
    seen: set[tuple[int, float, int]] = set()
    for n_qubits in preset.n_qubits:
        for alpha in ALPHAS:
            for seed in deterministic_seeds(preset.n_seeds):
                path = _task_path(outdir, n_qubits, alpha, seed)
                if not path.exists():
                    missing.append(str(path))
                    continue
                row = json.loads(path.read_text())
                _validate_task_checkpoint(
                    row,
                    preset=preset,
                    protocol_sha256=frozen["protocol_sha256"],
                    frozen_sha256=frozen_hash,
                    source_environment_sha256=frozen[
                        "source_environment_sha256"
                    ],
                    n_qubits=n_qubits,
                    alpha=alpha,
                    seed=seed,
                )
                key = (n_qubits, alpha, seed)
                if key in seen:
                    raise RuntimeError(f"duplicate task checkpoint {key}")
                seen.add(key)
                rows.append(row)
    if missing:
        raise RuntimeError(
            f"{len(missing)} task checkpoints are missing; first: {missing[0]}"
        )
    expected_count = (
        len(preset.n_qubits) * len(ALPHAS) * preset.n_seeds
    )
    if len(seen) != expected_count:
        raise RuntimeError("task checkpoint pairing grid is incomplete")
    return rows


def aggregate_results(preset: Preset, outdir: Path) -> Path:
    frozen_path, frozen = _load_frozen(preset, outdir)
    frozen_hash = _sha256_file(frozen_path)
    rows = _read_task_rows(preset, outdir, frozen, frozen_hash)
    by_n: dict[str, dict] = {}

    for n_qubits in preset.n_qubits:
        task_summary: list[dict] = []
        score_by_alpha_seed: dict[float, dict[int, float]] = {}
        for alpha in ALPHAS:
            group = [
                row
                for row in rows
                if row["N"] == n_qubits and math.isclose(row["alpha"], alpha)
            ]
            scores = [float(row["test_mc"]) for row in group]
            validations = [float(row["validation_mc"]) for row in group]
            mean, se = _mean_se(scores)
            val_mean, val_se = _mean_se(validations)
            task_summary.append(
                {
                    "alpha": alpha,
                    "test_mc_mean": mean,
                    "test_mc_se": se,
                    "validation_mc_mean": val_mean,
                    "validation_mc_se": val_se,
                    "n_seeds": len(group),
                    "selected_ridge_counts": {
                        f"{ridge:.12g}": sum(
                            math.isclose(float(row["selected_ridge"]), ridge)
                            for row in group
                        )
                        for ridge in RIDGES
                    },
                }
            )
            score_by_alpha_seed[alpha] = {
                int(row["seed"]): float(row["test_mc"]) for row in group
            }

        frozen_n = frozen["predictions_by_N"][str(n_qubits)]
        diagnostics = frozen_n["diagnostic_summary"]
        gaps = np.asarray(
            [item["spectral_gap_mean"] for item in diagnostics], dtype=float
        )
        masses = np.asarray(
            [item["retained_mode_mass_mean"] for item in diagnostics], dtype=float
        )
        task_means = np.asarray(
            [item["test_mc_mean"] for item in task_summary], dtype=float
        )
        gap_corr = _spearman_with_exact_p(-gaps, task_means)
        mass_corr = _spearman_with_exact_p(masses, task_means)

        selected = float(frozen_n["diagnostic_selected_intermediate_alpha"])
        seeds = deterministic_seeds(preset.n_seeds)
        selected_diff = [
            score_by_alpha_seed[selected][seed] - score_by_alpha_seed[0.0][seed]
            for seed in seeds
        ]
        selected_stats = _paired_interval(selected_diff)
        selected_stats.update(_paired_signflip_p(selected_diff))
        local_mean = next(
            item["test_mc_mean"] for item in task_summary if item["alpha"] == 0.0
        )
        selected_stats.update(
            {
                "selected_alpha": selected,
                "reference_alpha": 0.0,
                "relative_mean_advantage_percent": (
                    100.0 * selected_stats["mean"] / local_mean
                ),
            }
        )

        endpoint_diff = [
            score_by_alpha_seed[1.0][seed] - score_by_alpha_seed[0.0][seed]
            for seed in seeds
        ]
        endpoint_stats = _paired_interval(endpoint_diff)
        endpoint_stats.update(_paired_signflip_p(endpoint_diff))
        endpoint_stats["relative_mean_advantage_percent"] = (
            100.0 * endpoint_stats["mean"] / local_mean
        )

        primary_pass = bool(float(gap_corr["rho"]) >= 0.8)
        selected_pass = bool(selected_stats["ci95_low"] > 0.0)
        by_n[str(n_qubits)] = {
            "task_summary": task_summary,
            "prospective_rank_tests": {
                "gap_spearman_rho": gap_corr["rho"],
                "gap_spearman_p_two_sided": gap_corr["p_two_sided"],
                "gap_spearman_exact": gap_corr["exact"],
                "gap_spearman_n_permutations": gap_corr["n_permutations"],
                "retained_mass_spearman_rho": mass_corr["rho"],
                "retained_mass_spearman_p_two_sided": mass_corr["p_two_sided"],
                "retained_mass_spearman_exact": mass_corr["exact"],
                "retained_mass_spearman_n_permutations": mass_corr[
                    "n_permutations"
                ],
            },
            "diagnostic_selected_intermediate_vs_local": selected_stats,
            "collective_endpoint_vs_local": endpoint_stats,
            "criteria": {
                "primary_rank_validation_pass": primary_pass,
                "selected_candidate_validation_pass": selected_pass,
                "both_pass": primary_pass and selected_pass,
            },
        }

    payload = {
        "artifact_type": "quantum_strengthening_results",
        "created_utc": _utc_now(),
        "git_head": _git_head(),
        "protocol": protocol_dict(preset),
        "protocol_sha256": frozen["protocol_sha256"],
        "source_environment_sha256": frozen["source_environment_sha256"],
        "frozen_prediction_file": frozen_path.name,
        "frozen_prediction_sha256": frozen_hash,
        "results_by_N": by_n,
        "raw_task_rows": rows,
    }
    path = outdir / "quantum_strengthening_results.json"
    _atomic_write_json(path, payload)
    return path


def _fmt(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def write_report(preset: Preset, outdir: Path, report_path: Path) -> Path:
    frozen_path, frozen = _load_frozen(preset, outdir)
    aggregate_path = aggregate_results(preset, outdir)
    aggregate = json.loads(aggregate_path.read_text())
    lines: list[str] = [
        "# Quantum-strengthening prospective interpolation test",
        "",
        f"**Protocol:** `{PROTOCOL_VERSION}` (`{preset.name}` preset).  "
        f"**Frozen diagnostics:** `{_repo_relative(frozen_path)}`.  "
        f"**Raw/aggregate results:** `{_repo_relative(aggregate_path)}`.",
        "",
        "## Design and evidential status",
        "",
        "This is an **internal prospective freeze**, not an externally registered "
        "preregistration. Static diagnostics were computed and written before any "
        "task score. The later task stage is cryptographically linked to that "
        f"artifact (`SHA-256 {aggregate['frozen_prediction_sha256']}`).",
        "",
        "The four interior mixtures were not benchmarked in the earlier channel "
        "comparison; α=0 and α=1 are endpoint controls:",
        "",
        r"\[\mathcal D_\alpha=(1-\alpha)\mathcal D_{\rm local}"
        r"+\alpha\mathcal D_{\rm collective},\qquad "
        r"\alpha\in\{0,.2,.4,.6,.8,1\}.\]",
        "",
        "Every point has the same Frobenius jump-strength budget as uniform local "
        "loss at γ=1. Within each N and seed, every α uses the same random "
        "Hamiltonian and input. Ridge regularization is selected on validation "
        "only; weights are retrained on train+validation and scored once on the "
        "held-out test split.",
        "",
        f"- Sizes: {', '.join(str(n) for n in preset.n_qubits)}",
        f"- Paired seeds: {preset.n_seeds} per size",
        f"- Split: wash/train/validation/test = "
        f"{preset.wash}/{preset.train}/{preset.validation}/{preset.test}",
        f"- STM delays: {min(preset.delays)}–{max(preset.delays)}",
        f"- Operating point: h={preset.h}, Δt={preset.dt}",
        "",
        "Frozen primary prediction: the ensemble-mean held-out STM capacities "
        "rank inversely with the ensemble-mean Liouvillian gaps across α. The "
        "predefined success threshold was Spearman ρ≥0.8 within each N; the exact "
        "six-point rank p value is descriptive and was not an additional pass "
        "threshold. A second frozen criterion required the diagnostic-selected "
        "*intermediate candidate* to beat local loss with a paired 95% interval "
        "entirely above zero. Selection of α=0.8 identifies a candidate within "
        "the interior grid, not an optimum.",
        "",
        "## Frozen diagnostics and held-out outcomes",
        "",
    ]

    all_both_pass = True
    for n_qubits in preset.n_qubits:
        frozen_n = frozen["predictions_by_N"][str(n_qubits)]
        result_n = aggregate["results_by_N"][str(n_qubits)]
        task_by_alpha = {
            float(item["alpha"]): item for item in result_n["task_summary"]
        }
        lines.extend(
            [
                f"### N={n_qubits}",
                "",
                "| α | mean gap ± SE | slow modes ± SE | retained mass ± SE "
                "| held-out STM MC ± SE |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for diagnostic in frozen_n["diagnostic_summary"]:
            alpha = float(diagnostic["alpha"])
            task = task_by_alpha[alpha]
            lines.append(
                f"| {alpha:.1f} "
                f"| {_fmt(diagnostic['spectral_gap_mean'], diagnostic['spectral_gap_se'], 4)} "
                f"| {_fmt(diagnostic['slow_mode_count_mean'], diagnostic['slow_mode_count_se'], 2)} "
                f"| {_fmt(diagnostic['retained_mode_mass_mean'], diagnostic['retained_mode_mass_se'], 2)} "
                f"| {_fmt(task['test_mc_mean'], task['test_mc_se'], 3)} |"
            )
        rank = result_n["prospective_rank_tests"]
        selected = result_n["diagnostic_selected_intermediate_vs_local"]
        endpoint = result_n["collective_endpoint_vs_local"]
        criteria = result_n["criteria"]
        all_both_pass = all_both_pass and criteria["both_pass"]
        lines.extend(
            [
                "",
                f"- Frozen gap rank (best predicted first): "
                f"`{frozen_n['frozen_gap_rank_best_to_worst']}`.",
                f"- Gap prediction versus held-out means: Spearman "
                f"ρ={rank['gap_spearman_rho']:.3f}, "
                f"exact two-sided p={rank['gap_spearman_p_two_sided']:.4g}; "
                f"criterion **{'passed' if criteria['primary_rank_validation_pass'] else 'did not pass'}**.",
                f"- Retained-mode-mass association: Spearman "
                f"ρ={rank['retained_mass_spearman_rho']:.3f}, "
                f"exact two-sided p="
                f"{rank['retained_mass_spearman_p_two_sided']:.4g}.",
                f"- Diagnostic-selected interior candidate "
                f"α={selected['selected_alpha']:.1f} "
                f"versus local α=0: paired ΔMC={selected['mean']:.3f} "
                f"[95% CI {selected['ci95_low']:.3f}, {selected['ci95_high']:.3f}], "
                f"{selected['relative_mean_advantage_percent']:+.1f}%, "
                f"paired t={selected['paired_t_stat']:.2f}, exact sign-flip "
                f"p={selected['p_two_sided']:.4g}; criterion "
                f"**{'passed' if criteria['selected_candidate_validation_pass'] else 'did not pass'}**.",
                f"- Known endpoint α=1 versus local α=0: paired "
                f"ΔMC={endpoint['mean']:.3f} "
                f"[95% CI {endpoint['ci95_low']:.3f}, {endpoint['ci95_high']:.3f}], "
                f"{endpoint['relative_mean_advantage_percent']:+.1f}%, "
                f"exact sign-flip p={endpoint['p_two_sided']:.4g}.",
                "",
            ]
        )

    if all_both_pass:
        outcome = (
            "Both frozen criteria passed at every tested size. Within this "
            "predefined local-to-collective family, static Liouvillian diagnostics "
            "predicted the ensemble-mean ordering of the four previously "
            "unbenchmarked interior mixtures before task evaluation. The endpoint "
            "controls establish continuity with the earlier comparison."
        )
    else:
        outcome = (
            "At least one frozen criterion failed at one or more sizes. The "
            "prospective interpolation test therefore does not support an unqualified "
            "predictive-design claim; individual associations above remain "
            "descriptive."
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            outcome,
            "",
            "Permitted claim: a prospective, paired interpolation test at the "
            "stated operating point, sizes, Hamiltonian ensemble, readout, and STM "
            "protocol. The endpoints reproduce the established direction while "
            "the four interior α values are the previously unbenchmarked points. "
            "The diagnostic-selected α=0.8 is a candidate, not a claim of an "
            "optimum over continuous α.",
            "",
            "Not permitted from this experiment alone:",
            "",
            "- no claim that the chosen interpolation is globally optimal;",
            "- no claim that spectral gap predicts nonlinear-task performance;",
            "- no claim of universality across Hamiltonians, encodings, readouts, "
            "or dissipation budgets;",
            "- no causal proof that slow-mode count explains the N=4→8 scaling "
            "(only N=4/5 are tested here);",
            "- no claim of external preregistration.",
            "",
            "## Reproduction",
            "",
            "Run from the repository root:",
            "",
            "```bash",
            "PYTHONPATH=src:experiments .venv/bin/python "
            "experiments/run_quantum_strengthening.py freeze --preset "
            f"{preset.name} --workers 4",
            "PYTHONPATH=src:experiments .venv/bin/python "
            "experiments/run_quantum_strengthening.py score --preset "
            f"{preset.name} --workers 4",
            "PYTHONPATH=src:experiments .venv/bin/python "
            "experiments/run_quantum_strengthening.py report --preset "
            f"{preset.name}",
            "PYTHONPATH=src:experiments .venv/bin/python "
            "experiments/run_quantum_strengthening.py archive --preset "
            f"{preset.name}",
            "```",
            "",
            "Because the protocol hash includes the scientific source and "
            "environment manifest, commit the driver and tests first, then run "
            "freeze → score → report → archive without changing those inputs.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"REPORT {report_path}", flush=True)
    return report_path


def archive_results(
    preset: Preset,
    outdir: Path,
    report_path: Path,
    archive_path: Path,
) -> Path:
    """Validate and package the full prospective record plus provenance."""
    frozen_path, frozen = _load_frozen(preset, outdir)
    aggregate_path = aggregate_results(preset, outdir)
    report_path = write_report(preset, outdir, report_path)
    task_paths = sorted(_task_dir(outdir).glob("*.json"))
    expected_jobs = len(preset.n_qubits) * len(ALPHAS) * preset.n_seeds
    if len(task_paths) != expected_jobs:
        raise RuntimeError(
            f"archive requires exactly {expected_jobs} task checkpoints, "
            f"found {len(task_paths)}"
        )

    package_paths = [
        *(REPO_ROOT / relative for relative in SOURCE_FILES),
        frozen_path,
        _prediction_seal_path(outdir),
        outdir / "task_stage_metadata.json",
        aggregate_path,
        report_path,
        *task_paths,
    ]
    missing = [str(path) for path in package_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"archive inputs missing: {missing}")

    members: dict[str, str] = {}
    path_by_member: dict[str, Path] = {}
    for path in package_paths:
        member = _repo_relative(path)
        if member in path_by_member:
            raise RuntimeError(f"duplicate archive member {member}")
        path_by_member[member] = path
        members[member] = _sha256_file(path)

    manifest_path = outdir / "provenance_manifest.json"
    manifest = {
        "artifact_type": "quantum_strengthening_provenance_manifest",
        "created_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": frozen["protocol_sha256"],
        "frozen_prediction_sha256": _sha256_file(frozen_path),
        "source_environment_sha256": frozen["source_environment_sha256"],
        "source_environment": source_environment_manifest(),
        "git_head": _git_head(),
        "git_dirty_status": _git_dirty_status(),
        "git_dirty_status_at_freeze": frozen["git_dirty_status_at_freeze"],
        "archive_members_sha256": members,
        "reproduction_order": [
            "commit the scientific driver and tests",
            "freeze diagnostics in an empty output directory",
            "score task checkpoints without modifying source",
            "generate the report",
            "create this archive",
        ],
    }
    _atomic_write_json(manifest_path, manifest)
    manifest_member = _repo_relative(manifest_path)
    path_by_member[manifest_member] = manifest_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for member, path in sorted(path_by_member.items()):
            bundle.write(path, member)
    with zipfile.ZipFile(archive_path) as bundle:
        archived = set(bundle.namelist())
    expected = set(path_by_member)
    if archived != expected:
        raise RuntimeError(
            f"archive completeness failure: missing={sorted(expected-archived)}, "
            f"unexpected={sorted(archived-expected)}"
        )
    print(
        f"ARCHIVE {archive_path} members={len(expected)} "
        f"sha256={_sha256_file(archive_path)}",
        flush=True,
    )
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "score", "report", "archive"))
    parser.add_argument("--preset", choices=tuple(PRESETS), default="paper")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--outdir",
        type=Path,
        help="default: results/quantum_strengthening_v2_<preset>",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(REPORTS_DIR) / "quantum_strengthening_report.md",
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        help="default: <outdir>/quantum_strengthening_archive.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    outdir = (
        args.outdir
        if args.outdir is not None
        else Path(RESULTS_DIR) / f"quantum_strengthening_v2_{preset.name}"
    )
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.command == "freeze":
        freeze_predictions(preset, outdir, args.workers)
    elif args.command == "score":
        score_tasks(preset, outdir, args.workers)
    elif args.command == "report":
        write_report(preset, outdir, args.report_path)
    else:
        archive_path = (
            args.archive_path
            if args.archive_path is not None
            else outdir / "quantum_strengthening_archive.zip"
        )
        archive_results(preset, outdir, args.report_path, archive_path)


if __name__ == "__main__":
    main()

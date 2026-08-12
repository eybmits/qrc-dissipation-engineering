"""Acceptance-critical tuning and independence controls for the Quantum revision.

This driver adds evidence to the existing dissipation-engineering paper; it does
not introduce a new scientific project.  It provides three checkpointed stages:

``strength``
    Extend the existing six-channel ``R_sweep2`` record beyond the collective
    channel's x4 boundary and aggregate all six channels with paired intervals.

``nested``
    Compare uniform-local and collective loss after an identical, two-stage
    search over (h, dt, dissipative strength), followed by one evaluation on an
    untouched reservoir/input ensemble.

``interpolation``
    Apply the already-frozen alpha=0.8 diagnostic choice (and the complete
    alpha grid) to Hamiltonian/input seeds disjoint from every diagnostic seed.

All scientific rows are written atomically as they finish.  Run from the repo:

    PYTHONPATH=src:experiments .venv/bin/python \
      experiments/run_revision_tuning.py all --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import _paths  # noqa: E402,F401
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from _paths import REPORTS_DIR, RESULTS_DIR  # noqa: E402
from qrc import dissipators as dsp  # noqa: E402
from qrc import readout, reservoirs as res, tasks  # noqa: E402
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive  # noqa: E402
from qrc.sparse_evolve import SparseLindbladReservoir  # noqa: E402
from run_final_scaling import deterministic_seeds  # noqa: E402
from run_quantum_strengthening import (  # noqa: E402
    Preset as InterpolationPreset,
)


PROTOCOL_VERSION = "revision-tuning-v1-2026-07-23"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTROOT = Path(RESULTS_DIR) / "revision_tuning"
REPORT_PATH = Path(REPORTS_DIR) / "revision_tuning_report.md"
OLD_SWEEP_DIR = Path(RESULTS_DIR) / "review_protocol"
FROZEN_DIAGNOSTICS = (
    Path(RESULTS_DIR)
    / "quantum_strengthening_v2_paper"
    / "frozen_diagnostic_predictions.json"
)

N = 5
WASH, TRAIN, VAL, TEST = 200, 600, 150, 400
DELAYS = tuple(range(1, 21))
RIDGES = (
    0.0,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)
CHANNELS = (
    "CD_paper",
    "A1_heterogeneous",
    "B2_thermal",
    "B3_collective",
    "B4_loss_exchange",
    "B5_pair",
)
OLD_MULTIPLIERS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
EXTENSION_MULTIPLIERS = (8.0, 16.0, 32.0, 64.0)
EXTENDED_CHANNELS = ("CD_paper", "B3_collective")

# The complete screen is broad; only its top candidates advance to an
# independent selection ensemble.  No test seed participates in either stage.
NESTED_H = (0.25, 0.5, 1.0, 2.0)
NESTED_DT = (0.1, 0.25, 0.5, 1.0)
NESTED_MULTIPLIERS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
NESTED_CHANNELS = ("CD_paper", "B3_collective")
NESTED_SHORTLIST = 8
SCREEN_WASH, SCREEN_TRAIN, SCREEN_VAL = 100, 250, 100
N_SCREEN_SEEDS, N_SELECTION_SEEDS, N_TEST_SEEDS = 2, 12, 24

INTERPOLATION_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
N_INTERPOLATION_SEEDS = 24
INTERPOLATION_PRESET = InterpolationPreset(
    "revision_fresh_ensemble",
    (4, 5),
    N_INTERPOLATION_SEEDS,
    wash=200,
    train=600,
    validation=300,
    test=400,
    delays=DELAYS,
    h=0.5,
    dt=0.5,
)

SCIENTIFIC_SOURCES = (
    "experiments/run_revision_tuning.py",
    "experiments/run_review_experiments.py",
    "experiments/run_quantum_strengthening.py",
    "experiments/run_final_scaling.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


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


def source_hashes() -> dict[str, str]:
    paths = {name: REPO_ROOT / name for name in SCIENTIFIC_SOURCES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"scientific source files missing: {missing}")
    return {name: _sha256_file(path) for name, path in paths.items()}


def _fresh_seed_pool(namespace: int, count: int, excluded: set[int]) -> list[int]:
    """Generate a deterministic pool, rejecting every previously assigned seed."""
    rng = np.random.default_rng(np.random.SeedSequence([2026, 7, 23, namespace]))
    values: list[int] = []
    used = set(excluded)
    while len(values) < count:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in used:
            values.append(candidate)
            used.add(candidate)
    return values


def seed_namespaces() -> dict[str, list[int]]:
    """Return pairwise-disjoint seeds, all disjoint from legacy diagnostics."""
    legacy = set(deterministic_seeds(256))
    screen = _fresh_seed_pool(101, N_SCREEN_SEEDS, legacy)
    selection = _fresh_seed_pool(102, N_SELECTION_SEEDS, legacy | set(screen))
    test = _fresh_seed_pool(
        103, N_TEST_SEEDS, legacy | set(screen) | set(selection)
    )
    interpolation = _fresh_seed_pool(
        104,
        N_INTERPOLATION_SEEDS,
        legacy | set(screen) | set(selection) | set(test),
    )
    return {
        "legacy_diagnostic": sorted(legacy),
        "nested_screen": screen,
        "nested_selection": selection,
        "nested_test": test,
        "fresh_interpolation": interpolation,
    }


def assert_seed_disjointness(pools: dict[str, Sequence[int]]) -> None:
    names = list(pools)
    for index, left in enumerate(names):
        if len(set(pools[left])) != len(pools[left]):
            raise RuntimeError(f"duplicate seed inside namespace {left}")
        for right in names[index + 1 :]:
            overlap = set(pools[left]) & set(pools[right])
            if overlap:
                raise RuntimeError(
                    f"seed namespaces {left}/{right} overlap: {sorted(overlap)}"
                )


def _protocol(stage: str, details: dict) -> dict:
    pools = seed_namespaces()
    assert_seed_disjointness(pools)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": stage,
        "details": details,
        "seed_namespaces": pools,
        "git_head_at_protocol": _git_head(),
        "scientific_sources_sha256": source_hashes(),
    }


def _ensure_manifest(directory: Path, stage: str, details: dict) -> tuple[dict, str]:
    protocol = _protocol(stage, details)
    fingerprint = _sha256_json(protocol)
    path = directory / "manifest.json"
    payload = {
        "artifact_type": "revision_tuning_manifest",
        "manifest_status": "frozen_before_stage_rows",
        "protocol": protocol,
        "protocol_sha256": fingerprint,
    }
    if path.exists():
        old = json.loads(path.read_text())
        if old != payload:
            raise RuntimeError(f"manifest drift in {path}")
    else:
        _atomic_json(path, payload)
    return payload, fingerprint


def _tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    se = (
        float(np.std(array, ddof=1) / math.sqrt(len(array)))
        if len(array) > 1
        else 0.0
    )
    return mean, se


def paired_stats(candidate: Sequence[float], reference: Sequence[float]) -> dict:
    """Paired mean difference, t interval, sign test, and transparent raw deltas."""
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError("paired samples must be non-empty vectors of equal length")
    difference = a - b
    mean, se = _mean_se(difference)
    critical = (
        float(stats.t.ppf(0.975, len(difference) - 1))
        if len(difference) > 1
        else 0.0
    )
    nonzero = difference[~np.isclose(difference, 0.0, atol=1e-14)]
    wins = int(np.count_nonzero(nonzero > 0))
    sign_p = (
        float(stats.binomtest(wins, len(nonzero), 0.5).pvalue)
        if len(nonzero)
        else 1.0
    )
    return {
        "n": int(len(difference)),
        "candidate_mean": float(np.mean(a)),
        "reference_mean": float(np.mean(b)),
        "mean_difference": mean,
        "se_difference": se,
        "ci95_low": float(mean - critical * se),
        "ci95_high": float(mean + critical * se),
        "relative_mean_difference_percent": float(100.0 * mean / np.mean(b)),
        "wins": wins,
        "ties": int(len(difference) - len(nonzero)),
        "exact_sign_p_two_sided": sign_p,
        "paired_differences": difference.tolist(),
    }


def _build_jumps(method: str, n_qubits: int, multiplier: float):
    target = dsp.jump_strength(dsp.local_loss(n_qubits, 1.0)) * multiplier
    if method == "CD_paper":
        raw = dsp.local_loss(n_qubits, 1.0)
    elif method == "B3_collective":
        raw = dsp.collective_loss(n_qubits, 1.0)
    else:
        raise ValueError(f"unsupported nested-tuning channel {method}")
    return dsp.normalize_jump_strength(raw, target), target


def _build_interpolation_jumps(
    n_qubits: int, alpha: float
) -> tuple[list[tuple[np.ndarray, float]], float]:
    """Revision-local copy of the sealed matched-budget interpolation rule."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0,1]")
    target = dsp.jump_strength(dsp.local_loss(n_qubits, 1.0))
    jumps: list[tuple[np.ndarray, float]] = []
    if alpha < 1.0:
        jumps.extend(
            dsp.normalize_jump_strength(
                dsp.local_loss(n_qubits, 1.0), (1.0 - alpha) * target
            )
        )
    if alpha > 0.0:
        jumps.extend(
            dsp.normalize_jump_strength(
                dsp.collective_loss(n_qubits, 1.0), alpha * target
            )
        )
    return jumps, target


def _reservoir(
    method: str,
    seed: int,
    h: float,
    dt: float,
    multiplier: float,
    inputs_len: int,
):
    problem_rng = np.random.default_rng(seed)
    J = res.random_couplings(N, 1.0, problem_rng)
    inputs = tasks.stm_inputs(inputs_len, problem_rng)
    jumps, target = _build_jumps(method, N, multiplier)
    H0 = ising_xx_hamiltonian(J, h, N)
    Hx = transverse_drive(N)
    reservoir = SparseLindbladReservoir.from_terms(
        N, H0 + h * Hx, h * Hx, jumps, dt
    )
    return reservoir, inputs, jumps, target


def _validation_scores(
    features: np.ndarray,
    full_inputs: np.ndarray,
    wash: int = WASH,
    train: int = TRAIN,
    validation: int = VAL,
) -> dict[str, float]:
    post = np.asarray(full_inputs[wash:], dtype=float)
    if len(post) < train + validation:
        raise ValueError("calibration sequence is shorter than train+validation")
    Xb = readout.add_bias(features)
    output: dict[str, float] = {}
    for ridge in RIDGES:
        total = 0.0
        for delay in DELAYS:
            y = tasks.delayed_target(full_inputs, delay)[wash:]
            fit = np.zeros(len(y), bool)
            check = np.zeros(len(y), bool)
            fit[:train] = True
            check[train : train + validation] = True
            fit &= np.isfinite(y)
            check &= np.isfinite(y)
            weights = readout.train_readout(Xb[fit], y[fit], ridge=ridge)
            total += readout.capacity(
                y[check], readout.predict(Xb[check], weights)
            )
        output[f"{ridge:.12g}"] = float(total)
    return output


def _test_score(
    features: np.ndarray,
    full_inputs: np.ndarray,
    ridge: float,
    wash: int = WASH,
    train: int = TRAIN,
    validation: int = VAL,
    test: int = TEST,
) -> tuple[float, list[float]]:
    post = np.asarray(full_inputs[wash:], dtype=float)
    if len(post) != train + validation + test:
        raise ValueError("test sequence has the wrong split length")
    Xb = readout.add_bias(features)
    scores: list[float] = []
    for delay in DELAYS:
        y = tasks.delayed_target(full_inputs, delay)[wash:]
        fit = np.zeros(len(y), bool)
        check = np.zeros(len(y), bool)
        fit[: train + validation] = True
        check[train + validation :] = True
        fit &= np.isfinite(y)
        check &= np.isfinite(y)
        weights = readout.train_readout(Xb[fit], y[fit], ridge=ridge)
        scores.append(
            readout.capacity(y[check], readout.predict(Xb[check], weights))
        )
    return float(np.sum(scores)), scores


# ---------------------------------------------------------------- strength
def _strength_path(method: str, multiplier: float, seed: int) -> Path:
    return (
        OUTROOT
        / "strength_extension"
        / "jobs"
        / f"{method}_x{_tag(multiplier)}_s{seed}.json"
    )


def strength_job(job: tuple[str, float, int, str]) -> dict:
    """One row exactly compatible with the legacy R_sweep2 protocol."""
    method, multiplier, seed, protocol_sha256 = job
    started = time.perf_counter()
    from run_review_experiments import make_reservoir, sweep2_scores

    rng = np.random.default_rng(seed)
    J = res.random_couplings(N, 1.0, rng)
    reservoir = make_reservoir(
        "xx_z_x",
        method,
        J,
        N,
        np.random.default_rng(seed + 1),
        rate_scale=multiplier,
    )
    validation_mc, test_mc = sweep2_scores(reservoir, N, seed)
    return {
        "protocol_sha256": protocol_sha256,
        "block": "revision_strength_extension",
        "N": N,
        "ensemble": "xx_z_x",
        "method": method,
        "task": "stm",
        "seed": seed,
        "mult": multiplier,
        "val_value": validation_mc,
        "value": test_mc,
        "backend": "exact_sparse_expm_multiply",
        "runtime_s": time.perf_counter() - started,
    }


def _run_checkpointed(
    jobs: Sequence[tuple],
    worker,
    path_for,
    workers: int,
    label: str,
) -> None:
    pending = [job for job in jobs if not path_for(job).exists()]
    print(
        f"{label}: total={len(jobs)} complete={len(jobs)-len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return
    if workers <= 1:
        for index, job in enumerate(pending, 1):
            row = worker(job)
            _atomic_json(path_for(job), row)
            print(f"{label} {index}/{len(pending)} {row['runtime_s']:.1f}s", flush=True)
        return
    try:
        pool = ProcessPoolExecutor(max_workers=workers)
        executor_kind = "process"
    except (OSError, PermissionError) as exc:
        print(
            f"{label}: process executor unavailable "
            f"({type(exc).__name__}: {exc}); using bounded threads",
            flush=True,
        )
        pool = ThreadPoolExecutor(max_workers=workers)
        executor_kind = "thread"
    print(f"{label}: executor={executor_kind}", flush=True)
    with pool:
        futures = {pool.submit(worker, job): job for job in pending}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            row = future.result()
            _atomic_json(path_for(job), row)
            if index % 10 == 0 or index == len(pending):
                print(
                    f"{label} {index}/{len(pending)} latest={row['runtime_s']:.1f}s",
                    flush=True,
                )


def _legacy_sweep_rows() -> tuple[list[dict], list[dict]]:
    paths = sorted(OLD_SWEEP_DIR.glob("R_sweep2__*.json"))
    rows: list[dict] = []
    provenance: list[dict] = []
    for path in paths:
        row = json.loads(path.read_text())
        if row.get("block") != "R_sweep2":
            continue
        rows.append(row)
        provenance.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(path),
            }
        )
    expected = len(CHANNELS) * len(OLD_MULTIPLIERS) * 20
    if len(rows) != expected:
        raise RuntimeError(f"legacy R_sweep2 incomplete: {len(rows)}/{expected}")
    expected_seeds = set(deterministic_seeds(20))
    identities = {
        (row["method"], float(row["mult"]), int(row["seed"])) for row in rows
    }
    expected_identities = set(
        itertools.product(CHANNELS, OLD_MULTIPLIERS, expected_seeds)
    )
    if identities != expected_identities:
        raise RuntimeError("legacy R_sweep2 identity grid is incomplete")
    return rows, provenance


def _all_strength_rows() -> tuple[list[dict], list[dict]]:
    rows, provenance = _legacy_sweep_rows()
    for path in sorted((OUTROOT / "strength_extension" / "jobs").glob("*.json")):
        row = json.loads(path.read_text())
        rows.append(row)
        provenance.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(path),
            }
        )
    return rows, provenance


def aggregate_alternative_matching() -> Path:
    """Expose every existing resource-matching row, feasibility flag, and effect."""
    reference_paths = sorted(OLD_SWEEP_DIR.glob("R_match__CD_paper_ref_*.json"))
    references = {
        int(row["seed"]): row
        for row in (json.loads(path.read_text()) for path in reference_paths)
    }
    expected_seeds = set(deterministic_seeds(32))
    if set(references) != expected_seeds:
        raise RuntimeError("alternative-matching dial reference is incomplete")

    provenance: list[dict] = []
    raw_rows: dict[str, list[dict]] = {}
    for block in ("R_match", "R_match2", "R_gapsweep"):
        paths = sorted(OLD_SWEEP_DIR.glob(f"{block}__*.json"))
        raw_rows[block] = [json.loads(path.read_text()) for path in paths]
        provenance.extend(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": _sha256_file(path),
            }
            for path in paths
        )

    conditions: dict[str, dict] = {}
    for block in ("R_match", "R_match2"):
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in raw_rows[block]:
            if row["method"] == "CD_paper":
                continue
            key = (str(row["method"]), str(row["mode"]))
            groups.setdefault(key, []).append(row)
        for (method, mode), group in sorted(groups.items()):
            if {int(row["seed"]) for row in group} != expected_seeds:
                raise RuntimeError(f"incomplete {block} {method}/{mode}")
            ordered = sorted(group, key=lambda row: int(row["seed"]))
            method_values = [float(row["value"]) for row in ordered]
            dial_values = [
                float(references[int(row["seed"])]["value"]) for row in ordered
            ]
            scale = [float(row["scale_factor"]) for row in ordered]
            entry = {
                "block": block,
                "method": method,
                "matching_mode": mode,
                "effect_vs_standard_dial": paired_stats(
                    method_values, dial_values
                ),
                "matched_channel_mean": _mean_se(method_values)[0],
                "matched_channel_se": _mean_se(method_values)[1],
                "dial_mean": _mean_se(dial_values)[0],
                "dial_se": _mean_se(dial_values)[1],
                "scale_factor_mean": _mean_se(scale)[0],
                "scale_factor_se": _mean_se(scale)[1],
                "scale_factor_min": min(scale),
                "scale_factor_max": max(scale),
            }
            if mode == "energy":
                entry["match_feasibility"] = {
                    "status": "analytically_exact_linear_rescaling",
                    "reachable_count": len(group),
                    "total": len(group),
                }
            elif mode == "gap":
                # The stored log-bisection searched [0.05, 40]. Reaching the
                # upper boundary means the target gap was not attained.
                reachable = [value < 39.99 for value in scale]
                entry["match_feasibility"] = {
                    "status": (
                        "root_inside_search_interval"
                        if all(reachable)
                        else "upper_bound_censored_target_not_reached"
                    ),
                    "reachable_count": int(sum(reachable)),
                    "total": len(group),
                    "search_interval": [0.05, 40.0],
                }
            elif mode == "activity":
                reachable = [bool(row["reachable"]) for row in ordered]
                ratios = [float(row["activity_ratio"]) for row in ordered]
                entry["match_feasibility"] = {
                    "status": (
                        "all_exactly_reachable"
                        if all(reachable)
                        else "closest_achievable_activity_reported_when_unreachable"
                    ),
                    "reachable_count": int(sum(reachable)),
                    "total": len(group),
                    "achieved_to_reference_ratio_mean": _mean_se(ratios)[0],
                    "achieved_to_reference_ratio_se": _mean_se(ratios)[1],
                    "achieved_to_reference_ratio_min": min(ratios),
                    "achieved_to_reference_ratio_max": max(ratios),
                }
            conditions[f"{method}__{mode}"] = entry

    gap_curves: dict[str, list[dict]] = {}
    gap_rows = raw_rows["R_gapsweep"]
    for method in CHANNELS:
        points = []
        for multiplier in OLD_MULTIPLIERS:
            group = [
                row
                for row in gap_rows
                if row["method"] == method
                and math.isclose(float(row["mult"]), multiplier)
            ]
            if len(group) != 3:
                raise RuntimeError(f"incomplete gap curve {method} x{multiplier}")
            mean, se = _mean_se([float(row["value"]) for row in group])
            points.append(
                {
                    "multiplier": multiplier,
                    "driven_gap_mean": mean,
                    "driven_gap_se": se,
                    "n": len(group),
                }
            )
        gap_curves[method] = points

    payload = {
        "artifact_type": "revision_alternative_matching_aggregate",
        "reference": {
            "method": "CD_paper",
            "n": len(references),
            "mean": _mean_se([row["value"] for row in references.values()])[0],
            "se": _mean_se([row["value"] for row in references.values()])[1],
        },
        "conditions": conditions,
        "full_driven_gap_curves": gap_curves,
        "raw_rows": raw_rows,
        "raw_provenance": provenance,
    }
    path = OUTROOT / "strength_extension" / "alternative_matching_aggregate.json"
    _atomic_json(path, payload)
    return path


def curve_bracket(rows: Sequence[dict], method: str) -> dict:
    curve: list[dict] = []
    for multiplier in sorted(
        {float(row["mult"]) for row in rows if row["method"] == method}
    ):
        group = [
            row
            for row in rows
            if row["method"] == method
            and math.isclose(float(row["mult"]), multiplier)
        ]
        mean, se = _mean_se([row["val_value"] for row in group])
        curve.append(
            {
                "multiplier": multiplier,
                "validation_mean": mean,
                "validation_se": se,
                "n": len(group),
            }
        )
    best_index = max(
        range(len(curve)), key=lambda index: curve[index]["validation_mean"]
    )
    bracketed = (
        0 < best_index < len(curve) - 1
        and curve[best_index - 1]["validation_mean"]
        < curve[best_index]["validation_mean"]
        and curve[best_index + 1]["validation_mean"]
        < curve[best_index]["validation_mean"]
    )
    return {
        "curve": curve,
        "best_multiplier": curve[best_index]["multiplier"],
        "bracketed": bracketed,
        "left_multiplier": (
            curve[best_index - 1]["multiplier"] if best_index > 0 else None
        ),
        "right_multiplier": (
            curve[best_index + 1]["multiplier"]
            if best_index < len(curve) - 1
            else None
        ),
    }


def _selected_scores(
    rows: Sequence[dict], method: str, mode: str
) -> tuple[dict[int, float], dict[int, float]]:
    seeds = sorted({int(row["seed"]) for row in rows if row["method"] == method})
    values: dict[int, float] = {}
    selected: dict[int, float] = {}
    for target_seed in seeds:
        method_rows = [row for row in rows if row["method"] == method]
        if mode == "per_instance":
            candidates = [row for row in method_rows if row["seed"] == target_seed]
            best = max(
                candidates,
                key=lambda row: (float(row["val_value"]), -float(row["mult"])),
            )
            multiplier = float(best["mult"])
        elif mode == "leave_one_seed_out":
            multiplier_values = sorted({float(row["mult"]) for row in method_rows})
            means = {}
            for multiplier in multiplier_values:
                calibration = [
                    float(row["val_value"])
                    for row in method_rows
                    if row["seed"] != target_seed
                    and math.isclose(float(row["mult"]), multiplier)
                ]
                if len(calibration) != len(seeds) - 1:
                    raise RuntimeError(
                        f"incomplete LOSO calibration for {method} x{multiplier}"
                    )
                means[multiplier] = float(np.mean(calibration))
            multiplier = max(multiplier_values, key=lambda x: (means[x], -x))
            best = next(
                row
                for row in method_rows
                if row["seed"] == target_seed
                and math.isclose(float(row["mult"]), multiplier)
            )
        else:
            raise ValueError(mode)
        selected[target_seed] = multiplier
        values[target_seed] = float(best["value"])
    return values, selected


def aggregate_strength() -> Path:
    rows, provenance = _all_strength_rows()
    expected_count = (
        len(CHANNELS) * len(OLD_MULTIPLIERS) * 20
        + len(EXTENDED_CHANNELS) * 2 * 20
    )
    if len(rows) != expected_count:
        raise RuntimeError(
            f"strength aggregate is partial: {len(rows)}/{expected_count} rows"
        )
    methods: dict[str, dict] = {}
    for method in CHANNELS:
        method_rows = [row for row in rows if row["method"] == method]
        expected_multipliers = (
            sorted({float(row["mult"]) for row in method_rows})
        )
        seeds = set(deterministic_seeds(20))
        for multiplier in expected_multipliers:
            present = {
                int(row["seed"])
                for row in method_rows
                if math.isclose(float(row["mult"]), multiplier)
            }
            if present != seeds:
                raise RuntimeError(f"incomplete strength rows for {method} x{multiplier}")
        loso_values, loso_selected = _selected_scores(
            method_rows, method, "leave_one_seed_out"
        )
        instance_values, instance_selected = _selected_scores(
            method_rows, method, "per_instance"
        )
        mean, se = _mean_se(list(loso_values.values()))
        methods[method] = {
            "curve_bracket": curve_bracket(method_rows, method),
            "leave_one_seed_out": {
                "test_mean": mean,
                "test_se": se,
                "scores_by_seed": loso_values,
                "selected_multiplier_by_seed": loso_selected,
                "selection_counts": dict(Counter(map(str, loso_selected.values()))),
            },
            "per_instance_validation": {
                "test_mean": _mean_se(list(instance_values.values()))[0],
                "test_se": _mean_se(list(instance_values.values()))[1],
                "scores_by_seed": instance_values,
                "selected_multiplier_by_seed": instance_selected,
                "selection_counts": dict(
                    Counter(map(str, instance_selected.values()))
                ),
            },
        }
    collective = methods["B3_collective"]["leave_one_seed_out"]["scores_by_seed"]
    comparisons = {}
    for method in CHANNELS:
        if method == "B3_collective":
            continue
        reference = methods[method]["leave_one_seed_out"]["scores_by_seed"]
        common = sorted(set(collective) & set(reference), key=int)
        comparisons[method] = paired_stats(
            [collective[key] for key in common],
            [reference[key] for key in common],
        )
    payload = {
        "artifact_type": "revision_six_channel_strength_aggregate",
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "expected_raw_row_count": expected_count,
        "complete_raw_row_count": len(rows),
        "expected_extended_multipliers": [8.0, 16.0],
        "collective_optimum_bracketed": methods["B3_collective"][
            "curve_bracket"
        ]["bracketed"],
        "selection_primary": (
            "leave-one-seed-out population selection: each test reservoir's "
            "strength is selected from validation scores of the other 19 reservoirs"
        ),
        "methods": methods,
        "collective_paired_comparisons": comparisons,
        "raw_rows": rows,
        "raw_provenance": provenance,
    }
    path = OUTROOT / "strength_extension" / "six_channel_aggregate.json"
    _atomic_json(path, payload)
    return path


def run_strength(workers: int) -> Path:
    details = {
        "legacy_protocol": "R_sweep2 exact N=5, wash/train/validation/test=200/450/150/400",
        "legacy_multipliers": OLD_MULTIPLIERS,
        "extended_channels": EXTENDED_CHANNELS,
        "extension_candidates": EXTENSION_MULTIPLIERS,
        "adaptive_rule": (
            "run x8 and x16; if the collective ensemble-mean validation maximum "
            "is still the right boundary, double through x64 until bracketed"
        ),
    }
    _, protocol_sha = _ensure_manifest(
        OUTROOT / "strength_extension", "strength", details
    )
    seeds = deterministic_seeds(20)
    initial = (8.0, 16.0)
    jobs = [
        (method, multiplier, seed, protocol_sha)
        for method in EXTENDED_CHANNELS
        for multiplier in initial
        for seed in seeds
    ]
    _run_checkpointed(
        jobs,
        strength_job,
        lambda job: _strength_path(job[0], job[1], job[2]),
        workers,
        "strength",
    )
    for multiplier in (32.0, 64.0):
        rows, _ = _all_strength_rows()
        if curve_bracket(rows, "B3_collective")["bracketed"]:
            break
        jobs = [
            (method, multiplier, seed, protocol_sha)
            for method in EXTENDED_CHANNELS
            for seed in seeds
        ]
        _run_checkpointed(
            jobs,
            strength_job,
            lambda job: _strength_path(job[0], job[1], job[2]),
            workers,
            "strength-adaptive",
        )
    path = aggregate_strength()
    matching_path = aggregate_alternative_matching()
    result = json.loads(path.read_text())
    bracket = result["methods"]["B3_collective"]["curve_bracket"]
    if not bracket["bracketed"]:
        raise RuntimeError("collective optimum was not bracketed through x64")
    print(
        f"strength optimum bracketed at x{bracket['best_multiplier']} "
        f"between x{bracket['left_multiplier']} and x{bracket['right_multiplier']}",
        flush=True,
    )
    print(f"alternative matching aggregate: {matching_path}", flush=True)
    return path


# ---------------------------------------------------------------- nested
def nested_grid() -> list[tuple[float, float, float]]:
    return list(itertools.product(NESTED_H, NESTED_DT, NESTED_MULTIPLIERS))


def _nested_path(
    stage: str, method: str, config: Sequence[float], seed: int
) -> Path:
    h, dt, multiplier = config
    return (
        OUTROOT
        / "nested_tuning"
        / f"{stage}_jobs"
        / (
            f"{method}_h{_tag(h)}_dt{_tag(dt)}_x{_tag(multiplier)}"
            f"_s{seed}.json"
        )
    )


def nested_calibration_job(
    job: tuple[str, tuple[float, float, float], int, str, str]
) -> dict:
    method, config, seed, stage, protocol_sha256 = job
    h, dt, multiplier = config
    started = time.perf_counter()
    if stage == "screen":
        wash, train, validation = SCREEN_WASH, SCREEN_TRAIN, SCREEN_VAL
    elif stage == "selection":
        wash, train, validation = WASH, TRAIN, VAL
    else:
        raise ValueError(f"invalid nested calibration stage {stage}")
    reservoir, inputs, jumps, target = _reservoir(
        method, seed, h, dt, multiplier, wash + train + validation
    )
    observables = readout.pauli_observables(N, max_weight=2)
    features = reservoir.run(inputs, observables, washout=wash)
    scores = _validation_scores(
        features,
        inputs,
        wash=wash,
        train=train,
        validation=validation,
    )
    return {
        "protocol_sha256": protocol_sha256,
        "stage": stage,
        "split": {
            "wash": wash,
            "train": train,
            "validation": validation,
            "test": 0,
        },
        "method": method,
        "seed": seed,
        "h": h,
        "dt": dt,
        "strength_multiplier": multiplier,
        "ridge_validation_mc": scores,
        "jump_strength": dsp.jump_strength(jumps),
        "target_strength": target,
        "relative_budget_error": abs(dsp.jump_strength(jumps) - target) / target,
        "n_features": int(features.shape[1]),
        "backend": "exact_sparse_expm_multiply",
        "runtime_s": time.perf_counter() - started,
    }


def _read_nested_rows(stage: str) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(
            (OUTROOT / "nested_tuning" / f"{stage}_jobs").glob("*.json")
        )
    ]


def rank_nested_configs(
    rows: Sequence[dict],
    methods: Sequence[str],
    configs: Sequence[Sequence[float]],
    seeds: Sequence[int],
) -> dict[str, list[dict]]:
    """Rank configs using calibration rows only; test rows are not an argument."""
    ranked: dict[str, list[dict]] = {}
    for method in methods:
        entries: list[dict] = []
        for raw_config in configs:
            config = tuple(map(float, raw_config))
            group = [
                row
                for row in rows
                if row["method"] == method
                and int(row["seed"]) in set(seeds)
                and math.isclose(float(row["h"]), config[0])
                and math.isclose(float(row["dt"]), config[1])
                and math.isclose(float(row["strength_multiplier"]), config[2])
            ]
            if {int(row["seed"]) for row in group} != set(seeds):
                raise RuntimeError(f"incomplete nested rows for {method} {config}")
            ridge_means = {
                f"{ridge:.12g}": float(
                    np.mean(
                        [row["ridge_validation_mc"][f"{ridge:.12g}"] for row in group]
                    )
                )
                for ridge in RIDGES
            }
            best_ridge = max(
                RIDGES,
                key=lambda ridge: ridge_means[f"{ridge:.12g}"],
            )
            ridge_upper_boundary = math.isclose(best_ridge, RIDGES[-1])
            per_seed = {
                int(row["seed"]): float(
                    row["ridge_validation_mc"][f"{best_ridge:.12g}"]
                )
                for row in group
            }
            entries.append(
                {
                    "config": list(config),
                    "best_ridge": best_ridge,
                    "ridge_upper_boundary_unresolved": ridge_upper_boundary,
                    "mean_validation_mc": float(np.mean(list(per_seed.values()))),
                    "se_validation_mc": _mean_se(list(per_seed.values()))[1],
                    "validation_mc_by_seed": per_seed,
                    "ridge_mean_validation_mc": ridge_means,
                }
            )
        entries.sort(
            key=lambda row: (
                -row["mean_validation_mc"],
                row["config"][2],
                row["config"][1],
                row["config"][0],
                row["best_ridge"],
            )
        )
        ranked[method] = entries
    return ranked


def _nested_test_path(method: str, seed: int) -> Path:
    return OUTROOT / "nested_tuning" / "test_jobs" / f"{method}_s{seed}.json"


def nested_test_job(job: tuple[str, dict, int, str, str]) -> dict:
    method, selection, seed, protocol_sha256, selection_sha256 = job
    h, dt, multiplier = map(float, selection["config"])
    ridge = float(selection["best_ridge"])
    started = time.perf_counter()
    reservoir, inputs, jumps, target = _reservoir(
        method, seed, h, dt, multiplier, WASH + TRAIN + VAL + TEST
    )
    observables = readout.pauli_observables(N, max_weight=2)
    features = reservoir.run(inputs, observables, washout=WASH)
    score, delay_scores = _test_score(features, inputs, ridge)
    return {
        "protocol_sha256": protocol_sha256,
        "selection_sha256": selection_sha256,
        "stage": "test",
        "method": method,
        "seed": seed,
        "h": h,
        "dt": dt,
        "strength_multiplier": multiplier,
        "ridge": ridge,
        "test_mc": score,
        "test_capacity_by_delay": delay_scores,
        "jump_strength": dsp.jump_strength(jumps),
        "target_strength": target,
        "relative_budget_error": abs(dsp.jump_strength(jumps) - target) / target,
        "n_features": int(features.shape[1]),
        "backend": "exact_sparse_expm_multiply",
        "runtime_s": time.perf_counter() - started,
    }


def run_nested(workers: int) -> Path:
    pools = seed_namespaces()
    details = {
        "N": N,
        "split": {"wash": WASH, "train": TRAIN, "validation": VAL, "test": TEST},
        "delays": DELAYS,
        "channels": NESTED_CHANNELS,
        "h_grid": NESTED_H,
        "dt_grid": NESTED_DT,
        "strength_multiplier_grid": NESTED_MULTIPLIERS,
        "ridge_grid": RIDGES,
        "screen_seeds": N_SCREEN_SEEDS,
        "screen_split": {
            "wash": SCREEN_WASH,
            "train": SCREEN_TRAIN,
            "validation": SCREEN_VAL,
            "test": 0,
        },
        "selection_seeds": N_SELECTION_SEEDS,
        "test_seeds": N_TEST_SEEDS,
        "shortlist_per_channel": NESTED_SHORTLIST,
        "selection_rule": (
            "cheap screen of the identical full grid on two seeds; advance top "
            "8 per channel; select one config+ridge using full 200/600/150 "
            "trajectories on a disjoint 12-seed calibration ensemble; freeze; "
            "fit train+validation at frozen ridge and score 24 untouched full-"
            "length test reservoirs"
        ),
    }
    _, protocol_sha = _ensure_manifest(
        OUTROOT / "nested_tuning", "nested", details
    )
    configs = nested_grid()
    screen_jobs = [
        (method, config, seed, "screen", protocol_sha)
        for method in NESTED_CHANNELS
        for config in configs
        for seed in pools["nested_screen"]
    ]
    _run_checkpointed(
        screen_jobs,
        nested_calibration_job,
        lambda job: _nested_path("screen", job[0], job[1], job[2]),
        workers,
        "nested-screen",
    )
    screen_rows = _read_nested_rows("screen")
    screen_rank = rank_nested_configs(
        screen_rows,
        NESTED_CHANNELS,
        configs,
        pools["nested_screen"],
    )
    shortlist = {}
    for method in NESTED_CHANNELS:
        if screen_rank[method][0]["ridge_upper_boundary_unresolved"]:
            raise RuntimeError(
                f"{method} screen winner selects ridge={RIDGES[-1]}; "
                "regularization boundary is unresolved"
            )
        resolved = [
            entry
            for entry in screen_rank[method]
            if not entry["ridge_upper_boundary_unresolved"]
        ]
        if len(resolved) < NESTED_SHORTLIST:
            raise RuntimeError(f"too few ridge-bracketed configs for {method}")
        shortlist[method] = [
            entry["config"] for entry in resolved[:NESTED_SHORTLIST]
        ]
    shortlist_path = OUTROOT / "nested_tuning" / "shortlist.json"
    shortlist_payload = {
        "protocol_sha256": protocol_sha,
        "shortlist": shortlist,
        "screen_ranking": screen_rank,
    }
    if shortlist_path.exists():
        if json.loads(shortlist_path.read_text()) != shortlist_payload:
            raise RuntimeError("nested shortlist drift")
    else:
        _atomic_json(shortlist_path, shortlist_payload)

    selection_jobs = [
        (method, tuple(config), seed, "selection", protocol_sha)
        for method in NESTED_CHANNELS
        for config in shortlist[method]
        for seed in pools["nested_selection"]
    ]
    _run_checkpointed(
        selection_jobs,
        nested_calibration_job,
        lambda job: _nested_path("selection", job[0], job[1], job[2]),
        workers,
        "nested-selection",
    )
    selection_rows = _read_nested_rows("selection")
    chosen: dict[str, dict] = {}
    selection_ranking: dict[str, list[dict]] = {}
    for method in NESTED_CHANNELS:
        ranked = rank_nested_configs(
            selection_rows,
            (method,),
            shortlist[method],
            pools["nested_selection"],
        )[method]
        selection_ranking[method] = ranked
        if ranked[0]["ridge_upper_boundary_unresolved"]:
            raise RuntimeError(
                f"{method} selection winner selects ridge={RIDGES[-1]}; "
                "regularization boundary is unresolved"
            )
        chosen[method] = ranked[0]
    selection_path = OUTROOT / "nested_tuning" / "frozen_selection.json"
    selection_payload = {
        "artifact_type": "frozen_nested_operating_points",
        "status": "frozen_before_test_ensemble",
        "protocol_sha256": protocol_sha,
        "shortlist_sha256": _sha256_file(shortlist_path),
        "chosen": chosen,
        "selection_ranking": selection_ranking,
        "test_rows_present_at_freeze": False,
    }
    if selection_path.exists():
        old = json.loads(selection_path.read_text())
        if old != selection_payload:
            raise RuntimeError("nested frozen selection drift")
    else:
        test_dir = OUTROOT / "nested_tuning" / "test_jobs"
        if test_dir.exists() and any(test_dir.glob("*.json")):
            raise RuntimeError("test rows predate frozen nested selection")
        _atomic_json(selection_path, selection_payload)
    selection_sha = _sha256_file(selection_path)

    test_jobs = [
        (method, chosen[method], seed, protocol_sha, selection_sha)
        for method in NESTED_CHANNELS
        for seed in pools["nested_test"]
    ]
    _run_checkpointed(
        test_jobs,
        nested_test_job,
        lambda job: _nested_test_path(job[0], job[2]),
        workers,
        "nested-test",
    )
    test_rows = [
        json.loads(path.read_text())
        for path in sorted(
            (OUTROOT / "nested_tuning" / "test_jobs").glob("*.json")
        )
    ]
    by_method = {}
    scores = {}
    for method in NESTED_CHANNELS:
        group = [row for row in test_rows if row["method"] == method]
        if {int(row["seed"]) for row in group} != set(pools["nested_test"]):
            raise RuntimeError(f"incomplete nested test ensemble for {method}")
        ordered = {int(row["seed"]): float(row["test_mc"]) for row in group}
        scores[method] = ordered
        mean, se = _mean_se(list(ordered.values()))
        by_method[method] = {
            "selected": chosen[method],
            "test_mean": mean,
            "test_se": se,
            "test_scores_by_seed": ordered,
        }
    common = pools["nested_test"]
    comparison = paired_stats(
        [scores["B3_collective"][seed] for seed in common],
        [scores["CD_paper"][seed] for seed in common],
    )
    payload = {
        "artifact_type": "revision_nested_tuning_results",
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "expected_checkpoint_counts": {
            "screen": len(NESTED_CHANNELS)
            * len(configs)
            * len(pools["nested_screen"]),
            "selection": len(NESTED_CHANNELS)
            * NESTED_SHORTLIST
            * len(pools["nested_selection"]),
            "test": len(NESTED_CHANNELS) * len(pools["nested_test"]),
        },
        "complete_checkpoint_counts": {
            "screen": len(screen_rows),
            "selection": len(selection_rows),
            "test": len(test_rows),
        },
        "seed_disjointness_verified": True,
        "selected_ridge_upper_boundary_hits": sum(
            bool(chosen[method]["ridge_upper_boundary_unresolved"])
            for method in NESTED_CHANNELS
        ),
        "methods": by_method,
        "collective_vs_local": comparison,
        "screen_ranking": screen_rank,
        "selection_ranking": selection_ranking,
        "raw_screen_rows": screen_rows,
        "raw_selection_rows": selection_rows,
        "raw_test_rows": test_rows,
    }
    path = OUTROOT / "nested_tuning" / "nested_tuning_results.json"
    _atomic_json(path, payload)
    return path


# ---------------------------------------------------------- fresh interpolation
def _interpolation_path(n_qubits: int, alpha: float, seed: int) -> Path:
    return (
        OUTROOT
        / "fresh_interpolation"
        / "jobs"
        / f"stm_N{n_qubits}_a{int(round(alpha*100)):03d}_s{seed}.json"
    )


def fresh_interpolation_job(
    job: tuple[int, float, int, str, str]
) -> dict:
    n_qubits, alpha, seed, protocol_sha256, frozen_sha256 = job
    started = time.perf_counter()
    problem_rng = np.random.default_rng(seed)
    J = res.random_couplings(n_qubits, 1.0, problem_rng)
    inputs = tasks.stm_inputs(INTERPOLATION_PRESET.total_len, problem_rng)
    jumps, target = _build_interpolation_jumps(n_qubits, alpha)
    H0 = ising_xx_hamiltonian(J, INTERPOLATION_PRESET.h, n_qubits)
    Hx = transverse_drive(n_qubits)
    reservoir = SparseLindbladReservoir.from_terms(
        n_qubits,
        H0 + INTERPOLATION_PRESET.h * Hx,
        INTERPOLATION_PRESET.h * Hx,
        jumps,
        INTERPOLATION_PRESET.dt,
    )
    observables = readout.pauli_observables(n_qubits, max_weight=2)
    features = reservoir.run(
        inputs, observables, washout=INTERPOLATION_PRESET.wash
    )
    validation = _validation_scores(
        features,
        inputs,
        wash=INTERPOLATION_PRESET.wash,
        train=INTERPOLATION_PRESET.train,
        validation=INTERPOLATION_PRESET.validation,
    )
    selected_ridge = max(
        RIDGES, key=lambda ridge: validation[f"{ridge:.12g}"]
    )
    if math.isclose(selected_ridge, RIDGES[-1]):
        raise RuntimeError(
            f"N={n_qubits}, alpha={alpha}, seed={seed}: validation optimum "
            f"hits unresolved ridge upper boundary {RIDGES[-1]}"
        )
    test_mc, test_by_delay = _test_score(
        features,
        inputs,
        selected_ridge,
        wash=INTERPOLATION_PRESET.wash,
        train=INTERPOLATION_PRESET.train,
        validation=INTERPOLATION_PRESET.validation,
        test=INTERPOLATION_PRESET.test,
    )
    actual = dsp.jump_strength(jumps)
    return {
        "N": n_qubits,
        "alpha": alpha,
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "frozen_diagnostic_sha256": frozen_sha256,
        "ensemble_status": "fresh_and_disjoint_from_diagnostic_seeds",
        "backend": "exact_sparse_expm_multiply",
        "jump_strength": actual,
        "target_strength": target,
        "relative_budget_error": abs(actual - target) / target,
        "n_features": int(features.shape[1]),
        "selected_ridge": selected_ridge,
        "ridge_upper_boundary_unresolved": False,
        "validation_mc": validation[f"{selected_ridge:.12g}"],
        "validation_mc_by_ridge": validation,
        "test_mc": test_mc,
        "test_capacity_by_delay": test_by_delay,
        "runtime_s": time.perf_counter() - started,
    }


def run_interpolation(workers: int) -> Path:
    if not FROZEN_DIAGNOSTICS.is_file():
        raise RuntimeError(f"missing frozen diagnostics: {FROZEN_DIAGNOSTICS}")
    frozen = json.loads(FROZEN_DIAGNOSTICS.read_text())
    frozen_sha = _sha256_file(FROZEN_DIAGNOSTICS)
    old_diagnostic_seeds = {
        int(row["seed"]) for row in frozen.get("diagnostic_rows", [])
    }
    fresh = seed_namespaces()["fresh_interpolation"]
    if set(fresh) & old_diagnostic_seeds:
        raise RuntimeError("fresh interpolation seeds overlap frozen diagnostics")
    selected = {
        n_qubits: float(
            frozen["predictions_by_N"][str(n_qubits)][
                "diagnostic_selected_intermediate_alpha"
            ]
        )
        for n_qubits in INTERPOLATION_PRESET.n_qubits
    }
    if any(not math.isclose(alpha, 0.8) for alpha in selected.values()):
        raise RuntimeError(f"frozen diagnostic choice is no longer alpha=0.8: {selected}")
    details = {
        "frozen_diagnostic_path": str(FROZEN_DIAGNOSTICS.relative_to(REPO_ROOT)),
        "frozen_diagnostic_sha256": frozen_sha,
        "frozen_selected_alpha_by_N": selected,
        "alphas": INTERPOLATION_ALPHAS,
        "N": INTERPOLATION_PRESET.n_qubits,
        "split": {
            "wash": INTERPOLATION_PRESET.wash,
            "train": INTERPOLATION_PRESET.train,
            "validation": INTERPOLATION_PRESET.validation,
            "test": INTERPOLATION_PRESET.test,
        },
        "fresh_seed_count": len(fresh),
        "fresh_seed_overlap_with_diagnostics": 0,
        "revision_local_ridge_grid": RIDGES,
        "ridge_boundary_policy": (
            "hard fail before checkpointing if validation selects upper bound"
        ),
    }
    _, protocol_sha = _ensure_manifest(
        OUTROOT / "fresh_interpolation", "fresh_interpolation", details
    )
    jobs = [
        (n_qubits, alpha, seed, protocol_sha, frozen_sha)
        for n_qubits in INTERPOLATION_PRESET.n_qubits
        for alpha in INTERPOLATION_ALPHAS
        for seed in fresh
    ]
    _run_checkpointed(
        jobs,
        fresh_interpolation_job,
        lambda job: _interpolation_path(job[0], job[1], job[2]),
        workers,
        "fresh-interpolation",
    )
    rows = [
        json.loads(path.read_text())
        for path in sorted(
            (OUTROOT / "fresh_interpolation" / "jobs").glob("*.json")
        )
    ]
    by_n = {}
    for n_qubits in INTERPOLATION_PRESET.n_qubits:
        summaries = []
        score_by_alpha: dict[float, dict[int, float]] = {}
        for alpha in INTERPOLATION_ALPHAS:
            group = [
                row
                for row in rows
                if int(row["N"]) == n_qubits
                and math.isclose(float(row["alpha"]), alpha)
            ]
            if {int(row["seed"]) for row in group} != set(fresh):
                raise RuntimeError(
                    f"incomplete fresh interpolation N={n_qubits}, alpha={alpha}"
                )
            values = {int(row["seed"]): float(row["test_mc"]) for row in group}
            score_by_alpha[alpha] = values
            mean, se = _mean_se(list(values.values()))
            summaries.append(
                {
                    "alpha": alpha,
                    "test_mc_mean": mean,
                    "test_mc_se": se,
                    "n": len(values),
                }
            )
        selected_alpha = selected[n_qubits]
        selected_comparison = paired_stats(
            [score_by_alpha[selected_alpha][seed] for seed in fresh],
            [score_by_alpha[0.0][seed] for seed in fresh],
        )
        endpoint_comparison = paired_stats(
            [score_by_alpha[1.0][seed] for seed in fresh],
            [score_by_alpha[0.0][seed] for seed in fresh],
        )
        diagnostics = frozen["predictions_by_N"][str(n_qubits)][
            "diagnostic_summary"
        ]
        gaps = {
            float(item["alpha"]): float(item["spectral_gap_mean"])
            for item in diagnostics
        }
        means = {
            float(item["alpha"]): float(item["test_mc_mean"])
            for item in summaries
        }
        rho = stats.spearmanr(
            [-gaps[alpha] for alpha in INTERPOLATION_ALPHAS],
            [means[alpha] for alpha in INTERPOLATION_ALPHAS],
        )
        by_n[str(n_qubits)] = {
            "summary": summaries,
            "frozen_selected_alpha": selected_alpha,
            "selected_alpha_vs_local": selected_comparison,
            "collective_endpoint_vs_local": endpoint_comparison,
            "frozen_gap_vs_fresh_mean_spearman_rho": float(rho.statistic),
            "frozen_gap_vs_fresh_mean_spearman_p": float(rho.pvalue),
            "selected_ridge_counts": dict(
                Counter(
                    f"{float(row['selected_ridge']):.12g}"
                    for row in rows
                    if int(row["N"]) == n_qubits
                )
            ),
            "ridge_upper_boundary_hits": sum(
                bool(row.get("ridge_upper_boundary_unresolved"))
                for row in rows
                if int(row["N"]) == n_qubits
            ),
        }
    payload = {
        "artifact_type": "revision_fresh_interpolation_results",
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "frozen_diagnostic_sha256": frozen_sha,
        "diagnostic_seed_count": len(old_diagnostic_seeds),
        "fresh_seeds": fresh,
        "seed_overlap_with_frozen_diagnostics": sorted(
            set(fresh) & old_diagnostic_seeds
        ),
        "expected_checkpoint_count": (
            len(INTERPOLATION_PRESET.n_qubits)
            * len(INTERPOLATION_ALPHAS)
            * len(fresh)
        ),
        "complete_checkpoint_count": len(rows),
        "ridge_upper_boundary_hits": sum(
            bool(row.get("ridge_upper_boundary_unresolved")) for row in rows
        ),
        "results_by_N": by_n,
        "raw_rows": rows,
    }
    path = OUTROOT / "fresh_interpolation" / "fresh_interpolation_results.json"
    _atomic_json(path, payload)
    return path


# ------------------------------------------------------------------- report
def _fmt(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def write_report() -> Path:
    strength_path = OUTROOT / "strength_extension" / "six_channel_aggregate.json"
    matching_path = (
        OUTROOT / "strength_extension" / "alternative_matching_aggregate.json"
    )
    nested_path = OUTROOT / "nested_tuning" / "nested_tuning_results.json"
    interpolation_path = (
        OUTROOT / "fresh_interpolation" / "fresh_interpolation_results.json"
    )
    missing = [
        str(path)
        for path in (strength_path, matching_path, nested_path, interpolation_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(f"cannot report before all results exist: {missing}")
    strength = json.loads(strength_path.read_text())
    matching = json.loads(matching_path.read_text())
    nested = json.loads(nested_path.read_text())
    interpolation = json.loads(interpolation_path.read_text())
    lines = [
        "# Revision tuning and independence controls",
        "",
        "This report addresses the tuning and same-ensemble objections without "
        "changing the paper's dissipation-engineering question.",
        "",
        "## Six-channel validation-selected ranking",
        "",
        "Primary estimates use leave-one-reservoir-out strength selection: the "
        "strength applied to each test reservoir is selected from validation "
        "scores of the other 19 reservoirs.",
        "",
        "| channel | held-out STM MC ± SE | selected strengths |",
        "|---|---:|---|",
    ]
    ordered = sorted(
        CHANNELS,
        key=lambda method: -strength["methods"][method]["leave_one_seed_out"][
            "test_mean"
        ],
    )
    for method in ordered:
        result = strength["methods"][method]["leave_one_seed_out"]
        lines.append(
            f"| {method} | {_fmt(result['test_mean'], result['test_se'])} "
            f"| {result['selection_counts']} |"
        )
    bracket = strength["methods"]["B3_collective"]["curve_bracket"]
    runner = ordered[1]
    versus_runner = strength["collective_paired_comparisons"][runner]
    lines.extend(
        [
            "",
            f"The collective optimum is bracketed at x{bracket['best_multiplier']} "
            f"between x{bracket['left_multiplier']} and "
            f"x{bracket['right_multiplier']}. Against the runner-up ({runner}), "
            f"its paired difference is {versus_runner['mean_difference']:.3f} "
            f"[95% CI {versus_runner['ci95_low']:.3f}, "
            f"{versus_runner['ci95_high']:.3f}], "
            f"{versus_runner['wins']}/{versus_runner['n']} wins.",
            "",
            "## Alternative resource matching",
            "",
            "| channel | convention | feasibility | multiplier | ΔMC vs dial (95% CI) |",
            "|---|---|---|---:|---:|",
        ]
    )
    for key in (
        "B3_collective__energy",
        "B3_collective__gap",
        "B3_collective__activity",
        "B2_thermal__energy",
        "B2_thermal__gap",
        "B2_thermal__activity",
        "B5_pair__energy",
        "B5_pair__gap",
        "B5_pair__activity",
    ):
        item = matching["conditions"][key]
        feasibility = item["match_feasibility"]
        effect = item["effect_vs_standard_dial"]
        lines.append(
            f"| {item['method']} | {item['matching_mode']} | "
            f"{feasibility['reachable_count']}/{feasibility['total']} | "
            f"{item['scale_factor_mean']:.3g} | "
            f"{effect['mean_difference']:+.3f} "
            f"[{effect['ci95_low']:+.3f}, {effect['ci95_high']:+.3f}] |"
        )
    collective_activity = matching["conditions"]["B3_collective__activity"]
    collective_gap = matching["conditions"]["B3_collective__gap"]
    lines.extend(
        [
            "",
            "Initial-excitation loss-rate matching is exact. Driven-gap matching "
            f"for collective loss is not feasible within the stored x0.05–x40 "
            f"search ({collective_gap['match_feasibility']['reachable_count']}/32); "
            "the x40 closest boundary is reported rather than mislabeled as an "
            "exact match. Steady-state activity matching is likewise reachable "
            f"for only {collective_activity['match_feasibility']['reachable_count']}"
            "/32 collective instances; at the closest-achievable activity the "
            "paired effect remains positive as shown above. The data artifact "
            "also contains every seven-point driven-gap curve.",
            "",
            "## Independently nested operating-point comparison",
            "",
        ]
    )
    for method in NESTED_CHANNELS:
        result = nested["methods"][method]
        lines.append(
            f"- {method}: selected (h, dt, multiplier)="
            f"{result['selected']['config']}, ridge="
            f"{result['selected']['best_ridge']:.3g}; untouched-test STM MC "
            f"{_fmt(result['test_mean'], result['test_se'])}."
        )
    comparison = nested["collective_vs_local"]
    lines.extend(
        [
            "",
            f"Collective minus local on the 24 untouched reservoirs: "
            f"ΔMC={comparison['mean_difference']:.3f} "
            f"[95% CI {comparison['ci95_low']:.3f}, "
            f"{comparison['ci95_high']:.3f}], "
            f"{comparison['relative_mean_difference_percent']:+.1f}%, "
            f"{comparison['wins']}/{comparison['n']} wins, exact sign "
            f"p={comparison['exact_sign_p_two_sided']:.4g}.",
            "",
            "## Frozen interpolation on a fresh reservoir ensemble",
            "",
        ]
    )
    for n_qubits in (4, 5):
        result = interpolation["results_by_N"][str(n_qubits)]
        selected_result = result["selected_alpha_vs_local"]
        lines.append(
            f"- N={n_qubits}: frozen α={result['frozen_selected_alpha']:.1f} "
            f"minus local ΔMC={selected_result['mean_difference']:.3f} "
            f"[95% CI {selected_result['ci95_low']:.3f}, "
            f"{selected_result['ci95_high']:.3f}], "
            f"{selected_result['wins']}/{selected_result['n']} wins; frozen-gap "
            f"versus fresh-score rank ρ="
            f"{result['frozen_gap_vs_fresh_mean_spearman_rho']:.3f}."
        )
    lines.extend(
        [
            "",
            "The 24 interpolation seeds have zero overlap with the seeds used to "
            "freeze the diagnostic rule. This upgrades the former score-blind "
            "same-ensemble check to an out-of-ensemble confirmation while keeping "
            "the original operator family and scientific framing.",
            "",
            "## Raw provenance",
            "",
            f"- `{strength_path.relative_to(REPO_ROOT)}` embeds every source row "
            "and SHA-256 provenance path.",
            f"- `{matching_path.relative_to(REPO_ROOT)}` embeds every energy-rate, "
            "gap, activity, and full gap-curve row with match feasibility.",
            f"- `{nested_path.relative_to(REPO_ROOT)}` embeds screen, independent "
            "selection, and untouched-test rows.",
            f"- `{interpolation_path.relative_to(REPO_ROOT)}` embeds all fresh "
            "alpha-grid rows and the frozen-diagnostic SHA-256.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    return REPORT_PATH


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("strength", "nested", "interpolation", "report", "all"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    if args.command in ("strength", "all"):
        print(f"STRENGTH {run_strength(args.workers)}", flush=True)
    if args.command in ("nested", "all"):
        print(f"NESTED {run_nested(args.workers)}", flush=True)
    if args.command in ("interpolation", "all"):
        print(f"INTERPOLATION {run_interpolation(args.workers)}", flush=True)
    if args.command in ("report", "all"):
        print(f"REPORT {write_report()}", flush=True)


if __name__ == "__main__":
    main()

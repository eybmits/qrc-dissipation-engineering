"""Fresh-ensemble confirmation of the sealed local-to-collective interpolation.

This fresh-only driver deliberately leaves ``run_revision_tuning.py`` byte-exact
for its already-running strength/nested manifests.  It reuses that driver's
expanded-ridge exact task job, but owns the fresh manifest and aggregation:

* 24 Hamiltonian/input seeds disjoint from the 20 diagnostic seeds;
* the complete alpha={0,.2,...,1} grid at N=4,5 (288 atomic checkpoints);
* an exhaustive 6!=720 exact Spearman rank-permutation p value;
* exact sign tests for paired candidate/endpoint contrasts;
* embedded frozen diagnostic summaries and rows, so no legacy result directory
  is needed to rebuild the prospective figure;
* byte-exact source snapshots hash-linked to this stage manifest.

This is an out-of-ensemble confirmation within the same operator interpolation,
not an out-of-family claim.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from pathlib import Path
from typing import Sequence

import _paths  # noqa: F401
import numpy as np
from scipy import stats

import run_revision_tuning as base


PROTOCOL_VERSION = "revision-fresh-interpolation-v1-2026-07-24"
HELPER_RELATIVE = "experiments/run_revision_fresh_interpolation.py"
MAIN_RELATIVE = "experiments/run_revision_tuning.py"
OUTDIR = base.OUTROOT / "fresh_interpolation"
MANIFEST_PATH = OUTDIR / "manifest.json"
RESULT_PATH = OUTDIR / "fresh_interpolation_results.json"
SNAPSHOT_DIR = OUTDIR / "source_snapshot"
SNAPSHOT_MANIFEST = SNAPSHOT_DIR / "manifest.json"


def exact_spearman_permutation(
    predictor: Sequence[float], outcome: Sequence[float]
) -> dict:
    """Spearman rho with exhaustive two-sided permutation p for six points."""
    x = np.asarray(predictor, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("predictor/outcome must be equal-length vectors")
    if len(x) > 9:
        raise ValueError("exact enumeration is restricted to at most 9 points")
    observed = float(stats.spearmanr(x, y).statistic)
    extreme = 0
    total = math.factorial(len(y))
    for permutation in itertools.permutations(y.tolist()):
        rho = float(stats.spearmanr(x, permutation).statistic)
        extreme += int(abs(rho) >= abs(observed) - 1e-12)
    return {
        "rho": observed,
        "exact_p_two_sided": float(extreme / total),
        "exact": True,
        "n_permutations": total,
    }


def _load_frozen() -> tuple[dict, str, set[int]]:
    path = base.FROZEN_DIAGNOSTICS
    if not path.is_file():
        raise RuntimeError(f"missing frozen diagnostic artifact: {path}")
    frozen = json.loads(path.read_text())
    digest = base._sha256_file(path)
    diagnostic_seeds = {
        int(row["seed"]) for row in frozen.get("diagnostic_rows", [])
    }
    if len(diagnostic_seeds) != 20:
        raise RuntimeError(
            f"expected 20 frozen diagnostic seeds, found {len(diagnostic_seeds)}"
        )
    for n_qubits in base.INTERPOLATION_PRESET.n_qubits:
        selected = float(
            frozen["predictions_by_N"][str(n_qubits)][
                "diagnostic_selected_intermediate_alpha"
            ]
        )
        if not math.isclose(selected, 0.8):
            raise RuntimeError(
                f"sealed selected alpha changed for N={n_qubits}: {selected}"
            )
    return frozen, digest, diagnostic_seeds


def protocol_payload() -> dict:
    frozen, frozen_sha, diagnostic_seeds = _load_frozen()
    fresh = base.seed_namespaces()["fresh_interpolation"]
    overlap = sorted(set(fresh) & diagnostic_seeds)
    if overlap:
        raise RuntimeError(f"fresh/diagnostic seed overlap: {overlap}")
    sources = base.source_hashes()
    sources[HELPER_RELATIVE] = base._sha256_file(base.REPO_ROOT / HELPER_RELATIVE)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "fresh_interpolation",
        "scientific_sources_sha256": sources,
        "frozen_diagnostic_source": {
            "path": str(base.FROZEN_DIAGNOSTICS.relative_to(base.REPO_ROOT)),
            "sha256": frozen_sha,
            "diagnostic_seed_count": len(diagnostic_seeds),
        },
        "task_protocol": {
            "operator_family_status": (
                "out-of-ensemble confirmation within the same "
                "local-to-collective interpolation; not out-of-family"
            ),
            "N": list(base.INTERPOLATION_PRESET.n_qubits),
            "alphas": list(base.INTERPOLATION_ALPHAS),
            "fresh_task_seed_count": len(fresh),
            "fresh_task_seeds": fresh,
            "seed_overlap_with_frozen_diagnostics": overlap,
            "split": {
                "wash": base.INTERPOLATION_PRESET.wash,
                "train": base.INTERPOLATION_PRESET.train,
                "validation": base.INTERPOLATION_PRESET.validation,
                "test": base.INTERPOLATION_PRESET.test,
            },
            "delays": list(base.INTERPOLATION_PRESET.delays),
            "h": base.INTERPOLATION_PRESET.h,
            "dt": base.INTERPOLATION_PRESET.dt,
            "ridge_grid": list(base.RIDGES),
            "ridge_boundary_policy": (
                "hard fail before checkpoint if validation selects upper bound"
            ),
            "expected_checkpoint_count": (
                len(base.INTERPOLATION_PRESET.n_qubits)
                * len(base.INTERPOLATION_ALPHAS)
                * len(fresh)
            ),
        },
    }


def freeze_manifest() -> tuple[dict, str]:
    protocol = protocol_payload()
    fingerprint = base._sha256_json(protocol)
    payload = {
        "artifact_type": "revision_tuning_manifest",
        "manifest_status": "frozen_before_stage_rows",
        "protocol": protocol,
        "protocol_sha256": fingerprint,
    }
    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text())
        if base._canonical_json(old) != base._canonical_json(payload):
            raise RuntimeError("fresh manifest drift")
    else:
        jobs = OUTDIR / "jobs"
        if jobs.exists() and any(jobs.glob("*.json")):
            raise RuntimeError("fresh rows exist before manifest freeze")
        base._atomic_json(MANIFEST_PATH, payload)
    return payload, fingerprint


def snapshot_sources(protocol_sha256: str, source_hashes: dict[str, str]) -> Path:
    """Copy the exact main/helper bytes and write the strict linkage manifest."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    main_snapshot = SNAPSHOT_DIR / "run_revision_tuning.py"
    helper_snapshot = SNAPSHOT_DIR / "run_revision_fresh_interpolation.py"
    if not main_snapshot.exists():
        shutil.copyfile(base.REPO_ROOT / MAIN_RELATIVE, main_snapshot)
    if not helper_snapshot.exists():
        shutil.copyfile(base.REPO_ROOT / HELPER_RELATIVE, helper_snapshot)
    main_sha = base._sha256_file(main_snapshot)
    helper_sha = base._sha256_file(helper_snapshot)
    if main_sha != source_hashes[MAIN_RELATIVE]:
        raise RuntimeError("fresh main-driver snapshot hash mismatch")
    if helper_sha != source_hashes[HELPER_RELATIVE]:
        raise RuntimeError("fresh helper snapshot hash mismatch")
    payload = {
        "artifact_type": "fresh_stage_source_snapshot",
        "fresh_protocol_sha256": protocol_sha256,
        "source_path_in_protocol": MAIN_RELATIVE,
        "snapshot_path": (
            "results/revision_tuning/fresh_interpolation/source_snapshot/"
            "run_revision_tuning.py"
        ),
        "sha256": main_sha,
        "helper_path_in_protocol": HELPER_RELATIVE,
        "helper_snapshot_path": (
            "results/revision_tuning/fresh_interpolation/source_snapshot/"
            "run_revision_fresh_interpolation.py"
        ),
        "helper_sha256": helper_sha,
    }
    if SNAPSHOT_MANIFEST.exists():
        old = json.loads(SNAPSHOT_MANIFEST.read_text())
        if base._canonical_json(old) != base._canonical_json(payload):
            raise RuntimeError("fresh source-snapshot manifest drift")
    else:
        base._atomic_json(SNAPSHOT_MANIFEST, payload)
    return SNAPSHOT_MANIFEST


def _validate_row(
    row: dict,
    n_qubits: int,
    alpha: float,
    seed: int,
    protocol_sha256: str,
    frozen_sha256: str,
) -> None:
    if row.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError("fresh checkpoint protocol mismatch")
    if row.get("frozen_diagnostic_sha256") != frozen_sha256:
        raise RuntimeError("fresh checkpoint diagnostic hash mismatch")
    if (
        int(row.get("N")) != n_qubits
        or int(row.get("seed")) != seed
        or not math.isclose(float(row.get("alpha")), alpha)
    ):
        raise RuntimeError("fresh checkpoint identity mismatch")
    if bool(row.get("ridge_upper_boundary_unresolved")):
        raise RuntimeError("fresh checkpoint has unresolved ridge boundary")
    if math.isclose(float(row["selected_ridge"]), base.RIDGES[-1]):
        raise RuntimeError("fresh checkpoint selected ridge upper bound")


def aggregate(
    manifest: dict,
    protocol_sha256: str,
    frozen: dict,
    frozen_sha256: str,
    diagnostic_seeds: set[int],
) -> Path:
    fresh = base.seed_namespaces()["fresh_interpolation"]
    rows: list[dict] = []
    for n_qubits in base.INTERPOLATION_PRESET.n_qubits:
        for alpha in base.INTERPOLATION_ALPHAS:
            for seed in fresh:
                path = base._interpolation_path(n_qubits, alpha, seed)
                if not path.is_file():
                    raise RuntimeError(f"missing fresh checkpoint {path}")
                row = json.loads(path.read_text())
                _validate_row(
                    row,
                    n_qubits,
                    alpha,
                    seed,
                    protocol_sha256,
                    frozen_sha256,
                )
                rows.append(row)
    expected = (
        len(base.INTERPOLATION_PRESET.n_qubits)
        * len(base.INTERPOLATION_ALPHAS)
        * len(fresh)
    )
    if len(rows) != expected:
        raise RuntimeError(f"fresh row count {len(rows)}/{expected}")

    results_by_n: dict[str, dict] = {}
    for n_qubits in base.INTERPOLATION_PRESET.n_qubits:
        score_by_alpha: dict[float, dict[int, float]] = {}
        summary: list[dict] = []
        ridge_counts: dict[str, int] = {}
        for alpha in base.INTERPOLATION_ALPHAS:
            group = [
                row
                for row in rows
                if int(row["N"]) == n_qubits
                and math.isclose(float(row["alpha"]), alpha)
            ]
            scores = {int(row["seed"]): float(row["test_mc"]) for row in group}
            if set(scores) != set(fresh):
                raise RuntimeError(f"incomplete N={n_qubits}, alpha={alpha}")
            score_by_alpha[alpha] = scores
            mean, se = base._mean_se(list(scores.values()))
            summary.append(
                {
                    "alpha": alpha,
                    "test_mc_mean": mean,
                    "test_mc_se": se,
                    "n": len(scores),
                }
            )
            for row in group:
                key = f"{float(row['selected_ridge']):.12g}"
                ridge_counts[key] = ridge_counts.get(key, 0) + 1

        frozen_n = frozen["predictions_by_N"][str(n_qubits)]
        selected_alpha = float(
            frozen_n["diagnostic_selected_intermediate_alpha"]
        )
        selected = base.paired_stats(
            [score_by_alpha[selected_alpha][seed] for seed in fresh],
            [score_by_alpha[0.0][seed] for seed in fresh],
        )
        endpoint = base.paired_stats(
            [score_by_alpha[1.0][seed] for seed in fresh],
            [score_by_alpha[0.0][seed] for seed in fresh],
        )
        gaps = {
            float(item["alpha"]): float(item["spectral_gap_mean"])
            for item in frozen_n["diagnostic_summary"]
        }
        means = {
            float(item["alpha"]): float(item["test_mc_mean"])
            for item in summary
        }
        rank = exact_spearman_permutation(
            [-gaps[alpha] for alpha in base.INTERPOLATION_ALPHAS],
            [means[alpha] for alpha in base.INTERPOLATION_ALPHAS],
        )
        results_by_n[str(n_qubits)] = {
            "summary": summary,
            "frozen_selected_alpha": selected_alpha,
            "selected_alpha_vs_local": selected,
            "selected_alpha_test_label": "exact paired sign test",
            "collective_endpoint_vs_local": endpoint,
            "collective_endpoint_test_label": "exact paired sign test",
            "frozen_gap_vs_fresh_mean_spearman_rho": rank["rho"],
            "frozen_gap_vs_fresh_mean_exact_permutation_p": rank[
                "exact_p_two_sided"
            ],
            "frozen_gap_vs_fresh_mean_rank_test_exact": rank["exact"],
            "frozen_gap_vs_fresh_mean_rank_permutations": rank[
                "n_permutations"
            ],
            "selected_ridge_counts": ridge_counts,
            "ridge_upper_boundary_hits": 0,
        }

    frozen_rows = [
        row
        for row in frozen["diagnostic_rows"]
        if int(row["N"]) in base.INTERPOLATION_PRESET.n_qubits
    ]
    payload = {
        "artifact_type": "revision_fresh_interpolation_results",
        "status": "complete",
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": base._sha256_file(MANIFEST_PATH),
        "frozen_diagnostic_source": {
            "path": str(
                base.FROZEN_DIAGNOSTICS.relative_to(base.REPO_ROOT)
            ),
            "sha256": frozen_sha256,
            "diagnostic_seed_count": len(diagnostic_seeds),
        },
        "frozen_diagnostic_predictions_by_N": {
            str(n): frozen["predictions_by_N"][str(n)]
            for n in base.INTERPOLATION_PRESET.n_qubits
        },
        "frozen_diagnostic_rows": frozen_rows,
        "operator_family_status": (
            "out-of-ensemble confirmation within the same "
            "local-to-collective interpolation; not out-of-family"
        ),
        "diagnostic_seed_count": len(diagnostic_seeds),
        "fresh_task_seed_count": len(fresh),
        "fresh_seeds": fresh,
        "seed_overlap_with_frozen_diagnostics": sorted(
            set(fresh) & diagnostic_seeds
        ),
        "expected_checkpoint_count": expected,
        "complete_checkpoint_count": len(rows),
        "ridge_upper_boundary_hits": 0,
        "results_by_N": results_by_n,
        "raw_rows": rows,
    }
    base._atomic_json(RESULT_PATH, payload)
    return RESULT_PATH


def run(workers: int) -> Path:
    frozen, frozen_sha, diagnostic_seeds = _load_frozen()
    manifest, protocol_sha = freeze_manifest()
    snapshot_sources(
        protocol_sha, manifest["protocol"]["scientific_sources_sha256"]
    )
    fresh = base.seed_namespaces()["fresh_interpolation"]
    jobs = [
        (n_qubits, alpha, seed, protocol_sha, frozen_sha)
        for n_qubits in base.INTERPOLATION_PRESET.n_qubits
        for alpha in base.INTERPOLATION_ALPHAS
        for seed in fresh
    ]
    base._run_checkpointed(
        jobs,
        base.fresh_interpolation_job,
        lambda job: base._interpolation_path(job[0], job[1], job[2]),
        workers,
        "fresh-interpolation",
    )
    return aggregate(
        manifest, protocol_sha, frozen, frozen_sha, diagnostic_seeds
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(f"FRESH COMPLETE {run(args.workers)}", flush=True)


if __name__ == "__main__":
    main()

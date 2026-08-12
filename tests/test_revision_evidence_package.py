"""Tests for the deterministic reviewer-facing revision evidence package."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import math
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_revision_evidence_package.py"
)
SPEC = importlib.util.spec_from_file_location("revision_evidence_package", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package
SPEC.loader.exec_module(package)

ACTIVITY_FIXTURE_PROTOCOL_VERSION = (
    "activity-matched-response-v3-2026-07-25"
)
ACTIVITY_FIXTURE_BONFERRONI_CRITICAL = 2.8073356837675227


def _write(path: Path, data: str | bytes = "fixture\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")


def _safe_tar_bytes(name: str) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as bundle:
            payload = b"{}\n"
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(payload))
    return raw.getvalue()


def _safe_zip_bytes(name: str) -> bytes:
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w") as bundle:
        bundle.writestr(name, b"{}\n")
    return raw.getvalue()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _protocol_hash(protocol: dict) -> str:
    encoded = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary(values: list[float]) -> dict:
    mean = math.fsum(values) / len(values)
    se = math.sqrt(
        math.fsum((value - mean) ** 2 for value in values)
        / (len(values) - 1)
        / len(values)
    )
    return {"n": len(values), "mean": mean, "se": se}


def _paired_effect(candidate: list[float], reference: list[float]) -> dict:
    differences = [
        candidate_value - reference_value
        for candidate_value, reference_value in zip(candidate, reference)
    ]
    mean = sum(differences) / len(differences)
    candidate_mean = sum(candidate) / len(candidate)
    reference_mean = sum(reference) / len(reference)
    return {
        "n": 32,
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "mean_difference": mean,
        "se_difference": 0.01,
        "ci95_low": mean - 0.02,
        "ci95_high": mean + 0.02,
        "relative_mean_difference_percent": 100.0 * mean / reference_mean,
        "exact_sign_p_two_sided": 0.5,
        "wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "paired_differences": differences,
    }


def _complete_strength_aggregate() -> dict:
    methods = {
        "A1_heterogeneous",
        "B2_thermal",
        "B3_collective",
        "B4_loss_exchange",
        "B5_pair",
        "CD_paper",
    }
    base_grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
    seeds = list(range(20))
    raw_rows = []
    summaries = {}
    for method in sorted(methods):
        grid = (
            [*base_grid, 8.0, 16.0]
            if method in {"B3_collective", "CD_paper"}
            else base_grid
        )
        for multiplier in grid:
            raw_rows.extend(
                {
                    "method": method,
                    "mult": multiplier,
                    "seed": seed,
                }
                for seed in seeds
            )
        summaries[method] = {
            "curve_bracket": {
                "bracketed": True,
                "curve": [
                    {"multiplier": multiplier, "n": 20}
                    for multiplier in grid
                ],
            },
            "leave_one_seed_out": {
                "scores_by_seed": {str(seed): 1.0 for seed in seeds},
                "selected_multiplier_by_seed": {
                    str(seed): grid[0] for seed in seeds
                },
            },
        }
    return {
        "status": "complete",
        "expected_raw_row_count": 920,
        "complete_raw_row_count": 920,
        "expected_extended_multipliers": [8, 16],
        "collective_optimum_bracketed": True,
        "raw_rows": raw_rows,
        "methods": summaries,
    }


def _complete_alternative_matching_aggregate() -> dict:
    seeds = list(range(32))
    methods = ("B2_thermal", "B3_collective", "B5_pair")
    active_methods = (
        "A1_heterogeneous",
        "B2_thermal",
        "B3_collective",
        "B4_loss_exchange",
        "B5_pair",
        "CD_paper",
    )
    multipliers = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    reference_values = {seed: 8.0 + 0.01 * seed for seed in seeds}
    raw_match = [
        {
            "N": 5,
            "block": "R_match",
            "method": "CD_paper",
            "mode": None,
            "mult": 1.0,
            "seed": seed,
            "task": "stm",
            "value": reference_values[seed],
        }
        for seed in seeds
    ]
    raw_activity = []
    conditions = {}
    method_offsets = {
        "B2_thermal": 0.2,
        "B3_collective": 1.0,
        "B5_pair": -0.1,
    }
    for method in methods:
        for mode in ("energy", "gap", "activity"):
            rows = []
            for seed in seeds:
                if mode == "gap":
                    scale_factor = (
                        40.0
                        if method == "B3_collective"
                        or (method == "B5_pair" and seed < 3)
                        else 0.8
                    )
                elif mode == "activity":
                    scale_factor = 0.7
                else:
                    scale_factor = 1.0
                value = (
                    reference_values[seed]
                    + method_offsets[method]
                    + {"energy": 0.0, "gap": 0.1, "activity": -0.05}[mode]
                )
                if mode == "activity":
                    reachable = method == "B2_thermal" or (
                        method == "B3_collective" and seed == 0
                    )
                    row = {
                        "N": 5,
                        "block": "R_match2",
                        "method": method,
                        "mode": mode,
                        "reachable": reachable,
                        "activity_ratio": 1.0 if reachable else 0.8,
                        "scale_factor": scale_factor,
                        "seed": seed,
                        "task": "stm",
                        "value": value,
                    }
                    raw_activity.append(row)
                else:
                    row = {
                        "N": 5,
                        "block": "R_match",
                        "method": method,
                        "mode": mode,
                        "scale_factor": scale_factor,
                        "seed": seed,
                        "task": "stm",
                        "value": value,
                    }
                    raw_match.append(row)
                rows.append(row)

            rows.sort(key=lambda row: row["seed"])
            candidate = [row["value"] for row in rows]
            reference = [reference_values[row["seed"]] for row in rows]
            scales = [row["scale_factor"] for row in rows]
            if mode == "energy":
                feasibility = {
                    "status": "analytically_exact_linear_rescaling",
                    "reachable_count": 32,
                    "total": 32,
                }
            elif mode == "gap":
                reachable_count = sum(value < 39.99 for value in scales)
                feasibility = {
                    "status": (
                        "root_inside_search_interval"
                        if reachable_count == 32
                        else "upper_bound_censored_target_not_reached"
                    ),
                    "reachable_count": reachable_count,
                    "total": 32,
                    "search_interval": [0.05, 40.0],
                }
            else:
                reachable_count = sum(row["reachable"] for row in rows)
                ratios = [row["activity_ratio"] for row in rows]
                feasibility = {
                    "status": (
                        "all_exactly_reachable"
                        if reachable_count == 32
                        else "closest_achievable_activity_reported_when_unreachable"
                    ),
                    "reachable_count": reachable_count,
                    "total": 32,
                    "achieved_to_reference_ratio_mean": sum(ratios) / 32,
                    "achieved_to_reference_ratio_se": 0.01,
                    "achieved_to_reference_ratio_min": min(ratios),
                    "achieved_to_reference_ratio_max": max(ratios),
                }
            conditions[f"{method}__{mode}"] = {
                "block": "R_match2" if mode == "activity" else "R_match",
                "method": method,
                "matching_mode": mode,
                "effect_vs_standard_dial": _paired_effect(candidate, reference),
                "matched_channel_mean": sum(candidate) / 32,
                "matched_channel_se": 0.01,
                "dial_mean": sum(reference) / 32,
                "dial_se": 0.01,
                "scale_factor_mean": sum(scales) / 32,
                "scale_factor_se": 0.01,
                "scale_factor_min": min(scales),
                "scale_factor_max": max(scales),
                "match_feasibility": feasibility,
            }

    gap_rows = []
    curves = {}
    diagnostic_seeds = (0, 1, 2)
    for method_index, method in enumerate(active_methods):
        points = []
        for multiplier in multipliers:
            values = []
            for seed in diagnostic_seeds:
                value = method_index + multiplier + 0.01 * seed
                values.append(value)
                gap_rows.append(
                    {
                        "N": 5,
                        "block": "R_gapsweep",
                        "method": method,
                        "mult": multiplier,
                        "seed": seed,
                        "task": "gap",
                        "value": value,
                    }
                )
            mean = sum(values) / 3
            se = (
                math.sqrt(sum((value - mean) ** 2 for value in values) / 2)
                / math.sqrt(3)
            )
            points.append(
                {
                    "multiplier": multiplier,
                    "driven_gap_mean": mean,
                    "driven_gap_se": se,
                    "n": 3,
                }
            )
        curves[method] = points

    raw_rows = {
        "R_match": raw_match,
        "R_match2": raw_activity,
        "R_gapsweep": gap_rows,
    }
    provenance = []
    for row in raw_match:
        if row["method"] == "CD_paper":
            filename = f"R_match__CD_paper_ref_s{row['seed']}.json"
        else:
            filename = (
                f"R_match__{row['method']}_{row['mode']}_s{row['seed']}.json"
            )
        path = f"results/review_protocol/{filename}"
        provenance.append({"path": path, "sha256": _digest(path)})
    for row in raw_activity:
        path = (
            "results/review_protocol/"
            f"R_match2__{row['method']}_{row['mode']}_s{row['seed']}.json"
        )
        provenance.append({"path": path, "sha256": _digest(path)})
    for row in gap_rows:
        path = (
            "results/review_protocol/"
            f"R_gapsweep__{row['method']}_x{row['mult']:g}_s{row['seed']}.json"
        )
        provenance.append({"path": path, "sha256": _digest(path)})
    return {
        "artifact_type": "revision_alternative_matching_aggregate",
        "reference": {
            "method": "CD_paper",
            "n": 32,
            "mean": sum(reference_values.values()) / 32,
            "se": 0.01,
        },
        "conditions": conditions,
        "full_driven_gap_curves": curves,
        "raw_rows": raw_rows,
        "raw_provenance": provenance,
    }


def _complete_nested_result() -> dict:
    method_result = {
        "selected": {},
        "test_mean": 1.0,
        "test_se": 0.1,
        "test_scores_by_seed": {str(seed): 1.0 for seed in range(24)},
    }
    counts = {"screen": 512, "selection": 192, "test": 48}
    return {
        "status": "complete",
        "expected_checkpoint_counts": counts,
        "complete_checkpoint_counts": counts,
        "seed_disjointness_verified": True,
        "selected_ridge_upper_boundary_hits": 0,
        "methods": {
            "CD_paper": method_result,
            "B3_collective": method_result,
        },
        "collective_vs_local": {"n": 24},
    }


def _complete_fresh_result(
    frozen_payload: dict,
    frozen_source: dict,
    protocol_sha256: str,
) -> dict:
    by_size = {}
    for size in ("4", "5"):
        by_size[size] = {
            "summary": [
                {
                    "alpha": alpha,
                    "test_mc_mean": 1.0 + alpha,
                    "test_mc_se": 0.1,
                    "n": 24,
                }
                for alpha in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            ],
            "frozen_selected_alpha": 0.8,
            "selected_alpha_vs_local": {"n": 24},
            "collective_endpoint_vs_local": {"n": 24},
            "ridge_upper_boundary_hits": 0,
        }
    return {
        "status": "complete",
        "protocol_sha256": protocol_sha256,
        "frozen_diagnostic_source": frozen_source,
        "frozen_diagnostic_rows": frozen_payload["diagnostic_rows"],
        "frozen_diagnostic_predictions_by_N": (
            frozen_payload["predictions_by_N"]
        ),
        "expected_checkpoint_count": 288,
        "complete_checkpoint_count": 288,
        "seed_overlap_with_frozen_diagnostics": [],
        "ridge_upper_boundary_hits": 0,
        "results_by_N": by_size,
    }


def _complete_parity_results() -> dict[str, dict]:
    active_hash = "1" * 64
    reference_hash = "2" * 64
    active_methods = (
        "A1_heterogeneous",
        "B2_thermal",
        "B3_collective",
        "B4_loss_exchange",
        "B5_pair",
        "CD_paper",
    )
    active_rows = [
        {
            "method": method,
            "seed": seed,
            "status": "complete",
            "protocol_sha256": active_hash,
        }
        for method in active_methods
        for seed in range(16)
    ]
    reference_rows = [
        {
            "method": method,
            "seed": seed,
            "status": "complete",
            "protocol_sha256": reference_hash,
        }
        for method in ("FN", "B1_dephasing")
        for seed in range(16)
    ]
    boundary = {
        "upper_boundary_is_bracketed": True,
        "n_selected_maximum": 0,
        "n_unresolved_upper": 0,
    }
    return {
        "paper_protocol.json": {
            "protocol": {
                "feature_filter": {
                    "fit_on": "training rows only",
                    "threshold": 1e-12,
                }
            },
            "protocol_sha256": active_hash,
        },
        "paper_aggregate.json": {
            "status": "complete",
            "protocol_sha256": active_hash,
            "expected_checkpoints": 96,
            "complete_checkpoints": 96,
            "missing_checkpoints": [],
            "ridge_boundary_audit": boundary,
            "raw_rows": active_rows,
        },
        "paper_reference_protocol.json": {
            "protocol": {"active_protocol_sha256": active_hash},
            "protocol_sha256": reference_hash,
        },
        "paper_reference_aggregate.json": {
            "status": "complete",
            "protocol_sha256": reference_hash,
            "active_protocol_sha256": active_hash,
            "expected_checkpoints": 32,
            "complete_checkpoints": 32,
            "missing_checkpoints": [],
            "ridge_boundary_audit": boundary,
            "summary_by_method": {
                "FN": {"n_complete": 16, "selected_test_mean": 0.25},
                "B1_dephasing": {
                    "n_complete": 16,
                    "selected_test_mean": 0.0,
                },
            },
            "raw_rows": reference_rows,
        },
    }


def _complete_scaling_results() -> dict[str, dict]:
    seeds = list(range(100, 108))
    n_values = [4, 5, 6, 7, 8]
    methods = ("CD_paper", "B3_collective")
    protocol = {
        "protocol_version": "revision-controls-v1-2026-07-23",
        "control": "normalised_coupling_scaling",
        "preset": {
            "name": "paper",
            "n_seeds": 8,
            "n_values": n_values,
        },
        "seeds": seeds,
        "fresh_seed_namespace": 2026072302,
        "disjoint_from_definitive_seed_pool": True,
        "schemes": ["variance"],
        "normalisation_formulas": {"variance": "sqrt(4/(N-1))"},
        "methods": list(methods),
        "tasks": ["stm", "narma10"],
    }
    protocol_sha256 = _protocol_hash(protocol)
    rows = []
    checkpoints = {}
    for n_qubits in n_values:
        multiplier = math.sqrt(4.0 / (n_qubits - 1))
        target = float(2 ** (n_qubits - 1) * n_qubits)
        for seed in seeds:
            input_sha = _digest(f"input-{seed}")
            base_sha = _digest(f"base-{n_qubits}-{seed}")
            scaled_sha = _digest(f"scaled-{n_qubits}-{seed}")
            for method in methods:
                row = {
                    "status": "complete",
                    "control": "normalised_coupling_scaling",
                    "scheme": "variance",
                    "n_qubits": n_qubits,
                    "method": method,
                    "seed": seed,
                    "input_sha256": input_sha,
                    "base_coupling_sha256": base_sha,
                    "scaled_coupling_sha256": scaled_sha,
                    "coupling_multiplier": multiplier,
                    "jump_strength": target,
                    "target_jump_strength": target,
                    "relative_budget_error": 0.0,
                    "backend": "exact_sparse_expm_multiply",
                    "protocol_sha256": protocol_sha256,
                    "stm": {"selected_test": float(n_qubits)},
                    "narma10": {"selected_test": 0.1 * n_qubits},
                }
                rows.append(row)
                checkpoints[
                    f"paper__variance_N{n_qubits}_{method}_s{seed}.json"
                ] = row
    audit_checks = {
        "protocol_variance_only": True,
        "n_values_are_4_through_8": True,
        "seed_count_is_8": True,
        "fresh_seeds_disjoint": True,
        "both_methods_declared": True,
        "row_identities_unique": True,
        "paired_hashes_equal": True,
        "multipliers_match_formula": True,
        "n5_anchor_exact": True,
        "jump_budget_within_1e-10": True,
        "exact_backend_only": True,
    }
    aggregate = {
        "status": "complete",
        "control": "normalised_coupling_scaling",
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "expected_checkpoints": 80,
        "complete_checkpoints": 80,
        "missing_checkpoints": [],
        "ridge_boundary_audit": {
            "upper_boundary_is_bracketed": True,
            "n_unresolved_upper": 0,
        },
        "invariant_audit": {
            "production_contract_applies": True,
            "checks": audit_checks,
            "all_passed": True,
            "max_relative_jump_budget_error": 0.0,
            "pairing_violations": [],
            "multiplier_violations": [],
            "anchor_violations": [],
            "budget_violations": [],
            "backend_violations": [],
        },
        "summary_by_scheme": {
            "variance": {
                str(n_qubits): {"n_pairs": 8} for n_qubits in n_values
            }
        },
        "raw_rows": rows,
    }
    return {
        "paper_variance_protocol.json": {
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
        },
        "paper_variance_aggregate.json": aggregate,
        **checkpoints,
    }


def _write_primary_regularization_fixture(root: Path) -> None:
    methods = (
        "CD_paper",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    )
    seeds = list(range(32))
    tasks = {"stm": "capacity", "narma": "nmse"}
    source_environment = {
        "files": {
            "experiments/run_revision_primary_regularization.py": _digest(
                "primary-source"
            )
        },
        "python": "3.test",
        "numpy": "test",
        "scipy": "test",
    }
    baseline_entries = {}
    fixed_values: dict[tuple[str, str, int], float] = {}
    for task_index, task_name in enumerate(tasks):
        for method_index, method in enumerate(methods):
            for seed in seeds:
                value = 1.0 + task_index + 0.01 * method_index + 0.0001 * seed
                fixed_values[(task_name, method, seed)] = value
                key = f"{task_name}/{method}/{seed}"
                path = (
                    "results/final_protocol/"
                    f"A_table__{task_name}_N5_{method}_s{seed}"
                    "_h0.5_dt0.5_L200-600-400.json"
                )
                baseline_entries[key] = {
                    "path": path,
                    "sha256": _digest(path),
                    "value": value,
                }
    baseline_manifest = {
        "group": "results/final_protocol/A_table",
        "historical_ridge": 1e-8,
        "historical_split": {"wash": 200, "train": 600, "test": 400},
        "absolute_tolerance": 1e-9,
        "relative_tolerance": 1e-9,
        "guard_exception_methods": ["B1_dephasing"],
        "guard_exception_reason": "guarded dephasing roundoff features",
        "entries": baseline_entries,
        "entries_sha256": _protocol_hash(baseline_entries),
    }
    protocol = {
        "protocol_version": "fixture-primary-regularization",
        "methods": list(methods),
        "reference_method": "CD_paper",
        "seeds": seeds,
        "n_jobs": 224,
        "tasks": {
            "stm": {"metric": "capacity"},
            "narma": {"metric": "nmse"},
        },
        "readout": {
            "ridge_grid": [0.0, 1e-8, 1.0],
            "fixed_sensitivity_ridge": 1e-8,
            "feature_guard_fit_on": "raw training rows only",
            "feature_guard_std_threshold": 1e-12,
        },
        "source_environment": source_environment,
        "source_environment_sha256": _protocol_hash(source_environment),
        "baseline_reproduction": baseline_manifest,
    }
    protocol_hash = _protocol_hash(protocol)
    rows = []
    for method_index, method in enumerate(methods):
        for seed in seeds:
            common_hashes = {
                "coupling_sha256": _digest(f"coupling-{seed}"),
                "full_input_sha256": _digest(f"full-input-{seed}"),
                "post_wash_input_sha256": _digest(f"post-input-{seed}"),
                "target_sha256": {
                    "stm": _digest(f"stm-target-{seed}"),
                    "narma": _digest(f"narma-target-{seed}"),
                },
                "train_split_sha256": _digest(f"train-{seed}"),
                "validation_split_sha256": _digest(f"validation-{seed}"),
                "test_split_sha256": _digest(f"test-{seed}"),
            }
            task_results = {}
            for task_name, metric in tasks.items():
                fixed = fixed_values[(task_name, method, seed)]
                selected = (
                    fixed + 0.001 if metric == "capacity" else fixed - 0.001
                )
                target_count = 20 if task_name == "stm" else 1
                validation_totals = (
                    {"0": 0.5, "1e-08": 1.0, "1": 0.75}
                    if metric == "capacity"
                    else {"0": 1.0, "1e-08": 0.5, "1": 0.75}
                )
                task_results[task_name] = {
                    "metric": metric,
                    "selected_ridge": 1e-8,
                    "fixed_ridge": 1e-8,
                    "validation_by_ridge": validation_totals,
                    "validation_by_target_and_ridge": {
                        ridge: [value / target_count] * target_count
                        for ridge, value in validation_totals.items()
                    },
                    "selected_test": selected,
                    "selected_test_by_target": [
                        selected / target_count
                    ] * target_count,
                    "fixed_test": fixed,
                    "fixed_test_by_target": [
                        fixed / target_count
                    ] * target_count,
                    "selected_minus_fixed": selected - fixed,
                    "selection_improvement": 0.001,
                }
            dropped = 1 if method == "B1_dephasing" else 0
            row = {
                "protocol_sha256": protocol_hash,
                "source_environment_sha256": protocol[
                    "source_environment_sha256"
                ],
                "method": method,
                "seed": seed,
                "feature_guard": {
                    "fit_on": "training rows only",
                    "threshold": 1e-12,
                    "retained_nonbias_features": 45 - dropped,
                    "dropped_nonbias_features": dropped,
                    "retained_features_including_bias": 46 - dropped,
                },
                "jump_strength_error": 0.0,
                "task_results": task_results,
                "runtime_seconds": 1.0,
                **common_hashes,
            }
            rows.append(row)
    active = []
    guarded = []
    for row in rows:
        for task_name in tasks:
            expected = fixed_values[(task_name, row["method"], row["seed"])]
            item = {
                "task": task_name,
                "method": row["method"],
                "seed": row["seed"],
                "observed_fixed": expected,
                "sealed_primary": expected,
                "difference": 0.0,
                "absolute_difference": 0.0,
            }
            if row["method"] == "B1_dephasing":
                guarded.append({**item, "reason": "guarded exception"})
            else:
                active.append(
                    {
                        **item,
                        "tolerance": 1e-9 + 1e-9 * abs(expected),
                        "within_tolerance": True,
                    }
                )
    indexed_rows = {
        (row["method"], row["seed"]): row for row in rows
    }
    task_summaries = {}
    for task_name, metric in tasks.items():
        higher_is_better = metric == "capacity"
        method_summaries = {}
        selected_means = {}
        for method in methods:
            results = [
                indexed_rows[(method, seed)]["task_results"][task_name]
                for seed in seeds
            ]
            selected_values = [
                float(result["selected_test"]) for result in results
            ]
            selected_means[method] = _summary(selected_values)["mean"]
            method_summaries[method] = {
                "selected_test": _summary(selected_values),
                "fixed_test": _summary(
                    [float(result["fixed_test"]) for result in results]
                ),
                "selection_improvement": _summary(
                    [
                        float(result["selection_improvement"])
                        for result in results
                    ]
                ),
                "selection_better_than_fixed_count": 32,
                "selection_equal_to_fixed_count": 0,
                "selected_ridge_counts": {
                    "0": 0,
                    "1e-08": 32,
                    "1": 0,
                },
            }
        ranking_methods = sorted(
            methods,
            key=lambda method: (
                -selected_means[method]
                if higher_is_better
                else selected_means[method],
                method,
            ),
        )
        local_scores = {
            seed: float(
                indexed_rows[("CD_paper", seed)]["task_results"][task_name][
                    "selected_test"
                ]
            )
            for seed in seeds
        }
        comparisons = {}
        for method in methods:
            if method == "CD_paper":
                continue
            differences = []
            for seed in seeds:
                score = float(
                    indexed_rows[(method, seed)]["task_results"][task_name][
                        "selected_test"
                    ]
                )
                differences.append(
                    score - local_scores[seed]
                    if higher_is_better
                    else local_scores[seed] - score
                )
            comparisons[method] = {
                "method_advantage_over_local": _summary(differences),
                "method_better_count": sum(value > 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        task_summaries[task_name] = {
            "metric": metric,
            "direction": (
                "higher is better" if higher_is_better else "lower is better"
            ),
            "ranking": [
                {
                    "method": method,
                    "rank": rank,
                    "mean_selected_test": selected_means[method],
                }
                for rank, method in enumerate(ranking_methods, start=1)
            ],
            "method_summaries": method_summaries,
            "paired_vs_uniform_local": comparisons,
            "per_seed_winner_counts": {
                method: 32 if method == ranking_methods[0] else 0
                for method in methods
            },
        }
    aggregate = {
        "status": "complete",
        "protocol_sha256": protocol_hash,
        "source_environment_sha256": protocol["source_environment_sha256"],
        "n_jobs": 224,
        "invariant_audit": {
            "passed": True,
            "error_count": 0,
            "errors": [],
            "expected_jobs": 224,
            "observed_jobs": 224,
        },
        "ridge_boundary_audit": {
            "passed": True,
            "unresolved_upper_boundary_count": 0,
            "unresolved_upper_boundary": [],
        },
        "feature_guard_audit": {
            "passed": True,
            "threshold": 1e-12,
            "by_method": {
                method: {"jobs": 32} for method in methods
            },
        },
        "baseline_reproduction_audit": {
            "passed": True,
            "active_comparison_count": 384,
            "guarded_exception_count": 64,
            "guard_exception_methods": ["B1_dephasing"],
            "maximum_active_absolute_difference_by_task": {
                "stm": 0.0,
                "narma": 0.0,
            },
            "violations": [],
            "active_comparisons": active,
            "guarded_exceptions": guarded,
        },
        "task_summaries": task_summaries,
        "rows": rows,
    }
    directory = root / "results/revision_primary_regularization"
    _write(directory / "protocol.json", json.dumps(protocol))
    _write(directory / "aggregate.json", json.dumps(aggregate))
    for row in rows:
        _write(
            directory / "jobs" / f"{row['method']}__s{row['seed']}.json",
            json.dumps(row),
        )


def _write_collective_full_input_fixture(root: Path) -> None:
    seeds = list(range(10))
    s_grid = [index / 20.0 for index in range(21)]
    sparse_cases = {
        (0, 0.0),
        (0, 0.5),
        (0, 1.0),
        (4, 0.5),
        (9, 0.0),
        (9, 1.0),
    }
    protocol = {
        "protocol_version": "fixture-collective-full-input",
        "liouvillian_dimension": 1024,
        "s_grid": s_grid,
        "s_grid_count": 21,
        "seeds": seeds,
        "seed_count": 10,
        "dense_solver": {
            "stationary_abs_tolerance": 1e-8,
            "positive_real_tolerance": 1e-9,
            "minimum_gap_tolerance": 1e-10,
            "relative_residual_tolerance": 1e-11,
            "trace_preservation_tolerance": 1e-13,
        },
        "sparse_crosscheck": {
            "cases": [
                {"seed": seed, "s": s_value}
                for seed, s_value in sorted(sparse_cases)
            ],
            "matrix_tolerance": 1e-13,
            "eigenvalue_tolerance": 2e-6,
            "relative_residual_tolerance": 2e-9,
        },
    }
    protocol_hash = _protocol_hash(protocol)
    frozen = {
        "artifact_type": "collective_loss_full_input_protocol",
        "status": "frozen_before_diagnostic_rows",
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
    }
    rows = []
    for seed in seeds:
        for s_index, s_value in enumerate(s_grid):
            gap = 0.5 + 0.001 * seed + 0.0001 * s_index
            spectrum = [
                [0.0, 0.0],
                [-gap, 0.0],
                *([[-1.0, 0.0]] * 1022),
            ]
            packed = b"".join(
                struct.pack("<dd", *value) for value in spectrum
            )
            spectrum_hash = hashlib.sha256(packed).hexdigest()
            crosscheck = (
                {
                    "dense_sparse_matrix_max_abs_difference": 0.0,
                    "near_zero_sparse_stationary_count": 1,
                    "near_zero_max_dense_eigenvalue_abs_difference": 0.0,
                    "targeted_gap_eigenvalue_abs_difference": 0.0,
                    "near_zero_max_relative_residual": 0.0,
                    "targeted_gap_relative_residual": 0.0,
                }
                if (seed, s_value) in sparse_cases
                else None
            )
            rows.append(
                {
                    "seed": seed,
                    "s_index": s_index,
                    "s": s_value,
                    "coupling_sha256": _digest(f"collective-coupling-{seed}"),
                    "eigenvalue_count": 1024,
                    "stationary_mode_count": 1,
                    "stationary_eigenvalues": [[0.0, 0.0]],
                    "stationary_abs_max": 0.0,
                    "first_nonstationary_decay_gap": gap,
                    "first_nonstationary_eigenvalue": [-gap, 0.0],
                    "first_nonstationary_relative_residual": 0.0,
                    "spectral_abscissa": 0.0,
                    "positive_real_part_leakage": 0.0,
                    "max_all_mode_relative_residual": 0.0,
                    "trace_preservation_relative_residual": 0.0,
                    "relative_jump_budget_error": 0.0,
                    "spectrum_sha256": spectrum_hash,
                    "spectrum": spectrum,
                    "sparse_crosscheck": crosscheck,
                }
            )
    raw = {
        "artifact_type": "collective_loss_full_input_raw_spectrum",
        "protocol_sha256": protocol_hash,
        "row_count": 210,
        "rows": rows,
    }
    directory = root / "results/collective_loss_full_input_diagnostic"
    _write(directory / "protocol.json", json.dumps(frozen))
    _write(directory / "raw_spectrum.json", json.dumps(raw))
    gaps = [row["first_nonstationary_decay_gap"] for row in rows]
    minimum = rows[gaps.index(min(gaps))]
    maximum = rows[gaps.index(max(gaps))]
    crosschecks = [
        {"seed": row["seed"], "s": row["s"], **row["sparse_crosscheck"]}
        for row in rows
        if row["sparse_crosscheck"] is not None
    ]
    per_seed = []
    for seed in seeds:
        selected = [row for row in rows if row["seed"] == seed]
        selected_gaps = [
            row["first_nonstationary_decay_gap"] for row in selected
        ]
        seed_min = selected[selected_gaps.index(min(selected_gaps))]
        seed_max = selected[selected_gaps.index(max(selected_gaps))]

        def extreme(row: dict) -> dict:
            return {
                "seed": row["seed"],
                "s": row["s"],
                "gap": row["first_nonstationary_decay_gap"],
                "eigenvalue": row["first_nonstationary_eigenvalue"],
            }

        per_seed.append(
            {
                "seed": seed,
                "coupling_sha256": selected[0]["coupling_sha256"],
                "grid_rows": 21,
                "unique_stationary_mode_rows": 21,
                "minimum_gap": extreme(seed_min),
                "maximum_gap": extreme(seed_max),
                "mean_gap": math.fsum(selected_gaps) / len(selected_gaps),
                "maximum_positive_real_part_leakage": 0.0,
                "maximum_dense_relative_residual": 0.0,
            }
        )
    aggregate = {
        "artifact_type": "collective_loss_full_input_aggregate",
        "status": "complete",
        "all_declared_checks_passed": True,
        "protocol_sha256": protocol_hash,
        "protocol_file_sha256": hashlib.sha256(
            (directory / "protocol.json").read_bytes()
        ).hexdigest(),
        "raw_payload_sha256": _protocol_hash(raw),
        "raw_file_sha256": hashlib.sha256(
            (directory / "raw_spectrum.json").read_bytes()
        ).hexdigest(),
        "row_count": 210,
        "seed_count": 10,
        "s_grid_count": 21,
        "full_grid_complete": True,
        "all_rows_have_unique_stationary_mode": True,
        "unique_stationary_mode_rows": 210,
        "minimum_sampled_gap": {
            "seed": minimum["seed"],
            "s": minimum["s"],
            "gap": minimum["first_nonstationary_decay_gap"],
            "eigenvalue": minimum["first_nonstationary_eigenvalue"],
        },
        "maximum_sampled_gap": {
            "seed": maximum["seed"],
            "s": maximum["s"],
            "gap": maximum["first_nonstationary_decay_gap"],
            "eigenvalue": maximum["first_nonstationary_eigenvalue"],
        },
        "mean_sampled_gap": math.fsum(gaps) / len(gaps),
        "maximum_positive_real_part_leakage": 0.0,
        "maximum_dense_relative_residual": 0.0,
        "maximum_trace_preservation_relative_residual": 0.0,
        "maximum_relative_jump_budget_error": 0.0,
        "maximum_sparse_matrix_abs_difference": 0.0,
        "maximum_sparse_dense_eigenvalue_abs_difference": 0.0,
        "maximum_sparse_relative_residual": 0.0,
        "sparse_crosscheck_count": 6,
        "per_seed": per_seed,
        "sparse_crosschecks": crosschecks,
    }
    _write(directory / "aggregate.json", json.dumps(aggregate))


def _write_nested_extension_fixture(root: Path) -> None:
    tuning = root / "results/revision_tuning"
    stage = tuning / "nested_operating_point_extension"
    old_manifest_path = tuning / "nested_tuning/manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text())
    old_manifest_sha = hashlib.sha256(old_manifest_path.read_bytes()).hexdigest()
    methods = ("CD_paper", "B3_collective")
    h_grid = [0.25, 0.5, 1.0, 2.0]
    dt_grid = [0.1, 0.25, 0.5, 1.0]
    strengths = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    ridges = [0.0, 1e-8, 1.0]
    screen_seeds = [100, 101]
    selection_seeds = list(range(200, 212))
    old_test_seeds = list(range(300, 324))
    excluded = sorted(screen_seeds + selection_seeds + old_test_seeds)
    fresh_seeds = list(range(400, 424))
    snapshot_data = b"# nested extension source snapshot fixture\n"
    snapshot_sha = hashlib.sha256(snapshot_data).hexdigest()
    ledger = {
        "artifact_type": "nested_extension_seed_ledger",
        "source_manifest_sha256": old_manifest_sha,
        "reused_screen_seeds": screen_seeds,
        "reused_selection_seeds": selection_seeds,
        "known_old_test_seeds": old_test_seeds,
        "excluded_seeds": excluded,
        "excluded_seeds_sha256": _protocol_hash(excluded),
        "fresh_test_seeds": fresh_seeds,
        "fresh_test_seeds_sha256": _protocol_hash(fresh_seeds),
        "pairwise_disjoint_verified": True,
    }
    protocol = {
        "protocol_version": "fixture-nested-extension",
        "source_protocol_sha256": old_manifest["protocol_sha256"],
        "source_manifest_sha256": old_manifest_sha,
        "scientific_sources_sha256": {
            "experiments/run_nested_operating_point_extension.py": snapshot_sha
        },
        "details": {
            "channels": list(methods),
            "h_grid": h_grid,
            "dt_grid": dt_grid,
            "mandatory_common_strength_grid": strengths,
            "ridge_grid": ridges,
            "screen_split": {
                "wash": 100,
                "train": 250,
                "validation": 100,
                "test": 0,
            },
            "selection_split": {
                "wash": 200,
                "train": 450,
                "validation": 150,
                "test": 0,
            },
            "screen_seed_count": 2,
            "selection_seed_count": 12,
            "fresh_test_seed_count": 24,
            "shortlist_per_channel": 8,
            "maximum_adaptive_strength": 2048.0,
        },
        "seed_hashes": {
            "excluded_seeds_sha256": ledger["excluded_seeds_sha256"],
            "fresh_test_seeds_sha256": ledger["fresh_test_seeds_sha256"],
        },
    }
    protocol_sha = _protocol_hash(protocol)
    manifest = {
        "artifact_type": "nested_operating_point_extension_manifest",
        "status": "frozen_before_new_rows",
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
    }
    _write(stage / "manifest.json", json.dumps(manifest))
    _write(stage / "seed_ledger.json", json.dumps(ledger))
    snapshot_path = (
        stage / "source_snapshot/run_nested_operating_point_extension.py"
    )
    _write(snapshot_path, snapshot_data)
    _write(
        stage / "source_snapshot/manifest.json",
        json.dumps(
            {
                "artifact_type": "nested_extension_source_snapshot",
                "protocol_sha256": protocol_sha,
                "path": (
                    "results/revision_tuning/nested_operating_point_extension/"
                    "source_snapshot/run_nested_operating_point_extension.py"
                ),
                "sha256": snapshot_sha,
            }
        ),
    )

    def tag(value: float) -> str:
        return f"{value:g}".replace(".", "p")

    def write_job(
        base_directory: Path,
        stage_name: str,
        method: str,
        config: tuple[float, float, float],
        seed: int,
        row_protocol_sha: str,
    ) -> tuple[str, str]:
        h_value, dt_value, strength = config
        name = (
            f"{method}_h{tag(h_value)}_dt{tag(dt_value)}"
            f"_x{tag(strength)}_s{seed}.json"
        )
        path = base_directory / f"{stage_name}_jobs" / name
        row = {
            "stage": stage_name,
            "split": (
                protocol["details"]["screen_split"]
                if stage_name == "screen"
                else protocol["details"]["selection_split"]
            ),
            "method": method,
            "h": h_value,
            "dt": dt_value,
            "strength_multiplier": strength,
            "seed": seed,
            "protocol_sha256": row_protocol_sha,
            "runtime_s": 0.0,
        }
        _write(path, json.dumps(row))
        return (
            str(path.relative_to(root)),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    configs = [
        (h_value, dt_value, strength)
        for h_value in h_grid
        for dt_value in dt_grid
        for strength in strengths
    ]
    local_candidates = configs[:8]
    collective_candidates = [
        (0.5, 0.5, 32.0),
        (0.5, 0.5, 64.0),
        (0.5, 0.5, 128.0),
        *configs[:5],
    ]
    candidates = {
        "CD_paper": local_candidates,
        "B3_collective": collective_candidates,
    }
    provenance = {
        "screen_reused": [],
        "screen_new": [],
        "selection_reused": [],
        "selection_new": [],
    }
    for method in methods:
        for config in configs:
            for seed in screen_seeds:
                reused = config[2] <= 16.0
                base_directory = (
                    tuning / "nested_tuning" if reused else stage
                )
                path, digest = write_job(
                    base_directory,
                    "screen",
                    method,
                    config,
                    seed,
                    old_manifest["protocol_sha256"] if reused else protocol_sha,
                )
                provenance[
                    "screen_reused" if reused else "screen_new"
                ].append(
                    {
                        "stage": "screen",
                        "identity": [method, *config, seed],
                        "path": path,
                        "sha256": digest,
                    }
                )
        for config in candidates[method]:
            for seed in selection_seeds:
                reused = config[2] <= 16.0
                base_directory = (
                    tuning / "nested_tuning" if reused else stage
                )
                path, digest = write_job(
                    base_directory,
                    "selection",
                    method,
                    config,
                    seed,
                    old_manifest["protocol_sha256"] if reused else protocol_sha,
                )
                provenance[
                    "selection_reused" if reused else "selection_new"
                ].append(
                    {
                        "stage": "selection",
                        "identity": [method, *config, seed],
                        "path": path,
                        "sha256": digest,
                    }
                )
    reuse = {
        "artifact_type": "nested_extension_row_reuse_index",
        "protocol_sha256": protocol_sha,
        "source_protocol_sha256": old_manifest["protocol_sha256"],
        "source_manifest_sha256": old_manifest_sha,
        "counts": {
            key: len(value) for key, value in provenance.items()
        },
        **provenance,
    }
    _write(stage / "reuse_index.json", json.dumps(reuse))
    rankings = {}
    for method in methods:
        ordered_configs = [
            *candidates[method],
            *[config for config in configs if config not in candidates[method]],
        ]
        rankings[method] = [
            {
                "config": list(config),
                "ridge_upper_boundary_unresolved": False,
            }
            for config in ordered_configs
        ]
    shortlist = {
        "artifact_type": "nested_extension_screen_shortlist",
        "protocol_sha256": protocol_sha,
        "realized_common_strength_grid": strengths,
        "common_grid_identical_for_channels": True,
        "screen_config_count_per_channel": len(configs),
        "shortlist": {
            method: [list(config) for config in candidates[method]]
            for method in methods
        },
        "calibration_candidate_configs": {
            method: [list(config) for config in candidates[method]]
            for method in methods
        },
        "screen_ranking": rankings,
    }
    _write(stage / "screen_shortlist.json", json.dumps(shortlist))
    chosen = {
        "CD_paper": {
            "config": list(local_candidates[0]),
            "best_ridge": 1e-8,
            "ridge_upper_boundary_unresolved": False,
            "mean_validation_mc": 1.0,
        },
        "B3_collective": {
            "config": [0.5, 0.5, 64.0],
            "best_ridge": 1e-8,
            "ridge_upper_boundary_unresolved": False,
            "mean_validation_mc": 2.0,
        },
    }
    bracket = {
        "bracketed": True,
        "reason": "strict_local_maximum_on_full_selection_ensemble",
        "lower": {
            "config": [0.5, 0.5, 32.0],
            "mean_validation_mc": 1.5,
        },
        "selected": chosen["B3_collective"],
        "upper": {
            "config": [0.5, 0.5, 128.0],
            "mean_validation_mc": 1.4,
        },
        "required_configs": [],
    }
    selection_rankings = {
        "CD_paper": [
            chosen["CD_paper"],
            *[
                {
                    "config": list(config),
                    "best_ridge": 1e-8,
                    "ridge_upper_boundary_unresolved": False,
                    "mean_validation_mc": 0.5,
                }
                for config in local_candidates[1:]
            ],
        ],
        "B3_collective": [
            chosen["B3_collective"],
            bracket["lower"],
            bracket["upper"],
            *[
                {
                    "config": list(config),
                    "best_ridge": 1e-8,
                    "ridge_upper_boundary_unresolved": False,
                    "mean_validation_mc": 0.5,
                }
                for config in collective_candidates
                if list(config)
                not in (
                    chosen["B3_collective"]["config"],
                    bracket["lower"]["config"],
                    bracket["upper"]["config"],
                )
            ],
        ],
    }
    frozen = {
        "artifact_type": "frozen_nested_extension_operating_points",
        "status": "frozen_before_fresh_test_ensemble",
        "protocol_sha256": protocol_sha,
        "screen_shortlist_sha256": hashlib.sha256(
            (stage / "screen_shortlist.json").read_bytes()
        ).hexdigest(),
        "reuse_index_sha256": hashlib.sha256(
            (stage / "reuse_index.json").read_bytes()
        ).hexdigest(),
        "fresh_test_seeds_sha256": ledger["fresh_test_seeds_sha256"],
        "realized_common_strength_grid": strengths,
        "chosen": chosen,
        "selection_ranking": selection_rankings,
        "collective_strength_bracket": bracket,
        "test_rows_present_at_freeze": False,
    }
    _write(stage / "frozen_selection.json", json.dumps(frozen))
    selection_sha = hashlib.sha256(
        (stage / "frozen_selection.json").read_bytes()
    ).hexdigest()
    score_maps = {"CD_paper": {}, "B3_collective": {}}
    test_provenance = []
    for method in methods:
        for seed in fresh_seeds:
            score = 1.0 if method == "CD_paper" else 1.5
            score_maps[method][seed] = score
            path = stage / "test_jobs" / f"{method}_s{seed}.json"
            row = {
                "method": method,
                "seed": seed,
                "h": chosen[method]["config"][0],
                "dt": chosen[method]["config"][1],
                "strength_multiplier": chosen[method]["config"][2],
                "ridge": chosen[method]["best_ridge"],
                "test_mc": score,
                "protocol_sha256": protocol_sha,
                "selection_sha256": selection_sha,
                "runtime_s": 0.0,
            }
            _write(path, json.dumps(row))
            test_provenance.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    screen_count = len(methods) * len(configs) * len(screen_seeds)
    selection_count = sum(
        len(candidates[method]) * len(selection_seeds)
        for method in methods
    )
    aggregate = {
        "artifact_type": "nested_operating_point_extension_results",
        "status": "complete",
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "seed_ledger_sha256": hashlib.sha256(
            (stage / "seed_ledger.json").read_bytes()
        ).hexdigest(),
        "reuse_index_sha256": hashlib.sha256(
            (stage / "reuse_index.json").read_bytes()
        ).hexdigest(),
        "screen_shortlist_sha256": hashlib.sha256(
            (stage / "screen_shortlist.json").read_bytes()
        ).hexdigest(),
        "realized_common_strength_grid": strengths,
        "common_grid_identical_for_channels": True,
        "collective_strength_bracket": bracket,
        "seed_disjointness_verified": True,
        "freeze_before_test_verified": True,
        "selected_ridge_upper_boundary_hits": 0,
        "coverage": {
            "screen_expected": screen_count,
            "screen_complete": screen_count,
            "screen_reused": reuse["counts"]["screen_reused"],
            "screen_new": reuse["counts"]["screen_new"],
            "selection_expected": selection_count,
            "selection_complete": selection_count,
            "selection_reused": reuse["counts"]["selection_reused"],
            "selection_new": reuse["counts"]["selection_new"],
            "fresh_test_expected": 48,
            "fresh_test_complete": 48,
        },
        "methods": {
            method: {
                "selected": chosen[method],
                "fresh_test_scores_by_seed": score_maps[method],
            }
            for method in methods
        },
        "collective_vs_local": {
            "n": 24,
            "paired_differences": [0.5] * 24,
        },
        "runtime": {},
        "raw_provenance": {
            "new_screen_rows": [],
            "new_selection_rows": [],
            "fresh_test_rows": test_provenance,
        },
    }
    aggregate["deterministic_payload_sha256"] = _protocol_hash(aggregate)
    _write(stage / "aggregate.json", json.dumps(aggregate))

    screen_entries = [
        ("sealed_reused", entry)
        for entry in provenance["screen_reused"]
    ] + [
        ("extension_new", entry)
        for entry in provenance["screen_new"]
    ]
    screen_rows = [
        json.loads((root / entry["path"]).read_text())
        for _, entry in screen_entries
    ]
    screen_rows.sort(
        key=lambda row: (
            str(row["method"]),
            int(row["seed"]),
            float(row["h"]),
            float(row["dt"]),
            float(row["strength_multiplier"]),
        )
    )
    row_provenance = [
        {
            "source": source,
            "path": entry["path"],
            "sha256": entry["sha256"],
        }
        for source, entry in screen_entries
    ]
    row_provenance.sort(key=lambda item: (item["source"], item["path"]))
    method_audits = {}
    for method in methods:
        selected = tuple(map(float, chosen[method]["config"]))
        top_configs = [
            selected,
            *[config for config in configs if config != selected],
        ][:8]
        top_rows = [
            {
                "rank": rank,
                "config": list(config),
                "best_ridge": 1e-8,
                "validation_mc": 1.0 - 0.01 * rank,
            }
            for rank, config in enumerate(top_configs, start=1)
        ]
        per_seed = {
            str(seed): {
                "frozen_selected_config_rank": 1,
                "frozen_selected_config_in_top8": True,
                "frozen_selected_config_screen_best_ridge": 1e-8,
                "frozen_selected_config_validation_mc": 0.99,
                "winner": {
                    "config": top_rows[0]["config"],
                    "best_ridge": top_rows[0]["best_ridge"],
                    "validation_mc": top_rows[0]["validation_mc"],
                },
                "top8": top_rows,
            }
            for seed in screen_seeds
        }
        top_set = {tuple(row["config"]) for row in top_rows}
        method_audits[method] = {
            "frozen_selected_config": list(selected),
            "per_seed": per_seed,
            "top8_overlap": {
                "intersection_count": 8,
                "union_count": 8,
                "jaccard": 1.0,
                "intersection_configs": [
                    list(config) for config in sorted(top_set)
                ],
                "union_configs": [
                    list(config) for config in sorted(top_set)
                ],
            },
            "full_rank_spearman": 1.0,
        }
    prescreen = {
        "artifact_type": "nested_prescreen_stability_audit",
        "artifact_version": "nested-prescreen-stability-v1-2026-07-24",
        "status": "complete",
        "analysis_type": "post_hoc_descriptive_sensitivity_audit",
        "claim_boundary": (
            "This describes agreement between the two realized cheap-screen "
            "reservoirs. It does not remedy the two-seed prescreen limitation, "
            "provide selection-adjusted inference, or enlarge the calibration "
            "ensemble."
        ),
        "ridge_rule": "fixture ridge rule",
        "rank_rule": "fixture rank rule",
        "screen_seed_count": 2,
        "screen_seeds": screen_seeds,
        "configuration_count_per_method": len(configs),
        "top_k": 8,
        "provenance": {
            "protocol_sha256": protocol_sha,
            "frozen_selection_sha256": selection_sha,
            "screen_shortlist_sha256": hashlib.sha256(
                (stage / "screen_shortlist.json").read_bytes()
            ).hexdigest(),
            "source_manifest_sha256": old_manifest_sha,
            "screen_rows_expected": len(screen_rows),
            "screen_rows_loaded": len(screen_rows),
            "sealed_reused_rows": len(provenance["screen_reused"]),
            "extension_new_rows": len(provenance["screen_new"]),
            "screen_rows_sha256": _protocol_hash(screen_rows),
            "row_provenance_sha256": _protocol_hash(row_provenance),
        },
        "methods": method_audits,
    }
    prescreen["deterministic_payload_sha256"] = _protocol_hash(prescreen)
    _write(stage / "prescreen_stability.json", json.dumps(prescreen))


def _write_primary_driven_activity_fixture(root: Path) -> None:
    methods = [
        "CD_paper",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    ]
    seeds = list(range(32))
    source_relatives = (
        "experiments/run_primary_driven_activity.py",
        "experiments/run_final_scaling.py",
        "src/qrc/dissipators.py",
        "src/qrc/liouvillian.py",
        "src/qrc/operators.py",
        "src/qrc/reservoirs.py",
        "src/qrc/sparse_evolve.py",
        "src/qrc/tasks.py",
    )
    source_environment = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in source_relatives
    }
    baseline_path = root / "results/final_protocol_results.tar.gz"
    baseline_entries = {
        f"{method}/{seed}": {
            "member": (
                "final_protocol/"
                f"A_table__stm_N5_{method}_s{seed}_h0.5_dt0.5_L200-600-400.json"
            ),
            "checkpoint_sha256": _digest(f"baseline-{method}-{seed}"),
            "stm_capacity": 8.0 + 0.01 * seed,
        }
        for method in methods
        for seed in seeds
    }
    protocol = {
        "protocol_version": (
            "primary-driven-jump-activity-posthoc-v1-2026-07-25"
        ),
        "analysis_status": (
            "deterministic descriptive post-hoc reuse of sealed primary test "
            "trajectories"
        ),
        "n_qubits": 5,
        "h": 0.5,
        "dt": 0.5,
        "gamma": 1.0,
        "split": {
            "wash": 200,
            "train": 600,
            "test": 400,
            "test_start_input_index": 800,
            "total_inputs": 1200,
            "activity_uses_only_test_intervals": True,
        },
        "methods": methods,
        "reference_method": "CD_paper",
        "seeds": seeds,
        "n_jobs": 224,
        "baseline_archive": {
            "path": "results/final_protocol_results.tar.gz",
            "sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "entries": baseline_entries,
            "entries_sha256": _protocol_hash(baseline_entries),
        },
        "source_environment": source_environment,
        "source_environment_sha256": _protocol_hash(source_environment),
    }
    protocol_sha = _protocol_hash(protocol)
    output = root / "results/primary_driven_activity"
    _write(output / "protocol.json", json.dumps(protocol))

    offsets = {
        "CD_paper": 1.50,
        "B3_collective": 1.20,
        "A1_heterogeneous": 1.25,
        "B5_pair": 0.75,
        "B2_thermal": 2.20,
        "B4_loss_exchange": 2.22,
        "B1_dephasing": 2.50,
    }
    values_by_method: dict[str, list[float]] = {}
    for method in methods:
        values = []
        for seed in seeds:
            value = offsets[method] + 0.001 * seed
            values.append(value)
            row = {
                "protocol_version": protocol["protocol_version"],
                "protocol_sha256": protocol_sha,
                "source_environment_sha256": (
                    protocol["source_environment_sha256"]
                ),
                "method": method,
                "seed": seed,
                "test_intervals": 400,
                "baseline_checkpoint_member": baseline_entries[
                    f"{method}/{seed}"
                ]["member"],
                "baseline_checkpoint_sha256": baseline_entries[
                    f"{method}/{seed}"
                ]["checkpoint_sha256"],
                "coupling_sha256": _digest(f"coupling-{seed}"),
                "full_input_sha256": _digest(f"full-input-{seed}"),
                "test_input_sha256": _digest(f"test-input-{seed}"),
                "jump_family_sha256": _digest(f"jumps-{method}-{seed}"),
                "target_frobenius_jump_strength": 80.0,
                "actual_frobenius_jump_strength": 80.0,
                "time_averaged_jump_activity": value,
                "maximum_trace_error": 1e-13,
                "maximum_activity_imaginary_residue": 1e-16,
                "minimum_integrated_interval_activity": 0.2,
            }
            _write(
                output / "checkpoints" / f"{method}__seed_{seed}.json",
                json.dumps(row),
            )
        values_by_method[method] = values

    local = values_by_method["CD_paper"]
    method_summaries = {}
    effects = {}
    for method in methods:
        values = values_by_method[method]
        summary = _summary(values)
        differences = [
            candidate - reference
            for candidate, reference in zip(values, local)
        ]
        method_summaries[method] = {
            "n": 32,
            "mean_time_averaged_activity": summary["mean"],
            "standard_error": summary["se"],
        }
        effects[method] = {
            "mean_activity_difference": math.fsum(differences) / 32,
            "ratio_of_method_mean_to_local_mean": (
                math.fsum(values) / math.fsum(local)
            ),
            "higher_activity_count": sum(value > 0 for value in differences),
            "equal_activity_count": sum(value == 0 for value in differences),
            "lower_activity_count": sum(value < 0 for value in differences),
        }
    aggregate = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": protocol_sha,
        "source_environment_sha256": protocol["source_environment_sha256"],
        "status": "complete",
        "n_jobs": 224,
        "n_seeds": 32,
        "method_summaries": method_summaries,
        "paired_vs_uniform_local": effects,
        "invariant_audit": {
            "passed": True,
            "errors": [],
            "expected_jobs": 224,
            "observed_jobs": 224,
            "all_rows_use_400_test_intervals": True,
            "all_rows_link_sealed_checkpoint": True,
        },
    }
    _write(output / "aggregate.json", json.dumps(aggregate))


def _activity_fixture_band(values: list[float]) -> dict:
    summary = _summary(values)
    mean = summary["mean"]
    standard_error = summary["se"]
    critical = ACTIVITY_FIXTURE_BONFERRONI_CRITICAL
    nonzero = [
        value
        for value in values
        if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-14)
    ]
    return {
        "n": len(values),
        "mean": mean,
        "standard_error": standard_error,
        "critical_value": critical,
        "simultaneous_lower": mean - critical * standard_error,
        "simultaneous_upper": mean + critical * standard_error,
        "family_size": 5,
        "familywise_alpha": 0.05,
        "wins": sum(value > 0.0 for value in nonzero),
        "ties": len(values) - len(nonzero),
        "losses": sum(value < 0.0 for value in nonzero),
        "paired_values": values,
    }


def _write_activity_matched_response_fixture(root: Path) -> None:
    output = root / "results/activity_matched_response"
    designs = ("local", "collective")
    branches = {
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
    pilot_grids = {
        "local": [
            0.05,
            0.0666760716082,
            0.0889139705019,
            0.118568685283,
            0.158113883008,
            0.210848251714,
            0.25,
            0.281170662595,
            0.374947104666,
            0.5,
        ],
        "collective": [
            4.0,
            5.1873582186,
            6.72717132203,
            8.0,
            8.72406186132,
            11.313708499,
            14.6720646913,
            19.02731384,
            24.6753732065,
            32.0,
        ],
    }
    prior_seeds = list(range(100, 120))
    pilot_seeds = list(range(1000, 1008))
    task_seeds = list(range(2000, 2024))
    ledger = {
        "prior_sources": {
            "fixture_prior": {
                "count": len(prior_seeds),
                "sha256": _protocol_hash(prior_seeds),
            },
            "activity_v2_pilot": {
                "count": 8,
                "sha256": (
                    "e7de08e3669e2b722fca0e8c98c0472222f2c48c8146f6d376500e66728831b8"
                ),
            },
            "activity_v2_task": {
                "count": 24,
                "sha256": (
                    "172abb78c16ac2200b7645c2460959fde17d575da89209b0635110ddb2c940ac"
                ),
            },
        },
        "prior_seed_count": len(prior_seeds),
        "prior_seeds_sha256": _protocol_hash(prior_seeds),
        "prior_seeds": prior_seeds,
        "pilot_namespace": 505,
        "pilot_seeds": pilot_seeds,
        "task_namespace": 506,
        "task_seeds": task_seeds,
        "pilot_task_overlap": [],
        "pilot_prior_overlap": [],
        "task_prior_overlap": [],
    }
    source_relatives = (
        "experiments/run_activity_matched_response.py",
        "src/qrc/dissipators.py",
        "src/qrc/liouvillian.py",
        "src/qrc/operators.py",
        "src/qrc/readout.py",
        "src/qrc/reservoirs.py",
        "src/qrc/sparse_evolve.py",
        "src/qrc/tasks.py",
    )
    source_hashes = {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in source_relatives
    }
    source_hashes_sha = _protocol_hash(source_hashes)
    driver_sha = source_hashes[
        "experiments/run_activity_matched_response.py"
    ]
    pilot_protocol = {
        "protocol_version": ACTIVITY_FIXTURE_PROTOCOL_VERSION,
        "stage": "activity_only_pilot",
        "status": "must_be_frozen_before_pilot_rows",
        "scientific_sources_sha256": source_hashes,
        "scientific_sources_combined_sha256": source_hashes_sha,
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
            "driver_sha256": driver_sha,
        },
        "seed_ledger": ledger,
        "physics": {
            "N": 5,
            "h": 0.5,
            "dt": 0.5,
            "designs": list(designs),
            "branches": branches,
            "pilot_rate_grids": pilot_grids,
        },
        "calibration_input": {
            "distribution": "iid Uniform[0,1]",
            "labels_or_task_targets_used": False,
            "wash_intervals": 200,
            "unsupervised_prefix_intervals": 600,
            "measured_intervals": 400,
        },
        "target_freeze_rule": {
            "minimum_common_span_ratio": 1.5,
            "log_inset_fraction_each_side": 0.05,
            "n_targets": 5,
        },
        "expected_pilot_rows": len(pilot_seeds)
        * sum(len(grid) for grid in pilot_grids.values()),
        "supervised_boundary": {
            "constructs_task_targets": False,
            "fits_readout": False,
            "scores_task": False,
        },
    }
    pilot_protocol_sha = _protocol_hash(pilot_protocol)
    _write(
        output / "pilot_manifest.json",
        json.dumps(
            {
                "artifact_type": "activity_matched_pilot_manifest",
                "manifest_status": "frozen_before_pilot_rows",
                "protocol": pilot_protocol,
                "protocol_sha256": pilot_protocol_sha,
            }
        ),
    )
    driver_bytes = (root / "experiments/run_activity_matched_response.py").read_bytes()
    _write(
        output / "source_snapshot/run_activity_matched_response.py",
        driver_bytes,
    )
    _write(
        output / "source_snapshot/manifest.json",
        json.dumps(
            {
                "artifact_type": "activity_matched_source_snapshot",
                "pilot_protocol_sha256": pilot_protocol_sha,
                "source_path": "experiments/run_activity_matched_response.py",
                "snapshot_path": (
                    "results/activity_matched_response/source_snapshot/"
                    "run_activity_matched_response.py"
                ),
                "sha256": driver_sha,
                "all_scientific_source_hashes": source_hashes,
                "all_scientific_source_hashes_sha256": source_hashes_sha,
            }
        ),
    )

    pilot_rows: list[dict] = []
    curves: list[dict] = []
    reachable: list[tuple[float, float]] = []
    for design in designs:
        orientation = branches[design]["activity_orientation"]
        sign = 1.0 if orientation == "increasing" else -1.0
        for seed_offset, seed in enumerate(pilot_seeds):
            activities = []
            for rate_index, rate in enumerate(pilot_grids[design]):
                if design == "local":
                    activity = 0.2 + rate + 0.001 * seed_offset
                else:
                    activity = 1.5 - 0.04 * rate + 0.001 * seed_offset
                activities.append(activity)
                row = {
                    "artifact_type": "activity_only_pilot_row",
                    "protocol_version": ACTIVITY_FIXTURE_PROTOCOL_VERSION,
                    "pilot_protocol_sha256": pilot_protocol_sha,
                    "design": design,
                    "seed": seed,
                    "rate": rate,
                    "branch": branches[design],
                    "couplings_sha256": _digest(f"activity-coupling-{seed}"),
                    "calibration_input_sha256": _digest(
                        f"activity-calibration-input-{seed}"
                    ),
                    "activity": activity,
                    "total_expected_jumps": activity * 400 * 0.5,
                    "maximum_trace_error": 1e-12,
                    "maximum_activity_imaginary_residue": 1e-15,
                    "minimum_interval_integrated_activity": activity * 0.4,
                    "jump_strength": 80.0 * rate,
                    "runtime_seconds": 0.01,
                }
                pilot_rows.append(row)
                _write(
                    output
                    / "pilot/checkpoints"
                    / (
                        f"{design}__seed_{seed}"
                        f"__rate_{rate_index:02d}.json"
                    ),
                    json.dumps(row),
                )
            increments = [
                sign * (right - left)
                for left, right in zip(activities, activities[1:])
            ]
            tolerance = 2e-7 + 2e-5 * max(map(abs, activities))
            low = min(activities[0], activities[-1])
            high = max(activities[0], activities[-1])
            reachable.append((low, high))
            curves.append(
                {
                    "design": design,
                    "seed": seed,
                    "reachable_interval": [low, high],
                    "monotonicity": {
                        "passed": True,
                        "orientation": orientation,
                        "minimum_oriented_increment": min(increments),
                        "tolerance": tolerance,
                        "rates": pilot_grids[design],
                        "activities": activities,
                    },
                }
            )

    common_low = max(low for low, _ in reachable)
    common_high = min(high for _, high in reachable)
    log_low = math.log(common_low)
    log_high = math.log(common_high)
    log_span = log_high - log_low
    target_low = math.exp(log_low + 0.05 * log_span)
    target_high = math.exp(log_high - 0.05 * log_span)
    targets = [
        math.exp(
            math.log(target_low)
            + index
            * (math.log(target_high) - math.log(target_low))
            / 4
        )
        for index in range(5)
    ]
    pilot_row_index = sorted(
        {
            (
                row["design"],
                int(row["seed"]),
                float(row["rate"]),
                _protocol_hash(row),
            )
            for row in pilot_rows
        }
    )
    frozen_targets = {
        "artifact_type": "frozen_activity_targets",
        "freeze_status": (
            "frozen_before_fresh_calibration_or_task_scores"
        ),
        "pilot_protocol_sha256": pilot_protocol_sha,
        "pilot_rows_sha256": _protocol_hash(pilot_row_index),
        "uses_supervised_task_information": False,
        "common_activity_interval": [common_low, common_high],
        "common_activity_span_ratio": common_high / common_low,
        "target_interval_after_log_inset": [target_low, target_high],
        "targets": targets,
        "n_targets": 5,
        "pilot_curve_audits": curves,
    }
    _write(output / "frozen_targets.json", json.dumps(frozen_targets))
    frozen_targets_sha = hashlib.sha256(
        (output / "frozen_targets.json").read_bytes()
    ).hexdigest()

    task_protocol = {
        "protocol_version": ACTIVITY_FIXTURE_PROTOCOL_VERSION,
        "stage": "fresh_calibration_then_frozen_task_scoring",
        "status": "must_be_frozen_before_fresh_calibration_rows",
        "scientific_sources_sha256": source_hashes,
        "scientific_sources_combined_sha256": source_hashes_sha,
        "frozen_targets": {
            "path": "results/activity_matched_response/frozen_targets.json",
            "sha256": frozen_targets_sha,
            "pilot_protocol_sha256": pilot_protocol_sha,
            "targets": targets,
        },
        "seed_ledger": ledger,
        "fresh_boundary": {
            "task_seed_count": 24,
            "task_seeds": task_seeds,
            "task_prior_overlap": [],
            "task_pilot_overlap": [],
            "same_couplings_and_streams_across_designs_and_targets": True,
            "calibration_and_task_streams_independent": True,
        },
        "calibration": {
            "branches": branches,
            "input_split": {
                "wash": 200,
                "unsupervised_prefix": 600,
                "measured_activity": 400,
            },
            "target_count": 5,
            "expected_cells": 240,
            "bisection_scale": "geometric rate midpoint",
            "maximum_iterations": 18,
            "relative_match_tolerance": 0.005,
            "absolute_match_tolerance": 1e-5,
            "censored_statuses": [
                "censored_target_unreachable",
                "censored_branch_nonmonotone",
                "censored_nonconvergence",
            ],
        },
        "task": {
            "task_name": "STM",
            "N": 5,
            "h": 0.5,
            "dt": 0.5,
            "wash": 200,
            "train": 600,
            "test": 400,
            "delays": list(range(1, 21)),
            "ridge": 1e-8,
            "ridge_selection": "none; fixed before all fresh task scores",
            "test_evaluations_per_cell": 1,
        },
        "inference": {
            "unit": "paired reservoir seed",
            "family_size": 5,
            "method": (
                "two-sided Bonferroni-t simultaneous 95% bands over "
                "exactly five prespecified contrasts"
            ),
        },
    }
    task_protocol_sha = _protocol_hash(task_protocol)
    _write(
        output / "task_manifest.json",
        json.dumps(
            {
                "artifact_type": "activity_matched_task_manifest",
                "manifest_status": (
                    "frozen_before_fresh_calibration_rows"
                ),
                "protocol": task_protocol,
                "protocol_sha256": task_protocol_sha,
            }
        ),
    )

    calibration_rows: dict[tuple[str, int, int], dict] = {}
    calibration_paths: dict[tuple[str, int, int], Path] = {}
    for design in designs:
        branch = branches[design]
        orientation = branch["activity_orientation"]
        for seed in task_seeds:
            for target_index, target in enumerate(targets):
                rate = (
                    0.1 + 0.05 * target_index
                    if design == "local"
                    else 12.0 - target_index
                )
                if orientation == "increasing":
                    evaluation_activities = [
                        0.5 * target,
                        target,
                        1.5 * target,
                    ]
                    evaluation_rates = [
                        branch["lower_rate"],
                        rate,
                        branch["upper_rate"],
                    ]
                else:
                    evaluation_activities = [
                        1.5 * target,
                        target,
                        0.5 * target,
                    ]
                    evaluation_rates = [
                        branch["lower_rate"],
                        rate,
                        branch["upper_rate"],
                    ]
                evaluations = [
                    {
                        "rate": eval_rate,
                        "activity": eval_activity,
                        "jump_strength": 80.0 * eval_rate,
                        "maximum_trace_error": 1e-12,
                        "maximum_activity_imaginary_residue": 1e-15,
                    }
                    for eval_rate, eval_activity in zip(
                        evaluation_rates,
                        evaluation_activities,
                    )
                ]
                row = {
                    "artifact_type": "fresh_activity_calibration_row",
                    "protocol_version": ACTIVITY_FIXTURE_PROTOCOL_VERSION,
                    "task_protocol_sha256": task_protocol_sha,
                    "design": design,
                    "seed": seed,
                    "target_index": target_index,
                    "target_activity": target,
                    "branch": branch,
                    "couplings_sha256": _digest(
                        f"activity-coupling-{seed}"
                    ),
                    "calibration_input_sha256": _digest(
                        f"activity-calibration-input-{seed}"
                    ),
                    "calibration_uses_task_targets_or_scores": False,
                    "status": "matched",
                    "matched_rate": rate,
                    "matched_activity": target,
                    "absolute_error": 0.0,
                    "relative_error": 0.0,
                    "iterations": 1,
                    "reachable_interval": [
                        min(evaluation_activities),
                        max(evaluation_activities),
                    ],
                    "evaluations": evaluations,
                    "monotonicity": {
                        "passed": True,
                        "orientation": orientation,
                    },
                    "runtime_seconds": 0.01,
                }
                key = (design, seed, target_index)
                path = (
                    output
                    / "calibration/checkpoints"
                    / (
                        f"{design}__seed_{seed}"
                        f"__target_{target_index:02d}.json"
                    )
                )
                _write(path, json.dumps(row))
                calibration_rows[key] = row
                calibration_paths[key] = path

    cells = []
    for key in sorted(calibration_rows):
        row = calibration_rows[key]
        cells.append(
            {
                "design": row["design"],
                "seed": row["seed"],
                "target_index": row["target_index"],
                "target_activity": row["target_activity"],
                "matched_rate": row["matched_rate"],
                "matched_activity": row["matched_activity"],
                "relative_error": row["relative_error"],
                "calibration_row_sha256": hashlib.sha256(
                    calibration_paths[key].read_bytes()
                ).hexdigest(),
                "calibration_row_payload_sha256": _protocol_hash(row),
                "couplings_sha256": row["couplings_sha256"],
                "calibration_input_sha256": row[
                    "calibration_input_sha256"
                ],
                "status": "matched",
            }
        )
    frozen_calibration = {
        "artifact_type": "frozen_fresh_activity_calibration",
        "freeze_status": "frozen_before_any_task_score",
        "task_protocol_sha256": task_protocol_sha,
        "frozen_targets_sha256": frozen_targets_sha,
        "expected_cells": 240,
        "observed_cells": 240,
        "censored_cells": 0,
        "maximum_relative_match_error": 0.0,
        "gate_passed": True,
        "gate_errors": [],
        "cells": cells,
    }
    _write(
        output / "frozen_calibration.json",
        json.dumps(frozen_calibration),
    )
    frozen_calibration_sha = hashlib.sha256(
        (output / "frozen_calibration.json").read_bytes()
    ).hexdigest()
    frozen_cell_lookup = {
        (
            cell["design"],
            int(cell["seed"]),
            int(cell["target_index"]),
        ): cell
        for cell in cells
    }

    score_rows: dict[tuple[str, int, int], dict] = {}
    for design in designs:
        for seed_offset, seed in enumerate(task_seeds):
            task_input_sha = _digest(f"activity-task-input-{seed}")
            for target_index, target in enumerate(targets):
                cell = frozen_cell_lookup[(design, seed, target_index)]
                score = (
                    10.0 + 0.1 * target_index + 0.001 * seed_offset
                    if design == "local"
                    else 9.75 + 0.1 * target_index + 0.001 * seed_offset
                )
                seed_offset_activity = 0.0001 * (seed_offset - 11.5)
                activity = target * (
                    1.0
                    + seed_offset_activity
                    + (0.01 if design == "collective" else 0.0)
                )
                row = {
                    "artifact_type": "fresh_activity_matched_stm_row",
                    "protocol_version": ACTIVITY_FIXTURE_PROTOCOL_VERSION,
                    "task_protocol_sha256": task_protocol_sha,
                    "frozen_calibration_sha256": frozen_calibration_sha,
                    "calibration_row_sha256": cell[
                        "calibration_row_sha256"
                    ],
                    "design": design,
                    "seed": seed,
                    "target_index": target_index,
                    "target_activity": target,
                    "frozen_rate": cell["matched_rate"],
                    "couplings_sha256": cell["couplings_sha256"],
                    "calibration_input_sha256": cell[
                        "calibration_input_sha256"
                    ],
                    "task_input_sha256": task_input_sha,
                    "task_stream_is_independent_of_calibration_stream": True,
                    "ridge": 1e-8,
                    "test_stm_capacity": score,
                    "test_capacity_by_delay": [score / 20.0] * 20,
                    "time_averaged_test_activity": activity,
                    "total_expected_test_jumps": activity * 400 * 0.5,
                    "maximum_trace_error": 1e-12,
                    "maximum_activity_imaginary_residue": 1e-15,
                    "minimum_test_interval_integrated_activity": (
                        activity * 0.4
                    ),
                    "runtime_seconds": 0.01,
                }
                key = (design, seed, target_index)
                score_rows[key] = row
                _write(
                    output
                    / "score/checkpoints"
                    / (
                        f"{design}__seed_{seed}"
                        f"__target_{target_index:02d}.json"
                    ),
                    json.dumps(row),
                )

    target_results = []
    for target_index, target in enumerate(targets):
        local_rows = [
            score_rows[("local", seed, target_index)]
            for seed in task_seeds
        ]
        collective_rows = [
            score_rows[("collective", seed, target_index)]
            for seed in task_seeds
        ]
        local_scores = [
            float(row["test_stm_capacity"]) for row in local_rows
        ]
        collective_scores = [
            float(row["test_stm_capacity"])
            for row in collective_rows
        ]
        local_activity = [
            float(row["time_averaged_test_activity"])
            for row in local_rows
        ]
        collective_activity = [
            float(row["time_averaged_test_activity"])
            for row in collective_rows
        ]
        stm_band = _activity_fixture_band(
            [
                candidate - reference
                for candidate, reference in zip(
                    collective_scores,
                    local_scores,
                )
            ]
        )
        activity_band = _activity_fixture_band(
            [
                (candidate - reference) / target
                for candidate, reference in zip(
                    collective_activity,
                    local_activity,
                )
            ]
        )
        target_results.append(
            {
                "target_index": target_index,
                "target_activity": target,
                "local_stm_mean": math.fsum(local_scores) / 24,
                "collective_stm_mean": math.fsum(collective_scores) / 24,
                "stm_collective_minus_local": stm_band,
                "local_test_activity_mean": math.fsum(local_activity) / 24,
                "collective_test_activity_mean": (
                    math.fsum(collective_activity) / 24
                ),
                "relative_test_activity_collective_minus_local": (
                    activity_band
                ),
                "stm_dominance_at_target": (
                    stm_band["simultaneous_lower"] > 0.0
                ),
                "test_activity_equivalent_at_target": (
                    activity_band["simultaneous_lower"] >= -0.05
                    and activity_band["simultaneous_upper"] <= 0.05
                ),
            }
        )
    all_stm = all(
        row["stm_dominance_at_target"] for row in target_results
    )
    all_activity = all(
        row["test_activity_equivalent_at_target"]
        for row in target_results
    )
    aggregate = {
        "artifact_type": "activity_matched_response_aggregate",
        "protocol_version": ACTIVITY_FIXTURE_PROTOCOL_VERSION,
        "task_protocol_sha256": task_protocol_sha,
        "frozen_calibration_sha256": frozen_calibration_sha,
        "status": "complete",
        "n_rows": 240,
        "n_seeds": 24,
        "n_targets": 5,
        "target_results": target_results,
        "claim_gates": {
            "zero_censored_fresh_cells": True,
            "simultaneous_stm_dominance_all_targets": all_stm,
            "simultaneous_test_activity_equivalence_all_targets": (
                all_activity
            ),
            "range_wide_dominance_supported": all_stm,
            "task_activity_equivalence_supported": all_activity,
            "activity_matched_dominance_claim_allowed": (
                all_stm and all_activity
            ),
            "failure_rule": "fixture fail-closed rule",
        },
        "invariant_audit": {
            "passed": True,
            "errors": [],
            "expected_rows": 240,
            "observed_rows": 240,
            "maximum_trace_error": 1e-12,
            "maximum_activity_imaginary_residue": 1e-15,
            "minimum_test_interval_integrated_activity": min(
                float(row["minimum_test_interval_integrated_activity"])
                for row in score_rows.values()
            ),
            "fixed_ridge_all_rows": True,
            "delay_capacity_sums_match": True,
        },
        "limitations": ["fixture negative outcome remains complete"],
    }
    assert aggregate["claim_gates"][
        "activity_matched_dominance_claim_allowed"
    ] is False
    _write(output / "aggregate.json", json.dumps(aggregate))


def _make_activity_fixture_failed_terminal(root: Path) -> str:
    """Turn the complete fixture into an outcome-neutral feasibility failure."""
    output = root / "results/activity_matched_response"
    calibration = (
        output
        / "calibration/checkpoints"
        / "local__seed_2000__target_04.json"
    )
    row = json.loads(calibration.read_text(encoding="utf-8"))
    target = float(row["target_activity"])
    branch = row["branch"]
    row["status"] = "censored_target_unreachable"
    for key in (
        "matched_rate",
        "matched_activity",
        "absolute_error",
        "relative_error",
        "iterations",
        "monotonicity",
    ):
        row.pop(key, None)
    row["reachable_interval"] = [0.4 * target, 0.8 * target]
    row["evaluations"] = [
        {
            "rate": branch["lower_rate"],
            "activity": 0.4 * target,
            "jump_strength": 80.0 * branch["lower_rate"],
            "maximum_trace_error": 1e-12,
            "maximum_activity_imaginary_residue": 1e-15,
        },
        {
            "rate": branch["upper_rate"],
            "activity": 0.8 * target,
            "jump_strength": 80.0 * branch["upper_rate"],
            "maximum_trace_error": 1e-12,
            "maximum_activity_imaginary_residue": 1e-15,
        },
    ]
    _write(calibration, json.dumps(row))

    (output / "frozen_calibration.json").unlink()
    (output / "aggregate.json").unlink()
    for path in (output / "score/checkpoints").glob("*.json"):
        path.unlink()
    return calibration.relative_to(output).as_posix()


def _commit_all(root: Path, message: str = "fixture") -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", message, "--allow-empty"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _complete_repo(
    tmp_path: Path,
    *,
    initialize_git: bool = True,
    activity_failed: bool = False,
) -> Path:
    root = tmp_path / "repo"
    for relative in (
        *package.ROOT_SOURCE_FILES,
        *package.REQUIRED_SOURCE_FILES,
    ):
        _write(root / relative)
    # Add ordinary scientific files selected through the broad source patterns.
    _write(root / "src/qrc/extra.py")
    _write(root / "experiments/analysis/helper.py")
    _write(root / "HANDOFF.md", "FINAL-SEAL: complete\n")
    _write(root / "paper/FIGURE_QA.md", "Figure QA: complete\n")

    for relative in package.CURRENT_REPORTS:
        _write(root / relative)

    frozen_relative = package.RESULT_DEPENDENCIES[0]
    frozen_payload = {
        "diagnostic_rows": [{"seed": seed} for seed in range(20)],
        "predictions_by_N": {
            "4": {"diagnostic_selected_intermediate_alpha": 0.8},
            "5": {"diagnostic_selected_intermediate_alpha": 0.8},
        },
    }
    frozen_data = (
        json.dumps(frozen_payload, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write(root / "results" / frozen_relative, frozen_data)
    frozen_sha = hashlib.sha256(frozen_data).hexdigest()

    tuning = root / "results/revision_tuning"
    for relative in package.RESULT_GROUPS[0].required_files:
        _write(tuning / relative, "{}\n")
    _write(
        tuning / "critique_response_inference.json",
        json.dumps({"status": "complete"}) + "\n",
    )
    fresh_protocol_sha = ""
    for stage, stage_label in (
        ("strength_extension", "strength"),
        ("nested_tuning", "nested"),
        ("fresh_interpolation", "fresh"),
    ):
        protocol_sha = ""
        snapshot = f"# sealed {stage_label}-stage fixture\n".encode()
        snapshot_sha = hashlib.sha256(snapshot).hexdigest()
        source_hashes = {
            "experiments/run_revision_tuning.py": snapshot_sha,
        }
        snapshot_manifest = {
            "artifact_type": f"{stage_label}_stage_source_snapshot",
            f"{stage_label}_protocol_sha256": protocol_sha,
            "source_path_in_protocol": "experiments/run_revision_tuning.py",
            "snapshot_path": (
                f"results/revision_tuning/{stage}/source_snapshot/"
                "run_revision_tuning.py"
            ),
            "sha256": snapshot_sha,
        }
        if stage == "fresh_interpolation":
            helper = b"# sealed fresh-only finalizer fixture\n"
            helper_sha = hashlib.sha256(helper).hexdigest()
            helper_source = "experiments/run_revision_fresh_interpolation.py"
            helper_snapshot = (
                "results/revision_tuning/fresh_interpolation/"
                "source_snapshot/run_revision_fresh_interpolation.py"
            )
            source_hashes[helper_source] = helper_sha
            snapshot_manifest.update(
                {
                    "helper_path_in_protocol": helper_source,
                    "helper_snapshot_path": helper_snapshot,
                    "helper_sha256": helper_sha,
                }
            )
            _write(tuning / helper_snapshot.removeprefix(
                "results/revision_tuning/"
            ), helper)
        stage_protocol = {
            "scientific_sources_sha256": source_hashes,
        }
        if stage == "fresh_interpolation":
            stage_protocol["frozen_diagnostic_source"] = {
                "path": f"results/{frozen_relative}",
                "sha256": frozen_sha,
                "diagnostic_seed_count": 20,
            }
        protocol_sha = _protocol_hash(stage_protocol)
        snapshot_manifest[f"{stage_label}_protocol_sha256"] = protocol_sha
        if stage == "fresh_interpolation":
            fresh_protocol_sha = protocol_sha
        _write(
            tuning / f"{stage}/manifest.json",
            json.dumps(
                {
                    "manifest_status": "frozen_before_stage_rows",
                    "protocol": stage_protocol,
                    "protocol_sha256": protocol_sha,
                }
            ),
        )
        _write(
            tuning / f"{stage}/source_snapshot/run_revision_tuning.py",
            snapshot,
        )
        _write(
            tuning / f"{stage}/source_snapshot/manifest.json",
            json.dumps(snapshot_manifest),
        )
    _write(
        tuning / "strength_extension/six_channel_aggregate.json",
        json.dumps(_complete_strength_aggregate()),
    )
    _write(
        tuning / "strength_extension/alternative_matching_aggregate.json",
        json.dumps(_complete_alternative_matching_aggregate()),
    )
    _write(
        tuning / "nested_tuning/nested_tuning_results.json",
        json.dumps(_complete_nested_result()),
    )
    _write(
        tuning / "fresh_interpolation/fresh_interpolation_results.json",
        json.dumps(
            _complete_fresh_result(
                frozen_payload,
                {
                    "path": f"results/{frozen_relative}",
                    "sha256": frozen_sha,
                    "diagnostic_seed_count": 20,
                },
                fresh_protocol_sha,
            )
        ),
    )
    _write(tuning / "nested_tuning/test_jobs/B3_collective_s1.json", "{}\n")
    _write_nested_extension_fixture(root)

    measurement = root / "results/measurement_full_v3"
    _write(measurement / "protocol.json", "{}\n")
    _write(
        measurement / "measurement_full_aggregate.json",
        json.dumps({"validation": {"status": "complete"}}),
    )
    _write(measurement / "jobs/CD_paper__s1.json", "{}\n")

    parity = root / "results/revision_parity_control"
    parity_results = _complete_parity_results()
    for name, payload in parity_results.items():
        _write(parity / name, json.dumps(payload))
    for row in parity_results["paper_aggregate.json"]["raw_rows"]:
        _write(
            parity / f"paper__N5_{row['method']}_s{row['seed']}.json",
            json.dumps(row),
        )
    for row in parity_results["paper_reference_aggregate.json"]["raw_rows"]:
        _write(
            parity
            / f"paper_reference__N5_{row['method']}_s{row['seed']}.json",
            json.dumps(row),
        )

    scaling = root / "results/revision_normalized_scaling"
    for name, payload in _complete_scaling_results().items():
        _write(scaling / name, json.dumps(payload))

    _write_primary_regularization_fixture(root)
    _write_collective_full_input_fixture(root)

    archives = {
        "final_protocol_results.tar.gz": _safe_tar_bytes("final/data.json"),
        "review_protocol_results.tar.gz": _safe_tar_bytes("review/data.json"),
    }
    checksum_lines = []
    for name, data in archives.items():
        _write(root / "results" / name, data)
        checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    _write(
        root / "results" / package.BASELINE_CHECKSUM_FILE,
        "\n".join(checksum_lines) + "\n",
    )
    _write_primary_driven_activity_fixture(root)
    _write(
        root / "results/forecast_baseline_audit/aggregate.json",
        json.dumps({"status": "complete"}),
    )
    _write_activity_matched_response_fixture(root)
    if activity_failed:
        _make_activity_fixture_failed_terminal(root)
    if initialize_git:
        subprocess.run(
            ["git", "init", "-q"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Packaging Test"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "packaging-test@example.invalid"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        _commit_all(root)
    return root


def _member_names(path: Path) -> set[str]:
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as bundle:
            return set(bundle.namelist())
    with tarfile.open(path, "r:gz") as bundle:
        return {member.name for member in bundle.getmembers()}


def _clone(value):
    return json.loads(json.dumps(value))


def _activity_fixture_parsed(
    tmp_path: Path,
    *,
    failed: bool = False,
) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "activity-repo"
    for relative in (
        "experiments/run_activity_matched_response.py",
        "src/qrc/dissipators.py",
        "src/qrc/liouvillian.py",
        "src/qrc/operators.py",
        "src/qrc/readout.py",
        "src/qrc/reservoirs.py",
        "src/qrc/sparse_evolve.py",
        "src/qrc/tasks.py",
    ):
        _write(root / relative)
    _write_activity_matched_response_fixture(root)
    if failed:
        _make_activity_fixture_failed_terminal(root)
    directory = root / "results/activity_matched_response"
    parsed = {
        path.relative_to(directory).as_posix(): json.loads(
            path.read_text(encoding="utf-8")
        )
        for path in directory.rglob("*.json")
    }
    return directory, parsed


@pytest.mark.parametrize("suffix", (".zip", ".tar.gz"))
def test_final_build_is_byte_deterministic_and_verifiable(tmp_path, suffix):
    root = _complete_repo(tmp_path)
    first = tmp_path / f"first{suffix}"
    second = tmp_path / f"second{suffix}"
    summary_a = package.build_package(root, first, require_complete=True)
    summary_b = package.build_package(root, second, require_complete=True)

    assert summary_a.complete
    assert first.read_bytes() == second.read_bytes()
    assert summary_a.sha256 == summary_b.sha256
    verified = package.verify_archive(first)
    assert verified["complete"] is True
    assert verified["sha256"] == summary_a.sha256
    sidecar = first.with_name(first.name + ".sha256")
    assert sidecar.read_text().endswith(
        f"  {first.name}\n"
    )
    sidecar.write_text(f"{'0' * 64}  {first.name}\n")
    with pytest.raises(package.EvidencePackageError, match="sidecar mismatch"):
        package.verify_archive(first)


def test_strict_mode_requires_a_real_committed_git_repository(tmp_path):
    root = _complete_repo(tmp_path, initialize_git=False)
    with pytest.raises(
        package.IncompleteEvidenceError,
        match="valid Git repository",
    ):
        package.build_package(
            root,
            tmp_path / "not-git.zip",
            require_complete=True,
        )


def test_strict_mode_rejects_a_dirty_git_repository(tmp_path):
    root = _complete_repo(tmp_path)
    _write(root / "README.md", "uncommitted\n")
    with pytest.raises(
        package.IncompleteEvidenceError,
        match="clean committed tree",
    ):
        package.build_package(
            root,
            tmp_path / "dirty.zip",
            require_complete=True,
        )


def test_strict_mode_can_rebuild_its_tracked_output_in_place(tmp_path):
    root = _complete_repo(tmp_path)
    output = root / "results/qrc_dissipation_reproducibility_package.zip"
    sidecar = output.with_name(output.name + ".sha256")
    _write(output, b"old archive")
    _write(sidecar, "old sidecar\n")
    _commit_all(root, "add prior evidence package")

    _write(output, b"stale generated archive")
    _write(sidecar, "stale generated sidecar\n")
    summary = package.build_package(root, output, require_complete=True)

    assert summary.complete
    assert package.verify_archive(output, require_complete=True)["complete"]


@pytest.mark.parametrize(
    ("relative", "text"),
    (
        ("HANDOFF.md", "FINAL-SEAL — prospective control: PENDING.\n"),
        ("paper/FIGURE_QA.md", "Font audit: PENDING — record result.\n"),
    ),
)
def test_strict_mode_rejects_release_documentation_pending(
    tmp_path,
    relative,
    text,
):
    root = _complete_repo(tmp_path)
    _write(root / relative, text)
    _commit_all(root, "add release blocker")
    with pytest.raises(
        package.IncompleteEvidenceError,
        match="release_documentation",
    ):
        package.build_package(
            root,
            tmp_path / "pending.zip",
            require_complete=True,
        )


def test_generated_readme_has_executable_reconstruction_layout(tmp_path):
    root = _complete_repo(tmp_path)
    output = tmp_path / "evidence.zip"
    package.build_package(root, output, require_complete=True)
    with zipfile.ZipFile(output) as bundle:
        readme = bundle.read(f"{package.ARCHIVE_ROOT}/README.md").decode()
    for command in (
        "cp -R source/. reproduction/",
        "cp -R results reproduction/",
        "cp -R reports reproduction/",
        "tar -xzf results/final_protocol_results.tar.gz -C results",
        "tar -xzf results/review_protocol_results.tar.gz -C results",
        "python -m pip install -r requirements.txt",
        "PYTHONPATH=src:experiments python -m pytest -q",
        "experiments/run_activity_matched_response.py validate",
        "MPLCONFIGDIR=.mplconfig python paper/make_figures.py",
        "latexmk -pdf -interaction=nonstopmode -halt-on-error",
    ):
        assert command in readme


def test_normal_cli_verification_requires_the_sidecar(tmp_path):
    root = _complete_repo(tmp_path)
    output = tmp_path / "evidence.zip"
    summary = package.build_package(root, output, require_complete=True)
    summary.sidecar.unlink()
    with pytest.raises(package.EvidencePackageError, match="missing archive sidecar"):
        package.main(["verify", str(output)])


def test_cli_require_complete_rejects_a_partial_archive(tmp_path):
    root = _complete_repo(tmp_path)
    (
        root
        / "results/revision_normalized_scaling/paper_variance_aggregate.json"
    ).unlink()
    output = tmp_path / "partial.zip"
    package.build_package(root, output, require_complete=False)
    with pytest.raises(package.IncompleteEvidenceError, match="incomplete"):
        package.main(["verify", "--require-complete", str(output)])


def test_nested_zip_rejects_nonregular_unix_members(tmp_path):
    archive = tmp_path / "unsafe.zip"
    info = zipfile.ZipInfo("named-pipe")
    info.create_system = 3
    info.external_attr = (0o010000 | 0o644) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(info, b"")
    with pytest.raises(package.EvidencePackageError, match="unsupported member"):
        package._validate_nested_archive(archive)


def test_partial_snapshot_records_missing_groups_but_strict_mode_fails(tmp_path):
    root = _complete_repo(tmp_path)
    target = root / "results/revision_normalized_scaling/paper_variance_aggregate.json"
    target.unlink()
    output = tmp_path / "partial.zip"

    summary = package.build_package(root, output, require_complete=False)
    assert summary.complete is False
    with zipfile.ZipFile(output) as bundle:
        provenance = json.loads(
            bundle.read(f"{package.ARCHIVE_ROOT}/PROVENANCE.json")
        )
    scaling = provenance["result_groups"]["revision_normalized_scaling"]
    assert scaling["status"] == "partial"
    assert "paper_variance_aggregate.json" in scaling["missing_required_files"]

    _commit_all(root, "remove scaling aggregate")
    with pytest.raises(package.IncompleteEvidenceError):
        package.build_package(
            root,
            tmp_path / "strict.zip",
            require_complete=True,
        )


def test_allowlist_excludes_superseded_smoke_cache_secret_and_build_files(tmp_path):
    root = _complete_repo(tmp_path)
    _write(root / "results/measurement_full/jobs/old.json", "{}")
    _write(root / "results/measurement_full_v2/jobs/old.json", "{}")
    _write(
        root
        / "results/quantum_strengthening_v2_paper/"
        "same_seed_task_scores.json",
        "{}",
    )
    _write(root / "results/measurement_full_v3_smoke/jobs/smoke.json", "{}")
    _write(
        root / "results/revision_parity_control/smoke_aggregate.json",
        "{}",
    )
    _write(
        root
        / "results/revision_parity_control/"
        "paper_reference_prefilter_aggregate.json",
        "{}",
    )
    _write(
        root
        / "results/revision_parity_control/"
        "paper_reference_prefilter__N5_B1_dephasing_s1.json",
        "{}",
    )
    _write(
        root
        / "results/activity_matched_response_exploratory_branch_audit/"
        "pilot_manifest.json",
        '{"development_only": true}\n',
    )
    _write(
        root
        / "results/activity_matched_response_failed_v2_reachability_audit/"
        "pilot_manifest.json",
        '{"failed_reachability_audit": true}\n',
    )
    _write(root / ".env", "SECRET=do-not-package")
    _write(root / "experiments/__pycache__/driver.pyc", b"cache")
    _write(root / "paper/dissipation_qrc.pdf", b"%PDF")
    _write(root / "build/output.bin", b"build")
    _commit_all(root, "add excluded files")

    output = tmp_path / "evidence.zip"
    package.build_package(root, output, require_complete=True)
    names = _member_names(output)
    joined = "\n".join(sorted(names))
    assert "measurement_full/jobs" not in joined
    assert "measurement_full_v2" not in joined
    assert "measurement_full_v3_smoke" not in joined
    assert (
        "/results/quantum_strengthening_v2_paper/"
        "frozen_diagnostic_predictions.json"
    ) in joined
    assert "same_seed_task_scores.json" not in joined
    assert "smoke_aggregate" not in joined
    assert "/paper_reference_aggregate.json" in joined
    assert "paper_reference_prefilter" not in joined
    assert ".env" not in joined
    assert "__pycache__" not in joined
    assert "/paper/make_figures.py" in joined
    assert "/paper/dissipation_qrc.tex" in joined
    assert "/paper/dissipation_qrc.pdf" not in joined
    for stage in (
        "strength_extension",
        "nested_tuning",
        "fresh_interpolation",
    ):
        assert (
            f"/results/revision_tuning/{stage}/source_snapshot/"
            "run_revision_tuning.py"
        ) in joined
    assert (
        "/results/revision_tuning/fresh_interpolation/source_snapshot/"
        "run_revision_fresh_interpolation.py"
    ) in joined
    assert (
        "/results/revision_tuning/nested_operating_point_extension/"
        "source_snapshot/run_nested_operating_point_extension.py"
    ) in joined
    for source in (
        "experiments/run_revision_primary_regularization.py",
        "experiments/validate_revision_primary_regularization_artifacts.py",
        "experiments/run_collective_loss_full_input_diagnostic.py",
        "experiments/validate_collective_loss_full_input_artifacts.py",
        "experiments/run_nested_operating_point_extension.py",
        "experiments/validate_nested_operating_point_artifacts.py",
        "experiments/audit_nested_prescreen_stability.py",
        "tests/test_nested_prescreen_stability.py",
    ):
        assert f"/source/{source}" in joined
    assert "/results/revision_primary_regularization/aggregate.json" in joined
    assert "/results/revision_primary_regularization/jobs/" in joined
    assert (
        "/results/collective_loss_full_input_diagnostic/raw_spectrum.json"
        in joined
    )
    assert (
        "/results/revision_tuning/nested_operating_point_extension/"
        "prescreen_stability.json"
    ) in joined
    assert "/reports/nested_prescreen_stability_audit.md" in joined
    assert "/reports/review_response_v7.md" in joined
    assert "/reports/primary_driven_activity_report.md" in joined
    assert (
        "/reports/activity_matched_response_development_audit.md"
        in joined
    )
    assert (
        "/reports/activity_matched_response_v2_failure_audit.md"
        in joined
    )
    assert (
        "/reports/activity_matched_response_v2_recovery_plan.md"
        in joined
    )
    assert "/reports/activity_matched_response_report.md" in joined
    assert "/reports/review_response_v6.md" not in joined
    assert "/reports/review_response_v5.md" not in joined
    assert "/results/primary_driven_activity/aggregate.json" in joined
    assert "/results/primary_driven_activity/checkpoints/" in joined
    assert "/results/activity_matched_response/aggregate.json" in joined
    assert (
        "/results/activity_matched_response/pilot/checkpoints/" in joined
    )
    assert (
        "/results/activity_matched_response/calibration/checkpoints/"
        in joined
    )
    assert (
        "/results/activity_matched_response/score/checkpoints/" in joined
    )
    assert (
        "/results/activity_matched_response/source_snapshot/"
        "run_activity_matched_response.py"
    ) in joined
    assert "activity_matched_response_exploratory_branch_audit" not in joined
    assert (
        "activity_matched_response_failed_v2_reachability_audit"
        not in joined
    )
    assert "/build/" not in joined


@pytest.mark.parametrize("failure", ("missing", "tampered"))
def test_strict_mode_authenticates_frozen_diagnostic_dependency(
    tmp_path,
    failure,
):
    root = _complete_repo(tmp_path)
    dependency = (
        root
        / "results/quantum_strengthening_v2_paper/"
        "frozen_diagnostic_predictions.json"
    )
    if failure == "missing":
        dependency.unlink()
    else:
        dependency.write_text('{"tampered": true}\n', encoding="utf-8")
    _commit_all(root, f"{failure} frozen diagnostic dependency")

    with pytest.raises(
        package.IncompleteEvidenceError,
        match="result_dependencies",
    ):
        package.build_package(
            root,
            tmp_path / f"{failure}.zip",
            require_complete=True,
        )


def test_strict_mode_rejects_frozen_rows_changed_only_in_fresh_aggregate(
    tmp_path,
):
    root = _complete_repo(tmp_path)
    aggregate_path = (
        root
        / "results/revision_tuning/fresh_interpolation/"
        "fresh_interpolation_results.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["frozen_diagnostic_rows"] = aggregate[
        "frozen_diagnostic_rows"
    ][1:]
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    _commit_all(root, "tamper embedded frozen rows")

    with pytest.raises(
        package.IncompleteEvidenceError,
        match="result_dependencies",
    ):
        package.build_package(
            root,
            tmp_path / "tampered-embedded-rows.zip",
            require_complete=True,
        )


def test_revision_manifest_digest_authenticates_protocol():
    protocol = {"stage": "test", "value": 1}
    complete, reason = package._revision_manifest_is_frozen(
        {
            "manifest_status": "frozen_before_stage_rows",
            "protocol": protocol,
            "protocol_sha256": _protocol_hash(protocol),
        }
    )
    assert complete is True
    assert "hash-authenticated" in reason

    complete, reason = package._revision_manifest_is_frozen(
        {
            "manifest_status": "frozen_before_stage_rows",
            "protocol": {"stage": "test", "value": 2},
            "protocol_sha256": _protocol_hash(protocol),
        }
    )
    assert complete is False
    assert "does not authenticate" in reason


def test_strict_mode_rejects_unbracketed_strength_aggregate(tmp_path):
    root = _complete_repo(tmp_path)
    path = (
        root
        / "results/revision_tuning/strength_extension/six_channel_aggregate.json"
    )
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    aggregate["collective_optimum_bracketed"] = False
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    _commit_all(root, "break strength bracket")

    with pytest.raises(package.IncompleteEvidenceError, match="final evidence"):
        package.build_package(
            root,
            tmp_path / "unbracketed.zip",
            require_complete=True,
        )


def test_scaling_validator_accepts_only_hash_linked_production_grid():
    parsed = _complete_scaling_results()
    complete, reason = package._revision_scaling_is_complete(parsed)
    assert complete, reason

    broken = _clone(parsed)
    broken["paper_variance_aggregate.json"]["protocol_sha256"] = "f" * 64
    complete, reason = package._revision_scaling_is_complete(broken)
    assert complete is False
    assert "not linked" in reason

    broken = _clone(parsed)
    row = broken["paper_variance_aggregate.json"]["raw_rows"][1]
    row["input_sha256"] = _digest("unpaired-input")
    checkpoint = (
        f"paper__variance_N{row['n_qubits']}_{row['method']}_s{row['seed']}.json"
    )
    broken[checkpoint] = row
    complete, reason = package._revision_scaling_is_complete(broken)
    assert complete is False
    assert "paired hashes" in reason

    broken = _clone(parsed)
    broken["paper_variance_aggregate.json"]["ridge_boundary_audit"][
        "n_unresolved_upper"
    ] = 1
    complete, reason = package._revision_scaling_is_complete(broken)
    assert complete is False
    assert "ridge upper boundary" in reason

    broken = _clone(parsed)
    broken.pop(next(name for name in broken if name.startswith("paper__")))
    complete, reason = package._revision_scaling_is_complete(broken)
    assert complete is False
    assert "checkpoint file set" in reason


def test_primary_regularization_validator_requires_full_baseline_audit(tmp_path):
    root = tmp_path / "primary"
    _write_primary_regularization_fixture(root)
    directory = root / "results/revision_primary_regularization"
    parsed = {
        path.relative_to(directory).as_posix(): json.loads(path.read_text())
        for path in directory.rglob("*.json")
    }
    complete, reason = package._revision_primary_regularization_is_complete(
        parsed
    )
    assert complete, reason

    broken = _clone(parsed)
    broken["aggregate.json"]["baseline_reproduction_audit"]["passed"] = False
    complete, reason = package._revision_primary_regularization_is_complete(
        broken
    )
    assert complete is False
    assert "baseline audit failed" in reason

    broken = _clone(parsed)
    broken["protocol.json"]["baseline_reproduction"]["entries"].pop(
        next(iter(broken["protocol.json"]["baseline_reproduction"]["entries"]))
    )
    broken["aggregate.json"]["protocol_sha256"] = _protocol_hash(
        broken["protocol.json"]
    )
    complete, reason = package._revision_primary_regularization_is_complete(
        broken
    )
    assert complete is False
    assert "448-entry" in reason

    broken = _clone(parsed)
    broken["aggregate.json"]["task_summaries"]["stm"]["method_summaries"][
        "B3_collective"
    ]["selected_test"]["mean"] += 1.0
    complete, reason = package._revision_primary_regularization_is_complete(
        broken
    )
    assert complete is False
    assert "derived summary disagrees" in reason

    broken = _clone(parsed)
    row = broken["aggregate.json"]["rows"][0]
    row["task_results"]["stm"]["selected_ridge"] = 0.0
    job_name = f"jobs/{row['method']}__s{row['seed']}.json"
    broken[job_name] = _clone(row)
    complete, reason = package._revision_primary_regularization_is_complete(
        broken
    )
    assert complete is False
    assert "selection/test derivation failed" in reason


def test_activity_matched_validator_accepts_complete_negative_outcome(
    tmp_path,
):
    directory, parsed = _activity_fixture_parsed(tmp_path)
    aggregate = parsed["aggregate.json"]
    assert isinstance(aggregate, dict)
    assert aggregate["claim_gates"][
        "activity_matched_dominance_claim_allowed"
    ] is False

    complete, reason = package._activity_matched_response_is_complete(
        directory,
        parsed,
    )
    assert complete, reason
    assert "complete staged" in reason


def test_activity_matched_validator_accepts_complete_failed_feasibility(
    tmp_path,
):
    directory, parsed = _activity_fixture_parsed(tmp_path, failed=True)
    pilot_names = [
        name for name in parsed if name.startswith("pilot/checkpoints/")
    ]
    calibration_names = [
        name
        for name in parsed
        if name.startswith("calibration/checkpoints/")
    ]
    assert len(pilot_names) == 160
    assert len(calibration_names) == 240
    assert "frozen_calibration.json" not in parsed
    assert "aggregate.json" not in parsed
    assert not any(name.startswith("score/") for name in parsed)

    complete, reason = package._activity_matched_response_is_complete(
        directory,
        parsed,
    )
    assert complete, reason
    assert "outcome-neutral feasibility failure" in reason
    assert "1 of 240 calibration cells censored" in reason


def test_activity_matched_failed_feasibility_rejects_score_leakage(
    tmp_path,
):
    directory, parsed = _activity_fixture_parsed(tmp_path, failed=True)
    name = "score/checkpoints/leaked_task_score.json"
    leaked = {"test_stm_capacity": 9.0}
    _write(directory / name, json.dumps(leaked))
    parsed[name] = leaked

    complete, reason = package._activity_matched_response_is_complete(
        directory,
        parsed,
    )
    assert complete is False
    assert "task-score leakage" in reason


def test_activity_matched_failed_feasibility_rejects_partial_calibration(
    tmp_path,
):
    directory, parsed = _activity_fixture_parsed(tmp_path, failed=True)
    name = next(
        name
        for name in parsed
        if name.startswith("calibration/checkpoints/")
    )
    parsed.pop(name)

    complete, reason = package._activity_matched_response_is_complete(
        directory,
        parsed,
    )
    assert complete is False
    assert "calibration checkpoint coverage" in reason


def test_activity_matched_failed_feasibility_rejects_tampering(tmp_path):
    directory, parsed = _activity_fixture_parsed(tmp_path, failed=True)
    name = next(
        name
        for name, row in parsed.items()
        if name.startswith("calibration/checkpoints/")
        and isinstance(row, dict)
        and row.get("status") == "matched"
    )
    parsed[name]["target_activity"] *= 0.9

    complete, reason = package._activity_matched_response_is_complete(
        directory,
        parsed,
    )
    assert complete is False
    assert "calibration invariant" in reason


def test_failed_activity_terminal_state_is_complete_and_archives_raw_rows(
    tmp_path,
):
    root = _complete_repo(tmp_path, activity_failed=True)
    output = tmp_path / "failed-terminal.zip"
    summary = package.build_package(root, output, require_complete=True)
    assert summary.complete

    names = _member_names(output)
    prefix = (
        f"{package.ARCHIVE_ROOT}/results/activity_matched_response/"
    )
    pilot_rows = {
        name
        for name in names
        if name.startswith(f"{prefix}pilot/checkpoints/")
    }
    calibration_rows = {
        name
        for name in names
        if name.startswith(f"{prefix}calibration/checkpoints/")
    }
    assert len(pilot_rows) == 160
    assert len(calibration_rows) == 240
    assert f"{prefix}frozen_calibration.json" not in names
    assert f"{prefix}aggregate.json" not in names
    assert not any(name.startswith(f"{prefix}score/") for name in names)
    with zipfile.ZipFile(output) as bundle:
        readme = bundle.read(
            f"{package.ARCHIVE_ROOT}/README.md"
        ).decode("utf-8")
    assert "outcome-neutral feasibility failure" in readme
    assert "run_activity_matched_response.py validate" not in readme


def test_activity_matched_validator_is_fail_closed_across_stages(tmp_path):
    directory, parsed = _activity_fixture_parsed(tmp_path)

    broken = _clone(parsed)
    broken["pilot_manifest.json"]["protocol"]["status"] = "changed"
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "pilot manifest" in reason

    broken = _clone(parsed)
    broken["frozen_targets.json"]["common_activity_interval"][0] *= 0.9
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "frozen targets" in reason

    broken = _clone(parsed)
    calibration_name = next(
        name
        for name in broken
        if name.startswith("calibration/checkpoints/")
    )
    broken.pop(calibration_name)
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "calibration checkpoint coverage" in reason

    broken = _clone(parsed)
    broken["frozen_calibration.json"]["cells"][0][
        "calibration_row_sha256"
    ] = "0" * 64
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "cell/hash" in reason

    broken = _clone(parsed)
    score_names = [
        name for name in broken if name.startswith("score/checkpoints/")
    ]
    first_score = broken[score_names[0]]
    first_score["task_input_sha256"] = _digest("unpaired-task-input")
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "paired streams" in reason

    broken = _clone(parsed)
    first_score = broken[score_names[0]]
    first_score["task_input_sha256"] = first_score[
        "calibration_input_sha256"
    ]
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "score invariant/linkage" in reason

    broken = _clone(parsed)
    broken["aggregate.json"]["target_results"][0][
        "stm_collective_minus_local"
    ]["paired_values"][0] += 1.0
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "derived summary" in reason

    broken = _clone(parsed)
    broken["aggregate.json"]["claim_gates"][
        "activity_matched_dominance_claim_allowed"
    ] = True
    complete, reason = package._activity_matched_response_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "claim gates" in reason


def test_activity_matched_validator_authenticates_source_snapshot(tmp_path):
    directory, parsed = _activity_fixture_parsed(tmp_path)
    snapshot = (
        directory
        / "source_snapshot/run_activity_matched_response.py"
    )
    snapshot.write_text("# tampered\n", encoding="utf-8")

    complete, reason = package._activity_matched_response_is_complete(
        directory,
        parsed,
    )
    assert complete is False
    assert "source snapshot" in reason


def test_collective_full_input_validator_requires_grid_hashes_and_sparse_checks(
    tmp_path,
):
    root = tmp_path / "collective"
    _write_collective_full_input_fixture(root)
    directory = root / "results/collective_loss_full_input_diagnostic"
    parsed = {
        path.name: json.loads(path.read_text())
        for path in directory.glob("*.json")
    }
    complete, reason = package._collective_loss_full_input_is_complete(
        directory,
        parsed,
    )
    assert complete, reason

    broken = _clone(parsed)
    broken["raw_spectrum.json"]["rows"][0]["stationary_mode_count"] = 2
    complete, reason = package._collective_loss_full_input_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "dense audit failed" in reason

    broken = _clone(parsed)
    crosschecked = next(
        row
        for row in broken["raw_spectrum.json"]["rows"]
        if row["sparse_crosscheck"] is not None
    )
    crosschecked["sparse_crosscheck"]["near_zero_sparse_stationary_count"] = 2
    complete, reason = package._collective_loss_full_input_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "sparse audit failed" in reason

    broken = _clone(parsed)
    broken["aggregate.json"]["mean_sampled_gap"] += 0.1
    complete, reason = package._collective_loss_full_input_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "aggregate summary disagrees" in reason

    broken = _clone(parsed)
    spectrum_row = broken["raw_spectrum.json"]["rows"][0]
    spectrum_row["spectrum"][1][0] -= 0.1
    packed = b"".join(
        struct.pack("<dd", *value) for value in spectrum_row["spectrum"]
    )
    spectrum_row["spectrum_sha256"] = hashlib.sha256(packed).hexdigest()
    complete, reason = package._collective_loss_full_input_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "spectral fields disagree" in reason


def test_nested_extension_validator_requires_freeze_and_common_grid(tmp_path):
    root = _complete_repo(tmp_path)
    tuning_spec = next(
        spec for spec in package.RESULT_GROUPS if spec.name == "revision_tuning"
    )
    _, status = package._collect_result_group(root, tuning_spec)
    check = status["aggregate_checks"][
        "specific:nested-operating-point-extension"
    ]
    assert check["complete"] is True, check["reason"]

    aggregate_path = (
        root
        / "results/revision_tuning/nested_operating_point_extension/"
        "aggregate.json"
    )
    aggregate = json.loads(aggregate_path.read_text())
    aggregate["freeze_before_test_verified"] = False
    unhashed = {
        key: value
        for key, value in aggregate.items()
        if key != "deterministic_payload_sha256"
    }
    aggregate["deterministic_payload_sha256"] = _protocol_hash(unhashed)
    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
    _, status = package._collect_result_group(root, tuning_spec)
    check = status["aggregate_checks"][
        "specific:nested-operating-point-extension"
    ]
    assert check["complete"] is False
    assert "freeze hash chain" in check["reason"]

    root = _complete_repo(tmp_path / "row-linkage")
    directory = root / "results/revision_tuning"
    parsed = {
        path.relative_to(directory).as_posix(): json.loads(path.read_text())
        for path in directory.rglob("*.json")
    }
    reuse = parsed[
        "nested_operating_point_extension/reuse_index.json"
    ]
    entry = reuse["screen_new"][0]
    relative = Path(entry["path"]).relative_to("results/revision_tuning")
    parsed[relative.as_posix()]["h"] += 1.0
    complete, reason = package._nested_operating_point_extension_is_complete(
        directory,
        parsed,
    )
    assert complete is False
    assert "provenance row content/hash" in reason

    parsed = {
        path.relative_to(directory).as_posix(): json.loads(path.read_text())
        for path in directory.rglob("*.json")
    }
    frozen = parsed[
        "nested_operating_point_extension/frozen_selection.json"
    ]
    frozen["chosen"]["CD_paper"]["best_ridge"] = 0.5
    frozen["selection_ranking"]["CD_paper"][0]["best_ridge"] = 0.5
    complete, reason = package._nested_operating_point_extension_is_complete(
        directory,
        parsed,
    )
    assert complete is False
    assert "chosen ridge/config is invalid" in reason


def test_nested_prescreen_validator_is_fail_closed(tmp_path):
    root = _complete_repo(tmp_path)
    directory = root / "results/revision_tuning"
    parsed = {
        path.relative_to(directory).as_posix(): json.loads(path.read_text())
        for path in directory.rglob("*.json")
    }
    key = (
        "nested_operating_point_extension/prescreen_stability.json"
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        parsed,
    )
    assert complete, reason

    def with_audit(mutator):
        audit = _clone(parsed[key])
        mutator(audit)
        audit_without_hash = {
            field: value
            for field, value in audit.items()
            if field != "deterministic_payload_sha256"
        }
        audit["deterministic_payload_sha256"] = _protocol_hash(
            audit_without_hash
        )
        result = dict(parsed)
        result[key] = audit
        return result

    broken = _clone(parsed)
    broken[key]["deterministic_payload_sha256"] = "0" * 64
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "type/status/hash" in reason

    broken = with_audit(
        lambda audit: audit.__setitem__(
            "claim_boundary",
            "This is descriptive.",
        )
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "claim boundary" in reason

    broken = with_audit(
        lambda audit: audit.__setitem__("screen_seeds", [1])
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "two-seed" in reason

    broken = with_audit(
        lambda audit: audit.__setitem__(
            "configuration_count_per_method",
            audit["configuration_count_per_method"] - 1,
        )
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "configuration count" in reason

    broken = with_audit(
        lambda audit: audit["provenance"].__setitem__(
            "frozen_selection_sha256",
            "0" * 64,
        )
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "protocol/frozen provenance" in reason

    broken = with_audit(
        lambda audit: audit["provenance"].__setitem__(
            "screen_rows_sha256",
            "0" * 64,
        )
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "row provenance digest" in reason

    broken = with_audit(
        lambda audit: audit["methods"].pop("CD_paper")
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "method keys" in reason

    def invalidate_rank(audit):
        audit["methods"]["CD_paper"]["per_seed"][
            str(audit["screen_seeds"][0])
        ]["frozen_selected_config_rank"] = 0

    broken = with_audit(invalidate_rank)
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "ranks are invalid" in reason

    broken = with_audit(
        lambda audit: audit["methods"]["CD_paper"][
            "top8_overlap"
        ].__setitem__("intersection_count", 0)
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "overlap is invalid" in reason

    broken = with_audit(
        lambda audit: audit["methods"]["CD_paper"].__setitem__(
            "full_rank_spearman",
            2.0,
        )
    )
    complete, reason = package._nested_prescreen_stability_is_complete(
        directory,
        broken,
    )
    assert complete is False
    assert "method summary is invalid" in reason


def test_alternative_matching_validator_rejects_missing_or_false_censor_evidence():
    aggregate = _complete_alternative_matching_aggregate()
    complete, reason = package._revision_alternative_matching_is_complete(
        aggregate
    )
    assert complete, reason

    broken = _clone(aggregate)
    del broken["conditions"]["B5_pair__activity"]
    complete, reason = package._revision_alternative_matching_is_complete(
        broken
    )
    assert complete is False
    assert "complete 3x3 grid" in reason

    broken = _clone(aggregate)
    broken["conditions"]["B3_collective__gap"]["match_feasibility"][
        "reachable_count"
    ] = 32
    complete, reason = package._revision_alternative_matching_is_complete(
        broken
    )
    assert complete is False
    assert "censor flags disagree" in reason

    broken = _clone(aggregate)
    broken["full_driven_gap_curves"]["CD_paper"].pop()
    complete, reason = package._revision_alternative_matching_is_complete(
        broken
    )
    assert complete is False
    assert "curve is incomplete" in reason

    broken = _clone(aggregate)
    broken["raw_provenance"].pop()
    complete, reason = package._revision_alternative_matching_is_complete(
        broken
    )
    assert complete is False
    assert "446 rows" in reason

    broken = _clone(aggregate)
    broken["raw_provenance"][0]["path"] = (
        "results/review_protocol/R_match__unrelated_s0.json"
    )
    complete, reason = package._revision_alternative_matching_is_complete(
        broken
    )
    assert complete is False
    assert "do not cover raw rows" in reason


@pytest.mark.parametrize(
    ("stage", "filename"),
    (
        ("strength_extension", "run_revision_tuning.py"),
        ("nested_tuning", "run_revision_tuning.py"),
        ("fresh_interpolation", "run_revision_tuning.py"),
        ("fresh_interpolation", "run_revision_fresh_interpolation.py"),
    ),
)
def test_strict_mode_rejects_tampered_stage_source_snapshot(
    tmp_path,
    stage,
    filename,
):
    root = _complete_repo(tmp_path)
    snapshot = (
        root
        / "results/revision_tuning"
        / stage
        / "source_snapshot"
        / filename
    )
    snapshot.write_text("# tampered\n", encoding="utf-8")
    _commit_all(root, "tamper source snapshot")
    with pytest.raises(package.IncompleteEvidenceError, match="final evidence"):
        package.build_package(
            root,
            tmp_path / "tampered-snapshot.zip",
            require_complete=True,
        )


def test_partial_mode_records_missing_fresh_snapshot_without_hiding_other_stages(
    tmp_path,
):
    root = _complete_repo(tmp_path)
    missing = (
        root
        / "results/revision_tuning/fresh_interpolation/source_snapshot/"
        "run_revision_tuning.py"
    )
    missing.unlink()

    _, status = package._collect_result_group(root, package.RESULT_GROUPS[0])
    checks = status["aggregate_checks"]
    assert status["status"] == "partial"
    assert (
        "fresh_interpolation/source_snapshot/run_revision_tuning.py"
        in status["missing_required_files"]
    )
    assert checks["specific:strength-stage-source-snapshot"]["complete"] is True
    assert checks["specific:nested-stage-source-snapshot"]["complete"] is True
    assert checks["specific:fresh-stage-source-snapshot"]["complete"] is False


@pytest.mark.parametrize(
    ("stage", "field"),
    (
        ("strength_extension", "snapshot_path"),
        ("nested_tuning", "nested_protocol_sha256"),
        ("fresh_interpolation", "helper_snapshot_path"),
    ),
)
def test_stage_source_snapshot_metadata_must_match_frozen_protocol(
    tmp_path,
    stage,
    field,
):
    root = _complete_repo(tmp_path)
    path = (
        root
        / "results/revision_tuning"
        / stage
        / "source_snapshot/manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    _, status = package._collect_result_group(root, package.RESULT_GROUPS[0])
    check = status["aggregate_checks"][
        f"specific:{stage.split('_')[0]}-stage-source-snapshot"
    ]
    assert check["complete"] is False
    assert "snapshot" in check["reason"]


def test_flat_zero_parity_plateau_may_select_maximum_ridge(tmp_path):
    root = _complete_repo(tmp_path)
    path = (
        root
        / "results/revision_parity_control/paper_reference_aggregate.json"
    )
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    aggregate["ridge_boundary_audit"]["n_selected_maximum"] = 16
    aggregate["ridge_boundary_audit"]["n_unresolved_upper"] = 0
    aggregate["ridge_boundary_audit"]["upper_boundary_is_bracketed"] = True
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    _commit_all(root, "record flat plateau")

    summary = package.build_package(
        root,
        tmp_path / "flat-plateau.zip",
        require_complete=True,
    )
    assert summary.complete


def test_strict_mode_rejects_unresolved_parity_upper_boundary(tmp_path):
    root = _complete_repo(tmp_path)
    path = root / "results/revision_parity_control/paper_aggregate.json"
    aggregate = json.loads(path.read_text(encoding="utf-8"))
    aggregate["ridge_boundary_audit"]["n_unresolved_upper"] = 1
    aggregate["ridge_boundary_audit"]["upper_boundary_is_bracketed"] = False
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    _commit_all(root, "break parity ridge audit")

    with pytest.raises(package.IncompleteEvidenceError, match="final evidence"):
        package.build_package(
            root,
            tmp_path / "unresolved-parity.zip",
            require_complete=True,
        )


def test_present_baseline_with_wrong_checksum_is_rejected(tmp_path):
    root = _complete_repo(tmp_path)
    archive = root / "results/final_protocol_results.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(package.EvidencePackageError, match="checksum mismatch"):
        package.build_package(root, tmp_path / "bad.zip")


def test_verifier_rejects_hash_mismatch_and_unsafe_member(tmp_path):
    root = _complete_repo(tmp_path)
    valid = tmp_path / "valid.zip"
    package.build_package(root, valid, require_complete=True)

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(tampered, "w") as sink:
        for info in source.infolist():
            data = source.read(info)
            if info.filename.endswith("/README.md"):
                data += b"tamper"
            sink.writestr(info, data)
    with pytest.raises(package.EvidencePackageError, match="checksum mismatch"):
        package.verify_archive(tampered)

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as bundle:
        bundle.writestr("../escape", b"x")
    with pytest.raises(package.EvidencePackageError, match="unsafe archive path"):
        package.verify_archive(unsafe)

    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as bundle:
            bundle.writestr("same", b"one")
            bundle.writestr("same", b"two")
    with pytest.raises(package.EvidencePackageError, match="duplicate members"):
        package.verify_archive(duplicate)

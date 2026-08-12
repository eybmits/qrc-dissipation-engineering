"""Camera-ready extension of the nested local/collective operating-point search.

This is an append-only companion to ``run_revision_tuning.py``.  It reuses the
sealed screen and selection rows only after checking their protocol and source
hashes, extends the *common* local/collective strength screen through x128, and
uses a predeclared doubling rule if the collective choice still needs an upper
bracket.  Selection remains validation-only.  A deterministic test ensemble,
fresh relative to every namespace in the sealed source manifest, is evaluated
only after the selected operating points have been written and hashed.

Run from the repository root with the project environment:

    PYTHONPATH=src:experiments python \
      experiments/run_nested_operating_point_extension.py all --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
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

from _paths import REPORTS_DIR, RESULTS_DIR  # noqa: E402
import run_revision_tuning as base  # noqa: E402


PROTOCOL_VERSION = "nested-operating-point-extension-v1-2026-07-24"
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    Path(RESULTS_DIR)
    / "revision_tuning"
    / "nested_operating_point_extension"
)
SOURCE_ROOT = Path(RESULTS_DIR) / "revision_tuning" / "nested_tuning"
REPORT_PATH = (
    Path(REPORTS_DIR) / "nested_operating_point_extension_report.md"
)

CHANNELS = base.NESTED_CHANNELS
H_GRID = base.NESTED_H
DT_GRID = base.NESTED_DT
SOURCE_STRENGTHS = base.NESTED_MULTIPLIERS
BASE_STRENGTHS = SOURCE_STRENGTHS + (32.0, 64.0, 128.0)
MAX_ADAPTIVE_STRENGTH = 2048.0
SHORTLIST_PER_CHANNEL = base.NESTED_SHORTLIST
SCREEN_SEEDS = base.N_SCREEN_SEEDS
SELECTION_SEEDS = base.N_SELECTION_SEEDS
TEST_SEEDS = base.N_TEST_SEEDS
FRESH_TEST_NAMESPACE = 301
MAX_WORKERS = 8

MANIFEST_PATH = RESULT_ROOT / "manifest.json"
SEED_LEDGER_PATH = RESULT_ROOT / "seed_ledger.json"
REUSE_INDEX_PATH = RESULT_ROOT / "reuse_index.json"
SHORTLIST_PATH = RESULT_ROOT / "screen_shortlist.json"
SELECTION_PATH = RESULT_ROOT / "frozen_selection.json"
AGGREGATE_PATH = RESULT_ROOT / "aggregate.json"
SOURCE_SNAPSHOT_DIR = RESULT_ROOT / "source_snapshot"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def json_roundtrip_sha256(value: object) -> str:
    """Hash the canonical payload after applying JSON's key coercions."""
    return sha256_json(json.loads(canonical_json(value)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def git_head() -> str | None:
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


def tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def config_key(config: Sequence[float]) -> tuple[float, float, float]:
    return tuple(map(float, config))


def row_identity(row: dict) -> tuple[str, float, float, float, int]:
    return (
        str(row["method"]),
        float(row["h"]),
        float(row["dt"]),
        float(row["strength_multiplier"]),
        int(row["seed"]),
    )


def row_sort_key(row: dict) -> tuple:
    return row_identity(row)


def common_grid(strengths: Sequence[float]) -> list[tuple[float, float, float]]:
    values = tuple(map(float, strengths))
    if len(values) != len(set(values)) or tuple(sorted(values)) != values:
        raise ValueError("strength grid must be strictly increasing and unique")
    return list(itertools.product(H_GRID, DT_GRID, values))


def source_manifest() -> dict:
    if not (SOURCE_ROOT / "manifest.json").is_file():
        raise RuntimeError("sealed source manifest is missing")
    return json.loads((SOURCE_ROOT / "manifest.json").read_text())


def _current_source_hashes() -> dict[str, str]:
    paths = tuple(base.SCIENTIFIC_SOURCES) + (
        "experiments/run_nested_operating_point_extension.py",
    )
    missing = [name for name in paths if not (REPO_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"scientific source files missing: {missing}")
    return {name: sha256_file(REPO_ROOT / name) for name in paths}


def validate_source_protocol(manifest: dict | None = None) -> dict:
    manifest = source_manifest() if manifest is None else manifest
    if manifest.get("artifact_type") != "revision_tuning_manifest":
        raise RuntimeError("unexpected sealed source manifest type")
    protocol = manifest["protocol"]
    details = protocol["details"]
    expected = {
        "N": base.N,
        "channels": list(CHANNELS),
        "delays": list(base.DELAYS),
        "dt_grid": list(DT_GRID),
        "h_grid": list(H_GRID),
        "ridge_grid": list(base.RIDGES),
        "screen_seeds": SCREEN_SEEDS,
        "selection_seeds": SELECTION_SEEDS,
        "test_seeds": TEST_SEEDS,
        "shortlist_per_channel": SHORTLIST_PER_CHANNEL,
        "strength_multiplier_grid": list(SOURCE_STRENGTHS),
        "screen_split": {
            "wash": base.SCREEN_WASH,
            "train": base.SCREEN_TRAIN,
            "validation": base.SCREEN_VAL,
            "test": 0,
        },
        "split": {
            "wash": base.WASH,
            "train": base.TRAIN,
            "validation": base.VAL,
            "test": base.TEST,
        },
    }
    for key, value in expected.items():
        if details.get(key) != value:
            raise RuntimeError(f"sealed source protocol mismatch for {key}")
    current = _current_source_hashes()
    for path, expected_hash in protocol["scientific_sources_sha256"].items():
        if current.get(path) != expected_hash:
            raise RuntimeError(
                f"scientific source drift prevents row reuse: {path}"
            )
    if sha256_json(protocol) != manifest["protocol_sha256"]:
        raise RuntimeError("sealed source protocol fingerprint is invalid")
    return manifest


def seed_ledger_payload(manifest: dict | None = None) -> dict:
    manifest = validate_source_protocol(manifest)
    pools = {
        name: list(map(int, values))
        for name, values in manifest["protocol"]["seed_namespaces"].items()
    }
    exclusions = sorted({seed for values in pools.values() for seed in values})
    fresh = base._fresh_seed_pool(  # noqa: SLF001
        FRESH_TEST_NAMESPACE, TEST_SEEDS, set(exclusions)
    )
    if set(fresh) & set(exclusions) or len(fresh) != len(set(fresh)):
        raise RuntimeError("fresh test seed generation is not disjoint")
    return {
        "artifact_type": "nested_extension_seed_ledger",
        "source_manifest": str((SOURCE_ROOT / "manifest.json").relative_to(REPO_ROOT)),
        "source_manifest_sha256": sha256_file(SOURCE_ROOT / "manifest.json"),
        "reused_screen_seeds": pools["nested_screen"],
        "reused_selection_seeds": pools["nested_selection"],
        "known_old_test_seeds": pools["nested_test"],
        "excluded_seed_sources": {
            name: {
                "count": len(values),
                "sha256": sha256_json(values),
            }
            for name, values in sorted(pools.items())
        },
        "excluded_seed_count": len(exclusions),
        "excluded_seeds": exclusions,
        "excluded_seeds_sha256": sha256_json(exclusions),
        "fresh_test_namespace": FRESH_TEST_NAMESPACE,
        "fresh_test_seeds": fresh,
        "fresh_test_seeds_sha256": sha256_json(fresh),
        "pairwise_disjoint_verified": True,
    }


def protocol_payload() -> dict:
    source = validate_source_protocol()
    ledger = seed_ledger_payload(source)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "nested_operating_point_extension",
        "git_head_at_protocol": git_head(),
        "source_protocol_sha256": source["protocol_sha256"],
        "source_manifest_sha256": ledger["source_manifest_sha256"],
        "scientific_sources_sha256": _current_source_hashes(),
        "details": {
            "N": base.N,
            "channels": list(CHANNELS),
            "h_grid": list(H_GRID),
            "dt_grid": list(DT_GRID),
            "source_strength_grid": list(SOURCE_STRENGTHS),
            "mandatory_common_strength_grid": list(BASE_STRENGTHS),
            "ridge_grid": list(base.RIDGES),
            "screen_split": {
                "wash": base.SCREEN_WASH,
                "train": base.SCREEN_TRAIN,
                "validation": base.SCREEN_VAL,
                "test": 0,
            },
            "selection_split": {
                "wash": base.WASH,
                "train": base.TRAIN,
                "validation": base.VAL,
                "test": 0,
            },
            "test_split": {
                "wash": base.WASH,
                "train": base.TRAIN,
                "validation": base.VAL,
                "test": base.TEST,
            },
            "screen_seed_count": SCREEN_SEEDS,
            "selection_seed_count": SELECTION_SEEDS,
            "fresh_test_seed_count": TEST_SEEDS,
            "shortlist_per_channel": SHORTLIST_PER_CHANNEL,
            "adaptive_strength_rule": (
                "Start with the common grid through x128. If the collective "
                "screen winner or validation-selected winner is at the upper "
                "edge, append the next doubled strength for both channels and "
                "screen every h x dt cell. Continue through at most x2048. "
                "Before freezing, evaluate the immediate lower and upper "
                "strength neighbors at the selected collective h x dt on all "
                "12 selection reservoirs and require both validation means to "
                "be strictly lower. Fail without test scoring otherwise."
            ),
            "maximum_adaptive_strength": MAX_ADAPTIVE_STRENGTH,
            "selection_rule": (
                "Rank the common cheap screen separately by channel; advance "
                "the top eight ridge-resolved configs per channel; on the "
                "original disjoint 12-reservoir full-length selection pool, "
                "select config and ridge by validation STM only. Calibration-"
                "only immediate-neighbor expansion is permitted solely to "
                "bracket the collective strength choice. Freeze and hash both "
                "choices before scoring the deterministic fresh 24-seed test "
                "pool once."
            ),
            "row_reuse_rule": (
                "Reuse sealed screen/selection rows only when their identity, "
                "split, source protocol hash, and scientific-source hashes "
                "match; record path and SHA-256 for every reused row."
            ),
        },
        "seed_hashes": {
            "excluded_seeds_sha256": ledger["excluded_seeds_sha256"],
            "fresh_test_seeds_sha256": ledger["fresh_test_seeds_sha256"],
        },
    }
    return protocol


def _write_or_compare(path: Path, payload: dict, drift_label: str) -> None:
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f"{drift_label} drift: {path}")
    else:
        atomic_json(path, payload)


def ensure_protocol() -> tuple[dict, str, dict]:
    protocol = protocol_payload()
    protocol_sha = sha256_json(protocol)
    manifest = {
        "artifact_type": "nested_operating_point_extension_manifest",
        "status": "frozen_before_new_rows",
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
    }
    _write_or_compare(MANIFEST_PATH, manifest, "extension manifest")
    ledger = seed_ledger_payload()
    _write_or_compare(SEED_LEDGER_PATH, ledger, "seed ledger")

    SOURCE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SOURCE_SNAPSHOT_DIR / Path(__file__).name
    if snapshot_path.exists():
        if sha256_file(snapshot_path) != sha256_file(Path(__file__)):
            raise RuntimeError("extension source snapshot drift")
    else:
        shutil.copyfile(Path(__file__), snapshot_path)
    snapshot_manifest = {
        "artifact_type": "nested_extension_source_snapshot",
        "protocol_sha256": protocol_sha,
        "path": str(snapshot_path.relative_to(REPO_ROOT)),
        "sha256": sha256_file(snapshot_path),
    }
    _write_or_compare(
        SOURCE_SNAPSHOT_DIR / "manifest.json",
        snapshot_manifest,
        "source snapshot manifest",
    )
    return manifest, protocol_sha, ledger


def job_path(
    stage: str, method: str, config: Sequence[float], seed: int
) -> Path:
    h, dt, strength = config_key(config)
    return (
        RESULT_ROOT
        / f"{stage}_jobs"
        / (
            f"{method}_h{tag(h)}_dt{tag(dt)}_x{tag(strength)}"
            f"_s{seed}.json"
        )
    )


def _source_job_path(
    stage: str, method: str, config: Sequence[float], seed: int
) -> Path:
    h, dt, strength = config_key(config)
    return (
        SOURCE_ROOT
        / f"{stage}_jobs"
        / (
            f"{method}_h{tag(h)}_dt{tag(dt)}_x{tag(strength)}"
            f"_s{seed}.json"
        )
    )


def _validate_reused_row(
    path: Path,
    row: dict,
    stage: str,
    source_protocol_sha: str,
) -> None:
    if row.get("protocol_sha256") != source_protocol_sha:
        raise RuntimeError(f"source row protocol mismatch: {path}")
    if row.get("stage") != stage:
        raise RuntimeError(f"source row stage mismatch: {path}")
    expected_split = (
        {
            "wash": base.SCREEN_WASH,
            "train": base.SCREEN_TRAIN,
            "validation": base.SCREEN_VAL,
            "test": 0,
        }
        if stage == "screen"
        else {
            "wash": base.WASH,
            "train": base.TRAIN,
            "validation": base.VAL,
            "test": 0,
        }
    )
    if row.get("split") != expected_split:
        raise RuntimeError(f"source row split mismatch: {path}")


def rows_for(
    stage: str,
    configs_by_method: dict[str, Sequence[Sequence[float]]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load exact requested rows, preferring sealed compatible source rows."""
    source = validate_source_protocol()
    source_protocol_sha = source["protocol_sha256"]
    ledger = seed_ledger_payload(source)
    seeds = (
        ledger["reused_screen_seeds"]
        if stage == "screen"
        else ledger["reused_selection_seeds"]
    )
    rows: list[dict] = []
    reused: list[dict] = []
    new: list[dict] = []
    missing: list[tuple] = []
    seen: set[tuple] = set()
    for method in CHANNELS:
        for config in configs_by_method[method]:
            config = config_key(config)
            for seed in seeds:
                source_path = _source_job_path(stage, method, config, seed)
                new_path = job_path(stage, method, config, seed)
                if source_path.is_file():
                    row = json.loads(source_path.read_text())
                    _validate_reused_row(
                        source_path, row, stage, source_protocol_sha
                    )
                    provenance = {
                        "stage": stage,
                        "identity": list(row_identity(row)),
                        "path": str(source_path.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(source_path),
                    }
                    reused.append(provenance)
                elif new_path.is_file():
                    row = json.loads(new_path.read_text())
                    provenance = {
                        "stage": stage,
                        "identity": list(row_identity(row)),
                        "path": str(new_path.relative_to(REPO_ROOT)),
                        "sha256": sha256_file(new_path),
                    }
                    new.append(provenance)
                else:
                    missing.append((method, config, seed))
                    continue
                identity = row_identity(row)
                expected_identity = (method, *config, int(seed))
                if identity != expected_identity:
                    raise RuntimeError(
                        f"row identity mismatch: {identity} != {expected_identity}"
                    )
                if identity in seen:
                    raise RuntimeError(f"duplicate row identity: {identity}")
                seen.add(identity)
                rows.append(row)
    if missing:
        raise RuntimeError(f"{len(missing)} requested {stage} rows are missing")
    return (
        sorted(rows, key=row_sort_key),
        sorted(reused, key=lambda item: tuple(item["identity"])),
        sorted(new, key=lambda item: tuple(item["identity"])),
    )


def _run_calibration_jobs(
    stage: str,
    configs_by_method: dict[str, Sequence[Sequence[float]]],
    protocol_sha: str,
    workers: int,
) -> None:
    _, _, ledger = ensure_protocol()
    seeds = (
        ledger["reused_screen_seeds"]
        if stage == "screen"
        else ledger["reused_selection_seeds"]
    )
    jobs = []
    for method in CHANNELS:
        for raw_config in configs_by_method[method]:
            config = config_key(raw_config)
            for seed in seeds:
                if _source_job_path(stage, method, config, seed).is_file():
                    continue
                jobs.append((method, config, seed, stage, protocol_sha))
    base._run_checkpointed(  # noqa: SLF001
        jobs,
        base.nested_calibration_job,
        lambda job: job_path(stage, job[0], job[1], job[2]),
        workers,
        f"nested-extension-{stage}",
    )


def resolved_shortlist(screen_ranking: dict[str, list[dict]]) -> dict[str, list[list[float]]]:
    shortlist: dict[str, list[list[float]]] = {}
    for method in CHANNELS:
        if screen_ranking[method][0]["ridge_upper_boundary_unresolved"]:
            raise RuntimeError(
                f"{method} screen winner hits ridge upper boundary "
                f"{base.RIDGES[-1]}"
            )
        resolved = [
            entry
            for entry in screen_ranking[method]
            if not entry["ridge_upper_boundary_unresolved"]
        ]
        if len(resolved) < SHORTLIST_PER_CHANNEL:
            raise RuntimeError(f"too few ridge-resolved screen configs for {method}")
        shortlist[method] = [
            list(map(float, entry["config"]))
            for entry in resolved[:SHORTLIST_PER_CHANNEL]
        ]
    return shortlist


def immediate_bracket(
    selection_ranking: Sequence[dict],
    strengths: Sequence[float],
) -> dict:
    if not selection_ranking:
        raise ValueError("selection ranking is empty")
    selected = selection_ranking[0]
    h, dt, strength = config_key(selected["config"])
    strengths = tuple(map(float, strengths))
    try:
        index = strengths.index(strength)
    except ValueError as exc:
        raise RuntimeError("selected strength is outside the common grid") from exc
    if index == 0 or index == len(strengths) - 1:
        return {
            "bracketed": False,
            "reason": "selected_strength_at_common_grid_boundary",
            "selected": selected,
            "required_configs": [],
        }
    lower_config = (h, dt, strengths[index - 1])
    upper_config = (h, dt, strengths[index + 1])
    by_config = {
        config_key(entry["config"]): entry for entry in selection_ranking
    }
    missing = [
        list(config)
        for config in (lower_config, upper_config)
        if config not in by_config
    ]
    if missing:
        return {
            "bracketed": False,
            "reason": "immediate_selection_neighbors_missing",
            "selected": selected,
            "required_configs": missing,
        }
    lower = by_config[lower_config]
    upper = by_config[upper_config]
    if any(
        entry["ridge_upper_boundary_unresolved"]
        for entry in (selected, lower, upper)
    ):
        return {
            "bracketed": False,
            "reason": "ridge_upper_boundary_unresolved",
            "selected": selected,
            "lower": lower,
            "upper": upper,
            "required_configs": [],
        }
    bracketed = (
        selected["mean_validation_mc"] > lower["mean_validation_mc"]
        and selected["mean_validation_mc"] > upper["mean_validation_mc"]
    )
    return {
        "bracketed": bool(bracketed),
        "reason": (
            "strict_local_maximum_on_full_selection_ensemble"
            if bracketed
            else "neighbor_validation_mean_not_lower"
        ),
        "selected": selected,
        "lower": lower,
        "upper": upper,
        "required_configs": [],
    }


def _extend_strengths(strengths: list[float]) -> None:
    current = float(strengths[-1])
    proposed = current * 2.0
    if proposed > MAX_ADAPTIVE_STRENGTH:
        raise RuntimeError(
            "collective optimum not bracketed before the predeclared "
            f"x{MAX_ADAPTIVE_STRENGTH:g} safety limit"
        )
    strengths.append(proposed)


def _calibration_state(
    strengths: Sequence[float],
    selection_configs: dict[str, Sequence[Sequence[float]]],
) -> tuple[dict, dict, dict]:
    grid = common_grid(strengths)
    screen_configs = {method: grid for method in CHANNELS}
    screen_rows, screen_reused, screen_new = rows_for("screen", screen_configs)
    screen_ranking = base.rank_nested_configs(
        screen_rows,
        CHANNELS,
        grid,
        seed_ledger_payload()["reused_screen_seeds"],
    )
    selection_rows, selection_reused, selection_new = rows_for(
        "selection", selection_configs
    )
    selection_ranking = {
        method: base.rank_nested_configs(
            selection_rows,
            (method,),
            selection_configs[method],
            seed_ledger_payload()["reused_selection_seeds"],
        )[method]
        for method in CHANNELS
    }
    provenance = {
        "screen_reused": screen_reused,
        "screen_new": screen_new,
        "selection_reused": selection_reused,
        "selection_new": selection_new,
    }
    return screen_ranking, selection_ranking, provenance


def calibrate(workers: int) -> Path:
    _, protocol_sha, ledger = ensure_protocol()
    strengths = list(BASE_STRENGTHS)
    while True:
        grid = common_grid(strengths)
        screen_configs = {method: grid for method in CHANNELS}
        _run_calibration_jobs("screen", screen_configs, protocol_sha, workers)
        screen_rows, _, _ = rows_for("screen", screen_configs)
        screen_ranking = base.rank_nested_configs(
            screen_rows,
            CHANNELS,
            grid,
            ledger["reused_screen_seeds"],
        )
        shortlist = resolved_shortlist(screen_ranking)
        screen_collective_strength = float(
            screen_ranking["B3_collective"][0]["config"][2]
        )
        if math.isclose(screen_collective_strength, strengths[-1]):
            _extend_strengths(strengths)
            continue

        selection_configs: dict[str, list[list[float]]] = {
            method: [list(config) for config in shortlist[method]]
            for method in CHANNELS
        }
        while True:
            _run_calibration_jobs(
                "selection", selection_configs, protocol_sha, workers
            )
            (
                current_screen_ranking,
                selection_ranking,
                provenance,
            ) = _calibration_state(strengths, selection_configs)
            if any(
                selection_ranking[method][0][
                    "ridge_upper_boundary_unresolved"
                ]
                for method in CHANNELS
            ):
                raise RuntimeError(
                    "a validation-selected winner hits the ridge upper boundary"
                )
            bracket = immediate_bracket(
                selection_ranking["B3_collective"], strengths
            )
            if (
                bracket["reason"]
                == "selected_strength_at_common_grid_boundary"
                and math.isclose(
                    float(bracket["selected"]["config"][2]), strengths[-1]
                )
            ):
                _extend_strengths(strengths)
                break
            if bracket["required_configs"]:
                existing = {
                    config_key(config)
                    for config in selection_configs["B3_collective"]
                }
                for config in bracket["required_configs"]:
                    if config_key(config) not in existing:
                        selection_configs["B3_collective"].append(config)
                continue
            if not bracket["bracketed"]:
                # Both neighbours have been ranked.  If either is higher, it
                # must already be the first entry; reaching this branch means a
                # tie or an unresolved ridge boundary, neither is releasable.
                raise RuntimeError(
                    "collective selection is not a strict bracketed maximum: "
                    f"{bracket['reason']}"
                )

            shortlist_payload = {
                "artifact_type": "nested_extension_screen_shortlist",
                "protocol_sha256": protocol_sha,
                "realized_common_strength_grid": strengths,
                "common_grid_identical_for_channels": True,
                "screen_config_count_per_channel": len(
                    common_grid(strengths)
                ),
                "shortlist": shortlist,
                "calibration_candidate_configs": selection_configs,
                "screen_ranking": current_screen_ranking,
            }
            _write_or_compare(
                SHORTLIST_PATH, shortlist_payload, "screen shortlist"
            )

            reuse_payload = {
                "artifact_type": "nested_extension_row_reuse_index",
                "protocol_sha256": protocol_sha,
                "source_protocol_sha256": source_manifest()[
                    "protocol_sha256"
                ],
                "source_manifest_sha256": sha256_file(
                    SOURCE_ROOT / "manifest.json"
                ),
                "counts": {
                    name: len(entries)
                    for name, entries in provenance.items()
                },
                **provenance,
            }
            _write_or_compare(REUSE_INDEX_PATH, reuse_payload, "reuse index")

            chosen = {
                method: selection_ranking[method][0] for method in CHANNELS
            }
            selection_payload = {
                "artifact_type": "frozen_nested_extension_operating_points",
                "status": "frozen_before_fresh_test_ensemble",
                "protocol_sha256": protocol_sha,
                "screen_shortlist_sha256": sha256_file(SHORTLIST_PATH),
                "reuse_index_sha256": sha256_file(REUSE_INDEX_PATH),
                "fresh_test_seeds_sha256": ledger[
                    "fresh_test_seeds_sha256"
                ],
                "realized_common_strength_grid": strengths,
                "chosen": chosen,
                "selection_ranking": selection_ranking,
                "collective_strength_bracket": bracket,
                "test_rows_present_at_freeze": False,
            }
            if SELECTION_PATH.exists():
                if json.loads(SELECTION_PATH.read_text()) != selection_payload:
                    raise RuntimeError("frozen selection drift")
            else:
                test_dir = RESULT_ROOT / "test_jobs"
                if test_dir.exists() and any(test_dir.glob("*.json")):
                    raise RuntimeError(
                        "fresh test rows predate the frozen selection"
                    )
                atomic_json(SELECTION_PATH, selection_payload)
            print(
                "frozen selection "
                f"local={chosen['CD_paper']['config']} "
                f"collective={chosen['B3_collective']['config']} "
                f"selection_sha256={sha256_file(SELECTION_PATH)}",
                flush=True,
            )
            return SELECTION_PATH


def _load_frozen_selection() -> tuple[dict, str, dict, str]:
    manifest, protocol_sha, ledger = ensure_protocol()
    if not SELECTION_PATH.is_file():
        raise RuntimeError("selection is not frozen; run calibrate first")
    frozen = json.loads(SELECTION_PATH.read_text())
    if frozen.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("frozen selection protocol mismatch")
    if not frozen["collective_strength_bracket"]["bracketed"]:
        raise RuntimeError("frozen collective selection is not bracketed")
    return frozen, sha256_file(SELECTION_PATH), ledger, protocol_sha


def run_test(workers: int) -> Path:
    frozen, selection_sha, ledger, protocol_sha = _load_frozen_selection()
    jobs = [
        (
            method,
            frozen["chosen"][method],
            seed,
            protocol_sha,
            selection_sha,
        )
        for method in CHANNELS
        for seed in ledger["fresh_test_seeds"]
    ]
    base._run_checkpointed(  # noqa: SLF001
        jobs,
        base.nested_test_job,
        lambda job: job_path(
            "test", job[0], frozen["chosen"][job[0]]["config"], job[2]
        ),
        workers,
        "nested-extension-fresh-test",
    )
    validate_test_rows()
    return RESULT_ROOT / "test_jobs"


def validate_test_rows() -> list[dict]:
    frozen, selection_sha, ledger, protocol_sha = _load_frozen_selection()
    paths = sorted((RESULT_ROOT / "test_jobs").glob("*.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    expected = {
        (method, int(seed))
        for method in CHANNELS
        for seed in ledger["fresh_test_seeds"]
    }
    identities = {(str(row["method"]), int(row["seed"])) for row in rows}
    if identities != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"fresh test coverage mismatch: {len(rows)}/{len(expected)}"
        )
    for row in rows:
        method = str(row["method"])
        selected = frozen["chosen"][method]
        if row.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("fresh test protocol hash mismatch")
        if row.get("selection_sha256") != selection_sha:
            raise RuntimeError("fresh test selection hash mismatch")
        if config_key(
            (row["h"], row["dt"], row["strength_multiplier"])
        ) != config_key(selected["config"]):
            raise RuntimeError("fresh test row does not use frozen config")
        if not math.isclose(float(row["ridge"]), float(selected["best_ridge"])):
            raise RuntimeError("fresh test row does not use frozen ridge")
    return sorted(rows, key=lambda row: (str(row["method"]), int(row["seed"])))


def _job_provenance(directory: Path) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.glob("*.json"))
    ]


def _runtime_summary(paths: Iterable[Path]) -> dict:
    rows = [json.loads(path.read_text()) for path in sorted(paths)]
    values = [float(row["runtime_s"]) for row in rows]
    return {
        "jobs": len(values),
        "cpu_runtime_s": float(sum(values)),
        "mean_job_runtime_s": float(np.mean(values)) if values else 0.0,
        "max_job_runtime_s": float(max(values)) if values else 0.0,
    }


def build_aggregate() -> Path:
    frozen, selection_sha, ledger, protocol_sha = _load_frozen_selection()
    rows = validate_test_rows()
    reuse = json.loads(REUSE_INDEX_PATH.read_text())
    shortlist = json.loads(SHORTLIST_PATH.read_text())
    strengths = frozen["realized_common_strength_grid"]
    selection_configs = shortlist["calibration_candidate_configs"]
    screen_expected = (
        len(CHANNELS)
        * len(common_grid(strengths))
        * len(ledger["reused_screen_seeds"])
    )
    selection_expected = sum(
        len(selection_configs[method])
        * len(ledger["reused_selection_seeds"])
        for method in CHANNELS
    )
    screen_complete = (
        reuse["counts"]["screen_reused"]
        + reuse["counts"]["screen_new"]
    )
    selection_complete = (
        reuse["counts"]["selection_reused"]
        + reuse["counts"]["selection_new"]
    )
    if screen_complete != screen_expected:
        raise RuntimeError("aggregate screen coverage is incomplete")
    if selection_complete != selection_expected:
        raise RuntimeError("aggregate selection coverage is incomplete")

    scores: dict[str, dict[int, float]] = {}
    methods: dict[str, dict] = {}
    for method in CHANNELS:
        group = [row for row in rows if row["method"] == method]
        ordered = {
            int(row["seed"]): float(row["test_mc"])
            for row in sorted(group, key=lambda item: int(item["seed"]))
        }
        scores[method] = ordered
        mean, se = base._mean_se(list(ordered.values()))  # noqa: SLF001
        methods[method] = {
            "selected": frozen["chosen"][method],
            "fresh_test_mean": mean,
            "fresh_test_se": se,
            "fresh_test_scores_by_seed": ordered,
        }
    common = ledger["fresh_test_seeds"]
    comparison = base.paired_stats(
        [scores["B3_collective"][seed] for seed in common],
        [scores["CD_paper"][seed] for seed in common],
    )
    raw_provenance = {
        "new_screen_rows": _job_provenance(RESULT_ROOT / "screen_jobs"),
        "new_selection_rows": _job_provenance(
            RESULT_ROOT / "selection_jobs"
        ),
        "fresh_test_rows": _job_provenance(RESULT_ROOT / "test_jobs"),
    }
    runtime = {
        "new_screen": _runtime_summary(
            (RESULT_ROOT / "screen_jobs").glob("*.json")
        ),
        "new_selection": _runtime_summary(
            (RESULT_ROOT / "selection_jobs").glob("*.json")
        ),
        "fresh_test": _runtime_summary(
            (RESULT_ROOT / "test_jobs").glob("*.json")
        ),
    }
    runtime["new_compute_cpu_runtime_s"] = float(
        sum(value["cpu_runtime_s"] for value in runtime.values())
    )
    payload = {
        "artifact_type": "nested_operating_point_extension_results",
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "seed_ledger_sha256": sha256_file(SEED_LEDGER_PATH),
        "reuse_index_sha256": sha256_file(REUSE_INDEX_PATH),
        "screen_shortlist_sha256": sha256_file(SHORTLIST_PATH),
        "realized_common_strength_grid": strengths,
        "common_grid_identical_for_channels": True,
        "collective_strength_bracket": frozen[
            "collective_strength_bracket"
        ],
        "seed_disjointness_verified": True,
        "freeze_before_test_verified": True,
        "selected_ridge_upper_boundary_hits": sum(
            bool(
                frozen["chosen"][method][
                    "ridge_upper_boundary_unresolved"
                ]
            )
            for method in CHANNELS
        ),
        "coverage": {
            "screen_expected": screen_expected,
            "screen_complete": screen_complete,
            "screen_reused": reuse["counts"]["screen_reused"],
            "screen_new": reuse["counts"]["screen_new"],
            "selection_expected": selection_expected,
            "selection_complete": selection_complete,
            "selection_reused": reuse["counts"]["selection_reused"],
            "selection_new": reuse["counts"]["selection_new"],
            "fresh_test_expected": len(CHANNELS) * len(common),
            "fresh_test_complete": len(rows),
        },
        "methods": methods,
        "collective_vs_local": comparison,
        "runtime": runtime,
        "raw_provenance": raw_provenance,
    }
    payload["deterministic_payload_sha256"] = json_roundtrip_sha256(payload)
    atomic_json(AGGREGATE_PATH, payload)
    build_report(payload)
    return AGGREGATE_PATH


def build_report(aggregate: dict | None = None) -> Path:
    aggregate = (
        json.loads(AGGREGATE_PATH.read_text())
        if aggregate is None
        else aggregate
    )
    local = aggregate["methods"]["CD_paper"]["selected"]
    collective = aggregate["methods"]["B3_collective"]["selected"]
    bracket = aggregate["collective_strength_bracket"]
    effect = aggregate["collective_vs_local"]
    runtime = aggregate["runtime"]
    text = "\n".join(
        [
            "# Nested operating-point extension",
            "",
            "## Protocol and completion",
            "",
            (
                "The sealed original screen and selection pools were reused "
                "only after byte-level source/protocol verification. The "
                "common local/collective screen was extended through "
                + ", ".join(
                    f"x{value:g}"
                    for value in aggregate["realized_common_strength_grid"]
                )
                + ". Both choices were frozen and hashed before a deterministic "
                "fresh 24-seed paired test ensemble was scored once."
            ),
            "",
            (
                f"- Screen coverage: {aggregate['coverage']['screen_complete']}/"
                f"{aggregate['coverage']['screen_expected']} "
                f"({aggregate['coverage']['screen_reused']} reused, "
                f"{aggregate['coverage']['screen_new']} new)."
            ),
            (
                f"- Selection coverage: "
                f"{aggregate['coverage']['selection_complete']}/"
                f"{aggregate['coverage']['selection_expected']} "
                f"({aggregate['coverage']['selection_reused']} reused, "
                f"{aggregate['coverage']['selection_new']} new)."
            ),
            (
                f"- Fresh test coverage: "
                f"{aggregate['coverage']['fresh_test_complete']}/"
                f"{aggregate['coverage']['fresh_test_expected']}."
            ),
            (
                f"- Protocol SHA-256: `{aggregate['protocol_sha256']}`; "
                f"selection SHA-256: `{aggregate['selection_sha256']}`."
            ),
            "",
            "## Frozen choices",
            "",
            (
                "- Uniform local: "
                f"$h={local['config'][0]:g}$, "
                f"$\\Delta t={local['config'][1]:g}$, "
                f"strength x{local['config'][2]:g}, "
                f"ridge {local['best_ridge']:.12g}; "
                f"selection MC {local['mean_validation_mc']:.6f}."
            ),
            (
                "- Collective: "
                f"$h={collective['config'][0]:g}$, "
                f"$\\Delta t={collective['config'][1]:g}$, "
                f"strength x{collective['config'][2]:g}, "
                f"ridge {collective['best_ridge']:.12g}; "
                f"selection MC {collective['mean_validation_mc']:.6f}."
            ),
            (
                "- Collective bracket at the same $(h,\\Delta t)$: "
                f"x{bracket['lower']['config'][2]:g} "
                f"({bracket['lower']['mean_validation_mc']:.6f}), "
                f"x{bracket['selected']['config'][2]:g} "
                f"({bracket['selected']['mean_validation_mc']:.6f}), "
                f"x{bracket['upper']['config'][2]:g} "
                f"({bracket['upper']['mean_validation_mc']:.6f})."
            ),
            "",
            "## Fresh paired test",
            "",
            (
                f"Collective minus local on the 24 fresh reservoirs: "
                f"$\\Delta C_{{\\mathrm{{STM}}}}={effect['mean_difference']:.6f}$ "
                f"(95% paired $t$ interval "
                f"[{effect['ci95_low']:.6f}, {effect['ci95_high']:.6f}]), "
                f"{effect['relative_mean_difference_percent']:.2f}%, "
                f"{effect['wins']}/{effect['n']} wins, exact two-sided sign "
                f"$p={effect['exact_sign_p_two_sided']:.6g}$."
            ),
            "",
            "## New computation",
            "",
            (
                f"- Screen: {runtime['new_screen']['jobs']} jobs, "
                f"{runtime['new_screen']['cpu_runtime_s']:.1f} CPU-s."
            ),
            (
                f"- Selection: {runtime['new_selection']['jobs']} jobs, "
                f"{runtime['new_selection']['cpu_runtime_s']:.1f} CPU-s."
            ),
            (
                f"- Fresh test: {runtime['fresh_test']['jobs']} jobs, "
                f"{runtime['fresh_test']['cpu_runtime_s']:.1f} CPU-s."
            ),
            (
                f"- Total new row runtime: "
                f"{runtime['new_compute_cpu_runtime_s']:.1f} CPU-s."
            ),
            "",
            "## Claim boundary",
            "",
            (
                "This closes the upper-strength boundary in the declared "
                "common grid and freshens the paired test ensemble. It remains "
                "a finite-grid, equal-Frobenius, uniform-local versus equal-"
                "coefficient-collective comparison; it is not unrestricted "
                "profile or hardware-cost optimization."
            ),
            "",
        ]
    )
    atomic_text(REPORT_PATH, text)
    return REPORT_PATH


def validate_complete_artifacts() -> dict:
    if not AGGREGATE_PATH.is_file():
        raise RuntimeError("aggregate is missing")
    payload = json.loads(AGGREGATE_PATH.read_text())
    stored_hash = payload.pop("deterministic_payload_sha256")
    if sha256_json(payload) != stored_hash:
        raise RuntimeError("deterministic aggregate payload hash mismatch")
    if payload["coverage"]["screen_complete"] != payload["coverage"][
        "screen_expected"
    ]:
        raise RuntimeError("screen coverage incomplete")
    if payload["coverage"]["selection_complete"] != payload["coverage"][
        "selection_expected"
    ]:
        raise RuntimeError("selection coverage incomplete")
    if payload["coverage"]["fresh_test_complete"] != payload["coverage"][
        "fresh_test_expected"
    ]:
        raise RuntimeError("fresh test coverage incomplete")
    if not payload["collective_strength_bracket"]["bracketed"]:
        raise RuntimeError("collective strength is not bracketed")
    if payload["selected_ridge_upper_boundary_hits"] != 0:
        raise RuntimeError("selected ridge boundary is unresolved")
    if not REPORT_PATH.is_file():
        raise RuntimeError("extension report is missing")
    validate_test_rows()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("calibrate", "test", "aggregate", "validate", "all"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be between 1 and {MAX_WORKERS}")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if args.command in ("calibrate", "all"):
        print(f"SELECTION {calibrate(args.workers)}", flush=True)
    if args.command in ("test", "all"):
        print(f"TEST {run_test(args.workers)}", flush=True)
    if args.command in ("aggregate", "all"):
        print(f"AGGREGATE {build_aggregate()}", flush=True)
        print(f"REPORT {REPORT_PATH}", flush=True)
    if args.command == "validate":
        validate_complete_artifacts()
        print(f"VALID {AGGREGATE_PATH}", flush=True)
    print(f"WALL_RUNTIME_S {time.perf_counter() - started:.3f}", flush=True)


if __name__ == "__main__":
    main()

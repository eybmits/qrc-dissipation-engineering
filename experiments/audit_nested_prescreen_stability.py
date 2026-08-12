"""Descriptive stability audit for the two-seed nested-search prescreen.

This companion is intentionally post hoc and read only with respect to the
frozen nested operating-point extension.  It loads the realized common grid
only after ``frozen_selection.json`` exists, verifies and combines compatible
sealed/reused and extension screen rows, and asks how similarly the two cheap
screen reservoirs rank the full grid.

For every configuration and screen seed, the ridge maximizing that seed's
validation memory capacity is chosen first.  Configurations are then ranked
within the seed.  The resulting artifact records the frozen selected
configuration's rank, top-eight membership, the seedwise winner, top-eight
overlap, and the Spearman correlation of the two complete ordinal rankings.
This is descriptive sensitivity evidence; it does not repair or enlarge the
two-seed prescreen.

Run from the repository root:

    PYTHONPATH=src:experiments python \
      experiments/audit_nested_prescreen_stability.py build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import _paths  # noqa: F401

from _paths import REPORTS_DIR  # noqa: E402
import run_nested_operating_point_extension as extension  # noqa: E402


ARTIFACT_VERSION = "nested-prescreen-stability-v1-2026-07-24"
TOP_K = 8
RESULT_PATH = extension.RESULT_ROOT / "prescreen_stability.json"
REPORT_PATH = Path(REPORTS_DIR) / "nested_prescreen_stability_audit.md"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _config(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"expected (h, dt, strength), received {value}")
    config = tuple(map(float, value))
    if not all(math.isfinite(item) for item in config):
        raise ValueError(f"non-finite configuration: {value}")
    return config


def _ridge_tag(value: float) -> str:
    return f"{float(value):.12g}"


def _rank_sort_key(entry: Mapping[str, object]) -> tuple[float, ...]:
    h, dt, strength = _config(entry["config"])  # type: ignore[arg-type]
    return (
        -float(entry["validation_mc"]),
        strength,
        dt,
        h,
        float(entry["best_ridge"]),
    )


def _config_list(configs: set[tuple[float, float, float]]) -> list[list[float]]:
    return [list(config) for config in sorted(configs)]


def _ordinal_spearman(
    first: Mapping[tuple[float, float, float], int],
    second: Mapping[tuple[float, float, float], int],
) -> float:
    """Return Spearman rho for two deterministic complete ordinal rankings."""
    if set(first) != set(second):
        raise ValueError("rank maps do not cover the same configurations")
    n = len(first)
    if n == 0:
        raise ValueError("rank maps are empty")
    if n == 1:
        return 1.0
    squared_difference = sum(
        (int(first[config]) - int(second[config])) ** 2 for config in first
    )
    return float(1.0 - 6.0 * squared_difference / (n * (n * n - 1)))


def rank_seedwise(
    rows: Sequence[dict],
    *,
    methods: Sequence[str],
    configs: Sequence[Sequence[float]],
    seeds: Sequence[int],
    ridges: Sequence[float],
) -> dict[str, dict[int, list[dict]]]:
    """Choose ridge per row, then rank every config separately within each seed."""
    method_values = tuple(map(str, methods))
    seed_values = tuple(map(int, seeds))
    ridge_values = tuple(map(float, ridges))
    config_values = tuple(sorted({_config(config) for config in configs}))
    if len(seed_values) != 2 or len(set(seed_values)) != 2:
        raise ValueError("the prescreen stability audit requires two unique seeds")
    if len(config_values) != len(configs):
        raise ValueError("configuration grid contains duplicates")
    if not ridge_values or len(set(ridge_values)) != len(ridge_values):
        raise ValueError("ridge grid must be nonempty and unique")

    expected_ridge_tags = {_ridge_tag(ridge) for ridge in ridge_values}
    by_identity: dict[tuple[str, int, tuple[float, float, float]], dict] = {}
    for row in rows:
        identity = (
            str(row["method"]),
            int(row["seed"]),
            _config(
                (
                    row["h"],
                    row["dt"],
                    row["strength_multiplier"],
                )
            ),
        )
        if identity in by_identity:
            raise ValueError(f"duplicate screen row: {identity}")
        by_identity[identity] = row

    expected = {
        (method, seed, config)
        for method in method_values
        for seed in seed_values
        for config in config_values
    }
    if set(by_identity) != expected:
        missing = expected - set(by_identity)
        extra = set(by_identity) - expected
        raise ValueError(
            "screen row coverage mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    result: dict[str, dict[int, list[dict]]] = {}
    for method in method_values:
        result[method] = {}
        for seed in seed_values:
            entries: list[dict] = []
            for config in config_values:
                row = by_identity[(method, seed, config)]
                scores = row.get("ridge_validation_mc")
                if not isinstance(scores, dict) or set(scores) != expected_ridge_tags:
                    raise ValueError(
                        f"ridge-score coverage mismatch for {(method, seed, config)}"
                    )
                numeric_scores = {
                    tag: float(value) for tag, value in scores.items()
                }
                if not all(math.isfinite(value) for value in numeric_scores.values()):
                    raise ValueError(
                        f"non-finite ridge score for {(method, seed, config)}"
                    )
                # ``max`` retains the first ridge in the declared grid on ties,
                # matching the frozen nested-search ranking convention.
                best_ridge = max(
                    ridge_values,
                    key=lambda ridge: numeric_scores[_ridge_tag(ridge)],
                )
                entries.append(
                    {
                        "config": list(config),
                        "best_ridge": best_ridge,
                        "validation_mc": numeric_scores[_ridge_tag(best_ridge)],
                    }
                )
            entries.sort(key=_rank_sort_key)
            for rank, entry in enumerate(entries, 1):
                entry["rank"] = rank
            result[method][seed] = entries
    return result


def build_stability_payload(
    rows: Sequence[dict],
    *,
    methods: Sequence[str],
    configs: Sequence[Sequence[float]],
    seeds: Sequence[int],
    ridges: Sequence[float],
    frozen_selected: Mapping[str, Sequence[float]],
    provenance: Mapping[str, object],
    top_k: int = TOP_K,
) -> dict:
    """Build the deterministic audit payload from already validated rows."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    method_values = tuple(map(str, methods))
    seed_values = tuple(map(int, seeds))
    config_values = tuple(sorted({_config(config) for config in configs}))
    ranked = rank_seedwise(
        rows,
        methods=method_values,
        configs=config_values,
        seeds=seed_values,
        ridges=ridges,
    )
    used_top_k = min(int(top_k), len(config_values))

    method_payloads: dict[str, dict] = {}
    for method in method_values:
        selected_config = _config(frozen_selected[method])
        if selected_config not in set(config_values):
            raise ValueError(
                f"frozen selected config is outside screen grid: "
                f"{method} {selected_config}"
            )
        per_seed: dict[str, dict] = {}
        rank_maps: list[dict[tuple[float, float, float], int]] = []
        top_sets: list[set[tuple[float, float, float]]] = []
        for seed in seed_values:
            entries = ranked[method][seed]
            by_config = {_config(entry["config"]): entry for entry in entries}
            selected_entry = by_config[selected_config]
            top_entries = entries[:used_top_k]
            top_set = {_config(entry["config"]) for entry in top_entries}
            top_sets.append(top_set)
            rank_maps.append(
                {
                    _config(entry["config"]): int(entry["rank"])
                    for entry in entries
                }
            )
            per_seed[str(seed)] = {
                "frozen_selected_config_rank": int(selected_entry["rank"]),
                "frozen_selected_config_in_top8": selected_config in top_set,
                "frozen_selected_config_screen_best_ridge": float(
                    selected_entry["best_ridge"]
                ),
                "frozen_selected_config_validation_mc": float(
                    selected_entry["validation_mc"]
                ),
                "winner": {
                    "config": list(_config(entries[0]["config"])),
                    "best_ridge": float(entries[0]["best_ridge"]),
                    "validation_mc": float(entries[0]["validation_mc"]),
                },
                "top8": [
                    {
                        "rank": int(entry["rank"]),
                        "config": list(_config(entry["config"])),
                        "best_ridge": float(entry["best_ridge"]),
                        "validation_mc": float(entry["validation_mc"]),
                    }
                    for entry in top_entries
                ],
            }

        intersection = top_sets[0] & top_sets[1]
        union = top_sets[0] | top_sets[1]
        method_payloads[method] = {
            "frozen_selected_config": list(selected_config),
            "per_seed": per_seed,
            "top8_overlap": {
                "intersection_count": len(intersection),
                "union_count": len(union),
                "jaccard": float(len(intersection) / len(union)),
                "intersection_configs": _config_list(intersection),
                "union_configs": _config_list(union),
            },
            "full_rank_spearman": _ordinal_spearman(
                rank_maps[0], rank_maps[1]
            ),
        }

    payload = {
        "artifact_type": "nested_prescreen_stability_audit",
        "artifact_version": ARTIFACT_VERSION,
        "status": "complete",
        "analysis_type": "post_hoc_descriptive_sensitivity_audit",
        "claim_boundary": (
            "This describes agreement between the two realized cheap-screen "
            "reservoirs. It does not remedy the two-seed prescreen limitation, "
            "provide selection-adjusted inference, or enlarge the calibration "
            "ensemble."
        ),
        "ridge_rule": (
            "Choose the validation-MC-maximizing ridge independently for each "
            "configuration and screen seed; ties follow declared ridge-grid order."
        ),
        "rank_rule": (
            "Rank within each method and seed by descending validation MC, then "
            "ascending strength, dt, h, and ridge."
        ),
        "screen_seed_count": len(seed_values),
        "screen_seeds": list(seed_values),
        "configuration_count_per_method": len(config_values),
        "top_k": used_top_k,
        "provenance": dict(provenance),
        "methods": method_payloads,
    }
    payload["deterministic_payload_sha256"] = sha256_json(payload)
    return payload


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object: {path}")
    return payload


def _validate_manifest_payload(manifest: dict) -> None:
    if manifest.get("artifact_type") != (
        "nested_operating_point_extension_manifest"
    ):
        raise RuntimeError("unexpected extension manifest type")
    if extension.sha256_json(manifest["protocol"]) != manifest.get(
        "protocol_sha256"
    ):
        raise RuntimeError("extension manifest protocol hash mismatch")
    for relative, expected_hash in manifest["protocol"][
        "scientific_sources_sha256"
    ].items():
        path = extension.REPO_ROOT / relative
        if not path.is_file() or extension.sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"scientific source drift prevents audit: {relative}"
            )


def _validate_screen_row(
    row: dict,
    *,
    expected_identity: tuple[str, int, tuple[float, float, float]],
    expected_protocol_sha: str,
    expected_split: dict,
    expected_ridge_tags: set[str],
) -> None:
    identity = (
        str(row.get("method")),
        int(row.get("seed")),
        _config(
            (
                row.get("h"),
                row.get("dt"),
                row.get("strength_multiplier"),
            )
        ),
    )
    if identity != expected_identity:
        raise RuntimeError(f"screen row identity mismatch: {identity}")
    if row.get("protocol_sha256") != expected_protocol_sha:
        raise RuntimeError(f"screen row protocol mismatch: {identity}")
    if row.get("stage") != "screen" or row.get("split") != expected_split:
        raise RuntimeError(f"screen row split/stage mismatch: {identity}")
    scores = row.get("ridge_validation_mc")
    if not isinstance(scores, dict) or set(scores) != expected_ridge_tags:
        raise RuntimeError(f"screen row ridge grid mismatch: {identity}")


def load_frozen_screen_inputs() -> tuple[
    list[dict],
    tuple[str, ...],
    list[tuple[float, float, float]],
    tuple[int, ...],
    tuple[float, ...],
    dict[str, tuple[float, float, float]],
    dict,
]:
    """Load and verify all inputs without mutating the frozen experiment."""
    manifest = _load_json(extension.MANIFEST_PATH, "extension manifest")
    _validate_manifest_payload(manifest)
    protocol_sha = str(manifest["protocol_sha256"])
    details = manifest["protocol"]["details"]

    frozen = _load_json(extension.SELECTION_PATH, "frozen selection")
    if frozen.get("artifact_type") != (
        "frozen_nested_extension_operating_points"
    ):
        raise RuntimeError("unexpected frozen selection type")
    if frozen.get("status") != "frozen_before_fresh_test_ensemble":
        raise RuntimeError("frozen selection has unexpected status")
    if frozen.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("frozen selection protocol mismatch")

    shortlist = _load_json(extension.SHORTLIST_PATH, "screen shortlist")
    if extension.sha256_file(extension.SHORTLIST_PATH) != frozen.get(
        "screen_shortlist_sha256"
    ):
        raise RuntimeError("screen shortlist hash mismatch")
    strengths = tuple(map(float, frozen["realized_common_strength_grid"]))
    if list(strengths) != shortlist.get("realized_common_strength_grid"):
        raise RuntimeError("realized strength grid differs from shortlist")
    configs = extension.common_grid(strengths)

    source_manifest = _load_json(
        extension.SOURCE_ROOT / "manifest.json", "sealed source manifest"
    )
    if extension.sha256_file(extension.SOURCE_ROOT / "manifest.json") != (
        manifest["protocol"]["source_manifest_sha256"]
    ):
        raise RuntimeError("sealed source manifest hash mismatch")
    source_protocol_sha = str(source_manifest["protocol_sha256"])
    if source_protocol_sha != manifest["protocol"]["source_protocol_sha256"]:
        raise RuntimeError("sealed source protocol mismatch")

    ledger = _load_json(extension.SEED_LEDGER_PATH, "seed ledger")
    seeds = tuple(map(int, ledger["reused_screen_seeds"]))
    if len(seeds) != int(details["screen_seed_count"]):
        raise RuntimeError("screen seed count mismatch")
    methods = tuple(map(str, details["channels"]))
    ridges = tuple(map(float, details["ridge_grid"]))
    ridge_tags = {_ridge_tag(ridge) for ridge in ridges}
    expected_split = dict(details["screen_split"])

    rows: list[dict] = []
    provenance: list[dict] = []
    reused_count = 0
    new_count = 0
    for method in methods:
        for config in configs:
            config = _config(config)
            for seed in seeds:
                source_path = extension._source_job_path(  # noqa: SLF001
                    "screen", method, config, seed
                )
                new_path = extension.job_path("screen", method, config, seed)
                if source_path.is_file():
                    path = source_path
                    row_protocol_sha = source_protocol_sha
                    source_kind = "sealed_reused"
                    reused_count += 1
                elif new_path.is_file():
                    path = new_path
                    row_protocol_sha = protocol_sha
                    source_kind = "extension_new"
                    new_count += 1
                else:
                    raise RuntimeError(
                        "screen row missing after freeze: "
                        f"{method} {config} seed={seed}"
                    )
                row = _load_json(path, "screen row")
                expected_identity = (method, seed, config)
                _validate_screen_row(
                    row,
                    expected_identity=expected_identity,
                    expected_protocol_sha=row_protocol_sha,
                    expected_split=expected_split,
                    expected_ridge_tags=ridge_tags,
                )
                rows.append(row)
                provenance.append(
                    {
                        "source": source_kind,
                        "path": str(path.relative_to(extension.REPO_ROOT)),
                        "sha256": extension.sha256_file(path),
                    }
                )

    expected_rows = len(methods) * len(configs) * len(seeds)
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"screen coverage mismatch: {len(rows)}/{expected_rows}"
        )
    provenance.sort(key=lambda item: (item["source"], item["path"]))
    frozen_selected = {
        method: _config(frozen["chosen"][method]["config"])
        for method in methods
    }
    provenance_summary = {
        "protocol_sha256": protocol_sha,
        "frozen_selection_sha256": extension.sha256_file(
            extension.SELECTION_PATH
        ),
        "screen_shortlist_sha256": extension.sha256_file(
            extension.SHORTLIST_PATH
        ),
        "source_manifest_sha256": extension.sha256_file(
            extension.SOURCE_ROOT / "manifest.json"
        ),
        "screen_rows_expected": expected_rows,
        "screen_rows_loaded": len(rows),
        "sealed_reused_rows": reused_count,
        "extension_new_rows": new_count,
        "screen_rows_sha256": sha256_json(
            sorted(
                rows,
                key=lambda row: (
                    str(row["method"]),
                    int(row["seed"]),
                    float(row["h"]),
                    float(row["dt"]),
                    float(row["strength_multiplier"]),
                ),
            )
        ),
        "row_provenance_sha256": sha256_json(provenance),
    }
    return (
        rows,
        methods,
        configs,
        seeds,
        ridges,
        frozen_selected,
        provenance_summary,
    )


def recompute_payload() -> dict:
    (
        rows,
        methods,
        configs,
        seeds,
        ridges,
        frozen_selected,
        provenance,
    ) = load_frozen_screen_inputs()
    return build_stability_payload(
        rows,
        methods=methods,
        configs=configs,
        seeds=seeds,
        ridges=ridges,
        frozen_selected=frozen_selected,
        provenance=provenance,
    )


def render_report(payload: dict) -> str:
    seeds = [str(seed) for seed in payload["screen_seeds"]]
    lines = [
        "# Nested prescreen stability audit",
        "",
        (
            "This is a post-hoc descriptive sensitivity audit of the two "
            "realized cheap-screen reservoirs. It does not remedy the two-seed "
            "prescreen limitation, enlarge calibration, or supply "
            "selection-adjusted inference."
        ),
        "",
        (
            f"The complete realized common grid contains "
            f"{payload['configuration_count_per_method']} configurations per "
            "method. Within each seed and configuration, the validation-best "
            "ridge was chosen before configs were ranked."
        ),
        "",
        "| Method | Frozen config ranks | In each top 8 | Top-8 overlap (I/U) | Jaccard | Full-rank Spearman |",
        "|---|---:|:---:|---:|---:|---:|",
    ]
    for method in sorted(payload["methods"]):
        result = payload["methods"][method]
        ranks = "/".join(
            str(
                result["per_seed"][seed][
                    "frozen_selected_config_rank"
                ]
            )
            for seed in seeds
        )
        in_top = "/".join(
            "yes"
            if result["per_seed"][seed][
                "frozen_selected_config_in_top8"
            ]
            else "no"
            for seed in seeds
        )
        overlap = result["top8_overlap"]
        lines.append(
            f"| `{method}` | {ranks} | {in_top} | "
            f"{overlap['intersection_count']}/{overlap['union_count']} | "
            f"{overlap['jaccard']:.3f} | "
            f"{result['full_rank_spearman']:.3f} |"
        )
    lines.extend(["", "Seedwise winners:", ""])
    for method in sorted(payload["methods"]):
        result = payload["methods"][method]
        for seed in seeds:
            winner = result["per_seed"][seed]["winner"]
            h, dt, strength = winner["config"]
            lines.append(
                f"- `{method}`, seed {seed}: "
                f"$(h,\\Delta t,\\mathrm{{strength}})=({h:g},{dt:g},"
                f"{strength:g})$, ridge {winner['best_ridge']:.12g}, "
                f"validation MC {winner['validation_mc']:.6f}."
            )
    lines.extend(
        [
            "",
            "Claim boundary: agreement or disagreement here characterizes only "
            "the realized two-seed screening stage. The frozen operating points "
            "were selected later on the disjoint 12-reservoir calibration pool.",
            "",
            (
                f"Deterministic payload SHA-256: "
                f"`{payload['deterministic_payload_sha256']}`."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_artifacts() -> tuple[Path, Path]:
    payload = recompute_payload()
    extension.atomic_json(RESULT_PATH, payload)
    _atomic_text(REPORT_PATH, render_report(payload))
    return RESULT_PATH, REPORT_PATH


def validate_artifacts() -> dict:
    stored = _load_json(RESULT_PATH, "prescreen stability audit")
    stored_without_hash = dict(stored)
    stored_hash = stored_without_hash.pop("deterministic_payload_sha256", None)
    if stored_hash != sha256_json(stored_without_hash):
        raise RuntimeError("prescreen audit payload hash mismatch")
    expected = recompute_payload()
    if stored != expected:
        raise RuntimeError("prescreen audit differs from deterministic recompute")
    if not REPORT_PATH.is_file() or REPORT_PATH.read_text() != render_report(stored):
        raise RuntimeError("prescreen audit report is missing or stale")
    return stored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        result_path, report_path = write_artifacts()
        print(f"AUDIT {result_path}")
        print(f"REPORT {report_path}")
    else:
        validate_artifacts()
        print(f"VALID {RESULT_PATH}")


if __name__ == "__main__":
    main()

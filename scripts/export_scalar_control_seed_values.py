#!/usr/bin/env python3
"""Export the Figure 6 seed-level values from their sealed evidence."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "paper" / "data" / "experiment1_robustness_snapshot.json"
ACTIVITY = ROOT / "paper" / "data" / "activity_matched_confirmation.json"
GAP = (
    ROOT
    / "paper"
    / "evidence"
    / "switched_input_memory_control_v2"
    / "aggregate.json"
)
SELECTION = (
    ROOT
    / "results"
    / "revision_tuning"
    / "nested_operating_point_extension"
    / "aggregate.json"
)
FIXED_ARCHIVE = ROOT / "results" / "final_protocol_results.tar.gz"
REVIEW_ARCHIVE = ROOT / "results" / "review_protocol_results.tar.gz"
OUTPUT = (
    ROOT / "paper" / "data" / "experiment1_scalar_control_seed_values.json"
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_fixed_values() -> list[dict]:
    values = {"CD_paper": {}, "B3_collective": {}}
    with tarfile.open(FIXED_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"could not read {member.name}")
            row = json.load(stream)
            method = row.get("method")
            if (
                row.get("block") == "A_table"
                and row.get("task") == "stm"
                and row.get("N") == 5
                and method in values
                and row.get("value") is not None
            ):
                seed = int(row["seed"])
                if seed in values[method]:
                    raise RuntimeError(f"duplicate fixed-B row: {method}/{seed}")
                values[method][seed] = float(row["value"])
    seeds = sorted(set(values["CD_paper"]) & set(values["B3_collective"]))
    incomplete_method = any(
        len(method_values) != 32 for method_values in values.values()
    )
    if len(seeds) != 32 or incomplete_method:
        raise RuntimeError("fixed-B seed block is not a complete paired 32")
    return [
        {
            "seed": seed,
            "value": values["B3_collective"][seed] - values["CD_paper"][seed],
        }
        for seed in seeds
    ]


def paired_hamiltonian_values() -> dict[str, list[dict]]:
    ensembles = ("zz_x_z", "xy_z_x", "xx_ring")
    values = {
        ensemble: {"CD_paper": {}, "B3_collective": {}}
        for ensemble in ensembles
    }
    with tarfile.open(REVIEW_ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"could not read {member.name}")
            row = json.load(stream)
            ensemble = row.get("ensemble")
            method = row.get("method")
            if (
                row.get("block") == "R_ham"
                and row.get("task") == "stm"
                and row.get("N") == 5
                and ensemble in values
                and method in values[ensemble]
                and row.get("value") is not None
            ):
                seed = int(row["seed"])
                if seed in values[ensemble][method]:
                    raise RuntimeError(
                        f"duplicate Hamiltonian row: {ensemble}/{method}/{seed}"
                    )
                values[ensemble][method][seed] = float(row["value"])

    paired = {}
    for ensemble in ensembles:
        seeds = sorted(
            set(values[ensemble]["CD_paper"])
            & set(values[ensemble]["B3_collective"])
        )
        incomplete_method = any(
            len(method_values) != 32
            for method_values in values[ensemble].values()
        )
        if len(seeds) != 32 or incomplete_method:
            raise RuntimeError(
                f"Hamiltonian seed block is incomplete: {ensemble}"
            )
        paired[ensemble] = [
            {
                "seed": seed,
                "value": values[ensemble]["B3_collective"][seed]
                - values[ensemble]["CD_paper"][seed],
            }
            for seed in seeds
        ]
    return paired


def build_payload() -> dict:
    robustness = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    gap = json.loads(GAP.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    provenance = robustness["provenance"]
    expected_hashes = {
        "final_protocol_archive_sha256": file_digest(FIXED_ARCHIVE),
        "review_protocol_archive_sha256": file_digest(REVIEW_ARCHIVE),
        "activity_snapshot_sha256": file_digest(ACTIVITY),
        "forgetting_aggregate_sha256": file_digest(GAP),
        "independent_selection_aggregate_sha256": file_digest(SELECTION),
    }
    if any(provenance.get(key) != value for key, value in expected_hashes.items()):
        raise RuntimeError("scalar-control provenance hash mismatch")

    activity_rows = sorted(activity["rows"], key=lambda row: int(row["seed"]))
    gap_input = gap["input_response"]
    long_washout = gap["long_washout_stm"]
    selection_scores = {
        method: {
            int(seed): float(value)
            for seed, value in selection["methods"][method][
                "fresh_test_scores_by_seed"
            ].items()
        }
        for method in ("CD_paper", "B3_collective")
    }
    selection_seeds = sorted(
        set(selection_scores["CD_paper"])
        & set(selection_scores["B3_collective"])
    )
    selection_values = [
        selection_scores["B3_collective"][seed]
        - selection_scores["CD_paper"][seed]
        for seed in selection_seeds
    ]
    if not np.allclose(
        sorted(selection_values),
        sorted(selection["collective_vs_local"]["paired_differences"]),
        rtol=0,
        atol=5e-12,
    ):
        raise RuntimeError("independent-selection seed pairing mismatch")
    rows = {
        "fixed_b": paired_fixed_values(),
        "washout_800": [
            {"seed": int(seed), "value": float(value)}
            for seed, value in zip(
                long_washout["seeds"],
                long_washout["washouts"]["800"][
                    "collective_minus_local_ground_initialization"
                ]["values"],
                strict=True,
            )
        ],
        "activity_matched": [
            {
                "seed": int(row["seed"]),
                "value": float(row["collective_stm"]) - float(row["local_stm"]),
            }
            for row in activity_rows
        ],
        "gap_matched": [
            {"seed": int(seed), "value": float(value)}
            for seed, value in zip(
                gap_input["seeds"],
                gap_input["paired_stm_difference"],
                strict=True,
            )
        ],
        "independent_selection": [
            {"seed": seed, "value": float(value)}
            for seed, value in zip(selection_seeds, selection_values, strict=True)
        ],
        **paired_hamiltonian_values(),
    }

    summaries = {row["key"]: row for row in robustness["rows"]}
    expected_counts = {
        "fixed_b": 32,
        "washout_800": 10,
        "activity_matched": 8,
        "gap_matched": 24,
        "independent_selection": 24,
        "zz_x_z": 32,
        "xy_z_x": 32,
        "xx_ring": 32,
    }
    for key, seed_rows in rows.items():
        values = np.asarray([row["value"] for row in seed_rows], dtype=float)
        seeds = [int(row["seed"]) for row in seed_rows]
        expected = summaries[key]
        family_size = 2 if key == "gap_matched" else 1
        half_width = float(
            student_t.ppf(1 - 0.05 / (2 * family_size), len(values) - 1)
            * np.std(values, ddof=1)
            / np.sqrt(len(values))
        )
        if (
            len(values) != expected_counts[key]
            or len(set(seeds)) != len(seeds)
            or not np.all(np.isfinite(values))
            or not np.isclose(np.mean(values), expected["mean"], rtol=0, atol=5e-10)
            or not np.isclose(
                np.mean(values) - half_width,
                expected["ci95_low"],
                rtol=0,
                atol=5e-10,
            )
            or not np.isclose(
                np.mean(values) + half_width,
                expected["ci95_high"],
                rtol=0,
                atol=5e-10,
            )
        ):
            raise RuntimeError(f"scalar-control values disagree with {key}")

    return {
        "artifact_type": "figure3_figure6_forest_seed_values",
        "status": "complete",
        "metric": robustness["metric"],
        "sources": expected_hashes,
        "rows": rows,
    }


def main() -> int:
    payload = build_payload()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT} with "
        f"{sum(len(rows) for rows in payload['rows'].values())} seed effects"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

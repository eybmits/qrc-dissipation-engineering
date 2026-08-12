#!/usr/bin/env python3
"""Export the Figure 3 seed-level values from the sealed finite-size archive."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results" / "experiment1_finite_size_v2_results.tar.gz"
SNAPSHOT = ROOT / "paper" / "data" / "experiment1_finite_size_snapshot.json"
OUTPUT = (
    ROOT / "paper" / "data" / "experiment1_finite_size_seed_values.json"
)
SIZES = tuple(range(4, 9))
METHODS = ("CD_paper", "B3_collective")
TASKS = ("stm", "narma10")


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def checkpoint_digest(checkpoint: dict) -> str:
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_payload_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload() -> dict:
    summary = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    provenance = {
        (int(row["n_qubits"]), row["method"], int(row["seed"])): row[
            "checkpoint_payload_sha256"
        ]
        for row in summary["checkpoint_provenance"]
    }
    records: dict[int, dict[str, dict[int, dict[str, float]]]] = {
        size: {method: {} for method in METHODS} for size in SIZES
    }

    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or "/checkpoints/N" not in member.name:
                continue
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"could not read {member.name}")
            checkpoint = json.load(stream)
            size = int(checkpoint["n_qubits"])
            method = checkpoint["method"]
            if size not in records or method not in METHODS:
                continue
            seed = int(checkpoint["seed"])
            stored_digest = checkpoint.get("checkpoint_payload_sha256")
            expected_digest = provenance.get((size, method, seed))
            if (
                not isinstance(stored_digest, str)
                or stored_digest != checkpoint_digest(checkpoint)
                or stored_digest != expected_digest
            ):
                raise RuntimeError(f"checkpoint digest mismatch: {member.name}")
            if seed in records[size][method]:
                raise RuntimeError(f"duplicate checkpoint: N={size}/{method}/{seed}")
            records[size][method][seed] = {
                task: float(
                    checkpoint["task_results"][task]["primary_fixed"]["test"]
                )
                for task in TASKS
            }

    sizes_payload = {}
    summaries = summary["summaries"]["primary_fixed"]
    for size in SIZES:
        seed_sets = [set(records[size][method]) for method in METHODS]
        if seed_sets[0] != seed_sets[1] or len(seed_sets[0]) != 24:
            raise RuntimeError(f"incomplete paired seed block at N={size}")
        seeds = sorted(seed_sets[0])
        method_payload = {}
        for method in METHODS:
            task_payload = {
                task: [records[size][method][seed][task] for seed in seeds]
                for task in TASKS
            }
            for task, values in task_payload.items():
                observed_mean = float(np.mean(values))
                observed_se = float(np.std(values, ddof=1) / math.sqrt(len(values)))
                expected = summaries[str(size)][task][method]
                if not np.isclose(observed_mean, expected["mean"], rtol=0, atol=5e-12):
                    raise RuntimeError(f"mean mismatch: N={size}/{method}/{task}")
                if not np.isclose(observed_se, expected["se"], rtol=0, atol=5e-12):
                    raise RuntimeError(f"SE mismatch: N={size}/{method}/{task}")
            method_payload[method] = task_payload
        sizes_payload[str(size)] = {"seeds": seeds, "values": method_payload}

    return {
        "artifact_type": "figure3_finite_size_seed_values",
        "status": "complete",
        "source_archive": ARCHIVE.name,
        "source_archive_sha256": file_digest(ARCHIVE),
        "checkpoint_set_sha256": summary["checkpoint_set_sha256"],
        "readout": "primary_fixed",
        "methods": list(METHODS),
        "tasks": list(TASKS),
        "sizes": sizes_payload,
    }


def main() -> int:
    payload = build_payload()
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lineage_count = sum(len(value["seeds"]) for value in payload["sizes"].values())
    print(f"wrote {OUTPUT} with {lineage_count} paired lineages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

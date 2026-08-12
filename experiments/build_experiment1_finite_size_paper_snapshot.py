#!/usr/bin/env python3
"""Build the final finite-size snapshot used by the manuscript.

The helper validates every checkpoint in the frozen 24-lineage,
``N=4,...,8`` protocol and links the paper-facing summaries to the completed
62-test confirmatory aggregate.  It does not modify the production protocol.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import run_experiment1_finite_size as scaling  # noqa: E402


OUTDIR = ROOT / "results" / "experiment1_finite_size_v2"
AGGREGATE = OUTDIR / "aggregate.json"
OUTPUT = ROOT / "paper" / "data" / "experiment1_finite_size_snapshot.json"
COMPLETE_SIZES = (4, 5, 6, 7, 8)
LINEAGES_PER_SIZE = 24
TASKS = ("stm", "narma10")
READOUTS = ("primary_fixed", "validation_selected")


def _selected_seeds(protocol: dict, n_qubits: int) -> list[int]:
    if n_qubits in COMPLETE_SIZES:
        return [int(seed) for seed in protocol["seeds"]]
    raise ValueError(f"unsupported paper size N={n_qubits}")


def _validated_rows(
    protocol: dict,
) -> tuple[dict[tuple[int, str, int], dict], list[dict]]:
    indexed = {}
    provenance = []
    for n_qubits in COMPLETE_SIZES:
        for seed in _selected_seeds(protocol, n_qubits):
            paired_rows = []
            for method in protocol["methods"]:
                job = scaling.Job(n_qubits, method, seed)
                path = scaling.job_path(OUTDIR, job)
                row = scaling._validate_checkpoint(path, job, protocol)
                if row is None:
                    raise RuntimeError(
                        "paper snapshot requires the complete frozen grid; "
                        f"missing {path}"
                    )
                indexed[job.key] = row
                paired_rows.append(row)
                provenance.append(
                    {
                        "n_qubits": n_qubits,
                        "method": method,
                        "seed": seed,
                        "checkpoint_payload_sha256": row[
                            "checkpoint_payload_sha256"
                        ],
                    }
                )
            if len({row["full_input_sha256"] for row in paired_rows}) != 1:
                raise RuntimeError(
                    f"input pairing failed at N={n_qubits}, seed={seed}"
                )
            if len(
                {row["scaled_coupling_sha256"] for row in paired_rows}
            ) != 1:
                raise RuntimeError(
                    f"coupling pairing failed at N={n_qubits}, seed={seed}"
                )
            if len(
                {
                    scaling.canonical_json(row["target_sha256"])
                    for row in paired_rows
                }
            ) != 1:
                raise RuntimeError(
                    f"target pairing failed at N={n_qubits}, seed={seed}"
                )
            for row in paired_rows:
                if row["method"] in scaling.DISSIPATIVE_METHODS and float(
                    row["relative_jump_budget_error"]
                ) > 1e-10:
                    raise RuntimeError(
                        "jump-budget invariant failed at "
                        f"N={n_qubits}, method={row['method']}, seed={seed}"
                    )
    return indexed, provenance


def build_snapshot() -> dict:
    protocol = scaling.build_protocol()
    scaling.write_or_validate_protocol(OUTDIR, protocol)
    _, provenance = _validated_rows(protocol)
    if not AGGREGATE.is_file():
        raise RuntimeError(f"missing final aggregate: {AGGREGATE}")
    final = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    if (
        final.get("status") != "complete"
        or final.get("protocol_sha256") != scaling.protocol_sha256(protocol)
        or final.get("expected_checkpoints") != protocol["n_jobs"]
        or final.get("complete_checkpoints") != protocol["n_jobs"]
        or final.get("confirmatory_family_size") != 62
        or final.get("invariant_audit") != {"passed": True, "errors": []}
    ):
        raise RuntimeError("final aggregate violates the frozen paper contract")
    summaries = final["summary"]
    if set(summaries) != set(READOUTS):
        raise RuntimeError("final aggregate has an unexpected readout set")

    provenance_json = scaling.canonical_json(provenance).encode("utf-8")
    payload = {
        "status": "complete",
        "evidence_boundary": {
            "N4_to_N8": "complete 24-lineage paired blocks at every size",
            "confirmatory_analysis": "complete frozen 62-test Holm family",
        },
        "protocol_sha256": scaling.protocol_sha256(protocol),
        "source_environment_sha256": protocol[
            "source_environment_sha256"
        ],
        "complete_sizes": list(COMPLETE_SIZES),
        "lineages_per_size": LINEAGES_PER_SIZE,
        "methods": list(protocol["methods"]),
        "method_labels": protocol["method_labels"],
        "reference_method": scaling.REFERENCE_METHOD,
        "collective_method": scaling.COLLECTIVE_METHOD,
        "tasks": list(TASKS),
        "readouts": list(READOUTS),
        "measurement_scaling": {
            str(n_qubits): {
                "observables": scaling.expected_observable_count(n_qubits),
                "grouped_global_pauli_settings": 3,
            }
            for n_qubits in COMPLETE_SIZES
        },
        "summaries": summaries,
        "confirmatory_family_size": final["confirmatory_family_size"],
        "finite_range_endpoint_contrast": final[
            "finite_range_endpoint_contrast"
        ],
        "finite_range_slope_contrast": final[
            "finite_range_slope_contrast"
        ],
        "checkpoint_count": len(provenance),
        "checkpoint_set_sha256": hashlib.sha256(
            provenance_json
        ).hexdigest(),
        "checkpoint_provenance": provenance,
        "claim_boundary": (
            "Confirmatory finite-range strengthening over N=4,...,8 under "
            "the normalized nested-Hamiltonian protocol; not an asymptotic, "
            "Hamiltonian-universal, physical-cost, or shot-complexity result."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    scaling.atomic_write_json(OUTPUT, payload)
    return payload


def main() -> int:
    payload = build_snapshot()
    collective = payload["summaries"]["primary_fixed"]
    print(
        f"wrote {OUTPUT} from {payload['checkpoint_count']} validated "
        f"checkpoints; protocol={payload['protocol_sha256']}"
    )
    for n_qubits in COMPLETE_SIZES:
        summary = collective[str(n_qubits)]
        stm = summary["stm"][scaling.COLLECTIVE_METHOD][
            "versus_local"
        ]["relative"]
        narma = summary["narma10"][scaling.COLLECTIVE_METHOD][
            "versus_local"
        ]["relative"]
        print(
            f"N={n_qubits}: n={stm['n']}, "
            f"STM={100 * stm['mean']:+.2f}%, "
            f"NARMA={100 * narma['mean']:+.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

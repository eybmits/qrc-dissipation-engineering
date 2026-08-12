#!/usr/bin/env python3
"""Extend the frozen principal N=5 collective-convergence diagnostic.

The original switched-input control stops after 800 inputs.  This small
companion protocol continues exactly the same eight N=5 collective lineages
to 1,200 inputs so the convergence to numerical precision is visible in the
main figure.  The first 801 points are checked against the frozen aggregate;
the extension is never inferred or extrapolated from the earlier curve.
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
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrc import dissipators as dsp  # noqa: E402
from qrc import reservoirs as res  # noqa: E402
from qrc.liouvillian import unvec, vec  # noqa: E402
from qrc.reservoirs import (  # noqa: E402
    ising_xx_hamiltonian,
    transverse_drive,
)
from qrc.sparse_evolve import SparseLindbladReservoir  # noqa: E402


BASE_EVIDENCE_ROOT = (
    ROOT / "paper" / "evidence" / "switched_input_memory_control_v2"
)
EVIDENCE_ROOT = (
    ROOT / "paper" / "evidence" / "collective_N5_convergence_extension_v1"
)
BASE_AGGREGATE = BASE_EVIDENCE_ROOT / "aggregate.json"
PROTOCOL_PATH = EVIDENCE_ROOT / "protocol.json"
CHECKPOINT_ROOT = EVIDENCE_ROOT / "checkpoints"
RESULT_PATH = EVIDENCE_ROOT / "aggregate.json"
EVIDENCE_ZIP = ROOT / "results" / (
    "collective_loss_usable_memory_numerical_evidence.zip"
)
RAW_MEMBER_PREFIX = (
    "collective-loss-numerical-evidence/"
    "switched_input_memory_control_v2/results/convergence/"
)
RAW_MANIFEST_MEMBER = (
    "collective-loss-numerical-evidence/"
    "switched_input_memory_control_v2/SHA256SUMS"
)

PROTOCOL_VERSION = "collective-N5-convergence-extension-v1-2026-08-04"
LINEAGE_SEEDS = (
    181_031_542,
    307_836_921,
    613_734_097,
    818_242_779,
    1_067_952_113,
    1_319_483_087,
    1_673_840_529,
    1_984_601_307,
)
N_QUBITS = 5
STEPS = 1_200
BASE_STEPS = 800
H = 0.5
DT = 0.5
MULTIPLIER = 1.0
INPUT_SEED_OFFSET = 2_600_000
HAAR_SEED_OFFSET = 3_700_000
INITIAL_STATES = ("ground", "excited", "mixed", "haar")
PREFIX_ABSOLUTE_TOLERANCE = 5e-13


def canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    return sha256_bytes(canonical_json(payload).encode())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def protocol_payload() -> dict:
    source_paths = (
        ROOT / "src" / "qrc" / "dissipators.py",
        ROOT / "src" / "qrc" / "liouvillian.py",
        ROOT / "src" / "qrc" / "operators.py",
        ROOT / "src" / "qrc" / "reservoirs.py",
        ROOT / "src" / "qrc" / "sparse_evolve.py",
        Path(__file__),
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": "frozen_before_extension_results",
        "purpose": (
            "continue the frozen principal collective N=5 switched-input "
            "initial-state diagnostic from 800 to 1200 inputs"
        ),
        "base_aggregate_file_sha256": sha256_path(BASE_AGGREGATE),
        "base_evidence_zip_sha256": sha256_path(EVIDENCE_ZIP),
        "base_checkpoint_members_sha256": {
            raw_member(seed): raw_member_sha256(seed)
            for seed in LINEAGE_SEEDS
        },
        "scientific_sources_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path)
            for path in source_paths
        },
        "lineage_seeds": list(LINEAGE_SEEDS),
        "n_qubits": N_QUBITS,
        "steps": STEPS,
        "validated_prefix_steps": BASE_STEPS,
        "configuration": {
            "design": "collective",
            "regime": "principal",
            "h": H,
            "dt": DT,
            "frobenius_multiplier": MULTIPLIER,
        },
        "input": {
            "distribution": "iid Uniform[0,1]",
            "seed_rule": (
                "numpy.default_rng(2600000 + lineage_seed + "
                "1003 * n_qubits)"
            ),
            "prefix_identity": (
                "the first 800 draws are identical to the frozen v2 control"
            ),
        },
        "nested_couplings": (
            "top-left N-by-N block of one N=6 Uniform[-1,1] draw per "
            "lineage, scaled by sqrt(4/(N-1))"
        ),
        "initial_states": list(INITIAL_STATES),
        "trace_distance": "one half trace norm, maximum over all six pairs",
        "prefix_absolute_tolerance": PREFIX_ABSOLUTE_TOLERANCE,
    }


def freeze() -> None:
    protocol = protocol_payload()
    payload = {
        "artifact_type": "collective_convergence_extension_protocol",
        "protocol": protocol,
        "protocol_sha256": sha256_json(protocol),
    }
    if RESULT_PATH.exists():
        raise RuntimeError("refusing to refreeze after extension results exist")
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("existing extension protocol differs")
        print(f"verified frozen protocol: {PROTOCOL_PATH}")
        return
    atomic_json(PROTOCOL_PATH, payload)
    print(f"wrote frozen protocol: {PROTOCOL_PATH}")


def load_protocol() -> dict:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("freeze the extension protocol before running")
    payload = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    protocol = payload.get("protocol")
    if (
        payload.get("artifact_type")
        != "collective_convergence_extension_protocol"
        or not isinstance(protocol, dict)
        or payload.get("protocol_sha256") != sha256_json(protocol)
        or protocol != protocol_payload()
    ):
        raise RuntimeError("extension protocol or authenticated inputs drifted")
    return payload


def nested_couplings(seed: int) -> np.ndarray:
    parent = res.random_couplings(
        6,
        1.0,
        np.random.default_rng(seed),
    )
    return np.asarray(
        parent[:N_QUBITS, :N_QUBITS] * math.sqrt(4.0 / (N_QUBITS - 1)),
        dtype=float,
    )


def raw_member(seed: int) -> str:
    return (
        RAW_MEMBER_PREFIX
        + f"principal_collective_N5_seed_{seed}.json"
    )


def raw_member_bytes(seed: int) -> bytes:
    with zipfile.ZipFile(EVIDENCE_ZIP) as archive:
        payload = archive.read(raw_member(seed))
        manifest = archive.read(RAW_MANIFEST_MEMBER).decode("utf-8")
    relative = raw_member(seed).removeprefix(
        "collective-loss-numerical-evidence/"
        "switched_input_memory_control_v2/"
    )
    expected = None
    for line in manifest.splitlines():
        digest, _, name = line.partition("  ")
        if name == relative:
            expected = digest
            break
    if expected is None or sha256_bytes(payload) != expected:
        raise RuntimeError(
            f"frozen checkpoint checksum failed for seed {seed}"
        )
    return payload


def raw_member_sha256(seed: int) -> str:
    return sha256_bytes(raw_member_bytes(seed))


def build_reservoir(seed: int) -> SparseLindbladReservoir:
    raw = dsp.collective_loss(N_QUBITS, 1.0)
    target = (
        dsp.jump_strength(dsp.local_loss(N_QUBITS, 1.0)) * MULTIPLIER
    )
    jumps = dsp.normalize_jump_strength(raw, target)
    h0 = ising_xx_hamiltonian(nested_couplings(seed), H, N_QUBITS)
    hx = transverse_drive(N_QUBITS)
    return SparseLindbladReservoir.from_terms(
        N_QUBITS,
        h0 + H * hx,
        H * hx,
        jumps,
        DT,
    )


def density_states(seed: int) -> list[np.ndarray]:
    dimension = 2**N_QUBITS
    ground = np.zeros((dimension, dimension), dtype=complex)
    ground[0, 0] = 1.0
    excited = np.zeros_like(ground)
    excited[-1, -1] = 1.0
    mixed = np.eye(dimension, dtype=complex) / dimension
    rng = np.random.default_rng(
        HAAR_SEED_OFFSET + seed + 101 * N_QUBITS
    )
    vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    haar = np.outer(vector, vector.conjugate())
    return [ground, excited, mixed, haar]


def hermitian(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.conjugate().T) / 2


def state_metrics(state_vectors: np.ndarray, dimension: int) -> dict:
    states = [
        unvec(state_vectors[:, index], dimension)
        for index in range(state_vectors.shape[1])
    ]
    distances = []
    for left, right in combinations(range(len(states)), 2):
        eigenvalues = np.linalg.eigvalsh(
            hermitian(states[left] - states[right])
        )
        distances.append(float(0.5 * np.sum(np.abs(eigenvalues))))
    return {
        "max_trace_distance": float(max(distances)),
        "maximum_trace_error": float(
            max(abs(complex(np.trace(state)) - 1.0) for state in states)
        ),
        "maximum_hermiticity_error": float(
            max(
                np.max(np.abs(state - state.conjugate().T))
                for state in states
            )
        ),
    }


def compute_seed(seed: int, protocol_sha256: str) -> dict:
    started = time.time()
    reservoir = build_reservoir(seed)
    dimension = 2**N_QUBITS
    state_vectors = np.stack(
        [vec(state) for state in density_states(seed)],
        axis=1,
    )
    rng = np.random.default_rng(
        INPUT_SEED_OFFSET + seed + 1_003 * N_QUBITS
    )
    inputs = rng.random(STEPS)
    metrics = [state_metrics(state_vectors, dimension)]
    for input_value in inputs:
        state_vectors = expm_multiply(
            reservoir.liouvillian(float(input_value)) * reservoir.dt,
            state_vectors,
        )
        metrics.append(state_metrics(state_vectors, dimension))
    return {
        "artifact_type": (
            "collective_N5_convergence_extension_checkpoint"
        ),
        "status": "complete",
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "input_sha256": sha256_bytes(
            np.ascontiguousarray(inputs, dtype="<f8").tobytes()
        ),
        "input_prefix_sha256": sha256_bytes(
            np.ascontiguousarray(
                inputs[:BASE_STEPS],
                dtype="<f8",
            ).tobytes()
        ),
        "maximum_trace_distance": [
            row["max_trace_distance"] for row in metrics
        ],
        "maximum_numerical_trace_error": float(
            max(row["maximum_trace_error"] for row in metrics)
        ),
        "maximum_numerical_hermiticity_error": float(
            max(row["maximum_hermiticity_error"] for row in metrics)
        ),
        "runtime_seconds": float(time.time() - started),
    }


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"seed_{seed}.json"


def run_seed(seed: int, protocol_sha256: str) -> dict:
    path = checkpoint_path(seed)
    if path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("artifact_type")
            == "collective_N5_convergence_extension_checkpoint"
            and row.get("status") == "complete"
            and row.get("seed") == seed
            and row.get("protocol_sha256") == protocol_sha256
            and len(row.get("maximum_trace_distance", [])) == STEPS + 1
        ):
            return row
        raise RuntimeError(f"invalid existing checkpoint: {path}")
    row = compute_seed(seed, protocol_sha256)
    atomic_json(path, row)
    return row


def validate_result(payload: dict) -> dict:
    protocol = load_protocol()
    if (
        payload.get("artifact_type")
        != "collective_convergence_extension"
        or payload.get("status") != "complete"
        or payload.get("protocol_sha256") != protocol["protocol_sha256"]
        or payload.get("step") != list(range(STEPS + 1))
    ):
        raise RuntimeError("extension result metadata is invalid")
    rows = payload.get("per_seed")
    if (
        not isinstance(rows, list)
        or [row.get("seed") for row in rows] != list(LINEAGE_SEEDS)
    ):
        raise RuntimeError("extension seed ledger is invalid")
    curves = np.asarray(
        [row["maximum_trace_distance"] for row in rows],
        dtype=float,
    )
    if curves.shape != (len(LINEAGE_SEEDS), STEPS + 1):
        raise RuntimeError("extension curve shape is invalid")
    if not np.all(np.isfinite(curves)) or np.any(curves < 0):
        raise RuntimeError("extension curves contain invalid values")
    envelope = np.max(curves, axis=0)
    stored_envelope = np.asarray(
        payload["maximum_trace_distance_across_seeds"],
        dtype=float,
    )
    if not np.array_equal(envelope, stored_envelope):
        raise RuntimeError("stored extension envelope is not reconstructed")

    seed_prefix_errors: dict[str, float] = {}
    for row in rows:
        seed = int(row["seed"])
        raw = json.loads(raw_member_bytes(seed))
        raw_curve = np.asarray(raw["max_trace_distance"], dtype=float)
        if (
            raw.get("status") != "complete"
            or raw.get("job")
            != {
                "seed": seed,
                "n_qubits": N_QUBITS,
                "regime": "principal",
                "design": "collective",
            }
            or len(raw_curve) != BASE_STEPS + 1
            or row.get("input_prefix_sha256") != raw.get("input_sha256")
        ):
            raise RuntimeError(
                f"frozen checkpoint metadata differs for seed {seed}"
            )
        seed_prefix_errors[str(seed)] = float(
            np.max(
                np.abs(
                    np.asarray(
                        row["maximum_trace_distance"][: BASE_STEPS + 1],
                        dtype=float,
                    )
                    - raw_curve
                )
            )
        )
    prefix_error = float(max(seed_prefix_errors.values()))
    if prefix_error > PREFIX_ABSOLUTE_TOLERANCE:
        raise RuntimeError(
            f"extension prefix differs from frozen control by {prefix_error}"
        )
    tail_maximum = float(np.max(envelope[1_100 : STEPS + 1]))
    return {
        "maximum_prefix_absolute_error": prefix_error,
        "per_seed_prefix_absolute_error": seed_prefix_errors,
        "final_maximum_trace_distance": float(envelope[-1]),
        "tail_maximum_trace_distance_steps_1100_1200": tail_maximum,
        "convergence_gate": "tail maximum <= 1e-14",
        "convergence_gate_passed": bool(tail_maximum <= 1e-14),
    }


def run(workers: int) -> None:
    if RESULT_PATH.exists():
        raise RuntimeError("extension result already exists; use verify")
    protocol = load_protocol()
    protocol_sha256 = protocol["protocol_sha256"]
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_seed, seed, protocol_sha256): seed
            for seed in LINEAGE_SEEDS
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index}/{len(LINEAGE_SEEDS)}] seed={row['seed']} "
                f"runtime={row['runtime_seconds']:.1f}s",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["seed"]))
    curves = np.asarray(
        [row["maximum_trace_distance"] for row in rows],
        dtype=float,
    )
    payload = {
        "artifact_type": "collective_convergence_extension",
        "status": "complete",
        "protocol_sha256": protocol_sha256,
        "step": list(range(STEPS + 1)),
        "per_seed": rows,
        "maximum_trace_distance_across_seeds": np.max(
            curves,
            axis=0,
        ).tolist(),
    }
    validation = validate_result(payload)
    payload["validation"] = validation
    atomic_json(RESULT_PATH, payload)
    print(
        f"wrote {RESULT_PATH}; final worst trace distance "
        f"{validation['final_maximum_trace_distance']:.3e}"
    )


def verify() -> None:
    if not RESULT_PATH.is_file():
        raise RuntimeError("extension result is missing")
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    validation = validate_result(payload)
    stored_validation = payload.get("validation")
    if not isinstance(stored_validation, dict):
        raise RuntimeError("extension validation record is missing")
    for key in (
        "maximum_prefix_absolute_error",
        "final_maximum_trace_distance",
        "tail_maximum_trace_distance_steps_1100_1200",
    ):
        value = validation[key]
        if not np.isclose(
            float(stored_validation[key]),
            float(value),
            rtol=0,
            atol=1e-18,
        ):
            raise RuntimeError(f"stored validation drifted for {key}")
    if (
        stored_validation.get("per_seed_prefix_absolute_error")
        != validation["per_seed_prefix_absolute_error"]
    ):
        raise RuntimeError("stored per-seed prefix validation drifted")
    if (
        stored_validation.get("convergence_gate")
        != validation["convergence_gate"]
        or stored_validation.get("convergence_gate_passed")
        != validation["convergence_gate_passed"]
    ):
        raise RuntimeError("stored convergence gate validation drifted")
    print(
        "verified collective convergence extension: "
        f"prefix error={validation['maximum_prefix_absolute_error']:.3e}, "
        f"final={validation['final_maximum_trace_distance']:.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=4)
    subparsers.add_parser("verify")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "run":
        run(args.workers)
    elif args.command == "verify":
        verify()


if __name__ == "__main__":
    main()

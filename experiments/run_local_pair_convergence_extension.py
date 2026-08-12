#!/usr/bin/env python3
"""Extend frozen local and pair convergence records through 1,200 inputs.

The switched-input control used by the manuscript originally stops after 800
inputs for the local and pair-loss designs.  Figure 3(c) displays the
worst-case envelope over N=4,5,6, so this companion protocol continues all
2 designs x 3 sizes x 8 frozen lineages to 1,200 inputs.  Every 0,...,800
prefix is authenticated against the sealed raw archive; no tail is inferred
or graphically extrapolated.
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
from functools import lru_cache
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
    ROOT / "paper" / "evidence" / "local_pair_convergence_extension_v1"
)
BASE_AGGREGATE = BASE_EVIDENCE_ROOT / "aggregate.json"
PROTOCOL_PATH = EVIDENCE_ROOT / "protocol.json"
CHECKPOINT_ROOT = EVIDENCE_ROOT / "checkpoints"
RESULT_PATH = EVIDENCE_ROOT / "aggregate.json"
CHECKSUM_PATH = EVIDENCE_ROOT / "SHA256SUMS"
EVIDENCE_ZIP = (
    ROOT / "results" / "collective_loss_usable_memory_numerical_evidence.zip"
)
RAW_MEMBER_PREFIX = (
    "collective-loss-numerical-evidence/"
    "switched_input_memory_control_v2/results/convergence/"
)
RAW_MANIFEST_MEMBER = (
    "collective-loss-numerical-evidence/"
    "switched_input_memory_control_v2/SHA256SUMS"
)

PROTOCOL_VERSION = "local-pair-convergence-extension-v1-2026-08-05"
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
DESIGNS = ("local", "pair")
SIZES = (4, 5, 6)
STEPS = 1_200
BASE_STEPS = 800
H = 0.5
DT = 0.5
MULTIPLIER = 1.0
INPUT_SEED_OFFSET = 2_600_000
HAAR_SEED_OFFSET = 3_700_000
INITIAL_STATES = ("ground", "excited", "mixed", "haar")
PREFIX_ABSOLUTE_TOLERANCE = 5e-13
CONVERGENCE_GATE = 1e-14


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


def jobs() -> list[dict]:
    return [
        {"design": design, "n_qubits": size, "seed": seed}
        for design in DESIGNS
        for size in SIZES
        for seed in LINEAGE_SEEDS
    ]


def case_key(design: str, n_qubits: int) -> str:
    return f"principal_{design}_N{n_qubits}"


def raw_member(job: dict) -> str:
    return (
        RAW_MEMBER_PREFIX
        + f"{case_key(job['design'], job['n_qubits'])}"
        + f"_seed_{job['seed']}.json"
    )


@lru_cache(maxsize=1)
def raw_manifest() -> dict[str, str]:
    with zipfile.ZipFile(EVIDENCE_ZIP) as archive:
        text = archive.read(RAW_MANIFEST_MEMBER).decode("utf-8")
    manifest: dict[str, str] = {}
    for line in text.splitlines():
        digest, separator, name = line.partition("  ")
        if separator:
            manifest[name] = digest
    return manifest


@lru_cache(maxsize=None)
def raw_member_bytes(design: str, n_qubits: int, seed: int) -> bytes:
    job = {"design": design, "n_qubits": n_qubits, "seed": seed}
    member = raw_member(job)
    with zipfile.ZipFile(EVIDENCE_ZIP) as archive:
        payload = archive.read(member)
    relative = member.removeprefix(
        "collective-loss-numerical-evidence/"
        "switched_input_memory_control_v2/"
    )
    expected = raw_manifest().get(relative)
    if expected is None or sha256_bytes(payload) != expected:
        raise RuntimeError(f"frozen checkpoint checksum failed: {relative}")
    return payload


def raw_member_sha256(job: dict) -> str:
    return sha256_bytes(
        raw_member_bytes(job["design"], job["n_qubits"], job["seed"])
    )


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
            "continue every plotted principal local and pair-loss "
            "switched-input initial-state diagnostic from 800 to 1200 inputs"
        ),
        "base_aggregate_file_sha256": sha256_path(BASE_AGGREGATE),
        "base_evidence_zip_sha256": sha256_path(EVIDENCE_ZIP),
        "base_checkpoint_members_sha256": {
            raw_member(job): raw_member_sha256(job)
            for job in jobs()
        },
        "scientific_sources_sha256": {
            str(path.relative_to(ROOT)): sha256_path(path)
            for path in source_paths
        },
        "designs": list(DESIGNS),
        "sizes": list(SIZES),
        "lineage_seeds": list(LINEAGE_SEEDS),
        "steps": STEPS,
        "validated_prefix_steps": BASE_STEPS,
        "configuration": {
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
        "convergence_gate": CONVERGENCE_GATE,
        "jobs": jobs(),
    }


def freeze() -> None:
    protocol = protocol_payload()
    payload = {
        "artifact_type": "local_pair_convergence_extension_protocol",
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
        != "local_pair_convergence_extension_protocol"
        or not isinstance(protocol, dict)
        or payload.get("protocol_sha256") != sha256_json(protocol)
        or protocol != protocol_payload()
    ):
        raise RuntimeError("extension protocol or authenticated inputs drifted")
    return payload


def nested_couplings(seed: int, n_qubits: int) -> np.ndarray:
    parent = res.random_couplings(
        6,
        1.0,
        np.random.default_rng(seed),
    )
    return np.asarray(
        parent[:n_qubits, :n_qubits]
        * math.sqrt(4.0 / (n_qubits - 1)),
        dtype=float,
    )


def build_reservoir(
    seed: int,
    n_qubits: int,
    design: str,
) -> SparseLindbladReservoir:
    if design == "local":
        raw = dsp.local_loss(n_qubits, 1.0)
    elif design == "pair":
        raw = dsp.pair_loss(
            n_qubits,
            1.0,
            [
                (left, right)
                for left in range(n_qubits)
                for right in range(left + 1, n_qubits)
            ],
        )
    else:
        raise ValueError(design)
    target = (
        dsp.jump_strength(dsp.local_loss(n_qubits, 1.0)) * MULTIPLIER
    )
    jumps = dsp.normalize_jump_strength(raw, target)
    h0 = ising_xx_hamiltonian(
        nested_couplings(seed, n_qubits),
        H,
        n_qubits,
    )
    hx = transverse_drive(n_qubits)
    return SparseLindbladReservoir.from_terms(
        n_qubits,
        h0 + H * hx,
        H * hx,
        jumps,
        DT,
    )


def density_states(seed: int, n_qubits: int) -> list[np.ndarray]:
    dimension = 2**n_qubits
    ground = np.zeros((dimension, dimension), dtype=complex)
    ground[0, 0] = 1.0
    excited = np.zeros_like(ground)
    excited[-1, -1] = 1.0
    mixed = np.eye(dimension, dtype=complex) / dimension
    rng = np.random.default_rng(
        HAAR_SEED_OFFSET + seed + 101 * n_qubits
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


def compute_job(job: dict, protocol_sha256: str) -> dict:
    started = time.time()
    design = str(job["design"])
    n_qubits = int(job["n_qubits"])
    seed = int(job["seed"])
    reservoir = build_reservoir(seed, n_qubits, design)
    dimension = 2**n_qubits
    state_vectors = np.stack(
        [vec(state) for state in density_states(seed, n_qubits)],
        axis=1,
    )
    rng = np.random.default_rng(
        INPUT_SEED_OFFSET + seed + 1_003 * n_qubits
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
        "artifact_type": "local_pair_convergence_extension_checkpoint",
        "status": "complete",
        "job": {
            "design": design,
            "n_qubits": n_qubits,
            "regime": "principal",
            "seed": seed,
        },
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


def checkpoint_path(job: dict) -> Path:
    return CHECKPOINT_ROOT / (
        f"{case_key(job['design'], job['n_qubits'])}"
        f"_seed_{job['seed']}.json"
    )


def run_job(job: dict, protocol_sha256: str) -> dict:
    path = checkpoint_path(job)
    if path.exists():
        row = json.loads(path.read_text(encoding="utf-8"))
        if (
            row.get("artifact_type")
            == "local_pair_convergence_extension_checkpoint"
            and row.get("status") == "complete"
            and row.get("job")
            == {
                "design": job["design"],
                "n_qubits": job["n_qubits"],
                "regime": "principal",
                "seed": job["seed"],
            }
            and row.get("protocol_sha256") == protocol_sha256
            and len(row.get("maximum_trace_distance", [])) == STEPS + 1
        ):
            return row
        raise RuntimeError(f"invalid existing checkpoint: {path}")
    row = compute_job(job, protocol_sha256)
    atomic_json(path, row)
    return row


def load_checkpoint(job: dict, protocol_sha256: str) -> dict:
    path = checkpoint_path(job)
    if not path.is_file():
        raise RuntimeError(f"missing extension checkpoint: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    expected_job = {
        "design": job["design"],
        "n_qubits": job["n_qubits"],
        "regime": "principal",
        "seed": job["seed"],
    }
    if (
        row.get("artifact_type")
        != "local_pair_convergence_extension_checkpoint"
        or row.get("status") != "complete"
        or row.get("job") != expected_job
        or row.get("protocol_sha256") != protocol_sha256
        or len(row.get("maximum_trace_distance", [])) != STEPS + 1
    ):
        raise RuntimeError(f"invalid extension checkpoint: {path}")
    return row


def validate_result(payload: dict) -> dict:
    protocol = load_protocol()
    protocol_sha256 = protocol["protocol_sha256"]
    if (
        payload.get("artifact_type")
        != "local_pair_convergence_extension"
        or payload.get("status") != "complete"
        or payload.get("protocol_sha256") != protocol_sha256
    ):
        raise RuntimeError("extension result metadata is invalid")
    observed_cases = payload.get("cases")
    expected_case_names = {
        case_key(design, size)
        for design in DESIGNS
        for size in SIZES
    }
    if not isinstance(observed_cases, dict) or set(observed_cases) != expected_case_names:
        raise RuntimeError("extension result case set is invalid")

    prefix_errors: dict[str, float] = {}
    tail_maxima: dict[str, float] = {}
    finals: dict[str, float] = {}
    numerical_trace_errors: list[float] = []
    numerical_hermiticity_errors: list[float] = []
    for design in DESIGNS:
        for size in SIZES:
            name = case_key(design, size)
            case_jobs = [
                {"design": design, "n_qubits": size, "seed": seed}
                for seed in LINEAGE_SEEDS
            ]
            rows = [
                load_checkpoint(job, protocol_sha256)
                for job in case_jobs
            ]
            curves = np.asarray(
                [row["maximum_trace_distance"] for row in rows],
                dtype=float,
            )
            if (
                curves.shape != (len(LINEAGE_SEEDS), STEPS + 1)
                or not np.all(np.isfinite(curves))
                or np.any(curves < 0)
            ):
                raise RuntimeError(f"invalid extension curves: {name}")
            envelope = np.max(curves, axis=0)
            stored = observed_cases[name]
            if (
                stored.get("step") != list(range(STEPS + 1))
                or not np.array_equal(
                    envelope,
                    np.asarray(
                        stored["maximum_trace_distance_across_seeds"],
                        dtype=float,
                    ),
                )
            ):
                raise RuntimeError(f"stored extension envelope drifted: {name}")

            case_prefix_errors = []
            for job, row in zip(case_jobs, rows, strict=True):
                raw = json.loads(
                    raw_member_bytes(design, size, job["seed"])
                )
                raw_curve = np.asarray(raw["max_trace_distance"], dtype=float)
                if (
                    raw.get("status") != "complete"
                    or raw.get("job")
                    != {
                        "seed": job["seed"],
                        "n_qubits": size,
                        "regime": "principal",
                        "design": design,
                    }
                    or len(raw_curve) != BASE_STEPS + 1
                    or row.get("input_prefix_sha256")
                    != raw.get("input_sha256")
                ):
                    raise RuntimeError(
                        f"frozen checkpoint metadata differs: "
                        f"{name}, seed={job['seed']}"
                    )
                case_prefix_errors.append(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    row["maximum_trace_distance"][
                                        : BASE_STEPS + 1
                                    ],
                                    dtype=float,
                                )
                                - raw_curve
                            )
                        )
                    )
                )
                numerical_trace_errors.append(
                    float(row["maximum_numerical_trace_error"])
                )
                numerical_hermiticity_errors.append(
                    float(row["maximum_numerical_hermiticity_error"])
                )
            prefix_errors[name] = float(max(case_prefix_errors))
            tail_maxima[name] = float(np.max(envelope[1_100 : STEPS + 1]))
            finals[name] = float(envelope[-1])

    maximum_prefix_error = float(max(prefix_errors.values()))
    if maximum_prefix_error > PREFIX_ABSOLUTE_TOLERANCE:
        raise RuntimeError(
            "extension prefix differs from frozen control by "
            f"{maximum_prefix_error}"
        )
    gates = {
        name: bool(value <= CONVERGENCE_GATE)
        for name, value in tail_maxima.items()
    }
    if not all(gates.values()):
        raise RuntimeError(f"local/pair convergence gate failed: {gates}")
    return {
        "maximum_prefix_absolute_error": maximum_prefix_error,
        "per_case_prefix_absolute_error": prefix_errors,
        "per_case_tail_maximum_steps_1100_1200": tail_maxima,
        "per_case_final_maximum_trace_distance": finals,
        "convergence_gate": "every case tail maximum <= 1e-14",
        "per_case_convergence_gate_passed": gates,
        "all_convergence_gates_passed": True,
        "maximum_numerical_trace_error": float(
            max(numerical_trace_errors)
        ),
        "maximum_numerical_hermiticity_error": float(
            max(numerical_hermiticity_errors)
        ),
    }


def write_checksum_manifest() -> None:
    files = [PROTOCOL_PATH, RESULT_PATH, *sorted(CHECKPOINT_ROOT.glob("*.json"))]
    rows = [
        f"{sha256_path(path)}  {path.relative_to(EVIDENCE_ROOT).as_posix()}"
        for path in sorted(files, key=lambda item: item.as_posix())
    ]
    CHECKSUM_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_checksum_manifest() -> None:
    if not CHECKSUM_PATH.is_file():
        raise RuntimeError("extension checksum manifest is missing")
    observed: dict[str, str] = {}
    for line in CHECKSUM_PATH.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or relative in observed:
            raise RuntimeError("extension checksum manifest is malformed")
        observed[relative] = digest
    expected_paths = {
        path.relative_to(EVIDENCE_ROOT).as_posix(): path
        for path in (
            PROTOCOL_PATH,
            RESULT_PATH,
            *sorted(CHECKPOINT_ROOT.glob("*.json")),
        )
    }
    if set(observed) != set(expected_paths):
        raise RuntimeError("extension checksum manifest membership drifted")
    for relative, path in expected_paths.items():
        if sha256_path(path) != observed[relative]:
            raise RuntimeError(f"extension checksum mismatch: {relative}")


def run(workers: int) -> None:
    if RESULT_PATH.exists():
        raise RuntimeError("extension result already exists; use verify")
    protocol = load_protocol()
    protocol_sha256 = protocol["protocol_sha256"]
    all_jobs = jobs()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_job, job, protocol_sha256): job
            for job in all_jobs
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            job = row["job"]
            print(
                f"[{index}/{len(all_jobs)}] "
                f"{job['design']} N={job['n_qubits']} seed={job['seed']} "
                f"runtime={row['runtime_seconds']:.1f}s",
                flush=True,
            )

    cases: dict[str, dict] = {}
    for design in DESIGNS:
        for size in SIZES:
            name = case_key(design, size)
            rows = [
                load_checkpoint(
                    {"design": design, "n_qubits": size, "seed": seed},
                    protocol_sha256,
                )
                for seed in LINEAGE_SEEDS
            ]
            curves = np.asarray(
                [row["maximum_trace_distance"] for row in rows],
                dtype=float,
            )
            cases[name] = {
                "step": list(range(STEPS + 1)),
                "maximum_trace_distance_across_seeds": np.max(
                    curves,
                    axis=0,
                ).tolist(),
                "lineage_count": len(rows),
            }
    payload = {
        "artifact_type": "local_pair_convergence_extension",
        "status": "complete",
        "protocol_sha256": protocol_sha256,
        "cases": cases,
    }
    payload["validation"] = validate_result(payload)
    atomic_json(RESULT_PATH, payload)
    write_checksum_manifest()
    print(
        f"wrote {RESULT_PATH}; worst tail trace distance "
        f"{max(payload['validation']['per_case_tail_maximum_steps_1100_1200'].values()):.3e}"
    )


def verify() -> None:
    if not RESULT_PATH.is_file():
        raise RuntimeError("extension result is missing")
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    validation = validate_result(payload)
    if payload.get("validation") != validation:
        raise RuntimeError("stored extension validation drifted")
    verify_checksum_manifest()
    print(
        "verified local/pair convergence extension: "
        f"prefix error={validation['maximum_prefix_absolute_error']:.3e}, "
        "worst tail="
        f"{max(validation['per_case_tail_maximum_steps_1100_1200'].values()):.3e}"
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
    else:
        verify()


if __name__ == "__main__":
    main()

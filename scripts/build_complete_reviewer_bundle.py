#!/usr/bin/env python3
"""Build and verify the self-contained reviewer reproducibility bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "complete_reviewer_bundle.zip"
)
ARCHIVE_ROOT = "dissipative-architecture-complete-reproducibility"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o644

INPUTS = {
    "REPRODUCIBILITY_INDEX.md": REPO_ROOT
    / "results"
    / "REPRODUCIBILITY_INDEX.md",
    "manuscript.pdf": REPO_ROOT / "paper" / "dissipation_qrc.pdf",
    "manuscript_source.zip": REPO_ROOT
    / "results"
    / "arxiv_submission.zip",
    "numerical_evidence.zip": REPO_ROOT
    / "results"
    / "collective_loss_usable_memory_numerical_evidence.zip",
    "reset_architecture_evidence.tar.gz": REPO_ROOT
    / "results"
    / "reset_architecture_replication_results.tar.gz",
    "phase_direction_confirmatory_v1_results.tar.gz": REPO_ROOT
    / "results"
    / "phase_direction_confirmatory_v1_results.tar.gz",
    "rank_one_orientation_v1_results.tar.gz": REPO_ROOT
    / "results"
    / "rank_one_orientation_v1_results.tar.gz",
    "protocol_manifest.json": REPO_ROOT
    / "paper"
    / "data"
    / "reproducibility_manifest.json",
}


class ReviewerBundleError(RuntimeError):
    """Raised when the reviewer bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class Payload:
    name: str
    data: bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or ":" in path.parts[0]
    ):
        raise ReviewerBundleError(f"unsafe archive path: {value}")
    return path.as_posix()


def _readme() -> bytes:
    return (
        "Complete reviewer reproducibility bundle\n"
        "========================================\n\n"
        "This self-contained package accompanies the manuscript\n"
        "\"The Organization of Environmental Coupling Shapes What Quantum Reservoirs Remember.\"\n"
        "It does not require access to the project repository.\n\n"
        "Start here:\n"
        "  1. Run: python3 validate_complete_bundle.py\n"
        "  2. Read REPRODUCIBILITY_INDEX.md for the figure-to-record map.\n"
        "  3. Inspect protocol_manifest.json for every ordered seed array,\n"
        "     collective coefficient, grid, optimizer, tolerance, and split.\n\n"
        "numerical_evidence.zip contains the raw per-seed outputs, calibration\n"
        "histories, frozen protocols, code snapshots, negative controls, and\n"
        "its own nested validator. reset_architecture_evidence.tar.gz contains\n"
        "the 16-pair input-by-reset replication and four-state audit.\n"
        "phase_direction_confirmatory_v1_results.tar.gz contains the frozen\n"
        "32-pair phase-direction intervention, all task checkpoints, and its\n"
        "four-state convergence and numerical-replay audit.\n"
        "rank_one_orientation_v1_results.tar.gz contains the independent\n"
        "24-pair real sign-balanced replication at a second finite size, its\n"
        "matched-channel audit, raw checkpoints, and frozen source.\n"
        "manuscript_source.zip contains the complete\n"
        "journal source, figure code and data, the eight-lineage N=5\n"
        "collective continuation, and all 48 local/pair cross-size\n"
        "switched-input convergence continuations through 1200 inputs.\n"
    ).encode("utf-8")


def _embedded_validator() -> bytes:
    return r'''#!/usr/bin/env python3
"""Validate the complete reviewer bundle using the Python standard library."""
from __future__ import annotations

import hashlib
import csv
import io
import json
import math
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_name(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(
        bool(path.parts)
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in value
        and "\x00" not in value
        and ":" not in path.parts[0],
        f"unsafe ZIP member: {value}",
    )
    return path


def json_digest(value: object) -> str:
    return digest(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )


def verify_extension_manifest(
    source: zipfile.ZipFile,
    prefix: str,
    expected_relatives: set[str],
) -> None:
    rows = {}
    manifest_name = prefix + "SHA256SUMS"
    for number, line in enumerate(
        source.read(manifest_name).decode().splitlines(),
        1,
    ):
        parts = line.split("  ", 1)
        require(
            len(parts) == 2,
            f"{manifest_name}:{number}: malformed row",
        )
        sha, relative = parts
        safe_name(relative)
        require(
            relative not in rows,
            f"{manifest_name}: duplicate row: {relative}",
        )
        rows[relative] = sha
    require(
        set(rows) == expected_relatives,
        f"{manifest_name}: evidence file set mismatch",
    )
    for relative, sha in rows.items():
        require(
            digest(source.read(prefix + relative)) == sha,
            f"{manifest_name}: checksum mismatch: {relative}",
        )


def curve_envelope(curves: list[list[float]]) -> list[float]:
    require(bool(curves), "cannot construct an empty envelope")
    width = len(curves[0])
    require(
        all(len(curve) == width for curve in curves),
        "continuation curves have inconsistent lengths",
    )
    return [max(curve[index] for curve in curves) for index in range(width)]


def verify_manifest() -> None:
    rows = {}
    for number, line in enumerate((ROOT / "SHA256SUMS.txt").read_text().splitlines(), 1):
        parts = line.split("  ", 1)
        require(len(parts) == 2, f"SHA256SUMS.txt:{number}: malformed row")
        sha, relative = parts
        require(relative not in rows, f"duplicate manifest row: {relative}")
        rows[relative] = sha
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    }
    require(actual == set(rows), "top-level manifest file set mismatch")
    for relative, sha in rows.items():
        require(
            digest((ROOT / relative).read_bytes()) == sha,
            f"checksum mismatch: {relative}",
        )


def verify_source() -> tuple[dict, dict]:
    path = ROOT / "manuscript_source.zip"
    numerical_path = ROOT / "numerical_evidence.zip"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(
        numerical_path
    ) as numerical:
        names = source.namelist()
        require(len(names) == len(set(names)), "source ZIP has duplicate members")
        for name in names:
            safe_name(name)
        require("SHA256SUMS.txt" in names, "source ZIP lacks SHA256SUMS.txt")
        for number, line in enumerate(
            source.read("SHA256SUMS.txt").decode().splitlines(), 1
        ):
            parts = line.split("  ", 1)
            require(len(parts) == 2, f"source manifest:{number}: malformed row")
            sha, relative = parts
            require(relative in names, f"source ZIP missing {relative}")
            require(
                digest(source.read(relative)) == sha,
                f"source checksum mismatch: {relative}",
            )

        manifest = json.loads(source.read("data/reproducibility_manifest.json"))
        current_evidence_sha256 = digest(numerical_path.read_bytes())
        require(
            manifest["release"]["evidence_archive_sha256"]
            == current_evidence_sha256,
            "current numerical evidence checksum changed",
        )
        collective = manifest["collective_process"]
        require(
            collective["coefficients"] == "c_i=1 for every site",
            "collective coefficients are not explicit",
        )
        require(
            collective["redrawn_or_fitted_in_primary_and_controls"] is False,
            "collective coefficients unexpectedly vary",
        )

        prefix = "evidence/collective_N5_convergence_extension_v1/"
        protocol_record = json.loads(source.read(prefix + "protocol.json"))
        protocol = protocol_record["protocol"]
        aggregate = json.loads(source.read(prefix + "aggregate.json"))
        seeds = protocol["lineage_seeds"]
        checkpoint_relatives = {
            f"checkpoints/seed_{seed}.json"
            for seed in seeds
        }
        require(
            len(seeds) == 8 and len(set(seeds)) == 8,
            "N=5 continuation lineage set changed",
        )
        verify_extension_manifest(
            source,
            prefix,
            {
                "protocol.json",
                "aggregate.json",
                *checkpoint_relatives,
            },
        )
        require(
            protocol_record["protocol_sha256"] == json_digest(protocol),
            "N=5 continuation protocol hash failed",
        )
        require(protocol["steps"] == 1200, "continuation does not reach 1200")
        require(
            protocol["initial_states"] == ["ground", "excited", "mixed", "haar"],
            "continuation initial states changed",
        )
        require(
            len(protocol["base_evidence_zip_sha256"]) == 64
            and all(
                character in "0123456789abcdef"
                for character in protocol["base_evidence_zip_sha256"]
            ),
            "N=5 continuation base archive identity is malformed",
        )
        require(
            protocol["base_aggregate_file_sha256"]
            == digest(
                source.read(
                    "evidence/switched_input_memory_control_v2/aggregate.json"
                )
            ),
            "N=5 continuation base aggregate changed",
        )
        require(
            protocol["scientific_sources_sha256"][
                "experiments/run_collective_convergence_extension.py"
            ]
            == digest(
                source.read(
                    prefix + "run_collective_convergence_extension.py"
                )
            ),
            "N=5 continuation driver snapshot changed",
        )

        collective_rows = []
        collective_prefix_errors = {}
        collective_curves = []
        raw_prefix = (
            "collective-loss-numerical-evidence/"
            "switched_input_memory_control_v2/results/convergence/"
        )
        for seed in seeds:
            row = json.loads(
                source.read(prefix + f"checkpoints/seed_{seed}.json")
            )
            require(
                row["artifact_type"]
                == "collective_N5_convergence_extension_checkpoint"
                and row["status"] == "complete"
                and row["seed"] == seed
                and row["protocol_sha256"]
                == protocol_record["protocol_sha256"],
                f"N=5 continuation checkpoint metadata failed: {seed}",
            )
            curve = row["maximum_trace_distance"]
            require(
                len(curve) == 1201,
                f"N=5 continuation curve length failed: {seed}",
            )
            raw_name = (
                raw_prefix
                + f"principal_collective_N5_seed_{seed}.json"
            )
            raw_bytes = numerical.read(raw_name)
            require(
                digest(raw_bytes)
                == protocol["base_checkpoint_members_sha256"][raw_name],
                f"N=5 frozen checkpoint hash failed: {seed}",
            )
            raw = json.loads(raw_bytes)
            require(
                raw["job"]
                == {
                    "design": "collective",
                    "n_qubits": 5,
                    "regime": "principal",
                    "seed": seed,
                }
                and len(raw["max_trace_distance"]) == 801
                and row["input_prefix_sha256"] == raw["input_sha256"],
                f"N=5 frozen checkpoint metadata failed: {seed}",
            )
            prefix_error = max(
                abs(a - b)
                for a, b in zip(
                    curve[:801],
                    raw["max_trace_distance"],
                    strict=True,
                )
            )
            require(
                prefix_error <= protocol["prefix_absolute_tolerance"],
                f"N=5 frozen prefix failed: {seed}",
            )
            collective_prefix_errors[str(seed)] = prefix_error
            collective_curves.append(curve)
            collective_rows.append(row)

        collective_envelope = curve_envelope(collective_curves)
        require(aggregate["status"] == "complete", "continuation is incomplete")
        require(
            aggregate["protocol_sha256"] == protocol_record["protocol_sha256"]
            and aggregate["step"] == list(range(1201))
            and aggregate["per_seed"] == collective_rows
            and aggregate["maximum_trace_distance_across_seeds"]
            == collective_envelope,
            "N=5 continuation aggregate reconstruction failed",
        )
        collective_tail = max(collective_envelope[1100:])
        collective_validation = aggregate["validation"]
        require(
            collective_validation["convergence_gate_passed"] is True
            and collective_tail <= 1e-14,
            "continuation convergence gate failed",
        )
        require(
            abs(
                collective_validation["maximum_prefix_absolute_error"]
                - max(collective_prefix_errors.values())
            )
            <= 1e-18
            and collective_validation["per_seed_prefix_absolute_error"]
            == collective_prefix_errors
            and abs(
                collective_validation[
                    "tail_maximum_trace_distance_steps_1100_1200"
                ]
                - collective_tail
            )
            <= 1e-18
            and collective_validation["final_maximum_trace_distance"]
            == collective_envelope[-1],
            "N=5 continuation validation summary drifted",
        )

        local_pair_prefix = "evidence/local_pair_convergence_extension_v1/"
        local_pair_protocol_record = json.loads(
            source.read(local_pair_prefix + "protocol.json")
        )
        local_pair_protocol = local_pair_protocol_record["protocol"]
        local_pair_aggregate = json.loads(
            source.read(local_pair_prefix + "aggregate.json")
        )
        local_pair_seeds = local_pair_protocol["lineage_seeds"]
        expected_jobs = [
            {"design": design, "n_qubits": size, "seed": seed}
            for design in ("local", "pair")
            for size in (4, 5, 6)
            for seed in local_pair_seeds
        ]
        require(
            len(local_pair_seeds) == 8
            and len(set(local_pair_seeds)) == 8
            and local_pair_protocol["jobs"] == expected_jobs,
            "local/pair continuation lineage set changed",
        )
        local_pair_checkpoint_relatives = {
            "checkpoints/"
            f"principal_{job['design']}_N{job['n_qubits']}"
            f"_seed_{job['seed']}.json"
            for job in expected_jobs
        }
        verify_extension_manifest(
            source,
            local_pair_prefix,
            {
                "protocol.json",
                "aggregate.json",
                *local_pair_checkpoint_relatives,
            },
        )
        require(
            local_pair_protocol_record["protocol_sha256"]
            == json_digest(local_pair_protocol),
            "local/pair continuation protocol hash failed",
        )
        require(
            local_pair_protocol["steps"] == 1200,
            "local/pair continuation does not reach 1200",
        )
        require(
            local_pair_protocol["designs"] == ["local", "pair"]
            and local_pair_protocol["sizes"] == [4, 5, 6],
            "local/pair continuation coverage changed",
        )
        require(
            local_pair_protocol["initial_states"]
            == ["ground", "excited", "mixed", "haar"],
            "local/pair continuation initial states changed",
        )
        require(
            local_pair_protocol["base_evidence_zip_sha256"]
            == protocol["base_evidence_zip_sha256"]
            and local_pair_protocol["base_aggregate_file_sha256"]
            == digest(
                source.read(
                    "evidence/switched_input_memory_control_v2/aggregate.json"
                )
            ),
            "local/pair continuation frozen source changed",
        )
        require(
            local_pair_protocol["scientific_sources_sha256"][
                "experiments/run_local_pair_convergence_extension.py"
            ]
            == digest(
                source.read(
                    local_pair_prefix
                    + "run_local_pair_convergence_extension.py"
                )
            ),
            "local/pair continuation driver snapshot changed",
        )
        require(
            local_pair_aggregate["status"] == "complete",
            "local/pair continuation is incomplete",
        )
        expected_cases = {
            f"principal_{design}_N{size}"
            for design in ("local", "pair")
            for size in (4, 5, 6)
        }
        require(
            set(local_pair_aggregate["cases"]) == expected_cases,
            "local/pair continuation case set changed",
        )
        require(
            local_pair_aggregate["protocol_sha256"]
            == local_pair_protocol_record["protocol_sha256"],
            "local/pair aggregate protocol hash failed",
        )

        case_curves = {name: [] for name in expected_cases}
        case_prefix_errors = {name: [] for name in expected_cases}
        for job in expected_jobs:
            design = job["design"]
            size = job["n_qubits"]
            seed = job["seed"]
            case_name = f"principal_{design}_N{size}"
            relative = (
                "checkpoints/"
                f"{case_name}_seed_{seed}.json"
            )
            row = json.loads(source.read(local_pair_prefix + relative))
            require(
                row["artifact_type"]
                == "local_pair_convergence_extension_checkpoint"
                and row["status"] == "complete"
                and row["job"]
                == {
                    "design": design,
                    "n_qubits": size,
                    "regime": "principal",
                    "seed": seed,
                }
                and row["protocol_sha256"]
                == local_pair_protocol_record["protocol_sha256"],
                f"local/pair checkpoint metadata failed: {relative}",
            )
            curve = row["maximum_trace_distance"]
            require(
                len(curve) == 1201,
                f"local/pair checkpoint length failed: {relative}",
            )
            raw_name = (
                raw_prefix
                + f"{case_name}_seed_{seed}.json"
            )
            raw_bytes = numerical.read(raw_name)
            require(
                digest(raw_bytes)
                == local_pair_protocol[
                    "base_checkpoint_members_sha256"
                ][raw_name],
                f"local/pair frozen checkpoint hash failed: {relative}",
            )
            raw = json.loads(raw_bytes)
            require(
                raw["job"]
                == {
                    "design": design,
                    "n_qubits": size,
                    "regime": "principal",
                    "seed": seed,
                }
                and len(raw["max_trace_distance"]) == 801
                and row["input_prefix_sha256"] == raw["input_sha256"],
                f"local/pair frozen metadata failed: {relative}",
            )
            prefix_error = max(
                abs(a - b)
                for a, b in zip(
                    curve[:801],
                    raw["max_trace_distance"],
                    strict=True,
                )
            )
            require(
                prefix_error
                <= local_pair_protocol["prefix_absolute_tolerance"],
                f"local/pair frozen prefix failed: {relative}",
            )
            case_curves[case_name].append(curve)
            case_prefix_errors[case_name].append(prefix_error)

        reconstructed_prefix_errors = {}
        reconstructed_tails = {}
        reconstructed_finals = {}
        for case_name, case in local_pair_aggregate["cases"].items():
            steps = case["step"]
            curve = case["maximum_trace_distance_across_seeds"]
            reconstructed = curve_envelope(case_curves[case_name])
            require(
                len(steps) == 1201
                and steps[0] == 0
                and steps[-1] == 1200
                and len(curve) == len(steps),
                f"local/pair continuation grid invalid: {case_name}",
            )
            require(
                case["lineage_count"] == 8
                and curve == reconstructed,
                f"local/pair aggregate reconstruction failed: {case_name}",
            )
            reconstructed_prefix_errors[case_name] = max(
                case_prefix_errors[case_name]
            )
            reconstructed_tails[case_name] = max(reconstructed[1100:])
            reconstructed_finals[case_name] = reconstructed[-1]
            require(
                reconstructed_tails[case_name]
                <= local_pair_protocol["convergence_gate"],
                f"local/pair continuation gate failed: {case_name}",
            )
        local_pair_validation = local_pair_aggregate["validation"]
        require(
            local_pair_validation["all_convergence_gates_passed"] is True
            and local_pair_validation["per_case_convergence_gate_passed"]
            == {name: True for name in expected_cases}
            and local_pair_validation[
                "per_case_prefix_absolute_error"
            ]
            == reconstructed_prefix_errors
            and local_pair_validation[
                "per_case_tail_maximum_steps_1100_1200"
            ]
            == reconstructed_tails
            and local_pair_validation[
                "per_case_final_maximum_trace_distance"
            ]
            == reconstructed_finals
            and local_pair_validation["maximum_prefix_absolute_error"]
            == max(reconstructed_prefix_errors.values()),
            "local/pair continuation validation summary drifted",
        )
    return aggregate, local_pair_aggregate


def verify_numerical_archive() -> None:
    archive = ROOT / "numerical_evidence.zip"
    with tempfile.TemporaryDirectory(prefix="qrc-reviewer-evidence-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            require(len(names) == len(set(names)), "numerical ZIP has duplicate members")
            for name in names:
                safe_name(name)
            bundle.extractall(destination)
        evidence_root = destination / "collective-loss-numerical-evidence"
        validator = evidence_root / "validate_bundle.py"
        require(validator.is_file(), "numerical archive lacks validate_bundle.py")
        completed = subprocess.run(
            [sys.executable, str(validator)],
            cwd=evidence_root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="")
        require(completed.returncode == 0, "nested numerical validator failed")


def verify_reset_archive() -> dict:
    archive = ROOT / "reset_architecture_evidence.tar.gz"
    prefix = "reset_architecture_replication/"
    expected_payloads = {
        "initial_state_audit.csv",
        "ordered_seeds.json",
        "protocol.json",
        "strict_washout_arrays.npz",
        "strict_washout_lag_capacities.csv",
        "strict_washout_scores.csv",
        "strict_washout_summary.json",
    }
    expected_members = {
        prefix.rstrip("/"),
        prefix + "SHA256SUMS.txt",
        *{prefix + relative for relative in expected_payloads},
    }
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "reset TAR has duplicate members")
        require(set(names) == expected_members, "reset TAR file set mismatch")
        for member in members:
            safe_name(member.name)
            require(
                member.isdir() or member.isfile(),
                f"reset TAR has non-regular member: {member.name}",
            )
        files = {member.name: member for member in members if member.isfile()}

        checksum_file = bundle.extractfile(files[prefix + "SHA256SUMS.txt"])
        require(checksum_file is not None, "reset checksum manifest is unreadable")
        checksums = {}
        for number, line in enumerate(
            checksum_file.read().decode().splitlines(),
            1,
        ):
            parts = line.split("  ", 1)
            require(
                len(parts) == 2,
                f"reset checksum manifest:{number}: malformed row",
            )
            sha, relative = parts
            safe_name(relative)
            require(
                relative not in checksums,
                f"reset checksum manifest has duplicate row: {relative}",
            )
            checksums[relative] = sha
        require(
            set(checksums) == expected_payloads,
            "reset checksum manifest file set mismatch",
        )
        for relative, expected_sha in checksums.items():
            payload = bundle.extractfile(files[prefix + relative])
            require(payload is not None, f"reset payload is unreadable: {relative}")
            require(
                digest(payload.read()) == expected_sha,
                f"reset payload checksum mismatch: {relative}",
            )

        protocol_file = bundle.extractfile(files[prefix + "protocol.json"])
        summary_file = bundle.extractfile(
            files[prefix + "strict_washout_summary.json"]
        )
        require(
            protocol_file is not None and summary_file is not None,
            "reset protocol or summary is unreadable",
        )
        protocol_record = json.load(protocol_file)
        summary = json.load(summary_file)

    protocol = protocol_record["protocol"]
    seeds = protocol_record["ordered_seeds"]
    require(
        len(seeds) == 16 and len(set(seeds)) == 16,
        "reset paired seed ledger changed",
    )
    require(
        protocol["n_qubits"] == 5
        and protocol["washout"] == 800
        and protocol["train_len"] == 600
        and protocol["test_len"] == 400
        and protocol["gamma"] == 1.0
        and protocol["structural_budget"]["local"] == 80.0
        and protocol["structural_budget"]["collective"] == 80.0,
        "reset protocol changed",
    )
    stm = summary["stm_collective_vs_local"]
    narma = summary["narma10_local_minus_collective"]
    require(
        stm["n"] == narma["n"] == 16
        and stm["wins"] == narma["wins"] == 16
        and stm["ci95"][0] > 0
        and narma["ci95"][0] > 0,
        "reset paired replication gate failed",
    )
    require(
        abs(stm["mean_difference"] - 1.7734822613589252) <= 5e-7
        and abs(narma["mean_difference"] - 0.182040310840414) <= 5e-7,
        "reset audited mean effects changed",
    )
    initial = summary["initial_state_audit"]
    require(
        initial["n_rows"] == 64
        and initial["worst_max_score_spread"] < 7e-7
        and initial["worst_trace_distance_after_800_inputs"] < 1.4e-14,
        "reset initial-state audit gate failed",
    )
    return summary


def verify_phase_direction_archive() -> dict:
    archive = ROOT / "phase_direction_confirmatory_v1_results.tar.gz"
    archive_root = "phase_direction_confirmatory_v1"
    prefix = archive_root + "/"

    def load_json(bundle: tarfile.TarFile, files: dict, relative: str) -> dict:
        payload = bundle.extractfile(files[prefix + relative])
        require(payload is not None, f"phase payload is unreadable: {relative}")
        value = json.load(payload)
        require(isinstance(value, dict), f"phase JSON is not an object: {relative}")
        return value

    def verify_self_hash(value: dict, field: str, relative: str) -> None:
        unhashed = dict(value)
        require(field in unhashed, f"phase self-hash is missing: {relative}:{field}")
        stored = unhashed.pop(field)
        require(
            stored == json_digest(unhashed),
            f"phase self-hash mismatch: {relative}:{field}",
        )

    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), "phase TAR has duplicate members")
        for member in members:
            safe_name(member.name)
            require(
                member.isdir() or member.isfile(),
                f"phase TAR has non-regular member: {member.name}",
            )
            require(
                member.mtime == 0
                and member.uid == 0
                and member.gid == 0
                and member.uname == ""
                and member.gname == ""
                and member.mode == (0o755 if member.isdir() else 0o644),
                f"phase TAR metadata is not normalized: {member.name}",
            )
        directories = {member.name for member in members if member.isdir()}
        require(
            directories
            == {
                archive_root,
                prefix + "convergence_checkpoints",
                prefix + "task_checkpoints",
            },
            "phase TAR directory set mismatch",
        )
        files = {member.name: member for member in members if member.isfile()}
        checksum_name = prefix + "SHA256SUMS"
        require(checksum_name in files, "phase TAR lacks SHA256SUMS")
        checksum_file = bundle.extractfile(files[checksum_name])
        require(checksum_file is not None, "phase checksum manifest is unreadable")
        checksums = {}
        manifest_order = []
        for number, line in enumerate(
            checksum_file.read().decode().splitlines(),
            1,
        ):
            parts = line.split("  ", 1)
            require(
                len(parts) == 2,
                f"phase checksum manifest:{number}: malformed row",
            )
            sha, relative = parts
            path = safe_name(relative)
            require(
                path.as_posix() == relative
                and relative != "SHA256SUMS"
                and len(sha) == 64
                and sha == sha.lower()
                and all(character in "0123456789abcdef" for character in sha),
                f"phase checksum manifest:{number}: invalid row",
            )
            require(
                relative not in checksums,
                f"phase checksum manifest has duplicate row: {relative}",
            )
            checksums[relative] = sha
            manifest_order.append(relative)
        require(
            manifest_order == sorted(manifest_order),
            "phase checksum manifest is not sorted",
        )
        require(
            set(files)
            == {checksum_name, *{prefix + relative for relative in checksums}},
            "phase TAR file set mismatch",
        )
        require(
            names
            == [
                archive_root,
                *[
                    prefix + relative
                    for relative in sorted(
                        {
                            "SHA256SUMS",
                            "convergence_checkpoints",
                            "task_checkpoints",
                            *checksums,
                        }
                    )
                ],
            ],
            "phase TAR member order mismatch",
        )
        for relative, expected_sha in checksums.items():
            payload = bundle.extractfile(files[prefix + relative])
            require(payload is not None, f"phase payload is unreadable: {relative}")
            require(
                digest(payload.read()) == expected_sha,
                f"phase payload checksum mismatch: {relative}",
            )

        protocol = load_json(bundle, files, "protocol.json")
        verify_self_hash(protocol, "protocol_sha256", "protocol.json")
        conditions = protocol["conditions"]
        seeds = protocol["seeds"]
        audit_seeds = protocol["audit_seeds"]
        require(
            protocol["protocol_version"]
            == "phase-direction-confirmatory-v1-2026-08-12"
            and protocol["status"] == "confirmatory_frozen_before_scoring"
            and conditions
            == [
                "path_f0",
                "path_f025",
                "path_f05",
                "path_f075",
                "path_f1",
                "scrambled_r1",
                "scrambled_r2",
                "scrambled_r3",
                "scrambled_r4",
            ]
            and len(seeds) == len(set(seeds)) == protocol["n_seeds"] == 32
            and len(audit_seeds) == len(set(audit_seeds)) == 8
            and set(audit_seeds).issubset(seeds)
            and protocol["primary_condition"] == "path_f0"
            and protocol["primary_reference"] == "path_f1",
            "phase protocol ledger changed",
        )
        expected_tasks = {
            f"task_checkpoints/{condition}__s{seed}.json"
            for condition in conditions
            for seed in seeds
        }
        expected_convergence = {
            f"convergence_checkpoints/{condition}__s{seed}.json"
            for condition in conditions
            for seed in audit_seeds
        }
        expected_root_payloads = {
            "aggregate.json",
            "convergence_summary.json",
            "directions.json",
            "numerical_replay_audit.json",
            "phase_direction_confirmatory.pdf",
            "phase_direction_confirmatory.png",
            "protocol.json",
            "smoke.json",
            "validation_amendment.json",
            "validation_report.json",
        }
        require(
            set(checksums)
            == expected_root_payloads | expected_tasks | expected_convergence,
            "phase evidence membership changed",
        )

        protocol_sha = protocol["protocol_sha256"]
        source_sha = protocol["source_environment_sha256"]
        convergence_hashes = {}
        for relative in sorted(expected_convergence):
            row = load_json(bundle, files, relative)
            stem = PurePosixPath(relative).stem
            condition, seed_text = stem.rsplit("__s", 1)
            verify_self_hash(row, "checkpoint_sha256", relative)
            require(
                row["artifact_type"] == "phase_direction_convergence"
                and row["status"] == "complete"
                and row["condition"] == condition
                and row["seed"] == int(seed_text)
                and row["protocol_sha256"] == protocol_sha
                and row["source_environment_sha256"] == source_sha
                and row["all_gates_passed"] is True,
                f"phase convergence checkpoint failed: {relative}",
            )
            convergence_hashes[stem] = row["checkpoint_sha256"]

        task_hashes = {}
        for relative in sorted(expected_tasks):
            row = load_json(bundle, files, relative)
            stem = PurePosixPath(relative).stem
            condition, seed_text = stem.rsplit("__s", 1)
            verify_self_hash(row, "checkpoint_sha256", relative)
            require(
                row["artifact_type"] == "phase_direction_task"
                and row["status"] == "complete"
                and row["condition"] == condition
                and row["seed"] == int(seed_text)
                and row["protocol_sha256"] == protocol_sha
                and row["source_environment_sha256"] == source_sha
                and row["feature_shape"] == [1000, 45],
                f"phase task checkpoint failed: {relative}",
            )
            task_hashes[stem] = row["checkpoint_sha256"]

        convergence = load_json(bundle, files, "convergence_summary.json")
        verify_self_hash(convergence, "summary_sha256", "convergence_summary.json")
        require(
            convergence["status"] == "complete"
            and convergence["all_gates_passed"] is True
            and convergence["n_expected"] == convergence["n_complete"] == 72
            and convergence["failed_jobs"] == []
            and convergence["protocol_sha256"] == protocol_sha
            and convergence["checkpoint_sha256s"] == convergence_hashes,
            "phase convergence summary failed",
        )

        aggregate = load_json(bundle, files, "aggregate.json")
        verify_self_hash(aggregate, "aggregate_sha256", "aggregate.json")
        primary = aggregate["confirmatory_primary"]
        generality = aggregate["gated_zero_overlap_generality"]
        require(
            aggregate["artifact_type"] == "phase_direction_confirmatory_aggregate"
            and aggregate["status"] == "complete"
            and aggregate["protocol_sha256"] == protocol_sha
            and aggregate["source_environment_sha256"] == source_sha
            and aggregate["convergence_summary_sha256"]
            == convergence["summary_sha256"]
            and aggregate["n_conditions"] == 9
            and aggregate["n_seeds"] == 32
            and aggregate["n_task_checkpoints"] == 288
            and aggregate["task_checkpoint_sha256s"] == task_hashes
            and aggregate["pilot_scores_included"] is False
            and primary["n"] == primary["wins"] == 32
            and primary["ci95_student_t"][0] > 0
            and generality["n"] == generality["wins"] == 32
            and generality["ci95_student_t"][0] > 0
            and generality["gatekeeping_rejects_at_0.05"] is True,
            "phase aggregate gate failed",
        )

        replay = load_json(bundle, files, "numerical_replay_audit.json")
        verify_self_hash(replay, "replay_audit_sha256", "numerical_replay_audit.json")
        replay_cells = {
            f"{row['condition']}__s{row['seed']}"
            for row in replay["rows"]
            if row["all_gates_passed"] is True
        }
        require(
            replay["all_gates_passed"] is True
            and replay["n_expected"] == replay["n_complete"] == 72
            and replay["protocol_sha256"] == protocol_sha
            and replay_cells == set(convergence_hashes),
            "phase numerical replay audit failed",
        )

        amendment = load_json(bundle, files, "validation_amendment.json")
        verify_self_hash(amendment, "amendment_sha256", "validation_amendment.json")
        require(
            amendment["protocol_sha256"] == protocol_sha
            and amendment["scientific_protocol_changed"] is False
            and amendment["seeds_conditions_inference_or_scores_changed"] is False,
            "phase validation amendment failed",
        )

        report = load_json(bundle, files, "validation_report.json")
        verify_self_hash(report, "validation_report_sha256", "validation_report.json")
        require(
            report["status"]
            == "validated_confirmatory_result_with_numerical_replay_amendment"
            and report["protocol_sha256"] == protocol_sha
            and report["source_environment_sha256"] == source_sha
            and report["aggregate_sha256"] == aggregate["aggregate_sha256"]
            and report["convergence_summary_sha256"]
            == convergence["summary_sha256"]
            and report["n_task_checkpoints"] == 288
            and report["n_convergence_checkpoints"] == 72
            and report["n_numerical_replays"] == 72
            and report["all_convergence_gates_pass"] is True
            and report["all_numerical_replay_gates_pass"] is True
            and report["all_pairing_hashes_match"] is True
            and report["pilot_scores_included"] is False
            and report["primary"] == primary
            and report["gated_zero_overlap_generality"] == generality,
            "phase validation report failed",
        )
    return {
        "primary_mean_difference": primary["mean"],
        "primary_ci95": primary["ci95_student_t"],
        "primary_wins": primary["wins"],
    }


def verify_rank_one_orientation_archive() -> dict:
    """Verify the N=6 real sign-balanced intervention without third parties."""
    archive = ROOT / "rank_one_orientation_v1_results.tar.gz"
    archive_root = "rank_one_orientation_v1"
    prefix = archive_root + "/"
    conditions = ("drive_orthogonal", "equal_phase")
    expected_protocol_hashes = {
        "5d4d8c53cea9dfabbc5a0416e19097ad63227a2ada59c34bc69fcf9b459bf7a4",
        "8fcbb1c2a6f22677b062cd36c5b24b5c8c0d7f098bc3adb6a67ef85771314d80",
    }
    expected_jump_hashes = {
        "drive_orthogonal": "0b7c501fb88d59968692ac15a3680a182f6743cfa83064b7c5bce6637d851349",
        "equal_phase": "6aa82635a9e142fc2bee177a87deeeae4371d58a488e22af6410c08e4a6bf994",
    }
    t_critical_df23 = 2.0686576104190406
    tolerance = 5e-12

    def close(left: float, right: float, context: str, atol: float = tolerance) -> None:
        require(
            math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol),
            f"rank-one orientation numeric mismatch: {context}",
        )

    def mean_ci(values: list[float]) -> tuple[float, float, list[float]]:
        require(len(values) == 24, "rank-one orientation aggregate is not 24 rows")
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        standard_error = math.sqrt(variance / len(values))
        half_width = t_critical_df23 * standard_error
        return mean, standard_error, [mean - half_width, mean + half_width]

    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        require(
            len(names) == len(set(names)),
            "rank-one orientation TAR has duplicate members",
        )
        require(
            archive_root in names,
            "rank-one orientation TAR lacks its declared root",
        )
        for member in members:
            path = safe_name(member.name)
            require(
                path.parts[0] == archive_root,
                f"rank-one orientation TAR has unexpected root: {member.name}",
            )
            require(
                member.isdir() or member.isfile(),
                f"rank-one orientation TAR has non-regular member: {member.name}",
            )
            require(
                member.mtime == 0
                and member.uid == 0
                and member.gid == 0
                and member.uname == ""
                and member.gname == ""
                and member.mode == (0o755 if member.isdir() else 0o644),
                f"rank-one orientation TAR metadata is not normalized: {member.name}",
            )
        files = {member.name: member for member in members if member.isfile()}
        ledger_name = prefix + "SHA256SUMS"
        require(ledger_name in files, "rank-one orientation TAR lacks SHA256SUMS")
        ledger_file = bundle.extractfile(files[ledger_name])
        require(ledger_file is not None, "rank-one orientation ledger is unreadable")
        checksums = {}
        order = []
        for number, line in enumerate(ledger_file.read().decode().splitlines(), 1):
            parts = line.split("  ", 1)
            require(
                len(parts) == 2,
                f"rank-one orientation ledger:{number}: malformed row",
            )
            sha, relative = parts
            path = safe_name(relative)
            require(
                path.as_posix() == relative
                and relative != "SHA256SUMS"
                and len(sha) == 64
                and sha == sha.lower()
                and all(character in "0123456789abcdef" for character in sha),
                f"rank-one orientation ledger:{number}: invalid row",
            )
            require(
                relative not in checksums,
                f"rank-one orientation ledger has duplicate row: {relative}",
            )
            checksums[relative] = sha
            order.append(relative)
        require(order == sorted(order), "rank-one orientation ledger is not sorted")
        require(
            set(files) == {ledger_name, *{prefix + relative for relative in checksums}},
            "rank-one orientation TAR file set differs from its ledger",
        )
        payloads = {}
        for relative, expected_sha in checksums.items():
            source = bundle.extractfile(files[prefix + relative])
            require(source is not None, f"rank-one orientation payload unreadable: {relative}")
            data = source.read()
            require(
                digest(data) == expected_sha,
                f"rank-one orientation checksum mismatch: {relative}",
            )
            payloads[relative] = data

    def load_json(relative: str) -> dict:
        require(relative in payloads, f"rank-one orientation JSON missing: {relative}")
        value = json.loads(payloads[relative])
        require(isinstance(value, dict), f"rank-one orientation JSON is not an object: {relative}")
        return value

    required = {
        "protocol.json",
        "provenance.json",
        "environment.json",
        "validation_report.json",
        "derived/summary.json",
        "derived/per_seed.csv",
        "validate.py",
        *{f"checkpoints/seed_{index:02d}.json" for index in range(24)},
    }
    require(
        required.issubset(payloads),
        "rank-one orientation TAR is missing a required audit payload",
    )
    protocol = load_json("protocol.json")
    provenance = load_json("provenance.json")
    environment = load_json("environment.json")
    report = load_json("validation_report.json")
    summary = load_json("derived/summary.json")
    seeds = protocol["input_and_randomization"]["seeds_in_order"]
    matched = protocol["dissipator"]["matched_invariants"]
    require(
        protocol["schema_version"] == 1
        and protocol["experiment_version"] == "rank-one-orientation-v1-2026-08-12"
        and protocol["status"] == "confirmatory protocol frozen before task trajectories"
        and protocol["coherent_processor"]["n_qubits"] == 6
        and len(seeds) == len(set(seeds)) == 24
        and protocol["conditions"]["equal_phase"]["coefficient_vector"]
        == [1, 1, 1, 1, 1, 1]
        and protocol["conditions"]["drive_orthogonal"]["coefficient_vector"]
        == [1, 1, 1, -1, -1, -1]
        and protocol["conditions"]["equal_phase"][
            "normalized_squared_overlap_with_uniform_drive"
        ]
        == 1.0
        and protocol["conditions"]["drive_orthogonal"][
            "normalized_squared_overlap_with_uniform_drive"
        ]
        == 0.0
        and matched["number_of_jump_operators"] == 1
        and matched["kossakowski_block_rank"] == 1
        and matched["kossakowski_block_nonzero_spectrum"] == [6.0]
        and matched["kossakowski_block_trace"] == 6.0
        and matched["coefficient_magnitudes"] == [1, 1, 1, 1, 1, 1]
        and matched["kossakowski_block_diagonal"] == [1, 1, 1, 1, 1, 1]
        and matched["kossakowski_block_kernel_dimension"] == 5
        and matched["physical_jump_operator_kernel_dimension"] == 20
        and matched["operator_weight_budget_B"] == 192.0
        and protocol["task"]["delays"] == list(range(1, 21))
        and protocol["task"]["training_rows"] == 600
        and protocol["task"]["test_rows"] == 400
        and protocol["task"]["ridge"] == 1e-8
        and protocol["convergence"]["initial_states_for_all_24_pairs"]
        == ["ground", "maximally_mixed"]
        and protocol["convergence"][
            "additional_initial_states_for_seed_indices_0_through_5"
        ]
        == ["fully_excited", "haar_pure"],
        "rank-one orientation stable protocol changed",
    )

    rows = []
    raw_hashes = {}
    differences = []
    absolute = {condition: [] for condition in conditions}
    delay_differences = [[] for _ in range(20)]
    metric_differences = {
        metric: []
        for metric in (
            "feature_space_effective_rank",
            "leading_singular_energy_fraction",
            "long_lag_energy_fraction",
            "response_lag_centroid",
        )
    }
    worst_trace = {condition: 0.0 for condition in conditions}
    worst_feature = {condition: 0.0 for condition in conditions}
    for index, seed in enumerate(seeds):
        relative = f"checkpoints/seed_{index:02d}.json"
        row = load_json(relative)
        rows.append(row)
        raw_hashes[f"seed_{index:02d}.json"] = checksums[relative]
        unhashed = dict(row)
        stored_payload_sha = unhashed.pop("payload_sha256", None)
        require(
            stored_payload_sha == json_digest(unhashed),
            f"rank-one orientation checkpoint payload hash failed: {relative}",
        )
        require(
            row["seed_index"] == index
            and row["seed"] == seed
            and row["version"] == protocol["experiment_version"]
            and row["protocol_sha256"] in expected_protocol_hashes
            and row["full_four_state_audit"] is (index < 6)
            and row["convergence"]["both_conditions_passed"] is True
            and row["convergence"]["selected_common_washout"] == 800,
            f"rank-one orientation checkpoint identity failed: {relative}",
        )
        for condition in conditions:
            require(
                row["reservoirs"][condition]["budget"] == 192.0
                and row["reservoirs"][condition]["jump_sha256"]
                == expected_jump_hashes[condition],
                f"rank-one orientation channel metadata failed: {relative}:{condition}",
            )
            audits = row["convergence"]["audits"][condition]
            require(
                set(audits) == {"800"},
                f"rank-one orientation convergence horizon changed: {relative}:{condition}",
            )
            audit = audits["800"]
            pairwise = audit["pairwise"]
            require(
                audit["passed"] is True and len(pairwise) == (6 if index < 6 else 1),
                f"rank-one orientation initial-state audit failed: {relative}:{condition}",
            )
            maximum_trace = max(item["trace_distance"] for item in pairwise)
            maximum_feature = max(item["max_feature_distance"] for item in pairwise)
            close(maximum_trace, audit["maximum_trace_distance"], f"{relative}:{condition}:trace", 1e-20)
            close(maximum_feature, audit["maximum_feature_distance"], f"{relative}:{condition}:feature", 1e-20)
            require(
                maximum_trace <= 1e-8 and maximum_feature <= 2e-8,
                f"rank-one orientation convergence gate failed: {relative}:{condition}",
            )
            worst_trace[condition] = max(worst_trace[condition], maximum_trace)
            worst_feature[condition] = max(worst_feature[condition], maximum_feature)
            result = row["conditions"][condition]
            capacities = result["stm"]["capacity_by_delay"]
            require(
                len(capacities) == 20
                and all(0.0 <= value <= 1.0 + 1e-12 for value in capacities),
                f"rank-one orientation lag capacity failed: {relative}:{condition}",
            )
            close(
                math.fsum(capacities),
                result["stm"]["total_capacity"],
                f"{relative}:{condition}:STM sum",
                1e-12,
            )
            energy = result["kernel"]["normalized_lag_energy"]
            require(
                len(energy) == 20 and all(value >= 0.0 for value in energy),
                f"rank-one orientation response energy failed: {relative}:{condition}",
            )
            close(math.fsum(energy), 1.0, f"{relative}:{condition}:energy sum", 1e-12)
            close(
                math.fsum((lag + 1) * value for lag, value in enumerate(energy)),
                result["kernel"]["response_lag_centroid"],
                f"{relative}:{condition}:response centroid",
                1e-12,
            )
            close(
                math.fsum(energy[9:]),
                result["kernel"]["long_lag_energy_fraction"],
                f"{relative}:{condition}:long-lag energy",
                1e-12,
            )
            absolute[condition].append(result["stm"]["total_capacity"])
        effect = (
            row["conditions"]["equal_phase"]["stm"]["total_capacity"]
            - row["conditions"]["drive_orthogonal"]["stm"]["total_capacity"]
        )
        close(effect, row["stm_equal_minus_orthogonal"], f"{relative}:paired STM")
        differences.append(effect)
        for delay in range(20):
            delay_differences[delay].append(
                row["conditions"]["equal_phase"]["stm"]["capacity_by_delay"][delay]
                - row["conditions"]["drive_orthogonal"]["stm"]["capacity_by_delay"][delay]
            )
        for metric in metric_differences:
            metric_differences[metric].append(
                row["conditions"]["equal_phase"]["kernel"][metric]
                - row["conditions"]["drive_orthogonal"]["kernel"][metric]
            )

    require(
        {row["protocol_sha256"] for row in rows} == expected_protocol_hashes,
        "rank-one orientation protocol-hash provenance changed",
    )
    mean, standard_error, ci95 = mean_ci(differences)
    primary = report["primary_equal_minus_orthogonal"]
    summary_primary = summary["stm"]["paired_equal_minus_orthogonal"]
    require(
        len(differences) == 24
        and sum(value > 0 for value in differences) == 24
        and primary["n"] == primary["wins_positive"] == 24
        and primary["losses_negative"] == primary["ties"] == 0
        and primary["exact_sign_test_p_two_sided"] == 2 / (2 ** 24)
        and ci95[0] > 0,
        "rank-one orientation confirmatory decision gate failed",
    )
    for label, record in (("report", primary), ("summary", summary_primary)):
        close(record["mean"], mean, f"primary mean:{label}")
        close(record["standard_error"], standard_error, f"primary standard error:{label}")
        close(record["ci95"][0], ci95[0], f"primary CI low:{label}")
        close(record["ci95"][1], ci95[1], f"primary CI high:{label}")
    close(mean, 2.271492573239316, "declared primary mean")
    close(ci95[0], 1.825887561894783, "declared primary CI low")
    close(ci95[1], 2.7170975845838488, "declared primary CI high")

    for condition in conditions:
        observed_mean = math.fsum(absolute[condition]) / 24
        close(
            observed_mean,
            summary["stm"][condition]["mean"],
            f"absolute STM mean:{condition}",
        )
    close(summary["stm"]["equal_phase"]["mean"], 13.62584513645472, "equal-phase STM")
    close(summary["stm"]["drive_orthogonal"]["mean"], 11.354352563215405, "orthogonal STM")
    close(
        summary["stm"]["relative_gain_ratio_of_means_percent"],
        20.00547860913046,
        "relative gain",
    )
    for delay, values in enumerate(delay_differences, 1):
        delay_mean, _, delay_ci = mean_ci(values)
        stored = summary["stm"]["lag_resolved"][delay - 1]
        require(stored["delay"] == delay, f"rank-one orientation lag order changed: {delay}")
        close(delay_mean, stored["paired_difference_mean"], f"lag {delay} mean")
        close(delay_ci[0], stored["paired_difference_ci95_low"], f"lag {delay} CI low")
        close(delay_ci[1], stored["paired_difference_ci95_high"], f"lag {delay} CI high")
        require(delay_mean > 0 and delay_ci[0] > 0, f"rank-one orientation lag gate failed: {delay}")
    for metric, values in metric_differences.items():
        metric_mean, _, metric_ci = mean_ci(values)
        stored = summary["kernel"][metric]["paired_equal_minus_orthogonal"]
        close(metric_mean, stored["mean"], f"response diagnostic mean:{metric}")
        close(metric_ci[0], stored["ci95"][0], f"response diagnostic CI low:{metric}")
        close(metric_ci[1], stored["ci95"][1], f"response diagnostic CI high:{metric}")
    require(
        all(
            record["pearson_p"] > 0.05 and record["spearman_p"] > 0.05
            for record in summary["association"].values()
        ),
        "rank-one orientation diagnostic-association scope changed",
    )

    csv_rows = list(
        csv.DictReader(io.StringIO(payloads["derived/per_seed.csv"].decode("utf-8")))
    )
    require(len(csv_rows) == 24, "rank-one orientation per-seed table is not 24 rows")
    for index, (csv_row, raw_row) in enumerate(zip(csv_rows, rows)):
        require(
            int(csv_row["seed_index"]) == index
            and int(csv_row["seed"]) == seeds[index]
            and csv_row["convergence_passed"] == "True"
            and int(csv_row["washout"]) == 800,
            f"rank-one orientation per-seed row identity failed: {index}",
        )
        close(
            float(csv_row["stm_equal_minus_orthogonal"]),
            raw_row["stm_equal_minus_orthogonal"],
            f"per-seed aggregate STM:{index}",
        )

    invariants = report["channel_invariants"]
    for condition in conditions:
        require(
            invariants[condition]["kossakowski_rank"] == 1
            and invariants[condition]["kossakowski_nonzero_eigenvalues"] == [6.0]
            and invariants[condition]["kossakowski_trace"] == 6.0
            and invariants[condition]["kossakowski_kernel_dimension"] == 5
            and invariants[condition]["physical_jump_kernel_dimension"] == 20
            and invariants[condition]["coefficient_magnitudes"] == [1.0] * 6
            and invariants[condition]["kossakowski_diagonal"] == [1.0] * 6
            and invariants[condition]["operator_weight_budget_B"] == 192.0
            and invariants[condition]["jump_sha256"] == expected_jump_hashes[condition],
            f"rank-one orientation invariant report failed: {condition}",
        )
    require(
        report["schema_version"] == 1
        and report["status"] == "validated"
        and report["experiment_version"] == protocol["experiment_version"]
        and report["pair_count"] == report["ground_mixed_audit_pair_count"] == 24
        and report["additional_four_state_audit_pair_count"] == 6
        and report["all_convergence_gates_passed"] is True
        and report["protocol_sha256"] == json_digest(protocol)
        and report["provenance_sha256"] == json_digest(provenance)
        and report["environment_sha256"] == json_digest(environment)
        and report["raw_checkpoint_sha256s"] == raw_hashes
        and report["raw_checkpoint_set_sha256"] == json_digest(raw_hashes)
        and "does not establish universal" in report["claim_boundary"],
        "rank-one orientation validation report changed",
    )
    for condition in conditions:
        close(
            worst_trace[condition],
            summary["validation"]["worst_trace_distance"][condition],
            f"worst trace distance:{condition}",
            1e-20,
        )
        close(
            worst_feature[condition],
            summary["validation"]["worst_feature_distance"][condition],
            f"worst feature distance:{condition}",
            1e-20,
        )
    return {
        "mean_difference": mean,
        "ci95": ci95,
        "wins": 24,
        "equal_phase_mean": summary["stm"]["equal_phase"]["mean"],
        "orthogonal_mean": summary["stm"]["drive_orthogonal"]["mean"],
    }


def main() -> int:
    try:
        verify_manifest()
        aggregate, local_pair_aggregate = verify_source()
        verify_numerical_archive()
        reset_summary = verify_reset_archive()
        phase_summary = verify_phase_direction_archive()
        orientation_summary = verify_rank_one_orientation_archive()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    final_distance = aggregate["validation"]["final_maximum_trace_distance"]
    local_pair_tail = max(
        max(case["maximum_trace_distance_across_seeds"][1100:])
        for case in local_pair_aggregate["cases"].values()
    )
    print("complete reviewer bundle valid")
    print("  manuscript and source checksums: passed")
    print("  full numerical evidence validator: passed")
    print("  explicit protocol and seed manifest: passed")
    print("  N=5 convergence: 8 lineages, 4 initial states, 1200 inputs")
    print(f"  final worst trace distance: {final_distance:.3e}")
    print("  local/pair convergence: 48 lineages through 1200 inputs")
    print(f"  worst local/pair tail trace distance: {local_pair_tail:.3e}")
    print("  reset encoding: 16 paired reservoirs, 16/16 favorable on both tasks")
    print(
        "  reset worst trace distance after 800 inputs: "
        f"{reset_summary['initial_state_audit']['worst_trace_distance_after_800_inputs']:.3e}"
    )
    print(
        "  phase direction: 32 paired reservoirs, "
        f"{phase_summary['primary_wins']}/32 favorable"
    )
    print(
        "  phase-direction STM difference: "
        f"{phase_summary['primary_mean_difference']:.6f} "
        f"(95% CI {phase_summary['primary_ci95']})"
    )
    print(
        "  rank-one orientation: 24 paired N=6 reservoirs, "
        f"{orientation_summary['wins']}/24 favorable"
    )
    print(
        "  equal-phase-minus-sign-balanced STM difference: "
        f"{orientation_summary['mean_difference']:.6f} "
        f"(95% CI {orientation_summary['ci95']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''.encode("utf-8")


def collect_payloads() -> list[Payload]:
    payloads = [Payload("README.txt", _readme())]
    for name, path in INPUTS.items():
        _safe_name(name)
        if not path.is_file():
            raise ReviewerBundleError(f"missing required input: {path}")
        payloads.append(Payload(name, path.read_bytes()))
    payloads.append(Payload("validate_complete_bundle.py", _embedded_validator()))
    payloads.sort(key=lambda item: item.name)
    manifest = "".join(
        f"{sha256_bytes(payload.data)}  {payload.name}\n"
        for payload in payloads
    ).encode("utf-8")
    payloads.append(Payload("SHA256SUMS.txt", manifest))
    payloads.sort(key=lambda item: item.name)
    return payloads


def build(output: Path = DEFAULT_OUTPUT) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payloads = collect_payloads()
    with tempfile.NamedTemporaryFile(
        prefix=output.name + ".",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            root_info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/", ZIP_TIMESTAMP)
            root_info.external_attr = (0o755 | 0o040000) << 16
            bundle.writestr(root_info, b"")
            for payload in payloads:
                info = zipfile.ZipInfo(
                    f"{ARCHIVE_ROOT}/{payload.name}",
                    ZIP_TIMESTAMP,
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = FILE_MODE << 16
                bundle.writestr(info, payload.data, compresslevel=9)
        temporary_path.replace(output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return output


def verify(archive: Path = DEFAULT_OUTPUT) -> None:
    archive = archive.resolve()
    if not archive.is_file():
        raise ReviewerBundleError(f"missing archive: {archive}")
    with tempfile.TemporaryDirectory(prefix="qrc-complete-reviewer-") as temporary:
        destination = Path(temporary)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise ReviewerBundleError("archive contains duplicate members")
            for name in names:
                path = PurePosixPath(name)
                _safe_name(name)
                if path.parts[0] != ARCHIVE_ROOT:
                    raise ReviewerBundleError(
                        f"unexpected archive root: {path.parts[0]}"
                    )
            bundle.extractall(destination)
        root = destination / ARCHIVE_ROOT
        validator = root / "validate_complete_bundle.py"
        completed = subprocess.run(
            [sys.executable, str(validator)],
            cwd=root,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(completed.stdout, end="")
        if completed.returncode:
            raise ReviewerBundleError(
                f"embedded validator failed with exit code {completed.returncode}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if args.command == "build":
            output = build(args.output)
            print(f"built: {output}")
            print(f"sha256: {sha256_bytes(output.read_bytes())}")
            verify(output)
        else:
            verify(args.archive)
            print(f"verified: {args.archive.resolve()}")
    except (OSError, ReviewerBundleError, zipfile.BadZipFile) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

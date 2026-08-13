#!/usr/bin/env python3
"""Build deterministic, sidecar-free archives for the raw result bundles.

The archive format is deliberately simple: each tarball has one top-level
directory, contains only directories and regular files, and uses normalized tar
metadata.  Gzip timestamps are also fixed, so repeated builds from unchanged
inputs produce byte-identical output.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results"

PROTOCOL_BUNDLES = (
    ("final_protocol", "final_protocol_results.tar.gz"),
    ("review_protocol", "review_protocol_results.tar.gz"),
    ("experiment1_finite_size_v2", "experiment1_finite_size_v2_results.tar.gz"),
    (
        "reset_architecture_replication",
        "reset_architecture_replication_results.tar.gz",
    ),
    (
        "continuous_drive_narma_washout_v1",
        "continuous_drive_narma_washout_v1_results.tar.gz",
    ),
)
PHASE_DIRECTION_ROOT = "phase_direction_confirmatory_v1"
PHASE_DIRECTION_SOURCE = (
    ROOT / "paper" / "evidence" / PHASE_DIRECTION_ROOT
)
PHASE_DIRECTION_ARCHIVE_NAME = (
    "phase_direction_confirmatory_v1_results.tar.gz"
)
PHASE_DIRECTION_CHECKSUM_NAME = "SHA256SUMS"
PHASE_DIRECTION_CONDITIONS = (
    "path_f0",
    "path_f025",
    "path_f05",
    "path_f075",
    "path_f1",
    "scrambled_r1",
    "scrambled_r2",
    "scrambled_r3",
    "scrambled_r4",
)
PHASE_DIRECTION_ROOT_PAYLOADS = frozenset(
    {
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
)
RANK_ONE_ORIENTATION_ROOT = "rank_one_orientation_v1"
RANK_ONE_ORIENTATION_SOURCE = (
    ROOT / "paper" / "evidence" / RANK_ONE_ORIENTATION_ROOT
)
RANK_ONE_ORIENTATION_ARCHIVE_NAME = (
    "rank_one_orientation_v1_results.tar.gz"
)
RANK_ONE_ORIENTATION_CHECKSUM_NAME = "SHA256SUMS"
EXCLUDED_TREE_MEMBERS = {
    "experiment1_finite_size_v2": frozenset({"resume_production.zsh"}),
}
GROUPED_ROOT = "grouped_measurement"
GROUPED_DATA_NAME = "phys_shots.json"
GROUPED_MANIFEST_NAME = "provenance.json"
GROUPED_ARCHIVE_NAME = "grouped_measurement_results.tar.gz"

RESET_ARCHITECTURE_ROOT = "reset_architecture_replication"
RESET_ARCHITECTURE_CHECKSUM_NAME = "SHA256SUMS.txt"
RESET_ARCHITECTURE_FILES = (
    RESET_ARCHITECTURE_CHECKSUM_NAME,
    "initial_state_audit.csv",
    "ordered_seeds.json",
    "protocol.json",
    "strict_washout_arrays.npz",
    "strict_washout_lag_capacities.csv",
    "strict_washout_scores.csv",
    "strict_washout_summary.json",
)

NORMALIZED_MTIME = 0
NORMALIZED_UID = 0
NORMALIZED_GID = 0
DIRECTORY_MODE = 0o755
FILE_MODE = 0o644


@dataclass(frozen=True)
class ArchiveEntry:
    """One normalized archive member."""

    name: str
    data: bytes | None

    @property
    def is_directory(self) -> bool:
        return self.data is None


@dataclass(frozen=True)
class ArchiveSummary:
    """Validated output metadata printed after a successful build."""

    path: Path
    member_count: int
    file_count: int
    excluded_count: int
    sha256: str


def is_forbidden_sidecar(path: PurePosixPath) -> bool:
    """Return whether *path* is Apple metadata rather than scientific data."""
    return any(
        part == ".DS_Store" or part == "__MACOSX" or part.startswith("._")
        for part in path.parts
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256_manifest(data: bytes, context: str) -> dict[str, str]:
    """Parse the strict ``sha256sum`` format used by the reset result bundle."""
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"checksum manifest is not UTF-8: {context}") from error
    if not lines:
        raise ValueError(f"checksum manifest is empty: {context}")

    records: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        if (
            separator != "  "
            or not relative
            or PurePosixPath(relative).name != relative
            or len(digest) != 64
        ):
            raise ValueError(f"invalid checksum line in {context}: {line!r}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                f"invalid SHA-256 digest in {context}: {digest!r}"
            ) from error
        if digest != digest.lower():
            raise ValueError(f"SHA-256 digest must be lowercase in {context}")
        if relative in records:
            raise ValueError(
                f"duplicate checksum member in {context}: {relative}"
            )
        records[relative] = digest
    return records


def parse_tree_sha256_manifest(data: bytes, context: str) -> dict[str, str]:
    """Parse a sorted SHA-256 manifest whose members may be nested paths."""
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"checksum manifest is not UTF-8: {context}") from error
    if not lines:
        raise ValueError(f"checksum manifest is empty: {context}")

    records: dict[str, str] = {}
    for line in lines:
        digest, separator, relative = line.partition("  ")
        path = PurePosixPath(relative)
        if (
            separator != "  "
            or not relative
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or "\x00" in relative
            or ":" in path.parts[0]
            or relative != path.as_posix()
            or len(digest) != 64
        ):
            raise ValueError(f"invalid checksum line in {context}: {line!r}")
        try:
            int(digest, 16)
        except ValueError as error:
            raise ValueError(
                f"invalid SHA-256 digest in {context}: {digest!r}"
            ) from error
        if digest != digest.lower():
            raise ValueError(f"SHA-256 digest must be lowercase in {context}")
        if is_forbidden_sidecar(path):
            raise ValueError(f"forbidden sidecar in {context}: {relative}")
        if relative in {
            PHASE_DIRECTION_CHECKSUM_NAME,
            RANK_ONE_ORIENTATION_CHECKSUM_NAME,
        }:
            raise ValueError(
                f"checksum manifest may not hash itself in {context}"
            )
        if relative in records:
            raise ValueError(
                f"duplicate checksum member in {context}: {relative}"
            )
        records[relative] = digest
    if list(records) != sorted(records):
        raise ValueError(f"checksum manifest is not sorted: {context}")
    return records


def canonical_json_sha256(value: object) -> str:
    """Hash JSON using the canonical form used by the frozen experiment."""
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _load_json_payload(
    payloads: dict[str, bytes], relative: str, context: str
) -> dict:
    try:
        value = json.loads(payloads[relative])
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid or missing JSON payload in {context}: {relative}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON payload is not an object in {context}: {relative}")
    return value


def _require_self_hash(
    payload: dict, field: str, relative: str, context: str
) -> None:
    unhashed = dict(payload)
    try:
        stored = unhashed.pop(field)
    except KeyError as error:
        raise ValueError(
            f"missing self-hash field in {context}: {relative}:{field}"
        ) from error
    actual = canonical_json_sha256(unhashed)
    if stored != actual:
        raise ValueError(
            f"self-hash mismatch in {context}: {relative}:{field}"
        )


def validate_phase_direction_payloads(
    payloads: dict[str, bytes], context: str
) -> dict:
    """Validate the sealed phase-direction evidence without simulation code."""
    protocol = _load_json_payload(payloads, "protocol.json", context)
    conditions = protocol.get("conditions")
    seeds = protocol.get("seeds")
    audit_seeds = protocol.get("audit_seeds")
    if (
        protocol.get("protocol_version")
        != "phase-direction-confirmatory-v1-2026-08-12"
        or protocol.get("status") != "confirmatory_frozen_before_scoring"
        or conditions != list(PHASE_DIRECTION_CONDITIONS)
        or not isinstance(seeds, list)
        or len(seeds) != 32
        or len(set(seeds)) != 32
        or protocol.get("n_seeds") != 32
        or not isinstance(audit_seeds, list)
        or len(audit_seeds) != 8
        or len(set(audit_seeds)) != 8
        or not set(audit_seeds).issubset(seeds)
        or protocol.get("primary_condition") != "path_f0"
        or protocol.get("primary_reference") != "path_f1"
    ):
        raise ValueError(f"phase-direction protocol ledger mismatch: {context}")
    _require_self_hash(protocol, "protocol_sha256", "protocol.json", context)

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
    expected_payloads = (
        set(PHASE_DIRECTION_ROOT_PAYLOADS)
        | expected_tasks
        | expected_convergence
    )
    if set(payloads) != expected_payloads:
        missing = sorted(expected_payloads - set(payloads))
        unexpected = sorted(set(payloads) - expected_payloads)
        raise ValueError(
            "phase-direction evidence membership mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    directions = _load_json_payload(payloads, "directions.json", context)
    smoke = _load_json_payload(payloads, "smoke.json", context)
    direction_unhashed = {
        key: value
        for key, value in directions.items()
        if key != "direction_ledger_sha256"
    }
    if (
        protocol.get("directions_sha256")
        != directions.get("direction_ledger_sha256")
        or protocol.get("pre_score_smoke_sha256")
        != canonical_json_sha256(smoke)
        or directions.get("direction_ledger_sha256")
        != canonical_json_sha256(direction_unhashed)
        or [row.get("condition") for row in directions.get("conditions", [])]
        != conditions
        or smoke.get("status") != "passed"
        or smoke.get("n_conditions") != 9
        or smoke.get("n_seeds") != 32
    ):
        raise ValueError(
            f"phase-direction frozen direction/smoke binding failed: {context}"
        )

    protocol_sha256 = protocol["protocol_sha256"]
    source_environment_sha256 = protocol["source_environment_sha256"]
    convergence_hashes: dict[str, str] = {}
    for relative in sorted(expected_convergence):
        row = _load_json_payload(payloads, relative, context)
        filename = PurePosixPath(relative).stem
        condition, seed_text = filename.rsplit("__s", 1)
        if (
            row.get("artifact_type") != "phase_direction_convergence"
            or row.get("status") != "complete"
            or row.get("condition") != condition
            or row.get("seed") != int(seed_text)
            or row.get("protocol_sha256") != protocol_sha256
            or row.get("source_environment_sha256")
            != source_environment_sha256
            or row.get("all_gates_passed") is not True
        ):
            raise ValueError(
                f"phase-direction convergence checkpoint failed: {relative}"
            )
        _require_self_hash(row, "checkpoint_sha256", relative, context)
        convergence_hashes[filename] = row["checkpoint_sha256"]

    task_hashes: dict[str, str] = {}
    for relative in sorted(expected_tasks):
        row = _load_json_payload(payloads, relative, context)
        filename = PurePosixPath(relative).stem
        condition, seed_text = filename.rsplit("__s", 1)
        if (
            row.get("artifact_type") != "phase_direction_task"
            or row.get("status") != "complete"
            or row.get("condition") != condition
            or row.get("seed") != int(seed_text)
            or row.get("protocol_sha256") != protocol_sha256
            or row.get("source_environment_sha256")
            != source_environment_sha256
            or row.get("protocol_version") != protocol["protocol_version"]
            or row.get("feature_shape") != [1000, 45]
        ):
            raise ValueError(f"phase-direction task checkpoint failed: {relative}")
        _require_self_hash(row, "checkpoint_sha256", relative, context)
        task_hashes[filename] = row["checkpoint_sha256"]

    convergence = _load_json_payload(
        payloads, "convergence_summary.json", context
    )
    _require_self_hash(
        convergence,
        "summary_sha256",
        "convergence_summary.json",
        context,
    )
    if (
        convergence.get("status") != "complete"
        or convergence.get("all_gates_passed") is not True
        or convergence.get("n_expected") != 72
        or convergence.get("n_complete") != 72
        or convergence.get("failed_jobs") != []
        or convergence.get("protocol_sha256") != protocol_sha256
        or convergence.get("checkpoint_sha256s") != convergence_hashes
    ):
        raise ValueError(
            f"phase-direction convergence summary failed: {context}"
        )

    aggregate = _load_json_payload(payloads, "aggregate.json", context)
    _require_self_hash(
        aggregate, "aggregate_sha256", "aggregate.json", context
    )
    primary = aggregate.get("confirmatory_primary", {})
    generality = aggregate.get("gated_zero_overlap_generality", {})
    if (
        aggregate.get("artifact_type")
        != "phase_direction_confirmatory_aggregate"
        or aggregate.get("status") != "complete"
        or aggregate.get("protocol_sha256") != protocol_sha256
        or aggregate.get("source_environment_sha256")
        != source_environment_sha256
        or aggregate.get("convergence_summary_sha256")
        != convergence["summary_sha256"]
        or aggregate.get("n_conditions") != 9
        or aggregate.get("n_seeds") != 32
        or aggregate.get("n_task_checkpoints") != 288
        or aggregate.get("task_checkpoint_sha256s") != task_hashes
        or aggregate.get("pilot_scores_included") is not False
        or primary.get("n") != 32
        or primary.get("wins") != 32
        or primary.get("ci95_student_t", [0])[0] <= 0
        or generality.get("n") != 32
        or generality.get("wins") != 32
        or generality.get("ci95_student_t", [0])[0] <= 0
        or generality.get("gatekeeping_rejects_at_0.05") is not True
    ):
        raise ValueError(f"phase-direction aggregate gate failed: {context}")

    replay = _load_json_payload(
        payloads, "numerical_replay_audit.json", context
    )
    _require_self_hash(
        replay,
        "replay_audit_sha256",
        "numerical_replay_audit.json",
        context,
    )
    replay_cells = {
        f"{row.get('condition')}__s{row.get('seed')}"
        for row in replay.get("rows", [])
        if isinstance(row, dict) and row.get("all_gates_passed") is True
    }
    if (
        replay.get("all_gates_passed") is not True
        or replay.get("n_expected") != 72
        or replay.get("n_complete") != 72
        or replay.get("protocol_sha256") != protocol_sha256
        or replay_cells != set(convergence_hashes)
    ):
        raise ValueError(
            f"phase-direction numerical replay audit failed: {context}"
        )

    amendment = _load_json_payload(
        payloads, "validation_amendment.json", context
    )
    _require_self_hash(
        amendment,
        "amendment_sha256",
        "validation_amendment.json",
        context,
    )
    if (
        amendment.get("artifact_type") != "phase_direction_validation_amendment"
        or amendment.get("protocol_sha256") != protocol_sha256
        or amendment.get("scientific_protocol_changed") is not False
        or amendment.get("seeds_conditions_inference_or_scores_changed")
        is not False
    ):
        raise ValueError(
            f"phase-direction validation amendment failed: {context}"
        )

    report = _load_json_payload(payloads, "validation_report.json", context)
    _require_self_hash(
        report,
        "validation_report_sha256",
        "validation_report.json",
        context,
    )
    if (
        report.get("status")
        != "validated_confirmatory_result_with_numerical_replay_amendment"
        or report.get("protocol_sha256") != protocol_sha256
        or report.get("source_environment_sha256")
        != source_environment_sha256
        or report.get("aggregate_sha256") != aggregate["aggregate_sha256"]
        or report.get("convergence_summary_sha256")
        != convergence["summary_sha256"]
        or report.get("n_task_checkpoints") != 288
        or report.get("n_convergence_checkpoints") != 72
        or report.get("n_numerical_replays") != 72
        or report.get("all_convergence_gates_pass") is not True
        or report.get("all_numerical_replay_gates_pass") is not True
        or report.get("all_pairing_hashes_match") is not True
        or report.get("pilot_scores_included") is not False
        or report.get("primary") != primary
        or report.get("gated_zero_overlap_generality") != generality
    ):
        raise ValueError(
            f"phase-direction validation report failed: {context}"
        )

    return {
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "convergence_checkpoints": len(convergence_hashes),
        "primary_mean_difference": primary["mean"],
        "protocol_sha256": protocol_sha256,
        "task_checkpoints": len(task_hashes),
    }


def validate_phase_direction_source(source: Path) -> dict:
    """Validate exact source membership, checksums, and scientific gates."""
    if not source.is_dir():
        raise FileNotFoundError(f"required result directory is missing: {source}")
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        relative = PurePosixPath(path.relative_to(source).as_posix())
        if is_forbidden_sidecar(relative):
            raise ValueError(
                f"forbidden sidecar in phase-direction evidence: {path}"
            )
        if path.is_symlink():
            raise ValueError(
                f"symlinks are not permitted in phase-direction evidence: {path}"
            )
        if not path.is_dir() and not path.is_file():
            raise ValueError(
                f"unsupported phase-direction evidence entry: {path}"
            )
    actual_directories = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_dir()
    }
    if actual_directories != {
        "convergence_checkpoints",
        "task_checkpoints",
    }:
        raise ValueError(
            "phase-direction evidence directory set mismatch: "
            f"actual={sorted(actual_directories)}"
        )

    checksum_path = source / PHASE_DIRECTION_CHECKSUM_NAME
    if not checksum_path.is_file():
        raise FileNotFoundError(
            f"phase-direction checksum manifest is missing: {checksum_path}"
        )
    records = parse_tree_sha256_manifest(
        checksum_path.read_bytes(), str(checksum_path)
    )
    actual_files = {
        path.relative_to(source).as_posix(): path
        for path in source.rglob("*")
        if path.is_file() and path != checksum_path
    }
    if set(records) != set(actual_files):
        missing = sorted(set(records) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(records))
        raise ValueError(
            "phase-direction checksum membership mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    payloads = {}
    for relative, path in actual_files.items():
        data = path.read_bytes()
        if sha256_bytes(data) != records[relative]:
            raise ValueError(
                f"phase-direction checksum mismatch for {relative}"
            )
        payloads[relative] = data
    return validate_phase_direction_payloads(payloads, str(source))


def validate_phase_direction_archive(path: Path) -> dict:
    """Validate the published phase-direction TAR without repository access."""
    prefix = f"{PHASE_DIRECTION_ROOT}/"
    with tarfile.open(path, mode="r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(
                f"phase-direction archive contains duplicate members: {path}"
            )
        for member in members:
            pure_name = PurePosixPath(member.name)
            if pure_name.is_absolute() or ".." in pure_name.parts:
                raise RuntimeError(
                    f"unsafe member path in {path}: {member.name}"
                )
            if is_forbidden_sidecar(pure_name):
                raise RuntimeError(
                    f"forbidden sidecar in {path}: {member.name}"
                )
            expected_mode = DIRECTORY_MODE if member.isdir() else FILE_MODE
            if (
                (not member.isdir() and not member.isfile())
                or member.mtime != NORMALIZED_MTIME
                or member.uid != NORMALIZED_UID
                or member.gid != NORMALIZED_GID
                or member.uname != ""
                or member.gname != ""
                or member.mode != expected_mode
            ):
                raise RuntimeError(
                    f"invalid phase-direction archive metadata: {member.name}"
                )

        files = {member.name: member for member in members if member.isfile()}
        manifest_name = prefix + PHASE_DIRECTION_CHECKSUM_NAME
        if manifest_name not in files:
            raise RuntimeError(
                "phase-direction archive lacks its checksum manifest"
            )
        checksum_file = bundle.extractfile(files[manifest_name])
        if checksum_file is None:
            raise RuntimeError(
                "phase-direction archive checksum manifest is unreadable"
            )
        records = parse_tree_sha256_manifest(
            checksum_file.read(), f"{path}:{manifest_name}"
        )
        expected_files = {
            manifest_name,
            *{prefix + relative for relative in records},
        }
        if set(files) != expected_files:
            missing = sorted(expected_files - set(files))
            unexpected = sorted(set(files) - expected_files)
            raise RuntimeError(
                "phase-direction archive file set mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        expected_directories = {
            PHASE_DIRECTION_ROOT,
            prefix + "convergence_checkpoints",
            prefix + "task_checkpoints",
        }
        actual_directories = {
            member.name for member in members if member.isdir()
        }
        if actual_directories != expected_directories:
            raise RuntimeError(
                "phase-direction archive directory set mismatch: "
                f"actual={sorted(actual_directories)}"
            )
        expected_order = [
            PHASE_DIRECTION_ROOT,
            *[
                prefix + relative
                for relative in sorted(
                    {
                        PHASE_DIRECTION_CHECKSUM_NAME,
                        "convergence_checkpoints",
                        "task_checkpoints",
                        *records,
                    }
                )
            ],
        ]
        if names != expected_order:
            raise RuntimeError(
                f"phase-direction archive member order mismatch: {path}"
            )

        payloads: dict[str, bytes] = {}
        for relative, expected_digest in records.items():
            archived_file = bundle.extractfile(files[prefix + relative])
            if archived_file is None:
                raise RuntimeError(
                    f"phase-direction archive payload is unreadable: {relative}"
                )
            data = archived_file.read()
            if sha256_bytes(data) != expected_digest:
                raise RuntimeError(
                    f"phase-direction archive checksum mismatch: {relative}"
                )
            payloads[relative] = data
    return validate_phase_direction_payloads(payloads, str(path))


def _load_rank_one_orientation_validator():
    """Load the evidence validator from its portable, archive-owned source."""
    import importlib.util
    import sys

    validator_path = RANK_ONE_ORIENTATION_SOURCE / "validate.py"
    if not validator_path.is_file():
        raise FileNotFoundError(
            f"rank-one orientation validator is missing: {validator_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "rank_one_orientation_evidence_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"cannot load rank-one orientation validator: {validator_path}"
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def validate_rank_one_orientation_source(source: Path) -> dict:
    """Validate the sealed N=6 orientation evidence from repository files."""
    if not source.is_dir():
        raise FileNotFoundError(f"required result directory is missing: {source}")
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        relative = PurePosixPath(path.relative_to(source).as_posix())
        if is_forbidden_sidecar(relative):
            raise ValueError(
                f"forbidden sidecar in rank-one orientation evidence: {path}"
            )
        if path.is_symlink():
            raise ValueError(
                f"symlinks are not permitted in rank-one orientation evidence: {path}"
            )
        if not path.is_dir() and not path.is_file():
            raise ValueError(
                f"unsupported rank-one orientation evidence entry: {path}"
            )
    validator = _load_rank_one_orientation_validator()
    records = validator.parse_ledger(source)
    report = validator.build_report(source)
    validator.same_semantics(
        report,
        validator.load_json(source / validator.REPORT_NAME),
        "validation_report",
    )
    return {
        "file_count": len(records) + 1,
        "pair_count": report["pair_count"],
        "wins": report["primary_equal_minus_orthogonal"]["wins_positive"],
        "primary_mean_difference": report["primary_equal_minus_orthogonal"][
            "mean"
        ],
        "primary_ci95": report["primary_equal_minus_orthogonal"]["ci95"],
        "raw_checkpoint_set_sha256": report["raw_checkpoint_set_sha256"],
    }


def validate_rank_one_orientation_archive(path: Path) -> dict:
    """Validate the deterministic N=6 orientation TAR without the repository."""
    prefix = f"{RANK_ONE_ORIENTATION_ROOT}/"
    with tarfile.open(path, mode="r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(
                f"rank-one orientation archive contains duplicate members: {path}"
            )
        for member in members:
            pure_name = PurePosixPath(member.name)
            expected_mode = DIRECTORY_MODE if member.isdir() else FILE_MODE
            if (
                pure_name.is_absolute()
                or ".." in pure_name.parts
                or is_forbidden_sidecar(pure_name)
                or (not member.isdir() and not member.isfile())
                or member.mtime != NORMALIZED_MTIME
                or member.uid != NORMALIZED_UID
                or member.gid != NORMALIZED_GID
                or member.uname != ""
                or member.gname != ""
                or member.mode != expected_mode
            ):
                raise RuntimeError(
                    f"invalid rank-one orientation archive member: {member.name}"
                )
        files = {member.name: member for member in members if member.isfile()}
        ledger_name = prefix + RANK_ONE_ORIENTATION_CHECKSUM_NAME
        ledger_member = files.get(ledger_name)
        if ledger_member is None:
            raise RuntimeError("rank-one orientation archive lacks SHA256SUMS")
        ledger_file = bundle.extractfile(ledger_member)
        if ledger_file is None:
            raise RuntimeError("rank-one orientation checksum ledger is unreadable")
        records = parse_tree_sha256_manifest(
            ledger_file.read(), f"{path}:{ledger_name}"
        )
        expected_files = {ledger_name, *{prefix + relative for relative in records}}
        if set(files) != expected_files:
            missing = sorted(expected_files - set(files))
            unexpected = sorted(set(files) - expected_files)
            raise RuntimeError(
                "rank-one orientation archive file-set mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for relative, expected_digest in records.items():
            archived_file = bundle.extractfile(files[prefix + relative])
            if archived_file is None:
                raise RuntimeError(
                    f"rank-one orientation payload is unreadable: {relative}"
                )
            if sha256_bytes(archived_file.read()) != expected_digest:
                raise RuntimeError(
                    f"rank-one orientation payload checksum mismatch: {relative}"
                )

        temporary_root = Path(
            tempfile.mkdtemp(prefix="rank-one-orientation-validate-")
        )
        try:
            for member in members:
                target = temporary_root.joinpath(*PurePosixPath(member.name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                archived_file = bundle.extractfile(member)
                if archived_file is None:
                    raise RuntimeError(
                        f"rank-one orientation payload is unreadable: {member.name}"
                    )
                target.write_bytes(archived_file.read())
            source = temporary_root / RANK_ONE_ORIENTATION_ROOT
            validator_path = source / "validate.py"
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "archived_rank_one_orientation_validator", validator_path
            )
            if spec is None or spec.loader is None:
                raise RuntimeError("archived rank-one orientation validator is unloadable")
            validator = importlib.util.module_from_spec(spec)
            import sys

            previous = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                spec.loader.exec_module(validator)
            finally:
                sys.dont_write_bytecode = previous
            archived_records = validator.parse_ledger(source)
            report = validator.build_report(source)
            validator.same_semantics(
                report,
                validator.load_json(source / validator.REPORT_NAME),
                "validation_report",
            )
        finally:
            import shutil

            shutil.rmtree(temporary_root)
    return {
        "file_count": len(archived_records) + 1,
        "pair_count": report["pair_count"],
        "wins": report["primary_equal_minus_orthogonal"]["wins_positive"],
        "primary_mean_difference": report["primary_equal_minus_orthogonal"][
            "mean"
        ],
        "primary_ci95": report["primary_equal_minus_orthogonal"]["ci95"],
        "raw_checkpoint_set_sha256": report["raw_checkpoint_set_sha256"],
    }


def validate_reset_architecture_source(source: Path) -> None:
    """Require the exact sealed reset-replication evidence set and checksums."""
    if not source.is_dir():
        raise FileNotFoundError(f"required result directory is missing: {source}")
    members = sorted(source.iterdir(), key=lambda path: path.name)
    for member in members:
        if member.is_symlink() or not member.is_file():
            raise ValueError(
                "reset-architecture evidence must be a flat regular-file set: "
                f"{member}"
            )

    actual_names = tuple(member.name for member in members)
    if actual_names != RESET_ARCHITECTURE_FILES:
        missing = sorted(set(RESET_ARCHITECTURE_FILES) - set(actual_names))
        unexpected = sorted(set(actual_names) - set(RESET_ARCHITECTURE_FILES))
        raise ValueError(
            "reset-architecture evidence membership mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    checksum_path = source / RESET_ARCHITECTURE_CHECKSUM_NAME
    records = parse_sha256_manifest(
        checksum_path.read_bytes(),
        str(checksum_path),
    )
    expected_payloads = set(RESET_ARCHITECTURE_FILES) - {
        RESET_ARCHITECTURE_CHECKSUM_NAME
    }
    if set(records) != expected_payloads:
        missing = sorted(expected_payloads - set(records))
        unexpected = sorted(set(records) - expected_payloads)
        raise ValueError(
            "reset-architecture checksum membership mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, expected_digest in records.items():
        actual_digest = sha256_file(source / relative)
        if actual_digest != expected_digest:
            raise ValueError(
                "reset-architecture checksum mismatch for "
                f"{relative}: expected={expected_digest}, actual={actual_digest}"
            )


def validate_reset_architecture_archive(path: Path) -> None:
    """Validate the archived checksum manifest against the archived payload."""
    expected_names = [
        RESET_ARCHITECTURE_ROOT,
        *[
            f"{RESET_ARCHITECTURE_ROOT}/{relative}"
            for relative in RESET_ARCHITECTURE_FILES
        ],
    ]
    with tarfile.open(path, mode="r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if names != expected_names:
            missing = sorted(set(expected_names) - set(names))
            unexpected = sorted(set(names) - set(expected_names))
            raise RuntimeError(
                "reset-architecture archive membership/order mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        files = {member.name: member for member in members if member.isfile()}
        checksum_member = files[
            f"{RESET_ARCHITECTURE_ROOT}/{RESET_ARCHITECTURE_CHECKSUM_NAME}"
        ]
        checksum_file = bundle.extractfile(checksum_member)
        if checksum_file is None:
            raise RuntimeError("reset-architecture checksum member is unreadable")
        records = parse_sha256_manifest(
            checksum_file.read(),
            f"{path}:{checksum_member.name}",
        )
        expected_payloads = set(RESET_ARCHITECTURE_FILES) - {
            RESET_ARCHITECTURE_CHECKSUM_NAME
        }
        if set(records) != expected_payloads:
            missing = sorted(expected_payloads - set(records))
            unexpected = sorted(set(records) - expected_payloads)
            raise RuntimeError(
                "archived reset checksum membership mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for relative, expected_digest in records.items():
            member = files[f"{RESET_ARCHITECTURE_ROOT}/{relative}"]
            archived_file = bundle.extractfile(member)
            if archived_file is None:
                raise RuntimeError(
                    f"archived reset payload is unreadable: {relative}"
                )
            actual_digest = sha256_bytes(archived_file.read())
            if actual_digest != expected_digest:
                raise RuntimeError(
                    "archived reset checksum mismatch for "
                    f"{relative}: expected={expected_digest}, "
                    f"actual={actual_digest}"
                )


def normalized_tarinfo(entry: ArchiveEntry) -> tarfile.TarInfo:
    """Create fixed tar metadata for *entry*."""
    name = entry.name.rstrip("/") + "/" if entry.is_directory else entry.name
    info = tarfile.TarInfo(name)
    info.mtime = NORMALIZED_MTIME
    info.uid = NORMALIZED_UID
    info.gid = NORMALIZED_GID
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    if entry.is_directory:
        info.type = tarfile.DIRTYPE
        info.mode = DIRECTORY_MODE
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
        info.mode = FILE_MODE
        info.size = len(entry.data)
    return info


def collect_tree(
    source: Path,
    archive_root: str,
    excluded_members: frozenset[str] = frozenset(),
) -> tuple[list[ArchiveEntry], int]:
    """Collect a source directory in sorted order, filtering macOS sidecars."""
    if not source.is_dir():
        raise FileNotFoundError(f"required result directory is missing: {source}")

    entries = [ArchiveEntry(archive_root, None)]
    excluded = 0
    for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
        relative = PurePosixPath(path.relative_to(source).as_posix())
        if (
            is_forbidden_sidecar(relative)
            or relative.as_posix() in excluded_members
        ):
            excluded += 1
            continue
        member = str(PurePosixPath(archive_root) / relative)
        if path.is_symlink():
            raise ValueError(f"symlinks are not permitted in result archives: {path}")
        if path.is_dir():
            entries.append(ArchiveEntry(member, None))
        elif path.is_file():
            entries.append(ArchiveEntry(member, path.read_bytes()))
        else:
            raise ValueError(f"unsupported filesystem entry in result archive: {path}")
    return entries, excluded


def write_archive(path: Path, entries: Iterable[ArchiveEntry]) -> None:
    """Atomically write a deterministic gzip-compressed POSIX tar archive."""
    entries = list(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as raw:
            temporary = Path(raw.name)
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=NORMALIZED_MTIME,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as bundle:
                    for entry in entries:
                        info = normalized_tarinfo(entry)
                        payload = None if entry.is_directory else io.BytesIO(entry.data)
                        bundle.addfile(info, payload)
        os.replace(temporary, path)
        os.chmod(path, FILE_MODE)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def validate_archive(
    path: Path,
    expected_entries: Iterable[ArchiveEntry],
    excluded_count: int,
) -> ArchiveSummary:
    """Verify exact membership, normalized metadata, and absence of sidecars."""
    expected_entries = list(expected_entries)
    # ``tarfile`` canonicalizes directory member names by removing the trailing
    # slash when reading, even though command-line tar displays that slash.
    expected_names = [entry.name.rstrip("/") for entry in expected_entries]
    expected_by_name = dict(zip(expected_names, expected_entries, strict=True))

    with tarfile.open(path, mode="r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise RuntimeError(f"archive contains duplicate members: {path}")
        if names != expected_names:
            missing = sorted(set(expected_names) - set(names))
            unexpected = sorted(set(names) - set(expected_names))
            raise RuntimeError(
                f"archive membership/order mismatch for {path}: "
                f"missing={missing}, unexpected={unexpected}"
            )

        for member in members:
            pure_name = PurePosixPath(member.name)
            if pure_name.is_absolute() or ".." in pure_name.parts:
                raise RuntimeError(f"unsafe member path in {path}: {member.name}")
            if is_forbidden_sidecar(pure_name):
                raise RuntimeError(f"forbidden sidecar in {path}: {member.name}")
            expected = expected_by_name[member.name]
            expected_mode = DIRECTORY_MODE if expected.is_directory else FILE_MODE
            if (
                member.mtime != NORMALIZED_MTIME
                or member.uid != NORMALIZED_UID
                or member.gid != NORMALIZED_GID
                or member.uname != ""
                or member.gname != ""
                or member.mode != expected_mode
            ):
                raise RuntimeError(
                    f"non-normalized metadata in {path}: {member.name}"
                )
            if member.isdir() != expected.is_directory:
                raise RuntimeError(f"member type mismatch in {path}: {member.name}")
            if not member.isdir() and not member.isfile():
                raise RuntimeError(f"unsupported member type in {path}: {member.name}")
            if not expected.is_directory:
                extracted = bundle.extractfile(member)
                if extracted is None or extracted.read() != expected.data:
                    raise RuntimeError(f"content mismatch in {path}: {member.name}")

    return ArchiveSummary(
        path=path,
        member_count=len(expected_names),
        file_count=sum(not entry.is_directory for entry in expected_entries),
        excluded_count=excluded_count,
        sha256=sha256_file(path),
    )


def build_protocol_archive(
    source: Path,
    output: Path,
    archive_root: str,
    excluded_members: frozenset[str] = frozenset(),
) -> ArchiveSummary:
    if archive_root == RESET_ARCHITECTURE_ROOT:
        validate_reset_architecture_source(source)
    elif archive_root == PHASE_DIRECTION_ROOT:
        validate_phase_direction_source(source)
    elif archive_root == RANK_ONE_ORIENTATION_ROOT:
        validate_rank_one_orientation_source(source)
    entries, excluded = collect_tree(
        source,
        archive_root,
        excluded_members=excluded_members,
    )
    write_archive(output, entries)
    summary = validate_archive(output, entries, excluded)
    if archive_root == RESET_ARCHITECTURE_ROOT:
        validate_reset_architecture_archive(output)
    elif archive_root == PHASE_DIRECTION_ROOT:
        validate_phase_direction_archive(output)
    elif archive_root == RANK_ONE_ORIENTATION_ROOT:
        validate_rank_one_orientation_archive(output)
    return summary


def build_grouped_archive(source: Path, output: Path) -> ArchiveSummary:
    """Import grouped-measurement JSON with a deterministic provenance record."""
    if not source.is_file():
        raise FileNotFoundError(f"required grouped-measurement JSON is missing: {source}")
    if source.is_symlink():
        raise ValueError(f"grouped-measurement source may not be a symlink: {source}")

    raw = source.read_bytes()
    try:
        json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"grouped-measurement source is not valid JSON: {source}") from error

    source_digest = sha256_bytes(raw)
    manifest = {
        "archive_member": f"{GROUPED_ROOT}/{GROUPED_DATA_NAME}",
        "archive_member_sha256": source_digest,
        "schema_version": 1,
        "source_path": source.name,
        "source_path_note": (
            "Recovery-time filename; the machine-local absolute path is "
            "intentionally omitted from the portable archive."
        ),
        "source_sha256": source_digest,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    entries = [
        ArchiveEntry(GROUPED_ROOT, None),
        ArchiveEntry(f"{GROUPED_ROOT}/{GROUPED_DATA_NAME}", raw),
        ArchiveEntry(f"{GROUPED_ROOT}/{GROUPED_MANIFEST_NAME}", manifest_bytes),
    ]
    write_archive(output, entries)
    return validate_archive(output, entries, excluded_count=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic protocol result archives and, when supplied, "
            "the recovered grouped-measurement archive."
        )
    )
    parser.add_argument(
        "--grouped-source",
        type=Path,
        help="optional recovered phys_shots.json to import",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"source results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="archive destination (default: --results-dir)",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        choices=[
            *[directory for directory, _ in PROTOCOL_BUNDLES],
            PHASE_DIRECTION_ROOT,
            RANK_ONE_ORIENTATION_ROOT,
        ],
        help=(
            "build only the named protocol bundle; repeat for multiple bundles "
            "(default: build every available protocol bundle)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else results_dir
    )

    summaries: list[ArchiveSummary] = []
    selected = set(args.bundle or ())
    protocol_bundles = [
        bundle
        for bundle in PROTOCOL_BUNDLES
        if not selected or bundle[0] in selected
    ]
    for directory_name, archive_name in protocol_bundles:
        summaries.append(
            build_protocol_archive(
                results_dir / directory_name,
                output_dir / archive_name,
                directory_name,
                excluded_members=EXCLUDED_TREE_MEMBERS.get(
                    directory_name, frozenset()
                ),
            )
        )
    if not selected or PHASE_DIRECTION_ROOT in selected:
        summaries.append(
            build_protocol_archive(
                PHASE_DIRECTION_SOURCE,
                output_dir / PHASE_DIRECTION_ARCHIVE_NAME,
                PHASE_DIRECTION_ROOT,
            )
        )
    if not selected or RANK_ONE_ORIENTATION_ROOT in selected:
        summaries.append(
            build_protocol_archive(
                RANK_ONE_ORIENTATION_SOURCE,
                output_dir / RANK_ONE_ORIENTATION_ARCHIVE_NAME,
                RANK_ONE_ORIENTATION_ROOT,
            )
        )
    if args.grouped_source is not None:
        summaries.append(
            build_grouped_archive(
                args.grouped_source,
                output_dir / GROUPED_ARCHIVE_NAME,
            )
        )

    for summary in summaries:
        print(
            f"OK {summary.path}: members={summary.member_count} "
            f"files={summary.file_count} excluded={summary.excluded_count} "
            f"sha256={summary.sha256}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

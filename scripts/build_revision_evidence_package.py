#!/usr/bin/env python3
"""Build and verify the reviewer-facing Quantum-revision evidence package.

The builder is intentionally allowlist based.  It packages the scientific and
manuscript source needed to reproduce the analyses and figures, the current
revision reports, nine named revision result groups (including the extended
nested stage under revision tuning), and the checksum-verified baseline
evidence archives.  It does not walk the repository indiscriminately,
so rendered manuscript products, superseded analyses, caches, VCS metadata, and
local secrets cannot be swept into a submission archive accidentally.

Two build modes are supported:

* the default mode records absent or still-running result groups as partial in
  ``PROVENANCE.json`` and emits a valid snapshot; and
* ``--require-complete`` refuses to build until all final sentinels and reports
  are present and their aggregate status checks pass.

Both ZIP and ``.tar.gz`` outputs use stable ordering and normalized metadata.
The package contains a payload checksum manifest, and a sidecar checksum covers
the archive itself.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import struct
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "qrc_dissipation_reproducibility_package.zip"
)
ARCHIVE_ROOT = "qrc_dissipation_revision_evidence"
SCHEMA_VERSION = 3

ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
TAR_TIMESTAMP = 0
FILE_MODE = 0o644

ROOT_SOURCE_FILES = (
    "LICENSE",
    "pyproject.toml",
    "requirements.txt",
)
SOURCE_PATTERNS = (
    ".github/workflows/*.yml",
    "HANDOFF.md",
    "README.md",
    "configs/*.yaml",
    "experiments/**/*.py",
    "experiments/**/*.sbatch",
    "paper/*.md",
    "paper/dissipation_qrc.tex",
    "paper/l3_style.py",
    "paper/make_figures.py",
    "paper/quantum.bst",
    "paper/quantumarticle.cls",
    "paper/references.bib",
    "paper/sections/*.tex",
    "scripts/build_result_archives.py",
    "scripts/build_quantum_source_archive.py",
    "scripts/build_revision_evidence_package.py",
    "scripts/validate_submission.py",
    "src/**/*.py",
    "tests/**/*.py",
)
REQUIRED_SOURCE_FILES = (
    "configs/baseline.yaml",
    "experiments/_paths.py",
    "experiments/run_final_scaling.py",
    "experiments/run_measurement_full.py",
    "experiments/run_quantum_strengthening.py",
    "experiments/revision_inference.py",
    "experiments/run_revision_controls.py",
    "experiments/run_revision_fresh_interpolation.py",
    "experiments/run_revision_primary_regularization.py",
    "experiments/validate_revision_primary_regularization_artifacts.py",
    "experiments/run_collective_loss_full_input_diagnostic.py",
    "experiments/validate_collective_loss_full_input_artifacts.py",
    "experiments/run_nested_operating_point_extension.py",
    "experiments/validate_nested_operating_point_artifacts.py",
    "experiments/audit_nested_prescreen_stability.py",
    "experiments/run_revision_tuning.py",
    "experiments/build_revision_tuning_report.py",
    "experiments/run_primary_driven_activity.py",
    "experiments/run_activity_matched_response.py",
    "experiments/forecast_baseline_audit.py",
    "paper/dissipation_qrc.tex",
    "paper/l3_style.py",
    "paper/make_figures.py",
    "paper/quantum.bst",
    "paper/quantumarticle.cls",
    "paper/references.bib",
    "paper/sections/abstract.tex",
    "paper/sections/background.tex",
    "paper/sections/conclusion.tex",
    "paper/sections/evaluation.tex",
    "paper/sections/experimental-section.tex",
    "paper/sections/introduction.tex",
    "paper/sections/methodology.tex",
    "paper/sections/related-work.tex",
    "scripts/build_revision_evidence_package.py",
    "scripts/build_quantum_source_archive.py",
    "scripts/validate_submission.py",
    "src/qrc/__init__.py",
    "src/qrc/dissipators.py",
    "src/qrc/liouvillian.py",
    "src/qrc/operators.py",
    "src/qrc/readout.py",
    "src/qrc/reservoirs.py",
    "src/qrc/sparse_evolve.py",
    "src/qrc/tasks.py",
    "tests/test_dephasing_contraction_proof.py",
    "tests/test_measurement_full.py",
    "tests/test_revision_controls.py",
    "tests/test_revision_evidence_package.py",
    "tests/test_revision_fresh_interpolation.py",
    "tests/test_revision_primary_regularization.py",
    "tests/test_collective_loss_full_input_diagnostic.py",
    "tests/test_nested_operating_point_extension.py",
    "tests/test_nested_prescreen_stability.py",
    "tests/test_revision_tuning.py",
    "tests/test_revision_tuning_report.py",
    "tests/test_revision_inference.py",
    "tests/test_primary_driven_activity.py",
    "tests/test_activity_matched_response.py",
    "tests/test_forecast_baseline_audit.py",
    "tests/test_quantum_source_archive.py",
)

CURRENT_REPORTS = (
    "reports/theory_unitality.md",
    "reports/dephasing_uniform_contraction_proof.md",
    "reports/measurement_full_equal_total_shots.md",
    "reports/revision_controls_report.md",
    "reports/revision_tuning_report.md",
    "reports/revision_primary_regularization_report.md",
    "reports/collective_loss_full_input_diagnostic.md",
    "reports/nested_operating_point_extension_report.md",
    "reports/nested_prescreen_stability_audit.md",
    "reports/primary_driven_activity_report.md",
    "reports/activity_matched_response_development_audit.md",
    "reports/activity_matched_response_v2_failure_audit.md",
    "reports/activity_matched_response_v2_recovery_plan.md",
    "reports/activity_matched_response_report.md",
    "reports/forecast_baseline_audit.md",
    "reports/review_response_v7.md",
)
REQUIRED_FINAL_REPORTS = (
    "reports/dephasing_uniform_contraction_proof.md",
    "reports/measurement_full_equal_total_shots.md",
    "reports/revision_controls_report.md",
    "reports/revision_tuning_report.md",
    "reports/revision_primary_regularization_report.md",
    "reports/collective_loss_full_input_diagnostic.md",
    "reports/nested_operating_point_extension_report.md",
    "reports/nested_prescreen_stability_audit.md",
    "reports/primary_driven_activity_report.md",
    "reports/activity_matched_response_development_audit.md",
    "reports/activity_matched_response_v2_failure_audit.md",
    "reports/activity_matched_response_v2_recovery_plan.md",
    "reports/activity_matched_response_report.md",
    "reports/forecast_baseline_audit.md",
    "reports/review_response_v7.md",
)
RELEASE_GATE_FILES = (
    "HANDOFF.md",
    "paper/FIGURE_QA.md",
)

BASELINE_ARCHIVES = (
    "final_protocol_results.tar.gz",
    "review_protocol_results.tar.gz",
)
BASELINE_CHECKSUM_FILE = "ARCHIVE_SHA256SUMS.txt"

RESULT_DEPENDENCIES = (
    "quantum_strengthening_v2_paper/frozen_diagnostic_predictions.json",
)
FRESH_INTERPOLATION_MANIFEST = (
    "results/revision_tuning/fresh_interpolation/manifest.json"
)
FRESH_INTERPOLATION_AGGREGATE = (
    "results/revision_tuning/fresh_interpolation/"
    "fresh_interpolation_results.json"
)

FORBIDDEN_PATH_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "__MACOSX",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "tmp",
}
FORBIDDEN_EXACT_NAMES = {
    ".DS_Store",
    ".env",
    "credentials",
    "credentials.json",
    "id_rsa",
    "id_ed25519",
}
FORBIDDEN_NAME_FRAGMENTS = (
    "access_token",
    "api_key",
    "client_secret",
    "private_key",
)
FORBIDDEN_SUFFIXES = (
    ".aux",
    ".bbl",
    ".blg",
    ".coverage",
    ".fdb_latexmk",
    ".fls",
    ".key",
    ".log",
    ".out",
    ".pdf",
    ".pem",
    ".pyc",
    ".pyo",
    ".synctex.gz",
)


@dataclass(frozen=True)
class ResultGroup:
    """An allowlisted result tree and the sentinels that mark it complete."""

    name: str
    include_patterns: tuple[str, ...]
    required_files: tuple[str, ...]
    aggregate_files: tuple[str, ...]


RESULT_GROUPS = (
    ResultGroup(
        "revision_tuning",
        (
            "**/*.json",
            "strength_extension/source_snapshot/run_revision_tuning.py",
            "nested_tuning/source_snapshot/run_revision_tuning.py",
            "fresh_interpolation/source_snapshot/run_revision_tuning.py",
            (
                "fresh_interpolation/source_snapshot/"
                "run_revision_fresh_interpolation.py"
            ),
            (
                "nested_operating_point_extension/source_snapshot/"
                "run_nested_operating_point_extension.py"
            ),
        ),
        (
            "strength_extension/manifest.json",
            "strength_extension/six_channel_aggregate.json",
            "strength_extension/alternative_matching_aggregate.json",
            "strength_extension/source_snapshot/manifest.json",
            "strength_extension/source_snapshot/run_revision_tuning.py",
            "nested_tuning/manifest.json",
            "nested_tuning/nested_tuning_results.json",
            "nested_tuning/source_snapshot/manifest.json",
            "nested_tuning/source_snapshot/run_revision_tuning.py",
            "fresh_interpolation/manifest.json",
            "fresh_interpolation/fresh_interpolation_results.json",
            "critique_response_inference.json",
            "fresh_interpolation/source_snapshot/manifest.json",
            "fresh_interpolation/source_snapshot/run_revision_tuning.py",
            (
                "fresh_interpolation/source_snapshot/"
                "run_revision_fresh_interpolation.py"
            ),
            "nested_operating_point_extension/manifest.json",
            "nested_operating_point_extension/seed_ledger.json",
            "nested_operating_point_extension/reuse_index.json",
            "nested_operating_point_extension/screen_shortlist.json",
            "nested_operating_point_extension/frozen_selection.json",
            "nested_operating_point_extension/aggregate.json",
            "nested_operating_point_extension/prescreen_stability.json",
            "nested_operating_point_extension/source_snapshot/manifest.json",
            (
                "nested_operating_point_extension/source_snapshot/"
                "run_nested_operating_point_extension.py"
            ),
        ),
        (
            "strength_extension/six_channel_aggregate.json",
            "nested_tuning/nested_tuning_results.json",
            "fresh_interpolation/fresh_interpolation_results.json",
            "nested_operating_point_extension/aggregate.json",
            "nested_operating_point_extension/prescreen_stability.json",
            "critique_response_inference.json",
        ),
    ),
    ResultGroup(
        "measurement_full_v3",
        ("protocol.json", "measurement_full_aggregate.json", "jobs/*.json"),
        ("protocol.json", "measurement_full_aggregate.json"),
        ("measurement_full_aggregate.json",),
    ),
    ResultGroup(
        "revision_parity_control",
        (
            "paper__*.json",
            "paper_protocol.json",
            "paper_aggregate.json",
            "paper_reference__*.json",
            "paper_reference_protocol.json",
            "paper_reference_aggregate.json",
        ),
        (
            "paper_protocol.json",
            "paper_aggregate.json",
            "paper_reference_protocol.json",
            "paper_reference_aggregate.json",
        ),
        ("paper_aggregate.json", "paper_reference_aggregate.json"),
    ),
    ResultGroup(
        "revision_normalized_scaling",
        ("paper__*.json", "paper_variance_protocol.json", "paper_variance_aggregate.json"),
        ("paper_variance_protocol.json", "paper_variance_aggregate.json"),
        ("paper_variance_aggregate.json",),
    ),
    ResultGroup(
        "revision_primary_regularization",
        ("protocol.json", "aggregate.json", "jobs/*.json"),
        ("protocol.json", "aggregate.json"),
        ("aggregate.json",),
    ),
    ResultGroup(
        "collective_loss_full_input_diagnostic",
        ("protocol.json", "raw_spectrum.json", "aggregate.json"),
        ("protocol.json", "raw_spectrum.json", "aggregate.json"),
        ("aggregate.json",),
    ),
    ResultGroup(
        "primary_driven_activity",
        ("protocol.json", "aggregate.json", "checkpoints/*.json"),
        ("protocol.json", "aggregate.json"),
        ("aggregate.json",),
    ),
    ResultGroup(
        "forecast_baseline_audit",
        ("aggregate.json",),
        ("aggregate.json",),
        ("aggregate.json",),
    ),
    ResultGroup(
        "activity_matched_response",
        (
            "pilot_manifest.json",
            "pilot/checkpoints/*.json",
            "frozen_targets.json",
            "task_manifest.json",
            "calibration/checkpoints/*.json",
            "frozen_calibration.json",
            "score/checkpoints/*.json",
            "aggregate.json",
            "source_snapshot/manifest.json",
            "source_snapshot/run_activity_matched_response.py",
        ),
        (
            "pilot_manifest.json",
            "frozen_targets.json",
            "task_manifest.json",
            "source_snapshot/manifest.json",
            "source_snapshot/run_activity_matched_response.py",
        ),
        (),
    ),
)


class EvidencePackageError(RuntimeError):
    """Base class for evidence-package validation failures."""


class IncompleteEvidenceError(EvidencePackageError):
    """Raised when strict final mode encounters incomplete evidence."""


@dataclass(frozen=True)
class Payload:
    """One normalized regular file in the package."""

    name: str
    data: bytes


@dataclass(frozen=True)
class BuildSummary:
    """Machine-friendly summary of a successful build."""

    path: Path
    sidecar: Path
    sha256: str
    file_count: int
    complete: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_relative_path(value: str | PurePosixPath) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise EvidencePackageError(f"unsafe archive path: {value}")
    if (
        "\\" in str(value)
        or "\x00" in str(value)
        or ":" in path.parts[0]
    ):
        raise EvidencePackageError(f"unsafe archive path: {value}")
    return path


def _forbidden_source_path(relative: PurePosixPath) -> bool:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    lowered_name = relative.name.lower()
    forbidden_parts = {value.lower() for value in FORBIDDEN_PATH_PARTS}
    if any(part in forbidden_parts for part in lowered_parts):
        return True
    if lowered_name in {value.lower() for value in FORBIDDEN_EXACT_NAMES}:
        return True
    if lowered_name.startswith("._"):
        return True
    if any(fragment in lowered_name for fragment in FORBIDDEN_NAME_FRAGMENTS):
        return True
    return any(lowered_name.endswith(suffix.lower()) for suffix in FORBIDDEN_SUFFIXES)


def _read_regular_file(path: Path, repo_root: Path) -> bytes:
    try:
        relative = PurePosixPath(path.relative_to(repo_root).as_posix())
    except ValueError as error:
        raise EvidencePackageError(f"source escapes repository: {path}") from error
    _safe_relative_path(relative)
    if _forbidden_source_path(relative):
        raise EvidencePackageError(f"forbidden source path selected: {relative}")
    if path.is_symlink():
        raise EvidencePackageError(f"symlinks are not allowed: {relative}")
    if not path.is_file():
        raise EvidencePackageError(f"source is not a regular file: {relative}")
    return path.read_bytes()


def _strict_json_loads(data: bytes, label: str) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(data, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvidencePackageError(f"invalid strict JSON in {label}: {error}") from error


def _git_value(repo_root: Path, args: Sequence[str], fallback: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return fallback
    # Preserve the leading status column in ``git status --porcelain``.
    # Other callers only need trailing newlines removed.
    return completed.stdout.rstrip() or fallback


def _run_required_git(repo_root: Path, args: Sequence[str], label: str) -> str:
    """Run a Git query that must succeed for a final evidence snapshot."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise IncompleteEvidenceError(
            f"final evidence requires Git for {label}: {error}"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise IncompleteEvidenceError(
            f"final evidence requires a valid Git repository and {label}{suffix}"
        ) from error
    # ``strip`` corrupts the first row of porcelain output when its index
    # status is a space (for example, `` M path``).
    return completed.stdout.rstrip()


def _status_rows(
    raw: str,
    repo_root: Path,
    excluded_paths: Iterable[Path],
) -> list[str]:
    """Normalize porcelain status while excluding only the requested outputs."""
    excluded: set[str] = set()
    for path in excluded_paths:
        try:
            excluded.add(path.resolve().relative_to(repo_root.resolve()).as_posix())
        except ValueError:
            continue
    rows = []
    for line in raw.splitlines():
        # Porcelain v1 paths start after the two status columns and one space.
        path_text = line[3:] if len(line) >= 4 else ""
        candidates = path_text.split(" -> ")
        if any(candidate in excluded for candidate in candidates):
            continue
        rows.append(line)
    return sorted(rows)


def _git_status(repo_root: Path, excluded_paths: Iterable[Path]) -> list[str]:
    raw = _git_value(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "",
    )
    return _status_rows(raw, repo_root, excluded_paths)


def _require_clean_committed_git(
    repo_root: Path,
    excluded_paths: Iterable[Path],
) -> str:
    """Return HEAD only for an exact, clean repository-root snapshot."""
    top_level = _run_required_git(
        repo_root,
        ["rev-parse", "--show-toplevel"],
        "repository root",
    )
    if Path(top_level).resolve() != repo_root.resolve():
        raise IncompleteEvidenceError(
            "final evidence repo root does not match Git top level: "
            f"{repo_root.resolve()} != {Path(top_level).resolve()}"
        )
    head = _run_required_git(
        repo_root,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        "committed HEAD",
    )
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", head) is None:
        raise IncompleteEvidenceError(
            f"final evidence Git HEAD is not a commit hash: {head!r}"
        )
    raw_status = _run_required_git(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        "clean worktree status",
    )
    dirty_status = _status_rows(raw_status, repo_root, excluded_paths)
    if dirty_status:
        raise IncompleteEvidenceError(
            "final evidence must be built from a clean committed tree; "
            "uncommitted paths are:\n" + "\n".join(dirty_status)
        )
    return head.lower()


def _collect_scientific_source(
    repo_root: Path,
) -> tuple[list[Payload], dict]:
    selected: dict[str, Path] = {}
    for relative in ROOT_SOURCE_FILES:
        path = repo_root / relative
        if path.is_file():
            selected[relative] = path
    for pattern in SOURCE_PATTERNS:
        for path in repo_root.glob(pattern):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = path.relative_to(repo_root).as_posix()
            if _forbidden_source_path(PurePosixPath(relative)):
                continue
            selected[relative] = path

    missing = sorted(
        relative
        for relative in (*ROOT_SOURCE_FILES, *REQUIRED_SOURCE_FILES)
        if not (repo_root / relative).is_file()
    )
    payloads = [
        Payload(
            f"source/{relative}",
            _read_regular_file(path, repo_root),
        )
        for relative, path in sorted(selected.items())
    ]
    return payloads, {
        "status": "complete" if not missing else "partial",
        "included_files": len(payloads),
        "missing_required_files": missing,
    }


def _collect_reports(repo_root: Path) -> tuple[list[Payload], dict]:
    payloads = []
    for relative in CURRENT_REPORTS:
        path = repo_root / relative
        if path.is_file():
            payloads.append(
                Payload(relative, _read_regular_file(path, repo_root))
            )
    missing = sorted(
        relative
        for relative in REQUIRED_FINAL_REPORTS
        if not (repo_root / relative).is_file()
    )
    return payloads, {
        "status": "complete" if not missing else "partial",
        "included_files": len(payloads),
        "missing_required_files": missing,
    }


def _collect_result_dependencies(
    repo_root: Path,
) -> tuple[list[Payload], dict[str, object]]:
    """Package and authenticate named inputs needed by the included drivers."""
    payloads: list[Payload] = []
    missing: list[str] = []
    checks: dict[str, dict[str, object]] = {}
    dependency_data: dict[str, bytes] = {}
    dependency_json: dict[str, object] = {}

    for relative in RESULT_DEPENDENCIES:
        path = repo_root / "results" / relative
        archive_path = f"results/{relative}"
        if not path.is_file():
            missing.append(archive_path)
            continue
        data = _read_regular_file(path, repo_root)
        dependency_json[archive_path] = _strict_json_loads(data, archive_path)
        dependency_data[archive_path] = data
        payloads.append(Payload(archive_path, data))

    manifest_path = repo_root / FRESH_INTERPOLATION_MANIFEST
    manifest_check: dict[str, object] = {
        "complete": False,
        "reason": "fresh-interpolation manifest is missing",
    }
    if manifest_path.is_file():
        manifest = _strict_json_loads(
            _read_regular_file(manifest_path, repo_root),
            FRESH_INTERPOLATION_MANIFEST,
        )
        try:
            frozen_source = manifest["protocol"]["frozen_diagnostic_source"]
            expected_path = str(frozen_source["path"])
            expected_sha = str(frozen_source["sha256"]).lower()
        except (KeyError, TypeError):
            manifest_check["reason"] = (
                "fresh-interpolation manifest lacks frozen diagnostic provenance"
            )
        else:
            data = dependency_data.get(expected_path)
            if expected_path not in {
                f"results/{relative}" for relative in RESULT_DEPENDENCIES
            }:
                manifest_check["reason"] = (
                    "fresh-interpolation manifest names an unallowlisted "
                    f"dependency: {expected_path}"
                )
            elif data is None:
                manifest_check["reason"] = (
                    f"frozen diagnostic dependency is missing: {expected_path}"
                )
            elif re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
                manifest_check["reason"] = (
                    "fresh-interpolation manifest has an invalid dependency hash"
                )
            elif sha256_bytes(data) != expected_sha:
                manifest_check["reason"] = (
                    "frozen diagnostic dependency hash does not match the "
                    "fresh-interpolation manifest"
                )
            else:
                manifest_check = {
                    "complete": True,
                    "reason": (
                        "frozen diagnostic dependency matches the sealed "
                        "fresh-interpolation manifest"
                    ),
                    "path": expected_path,
                    "sha256": expected_sha,
                }
    checks["fresh_interpolation_frozen_diagnostic"] = manifest_check

    aggregate_check: dict[str, object] = {
        "complete": False,
        "reason": "fresh-interpolation aggregate is missing",
    }
    aggregate_path = repo_root / FRESH_INTERPOLATION_AGGREGATE
    if manifest_check["complete"] and aggregate_path.is_file():
        aggregate = _strict_json_loads(
            _read_regular_file(aggregate_path, repo_root),
            FRESH_INTERPOLATION_AGGREGATE,
        )
        manifest = _strict_json_loads(
            _read_regular_file(manifest_path, repo_root),
            FRESH_INTERPOLATION_MANIFEST,
        )
        frozen_source = manifest["protocol"]["frozen_diagnostic_source"]
        expected_path = str(frozen_source["path"])
        frozen = dependency_json[expected_path]
        if not isinstance(aggregate, dict) or not isinstance(frozen, dict):
            aggregate_check["reason"] = (
                "fresh aggregate or frozen diagnostic root is not an object"
            )
        elif aggregate.get("protocol_sha256") != manifest.get(
            "protocol_sha256"
        ):
            aggregate_check["reason"] = (
                "fresh aggregate is not linked to the sealed protocol hash"
            )
        elif aggregate.get("frozen_diagnostic_source") != frozen_source:
            aggregate_check["reason"] = (
                "fresh aggregate frozen-source metadata differs from manifest"
            )
        elif aggregate.get("frozen_diagnostic_rows") != frozen.get(
            "diagnostic_rows"
        ):
            aggregate_check["reason"] = (
                "fresh aggregate diagnostic rows differ from frozen input"
            )
        elif aggregate.get("frozen_diagnostic_predictions_by_N") != frozen.get(
            "predictions_by_N"
        ):
            aggregate_check["reason"] = (
                "fresh aggregate predictions differ from frozen input"
            )
        else:
            aggregate_check = {
                "complete": True,
                "reason": (
                    "fresh aggregate embeds the exact hash-linked frozen "
                    "diagnostic rows and predictions"
                ),
            }
    checks["fresh_interpolation_aggregate_linkage"] = aggregate_check

    complete = not missing and all(
        bool(check["complete"]) for check in checks.values()
    )
    return payloads, {
        "status": "complete" if complete else "partial",
        "included_files": len(payloads),
        "missing_required_files": sorted(missing),
        "checks": checks,
    }


def _release_documentation_status(repo_root: Path) -> dict[str, object]:
    """Fail closed on explicit final-seal and figure-QA release blockers."""
    missing: list[str] = []
    blockers: list[str] = []
    for relative in RELEASE_GATE_FILES:
        path = repo_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise EvidencePackageError(
                f"release-gate document is not UTF-8: {relative}"
            ) from error
        for line_number, line in enumerate(lines, start=1):
            pending = re.search(r"\bPENDING\b", line, flags=re.IGNORECASE)
            if not pending:
                continue
            if relative == "HANDOFF.md" and "FINAL-SEAL" not in line.upper():
                continue
            blockers.append(f"{relative}:{line_number}: {line.strip()}")
    complete = not missing and not blockers
    return {
        "status": "complete" if complete else "partial",
        "checked_files": list(RELEASE_GATE_FILES),
        "missing_required_files": missing,
        "pending_blockers": blockers,
    }


def _aggregate_is_complete(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "aggregate root is not an object"

    validation = payload.get("validation")
    if isinstance(validation, dict) and validation.get("status") == "complete":
        return True, "validation.status=complete"

    status = payload.get("status")
    if status == "complete":
        missing = payload.get("missing_checkpoints", [])
        expected = payload.get("expected_checkpoints")
        complete = payload.get("complete_checkpoints")
        if missing:
            return False, "status is complete but missing_checkpoints is nonempty"
        if expected is not None and complete is not None and expected != complete:
            return False, "status is complete but checkpoint counts differ"
        return True, "status=complete"
    return False, f"aggregate status is {status!r}"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _close(left: object, right: object, *, atol: float = 1e-10) -> bool:
    left_value = _finite_float(left)
    right_value = _finite_float(right)
    return (
        left_value is not None
        and right_value is not None
        and math.isclose(left_value, right_value, rel_tol=1e-10, abs_tol=atol)
    )


def _mean_and_sample_se(values: Sequence[float]) -> tuple[float, float]:
    """Return the arithmetic mean and standard error without NumPy."""
    numbers = [float(value) for value in values]
    if not numbers or any(not math.isfinite(value) for value in numbers):
        raise ValueError("summary values must be finite and nonempty")
    mean = math.fsum(numbers) / len(numbers)
    if len(numbers) == 1:
        return mean, 0.0
    squared = math.fsum((value - mean) ** 2 for value in numbers)
    return mean, math.sqrt(squared / (len(numbers) - 1) / len(numbers))


def _summary_mean_se_matches(payload: object, values: Sequence[float]) -> bool:
    """Check that a stored headline summary is derived from its raw rows."""
    if not isinstance(payload, dict) or payload.get("n") != len(values):
        return False
    try:
        mean, se = _mean_and_sample_se(values)
    except (TypeError, ValueError):
        return False
    return _close(payload.get("mean"), mean) and _close(payload.get("se"), se)


def _scientific_protocol_sha256(protocol: object) -> str | None:
    """Match the compact canonical JSON hash used by experiment drivers."""
    try:
        encoded = json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _revision_strength_is_complete(payload: object) -> tuple[bool, str]:
    """Validate the sealed all-channel strength grid, not just its existence."""
    if not isinstance(payload, dict):
        return False, "strength aggregate root is not an object"
    if payload.get("status") != "complete":
        return False, f"strength status is {payload.get('status')!r}"
    expected_rows = 920
    if (
        payload.get("expected_raw_row_count") != expected_rows
        or payload.get("complete_raw_row_count") != expected_rows
    ):
        return False, "strength expected/complete raw-row counts are not 920"
    if payload.get("expected_extended_multipliers") != [8, 16]:
        return False, "strength extended multipliers are not [8, 16]"
    if payload.get("collective_optimum_bracketed") is not True:
        return False, "collective strength optimum is not bracketed"

    expected_methods = {
        "A1_heterogeneous",
        "B2_thermal",
        "B3_collective",
        "B4_loss_exchange",
        "B5_pair",
        "CD_paper",
    }
    methods = payload.get("methods")
    if not isinstance(methods, dict) or set(methods) != expected_methods:
        return False, "strength method set is incomplete"

    base_grid = {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0}
    extended_methods = {"B3_collective", "CD_paper"}
    raw_rows = payload.get("raw_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_rows:
        return False, "strength raw_rows length is not 920"
    try:
        raw_keys = [
            (str(row["method"]), float(row["mult"]), int(row["seed"]))
            for row in raw_rows
        ]
    except (KeyError, TypeError, ValueError):
        return False, "strength raw row is malformed"
    if len(set(raw_keys)) != expected_rows:
        return False, "strength raw rows contain duplicate checkpoints"
    seeds = {seed for _, _, seed in raw_keys}
    if len(seeds) != 20:
        return False, "strength grid does not contain exactly 20 seeds"
    for method in expected_methods:
        grid = base_grid | ({8.0, 16.0} if method in extended_methods else set())
        for multiplier in grid:
            observed = {
                seed
                for row_method, row_multiplier, seed in raw_keys
                if row_method == method and row_multiplier == multiplier
            }
            if observed != seeds:
                return False, (
                    f"strength grid is incomplete for {method} at x{multiplier:g}"
                )
        curve = methods[method].get("curve_bracket")
        if not isinstance(curve, dict) or curve.get("bracketed") is not True:
            return False, f"strength curve is not bracketed for {method}"
        points = curve.get("curve")
        if not isinstance(points, list):
            return False, f"strength curve is malformed for {method}"
        try:
            curve_counts = {
                float(point["multiplier"]): int(point["n"])
                for point in points
            }
        except (KeyError, TypeError, ValueError):
            return False, f"strength curve point is malformed for {method}"
        if set(curve_counts) != grid or any(
            count != 20 for count in curve_counts.values()
        ):
            return False, f"strength curve coverage is incomplete for {method}"
        selected = methods[method].get("leave_one_seed_out")
        if not isinstance(selected, dict):
            return False, f"strength LOSO result is missing for {method}"
        scores = selected.get("scores_by_seed")
        selected_by_seed = selected.get("selected_multiplier_by_seed")
        if (
            not isinstance(scores, dict)
            or not isinstance(selected_by_seed, dict)
            or len(scores) != 20
            or set(scores) != set(selected_by_seed)
        ):
            return False, f"strength LOSO coverage is incomplete for {method}"
    return True, "sealed 920-row strength grid with bracketed optima"


def _paired_effect_is_complete(effect: object) -> bool:
    if not isinstance(effect, dict) or effect.get("n") != 32:
        return False
    differences = effect.get("paired_differences")
    if (
        not isinstance(differences, list)
        or len(differences) != 32
        or any(_finite_float(value) is None for value in differences)
    ):
        return False
    numeric = (
        "candidate_mean",
        "reference_mean",
        "mean_difference",
        "se_difference",
        "ci95_low",
        "ci95_high",
        "relative_mean_difference_percent",
        "exact_sign_p_two_sided",
    )
    if any(_finite_float(effect.get(key)) is None for key in numeric):
        return False
    p_value = float(effect["exact_sign_p_two_sided"])
    if not 0.0 <= p_value <= 1.0:
        return False
    wins = effect.get("wins")
    ties = effect.get("ties")
    return (
        type(wins) is int
        and type(ties) is int
        and 0 <= wins <= 32
        and 0 <= ties <= 32
        and wins + ties <= 32
        and float(effect["ci95_low"])
        <= float(effect["mean_difference"])
        <= float(effect["ci95_high"])
    )


def _revision_alternative_matching_is_complete(
    payload: object,
) -> tuple[bool, str]:
    """Validate all operational-matching rows, censoring, and gap curves."""
    if not isinstance(payload, dict):
        return False, "alternative-matching root is not an object"
    if payload.get("artifact_type") != "revision_alternative_matching_aggregate":
        return False, "alternative-matching artifact type is wrong"

    methods = ("B2_thermal", "B3_collective", "B5_pair")
    modes = ("energy", "gap", "activity")
    active_methods = {
        "A1_heterogeneous",
        "B2_thermal",
        "B3_collective",
        "B4_loss_exchange",
        "B5_pair",
        "CD_paper",
    }
    expected_conditions = {
        f"{method}__{mode}" for method in methods for mode in modes
    }

    reference = payload.get("reference")
    if (
        not isinstance(reference, dict)
        or reference.get("method") != "CD_paper"
        or reference.get("n") != 32
        or _finite_float(reference.get("mean")) is None
        or _finite_float(reference.get("se")) is None
    ):
        return False, "alternative-matching reference is incomplete"

    raw = payload.get("raw_rows")
    expected_raw_counts = {"R_match": 224, "R_match2": 96, "R_gapsweep": 126}
    if not isinstance(raw, dict) or set(raw) != set(expected_raw_counts):
        return False, "alternative-matching raw blocks are incomplete"
    for block, expected_count in expected_raw_counts.items():
        if not isinstance(raw[block], list) or len(raw[block]) != expected_count:
            return False, f"alternative-matching {block} row count is incomplete"

    try:
        r_match_keys = [
            (str(row["method"]), row.get("mode"), int(row["seed"]))
            for row in raw["R_match"]
        ]
        r_match2_keys = [
            (str(row["method"]), str(row["mode"]), int(row["seed"]))
            for row in raw["R_match2"]
        ]
        gap_keys = [
            (str(row["method"]), float(row["mult"]), int(row["seed"]))
            for row in raw["R_gapsweep"]
        ]
    except (KeyError, TypeError, ValueError):
        return False, "alternative-matching raw row is malformed"
    if (
        len(set(r_match_keys)) != 224
        or len(set(r_match2_keys)) != 96
        or len(set(gap_keys)) != 126
    ):
        return False, "alternative-matching raw identities are duplicated"

    reference_rows = {
        int(row["seed"]): row
        for row in raw["R_match"]
        if row.get("method") == "CD_paper" and row.get("mode") is None
    }
    if len(reference_rows) != 32:
        return False, "alternative-matching dial does not contain 32 seeds"
    seeds = set(reference_rows)
    expected_r_match = {
        ("CD_paper", None, seed) for seed in seeds
    } | {
        (method, mode, seed)
        for method in methods
        for mode in ("energy", "gap")
        for seed in seeds
    }
    expected_r_match2 = {
        (method, "activity", seed)
        for method in methods
        for seed in seeds
    }
    if set(r_match_keys) != expected_r_match:
        return False, "energy/gap matching coverage is not 32 rows per condition"
    if set(r_match2_keys) != expected_r_match2:
        return False, "activity matching coverage is not 32 rows per condition"

    for block, rows in (("R_match", raw["R_match"]), ("R_match2", raw["R_match2"])):
        for row in rows:
            if (
                row.get("block") != block
                or row.get("N") != 5
                or row.get("task") != "stm"
                or _finite_float(row.get("value")) is None
            ):
                return False, f"alternative-matching {block} row metadata is invalid"
            if row.get("method") != "CD_paper":
                scale = _finite_float(row.get("scale_factor"))
                if scale is None or scale <= 0:
                    return False, f"alternative-matching {block} scale is invalid"
            if block == "R_match2" and (
                type(row.get("reachable")) is not bool
                or _finite_float(row.get("activity_ratio")) is None
            ):
                return False, "activity matching lacks reachability/ratio flags"

    conditions = payload.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != expected_conditions:
        return False, "alternative-matching condition set is not the complete 3x3 grid"
    for method in methods:
        for mode in modes:
            key = f"{method}__{mode}"
            condition = conditions[key]
            expected_block = "R_match2" if mode == "activity" else "R_match"
            if (
                not isinstance(condition, dict)
                or condition.get("block") != expected_block
                or condition.get("method") != method
                or condition.get("matching_mode") != mode
            ):
                return False, f"alternative-matching condition {key} is malformed"
            feasibility = condition.get("match_feasibility")
            if not isinstance(feasibility, dict) or feasibility.get("total") != 32:
                return False, f"alternative-matching feasibility is missing for {key}"
            reachable = feasibility.get("reachable_count")
            if type(reachable) is not int or not 0 <= reachable <= 32:
                return False, f"alternative-matching reachable count is invalid for {key}"

            source_rows = [
                row
                for row in raw[expected_block]
                if row.get("method") == method and row.get("mode") == mode
            ]
            source_rows.sort(key=lambda row: int(row["seed"]))
            if len(source_rows) != 32:
                return False, f"alternative-matching raw coverage is incomplete for {key}"
            if mode == "energy":
                observed_reachable = 32
                expected_status = "analytically_exact_linear_rescaling"
            elif mode == "gap":
                observed_reachable = sum(
                    float(row["scale_factor"]) < 39.99 for row in source_rows
                )
                expected_status = (
                    "root_inside_search_interval"
                    if observed_reachable == 32
                    else "upper_bound_censored_target_not_reached"
                )
                if feasibility.get("search_interval") != [0.05, 40.0]:
                    return False, f"gap search/censor interval is missing for {key}"
            else:
                observed_reachable = sum(
                    bool(row["reachable"]) for row in source_rows
                )
                expected_status = (
                    "all_exactly_reachable"
                    if observed_reachable == 32
                    else "closest_achievable_activity_reported_when_unreachable"
                )
            if (
                reachable != observed_reachable
                or feasibility.get("status") != expected_status
            ):
                return False, f"alternative-matching censor flags disagree for {key}"

            effect = condition.get("effect_vs_standard_dial")
            if not _paired_effect_is_complete(effect):
                return False, f"alternative-matching paired effect is incomplete for {key}"
            expected_differences = [
                float(row["value"])
                - float(reference_rows[int(row["seed"])]["value"])
                for row in source_rows
            ]
            if any(
                not _close(actual, expected, atol=1e-12)
                for actual, expected in zip(
                    effect["paired_differences"], expected_differences
                )
            ):
                return False, f"alternative-matching paired deltas disagree for {key}"
            numeric_fields = (
                "matched_channel_mean",
                "matched_channel_se",
                "dial_mean",
                "dial_se",
                "scale_factor_mean",
                "scale_factor_se",
                "scale_factor_min",
                "scale_factor_max",
            )
            if any(
                _finite_float(condition.get(field)) is None
                for field in numeric_fields
            ):
                return False, f"alternative-matching summary is non-finite for {key}"

    gap_rows = raw["R_gapsweep"]
    multipliers = {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0}
    diagnostic_seeds = {seed for _, _, seed in gap_keys}
    expected_gap_keys = {
        (method, multiplier, seed)
        for method in active_methods
        for multiplier in multipliers
        for seed in diagnostic_seeds
    }
    if len(diagnostic_seeds) != 3 or set(gap_keys) != expected_gap_keys:
        return False, "full driven-gap raw grid is not six channels x 7 x 3"
    for row in gap_rows:
        if (
            row.get("block") != "R_gapsweep"
            or row.get("N") != 5
            or row.get("task") != "gap"
            or _finite_float(row.get("value")) is None
        ):
            return False, "driven-gap raw row metadata is invalid"

    curves = payload.get("full_driven_gap_curves")
    if not isinstance(curves, dict) or set(curves) != active_methods:
        return False, "full driven-gap curve method set is incomplete"
    for method, points in curves.items():
        if not isinstance(points, list) or len(points) != 7:
            return False, f"driven-gap curve is incomplete for {method}"
        try:
            point_by_multiplier = {
                float(point["multiplier"]): point for point in points
            }
        except (KeyError, TypeError, ValueError):
            return False, f"driven-gap curve is malformed for {method}"
        if set(point_by_multiplier) != multipliers:
            return False, f"driven-gap multiplier grid is incomplete for {method}"
        for multiplier, point in point_by_multiplier.items():
            values = [
                float(row["value"])
                for row in gap_rows
                if row["method"] == method
                and math.isclose(float(row["mult"]), multiplier)
            ]
            mean = sum(values) / len(values)
            se = (
                math.sqrt(
                    sum((value - mean) ** 2 for value in values)
                    / (len(values) - 1)
                )
                / math.sqrt(len(values))
            )
            if (
                point.get("n") != 3
                or not _close(point.get("driven_gap_mean"), mean)
                or not _close(point.get("driven_gap_se"), se)
            ):
                return False, f"driven-gap summary disagrees for {method} x{multiplier:g}"

    provenance = payload.get("raw_provenance")
    if not isinstance(provenance, list) or len(provenance) != 446:
        return False, "alternative-matching provenance does not cover all 446 rows"
    paths = []
    for entry in provenance:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"].startswith("results/review_protocol/")
            or not _is_sha256(entry.get("sha256"))
        ):
            return False, "alternative-matching provenance entry is invalid"
        paths.append(entry["path"])
    expected_paths = set()
    for row in raw["R_match"]:
        seed = int(row["seed"])
        if row["method"] == "CD_paper":
            filename = f"R_match__CD_paper_ref_s{seed}.json"
        else:
            filename = (
                f"R_match__{row['method']}_{row['mode']}_s{seed}.json"
            )
        expected_paths.add(f"results/review_protocol/{filename}")
    for row in raw["R_match2"]:
        expected_paths.add(
            "results/review_protocol/"
            f"R_match2__{row['method']}_{row['mode']}_s{int(row['seed'])}.json"
        )
    for row in raw["R_gapsweep"]:
        expected_paths.add(
            "results/review_protocol/"
            f"R_gapsweep__{row['method']}_x{float(row['mult']):g}"
            f"_s{int(row['seed'])}.json"
        )
    if set(paths) != expected_paths:
        return False, "alternative-matching provenance paths do not cover raw rows"
    return True, "complete nine-condition matching record and six 7x3 gap curves"


def _revision_nested_is_complete(payload: object) -> tuple[bool, str]:
    """Validate the disjoint-seed joint tuning control."""
    if not isinstance(payload, dict):
        return False, "nested-tuning root is not an object"
    if payload.get("status") != "complete":
        return False, f"nested-tuning status is {payload.get('status')!r}"
    expected = {"screen": 512, "selection": 192, "test": 48}
    if (
        payload.get("expected_checkpoint_counts") != expected
        or payload.get("complete_checkpoint_counts") != expected
    ):
        return False, "nested-tuning checkpoint counts are incomplete"
    if payload.get("seed_disjointness_verified") is not True:
        return False, "nested-tuning seed disjointness is not verified"
    if payload.get("selected_ridge_upper_boundary_hits") != 0:
        return False, "nested-tuning selected ridge reaches the upper boundary"
    methods = payload.get("methods")
    if not isinstance(methods, dict) or set(methods) != {
        "CD_paper",
        "B3_collective",
    }:
        return False, "nested-tuning method set is incomplete"
    for method, result in methods.items():
        if not isinstance(result, dict):
            return False, f"nested-tuning result is malformed for {method}"
        scores = result.get("test_scores_by_seed")
        if (
            not isinstance(scores, dict)
            or len(scores) != 24
            or "selected" not in result
            or "test_mean" not in result
            or "test_se" not in result
        ):
            return False, f"nested-tuning test coverage is incomplete for {method}"
    comparison = payload.get("collective_vs_local")
    if not isinstance(comparison, dict) or comparison.get("n") != 24:
        return False, "nested-tuning paired comparison is incomplete"
    return True, "complete disjoint-seed nested tuning control"


def _revision_fresh_is_complete(payload: object) -> tuple[bool, str]:
    """Validate the fresh-ensemble prospective interpolation."""
    if not isinstance(payload, dict):
        return False, "fresh-interpolation root is not an object"
    if payload.get("status") != "complete":
        return False, f"fresh-interpolation status is {payload.get('status')!r}"
    if (
        payload.get("expected_checkpoint_count") != 288
        or payload.get("complete_checkpoint_count") != 288
    ):
        return False, "fresh-interpolation checkpoint count is incomplete"
    if payload.get("seed_overlap_with_frozen_diagnostics") != []:
        return False, "fresh task seeds overlap frozen diagnostic seeds"
    if payload.get("ridge_upper_boundary_hits") != 0:
        return False, "fresh-interpolation ridge reaches the upper boundary"
    results = payload.get("results_by_N")
    if not isinstance(results, dict) or set(results) != {"4", "5"}:
        return False, "fresh-interpolation size set is incomplete"
    expected_alphas = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
    for size, result in results.items():
        if not isinstance(result, dict):
            return False, f"fresh-interpolation result is malformed at N={size}"
        summary = result.get("summary")
        if not isinstance(summary, list) or len(summary) != 6:
            return False, f"fresh-interpolation alpha grid is incomplete at N={size}"
        try:
            alphas = {float(row["alpha"]) for row in summary}
            sample_sizes = {int(row["n"]) for row in summary}
        except (KeyError, TypeError, ValueError):
            return False, f"fresh-interpolation summary is malformed at N={size}"
        if alphas != expected_alphas or sample_sizes != {24}:
            return False, f"fresh-interpolation coverage is incomplete at N={size}"
        try:
            frozen_alpha = float(result.get("frozen_selected_alpha", -1))
        except (TypeError, ValueError):
            return False, f"fresh-interpolation frozen alpha is malformed at N={size}"
        if frozen_alpha != 0.8:
            return False, f"fresh-interpolation frozen alpha changed at N={size}"
        if result.get("ridge_upper_boundary_hits") != 0:
            return False, f"fresh-interpolation ridge boundary hit at N={size}"
        for key in ("selected_alpha_vs_local", "collective_endpoint_vs_local"):
            comparison = result.get(key)
            if not isinstance(comparison, dict) or comparison.get("n") != 24:
                return False, (
                    f"fresh-interpolation {key} comparison is incomplete at N={size}"
                )
    return True, "complete 288-checkpoint fresh-ensemble interpolation"


def _revision_manifest_is_frozen(payload: object) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "manifest root is not an object"
    if payload.get("manifest_status") != "frozen_before_stage_rows":
        return False, f"manifest status is {payload.get('manifest_status')!r}"
    protocol = payload.get("protocol")
    digest = payload.get("protocol_sha256")
    if not isinstance(protocol, dict) or not isinstance(digest, str):
        return False, "manifest protocol or digest is missing"
    if not _is_sha256(digest):
        return False, "manifest protocol digest is not SHA-256 shaped"
    computed = _scientific_protocol_sha256(protocol)
    if computed != digest:
        return False, "manifest protocol digest does not authenticate protocol"
    return True, "protocol frozen and hash-authenticated before stage rows"


def _revision_parity_is_complete(
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate active and reference parity controls as one hash-linked unit."""
    active_protocol = parsed.get("paper_protocol.json")
    active = parsed.get("paper_aggregate.json")
    reference_protocol = parsed.get("paper_reference_protocol.json")
    reference = parsed.get("paper_reference_aggregate.json")
    if not all(
        isinstance(item, dict)
        for item in (active_protocol, active, reference_protocol, reference)
    ):
        return False, "parity protocol/aggregate unit is incomplete"

    active_hash = active_protocol.get("protocol_sha256")
    reference_hash = reference_protocol.get("protocol_sha256")
    if not all(
        isinstance(value, str) and len(value) == 64
        for value in (active_hash, reference_hash)
    ):
        return False, "parity protocol hashes are missing"
    try:
        int(active_hash, 16)
        int(reference_hash, 16)
    except ValueError:
        return False, "parity protocol hash is not hexadecimal"

    active_payload = active_protocol.get("protocol")
    reference_payload = reference_protocol.get("protocol")
    if not isinstance(active_payload, dict) or not isinstance(
        reference_payload, dict
    ):
        return False, "parity protocol payload is missing"
    feature_filter = active_payload.get("feature_filter")
    try:
        feature_threshold = (
            float(feature_filter.get("threshold", -1))
            if isinstance(feature_filter, dict)
            else -1.0
        )
    except (TypeError, ValueError):
        feature_threshold = -1.0
    if (
        not isinstance(feature_filter, dict)
        or feature_filter.get("fit_on") != "training rows only"
        or feature_threshold != 1e-12
    ):
        return False, "parity train-only feature floor is not sealed at 1e-12"
    if reference_payload.get("active_protocol_sha256") != active_hash:
        return False, "reference protocol is not linked to the active protocol"

    checks = (
        (active, active_hash, 96, "active"),
        (reference, reference_hash, 32, "reference"),
    )
    for aggregate, protocol_hash, expected, label in checks:
        if (
            aggregate.get("status") != "complete"
            or aggregate.get("expected_checkpoints") != expected
            or aggregate.get("complete_checkpoints") != expected
            or aggregate.get("missing_checkpoints") != []
            or aggregate.get("protocol_sha256") != protocol_hash
        ):
            return False, f"{label} parity aggregate is incomplete"
        rows = aggregate.get("raw_rows")
        if not isinstance(rows, list) or len(rows) != expected:
            return False, f"{label} parity raw-row count is incomplete"
        if any(
            not isinstance(row, dict)
            or row.get("protocol_sha256") != protocol_hash
            or row.get("status") != "complete"
            for row in rows
        ):
            return False, f"{label} parity raw row is not hash-linked and complete"
        boundary = aggregate.get("ridge_boundary_audit")
        if (
            not isinstance(boundary, dict)
            or boundary.get("upper_boundary_is_bracketed") is not True
            or boundary.get("n_unresolved_upper") != 0
        ):
            return False, f"{label} parity ridge boundary is unresolved"

    if reference.get("active_protocol_sha256") != active_hash:
        return False, "reference aggregate is not linked to the active protocol"
    active_files = {
        name: payload
        for name, payload in parsed.items()
        if name.startswith("paper__") and name.endswith(".json")
    }
    reference_files = {
        name: payload
        for name, payload in parsed.items()
        if name.startswith("paper_reference__") and name.endswith(".json")
    }
    if len(active_files) != 96 or len(reference_files) != 32:
        return False, "parity raw checkpoint files are not complete"
    for label, files, protocol_hash in (
        ("active", active_files, active_hash),
        ("reference", reference_files, reference_hash),
    ):
        if any(
            not isinstance(payload, dict)
            or payload.get("status") != "complete"
            or payload.get("protocol_sha256") != protocol_hash
            for payload in files.values()
        ):
            return False, f"{label} parity checkpoint file is not hash-linked"
    reference_rows = reference["raw_rows"]
    methods = {row.get("method") for row in reference_rows}
    if methods != {"FN", "B1_dephasing"}:
        return False, "reference parity method set is incomplete"
    counts = {
        method: sum(row.get("method") == method for row in reference_rows)
        for method in methods
    }
    if counts != {"FN": 16, "B1_dephasing": 16}:
        return False, "reference parity method coverage is not 16+16"
    summary = reference.get("summary_by_method")
    if not isinstance(summary, dict):
        return False, "reference parity summary is missing"
    dephasing = summary.get("B1_dephasing")
    try:
        dephasing_mean = (
            float(dephasing.get("selected_test_mean", 1.0))
            if isinstance(dephasing, dict)
            else 1.0
        )
    except (TypeError, ValueError):
        dephasing_mean = 1.0
    if (
        not isinstance(dephasing, dict)
        or dephasing.get("n_complete") != 16
        or abs(dephasing_mean) > 1e-12
    ):
        return False, "dephasing parity reference is not at the finite-precision floor"
    return True, "complete hash-linked 96+32 parity controls"


def _revision_scaling_is_complete(
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate the production variance-normalised scaling unit end to end."""
    protocol_file = parsed.get("paper_variance_protocol.json")
    aggregate = parsed.get("paper_variance_aggregate.json")
    if not isinstance(protocol_file, dict) or not isinstance(aggregate, dict):
        return False, "normalised-scaling protocol/aggregate unit is incomplete"
    protocol = protocol_file.get("protocol")
    protocol_hash = protocol_file.get("protocol_sha256")
    if (
        not isinstance(protocol, dict)
        or not _is_sha256(protocol_hash)
        or _scientific_protocol_sha256(protocol) != protocol_hash
    ):
        return False, "normalised-scaling protocol hash is invalid"
    if (
        aggregate.get("protocol") != protocol
        or aggregate.get("protocol_sha256") != protocol_hash
    ):
        return False, "normalised-scaling aggregate is not linked to its protocol"

    preset = protocol.get("preset")
    seeds = protocol.get("seeds")
    expected_methods = {"CD_paper", "B3_collective"}
    expected_n = {4, 5, 6, 7, 8}
    if (
        protocol.get("control") != "normalised_coupling_scaling"
        or protocol.get("schemes") != ["variance"]
        or set(protocol.get("methods", [])) != expected_methods
        or protocol.get("disjoint_from_definitive_seed_pool") is not True
        or not isinstance(preset, dict)
        or preset.get("name") != "paper"
        or preset.get("n_seeds") != 8
        or set(preset.get("n_values", [])) != expected_n
        or not isinstance(seeds, list)
        or len(seeds) != 8
        or any(type(seed) is not int for seed in seeds)
        or len(set(seeds)) != 8
    ):
        return False, "normalised-scaling production protocol is not variance-only 5x8"

    if (
        aggregate.get("status") != "complete"
        or aggregate.get("expected_checkpoints") != 80
        or aggregate.get("complete_checkpoints") != 80
        or aggregate.get("missing_checkpoints") != []
    ):
        return False, "normalised-scaling aggregate is not complete at 80/80"
    boundary = aggregate.get("ridge_boundary_audit")
    if (
        not isinstance(boundary, dict)
        or boundary.get("upper_boundary_is_bracketed") is not True
        or boundary.get("n_unresolved_upper") != 0
    ):
        return False, "normalised-scaling ridge upper boundary is unresolved"

    audit = aggregate.get("invariant_audit")
    expected_audit_checks = {
        "protocol_variance_only",
        "n_values_are_4_through_8",
        "seed_count_is_8",
        "fresh_seeds_disjoint",
        "both_methods_declared",
        "row_identities_unique",
        "paired_hashes_equal",
        "multipliers_match_formula",
        "n5_anchor_exact",
        "jump_budget_within_1e-10",
        "exact_backend_only",
    }
    if (
        not isinstance(audit, dict)
        or audit.get("production_contract_applies") is not True
        or audit.get("all_passed") is not True
        or not isinstance(audit.get("checks"), dict)
        or set(audit["checks"]) != expected_audit_checks
        or any(value is not True for value in audit["checks"].values())
        or any(
            audit.get(key) != []
            for key in (
                "pairing_violations",
                "multiplier_violations",
                "anchor_violations",
                "budget_violations",
                "backend_violations",
            )
        )
    ):
        return False, "normalised-scaling invariant audit did not pass"
    max_budget_error = _finite_float(audit.get("max_relative_jump_budget_error"))
    if max_budget_error is None or not 0.0 <= max_budget_error <= 1e-10:
        return False, "normalised-scaling invariant audit has invalid budget error"

    rows = aggregate.get("raw_rows")
    if not isinstance(rows, list) or len(rows) != 80:
        return False, "normalised-scaling aggregate does not embed 80 raw rows"
    indexed: dict[tuple[str, int, str, int], dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            return False, "normalised-scaling raw row is not an object"
        try:
            identity = (
                str(row["scheme"]),
                int(row["n_qubits"]),
                str(row["method"]),
                int(row["seed"]),
            )
        except (KeyError, TypeError, ValueError):
            return False, "normalised-scaling raw identity is malformed"
        if identity in indexed:
            return False, "normalised-scaling raw identities are duplicated"
        indexed[identity] = row
        n_qubits = identity[1]
        expected_multiplier = math.sqrt(4.0 / (n_qubits - 1))
        error = _finite_float(row.get("relative_budget_error"))
        jump_strength = _finite_float(row.get("jump_strength"))
        target_strength = _finite_float(row.get("target_jump_strength"))
        if (
            row.get("status") != "complete"
            or row.get("control") != "normalised_coupling_scaling"
            or row.get("protocol_sha256") != protocol_hash
            or row.get("backend") != "exact_sparse_expm_multiply"
            or identity[0] != "variance"
            or identity[1] not in expected_n
            or identity[2] not in expected_methods
            or identity[3] not in seeds
            or not _close(row.get("coupling_multiplier"), expected_multiplier, atol=1e-15)
            or error is None
            or not 0.0 <= error <= 1e-10
            or jump_strength is None
            or target_strength is None
            or jump_strength <= 0.0
            or target_strength <= 0.0
            or abs(jump_strength - target_strength) / target_strength > 1e-10
            or any(
                not _is_sha256(row.get(field))
                for field in (
                    "input_sha256",
                    "base_coupling_sha256",
                    "scaled_coupling_sha256",
                )
            )
        ):
            return False, "normalised-scaling raw row violates the production contract"
        for task in ("stm", "narma10"):
            result = row.get(task)
            if (
                not isinstance(result, dict)
                or _finite_float(result.get("selected_test")) is None
            ):
                return False, f"normalised-scaling {task} result is non-finite"

    expected_identities = {
        ("variance", n_qubits, method, seed)
        for n_qubits in expected_n
        for method in expected_methods
        for seed in seeds
    }
    if set(indexed) != expected_identities:
        return False, "normalised-scaling raw grid is not N=4..8 x 2 x 8"

    for n_qubits in expected_n:
        for seed in seeds:
            local = indexed[("variance", n_qubits, "CD_paper", seed)]
            collective = indexed[("variance", n_qubits, "B3_collective", seed)]
            if any(
                local[field] != collective[field]
                for field in (
                    "input_sha256",
                    "base_coupling_sha256",
                    "scaled_coupling_sha256",
                    "target_jump_strength",
                )
            ):
                return False, "normalised-scaling paired hashes/budget differ by method"

    checkpoints = {
        name: value
        for name, value in parsed.items()
        if name.startswith("paper__") and name.endswith(".json")
    }
    expected_names = {
        (
            f"paper__variance_N{n_qubits}_{method}_s{seed}.json"
        ): identity
        for identity in expected_identities
        for _, n_qubits, method, seed in (identity,)
    }
    if set(checkpoints) != set(expected_names):
        return False, "normalised-scaling checkpoint file set is not exactly 80 rows"
    for name, identity in expected_names.items():
        if checkpoints[name] != indexed[identity]:
            return False, f"normalised-scaling checkpoint disagrees with aggregate: {name}"

    summaries = aggregate.get("summary_by_scheme")
    if not isinstance(summaries, dict) or set(summaries) != {"variance"}:
        return False, "normalised-scaling summary is not variance-only"
    variance = summaries["variance"]
    if not isinstance(variance, dict):
        return False, "normalised-scaling variance summary is malformed"
    for n_qubits in expected_n:
        summary = variance.get(str(n_qubits))
        if not isinstance(summary, dict) or summary.get("n_pairs") != 8:
            return False, f"normalised-scaling summary lacks eight pairs at N={n_qubits}"
    return True, "complete hash-linked variance-only 80-row scaling control"


def _revision_primary_regularization_is_complete(
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate the complete paired seven-design, two-task ridge control."""
    protocol = parsed.get("protocol.json")
    aggregate = parsed.get("aggregate.json")
    if not isinstance(protocol, dict) or not isinstance(aggregate, dict):
        return False, "primary-regularization protocol/aggregate unit is incomplete"

    methods = {
        "CD_paper",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    }
    tasks = {"stm": "capacity", "narma": "nmse"}
    declared_methods = protocol.get("methods")
    seeds = protocol.get("seeds")
    if (
        not isinstance(declared_methods, list)
        or set(declared_methods) != methods
        or len(declared_methods) != len(methods)
        or protocol.get("reference_method") != "CD_paper"
        or not isinstance(seeds, list)
        or len(seeds) != 32
        or len(set(seeds)) != 32
        or any(type(seed) is not int for seed in seeds)
        or protocol.get("n_jobs") != 224
    ):
        return False, "primary-regularization method/seed protocol is not 7x32"
    task_protocol = protocol.get("tasks")
    if not isinstance(task_protocol, dict) or set(task_protocol) & set(tasks) != set(
        tasks
    ):
        return False, "primary-regularization task protocol is incomplete"
    readout = protocol.get("readout")
    if not isinstance(readout, dict):
        return False, "primary-regularization readout protocol is missing"
    ridges = readout.get("ridge_grid")
    try:
        ridge_values = [float(value) for value in ridges]
    except (TypeError, ValueError):
        ridge_values = []
    if (
        not isinstance(ridges, list)
        or len(ridges) < 3
        or len(ridge_values) != len(ridges)
        or any(not math.isfinite(value) or value < 0 for value in ridge_values)
        or len(set(ridge_values)) != len(ridges)
        or readout.get("feature_guard_fit_on") != "raw training rows only"
        or not _close(readout.get("feature_guard_std_threshold"), 1e-12)
        or not _close(readout.get("fixed_sensitivity_ridge"), 1e-8)
    ):
        return False, "primary-regularization ridge/feature-guard protocol drifted"
    source_environment = protocol.get("source_environment")
    if (
        not isinstance(source_environment, dict)
        or aggregate.get("source_environment_sha256")
        != protocol.get("source_environment_sha256")
        or _scientific_protocol_sha256(source_environment)
        != protocol.get("source_environment_sha256")
    ):
        return False, "primary-regularization source environment is not hash-linked"

    protocol_hash = _scientific_protocol_sha256(protocol)
    if (
        aggregate.get("status") != "complete"
        or aggregate.get("protocol_sha256") != protocol_hash
        or aggregate.get("n_jobs") != 224
    ):
        return False, "primary-regularization aggregate is not complete/hash-linked"

    invariants = aggregate.get("invariant_audit")
    if (
        not isinstance(invariants, dict)
        or invariants.get("passed") is not True
        or invariants.get("error_count") != 0
        or invariants.get("errors") != []
        or invariants.get("expected_jobs") != 224
        or invariants.get("observed_jobs") != 224
    ):
        return False, "primary-regularization invariant audit did not pass"
    boundary = aggregate.get("ridge_boundary_audit")
    if (
        not isinstance(boundary, dict)
        or boundary.get("passed") is not True
        or boundary.get("unresolved_upper_boundary_count") != 0
        or boundary.get("unresolved_upper_boundary") != []
    ):
        return False, "primary-regularization ridge boundary is unresolved"
    feature = aggregate.get("feature_guard_audit")
    by_method = feature.get("by_method") if isinstance(feature, dict) else None
    if (
        not isinstance(feature, dict)
        or feature.get("passed") is not True
        or not _close(feature.get("threshold"), 1e-12)
        or not isinstance(by_method, dict)
        or set(by_method) != methods
        or any(
            not isinstance(item, dict) or item.get("jobs") != 32
            for item in by_method.values()
        )
    ):
        return False, "primary-regularization train-only feature audit is incomplete"

    baseline_manifest = protocol.get("baseline_reproduction")
    if not isinstance(baseline_manifest, dict):
        return False, "primary-regularization sealed baseline manifest is missing"
    baseline_entries = baseline_manifest.get("entries")
    expected_baseline_keys = {
        f"{task_name}/{method}/{seed}"
        for task_name in tasks
        for method in methods
        for seed in seeds
    }
    if (
        baseline_manifest.get("group") != "results/final_protocol/A_table"
        or not _close(baseline_manifest.get("historical_ridge"), 1e-8)
        or not _close(baseline_manifest.get("absolute_tolerance"), 1e-9)
        or not _close(baseline_manifest.get("relative_tolerance"), 1e-9)
        or baseline_manifest.get("guard_exception_methods")
        != ["B1_dephasing"]
        or not isinstance(baseline_entries, dict)
        or set(baseline_entries) != expected_baseline_keys
        or len(baseline_entries) != 448
        or baseline_manifest.get("entries_sha256")
        != _scientific_protocol_sha256(baseline_entries)
    ):
        return False, "primary-regularization 448-entry baseline manifest drifted"
    for key, entry in baseline_entries.items():
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"].startswith("results/final_protocol/A_table__")
            or not _is_sha256(entry.get("sha256"))
            or _finite_float(entry.get("value")) is None
        ):
            return False, (
                "primary-regularization baseline entry is malformed for "
                f"{key}"
            )

    baseline = aggregate.get("baseline_reproduction_audit")
    if (
        not isinstance(baseline, dict)
        or baseline.get("passed") is not True
        or baseline.get("active_comparison_count") != 384
        or baseline.get("guarded_exception_count") != 64
        or baseline.get("guard_exception_methods") != ["B1_dephasing"]
        or baseline.get("violations") != []
        or not isinstance(baseline.get("active_comparisons"), list)
        or len(baseline["active_comparisons"]) != 384
        or not isinstance(baseline.get("guarded_exceptions"), list)
        or len(baseline["guarded_exceptions"]) != 64
    ):
        return False, "primary-regularization fixed-ridge baseline audit failed"
    active_methods = methods - {"B1_dephasing"}
    expected_active = {
        (task_name, method, int(seed))
        for task_name in tasks
        for method in active_methods
        for seed in seeds
    }
    expected_guarded = {
        (task_name, "B1_dephasing", int(seed))
        for task_name in tasks
        for seed in seeds
    }
    try:
        active_identities = {
            (str(item["task"]), str(item["method"]), int(item["seed"]))
            for item in baseline["active_comparisons"]
        }
        guarded_identities = {
            (str(item["task"]), str(item["method"]), int(item["seed"]))
            for item in baseline["guarded_exceptions"]
        }
    except (KeyError, TypeError, ValueError):
        return False, "primary-regularization baseline comparison identity is malformed"
    if active_identities != expected_active or guarded_identities != expected_guarded:
        return False, "primary-regularization baseline comparison coverage is incomplete"
    active_differences: dict[str, list[float]] = {task: [] for task in tasks}
    for item in baseline["active_comparisons"]:
        difference = _finite_float(item.get("difference"))
        absolute = _finite_float(item.get("absolute_difference"))
        tolerance = _finite_float(item.get("tolerance"))
        if (
            difference is None
            or absolute is None
            or tolerance is None
            or tolerance <= 0
            or not _close(absolute, abs(difference), atol=1e-15)
            or absolute > tolerance
            or item.get("within_tolerance") is not True
        ):
            return False, "primary-regularization active baseline comparison failed"
        active_differences[str(item["task"])].append(absolute)
    for item in baseline["guarded_exceptions"]:
        if (
            _finite_float(item.get("difference")) is None
            or _finite_float(item.get("absolute_difference")) is None
            or not isinstance(item.get("reason"), str)
            or not item["reason"]
        ):
            return False, "primary-regularization guarded baseline exception is malformed"
    maxima = baseline.get("maximum_active_absolute_difference_by_task")
    if (
        not isinstance(maxima, dict)
        or set(maxima) != set(tasks)
        or any(
            _finite_float(maxima.get(task_name)) is None
            or not _close(
                maxima[task_name],
                max(active_differences[task_name]),
                atol=1e-15,
            )
            or float(maxima[task_name]) > 1e-8
            for task_name in tasks
        )
    ):
        return False, "primary-regularization maximum baseline drift is not tight"

    rows = aggregate.get("rows")
    if not isinstance(rows, list) or len(rows) != 224:
        return False, "primary-regularization aggregate does not contain 224 rows"
    expected_identities = {
        (method, int(seed)) for method in methods for seed in seeds
    }
    try:
        identities = [
            (str(row["method"]), int(row["seed"])) for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        return False, "primary-regularization row identity is malformed"
    if set(identities) != expected_identities or len(set(identities)) != 224:
        return False, "primary-regularization method/seed row coverage is incomplete"
    ridge_set = set(ridge_values)
    ridge_keys = {f"{ridge:.12g}" for ridge in ridge_values}
    pairing_fields = (
        "coupling_sha256",
        "full_input_sha256",
        "post_wash_input_sha256",
        "target_sha256",
        "train_split_sha256",
        "validation_split_sha256",
        "test_split_sha256",
    )
    indexed: dict[tuple[str, int], dict] = {}
    for row in rows:
        identity = (str(row["method"]), int(row["seed"]))
        indexed[identity] = row
        task_results = row.get("task_results")
        guard = row.get("feature_guard")
        if (
            row.get("protocol_sha256") != protocol_hash
            or row.get("source_environment_sha256")
            != protocol.get("source_environment_sha256")
            or not isinstance(task_results, dict)
            or set(task_results) != set(tasks)
            or not isinstance(guard, dict)
            or guard.get("fit_on") != "training rows only"
            or not _close(guard.get("threshold"), 1e-12)
            or abs(float(row.get("jump_strength_error", math.inf))) > 1e-10
            or any(
                not _is_sha256(row.get(field))
                for field in (
                    "coupling_sha256",
                    "full_input_sha256",
                    "post_wash_input_sha256",
                    "train_split_sha256",
                    "validation_split_sha256",
                    "test_split_sha256",
                )
            )
            or not isinstance(row.get("target_sha256"), dict)
            or set(row["target_sha256"]) != set(tasks)
            or any(
                not _is_sha256(value)
                for value in row["target_sha256"].values()
            )
        ):
            return False, f"primary-regularization row contract failed for {identity}"
        for task_name, metric in tasks.items():
            result = task_results[task_name]
            if not isinstance(result, dict):
                return False, (
                    "primary-regularization task result is malformed for "
                    f"{identity}/{task_name}"
                )
            selected = _finite_float(result.get("selected_ridge"))
            validation = result.get("validation_by_ridge")
            validation_by_target = result.get(
                "validation_by_target_and_ridge"
            )
            selected_targets = result.get("selected_test_by_target")
            fixed_targets = result.get("fixed_test_by_target")
            expected_targets = 20 if task_name == "stm" else 1
            if (
                result.get("metric") != metric
                or selected is None
                or selected not in ridge_set
                or not _close(result.get("fixed_ridge"), 1e-8)
                or not isinstance(validation, dict)
                or set(validation) != ridge_keys
                or any(_finite_float(value) is None for value in validation.values())
                or not isinstance(validation_by_target, dict)
                or set(validation_by_target) != ridge_keys
                or any(
                    not isinstance(values, list)
                    or len(values) != expected_targets
                    or any(_finite_float(value) is None for value in values)
                    for values in validation_by_target.values()
                )
                or not isinstance(selected_targets, list)
                or len(selected_targets) != expected_targets
                or any(_finite_float(value) is None for value in selected_targets)
                or not isinstance(fixed_targets, list)
                or len(fixed_targets) != expected_targets
                or any(_finite_float(value) is None for value in fixed_targets)
                or any(
                    _finite_float(result.get(field)) is None
                    for field in (
                        "selected_test",
                        "fixed_test",
                        "selected_minus_fixed",
                        "selection_improvement",
                    )
                )
            ):
                return False, (
                    "primary-regularization task result is malformed for "
                    f"{identity}/{task_name}"
                )
            validation_values = {
                float(key): float(value) for key, value in validation.items()
            }
            best = (
                max(validation_values.values())
                if metric == "capacity"
                else min(validation_values.values())
            )
            expected_selected = max(
                ridge
                for ridge, value in validation_values.items()
                if math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)
            )
            target_totals = {
                float(key): (
                    math.fsum(float(value) for value in values)
                    if metric == "capacity"
                    else math.fsum(float(value) for value in values) / len(values)
                )
                for key, values in validation_by_target.items()
            }
            selected_total = (
                math.fsum(float(value) for value in selected_targets)
                if metric == "capacity"
                else math.fsum(float(value) for value in selected_targets)
                / len(selected_targets)
            )
            fixed_total = (
                math.fsum(float(value) for value in fixed_targets)
                if metric == "capacity"
                else math.fsum(float(value) for value in fixed_targets)
                / len(fixed_targets)
            )
            selected_minus_fixed = selected_total - fixed_total
            improvement = (
                selected_minus_fixed
                if metric == "capacity"
                else -selected_minus_fixed
            )
            if (
                not _close(selected, expected_selected, atol=1e-15)
                or any(
                    not _close(validation_values[ridge], total)
                    for ridge, total in target_totals.items()
                )
                or not _close(result["selected_test"], selected_total)
                or not _close(result["fixed_test"], fixed_total)
                or not _close(
                    result["selected_minus_fixed"],
                    selected_minus_fixed,
                )
                or not _close(result["selection_improvement"], improvement)
            ):
                return False, (
                    "primary-regularization selection/test derivation failed for "
                    f"{identity}/{task_name}"
                )

    for seed in seeds:
        paired = [indexed[(method, int(seed))] for method in methods]
        for field in pairing_fields:
            if len(
                {
                    json.dumps(row[field], sort_keys=True, allow_nan=False)
                    for row in paired
                }
            ) != 1:
                return False, (
                    f"primary-regularization paired field {field} differs at "
                    f"seed {seed}"
                )

    summaries = aggregate.get("task_summaries")
    if not isinstance(summaries, dict) or set(summaries) != set(tasks):
        return False, "primary-regularization task summaries are incomplete"
    for task_name, metric in tasks.items():
        summary = summaries[task_name]
        higher_is_better = metric == "capacity"
        method_summaries = (
            summary.get("method_summaries") if isinstance(summary, dict) else None
        )
        comparisons = (
            summary.get("paired_vs_uniform_local")
            if isinstance(summary, dict)
            else None
        )
        ranking = summary.get("ranking") if isinstance(summary, dict) else None
        winners = (
            summary.get("per_seed_winner_counts")
            if isinstance(summary, dict)
            else None
        )
        if (
            not isinstance(summary, dict)
            or summary.get("metric") != metric
            or summary.get("direction")
            != ("higher is better" if higher_is_better else "lower is better")
            or not isinstance(method_summaries, dict)
            or set(method_summaries) != methods
            or not isinstance(comparisons, dict)
            or set(comparisons) != methods - {"CD_paper"}
            or not isinstance(ranking, list)
            or len(ranking) != 7
            or {item.get("method") for item in ranking} != methods
            or not isinstance(winners, dict)
            or set(winners) != methods
            or sum(winners.values()) != 32
        ):
            return False, (
                f"primary-regularization {task_name} method coverage is incomplete"
            )
        expected_method_means: dict[str, float] = {}
        for method, method_summary in method_summaries.items():
            if not isinstance(method_summary, dict):
                return False, f"primary-regularization summary malformed for {method}"
            method_results = [
                indexed[(method, int(seed))]["task_results"][task_name]
                for seed in seeds
            ]
            for field in (
                "selected_test",
                "fixed_test",
                "selection_improvement",
            ):
                values = [float(result[field]) for result in method_results]
                if not _summary_mean_se_matches(
                    method_summary.get(field),
                    values,
                ):
                    return False, (
                        "primary-regularization derived summary disagrees for "
                        f"{task_name}/{method}/{field}"
                    )
            expected_method_means[method] = _mean_and_sample_se(
                [float(result["selected_test"]) for result in method_results]
            )[0]
            expected_ridge_counts = {
                f"{ridge:.12g}": sum(
                    math.isclose(
                        float(result["selected_ridge"]),
                        ridge,
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    for result in method_results
                )
                for ridge in ridge_values
            }
            if (
                method_summary.get("selected_ridge_counts")
                != expected_ridge_counts
                or method_summary.get("selection_better_than_fixed_count")
                != sum(
                    float(result["selection_improvement"]) > 0.0
                    for result in method_results
                )
                or method_summary.get("selection_equal_to_fixed_count")
                != sum(
                    math.isclose(
                        float(result["selection_improvement"]),
                        0.0,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    for result in method_results
                )
            ):
                return False, (
                    "primary-regularization selection-count summary disagrees for "
                    f"{task_name}/{method}"
                )
        local_scores = {
            int(seed): float(
                indexed[("CD_paper", int(seed))]["task_results"][task_name][
                    "selected_test"
                ]
            )
            for seed in seeds
        }
        for method, comparison in comparisons.items():
            effect = comparison.get("method_advantage_over_local")
            method_scores = {
                int(seed): float(
                    indexed[(method, int(seed))]["task_results"][task_name][
                        "selected_test"
                    ]
                )
                for seed in seeds
            }
            differences = [
                (
                    method_scores[int(seed)] - local_scores[int(seed)]
                    if higher_is_better
                    else local_scores[int(seed)] - method_scores[int(seed)]
                )
                for seed in seeds
            ]
            if (
                not isinstance(comparison, dict)
                or not _summary_mean_se_matches(effect, differences)
                or comparison.get("method_better_count")
                != sum(value > 0.0 for value in differences)
                or comparison.get("ties")
                != sum(value == 0.0 for value in differences)
            ):
                return False, (
                    f"primary-regularization {task_name} paired effect disagrees"
                )
        expected_ranking = sorted(
            methods,
            key=lambda method: (
                -expected_method_means[method]
                if higher_is_better
                else expected_method_means[method],
                method,
            ),
        )
        for expected_rank, (method, item) in enumerate(
            zip(expected_ranking, ranking),
            start=1,
        ):
            if (
                item.get("method") != method
                or item.get("rank") != expected_rank
                or not _close(
                    item.get("mean_selected_test"),
                    expected_method_means[method],
                )
            ):
                return False, (
                    f"primary-regularization {task_name} ranking disagrees"
                )
        expected_winners = {method: 0 for method in methods}
        for seed in seeds:
            ordered = sorted(
                methods,
                key=lambda method: (
                    -float(
                        indexed[(method, int(seed))]["task_results"][task_name][
                            "selected_test"
                        ]
                    )
                    if higher_is_better
                    else float(
                        indexed[(method, int(seed))]["task_results"][task_name][
                            "selected_test"
                        ]
                    ),
                    method,
                ),
            )
            expected_winners[ordered[0]] += 1
        if winners != expected_winners:
            return False, (
                f"primary-regularization {task_name} winner counts disagree"
            )

    job_files = {
        name: value
        for name, value in parsed.items()
        if name.startswith("jobs/") and name.endswith(".json")
    }
    expected_names = {
        f"jobs/{method}__s{seed}.json": (method, int(seed))
        for method, seed in expected_identities
    }
    if set(job_files) != set(expected_names):
        return False, "primary-regularization job file set is not exactly 224"
    for name, identity in expected_names.items():
        if job_files[name] != indexed[identity]:
            return False, f"primary-regularization job disagrees with aggregate: {name}"
    return True, "complete hash-linked 7x32 STM/NARMA regularization control"


def _collective_loss_full_input_is_complete(
    directory: Path,
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate the 210-point full-spectrum collective-loss diagnostic."""
    frozen = parsed.get("protocol.json")
    raw = parsed.get("raw_spectrum.json")
    aggregate = parsed.get("aggregate.json")
    if not all(isinstance(item, dict) for item in (frozen, raw, aggregate)):
        return False, "collective full-input protocol/raw/aggregate unit is incomplete"
    protocol = frozen.get("protocol")
    protocol_hash = frozen.get("protocol_sha256")
    if (
        frozen.get("artifact_type") != "collective_loss_full_input_protocol"
        or frozen.get("status") != "frozen_before_diagnostic_rows"
        or not isinstance(protocol, dict)
        or not _is_sha256(protocol_hash)
        or _scientific_protocol_sha256(protocol) != protocol_hash
    ):
        return False, "collective full-input protocol is not frozen/hash-authenticated"
    seeds = protocol.get("seeds")
    s_grid = protocol.get("s_grid")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 10
        or len(set(seeds)) != 10
        or protocol.get("seed_count") != 10
        or not isinstance(s_grid, list)
        or len(s_grid) != 21
        or protocol.get("s_grid_count") != 21
        or any(not _close(value, index / 20.0) for index, value in enumerate(s_grid))
        or protocol.get("liouvillian_dimension") != 1024
    ):
        return False, "collective full-input seed/input grid is not 10x21"
    dense = protocol.get("dense_solver")
    sparse = protocol.get("sparse_crosscheck")
    cases = sparse.get("cases") if isinstance(sparse, dict) else None
    if (
        not isinstance(dense, dict)
        or not isinstance(sparse, dict)
        or not isinstance(cases, list)
        or len(cases) != 6
    ):
        return False, "collective full-input solver audit protocol is incomplete"
    try:
        sparse_cases = {
            (int(item["seed"]), float(item["s"])) for item in cases
        }
    except (KeyError, TypeError, ValueError):
        return False, "collective full-input sparse case declaration is malformed"
    if len(sparse_cases) != 6:
        return False, "collective full-input sparse case declaration is duplicated"

    if (
        raw.get("artifact_type") != "collective_loss_full_input_raw_spectrum"
        or raw.get("protocol_sha256") != protocol_hash
        or raw.get("row_count") != 210
    ):
        return False, "collective full-input raw payload is not hash-linked/complete"
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != 210:
        return False, "collective full-input raw grid does not contain 210 rows"
    expected = {
        (int(seed), index)
        for seed in seeds
        for index in range(len(s_grid))
    }
    try:
        identities = [(int(row["seed"]), int(row["s_index"])) for row in rows]
    except (KeyError, TypeError, ValueError):
        return False, "collective full-input row identity is malformed"
    if set(identities) != expected or len(set(identities)) != 210:
        return False, "collective full-input raw identities do not cover 10x21"

    stationary_tol = _finite_float(dense.get("stationary_abs_tolerance"))
    gap_tol = _finite_float(dense.get("minimum_gap_tolerance"))
    leakage_tol = _finite_float(dense.get("positive_real_tolerance"))
    residual_tol = _finite_float(dense.get("relative_residual_tolerance"))
    trace_tol = _finite_float(dense.get("trace_preservation_tolerance"))
    sparse_matrix_tol = _finite_float(sparse.get("matrix_tolerance"))
    sparse_eigen_tol = _finite_float(sparse.get("eigenvalue_tolerance"))
    sparse_residual_tol = _finite_float(sparse.get("relative_residual_tolerance"))
    if any(
        value is None or value < 0
        for value in (
            stationary_tol,
            gap_tol,
            leakage_tol,
            residual_tol,
            trace_tol,
            sparse_matrix_tol,
            sparse_eigen_tol,
            sparse_residual_tol,
        )
    ):
        return False, "collective full-input numerical tolerances are invalid"

    indexed: dict[tuple[int, int], dict] = {}
    observed_sparse: set[tuple[int, float]] = set()
    gaps: list[float] = []
    coupling_by_seed: dict[int, str] = {}
    for row in rows:
        identity = (int(row["seed"]), int(row["s_index"]))
        indexed[identity] = row
        s_value = _finite_float(row.get("s"))
        gap = _finite_float(row.get("first_nonstationary_decay_gap"))
        leakage = _finite_float(row.get("positive_real_part_leakage"))
        residual = _finite_float(row.get("max_all_mode_relative_residual"))
        trace = _finite_float(row.get("trace_preservation_relative_residual"))
        budget = _finite_float(row.get("relative_jump_budget_error"))
        spectrum = row.get("spectrum")
        if (
            s_value is None
            or not _close(s_value, s_grid[identity[1]])
            or row.get("eigenvalue_count") != 1024
            or row.get("stationary_mode_count") != 1
            or gap is None
            or gap <= gap_tol
            or leakage is None
            or leakage > leakage_tol
            or residual is None
            or residual > residual_tol
            or trace is None
            or trace > trace_tol
            or budget is None
            or budget > 1e-14
            or not _is_sha256(row.get("coupling_sha256"))
            or not _is_sha256(row.get("spectrum_sha256"))
            or not isinstance(spectrum, list)
            or len(spectrum) != 1024
        ):
            return False, f"collective full-input dense audit failed at {identity}"
        packed = bytearray()
        complex_pairs: list[list[float]] = []
        for value in spectrum:
            if (
                not isinstance(value, list)
                or len(value) != 2
                or _finite_float(value[0]) is None
                or _finite_float(value[1]) is None
            ):
                return False, f"collective full-input spectrum is malformed at {identity}"
            pair = [float(value[0]), float(value[1])]
            complex_pairs.append(pair)
            packed.extend(struct.pack("<dd", *pair))
        if sha256_bytes(bytes(packed)) != row["spectrum_sha256"]:
            return False, f"collective full-input spectrum hash failed at {identity}"
        stationary_pairs = [
            pair
            for pair in complex_pairs
            if math.hypot(pair[0], pair[1]) <= stationary_tol
        ]
        nonstationary_pairs = [
            pair
            for pair in complex_pairs
            if math.hypot(pair[0], pair[1]) > stationary_tol
        ]
        derived_spectral_abscissa = max(pair[0] for pair in complex_pairs)
        derived_gap_pair = nonstationary_pairs[0]
        if (
            len(stationary_pairs) != 1
            or row.get("stationary_eigenvalues") != stationary_pairs
            or not _close(
                row.get("stationary_abs_max"),
                max(math.hypot(*pair) for pair in stationary_pairs),
                atol=1e-15,
            )
            or row.get("first_nonstationary_eigenvalue")
            != derived_gap_pair
            or not _close(gap, -derived_gap_pair[0], atol=1e-15)
            or not _close(
                row.get("spectral_abscissa"),
                derived_spectral_abscissa,
                atol=1e-15,
            )
            or not _close(
                leakage,
                max(0.0, derived_spectral_abscissa),
                atol=1e-15,
            )
            or _finite_float(
                row.get("first_nonstationary_relative_residual")
            )
            is None
            or float(row["first_nonstationary_relative_residual"]) > residual
        ):
            return False, (
                "collective full-input spectral fields disagree with the "
                f"authenticated spectrum at {identity}"
            )
        previous_coupling = coupling_by_seed.setdefault(
            identity[0],
            str(row["coupling_sha256"]),
        )
        if previous_coupling != row["coupling_sha256"]:
            return False, (
                "collective full-input coupling hash changes across the input "
                f"grid for seed {identity[0]}"
            )
        gaps.append(gap)

        crosscheck = row.get("sparse_crosscheck")
        case = (identity[0], float(s_value))
        if case in sparse_cases:
            if not isinstance(crosscheck, dict):
                return False, f"collective full-input sparse check missing at {case}"
            observed_sparse.add(case)
            values = (
                crosscheck.get("dense_sparse_matrix_max_abs_difference"),
                crosscheck.get("near_zero_max_dense_eigenvalue_abs_difference"),
                crosscheck.get("targeted_gap_eigenvalue_abs_difference"),
                crosscheck.get("near_zero_max_relative_residual"),
                crosscheck.get("targeted_gap_relative_residual"),
            )
            if (
                crosscheck.get("near_zero_sparse_stationary_count") != 1
                or any(_finite_float(value) is None for value in values)
                or float(values[0]) > sparse_matrix_tol
                or max(float(values[1]), float(values[2])) > sparse_eigen_tol
                or max(float(values[3]), float(values[4])) > sparse_residual_tol
            ):
                return False, f"collective full-input sparse audit failed at {case}"
        elif crosscheck is not None:
            return False, f"collective full-input has an undeclared sparse check at {case}"
    if observed_sparse != sparse_cases:
        return False, "collective full-input sparse cross-check coverage is not six"

    protocol_path = directory / "protocol.json"
    raw_path = directory / "raw_spectrum.json"
    if (
        aggregate.get("artifact_type")
        != "collective_loss_full_input_aggregate"
        or aggregate.get("status") != "complete"
        or aggregate.get("all_declared_checks_passed") is not True
        or aggregate.get("protocol_sha256") != protocol_hash
        or aggregate.get("protocol_file_sha256") != sha256_file(protocol_path)
        or aggregate.get("raw_payload_sha256")
        != _scientific_protocol_sha256(raw)
        or aggregate.get("raw_file_sha256") != sha256_file(raw_path)
        or aggregate.get("row_count") != 210
        or aggregate.get("seed_count") != 10
        or aggregate.get("s_grid_count") != 21
        or aggregate.get("full_grid_complete") is not True
        or aggregate.get("all_rows_have_unique_stationary_mode") is not True
        or aggregate.get("unique_stationary_mode_rows") != 210
        or aggregate.get("sparse_crosscheck_count") != 6
    ):
        return False, "collective full-input aggregate/hash chain is incomplete"
    aggregate_checks = (
        (
            aggregate.get("maximum_positive_real_part_leakage"),
            leakage_tol,
        ),
        (aggregate.get("maximum_dense_relative_residual"), residual_tol),
        (
            aggregate.get("maximum_trace_preservation_relative_residual"),
            trace_tol,
        ),
        (
            aggregate.get("maximum_sparse_matrix_abs_difference"),
            sparse_matrix_tol,
        ),
        (
            aggregate.get("maximum_sparse_dense_eigenvalue_abs_difference"),
            sparse_eigen_tol,
        ),
        (
            aggregate.get("maximum_sparse_relative_residual"),
            sparse_residual_tol,
        ),
    )
    if any(
        _finite_float(value) is None or float(value) > tolerance
        for value, tolerance in aggregate_checks
    ):
        return False, "collective full-input aggregate residual audit failed"
    minimum = aggregate.get("minimum_sampled_gap")
    maximum = aggregate.get("maximum_sampled_gap")
    minimum_row = rows[gaps.index(min(gaps))]
    maximum_row = rows[gaps.index(max(gaps))]

    def extreme_record(row: dict) -> dict[str, object]:
        return {
            "seed": int(row["seed"]),
            "s": float(row["s"]),
            "gap": float(row["first_nonstationary_decay_gap"]),
            "eigenvalue": list(row["first_nonstationary_eigenvalue"]),
        }

    if (
        not isinstance(minimum, dict)
        or not isinstance(maximum, dict)
        or minimum != extreme_record(minimum_row)
        or maximum != extreme_record(maximum_row)
        or float(minimum["gap"]) <= gap_tol
    ):
        return False, "collective full-input aggregate gap extrema disagree"
    crosscheck_rows = [
        {
            "seed": row["seed"],
            "s": row["s"],
            **row["sparse_crosscheck"],
        }
        for row in rows
        if row["sparse_crosscheck"] is not None
    ]
    expected_aggregate_values = {
        "mean_sampled_gap": math.fsum(gaps) / len(gaps),
        "maximum_positive_real_part_leakage": max(
            float(row["positive_real_part_leakage"]) for row in rows
        ),
        "maximum_dense_relative_residual": max(
            float(row["max_all_mode_relative_residual"]) for row in rows
        ),
        "maximum_trace_preservation_relative_residual": max(
            float(row["trace_preservation_relative_residual"]) for row in rows
        ),
        "maximum_relative_jump_budget_error": max(
            float(row["relative_jump_budget_error"]) for row in rows
        ),
        "maximum_sparse_matrix_abs_difference": max(
            float(item["dense_sparse_matrix_max_abs_difference"])
            for item in crosscheck_rows
        ),
        "maximum_sparse_dense_eigenvalue_abs_difference": max(
            max(
                float(item["near_zero_max_dense_eigenvalue_abs_difference"]),
                float(item["targeted_gap_eigenvalue_abs_difference"]),
            )
            for item in crosscheck_rows
        ),
        "maximum_sparse_relative_residual": max(
            max(
                float(item["near_zero_max_relative_residual"]),
                float(item["targeted_gap_relative_residual"]),
            )
            for item in crosscheck_rows
        ),
    }
    if any(
        not _close(aggregate.get(field), expected)
        for field, expected in expected_aggregate_values.items()
    ):
        return False, "collective full-input aggregate summary disagrees with raw rows"
    per_seed = aggregate.get("per_seed")
    expected_per_seed = []
    for seed in seeds:
        selected_rows = [
            indexed[(int(seed), index)] for index in range(len(s_grid))
        ]
        selected_gaps = [
            float(row["first_nonstationary_decay_gap"])
            for row in selected_rows
        ]
        seed_min = selected_rows[selected_gaps.index(min(selected_gaps))]
        seed_max = selected_rows[selected_gaps.index(max(selected_gaps))]
        expected_per_seed.append(
            {
                "seed": int(seed),
                "coupling_sha256": coupling_by_seed[int(seed)],
                "unique_stationary_mode_rows": 21,
                "grid_rows": 21,
                "minimum_gap": extreme_record(seed_min),
                "maximum_gap": extreme_record(seed_max),
                "mean_gap": math.fsum(selected_gaps) / len(selected_gaps),
                "maximum_positive_real_part_leakage": max(
                    float(row["positive_real_part_leakage"])
                    for row in selected_rows
                ),
                "maximum_dense_relative_residual": max(
                    float(row["max_all_mode_relative_residual"])
                    for row in selected_rows
                ),
            }
        )
    if not isinstance(per_seed, list) or len(per_seed) != len(expected_per_seed):
        return False, "collective full-input per-seed coverage is incomplete"
    for observed, expected_seed in zip(per_seed, expected_per_seed):
        if (
            not isinstance(observed, dict)
            or any(
                observed.get(field) != expected_seed[field]
                for field in (
                    "seed",
                    "coupling_sha256",
                    "unique_stationary_mode_rows",
                    "grid_rows",
                    "minimum_gap",
                    "maximum_gap",
                )
            )
            or any(
                not _close(observed.get(field), expected_seed[field])
                for field in (
                    "mean_gap",
                    "maximum_positive_real_part_leakage",
                    "maximum_dense_relative_residual",
                )
            )
        ):
            return False, "collective full-input per-seed coverage is incomplete"
    aggregate_sparse = aggregate.get("sparse_crosschecks")
    if aggregate_sparse != crosscheck_rows:
        return False, "collective full-input aggregate sparse coverage is incomplete"
    return True, "complete hash-linked 10x21 full-spectrum collective diagnostic"


def _nested_operating_point_extension_is_complete(
    directory: Path,
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate common-grid calibration, freeze, and fresh-test linkage."""
    stage = "nested_operating_point_extension"
    prefix = f"{stage}/"
    manifest = parsed.get(f"{prefix}manifest.json")
    ledger = parsed.get(f"{prefix}seed_ledger.json")
    reuse = parsed.get(f"{prefix}reuse_index.json")
    shortlist = parsed.get(f"{prefix}screen_shortlist.json")
    frozen = parsed.get(f"{prefix}frozen_selection.json")
    aggregate = parsed.get(f"{prefix}aggregate.json")
    snapshot = parsed.get(f"{prefix}source_snapshot/manifest.json")
    if not all(
        isinstance(item, dict)
        for item in (
            manifest,
            ledger,
            reuse,
            shortlist,
            frozen,
            aggregate,
            snapshot,
        )
    ):
        return False, "nested extension control/artifact chain is incomplete"

    protocol = manifest.get("protocol")
    protocol_hash = manifest.get("protocol_sha256")
    if (
        manifest.get("artifact_type")
        != "nested_operating_point_extension_manifest"
        or manifest.get("status") != "frozen_before_new_rows"
        or not isinstance(protocol, dict)
        or not _is_sha256(protocol_hash)
        or _scientific_protocol_sha256(protocol) != protocol_hash
    ):
        return False, "nested extension manifest is not frozen/hash-authenticated"
    details = protocol.get("details")
    if not isinstance(details, dict) or details.get("channels") != [
        "CD_paper",
        "B3_collective",
    ]:
        return False, "nested extension channel protocol is not local/collective"
    h_grid = details.get("h_grid")
    dt_grid = details.get("dt_grid")
    mandatory_strengths = details.get("mandatory_common_strength_grid")
    ridges = details.get("ridge_grid")
    try:
        ridge_values = [float(value) for value in ridges]
    except (TypeError, ValueError):
        ridge_values = []
    if (
        h_grid != [0.25, 0.5, 1.0, 2.0]
        or dt_grid != [0.1, 0.25, 0.5, 1.0]
        or mandatory_strengths
        != [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        or not isinstance(ridges, list)
        or len(ridges) < 3
        or len(ridge_values) != len(ridges)
        or any(not math.isfinite(value) or value < 0 for value in ridge_values)
        or len(set(ridge_values)) != len(ridge_values)
        or details.get("screen_seed_count") != 2
        or details.get("selection_seed_count") != 12
        or details.get("fresh_test_seed_count") != 24
        or details.get("shortlist_per_channel") != 8
    ):
        return False, "nested extension declared grid/split dimensions drifted"

    old_manifest_path = directory / "nested_tuning/manifest.json"
    if not old_manifest_path.is_file():
        return False, "nested extension sealed source manifest is absent"
    old_manifest = parsed.get("nested_tuning/manifest.json")
    if not isinstance(old_manifest, dict):
        return False, "nested extension sealed source manifest is not selected"
    old_manifest_sha = sha256_file(old_manifest_path)
    if (
        ledger.get("artifact_type") != "nested_extension_seed_ledger"
        or ledger.get("source_manifest_sha256") != old_manifest_sha
        or protocol.get("source_manifest_sha256") != old_manifest_sha
        or protocol.get("source_protocol_sha256")
        != old_manifest.get("protocol_sha256")
    ):
        return False, "nested extension source-manifest hash chain is broken"
    screen_seeds = ledger.get("reused_screen_seeds")
    selection_seeds = ledger.get("reused_selection_seeds")
    old_test_seeds = ledger.get("known_old_test_seeds")
    fresh_seeds = ledger.get("fresh_test_seeds")
    excluded = ledger.get("excluded_seeds")
    if (
        not isinstance(screen_seeds, list)
        or len(screen_seeds) != 2
        or len(set(screen_seeds)) != 2
        or not isinstance(selection_seeds, list)
        or len(selection_seeds) != 12
        or len(set(selection_seeds)) != 12
        or not isinstance(old_test_seeds, list)
        or len(old_test_seeds) != 24
        or not isinstance(fresh_seeds, list)
        or len(fresh_seeds) != 24
        or len(set(fresh_seeds)) != 24
        or not isinstance(excluded, list)
        or ledger.get("pairwise_disjoint_verified") is not True
        or set(fresh_seeds) & set(excluded)
        or set(screen_seeds) & set(selection_seeds)
        or set(screen_seeds) & set(old_test_seeds)
        or set(selection_seeds) & set(old_test_seeds)
        or ledger.get("fresh_test_seeds_sha256")
        != _scientific_protocol_sha256(fresh_seeds)
        or ledger.get("excluded_seeds_sha256")
        != _scientific_protocol_sha256(excluded)
        or protocol.get("seed_hashes", {}).get("fresh_test_seeds_sha256")
        != ledger.get("fresh_test_seeds_sha256")
        or protocol.get("seed_hashes", {}).get("excluded_seeds_sha256")
        != ledger.get("excluded_seeds_sha256")
    ):
        return False, "nested extension seed-disjointness/hash ledger failed"

    source_snapshot_path = (
        directory
        / stage
        / "source_snapshot/run_nested_operating_point_extension.py"
    )
    expected_snapshot_path = (
        "results/revision_tuning/nested_operating_point_extension/"
        "source_snapshot/run_nested_operating_point_extension.py"
    )
    source_hashes = protocol.get("scientific_sources_sha256")
    if (
        snapshot.get("artifact_type") != "nested_extension_source_snapshot"
        or snapshot.get("protocol_sha256") != protocol_hash
        or snapshot.get("path") != expected_snapshot_path
        or not source_snapshot_path.is_file()
        or source_snapshot_path.is_symlink()
        or snapshot.get("sha256") != sha256_file(source_snapshot_path)
        or not isinstance(source_hashes, dict)
        or source_hashes.get(
            "experiments/run_nested_operating_point_extension.py"
        )
        != snapshot.get("sha256")
    ):
        return False, "nested extension source snapshot is not hash-linked"

    realized = frozen.get("realized_common_strength_grid")
    if (
        not isinstance(realized, list)
        or len(realized) < len(mandatory_strengths)
        or realized[: len(mandatory_strengths)] != mandatory_strengths
        or any(
            float(right) <= float(left)
            for left, right in zip(realized, realized[1:])
        )
        or any(
            not _close(float(right), 2.0 * float(left))
            for left, right in zip(
                realized[len(mandatory_strengths) - 1 :],
                realized[len(mandatory_strengths) :],
            )
        )
        or float(realized[-1])
        > float(details.get("maximum_adaptive_strength", 0))
    ):
        return False, "nested extension realized strength grid violates doubling rule"
    methods = {"CD_paper", "B3_collective"}
    if (
        shortlist.get("artifact_type") != "nested_extension_screen_shortlist"
        or shortlist.get("protocol_sha256") != protocol_hash
        or shortlist.get("common_grid_identical_for_channels") is not True
        or shortlist.get("realized_common_strength_grid") != realized
        or shortlist.get("screen_config_count_per_channel")
        != len(h_grid) * len(dt_grid) * len(realized)
    ):
        return False, "nested extension common screen-grid declaration is invalid"
    shortlist_by_method = shortlist.get("shortlist")
    candidates = shortlist.get("calibration_candidate_configs")
    rankings = shortlist.get("screen_ranking")
    if (
        not isinstance(shortlist_by_method, dict)
        or set(shortlist_by_method) != methods
        or any(len(value) != 8 for value in shortlist_by_method.values())
        or not isinstance(candidates, dict)
        or set(candidates) != methods
        or any(len(value) < 8 for value in candidates.values())
        or not isinstance(rankings, dict)
        or set(rankings) != methods
    ):
        return False, "nested extension shortlist/candidate coverage is incomplete"
    common_configs = {
        (float(h), float(dt), float(strength))
        for h in h_grid
        for dt in dt_grid
        for strength in realized
    }
    try:
        candidate_configs = {
            method: {tuple(map(float, value)) for value in candidates[method]}
            for method in methods
        }
        ranked_configs = {
            method: {
                tuple(map(float, item["config"])) for item in rankings[method]
            }
            for method in methods
        }
    except (KeyError, TypeError, ValueError):
        return False, "nested extension config record is malformed"
    if (
        any(not values.issubset(common_configs) for values in candidate_configs.values())
        or any(values != common_configs for values in ranked_configs.values())
    ):
        return False, "nested extension channel grids are not identical/complete"
    for method in methods:
        expected_shortlist = [
            list(map(float, item["config"]))
            for item in rankings[method]
            if item.get("ridge_upper_boundary_unresolved") is False
        ][:8]
        if (
            shortlist_by_method[method] != expected_shortlist
            or candidates[method][:8] != expected_shortlist
        ):
            return False, (
                "nested extension shortlist is not the top eight resolved "
                f"screen configs for {method}"
            )

    counts = reuse.get("counts")
    provenance_fields = (
        "screen_reused",
        "screen_new",
        "selection_reused",
        "selection_new",
    )
    if (
        reuse.get("artifact_type") != "nested_extension_row_reuse_index"
        or reuse.get("protocol_sha256") != protocol_hash
        or reuse.get("source_protocol_sha256")
        != old_manifest.get("protocol_sha256")
        or reuse.get("source_manifest_sha256") != old_manifest_sha
        or not isinstance(counts, dict)
        or any(not isinstance(reuse.get(field), list) for field in provenance_fields)
        or any(counts.get(field) != len(reuse[field]) for field in provenance_fields)
    ):
        return False, "nested extension row-reuse index/hash chain is invalid"

    provenance: dict[str, tuple[str, float, float, float, int]] = {}
    for field in provenance_fields:
        expected_stage = "screen" if field.startswith("screen") else "selection"
        expected_prefix = (
            "results/revision_tuning/nested_tuning/"
            if field.endswith("reused")
            else f"results/revision_tuning/{stage}/"
        )
        for entry in reuse[field]:
            if (
                not isinstance(entry, dict)
                or entry.get("stage") != expected_stage
                or not isinstance(entry.get("identity"), list)
                or len(entry["identity"]) != 5
                or not isinstance(entry.get("path"), str)
                or not entry["path"].startswith(expected_prefix)
                or not _is_sha256(entry.get("sha256"))
            ):
                return False, f"nested extension {field} provenance is malformed"
            try:
                identity = (
                    str(entry["identity"][0]),
                    float(entry["identity"][1]),
                    float(entry["identity"][2]),
                    float(entry["identity"][3]),
                    int(entry["identity"][4]),
                )
            except (TypeError, ValueError):
                return False, f"nested extension {field} identity is malformed"
            if entry["path"] in provenance:
                return False, "nested extension provenance path is duplicated"
            relative = PurePosixPath(entry["path"])
            try:
                result_relative = relative.relative_to("results/revision_tuning")
            except ValueError:
                return False, "nested extension provenance escapes result group"
            path = directory / result_relative
            parsed_row = parsed.get(result_relative.as_posix())
            expected_protocol_hash = (
                old_manifest.get("protocol_sha256")
                if field.endswith("reused")
                else protocol_hash
            )
            expected_split = details.get(f"{expected_stage}_split")
            if (
                not path.is_file()
                or path.is_symlink()
                or sha256_file(path) != entry["sha256"]
                or not isinstance(parsed_row, dict)
                or parsed_row.get("stage") != expected_stage
                or parsed_row.get("method") != identity[0]
                or not _close(parsed_row.get("h"), identity[1])
                or not _close(parsed_row.get("dt"), identity[2])
                or not _close(
                    parsed_row.get("strength_multiplier"),
                    identity[3],
                )
                or parsed_row.get("seed") != identity[4]
                or parsed_row.get("protocol_sha256")
                != expected_protocol_hash
                or not isinstance(expected_split, dict)
                or parsed_row.get("split") != expected_split
            ):
                return False, (
                    "nested extension provenance row content/hash does not "
                    f"match {entry['path']}"
                )
            provenance[entry["path"]] = identity

    screen_identities = {
        identity
        for field in ("screen_reused", "screen_new")
        for identity in (
            (
                str(entry["identity"][0]),
                float(entry["identity"][1]),
                float(entry["identity"][2]),
                float(entry["identity"][3]),
                int(entry["identity"][4]),
            )
            for entry in reuse[field]
        )
    }
    expected_screen = {
        (method, *config, int(seed))
        for method in methods
        for config in common_configs
        for seed in screen_seeds
    }
    selection_identities = {
        identity
        for field in ("selection_reused", "selection_new")
        for identity in (
            (
                str(entry["identity"][0]),
                float(entry["identity"][1]),
                float(entry["identity"][2]),
                float(entry["identity"][3]),
                int(entry["identity"][4]),
            )
            for entry in reuse[field]
        )
    }
    expected_selection = {
        (method, *config, int(seed))
        for method in methods
        for config in candidate_configs[method]
        for seed in selection_seeds
    }
    if (
        screen_identities != expected_screen
        or sum(counts[field] for field in ("screen_reused", "screen_new"))
        != len(expected_screen)
    ):
        return False, "nested extension common screen row coverage is incomplete"
    if (
        selection_identities != expected_selection
        or sum(counts[field] for field in ("selection_reused", "selection_new"))
        != len(expected_selection)
    ):
        return False, "nested extension selection row coverage is incomplete"

    stage_directory = directory / stage
    shortlist_path = stage_directory / "screen_shortlist.json"
    reuse_path = stage_directory / "reuse_index.json"
    frozen_path = stage_directory / "frozen_selection.json"
    ledger_path = stage_directory / "seed_ledger.json"
    if (
        frozen.get("artifact_type")
        != "frozen_nested_extension_operating_points"
        or frozen.get("status") != "frozen_before_fresh_test_ensemble"
        or frozen.get("protocol_sha256") != protocol_hash
        or frozen.get("screen_shortlist_sha256") != sha256_file(shortlist_path)
        or frozen.get("reuse_index_sha256") != sha256_file(reuse_path)
        or frozen.get("fresh_test_seeds_sha256")
        != ledger.get("fresh_test_seeds_sha256")
        or frozen.get("realized_common_strength_grid") != realized
        or frozen.get("test_rows_present_at_freeze") is not False
    ):
        return False, "nested extension frozen selection/hash chain is invalid"
    chosen = frozen.get("chosen")
    bracket = frozen.get("collective_strength_bracket")
    selection_rankings = frozen.get("selection_ranking")
    if (
        not isinstance(chosen, dict)
        or set(chosen) != methods
        or not isinstance(selection_rankings, dict)
        or set(selection_rankings) != methods
        or any(
            not isinstance(selection_rankings[method], list)
            or not selection_rankings[method]
            or selection_rankings[method][0] != chosen[method]
            for method in methods
        )
        or not isinstance(bracket, dict)
        or bracket.get("bracketed") is not True
        or bracket.get("reason")
        != "strict_local_maximum_on_full_selection_ensemble"
    ):
        return False, "nested extension collective optimum is not bracketed"
    maximum_ridge = ridge_values[-1]
    for method, item in chosen.items():
        if (
            not isinstance(item, dict)
            or item.get("ridge_upper_boundary_unresolved") is not False
            or _finite_float(item.get("best_ridge")) is None
            or float(item["best_ridge"]) not in set(ridge_values)
            or float(item["best_ridge"]) == maximum_ridge
            or tuple(map(float, item.get("config", ())))
            not in candidate_configs[method]
        ):
            return False, f"nested extension chosen ridge/config is invalid for {method}"
    try:
        lower = bracket["lower"]
        selected = bracket["selected"]
        upper = bracket["upper"]
        lower_config = tuple(map(float, lower["config"]))
        selected_config = tuple(map(float, selected["config"]))
        upper_config = tuple(map(float, upper["config"]))
        strength_index = list(map(float, realized)).index(selected_config[2])
    except (KeyError, TypeError, ValueError, IndexError):
        return False, "nested extension collective bracket record is malformed"
    if (
        strength_index == 0
        or strength_index == len(realized) - 1
        or lower_config[:2] != selected_config[:2]
        or upper_config[:2] != selected_config[:2]
        or not _close(lower_config[2], realized[strength_index - 1])
        or not _close(upper_config[2], realized[strength_index + 1])
        or selected_config
        != tuple(map(float, chosen["B3_collective"]["config"]))
        or selected != chosen["B3_collective"]
        or lower not in selection_rankings["B3_collective"]
        or selected not in selection_rankings["B3_collective"]
        or upper not in selection_rankings["B3_collective"]
        or float(selected["mean_validation_mc"])
        <= float(lower["mean_validation_mc"])
        or float(selected["mean_validation_mc"])
        <= float(upper["mean_validation_mc"])
    ):
        return False, "nested extension selected collective point is not bracketed"

    if (
        aggregate.get("artifact_type")
        != "nested_operating_point_extension_results"
        or aggregate.get("status") != "complete"
        or aggregate.get("protocol_sha256") != protocol_hash
        or aggregate.get("selection_sha256") != sha256_file(frozen_path)
        or aggregate.get("seed_ledger_sha256") != sha256_file(ledger_path)
        or aggregate.get("reuse_index_sha256") != sha256_file(reuse_path)
        or aggregate.get("screen_shortlist_sha256")
        != sha256_file(shortlist_path)
        or aggregate.get("realized_common_strength_grid") != realized
        or aggregate.get("common_grid_identical_for_channels") is not True
        or aggregate.get("collective_strength_bracket") != bracket
        or aggregate.get("seed_disjointness_verified") is not True
        or aggregate.get("freeze_before_test_verified") is not True
        or aggregate.get("selected_ridge_upper_boundary_hits") != 0
    ):
        return False, "nested extension aggregate/freeze hash chain is invalid"
    stored_digest = aggregate.get("deterministic_payload_sha256")
    unhashed = {
        key: value
        for key, value in aggregate.items()
        if key != "deterministic_payload_sha256"
    }
    if (
        not _is_sha256(stored_digest)
        or _scientific_protocol_sha256(unhashed) != stored_digest
    ):
        return False, "nested extension deterministic aggregate hash is invalid"
    coverage = aggregate.get("coverage")
    expected_screen_count = len(expected_screen)
    expected_selection_count = len(expected_selection)
    if (
        not isinstance(coverage, dict)
        or coverage.get("screen_expected") != expected_screen_count
        or coverage.get("screen_complete") != expected_screen_count
        or coverage.get("screen_reused") != counts.get("screen_reused")
        or coverage.get("screen_new") != counts.get("screen_new")
        or coverage.get("selection_expected") != expected_selection_count
        or coverage.get("selection_complete") != expected_selection_count
        or coverage.get("selection_reused") != counts.get("selection_reused")
        or coverage.get("selection_new") != counts.get("selection_new")
        or coverage.get("fresh_test_expected") != 48
        or coverage.get("fresh_test_complete") != 48
    ):
        return False, "nested extension aggregate row coverage is incomplete"
    aggregate_methods = aggregate.get("methods")
    if not isinstance(aggregate_methods, dict) or set(aggregate_methods) != methods:
        return False, "nested extension fresh-test method coverage is incomplete"
    score_maps: dict[str, dict[int, float]] = {}
    for method in methods:
        item = aggregate_methods[method]
        raw_scores = (
            item.get("fresh_test_scores_by_seed")
            if isinstance(item, dict)
            else None
        )
        if (
            not isinstance(item, dict)
            or item.get("selected") != chosen[method]
            or not isinstance(raw_scores, dict)
            or len(raw_scores) != 24
        ):
            return False, f"nested extension fresh scores are incomplete for {method}"
        try:
            scores = {
                int(seed): float(value) for seed, value in raw_scores.items()
            }
        except (TypeError, ValueError):
            return False, f"nested extension fresh scores are malformed for {method}"
        if set(scores) != set(fresh_seeds) or any(
            not math.isfinite(value) for value in scores.values()
        ):
            return False, f"nested extension fresh seed set differs for {method}"
        score_maps[method] = scores
    comparison = aggregate.get("collective_vs_local")
    differences = [
        score_maps["B3_collective"][int(seed)]
        - score_maps["CD_paper"][int(seed)]
        for seed in fresh_seeds
    ]
    if (
        not isinstance(comparison, dict)
        or comparison.get("n") != 24
        or not isinstance(comparison.get("paired_differences"), list)
        or len(comparison["paired_differences"]) != 24
        or any(
            not _close(actual, expected, atol=1e-12)
            for actual, expected in zip(
                comparison["paired_differences"],
                differences,
            )
        )
    ):
        return False, "nested extension paired fresh-test comparison is incomplete"

    test_files = {
        name: value
        for name, value in parsed.items()
        if name.startswith(f"{prefix}test_jobs/") and name.endswith(".json")
    }
    test_identities: set[tuple[str, int]] = set()
    selection_sha = sha256_file(frozen_path)
    for name, row in test_files.items():
        if not isinstance(row, dict):
            return False, f"nested extension test row is malformed: {name}"
        try:
            method = str(row["method"])
            seed = int(row["seed"])
            config = (
                float(row["h"]),
                float(row["dt"]),
                float(row["strength_multiplier"]),
            )
        except (KeyError, TypeError, ValueError):
            return False, f"nested extension test identity is malformed: {name}"
        if (
            method not in methods
            or row.get("protocol_sha256") != protocol_hash
            or row.get("selection_sha256") != selection_sha
            or config != tuple(map(float, chosen[method]["config"]))
            or not _close(row.get("ridge"), chosen[method]["best_ridge"])
            or not _close(
                row.get("test_mc"),
                score_maps[method].get(seed),
            )
        ):
            return False, f"nested extension test row is not freeze-linked: {name}"
        test_identities.add((method, seed))
    expected_test = {
        (method, int(seed)) for method in methods for seed in fresh_seeds
    }
    if len(test_files) != 48 or test_identities != expected_test:
        return False, "nested extension test job set is not exactly 2x24"

    raw_provenance = aggregate.get("raw_provenance")
    test_provenance = (
        raw_provenance.get("fresh_test_rows")
        if isinstance(raw_provenance, dict)
        else None
    )
    if not isinstance(test_provenance, list) or len(test_provenance) != 48:
        return False, "nested extension test provenance is incomplete"
    observed_test_provenance_paths: list[str] = []
    for entry in test_provenance:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not entry["path"].startswith(
                f"results/revision_tuning/{stage}/test_jobs/"
            )
            or not _is_sha256(entry.get("sha256"))
        ):
            return False, "nested extension test provenance entry is malformed"
        relative = PurePosixPath(entry["path"]).relative_to(
            "results/revision_tuning"
        )
        path = directory / relative
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            return False, "nested extension test provenance hash disagrees"
        observed_test_provenance_paths.append(entry["path"])
    expected_test_provenance_paths = {
        f"results/revision_tuning/{name}" for name in test_files
    }
    if (
        len(set(observed_test_provenance_paths))
        != len(observed_test_provenance_paths)
        or set(observed_test_provenance_paths)
        != expected_test_provenance_paths
    ):
        return False, "nested extension test provenance path set is not exact"
    return True, "complete common-grid, bracketed, freeze-before-test nested extension"


def _nested_prescreen_stability_is_complete(
    directory: Path,
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate the post-hoc two-seed prescreen stability audit."""
    stage = "nested_operating_point_extension"
    prefix = f"{stage}/"
    audit = parsed.get(f"{prefix}prescreen_stability.json")
    manifest = parsed.get(f"{prefix}manifest.json")
    ledger = parsed.get(f"{prefix}seed_ledger.json")
    reuse = parsed.get(f"{prefix}reuse_index.json")
    shortlist = parsed.get(f"{prefix}screen_shortlist.json")
    frozen = parsed.get(f"{prefix}frozen_selection.json")
    old_manifest = parsed.get("nested_tuning/manifest.json")
    if not all(
        isinstance(item, dict)
        for item in (
            audit,
            manifest,
            ledger,
            reuse,
            shortlist,
            frozen,
            old_manifest,
        )
    ):
        return False, "nested prescreen audit/provenance chain is incomplete"

    expected_claim_boundary = (
        "This describes agreement between the two realized cheap-screen "
        "reservoirs. It does not remedy the two-seed prescreen limitation, "
        "provide selection-adjusted inference, or enlarge the calibration "
        "ensemble."
    )
    stored_digest = audit.get("deterministic_payload_sha256")
    unhashed = {
        key: value
        for key, value in audit.items()
        if key != "deterministic_payload_sha256"
    }
    if (
        audit.get("artifact_type") != "nested_prescreen_stability_audit"
        or audit.get("artifact_version")
        != "nested-prescreen-stability-v1-2026-07-24"
        or audit.get("status") != "complete"
        or audit.get("analysis_type")
        != "post_hoc_descriptive_sensitivity_audit"
        or audit.get("claim_boundary") != expected_claim_boundary
        or not _is_sha256(stored_digest)
        or _scientific_protocol_sha256(unhashed) != stored_digest
    ):
        return False, (
            "nested prescreen audit type/status/hash or descriptive claim "
            "boundary is invalid"
        )

    protocol = manifest.get("protocol")
    protocol_hash = manifest.get("protocol_sha256")
    details = protocol.get("details") if isinstance(protocol, dict) else None
    methods = {"CD_paper", "B3_collective"}
    if (
        manifest.get("artifact_type")
        != "nested_operating_point_extension_manifest"
        or manifest.get("status") != "frozen_before_new_rows"
        or not isinstance(protocol, dict)
        or not _is_sha256(protocol_hash)
        or _scientific_protocol_sha256(protocol) != protocol_hash
        or not isinstance(details, dict)
        or details.get("channels") != ["CD_paper", "B3_collective"]
    ):
        return False, "nested prescreen audit protocol linkage is invalid"
    seeds = ledger.get("reused_screen_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 2
        or len(set(seeds)) != 2
        or any(type(seed) is not int for seed in seeds)
        or audit.get("screen_seed_count") != 2
        or audit.get("screen_seeds") != seeds
    ):
        return False, "nested prescreen audit is not the declared two-seed audit"
    h_grid = details.get("h_grid")
    dt_grid = details.get("dt_grid")
    realized = frozen.get("realized_common_strength_grid")
    try:
        configs = {
            (float(h), float(dt), float(strength))
            for h in h_grid
            for dt in dt_grid
            for strength in realized
        }
    except (TypeError, ValueError):
        return False, "nested prescreen audit realized grid is malformed"
    if (
        not configs
        or any(
            not all(math.isfinite(value) for value in config)
            for config in configs
        )
        or len(configs) != len(h_grid) * len(dt_grid) * len(realized)
        or shortlist.get("realized_common_strength_grid") != realized
        or audit.get("configuration_count_per_method") != len(configs)
        or audit.get("top_k") != min(8, len(configs))
    ):
        return False, "nested prescreen audit realized configuration count is invalid"

    stage_directory = directory / stage
    provenance = audit.get("provenance")
    frozen_path = stage_directory / "frozen_selection.json"
    shortlist_path = stage_directory / "screen_shortlist.json"
    old_manifest_path = directory / "nested_tuning/manifest.json"
    if (
        not isinstance(provenance, dict)
        or provenance.get("protocol_sha256") != protocol_hash
        or provenance.get("frozen_selection_sha256")
        != sha256_file(frozen_path)
        or provenance.get("screen_shortlist_sha256")
        != sha256_file(shortlist_path)
        or provenance.get("source_manifest_sha256")
        != sha256_file(old_manifest_path)
        or protocol.get("source_manifest_sha256")
        != provenance.get("source_manifest_sha256")
    ):
        return False, "nested prescreen audit protocol/frozen provenance is invalid"

    screen_entries: list[tuple[str, dict]] = []
    for field, source in (
        ("screen_reused", "sealed_reused"),
        ("screen_new", "extension_new"),
    ):
        entries = reuse.get(field)
        if not isinstance(entries, list):
            return False, "nested prescreen audit screen provenance is missing"
        for entry in entries:
            if not isinstance(entry, dict):
                return False, "nested prescreen audit screen provenance is malformed"
            screen_entries.append((source, entry))
    expected_row_count = len(methods) * len(configs) * len(seeds)
    if (
        len(screen_entries) != expected_row_count
        or provenance.get("screen_rows_expected") != expected_row_count
        or provenance.get("screen_rows_loaded") != expected_row_count
        or provenance.get("sealed_reused_rows")
        != len(reuse["screen_reused"])
        or provenance.get("extension_new_rows") != len(reuse["screen_new"])
    ):
        return False, "nested prescreen audit screen row count is incomplete"

    expected_identities = {
        (method, int(seed), config)
        for method in methods
        for seed in seeds
        for config in configs
    }
    observed_identities: set[
        tuple[str, int, tuple[float, float, float]]
    ] = set()
    rows: list[dict] = []
    row_provenance: list[dict[str, str]] = []
    for source, entry in screen_entries:
        path_text = entry.get("path")
        if (
            entry.get("stage") != "screen"
            or not isinstance(path_text, str)
            or not _is_sha256(entry.get("sha256"))
        ):
            return False, "nested prescreen audit screen provenance is malformed"
        try:
            relative = PurePosixPath(path_text).relative_to(
                "results/revision_tuning"
            )
        except ValueError:
            return False, "nested prescreen audit screen path escapes result group"
        path = directory / relative
        row = parsed.get(relative.as_posix())
        entry_identity = entry.get("identity")
        expected_protocol_hash = (
            old_manifest.get("protocol_sha256")
            if source == "sealed_reused"
            else protocol_hash
        )
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != entry["sha256"]
            or not isinstance(row, dict)
            or not isinstance(entry_identity, list)
            or len(entry_identity) != 5
            or row.get("protocol_sha256") != expected_protocol_hash
            or row.get("stage") != "screen"
            or row.get("split") != details.get("screen_split")
        ):
            return False, "nested prescreen audit screen row hash is invalid"
        try:
            identity = (
                str(row["method"]),
                int(row["seed"]),
                (
                    float(row["h"]),
                    float(row["dt"]),
                    float(row["strength_multiplier"]),
                ),
            )
        except (KeyError, TypeError, ValueError):
            return False, "nested prescreen audit screen row identity is malformed"
        try:
            indexed_identity = (
                str(entry_identity[0]),
                int(entry_identity[4]),
                (
                    float(entry_identity[1]),
                    float(entry_identity[2]),
                    float(entry_identity[3]),
                ),
            )
        except (TypeError, ValueError):
            return False, "nested prescreen audit indexed row identity is malformed"
        if indexed_identity != identity:
            return False, "nested prescreen audit indexed row identity disagrees"
        if identity in observed_identities:
            return False, "nested prescreen audit screen row identity is duplicated"
        observed_identities.add(identity)
        rows.append(row)
        row_provenance.append(
            {
                "source": source,
                "path": path_text,
                "sha256": entry["sha256"],
            }
        )
    if observed_identities != expected_identities:
        return False, "nested prescreen audit screen row coverage is incomplete"
    rows.sort(
        key=lambda row: (
            str(row["method"]),
            int(row["seed"]),
            float(row["h"]),
            float(row["dt"]),
            float(row["strength_multiplier"]),
        )
    )
    row_provenance.sort(key=lambda item: (item["source"], item["path"]))
    if (
        provenance.get("screen_rows_sha256")
        != _scientific_protocol_sha256(rows)
        or provenance.get("row_provenance_sha256")
        != _scientific_protocol_sha256(row_provenance)
    ):
        return False, "nested prescreen audit row provenance digest is invalid"

    method_payloads = audit.get("methods")
    chosen = frozen.get("chosen")
    try:
        ridge_values = {float(value) for value in details["ridge_grid"]}
    except (KeyError, TypeError, ValueError):
        return False, "nested prescreen audit ridge grid is malformed"
    if (
        not ridge_values
        or any(not math.isfinite(value) for value in ridge_values)
    ):
        return False, "nested prescreen audit ridge grid is malformed"
    if (
        not isinstance(method_payloads, dict)
        or set(method_payloads) != methods
        or not isinstance(chosen, dict)
        or set(chosen) != methods
    ):
        return False, "nested prescreen audit method keys are incomplete"

    def parsed_config(value: object) -> tuple[float, float, float] | None:
        if not isinstance(value, list) or len(value) != 3:
            return None
        try:
            config = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in config):
            return None
        return config  # type: ignore[return-value]

    top_k = int(audit["top_k"])
    for method in methods:
        result = method_payloads[method]
        selected = parsed_config(
            result.get("frozen_selected_config")
            if isinstance(result, dict)
            else None
        )
        try:
            chosen_config = tuple(
                float(value) for value in chosen[method]["config"]
            )
        except (KeyError, TypeError, ValueError):
            return False, f"nested prescreen frozen config is malformed for {method}"
        per_seed = result.get("per_seed") if isinstance(result, dict) else None
        overlap = (
            result.get("top8_overlap") if isinstance(result, dict) else None
        )
        spearman = (
            _finite_float(result.get("full_rank_spearman"))
            if isinstance(result, dict)
            else None
        )
        if (
            selected != chosen_config
            or selected not in configs
            or not isinstance(per_seed, dict)
            or set(per_seed) != {str(seed) for seed in seeds}
            or not isinstance(overlap, dict)
            or spearman is None
            or not -1.0 <= spearman <= 1.0
        ):
            return False, f"nested prescreen method summary is invalid for {method}"

        seed_top_sets: list[set[tuple[float, float, float]]] = []
        for seed in seeds:
            seed_result = per_seed[str(seed)]
            if not isinstance(seed_result, dict):
                return False, f"nested prescreen seed result is invalid for {method}"
            rank = seed_result.get("frozen_selected_config_rank")
            in_top = seed_result.get("frozen_selected_config_in_top8")
            selected_ridge = _finite_float(
                seed_result.get("frozen_selected_config_screen_best_ridge")
            )
            selected_score = _finite_float(
                seed_result.get("frozen_selected_config_validation_mc")
            )
            winner = seed_result.get("winner")
            top_rows = seed_result.get("top8")
            if (
                type(rank) is not int
                or not 1 <= rank <= len(configs)
                or type(in_top) is not bool
                or selected_ridge not in ridge_values
                or selected_score is None
                or not isinstance(winner, dict)
                or not isinstance(top_rows, list)
                or len(top_rows) != top_k
            ):
                return False, f"nested prescreen ranks are invalid for {method}/{seed}"
            top_configs: set[tuple[float, float, float]] = set()
            for expected_rank, top_row in enumerate(top_rows, start=1):
                top_config = parsed_config(
                    top_row.get("config")
                    if isinstance(top_row, dict)
                    else None
                )
                top_ridge = (
                    _finite_float(top_row.get("best_ridge"))
                    if isinstance(top_row, dict)
                    else None
                )
                top_score = (
                    _finite_float(top_row.get("validation_mc"))
                    if isinstance(top_row, dict)
                    else None
                )
                if (
                    not isinstance(top_row, dict)
                    or top_row.get("rank") != expected_rank
                    or top_config not in configs
                    or top_config in top_configs
                    or top_ridge not in ridge_values
                    or top_score is None
                ):
                    return False, (
                        f"nested prescreen top-eight rows are invalid for "
                        f"{method}/{seed}"
                    )
                top_configs.add(top_config)
            first = top_rows[0]
            if (
                winner.get("config") != first.get("config")
                or not _close(winner.get("best_ridge"), first.get("best_ridge"))
                or not _close(
                    winner.get("validation_mc"),
                    first.get("validation_mc"),
                )
                or in_top != (selected in top_configs)
                or (rank <= top_k)
                != (
                    in_top
                    and parsed_config(top_rows[rank - 1]["config"]) == selected
                )
            ):
                return False, (
                    f"nested prescreen selected/winner rank linkage is invalid "
                    f"for {method}/{seed}"
                )
            seed_top_sets.append(top_configs)

        intersection = seed_top_sets[0] & seed_top_sets[1]
        union = seed_top_sets[0] | seed_top_sets[1]
        expected_intersection = [list(config) for config in sorted(intersection)]
        expected_union = [list(config) for config in sorted(union)]
        if (
            type(overlap.get("intersection_count")) is not int
            or type(overlap.get("union_count")) is not int
            or overlap.get("intersection_count") != len(intersection)
            or overlap.get("union_count") != len(union)
            or not _close(
                overlap.get("jaccard"),
                len(intersection) / len(union),
            )
            or overlap.get("intersection_configs") != expected_intersection
            or overlap.get("union_configs") != expected_union
        ):
            return False, f"nested prescreen overlap is invalid for {method}"
    return True, "complete descriptive two-seed prescreen stability audit"


def _revision_stage_source_snapshot_is_complete(
    directory: Path,
    parsed: dict[str, object],
    stage: str,
    stage_label: str,
) -> tuple[bool, str]:
    """Verify source snapshots against one stage's frozen protocol manifest."""
    stage_manifest = parsed.get(f"{stage}/manifest.json")
    snapshot_manifest = parsed.get(f"{stage}/source_snapshot/manifest.json")
    if not isinstance(stage_manifest, dict) or not isinstance(
        snapshot_manifest, dict
    ):
        return False, f"{stage_label} source-snapshot manifests are incomplete"

    protocol = stage_manifest.get("protocol")
    source_hashes = (
        protocol.get("scientific_sources_sha256")
        if isinstance(protocol, dict)
        else None
    )
    if not isinstance(source_hashes, dict):
        return False, (
            f"{stage_label} protocol lacks scientific source hashes"
        )

    protocol_digest = stage_manifest.get("protocol_sha256")
    if (
        snapshot_manifest.get("artifact_type")
        != f"{stage_label}_stage_source_snapshot"
        or not _is_sha256(protocol_digest)
        or snapshot_manifest.get(f"{stage_label}_protocol_sha256")
        != protocol_digest
    ):
        return False, (
            f"{stage_label} source snapshot is not linked to its frozen protocol"
        )

    source_entries = [
        (
            "source_path_in_protocol",
            "snapshot_path",
            "sha256",
            "experiments/run_revision_tuning.py",
            (
                f"results/revision_tuning/{stage}/source_snapshot/"
                "run_revision_tuning.py"
            ),
        )
    ]
    if stage == "fresh_interpolation":
        source_entries.append(
            (
                "helper_path_in_protocol",
                "helper_snapshot_path",
                "helper_sha256",
                "experiments/run_revision_fresh_interpolation.py",
                (
                    "results/revision_tuning/fresh_interpolation/"
                    "source_snapshot/run_revision_fresh_interpolation.py"
                ),
            )
        )

    for (
        source_field,
        snapshot_field,
        hash_field,
        expected_source,
        expected_snapshot,
    ) in source_entries:
        snapshot = directory / PurePosixPath(expected_snapshot).relative_to(
            "results/revision_tuning"
        )
        expected_hash = source_hashes.get(expected_source)
        if (
            snapshot_manifest.get(source_field) != expected_source
            or snapshot_manifest.get(snapshot_field) != expected_snapshot
            or not _is_sha256(expected_hash)
            or snapshot_manifest.get(hash_field) != expected_hash
        ):
            return False, (
                f"{stage_label} source snapshot metadata is not hash-linked "
                f"for {expected_source}"
            )
        if not snapshot.is_file() or snapshot.is_symlink():
            return False, (
                f"{stage_label} source snapshot is missing or is a symlink: "
                f"{expected_snapshot}"
            )
        if sha256_file(snapshot) != expected_hash:
            return False, (
                f"{stage_label} source snapshot content hash differs for "
                f"{expected_source}"
            )

    return True, (
        f"{stage_label}-stage source snapshots match their recorded source hashes"
    )


def _primary_driven_activity_is_complete(
    directory: Path,
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate the complete seven-design driven-trajectory activity replay."""
    protocol = parsed.get("protocol.json")
    aggregate = parsed.get("aggregate.json")
    if not isinstance(protocol, dict) or not isinstance(aggregate, dict):
        return False, "driven-activity protocol/aggregate unit is incomplete"

    expected_methods = {
        "CD_paper",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    }
    methods = protocol.get("methods")
    seeds = protocol.get("seeds")
    split = protocol.get("split")
    if (
        protocol.get("protocol_version")
        != "primary-driven-jump-activity-posthoc-v1-2026-07-25"
        or not isinstance(methods, list)
        or set(methods) != expected_methods
        or len(methods) != 7
        or protocol.get("reference_method") != "CD_paper"
        or not isinstance(seeds, list)
        or len(seeds) != 32
        or len(set(seeds)) != 32
        or any(type(seed) is not int for seed in seeds)
        or protocol.get("n_jobs") != 224
        or protocol.get("n_qubits") != 5
        or not _close(protocol.get("h"), 0.5)
        or not _close(protocol.get("dt"), 0.5)
        or not isinstance(split, dict)
        or split.get("wash") != 200
        or split.get("train") != 600
        or split.get("test") != 400
        or split.get("activity_uses_only_test_intervals") is not True
    ):
        return False, "driven-activity protocol is not the declared 7x32 replay"

    source_environment = protocol.get("source_environment")
    if (
        not isinstance(source_environment, dict)
        or not source_environment
        or _scientific_protocol_sha256(source_environment)
        != protocol.get("source_environment_sha256")
        or aggregate.get("source_environment_sha256")
        != protocol.get("source_environment_sha256")
    ):
        return False, "driven-activity source environment is not hash-linked"
    repo_root = directory.parents[1]
    for relative, expected_sha in source_environment.items():
        if not isinstance(relative, str) or not _is_sha256(expected_sha):
            return False, "driven-activity source manifest is malformed"
        try:
            safe_relative = _safe_relative_path(relative)
        except EvidencePackageError:
            return False, "driven-activity source manifest contains an unsafe path"
        source = repo_root / Path(*safe_relative.parts)
        if (
            not source.is_file()
            or source.is_symlink()
            or sha256_file(source) != expected_sha
        ):
            return False, f"driven-activity source hash differs: {relative}"

    baseline = protocol.get("baseline_archive")
    entries = baseline.get("entries") if isinstance(baseline, dict) else None
    expected_entry_keys = {
        f"{method}/{seed}" for method in expected_methods for seed in seeds
    }
    if (
        not isinstance(baseline, dict)
        or baseline.get("path") != "results/final_protocol_results.tar.gz"
        or not _is_sha256(baseline.get("sha256"))
        or not isinstance(entries, dict)
        or set(entries) != expected_entry_keys
        or baseline.get("entries_sha256")
        != _scientific_protocol_sha256(entries)
    ):
        return False, "driven-activity sealed baseline manifest drifted"
    baseline_path = repo_root / baseline["path"]
    if (
        not baseline_path.is_file()
        or baseline_path.is_symlink()
        or sha256_file(baseline_path) != baseline["sha256"]
    ):
        return False, "driven-activity sealed baseline archive hash differs"

    protocol_hash = _scientific_protocol_sha256(protocol)
    invariants = aggregate.get("invariant_audit")
    if (
        aggregate.get("status") != "complete"
        or aggregate.get("protocol_version") != protocol.get("protocol_version")
        or aggregate.get("protocol_sha256") != protocol_hash
        or aggregate.get("n_jobs") != 224
        or aggregate.get("n_seeds") != 32
        or not isinstance(invariants, dict)
        or invariants.get("passed") is not True
        or invariants.get("errors") != []
        or invariants.get("expected_jobs") != 224
        or invariants.get("observed_jobs") != 224
        or invariants.get("all_rows_use_400_test_intervals") is not True
        or invariants.get("all_rows_link_sealed_checkpoint") is not True
    ):
        return False, "driven-activity aggregate/invariant audit is incomplete"

    checkpoints = {
        name: row
        for name, row in parsed.items()
        if name.startswith("checkpoints/") and name.endswith(".json")
    }
    expected_names = {
        f"checkpoints/{method}__seed_{seed}.json": (method, seed)
        for method in methods
        for seed in seeds
    }
    if set(checkpoints) != set(expected_names):
        return False, "driven-activity checkpoint set is not exactly 224 jobs"

    values_by_method: dict[str, list[float]] = {method: [] for method in methods}
    paired_hashes: dict[int, tuple[object, object, object]] = {}
    target_strength: float | None = None
    for name, (method, seed) in expected_names.items():
        row = checkpoints[name]
        baseline_entry = entries[f"{method}/{seed}"]
        value = (
            _finite_float(row.get("time_averaged_jump_activity"))
            if isinstance(row, dict)
            else None
        )
        if (
            not isinstance(row, dict)
            or row.get("method") != method
            or row.get("seed") != seed
            or row.get("protocol_sha256") != protocol_hash
            or row.get("source_environment_sha256")
            != protocol.get("source_environment_sha256")
            or row.get("test_intervals") != 400
            or value is None
            or value < 0
            or not _is_sha256(row.get("coupling_sha256"))
            or not _is_sha256(row.get("full_input_sha256"))
            or not _is_sha256(row.get("test_input_sha256"))
            or not _is_sha256(row.get("jump_family_sha256"))
            or row.get("baseline_checkpoint_member")
            != baseline_entry.get("member")
            or row.get("baseline_checkpoint_sha256")
            != baseline_entry.get("checkpoint_sha256")
        ):
            return False, f"driven-activity checkpoint is malformed: {name}"
        target = _finite_float(row.get("target_frobenius_jump_strength"))
        actual = _finite_float(row.get("actual_frobenius_jump_strength"))
        if (
            target is None
            or actual is None
            or not _close(target, actual)
            or _finite_float(row.get("maximum_trace_error")) is None
            or _finite_float(
                row.get("maximum_activity_imaginary_residue")
            )
            is None
            or _finite_float(row.get("minimum_integrated_interval_activity"))
            is None
        ):
            return False, f"driven-activity numerical audit is malformed: {name}"
        if target_strength is None:
            target_strength = target
        elif not _close(target_strength, target):
            return False, "driven-activity Frobenius target differs across jobs"
        hashes = (
            row["coupling_sha256"],
            row["full_input_sha256"],
            row["test_input_sha256"],
        )
        if seed in paired_hashes and paired_hashes[seed] != hashes:
            return False, "driven-activity paired coupling/input hashes differ"
        paired_hashes[seed] = hashes
        values_by_method[method].append(value)

    summaries = aggregate.get("method_summaries")
    effects = aggregate.get("paired_vs_uniform_local")
    if (
        not isinstance(summaries, dict)
        or set(summaries) != expected_methods
        or not isinstance(effects, dict)
        or set(effects) != expected_methods
    ):
        return False, "driven-activity method summaries are incomplete"
    local = values_by_method["CD_paper"]
    for method in methods:
        values = values_by_method[method]
        mean, se = _mean_and_sample_se(values)
        summary = summaries[method]
        effect = effects[method]
        differences = [
            candidate - reference
            for candidate, reference in zip(values, local)
        ]
        if (
            not isinstance(summary, dict)
            or summary.get("n") != 32
            or not _close(summary.get("mean_time_averaged_activity"), mean)
            or not _close(summary.get("standard_error"), se)
            or not isinstance(effect, dict)
            or not _close(
                effect.get("mean_activity_difference"),
                math.fsum(differences) / 32,
            )
            or not _close(
                effect.get("ratio_of_method_mean_to_local_mean"),
                mean / (math.fsum(local) / 32),
            )
            or effect.get("higher_activity_count")
            != sum(difference > 0 for difference in differences)
            or effect.get("equal_activity_count")
            != sum(difference == 0 for difference in differences)
            or effect.get("lower_activity_count")
            != sum(difference < 0 for difference in differences)
        ):
            return False, f"driven-activity summary differs from rows: {method}"

    return True, "complete hash-linked 224-job driven-trajectory activity replay"


_ACTIVITY_PROTOCOL_VERSION = "activity-matched-response-v3-2026-07-25"
_ACTIVITY_PILOT_NAMESPACE = 505
_ACTIVITY_TASK_NAMESPACE = 506
_ACTIVITY_V2_PRIOR_SOURCES = {
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
}
_ACTIVITY_DESIGNS = ("local", "collective")
_ACTIVITY_BRANCHES = {
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
_ACTIVITY_BONFERRONI_CRITICAL = 2.8073356837675227
_ACTIVITY_TRACE_TOLERANCE = 2e-9
_ACTIVITY_IMAGINARY_TOLERANCE = 2e-9
_ACTIVITY_NEGATIVE_COUNT_TOLERANCE = 2e-10
_ACTIVITY_OUTCOME_FIELDS = {
    "test_stm_capacity",
    "test_capacity_by_delay",
    "time_averaged_test_activity",
    "total_expected_test_jumps",
    "task_input_sha256",
    "frozen_calibration_sha256",
    "target_results",
    "claim_gates",
}


def _activity_require(condition: bool, message: str) -> None:
    """Fail one activity-response invariant with a concise diagnostic."""
    if not condition:
        raise ValueError(message)


def _activity_float_list(
    value: object,
    *,
    length: int | None = None,
) -> list[float]:
    """Return a finite numeric list or fail the activity validator."""
    _activity_require(isinstance(value, list), "numeric vector is missing")
    numbers = [_finite_float(item) for item in value]
    _activity_require(
        all(item is not None for item in numbers),
        "numeric vector contains a non-finite value",
    )
    if length is not None:
        _activity_require(
            len(numbers) == length,
            f"numeric vector length is not {length}",
        )
    return [float(item) for item in numbers if item is not None]


def _activity_contains_outcome_field(value: object) -> bool:
    """Detect task-result leakage anywhere in a pre-score artifact."""
    if isinstance(value, dict):
        return bool(_ACTIVITY_OUTCOME_FIELDS.intersection(value)) or any(
            _activity_contains_outcome_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_activity_contains_outcome_field(item) for item in value)
    return False


def _activity_band_matches(
    observed: object,
    values: Sequence[float],
) -> bool:
    """Check one five-contrast Bonferroni-t band from its paired values."""
    if not isinstance(observed, dict) or len(values) != 24:
        return False
    try:
        mean, standard_error = _mean_and_sample_se(values)
    except (TypeError, ValueError):
        return False
    paired = observed.get("paired_values")
    if not isinstance(paired, list) or len(paired) != len(values):
        return False
    if any(
        not _close(left, right)
        for left, right in zip(paired, values)
    ):
        return False
    nonzero = [
        value
        for value in values
        if not math.isclose(value, 0.0, rel_tol=0.0, abs_tol=1e-14)
    ]
    lower = mean - _ACTIVITY_BONFERRONI_CRITICAL * standard_error
    upper = mean + _ACTIVITY_BONFERRONI_CRITICAL * standard_error
    return (
        observed.get("n") == len(values)
        and observed.get("family_size") == 5
        and _close(observed.get("familywise_alpha"), 0.05)
        and _close(
            observed.get("critical_value"),
            _ACTIVITY_BONFERRONI_CRITICAL,
        )
        and _close(observed.get("mean"), mean)
        and _close(observed.get("standard_error"), standard_error)
        and _close(observed.get("simultaneous_lower"), lower)
        and _close(observed.get("simultaneous_upper"), upper)
        and observed.get("wins")
        == sum(value > 0.0 for value in nonzero)
        and observed.get("ties") == len(values) - len(nonzero)
        and observed.get("losses")
        == sum(value < 0.0 for value in nonzero)
    )


def _activity_matched_response_is_complete(
    directory: Path,
    parsed: dict[str, object],
) -> tuple[bool, str]:
    """Validate a scored response or its complete frozen feasibility failure."""
    try:
        pilot_manifest = parsed.get("pilot_manifest.json")
        frozen_targets = parsed.get("frozen_targets.json")
        task_manifest = parsed.get("task_manifest.json")
        frozen_calibration = parsed.get("frozen_calibration.json")
        aggregate = parsed.get("aggregate.json")
        snapshot_manifest = parsed.get("source_snapshot/manifest.json")
        _activity_require(
            all(
                isinstance(item, dict)
                for item in (
                    pilot_manifest,
                    frozen_targets,
                    task_manifest,
                    snapshot_manifest,
                )
            ),
            "common control/artifact chain is incomplete",
        )
        assert isinstance(pilot_manifest, dict)
        assert isinstance(frozen_targets, dict)
        assert isinstance(task_manifest, dict)
        assert isinstance(snapshot_manifest, dict)
        _activity_require(
            not any(
                _activity_contains_outcome_field(item)
                for item in (
                    pilot_manifest,
                    frozen_targets,
                    task_manifest,
                    snapshot_manifest,
                )
            ),
            "task outcome leaked into the common pre-score chain",
        )

        pilot_protocol = pilot_manifest.get("protocol")
        pilot_hash = pilot_manifest.get("protocol_sha256")
        task_protocol = task_manifest.get("protocol")
        task_hash = task_manifest.get("protocol_sha256")
        _activity_require(
            pilot_manifest.get("artifact_type")
            == "activity_matched_pilot_manifest"
            and pilot_manifest.get("manifest_status")
            == "frozen_before_pilot_rows"
            and isinstance(pilot_protocol, dict)
            and _is_sha256(pilot_hash)
            and _scientific_protocol_sha256(pilot_protocol) == pilot_hash,
            "pilot manifest is not frozen/hash-authenticated",
        )
        _activity_require(
            task_manifest.get("artifact_type")
            == "activity_matched_task_manifest"
            and task_manifest.get("manifest_status")
            == "frozen_before_fresh_calibration_rows"
            and isinstance(task_protocol, dict)
            and _is_sha256(task_hash)
            and _scientific_protocol_sha256(task_protocol) == task_hash,
            "task manifest is not frozen/hash-authenticated",
        )
        assert isinstance(pilot_protocol, dict)
        assert isinstance(task_protocol, dict)
        assert isinstance(pilot_hash, str)
        assert isinstance(task_hash, str)
        _activity_require(
            pilot_protocol.get("protocol_version")
            == _ACTIVITY_PROTOCOL_VERSION
            and pilot_protocol.get("stage") == "activity_only_pilot"
            and pilot_protocol.get("status")
            == "must_be_frozen_before_pilot_rows",
            "pilot protocol version/stage drifted",
        )
        _activity_require(
            task_protocol.get("protocol_version")
            == _ACTIVITY_PROTOCOL_VERSION
            and task_protocol.get("stage")
            == "fresh_calibration_then_frozen_task_scoring"
            and task_protocol.get("status")
            == "must_be_frozen_before_fresh_calibration_rows",
            "task protocol version/stage drifted",
        )

        pilot_sources = pilot_protocol.get("scientific_sources_sha256")
        task_sources = task_protocol.get("scientific_sources_sha256")
        _activity_require(
            isinstance(pilot_sources, dict)
            and pilot_sources
            and task_sources == pilot_sources
            and pilot_protocol.get("scientific_sources_combined_sha256")
            == _scientific_protocol_sha256(pilot_sources)
            and task_protocol.get("scientific_sources_combined_sha256")
            == _scientific_protocol_sha256(pilot_sources),
            "scientific source manifests are not identical/hash-linked",
        )
        assert isinstance(pilot_sources, dict)
        repo_root = directory.parents[1]
        for relative, expected_sha in pilot_sources.items():
            _activity_require(
                isinstance(relative, str) and _is_sha256(expected_sha),
                "scientific source manifest is malformed",
            )
            safe = _safe_relative_path(relative)
            source = repo_root / Path(*safe.parts)
            _activity_require(
                source.is_file()
                and not source.is_symlink()
                and sha256_file(source) == expected_sha,
                f"scientific source hash differs: {relative}",
            )

        snapshot_contract = pilot_protocol.get("source_snapshot_contract")
        snapshot_path = (
            directory / "source_snapshot/run_activity_matched_response.py"
        )
        expected_driver_sha = pilot_sources.get(
            "experiments/run_activity_matched_response.py"
        )
        _activity_require(
            isinstance(snapshot_contract, dict)
            and snapshot_contract.get("manifest")
            == (
                "results/activity_matched_response/"
                "source_snapshot/manifest.json"
            )
            and snapshot_contract.get("driver_source")
            == "experiments/run_activity_matched_response.py"
            and snapshot_contract.get("driver_snapshot")
            == (
                "results/activity_matched_response/source_snapshot/"
                "run_activity_matched_response.py"
            )
            and snapshot_contract.get("driver_sha256")
            == expected_driver_sha
            and snapshot_manifest.get("artifact_type")
            == "activity_matched_source_snapshot"
            and snapshot_manifest.get("pilot_protocol_sha256") == pilot_hash
            and snapshot_manifest.get("source_path")
            == "experiments/run_activity_matched_response.py"
            and snapshot_manifest.get("snapshot_path")
            == (
                "results/activity_matched_response/source_snapshot/"
                "run_activity_matched_response.py"
            )
            and snapshot_manifest.get("sha256") == expected_driver_sha
            and snapshot_manifest.get("all_scientific_source_hashes")
            == pilot_sources
            and snapshot_manifest.get(
                "all_scientific_source_hashes_sha256"
            )
            == _scientific_protocol_sha256(pilot_sources)
            and snapshot_path.is_file()
            and not snapshot_path.is_symlink()
            and sha256_file(snapshot_path) == expected_driver_sha,
            "driver source snapshot is not linked to the pilot freeze",
        )

        pilot_ledger = pilot_protocol.get("seed_ledger")
        task_ledger = task_protocol.get("seed_ledger")
        _activity_require(
            isinstance(pilot_ledger, dict)
            and task_ledger == pilot_ledger,
            "pilot/task seed ledgers differ",
        )
        assert isinstance(pilot_ledger, dict)
        _activity_require(
            pilot_ledger.get("pilot_namespace")
            == _ACTIVITY_PILOT_NAMESPACE
            and pilot_ledger.get("task_namespace")
            == _ACTIVITY_TASK_NAMESPACE,
            "activity-response seed namespaces drifted",
        )
        prior_sources = pilot_ledger.get("prior_sources")
        _activity_require(
            isinstance(prior_sources, dict)
            and all(
                prior_sources.get(name) == metadata
                for name, metadata in _ACTIVITY_V2_PRIOR_SOURCES.items()
            ),
            "failed-v2 seed pools are not sealed as prior evidence",
        )
        prior_seeds = pilot_ledger.get("prior_seeds")
        pilot_seeds = pilot_ledger.get("pilot_seeds")
        task_seeds = pilot_ledger.get("task_seeds")
        _activity_require(
            isinstance(prior_seeds, list)
            and all(type(seed) is int for seed in prior_seeds)
            and len(prior_seeds) == len(set(prior_seeds))
            and pilot_ledger.get("prior_seed_count") == len(prior_seeds)
            and pilot_ledger.get("prior_seeds_sha256")
            == _scientific_protocol_sha256(sorted(prior_seeds)),
            "prior seed ledger is malformed",
        )
        _activity_require(
            isinstance(pilot_seeds, list)
            and len(pilot_seeds) == 8
            and len(set(pilot_seeds)) == 8
            and all(type(seed) is int for seed in pilot_seeds)
            and isinstance(task_seeds, list)
            and len(task_seeds) == 24
            and len(set(task_seeds)) == 24
            and all(type(seed) is int for seed in task_seeds),
            "new seed pools are not the declared 8-pilot/24-task design",
        )
        assert isinstance(prior_seeds, list)
        assert isinstance(pilot_seeds, list)
        assert isinstance(task_seeds, list)
        _activity_require(
            not set(prior_seeds) & set(pilot_seeds)
            and not set(prior_seeds) & set(task_seeds)
            and not set(pilot_seeds) & set(task_seeds)
            and pilot_ledger.get("pilot_prior_overlap") == []
            and pilot_ledger.get("task_prior_overlap") == []
            and pilot_ledger.get("pilot_task_overlap") == [],
            "new seed pools overlap pilot or prior evidence",
        )

        physics = pilot_protocol.get("physics")
        _activity_require(
            isinstance(physics, dict)
            and physics.get("N") == 5
            and _close(physics.get("h"), 0.5)
            and _close(physics.get("dt"), 0.5)
            and physics.get("designs") == list(_ACTIVITY_DESIGNS)
            and physics.get("branches") == _ACTIVITY_BRANCHES,
            "pilot physics/branch protocol drifted",
        )
        assert isinstance(physics, dict)
        rate_grids = physics.get("pilot_rate_grids")
        _activity_require(
            isinstance(rate_grids, dict)
            and set(rate_grids) == set(_ACTIVITY_DESIGNS),
            "pilot rate grids are incomplete",
        )
        assert isinstance(rate_grids, dict)
        normalized_grids: dict[str, list[float]] = {}
        for design in _ACTIVITY_DESIGNS:
            grid = _activity_float_list(rate_grids[design])
            branch = _ACTIVITY_BRANCHES[design]
            _activity_require(
                len(grid) >= 3
                and grid == sorted(set(grid))
                and _close(grid[0], branch["lower_rate"])
                and _close(grid[-1], branch["upper_rate"])
                and any(
                    _close(value, branch["anchor_rate"]) for value in grid
                ),
                f"pilot rate grid is invalid for {design}",
            )
            normalized_grids[design] = grid
        expected_pilot_rows = len(pilot_seeds) * sum(
            len(grid) for grid in normalized_grids.values()
        )
        _activity_require(
            expected_pilot_rows == 160
            and pilot_protocol.get("expected_pilot_rows")
            == expected_pilot_rows,
            "pilot expected-row count is not the frozen 160-row grid",
        )
        calibration_input = pilot_protocol.get("calibration_input")
        target_rule = pilot_protocol.get("target_freeze_rule")
        supervised = pilot_protocol.get("supervised_boundary")
        _activity_require(
            isinstance(calibration_input, dict)
            and calibration_input.get("distribution") == "iid Uniform[0,1]"
            and calibration_input.get("labels_or_task_targets_used") is False
            and calibration_input.get("wash_intervals") == 200
            and calibration_input.get("unsupervised_prefix_intervals") == 600
            and calibration_input.get("measured_intervals") == 400
            and isinstance(target_rule, dict)
            and target_rule.get("n_targets") == 5
            and _close(
                target_rule.get("minimum_common_span_ratio"),
                1.5,
            )
            and _close(
                target_rule.get("log_inset_fraction_each_side"),
                0.05,
            )
            and isinstance(supervised, dict)
            and supervised.get("constructs_task_targets") is False
            and supervised.get("fits_readout") is False
            and supervised.get("scores_task") is False,
            "pilot calibration/supervised boundary drifted",
        )

        pilot_rows_by_name = {
            name: row
            for name, row in parsed.items()
            if name.startswith("pilot/checkpoints/")
            and name.endswith(".json")
        }
        _activity_require(
            len(pilot_rows_by_name) == expected_pilot_rows,
            "pilot checkpoint coverage is incomplete",
        )
        pilot_rows: dict[tuple[str, int, float], dict] = {}
        pilot_hashes_by_seed: dict[int, set[tuple[str, str]]] = {}
        for name, candidate in pilot_rows_by_name.items():
            _activity_require(
                isinstance(candidate, dict),
                f"pilot checkpoint is malformed: {name}",
            )
            row = candidate
            design = row.get("design")
            seed = row.get("seed")
            rate = _finite_float(row.get("rate"))
            activity = _finite_float(row.get("activity"))
            total = _finite_float(row.get("total_expected_jumps"))
            trace_error = _finite_float(row.get("maximum_trace_error"))
            imaginary = _finite_float(
                row.get("maximum_activity_imaginary_residue")
            )
            minimum_count = _finite_float(
                row.get("minimum_interval_integrated_activity")
            )
            _activity_require(
                row.get("artifact_type") == "activity_only_pilot_row"
                and row.get("protocol_version")
                == _ACTIVITY_PROTOCOL_VERSION
                and row.get("pilot_protocol_sha256") == pilot_hash
                and design in _ACTIVITY_DESIGNS
                and type(seed) is int
                and seed in pilot_seeds
                and rate is not None
                and any(
                    _close(rate, value)
                    for value in normalized_grids[str(design)]
                )
                and row.get("branch")
                == _ACTIVITY_BRANCHES[str(design)]
                and activity is not None
                and activity > 0.0
                and total is not None
                and _close(total, activity * 400 * 0.5)
                and trace_error is not None
                and trace_error <= _ACTIVITY_TRACE_TOLERANCE
                and imaginary is not None
                and imaginary <= _ACTIVITY_IMAGINARY_TOLERANCE
                and minimum_count is not None
                and minimum_count >= -_ACTIVITY_NEGATIVE_COUNT_TOLERANCE
                and _finite_float(row.get("jump_strength")) is not None
                and _is_sha256(row.get("couplings_sha256"))
                and _is_sha256(row.get("calibration_input_sha256"))
                and "test_stm_capacity" not in row
                and "task_input_sha256" not in row,
                f"pilot checkpoint invariant failed: {name}",
            )
            assert isinstance(design, str)
            assert isinstance(seed, int)
            assert rate is not None
            key = (design, seed, rate)
            _activity_require(
                key not in pilot_rows,
                "pilot checkpoints contain duplicate identities",
            )
            pilot_rows[key] = row
            pilot_hashes_by_seed.setdefault(seed, set()).add(
                (
                    str(row["couplings_sha256"]),
                    str(row["calibration_input_sha256"]),
                )
            )
        expected_pilot_keys = {
            (design, seed, rate)
            for design in _ACTIVITY_DESIGNS
            for seed in pilot_seeds
            for rate in normalized_grids[design]
        }
        _activity_require(
            set(pilot_rows) == expected_pilot_keys
            and all(
                len(hashes) == 1
                for hashes in pilot_hashes_by_seed.values()
            ),
            "pilot identity grid or paired streams are incomplete",
        )

        _activity_require(
            frozen_targets.get("artifact_type")
            == "frozen_activity_targets"
            and frozen_targets.get("freeze_status")
            == "frozen_before_fresh_calibration_or_task_scores"
            and frozen_targets.get("pilot_protocol_sha256") == pilot_hash
            and frozen_targets.get("uses_supervised_task_information")
            is False
            and frozen_targets.get("n_targets") == 5,
            "frozen target metadata is invalid",
        )
        reachable: list[tuple[float, float]] = []
        expected_curve_keys = {
            (design, seed)
            for design in _ACTIVITY_DESIGNS
            for seed in pilot_seeds
        }
        curve_audits = frozen_targets.get("pilot_curve_audits")
        _activity_require(
            isinstance(curve_audits, list)
            and len(curve_audits) == len(expected_curve_keys),
            "frozen pilot-curve audits are incomplete",
        )
        observed_curve_keys: set[tuple[str, int]] = set()
        for curve in curve_audits:
            _activity_require(
                isinstance(curve, dict),
                "frozen pilot-curve audit is malformed",
            )
            design = curve.get("design")
            seed = curve.get("seed")
            _activity_require(
                design in _ACTIVITY_DESIGNS
                and type(seed) is int
                and seed in pilot_seeds,
                "frozen pilot-curve identity is invalid",
            )
            assert isinstance(design, str)
            assert isinstance(seed, int)
            ordered = [
                pilot_rows[(design, seed, rate)]
                for rate in normalized_grids[design]
            ]
            activities = [float(row["activity"]) for row in ordered]
            orientation = _ACTIVITY_BRANCHES[design][
                "activity_orientation"
            ]
            sign = 1.0 if orientation == "increasing" else -1.0
            increments = [
                sign * (right - left)
                for left, right in zip(activities, activities[1:])
            ]
            tolerance = 2e-7 + 2e-5 * max(
                abs(value) for value in activities
            )
            _activity_require(
                increments
                and all(value > -tolerance for value in increments),
                f"pilot branch is not monotone for {design}/{seed}",
            )
            low = min(activities[0], activities[-1])
            high = max(activities[0], activities[-1])
            audit = curve.get("monotonicity")
            interval = curve.get("reachable_interval")
            _activity_require(
                low > 0.0
                and high > low
                and isinstance(interval, list)
                and len(interval) == 2
                and _close(interval[0], low)
                and _close(interval[1], high)
                and isinstance(audit, dict)
                and audit.get("passed") is True
                and audit.get("orientation") == orientation
                and _close(audit.get("tolerance"), tolerance)
                and _close(
                    audit.get("minimum_oriented_increment"),
                    min(increments),
                )
                and all(
                    _close(left, right)
                    for left, right in zip(
                        _activity_float_list(
                            audit.get("rates"),
                            length=len(normalized_grids[design]),
                        ),
                        normalized_grids[design],
                    )
                )
                and all(
                    _close(left, right)
                    for left, right in zip(
                        _activity_float_list(
                            audit.get("activities"),
                            length=len(activities),
                        ),
                        activities,
                    )
                ),
                f"pilot-curve audit disagrees for {design}/{seed}",
            )
            observed_curve_keys.add((design, seed))
            reachable.append((low, high))
        _activity_require(
            observed_curve_keys == expected_curve_keys,
            "frozen pilot-curve identities are incomplete",
        )

        common_low = max(low for low, _ in reachable)
        common_high = min(high for _, high in reachable)
        span_ratio = common_high / common_low
        log_low = math.log(common_low)
        log_high = math.log(common_high)
        log_span = log_high - log_low
        target_low = math.exp(log_low + 0.05 * log_span)
        target_high = math.exp(log_high - 0.05 * log_span)
        expected_targets = [
            math.exp(
                math.log(target_low)
                + index
                * (math.log(target_high) - math.log(target_low))
                / 4
            )
            for index in range(5)
        ]
        targets = _activity_float_list(
            frozen_targets.get("targets"),
            length=5,
        )
        row_index = sorted(
            (
                design,
                seed,
                rate,
                _scientific_protocol_sha256(row),
            )
            for (design, seed, rate), row in pilot_rows.items()
        )
        _activity_require(
            common_high > common_low > 0.0
            and span_ratio >= 1.5
            and _close(
                frozen_targets.get("common_activity_interval")[0],
                common_low,
            )
            and _close(
                frozen_targets.get("common_activity_interval")[1],
                common_high,
            )
            and _close(
                frozen_targets.get("common_activity_span_ratio"),
                span_ratio,
            )
            and _close(
                frozen_targets.get("target_interval_after_log_inset")[0],
                target_low,
            )
            and _close(
                frozen_targets.get("target_interval_after_log_inset")[1],
                target_high,
            )
            and all(
                _close(left, right)
                for left, right in zip(targets, expected_targets)
            )
            and frozen_targets.get("pilot_rows_sha256")
            == _scientific_protocol_sha256(row_index),
            "frozen targets do not reproduce from pilot rows",
        )

        frozen_target_path = directory / "frozen_targets.json"
        frozen_target_link = task_protocol.get("frozen_targets")
        fresh_boundary = task_protocol.get("fresh_boundary")
        calibration_protocol = task_protocol.get("calibration")
        task_spec = task_protocol.get("task")
        inference = task_protocol.get("inference")
        censored_statuses = {
            "censored_target_unreachable",
            "censored_branch_nonmonotone",
            "censored_nonconvergence",
        }
        _activity_require(
            isinstance(frozen_target_link, dict)
            and frozen_target_link.get("path")
            == "results/activity_matched_response/frozen_targets.json"
            and frozen_target_link.get("sha256")
            == sha256_file(frozen_target_path)
            and frozen_target_link.get("pilot_protocol_sha256")
            == pilot_hash
            and all(
                _close(left, right)
                for left, right in zip(
                    _activity_float_list(
                        frozen_target_link.get("targets"),
                        length=5,
                    ),
                    targets,
                )
            ),
            "task manifest is not byte-linked to frozen targets",
        )
        _activity_require(
            isinstance(fresh_boundary, dict)
            and fresh_boundary.get("task_seed_count") == 24
            and fresh_boundary.get("task_seeds") == task_seeds
            and fresh_boundary.get("task_prior_overlap") == []
            and fresh_boundary.get("task_pilot_overlap") == []
            and fresh_boundary.get(
                "same_couplings_and_streams_across_designs_and_targets"
            )
            is True
            and fresh_boundary.get(
                "calibration_and_task_streams_independent"
            )
            is True,
            "fresh task boundary is incomplete",
        )
        _activity_require(
            isinstance(calibration_protocol, dict)
            and calibration_protocol.get("branches") == _ACTIVITY_BRANCHES
            and calibration_protocol.get("input_split")
            == {
                "wash": 200,
                "unsupervised_prefix": 600,
                "measured_activity": 400,
            }
            and calibration_protocol.get("target_count") == 5
            and calibration_protocol.get("expected_cells") == 240
            and calibration_protocol.get("censored_statuses")
            == [
                "censored_target_unreachable",
                "censored_branch_nonmonotone",
                "censored_nonconvergence",
            ]
            and _close(
                calibration_protocol.get("relative_match_tolerance"),
                0.005,
            )
            and _close(
                calibration_protocol.get("absolute_match_tolerance"),
                1e-5,
            ),
            "fresh calibration protocol drifted",
        )
        _activity_require(
            isinstance(task_spec, dict)
            and task_spec.get("task_name") == "STM"
            and task_spec.get("N") == 5
            and _close(task_spec.get("h"), 0.5)
            and _close(task_spec.get("dt"), 0.5)
            and task_spec.get("wash") == 200
            and task_spec.get("train") == 600
            and task_spec.get("test") == 400
            and task_spec.get("delays") == list(range(1, 21))
            and _close(task_spec.get("ridge"), 1e-8, atol=1e-18)
            and task_spec.get("ridge_selection")
            == "none; fixed before all fresh task scores"
            and isinstance(inference, dict)
            and inference.get("family_size") == 5
            and "Bonferroni-t" in str(inference.get("method")),
            "task/inference protocol drifted",
        )

        calibration_rows_by_name = {
            name: row
            for name, row in parsed.items()
            if name.startswith("calibration/checkpoints/")
            and name.endswith(".json")
        }
        expected_fresh_keys = {
            (design, seed, target_index)
            for design in _ACTIVITY_DESIGNS
            for seed in task_seeds
            for target_index in range(5)
        }
        _activity_require(
            len(calibration_rows_by_name) == 240,
            "fresh calibration checkpoint coverage is incomplete",
        )
        calibration_rows: dict[tuple[str, int, int], dict] = {}
        calibration_paths: dict[tuple[str, int, int], Path] = {}
        pairing_by_seed: dict[int, set[tuple[str, str]]] = {}
        censored_keys: set[tuple[str, int, int]] = set()
        for name, candidate in calibration_rows_by_name.items():
            _activity_require(
                isinstance(candidate, dict),
                f"fresh calibration checkpoint is malformed: {name}",
            )
            row = candidate
            design = row.get("design")
            seed = row.get("seed")
            target_index = row.get("target_index")
            _activity_require(
                design in _ACTIVITY_DESIGNS
                and type(seed) is int
                and seed in task_seeds
                and type(target_index) is int
                and 0 <= target_index < 5,
                f"fresh calibration identity is invalid: {name}",
            )
            assert isinstance(design, str)
            assert isinstance(seed, int)
            assert isinstance(target_index, int)
            key = (design, seed, target_index)
            expected_name = (
                "calibration/checkpoints/"
                f"{design}__seed_{seed}__target_{target_index:02d}.json"
            )
            target = targets[target_index]
            matched_rate = _finite_float(row.get("matched_rate"))
            matched_activity = _finite_float(row.get("matched_activity"))
            relative_error = _finite_float(row.get("relative_error"))
            absolute_error = _finite_float(row.get("absolute_error"))
            branch = _ACTIVITY_BRANCHES[design]
            monotonicity = row.get("monotonicity")
            evaluations = row.get("evaluations")
            status = row.get("status")
            _activity_require(
                name == expected_name
                and row.get("artifact_type")
                == "fresh_activity_calibration_row"
                and row.get("protocol_version")
                == _ACTIVITY_PROTOCOL_VERSION
                and row.get("task_protocol_sha256") == task_hash
                and _close(row.get("target_activity"), target)
                and row.get("branch") == branch
                and row.get("calibration_uses_task_targets_or_scores")
                is False
                and status in ({"matched"} | censored_statuses)
                and _is_sha256(row.get("couplings_sha256"))
                and _is_sha256(row.get("calibration_input_sha256"))
                and isinstance(evaluations, list)
                and len(evaluations) >= 2
                and not _activity_contains_outcome_field(row),
                f"fresh calibration invariant failed: {name}",
            )
            evaluation_rates: list[float] = []
            evaluation_activities: list[float] = []
            for evaluation in evaluations:
                evaluation_rate = _finite_float(evaluation.get("rate"))
                evaluation_activity = _finite_float(
                    evaluation.get("activity")
                )
                _activity_require(
                    isinstance(evaluation, dict)
                    and evaluation_rate is not None
                    and float(branch["lower_rate"])
                    <= evaluation_rate
                    <= float(branch["upper_rate"])
                    and evaluation_activity is not None
                    and evaluation_activity > 0.0
                    and (
                        _finite_float(evaluation.get("jump_strength"))
                        is not None
                    )
                    and float(evaluation["jump_strength"]) >= 0.0
                    and _finite_float(
                        evaluation.get("maximum_trace_error")
                    )
                    is not None
                    and float(evaluation["maximum_trace_error"]) >= 0.0
                    and float(evaluation["maximum_trace_error"])
                    <= _ACTIVITY_TRACE_TOLERANCE
                    and _finite_float(
                        evaluation.get(
                            "maximum_activity_imaginary_residue"
                        )
                    )
                    is not None
                    and float(
                        evaluation[
                            "maximum_activity_imaginary_residue"
                        ]
                    )
                    >= 0.0
                    and float(
                        evaluation[
                            "maximum_activity_imaginary_residue"
                        ]
                    )
                    <= _ACTIVITY_IMAGINARY_TOLERANCE,
                    f"fresh calibration evaluation failed: {name}",
                )
                assert evaluation_rate is not None
                assert evaluation_activity is not None
                evaluation_rates.append(evaluation_rate)
                evaluation_activities.append(evaluation_activity)
            _activity_require(
                evaluation_rates == sorted(set(evaluation_rates))
                and _close(
                    evaluation_rates[0],
                    branch["lower_rate"],
                )
                and _close(
                    evaluation_rates[-1],
                    branch["upper_rate"],
                ),
                f"fresh calibration evaluation grid failed: {name}",
            )

            if status == "matched":
                _activity_require(
                    matched_rate is not None
                    and float(branch["lower_rate"])
                    <= matched_rate
                    <= float(branch["upper_rate"])
                    and matched_activity is not None
                    and matched_activity > 0.0
                    and relative_error is not None
                    and relative_error <= 0.005
                    and _close(
                        relative_error,
                        abs(matched_activity - target) / target,
                    )
                    and absolute_error is not None
                    and _close(
                        absolute_error,
                        abs(matched_activity - target),
                    )
                    and any(
                        _close(rate, matched_rate)
                        and _close(activity, matched_activity)
                        for rate, activity in zip(
                            evaluation_rates,
                            evaluation_activities,
                        )
                    )
                    and isinstance(monotonicity, dict)
                    and monotonicity.get("passed") is True
                    and monotonicity.get("orientation")
                    == branch["activity_orientation"],
                    f"matched fresh calibration cell failed: {name}",
                )
            elif status == "censored_target_unreachable":
                reachable_interval = _activity_float_list(
                    row.get("reachable_interval"),
                    length=2,
                )
                reach_low = min(
                    evaluation_activities[0],
                    evaluation_activities[-1],
                )
                reach_high = max(
                    evaluation_activities[0],
                    evaluation_activities[-1],
                )
                bracket_tolerance = max(1e-5, 0.005 * abs(target))
                _activity_require(
                    matched_rate is None
                    and matched_activity is None
                    and relative_error is None
                    and absolute_error is None
                    and _close(reachable_interval[0], reach_low)
                    and _close(reachable_interval[1], reach_high)
                    and (
                        target < reach_low - bracket_tolerance
                        or target > reach_high + bracket_tolerance
                    ),
                    f"unreachable calibration censor is unsupported: {name}",
                )
                censored_keys.add(key)
            elif status == "censored_branch_nonmonotone":
                sign = (
                    1.0
                    if branch["activity_orientation"] == "increasing"
                    else -1.0
                )
                tolerance = 2e-7 + 2e-5 * max(
                    abs(value) for value in evaluation_activities
                )
                increments = [
                    sign * (right - left)
                    for left, right in zip(
                        evaluation_activities,
                        evaluation_activities[1:],
                    )
                ]
                _activity_require(
                    any(value < -tolerance for value in increments),
                    f"nonmonotone calibration censor is unsupported: {name}",
                )
                censored_keys.add(key)
            elif status == "censored_nonconvergence":
                tolerance = max(1e-5, 0.005 * abs(target))
                _activity_require(
                    matched_rate is not None
                    and float(branch["lower_rate"])
                    <= matched_rate
                    <= float(branch["upper_rate"])
                    and matched_activity is not None
                    and matched_activity > 0.0
                    and relative_error is not None
                    and _close(
                        relative_error,
                        abs(matched_activity - target) / target,
                    )
                    and absolute_error is not None
                    and _close(
                        absolute_error,
                        abs(matched_activity - target),
                    )
                    and any(
                        _close(rate, matched_rate)
                        and _close(activity, matched_activity)
                        for rate, activity in zip(
                            evaluation_rates,
                            evaluation_activities,
                        )
                    )
                    and absolute_error > tolerance,
                    f"nonconverged calibration censor is unsupported: {name}",
                )
                censored_keys.add(key)
            _activity_require(
                key not in calibration_rows,
                "fresh calibration contains duplicate identities",
            )
            calibration_rows[key] = row
            calibration_paths[key] = directory / name
            pairing_by_seed.setdefault(seed, set()).add(
                (
                    str(row["couplings_sha256"]),
                    str(row["calibration_input_sha256"]),
                )
            )
        _activity_require(
            set(calibration_rows) == expected_fresh_keys
            and all(
                len(hashes) == 1 for hashes in pairing_by_seed.values()
            ),
            "fresh calibration identity grid or pairing is incomplete",
        )
        for design in _ACTIVITY_DESIGNS:
            sign = (
                1.0
                if _ACTIVITY_BRANCHES[design]["activity_orientation"]
                == "increasing"
                else -1.0
            )
            for seed in task_seeds:
                rates = [
                    (
                        index,
                        float(
                            calibration_rows[(design, seed, index)][
                                "matched_rate"
                            ]
                        ),
                    )
                    for index in range(5)
                    if calibration_rows[(design, seed, index)].get("status")
                    == "matched"
                ]
                _activity_require(
                    all(
                        sign * (right[1] - left[1]) >= -1e-12
                        for left, right in zip(rates, rates[1:])
                    ),
                    f"matched-rate order failed for {design}/{seed}",
                )

        if censored_keys:
            score_artifacts = [
                name
                for name in parsed
                if name.startswith("score/")
            ]
            score_directory = directory / "score"
            score_files = (
                [
                    path
                    for path in score_directory.rglob("*")
                    if path.is_file() or path.is_symlink()
                ]
                if score_directory.exists()
                else []
            )
            _activity_require(
                frozen_calibration is None
                and aggregate is None
                and not (directory / "frozen_calibration.json").exists()
                and not (directory / "aggregate.json").exists()
                and not score_artifacts
                and not score_files,
                "task-score leakage followed failed feasibility gate",
            )
            return (
                True,
                "complete outcome-neutral feasibility failure: "
                f"{len(censored_keys)} of 240 calibration cells censored",
            )

        _activity_require(
            isinstance(frozen_calibration, dict)
            and isinstance(aggregate, dict),
            "successful calibration lacks frozen scoring artifacts",
        )
        assert isinstance(frozen_calibration, dict)
        assert isinstance(aggregate, dict)
        frozen_cells = frozen_calibration.get("cells")
        _activity_require(
            frozen_calibration.get("artifact_type")
            == "frozen_fresh_activity_calibration"
            and frozen_calibration.get("freeze_status")
            == "frozen_before_any_task_score"
            and frozen_calibration.get("task_protocol_sha256") == task_hash
            and frozen_calibration.get("frozen_targets_sha256")
            == sha256_file(frozen_target_path)
            and frozen_calibration.get("expected_cells") == 240
            and frozen_calibration.get("observed_cells") == 240
            and frozen_calibration.get("censored_cells") == 0
            and frozen_calibration.get("gate_passed") is True
            and frozen_calibration.get("gate_errors") == []
            and isinstance(frozen_cells, list)
            and len(frozen_cells) == 240,
            "frozen fresh calibration gate is invalid",
        )
        assert isinstance(frozen_cells, list)
        observed_cells: dict[tuple[str, int, int], dict] = {}
        for cell in frozen_cells:
            _activity_require(
                isinstance(cell, dict),
                "frozen fresh calibration cell is malformed",
            )
            key = (
                str(cell.get("design")),
                int(cell.get("seed")),
                int(cell.get("target_index")),
            )
            row = calibration_rows.get(key)
            _activity_require(
                row is not None and key not in observed_cells,
                "frozen fresh calibration identities drifted",
            )
            assert row is not None
            _activity_require(
                cell.get("status") == "matched"
                and _close(
                    cell.get("target_activity"),
                    row.get("target_activity"),
                )
                and _close(
                    cell.get("matched_rate"),
                    row.get("matched_rate"),
                )
                and _close(
                    cell.get("matched_activity"),
                    row.get("matched_activity"),
                )
                and _close(
                    cell.get("relative_error"),
                    row.get("relative_error"),
                )
                and cell.get("calibration_row_sha256")
                == sha256_file(calibration_paths[key])
                and cell.get("calibration_row_payload_sha256")
                == _scientific_protocol_sha256(row)
                and cell.get("couplings_sha256")
                == row.get("couplings_sha256")
                and cell.get("calibration_input_sha256")
                == row.get("calibration_input_sha256"),
                "frozen fresh calibration cell/hash disagrees with row",
            )
            observed_cells[key] = cell
        maximum_error = max(
            float(row["relative_error"])
            for row in calibration_rows.values()
        )
        _activity_require(
            set(observed_cells) == expected_fresh_keys
            and _close(
                frozen_calibration.get("maximum_relative_match_error"),
                maximum_error,
            ),
            "frozen fresh calibration summary disagrees with rows",
        )

        frozen_calibration_path = directory / "frozen_calibration.json"
        frozen_calibration_sha = sha256_file(frozen_calibration_path)
        score_rows_by_name = {
            name: row
            for name, row in parsed.items()
            if name.startswith("score/checkpoints/")
            and name.endswith(".json")
        }
        _activity_require(
            len(score_rows_by_name) == 240,
            "fresh score checkpoint coverage is incomplete",
        )
        score_rows: dict[tuple[str, int, int], dict] = {}
        score_hashes_by_seed: dict[int, set[tuple[str, str, str]]] = {}
        for name, candidate in score_rows_by_name.items():
            _activity_require(
                isinstance(candidate, dict),
                f"fresh score checkpoint is malformed: {name}",
            )
            row = candidate
            design = row.get("design")
            seed = row.get("seed")
            target_index = row.get("target_index")
            _activity_require(
                design in _ACTIVITY_DESIGNS
                and type(seed) is int
                and seed in task_seeds
                and type(target_index) is int
                and 0 <= target_index < 5,
                f"fresh score identity is invalid: {name}",
            )
            assert isinstance(design, str)
            assert isinstance(seed, int)
            assert isinstance(target_index, int)
            key = (design, seed, target_index)
            cell = observed_cells[key]
            by_delay = _activity_float_list(
                row.get("test_capacity_by_delay"),
                length=20,
            )
            stm = _finite_float(row.get("test_stm_capacity"))
            activity = _finite_float(
                row.get("time_averaged_test_activity")
            )
            total = _finite_float(row.get("total_expected_test_jumps"))
            trace_error = _finite_float(row.get("maximum_trace_error"))
            imaginary = _finite_float(
                row.get("maximum_activity_imaginary_residue")
            )
            minimum_count = _finite_float(
                row.get("minimum_test_interval_integrated_activity")
            )
            _activity_require(
                row.get("artifact_type")
                == "fresh_activity_matched_stm_row"
                and row.get("protocol_version")
                == _ACTIVITY_PROTOCOL_VERSION
                and row.get("task_protocol_sha256") == task_hash
                and row.get("frozen_calibration_sha256")
                == frozen_calibration_sha
                and row.get("calibration_row_sha256")
                == cell.get("calibration_row_sha256")
                and _close(
                    row.get("target_activity"),
                    cell.get("target_activity"),
                )
                and _close(
                    row.get("frozen_rate"),
                    cell.get("matched_rate"),
                )
                and row.get("couplings_sha256")
                == cell.get("couplings_sha256")
                and row.get("calibration_input_sha256")
                == cell.get("calibration_input_sha256")
                and _is_sha256(row.get("task_input_sha256"))
                and row.get(
                    "task_stream_is_independent_of_calibration_stream"
                )
                is True
                and row.get("task_input_sha256")
                != row.get("calibration_input_sha256")
                and _close(row.get("ridge"), 1e-8, atol=1e-18)
                and stm is not None
                and _close(stm, math.fsum(by_delay))
                and activity is not None
                and activity >= 0.0
                and total is not None
                and _close(total, activity * 400 * 0.5)
                and trace_error is not None
                and trace_error <= _ACTIVITY_TRACE_TOLERANCE
                and imaginary is not None
                and imaginary <= _ACTIVITY_IMAGINARY_TOLERANCE
                and minimum_count is not None
                and minimum_count >= -_ACTIVITY_NEGATIVE_COUNT_TOLERANCE,
                f"fresh score invariant/linkage failed: {name}",
            )
            _activity_require(
                key not in score_rows,
                "fresh scores contain duplicate identities",
            )
            score_rows[key] = row
            score_hashes_by_seed.setdefault(seed, set()).add(
                (
                    str(row["couplings_sha256"]),
                    str(row["calibration_input_sha256"]),
                    str(row["task_input_sha256"]),
                )
            )
        _activity_require(
            set(score_rows) == expected_fresh_keys
            and all(
                len(hashes) == 1
                for hashes in score_hashes_by_seed.values()
            ),
            "fresh score identity grid or paired streams are incomplete",
        )

        target_results = aggregate.get("target_results")
        _activity_require(
            aggregate.get("artifact_type")
            == "activity_matched_response_aggregate"
            and aggregate.get("protocol_version")
            == _ACTIVITY_PROTOCOL_VERSION
            and aggregate.get("task_protocol_sha256") == task_hash
            and aggregate.get("frozen_calibration_sha256")
            == frozen_calibration_sha
            and aggregate.get("status") == "complete"
            and aggregate.get("n_rows") == 240
            and aggregate.get("n_seeds") == 24
            and aggregate.get("n_targets") == 5
            and isinstance(target_results, list)
            and len(target_results) == 5,
            "activity-matched aggregate header/coverage is incomplete",
        )
        assert isinstance(target_results, list)
        stm_flags: list[bool] = []
        activity_flags: list[bool] = []
        for index, observed in enumerate(target_results):
            _activity_require(
                isinstance(observed, dict)
                and observed.get("target_index") == index
                and _close(observed.get("target_activity"), targets[index]),
                "activity-matched target summary identity drifted",
            )
            assert isinstance(observed, dict)
            local_rows = [
                score_rows[("local", seed, index)] for seed in task_seeds
            ]
            collective_rows = [
                score_rows[("collective", seed, index)]
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
            stm_differences = [
                candidate - reference
                for candidate, reference in zip(
                    collective_scores,
                    local_scores,
                )
            ]
            relative_activity_differences = [
                (candidate - reference) / targets[index]
                for candidate, reference in zip(
                    collective_activity,
                    local_activity,
                )
            ]
            stm_band = observed.get("stm_collective_minus_local")
            activity_band = observed.get(
                "relative_test_activity_collective_minus_local"
            )
            _activity_require(
                _close(
                    observed.get("local_stm_mean"),
                    math.fsum(local_scores) / 24,
                )
                and _close(
                    observed.get("collective_stm_mean"),
                    math.fsum(collective_scores) / 24,
                )
                and _close(
                    observed.get("local_test_activity_mean"),
                    math.fsum(local_activity) / 24,
                )
                and _close(
                    observed.get("collective_test_activity_mean"),
                    math.fsum(collective_activity) / 24,
                )
                and _activity_band_matches(stm_band, stm_differences)
                and _activity_band_matches(
                    activity_band,
                    relative_activity_differences,
                ),
                f"activity-matched derived summary disagrees at target {index}",
            )
            assert isinstance(stm_band, dict)
            assert isinstance(activity_band, dict)
            stm_flag = float(stm_band["simultaneous_lower"]) > 0.0
            activity_flag = (
                float(activity_band["simultaneous_lower"]) >= -0.05
                and float(activity_band["simultaneous_upper"]) <= 0.05
            )
            _activity_require(
                observed.get("stm_dominance_at_target") is stm_flag
                and observed.get("test_activity_equivalent_at_target")
                is activity_flag,
                f"activity-matched target gate disagrees at target {index}",
            )
            stm_flags.append(stm_flag)
            activity_flags.append(activity_flag)

        claim_gates = aggregate.get("claim_gates")
        all_stm = all(stm_flags)
        all_activity = all(activity_flags)
        claim_allowed = all_stm and all_activity
        _activity_require(
            isinstance(claim_gates, dict)
            and claim_gates.get("zero_censored_fresh_cells") is True
            and claim_gates.get(
                "simultaneous_stm_dominance_all_targets"
            )
            is all_stm
            and claim_gates.get(
                "simultaneous_test_activity_equivalence_all_targets"
            )
            is all_activity
            and claim_gates.get("range_wide_dominance_supported")
            is all_stm
            and claim_gates.get("task_activity_equivalence_supported")
            is all_activity
            and claim_gates.get(
                "activity_matched_dominance_claim_allowed"
            )
            is claim_allowed,
            "activity-matched claim gates disagree with raw rows",
        )

        invariant = aggregate.get("invariant_audit")
        maximum_trace = max(
            float(row["maximum_trace_error"])
            for row in score_rows.values()
        )
        maximum_imaginary = max(
            float(row["maximum_activity_imaginary_residue"])
            for row in score_rows.values()
        )
        minimum_test_count = min(
            float(row["minimum_test_interval_integrated_activity"])
            for row in score_rows.values()
        )
        _activity_require(
            isinstance(invariant, dict)
            and invariant.get("passed") is True
            and invariant.get("errors") == []
            and invariant.get("expected_rows") == 240
            and invariant.get("observed_rows") == 240
            and invariant.get("fixed_ridge_all_rows") is True
            and invariant.get("delay_capacity_sums_match") is True
            and _close(
                invariant.get("maximum_trace_error"),
                maximum_trace,
            )
            and _close(
                invariant.get("maximum_activity_imaginary_residue"),
                maximum_imaginary,
            )
            and _close(
                invariant.get(
                    "minimum_test_interval_integrated_activity"
                ),
                minimum_test_count,
            ),
            "activity-matched invariant audit disagrees with score rows",
        )
    except (
        AssertionError,
        EvidencePackageError,
        KeyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
    ) as error:
        return False, f"activity-matched response validation failed: {error}"

    return (
        True,
        "complete staged 8-pilot/24-fresh activity-matched response",
    )


def _result_specific_checks(
    spec: ResultGroup,
    parsed: dict[str, object],
    directory: Path,
) -> dict[str, dict[str, object]]:
    if spec.name == "activity_matched_response":
        complete, reason = _activity_matched_response_is_complete(
            directory,
            parsed,
        )
        return {
            "specific:activity-matched-response": {
                "complete": complete,
                "reason": reason,
            }
        }
    if spec.name == "primary_driven_activity":
        complete, reason = _primary_driven_activity_is_complete(
            directory,
            parsed,
        )
        return {
            "specific:primary-driven-activity": {
                "complete": complete,
                "reason": reason,
            }
        }
    if spec.name == "revision_parity_control":
        complete, reason = _revision_parity_is_complete(parsed)
        return {
            "specific:active-and-reference-parity": {
                "complete": complete,
                "reason": reason,
            }
        }
    if spec.name == "revision_normalized_scaling":
        complete, reason = _revision_scaling_is_complete(parsed)
        return {
            "specific:variance-normalized-scaling": {
                "complete": complete,
                "reason": reason,
            }
        }
    if spec.name == "revision_primary_regularization":
        complete, reason = _revision_primary_regularization_is_complete(parsed)
        return {
            "specific:seven-design-primary-regularization": {
                "complete": complete,
                "reason": reason,
            }
        }
    if spec.name == "collective_loss_full_input_diagnostic":
        complete, reason = _collective_loss_full_input_is_complete(
            directory,
            parsed,
        )
        return {
            "specific:collective-full-input-spectrum": {
                "complete": complete,
                "reason": reason,
            }
        }
    if spec.name != "revision_tuning":
        return {}
    validators = {
        "strength_extension/manifest.json": _revision_manifest_is_frozen,
        "nested_tuning/manifest.json": _revision_manifest_is_frozen,
        "fresh_interpolation/manifest.json": _revision_manifest_is_frozen,
        "strength_extension/six_channel_aggregate.json":
            _revision_strength_is_complete,
        "strength_extension/alternative_matching_aggregate.json":
            _revision_alternative_matching_is_complete,
        "nested_tuning/nested_tuning_results.json": _revision_nested_is_complete,
        "fresh_interpolation/fresh_interpolation_results.json":
            _revision_fresh_is_complete,
    }
    checks: dict[str, dict[str, object]] = {}
    for relative, validator in validators.items():
        if relative not in parsed:
            checks[f"specific:{relative}"] = {
                "complete": False,
                "reason": "missing or not selected",
            }
            continue
        complete, reason = validator(parsed[relative])
        checks[f"specific:{relative}"] = {
            "complete": complete,
            "reason": reason,
        }
    nested_complete, nested_reason = (
        _nested_operating_point_extension_is_complete(directory, parsed)
    )
    checks["specific:nested-operating-point-extension"] = {
        "complete": nested_complete,
        "reason": nested_reason,
    }
    prescreen_complete, prescreen_reason = (
        _nested_prescreen_stability_is_complete(directory, parsed)
    )
    checks["specific:nested-prescreen-stability"] = {
        "complete": prescreen_complete,
        "reason": prescreen_reason,
    }
    return checks


def _collect_result_group(
    repo_root: Path,
    spec: ResultGroup,
) -> tuple[list[Payload], dict]:
    directory = repo_root / "results" / spec.name
    if not directory.is_dir():
        return [], {
            "status": "missing",
            "included_files": 0,
            "missing_required_files": list(spec.required_files),
            "aggregate_checks": {},
        }

    selected: dict[str, Path] = {}
    for pattern in spec.include_patterns:
        for path in directory.glob(pattern):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = path.relative_to(directory).as_posix()
            if any(
                "smoke" in part.lower()
                for part in PurePosixPath(relative).parts
            ):
                continue
            selected[relative] = path

    payloads: list[Payload] = []
    parsed: dict[str, object] = {}
    for relative, path in sorted(selected.items()):
        data = _read_regular_file(path, repo_root)
        if path.suffix == ".json":
            parsed[relative] = _strict_json_loads(
                data, f"results/{spec.name}/{relative}"
            )
        payloads.append(Payload(f"results/{spec.name}/{relative}", data))

    missing = sorted(
        relative
        for relative in spec.required_files
        if not (directory / relative).is_file()
    )
    aggregate_checks: dict[str, dict[str, object]] = {}
    aggregates_complete = True
    for relative in spec.aggregate_files:
        if relative not in parsed:
            aggregate_checks[relative] = {
                "complete": False,
                "reason": "missing or not selected",
            }
            aggregates_complete = False
            continue
        complete, reason = _aggregate_is_complete(parsed[relative])
        aggregate_checks[relative] = {
            "complete": complete,
            "reason": reason,
        }
        aggregates_complete &= complete

    specific_checks = _result_specific_checks(spec, parsed, directory)
    if spec.name == "revision_tuning":
        for stage, stage_label in (
            ("strength_extension", "strength"),
            ("nested_tuning", "nested"),
            ("fresh_interpolation", "fresh"),
        ):
            complete_snapshot, snapshot_reason = (
                _revision_stage_source_snapshot_is_complete(
                    directory,
                    parsed,
                    stage,
                    stage_label,
                )
            )
            specific_checks[f"specific:{stage_label}-stage-source-snapshot"] = {
                "complete": complete_snapshot,
                "reason": snapshot_reason,
            }
    for label, check in specific_checks.items():
        aggregate_checks[label] = check
        aggregates_complete &= bool(check["complete"])

    complete = not missing and aggregates_complete
    return payloads, {
        "status": "complete" if complete else "partial",
        "included_files": len(payloads),
        "missing_required_files": missing,
        "aggregate_checks": aggregate_checks,
    }


def _parse_checksum_file(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidencePackageError("baseline checksum file is not UTF-8") from error
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        pieces = line.split()
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise EvidencePackageError(
                f"invalid baseline checksum line {line_number}: {line!r}"
            )
        digest, name = pieces
        try:
            int(digest, 16)
        except ValueError as error:
            raise EvidencePackageError(
                f"invalid baseline digest on line {line_number}"
            ) from error
        safe_name = _safe_relative_path(name)
        if len(safe_name.parts) != 1:
            raise EvidencePackageError(
                f"baseline checksum must name a direct child: {name}"
            )
        if name in checksums:
            raise EvidencePackageError(
                f"duplicate baseline checksum entry: {name}"
            )
        checksums[name] = digest.lower()
    return checksums


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return (mode & 0o170000) == 0o120000


def _validate_nested_archive(path: Path) -> None:
    """Reject unsafe members in an opaque baseline archive."""
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise EvidencePackageError(
                    f"baseline archive has duplicate members: {path.name}"
                )
            for info in infos:
                _safe_relative_path(info.filename.rstrip("/"))
                if info.flag_bits & 0x1:
                    raise EvidencePackageError(
                        f"encrypted baseline member: {path.name}:{info.filename}"
                    )
                unix_type = (info.external_attr >> 16) & 0o170000
                supported_types = (
                    (0, 0o040000)
                    if info.is_dir()
                    else (0, 0o100000)
                )
                if _zip_member_is_symlink(info) or unix_type not in supported_types:
                    raise EvidencePackageError(
                        "baseline archive has unsupported member: "
                        f"{path.name}:{info.filename}"
                    )
        return

    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise EvidencePackageError(
                    f"baseline archive has duplicate members: {path.name}"
                )
            for member in members:
                _safe_relative_path(member.name.rstrip("/"))
                if not (member.isdir() or member.isfile()):
                    raise EvidencePackageError(
                        f"baseline archive has unsupported member: "
                        f"{path.name}:{member.name}"
                    )
        return

    raise EvidencePackageError(f"unsupported baseline archive type: {path.name}")


def _collect_baseline_archives(
    repo_root: Path,
) -> tuple[list[Payload], dict]:
    results_dir = repo_root / "results"
    checksum_path = results_dir / BASELINE_CHECKSUM_FILE
    payloads: list[Payload] = []
    checksums: dict[str, str] = {}
    if checksum_path.is_file():
        checksum_data = _read_regular_file(checksum_path, repo_root)
        checksums = _parse_checksum_file(checksum_data)

    missing: list[str] = []
    validated: dict[str, str] = {}
    for name in BASELINE_ARCHIVES:
        path = results_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        data = _read_regular_file(path, repo_root)
        actual = sha256_bytes(data)
        expected = checksums.get(name)
        if expected is None:
            missing.append(f"checksum:{name}")
        elif actual != expected:
            raise EvidencePackageError(
                f"baseline archive checksum mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
        else:
            validated[name] = actual
        _validate_nested_archive(path)
        payloads.append(Payload(f"results/{name}", data))

    if not checksum_path.is_file():
        missing.append(BASELINE_CHECKSUM_FILE)
    extra_checksum_entries = sorted(set(checksums) - set(BASELINE_ARCHIVES))
    if checksums:
        subset_manifest = "".join(
            f"{validated[name]}  {name}\n"
            for name in BASELINE_ARCHIVES
            if name in validated
        )
        payloads.append(
            Payload(
                f"results/{BASELINE_CHECKSUM_FILE}",
                subset_manifest.encode("utf-8"),
            )
        )
    return payloads, {
        "status": "complete" if not missing else "partial",
        "included_files": len(payloads),
        "missing_required_files": sorted(missing),
        "validated_sha256": validated,
        "checksum_entries_not_packaged": extra_checksum_entries,
    }


def _ensure_unique_payloads(payloads: Iterable[Payload]) -> list[Payload]:
    by_name: dict[str, Payload] = {}
    for payload in payloads:
        safe = _safe_relative_path(payload.name)
        normalized = safe.as_posix()
        if normalized in by_name:
            raise EvidencePackageError(f"duplicate payload path: {normalized}")
        by_name[normalized] = Payload(normalized, payload.data)
    return [by_name[name] for name in sorted(by_name)]


def _readme_bytes(statuses: dict[str, object]) -> bytes:
    result_rows = []
    for name, status in statuses["result_groups"].items():
        result_rows.append(
            f"| `{name}` | {status['status']} | {status['included_files']} |"
        )
    activity_group = statuses["result_groups"].get(
        "activity_matched_response",
        {},
    )
    activity_checks = activity_group.get("aggregate_checks", {})
    activity_check = activity_checks.get(
        "specific:activity-matched-response",
        {},
    )
    activity_reason = str(activity_check.get("reason", "not available"))
    if "outcome-neutral feasibility failure" in activity_reason:
        activity_verification = (
            "# Activity protocol terminal state: outcome-neutral feasibility "
            "failure.\n"
            "# PROVENANCE.json records the authenticated 160-row pilot and "
            "240-cell calibration audit;\n"
            "# no frozen calibration, task scores, or aggregate exist by "
            "design."
        )
    elif activity_check.get("complete") is True:
        activity_verification = (
            "PYTHONPATH=src:experiments python \\\n"
            "  experiments/run_activity_matched_response.py validate"
        )
    else:
        activity_verification = (
            "# Activity protocol artifacts are absent or incomplete in this "
            "snapshot."
        )
    completeness = (
        "complete and ready for a final evidence snapshot"
        if statuses["complete"]
        else "a partial snapshot while one or more analyses are still running"
    )
    text = f"""# QRC dissipation-engineering revision evidence

This is {completeness}.  It accompanies the Quantum revision and contains the
scientific and manuscript source, tests, current analysis reports, named
revision result groups, and checksum-validated baseline archives.  It
intentionally excludes rendered PDFs, caches, version-control metadata,
credential-like paths, smoke outputs, and superseded
`measurement_full`/`measurement_full_v2` results.

## Evidence authority

The reports directly under `reports/` in this package are the current narrative
authority.  The two nested baseline archives provide the raw main/review
protocol evidence; historical measurement and same-seed prospective archives
are deliberately omitted because the sealed revision groups supersede them.
The single frozen diagnostic file consumed by the fresh-interpolation driver
is retained at its original relative path and authenticated against that
stage's sealed manifest.
For parity and system-size conclusions, use `revision_parity_control` and
`revision_normalized_scaling`, respectively, rather than the corresponding
legacy rows in the baseline archive.

## Result groups

| group | status | files |
|---|---:|---:|
{chr(10).join(result_rows)}

Release-documentation gate: **{statuses["release_documentation"]["status"]}**.
Exact missing files or pending final-seal/figure-QA lines are recorded in
`PROVENANCE.json`.

`PROVENANCE.json` records exact group checks, Git state, and the builder
contract. `SHA256SUMS.txt` authenticates every other member in this package.
The archive's adjacent `.sha256` file authenticates the archive itself.

## Verification

From the repository used to build this archive:

```bash
python3 scripts/build_revision_evidence_package.py verify \\
  --require-complete PATH_TO_ARCHIVE
```

To reproduce the final package, first complete the experiment drivers described
in the included source, then run:

```bash
python3 scripts/build_revision_evidence_package.py build \\
  --require-complete \\
  --output results/qrc_dissipation_reproducibility_package.zip
```

To reconstruct a runnable checkout from an extracted evidence package, run the
following commands from the package's top-level `{ARCHIVE_ROOT}` directory.
The separate reconstruction directory is necessary because repository files
live under `source/`, while the packaged result groups live under `results/`:

```bash
mkdir reproduction
cp -R source/. reproduction/
cp -R results reproduction/
cp -R reports reproduction/
cd reproduction
tar -xzf results/final_protocol_results.tar.gz -C results
tar -xzf results/review_protocol_results.tar.gz -C results
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src:experiments python -m pytest -q
PYTHONPATH=src:experiments python \
  experiments/validate_revision_primary_regularization_artifacts.py
PYTHONPATH=src:experiments python \
  experiments/validate_collective_loss_full_input_artifacts.py
PYTHONPATH=src:experiments python \
  experiments/validate_nested_operating_point_artifacts.py
{activity_verification}
MPLCONFIGDIR=.mplconfig python paper/make_figures.py
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error dissipation_qrc.tex
```

No machine-local absolute path or environment variable is stored in the
package.  All archive paths are relative to the single top-level directory
`{ARCHIVE_ROOT}`.
"""
    return text.encode("utf-8")


def _build_provenance(
    repo_root: Path,
    output: Path,
    statuses: dict[str, object],
) -> bytes:
    sidecar = output.with_name(output.name + ".sha256")
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "qrc_dissipation_quantum_revision_evidence",
        "archive_root": ARCHIVE_ROOT,
        "determinism": {
            "member_order": "UTF-8 path sort",
            "file_mode": f"{FILE_MODE:o}",
            "tar_mtime": TAR_TIMESTAMP,
            "zip_timestamp": list(ZIP_TIMESTAMP),
            "generated_wallclock_time_included": False,
        },
        "git": {
            "head": _git_value(repo_root, ["rev-parse", "HEAD"], "unavailable"),
            "branch": _git_value(
                repo_root, ["branch", "--show-current"], "unavailable"
            ),
            "dirty_status": _git_status(
                repo_root,
                (output, sidecar),
            ),
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "selection_policy": {
            "source_patterns": list(SOURCE_PATTERNS),
            "current_reports": list(CURRENT_REPORTS),
            "result_groups": [spec.name for spec in RESULT_GROUPS],
            "result_dependencies": list(RESULT_DEPENDENCIES),
            "baseline_archives": list(BASELINE_ARCHIVES),
            "superseded_result_groups_excluded": [
                "measurement_full",
                "measurement_full_v2",
                "grouped_measurement_results",
                "activity_matched_response_exploratory_branch_audit",
                "activity_matched_response_failed_v2_reachability_audit",
            ],
            "superseded_result_groups_partially_retained": {
                "quantum_strengthening_v2_paper": [
                    "frozen_diagnostic_predictions.json"
                ]
            },
            "smoke_outputs_excluded": True,
            "rendered_manuscript_and_build_outputs_excluded": True,
            "credential_like_paths_excluded": True,
        },
        **statuses,
    }
    return _canonical_json(provenance)


def collect_payloads(
    repo_root: Path,
    output: Path,
    require_complete: bool,
) -> tuple[list[Payload], bool]:
    repo_root = repo_root.resolve()
    sidecar = output.with_name(output.name + ".sha256")
    if require_complete:
        _require_clean_committed_git(repo_root, (output, sidecar))
    source_payloads, source_status = _collect_scientific_source(repo_root)
    report_payloads, report_status = _collect_reports(repo_root)
    release_status = _release_documentation_status(repo_root)
    baseline_payloads, baseline_status = _collect_baseline_archives(repo_root)
    dependency_payloads, dependency_status = _collect_result_dependencies(
        repo_root
    )

    result_payloads: list[Payload] = []
    result_statuses: dict[str, dict] = {}
    for spec in RESULT_GROUPS:
        group_payloads, status = _collect_result_group(repo_root, spec)
        result_payloads.extend(group_payloads)
        result_statuses[spec.name] = status

    complete = (
        source_status["status"] == "complete"
        and report_status["status"] == "complete"
        and release_status["status"] == "complete"
        and baseline_status["status"] == "complete"
        and dependency_status["status"] == "complete"
        and all(
            status["status"] == "complete"
            for status in result_statuses.values()
        )
    )
    statuses = {
        "complete": complete,
        "source": source_status,
        "reports": report_status,
        "release_documentation": release_status,
        "baseline_archives": baseline_status,
        "result_dependencies": dependency_status,
        "result_groups": result_statuses,
    }
    if require_complete and not complete:
        incomplete = {
            "source": source_status,
            "reports": report_status,
            "release_documentation": release_status,
            "baseline_archives": baseline_status,
            "result_dependencies": dependency_status,
            "result_groups": {
                name: status
                for name, status in result_statuses.items()
                if status["status"] != "complete"
            },
        }
        raise IncompleteEvidenceError(
            "final evidence is incomplete:\n"
            + json.dumps(incomplete, indent=2, sort_keys=True)
        )

    core = _ensure_unique_payloads(
        [
            *source_payloads,
            *report_payloads,
            *baseline_payloads,
            *dependency_payloads,
            *result_payloads,
        ]
    )
    readme = Payload("README.md", _readme_bytes(statuses))
    provenance = Payload(
        "PROVENANCE.json",
        _build_provenance(repo_root, output, statuses),
    )
    checksummed = _ensure_unique_payloads([*core, readme, provenance])
    manifest_lines = [
        f"{sha256_bytes(payload.data)}  {payload.name}"
        for payload in checksummed
    ]
    manifest = Payload(
        "SHA256SUMS.txt",
        ("\n".join(manifest_lines) + "\n").encode("utf-8"),
    )
    return _ensure_unique_payloads([*checksummed, manifest]), complete


def _prefixed_payloads(payloads: Iterable[Payload]) -> list[Payload]:
    return [
        Payload(
            (PurePosixPath(ARCHIVE_ROOT) / payload.name).as_posix(),
            payload.data,
        )
        for payload in payloads
    ]


def _archive_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar.gz"
    raise EvidencePackageError(
        "output must end in .zip, .tar.gz, or .tgz"
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o100000 | FILE_MODE) << 16
    info.flag_bits = 0
    return info


def _write_zip(path: Path, payloads: Iterable[Payload]) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as bundle:
        for payload in payloads:
            bundle.writestr(
                _zip_info(payload.name),
                payload.data,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _tar_info(payload: Payload) -> tarfile.TarInfo:
    info = tarfile.TarInfo(payload.name)
    info.type = tarfile.REGTYPE
    info.size = len(payload.data)
    info.mode = FILE_MODE
    info.mtime = TAR_TIMESTAMP
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.pax_headers = {}
    return info


def _write_tar_gz(path: Path, payloads: Iterable[Payload]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=9,
            mtime=TAR_TIMESTAMP,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as bundle:
                for payload in payloads:
                    bundle.addfile(
                        _tar_info(payload),
                        io.BytesIO(payload.data),
                    )


def _atomic_write_archive(
    output: Path,
    payloads: Sequence[Payload],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        suffix = ".zip" if _archive_format(output) == "zip" else ".tar.gz"
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=suffix,
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        if _archive_format(output) == "zip":
            _write_zip(temporary, payloads)
        else:
            _write_tar_gz(temporary, payloads)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _archive_members(path: Path) -> list[Payload]:
    archive_format = _archive_format(path)
    payloads: list[Payload] = []
    if archive_format == "zip":
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise EvidencePackageError("archive contains duplicate members")
            if names != sorted(names):
                raise EvidencePackageError("archive members are not sorted")
            for info in infos:
                _safe_relative_path(info.filename)
                if info.is_dir() or _zip_member_is_symlink(info):
                    raise EvidencePackageError(
                        f"archive contains non-regular member: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise EvidencePackageError(
                        f"archive contains encrypted member: {info.filename}"
                    )
                if info.date_time != ZIP_TIMESTAMP:
                    raise EvidencePackageError(
                        f"ZIP timestamp is not normalized: {info.filename}"
                    )
                mode = (info.external_attr >> 16) & 0o777
                if mode != FILE_MODE:
                    raise EvidencePackageError(
                        f"ZIP mode is not normalized: {info.filename}"
                    )
                payloads.append(Payload(info.filename, bundle.read(info)))
        return payloads

    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise EvidencePackageError("archive contains duplicate members")
        if names != sorted(names):
            raise EvidencePackageError("archive members are not sorted")
        for member in members:
            _safe_relative_path(member.name)
            if not member.isfile():
                raise EvidencePackageError(
                    f"archive contains non-regular member: {member.name}"
                )
            if (
                member.mtime != TAR_TIMESTAMP
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
                or member.mode != FILE_MODE
            ):
                raise EvidencePackageError(
                    f"tar metadata is not normalized: {member.name}"
                )
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise EvidencePackageError(
                    f"cannot read archive member: {member.name}"
                )
            payloads.append(Payload(member.name, extracted.read()))
    return payloads


def _parse_payload_manifest(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidencePackageError("payload manifest is not UTF-8") from error
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        pieces = line.split(maxsplit=1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise EvidencePackageError(
                f"invalid payload manifest line {line_number}"
            )
        digest = pieces[0].lower()
        try:
            int(digest, 16)
        except ValueError as error:
            raise EvidencePackageError(
                f"invalid payload digest on line {line_number}"
            ) from error
        name = pieces[1].lstrip()
        _safe_relative_path(name)
        if name in checksums:
            raise EvidencePackageError(
                f"duplicate payload checksum entry: {name}"
            )
        checksums[name] = digest
    return checksums


def _verify_sidecar(path: Path, digest: str) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    if not sidecar.exists():
        raise EvidencePackageError(f"missing archive sidecar: {sidecar}")
    try:
        fields = sidecar.read_text(encoding="utf-8").split()
    except UnicodeDecodeError as error:
        raise EvidencePackageError("archive sidecar is not UTF-8") from error
    if fields != [digest, path.name]:
        raise EvidencePackageError(
            f"archive sidecar mismatch: expected {digest}  {path.name}"
        )


def verify_archive(
    path: Path,
    *,
    check_sidecar: bool = True,
    require_complete: bool = False,
) -> dict:
    """Verify safety, metadata, exact membership, and every payload checksum."""
    path = path.resolve()
    payloads = _archive_members(path)
    by_name = {payload.name: payload.data for payload in payloads}
    manifest_name = f"{ARCHIVE_ROOT}/SHA256SUMS.txt"
    if manifest_name not in by_name:
        raise EvidencePackageError("archive is missing SHA256SUMS.txt")
    checksums = _parse_payload_manifest(by_name[manifest_name])

    expected = {
        f"{ARCHIVE_ROOT}/{relative}"
        for relative in checksums
    } | {manifest_name}
    actual = set(by_name)
    if actual != expected:
        raise EvidencePackageError(
            "archive membership does not match payload manifest: "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for relative, expected_digest in checksums.items():
        member = f"{ARCHIVE_ROOT}/{relative}"
        actual_digest = sha256_bytes(by_name[member])
        if actual_digest != expected_digest:
            raise EvidencePackageError(
                f"payload checksum mismatch for {relative}: "
                f"expected {expected_digest}, got {actual_digest}"
            )

    provenance_name = f"{ARCHIVE_ROOT}/PROVENANCE.json"
    readme_name = f"{ARCHIVE_ROOT}/README.md"
    for required in (provenance_name, readme_name):
        if required not in by_name:
            raise EvidencePackageError(
                f"archive is missing required top-level file: {required}"
            )
    provenance = _strict_json_loads(
        by_name[provenance_name],
        provenance_name,
    )
    archive_digest = sha256_file(path)
    if check_sidecar:
        _verify_sidecar(path, archive_digest)
    complete = bool(provenance["complete"])
    if require_complete and not complete:
        raise IncompleteEvidenceError(
            f"evidence archive is structurally valid but incomplete: {path}"
        )
    return {
        "path": str(path),
        "sha256": archive_digest,
        "file_count": len(payloads),
        "complete": complete,
    }


def build_package(
    repo_root: Path,
    output: Path,
    *,
    require_complete: bool = False,
) -> BuildSummary:
    """Collect, write, validate, and checksum one deterministic package."""
    output = output.resolve()
    payloads, complete = collect_payloads(
        repo_root,
        output,
        require_complete=require_complete,
    )
    prefixed = _prefixed_payloads(payloads)
    _atomic_write_archive(output, prefixed)
    verified = verify_archive(output, check_sidecar=False)
    sidecar = output.with_name(output.name + ".sha256")
    digest = verified["sha256"]
    temporary_sidecar: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{sidecar.name}.",
            suffix=".tmp",
            dir=sidecar.parent,
            delete=False,
        ) as handle:
            temporary_sidecar = Path(handle.name)
            handle.write(f"{digest}  {output.name}\n".encode("utf-8"))
        os.replace(temporary_sidecar, sidecar)
        temporary_sidecar = None
    finally:
        if temporary_sidecar is not None:
            temporary_sidecar.unlink(missing_ok=True)
    verify_archive(output)
    return BuildSummary(
        path=output,
        sidecar=sidecar,
        sha256=digest,
        file_count=int(verified["file_count"]),
        complete=complete,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build and validate a package")
    build.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"repository root (default: {REPO_ROOT})",
    )
    build.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f".zip or .tar.gz destination (default: {DEFAULT_OUTPUT})",
    )
    build.add_argument(
        "--require-complete",
        action="store_true",
        help="fail unless all final source, report, baseline, and result checks pass",
    )

    verify = subparsers.add_parser(
        "verify", help="verify paths, metadata, membership, and hashes"
    )
    verify.add_argument("archive", type=Path)
    verify.add_argument(
        "--require-complete",
        action="store_true",
        help="also fail unless PROVENANCE.json marks the evidence complete",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        summary = build_package(
            args.repo_root,
            args.output,
            require_complete=args.require_complete,
        )
        print(
            f"OK {summary.path} files={summary.file_count} "
            f"complete={str(summary.complete).lower()} "
            f"sha256={summary.sha256}"
        )
        print(f"OK {summary.sidecar}")
        return 0

    verified = verify_archive(
        args.archive,
        require_complete=args.require_complete,
    )
    print(
        f"OK {verified['path']} files={verified['file_count']} "
        f"complete={str(verified['complete']).lower()} "
        f"sha256={verified['sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

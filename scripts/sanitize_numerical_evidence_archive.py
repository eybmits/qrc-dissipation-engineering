#!/usr/bin/env python3
"""Deterministically sanitize the canonical numerical-evidence archive.

The release archive is otherwise immutable scientific evidence, so this
script performs only three narrowly scoped transformations:

* remove the accidental Python bytecode/cache member;
* replace two transient build-environment paths with a stable placeholder;
* regenerate every affected checksum manifest.

The ZIP is then rewritten atomically with sorted members, a fixed timestamp,
regular-file permissions, and deterministic DEFLATE settings.  Re-running the
script on an already sanitized archive produces the same bytes and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "results"
    / "collective_loss_usable_memory_numerical_evidence.zip"
)
ARCHIVE_ROOT = "collective-loss-numerical-evidence"
CHECKSUM_NAME = "SHA256SUMS"
NORMALIZED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
NORMALIZED_MODE = stat.S_IFREG | 0o644
COMPRESSION_LEVEL = 9

ACCIDENTAL_CACHE_MEMBER = (
    f"{ARCHIVE_ROOT}/switched_input_memory_control_v2/__pycache__/"
    "run_switched_input_memory_control.cpython-313.pyc"
)
ENVIRONMENT_MEMBERS = (
    f"{ARCHIVE_ROOT}/canonical_gap_control/environment.json",
    f"{ARCHIVE_ROOT}/midpoint_gap_control/"
    "canonical_gap_control/environment.json",
)
PYTHON_PATH_PREFIX = b"Python Information:\\n  path: "
PYTHON_PATH_SUFFIX = b"\\n  version: '3.13'"
NEUTRAL_PYTHON_PATH = b"<isolated-build-environment>/bin/python"
FORBIDDEN_CONTENT_MARKERS = (
    b"/" + b"Users" + b"/",
    b"/private/" + b"var/folders" + b"/",
    b"/var/folders/",
)


class SanitizationError(RuntimeError):
    """Raised when the archive does not match the expected safe structure."""


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of *data*."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_name(value: str) -> PurePosixPath:
    """Validate and return one portable, relative archive member path."""

    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or ":" in path.parts[0]
    ):
        raise SanitizationError(f"unsafe archive member: {value!r}")
    if path.parts[0] != ARCHIVE_ROOT:
        raise SanitizationError(f"unexpected archive root: {value!r}")
    return path


def load_archive(path: Path) -> dict[str, bytes]:
    """Load a normalized regular-file-only ZIP into memory."""

    if not path.is_file():
        raise SanitizationError(f"missing archive: {path}")
    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SanitizationError("archive contains duplicate members")
            if names != sorted(names):
                raise SanitizationError("archive members are not sorted")
            files: dict[str, bytes] = {}
            for info in infos:
                safe_member_name(info.filename)
                if info.is_dir():
                    raise SanitizationError(
                        f"directory member is not permitted: {info.filename}"
                    )
                unix_type = (info.external_attr >> 16) & 0o170000
                if unix_type not in (0, stat.S_IFREG):
                    raise SanitizationError(
                        f"non-regular archive member: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise SanitizationError(
                        f"encrypted archive member: {info.filename}"
                    )
                files[info.filename] = bundle.read(info)
            return files
    except zipfile.BadZipFile as error:
        raise SanitizationError(f"invalid ZIP archive: {path}") from error


def checksum_manifest(files: dict[str, bytes], manifest_name: str) -> bytes:
    """Render the checksum manifest for one archive subtree.

    Each manifest covers every non-manifest file below its parent directory.
    This matches the evidence bundle's existing nested and outer conventions.
    """

    parent = str(PurePosixPath(manifest_name).parent)
    prefix = parent + "/"
    rows: list[str] = []
    for name in sorted(files):
        if not name.startswith(prefix):
            continue
        relative = name[len(prefix) :]
        if PurePosixPath(relative).name == CHECKSUM_NAME:
            continue
        rows.append(f"{sha256_bytes(files[name])}  {relative}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def refresh_checksum_manifests(files: dict[str, bytes]) -> None:
    """Regenerate all existing nested and outer checksum manifests."""

    manifests = sorted(
        name
        for name in files
        if PurePosixPath(name).name == CHECKSUM_NAME
    )
    if not manifests:
        raise SanitizationError("archive contains no checksum manifests")
    expected_outer = f"{ARCHIVE_ROOT}/{CHECKSUM_NAME}"
    if expected_outer not in manifests:
        raise SanitizationError(f"missing outer checksum manifest: {expected_outer}")
    for manifest in manifests:
        files[manifest] = checksum_manifest(files, manifest)


def verify_checksum_manifests(files: dict[str, bytes]) -> None:
    """Verify all checksum manifests against the transformed payload."""

    for name, data in files.items():
        if PurePosixPath(name).name != CHECKSUM_NAME:
            continue
        expected = checksum_manifest(files, name)
        if data != expected:
            raise SanitizationError(f"checksum manifest mismatch: {name}")


def sanitize_payload(files: dict[str, bytes]) -> tuple[int, int]:
    """Apply the allowlisted sanitation and return change counts."""

    cache_members = sorted(
        name
        for name in files
        if "__pycache__" in PurePosixPath(name).parts
        or name.lower().endswith((".pyc", ".pyo"))
    )
    unexpected = [
        name for name in cache_members if name != ACCIDENTAL_CACHE_MEMBER
    ]
    if unexpected:
        raise SanitizationError(
            "unexpected cache members require manual review: "
            + ", ".join(unexpected)
        )
    removed = int(files.pop(ACCIDENTAL_CACHE_MEMBER, None) is not None)

    neutralized = 0
    for name in ENVIRONMENT_MEMBERS:
        try:
            data = files[name]
        except KeyError as error:
            raise SanitizationError(f"missing environment record: {name}") from error
        prefix_index = data.find(PYTHON_PATH_PREFIX)
        if prefix_index < 0:
            raise SanitizationError(f"missing Python path record in {name}")
        path_start = prefix_index + len(PYTHON_PATH_PREFIX)
        path_end = data.find(PYTHON_PATH_SUFFIX, path_start)
        if path_end < 0:
            raise SanitizationError(f"malformed Python path record in {name}")
        recorded_path = data[path_start:path_end]
        if recorded_path == NEUTRAL_PYTHON_PATH:
            continue
        if not recorded_path.endswith(b"/bin/python"):
            raise SanitizationError(
                f"unexpected Python executable record in {name}"
            )
        if recorded_path.startswith(b"<"):
            raise SanitizationError(
                f"unexpected pre-normalized Python path in {name}"
            )
        files[name] = (
            data[:path_start]
            + NEUTRAL_PYTHON_PATH
            + data[path_end:]
        )
        neutralized += 1

    remaining_cache = [
        name
        for name in files
        if "__pycache__" in PurePosixPath(name).parts
        or name.lower().endswith((".pyc", ".pyo"))
    ]
    if remaining_cache:
        raise SanitizationError(
            "cache members remain after sanitation: " + ", ".join(remaining_cache)
        )
    for name, data in files.items():
        for marker in FORBIDDEN_CONTENT_MARKERS:
            if marker.lower() in data.lower():
                raise SanitizationError(
                    f"machine-local marker {marker!r} remains in {name}"
                )
    return removed, neutralized


def normalized_zip_info(name: str) -> zipfile.ZipInfo:
    """Create deterministic metadata for one regular archive member."""

    info = zipfile.ZipInfo(name, NORMALIZED_TIMESTAMP)
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = NORMALIZED_MODE << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def write_archive(path: Path, files: dict[str, bytes]) -> None:
    """Atomically replace *path* with a deterministic ZIP."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=COMPRESSION_LEVEL,
            strict_timestamps=True,
        ) as bundle:
            bundle.comment = b""
            for name in sorted(files):
                bundle.writestr(
                    normalized_zip_info(name),
                    files[name],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=COMPRESSION_LEVEL,
                )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sanitize_archive(path: Path) -> dict[str, object]:
    """Sanitize *path* and return a machine-readable summary."""

    path = path.resolve()
    old_sha256 = sha256_file(path)
    files = load_archive(path)
    old_member_count = len(files)
    removed, neutralized = sanitize_payload(files)
    refresh_checksum_manifests(files)
    verify_checksum_manifests(files)
    write_archive(path, files)

    written = load_archive(path)
    verify_checksum_manifests(written)
    if written != files:
        raise SanitizationError("written archive payload differs from input payload")
    return {
        "archive": str(path),
        "old_sha256": old_sha256,
        "new_sha256": sha256_file(path),
        "old_member_count": old_member_count,
        "new_member_count": len(written),
        "removed_cache_members": removed,
        "neutralized_paths": neutralized,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    try:
        summary = sanitize_archive(args.archive)
    except (OSError, SanitizationError) as error:
        print(f"FAIL: {error}")
        return 1
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

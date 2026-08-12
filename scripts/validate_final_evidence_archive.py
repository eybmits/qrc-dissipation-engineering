#!/usr/bin/env python3
"""Validate the canonical numerical-evidence archive from a clean checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT / "results" / "collective_loss_usable_memory_numerical_evidence.zip"
)
EXPECTED_ROOT = "collective-loss-numerical-evidence"
FORBIDDEN_MEMBER_PARTS = {"__pycache__", ".pytest_cache", ".DS_Store"}
FORBIDDEN_MEMBER_SUFFIXES = {".pyc", ".pyo"}
FORBIDDEN_MACHINE_PATHS = (
    b"/" + b"Users" + b"/",
    b"/private/" + b"var/folders" + b"/",
    b"/private/" + b"tmp" + b"/",
    b"C:\\Users\\",
)


class EvidenceArchiveError(RuntimeError):
    """Raised when the outer archive cannot be validated safely."""


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or "\x00" in name
        or ":" in path.parts[0]
    ):
        raise EvidenceArchiveError(f"unsafe archive path: {name}")
    return path


def validate(archive: Path) -> None:
    archive = archive.resolve()
    if not archive.is_file():
        raise EvidenceArchiveError(f"missing archive: {archive}")

    with tempfile.TemporaryDirectory(prefix="qrc-final-evidence-") as temporary:
        destination = Path(temporary)
        try:
            with zipfile.ZipFile(archive) as bundle:
                infos = bundle.infolist()
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    raise EvidenceArchiveError("archive contains duplicate members")
                for info in infos:
                    path = _safe_member(info.filename)
                    if path.parts[0] != EXPECTED_ROOT:
                        raise EvidenceArchiveError(
                            f"unexpected archive root: {path.parts[0]}"
                        )
                    if (
                        any(part in FORBIDDEN_MEMBER_PARTS for part in path.parts)
                        or path.suffix.lower() in FORBIDDEN_MEMBER_SUFFIXES
                    ):
                        raise EvidenceArchiveError(
                            f"archive contains a cache artifact: {info.filename}"
                        )
                    unix_type = (info.external_attr >> 16) & 0o170000
                    if unix_type not in (0, 0o040000, 0o100000):
                        raise EvidenceArchiveError(
                            f"archive contains a non-regular member: {info.filename}"
                        )
                    if info.flag_bits & 0x1:
                        raise EvidenceArchiveError(
                            f"archive contains an encrypted member: {info.filename}"
                        )
                bundle.extractall(destination)
        except (OSError, zipfile.BadZipFile) as error:
            raise EvidenceArchiveError(f"cannot read archive: {archive}") from error

        root = destination / EXPECTED_ROOT
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            payload = path.read_bytes()
            for marker in FORBIDDEN_MACHINE_PATHS:
                if marker in payload:
                    relative = path.relative_to(root).as_posix()
                    raise EvidenceArchiveError(
                        f"archive contains a machine-local path in {relative}"
                    )
        validator = root / "validate_bundle.py"
        if not validator.is_file():
            raise EvidenceArchiveError("archive lacks validate_bundle.py")
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
            raise EvidenceArchiveError(
                f"nested validator failed with exit code {completed.returncode}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    try:
        validate(args.archive)
    except EvidenceArchiveError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Validated canonical evidence archive: {args.archive.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

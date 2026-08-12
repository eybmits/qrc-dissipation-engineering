#!/usr/bin/env python3
"""Portably validate the frozen nested operating-point extension.

The experiment driver intentionally records the Git commit present when its
protocol is frozen.  Recomputing that historical field from the current
checkout makes the driver's direct ``validate`` command unsuitable after later
documentation commits and in an extracted evidence archive without ``.git``.

This companion validator authenticates the frozen protocol, every scientific
source hash, and the byte-identical driver snapshot first.  It then replays the
driver's existing validation with the authenticated historical commit value.
No scientific payload is rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import run_nested_operating_point_extension as extension


HEX_SHA1 = re.compile(r"[0-9a-f]{40}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _recorded_commit_available(recorded_head: str) -> bool:
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"{recorded_head}^{{commit}}"],
        cwd=extension.REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if probe.returncode == 0:
        return True
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=extension.REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if shallow.returncode == 0 and shallow.stdout.strip() == "true":
        return False
    raise RuntimeError("frozen protocol commit is unavailable from full Git history")


def _authenticated_protocol() -> tuple[dict, str]:
    if not extension.MANIFEST_PATH.is_file():
        raise RuntimeError("nested extension manifest is missing")
    manifest = json.loads(extension.MANIFEST_PATH.read_text())
    if (
        manifest.get("artifact_type")
        != "nested_operating_point_extension_manifest"
    ):
        raise RuntimeError("unexpected nested extension manifest type")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict):
        raise RuntimeError("nested extension protocol is missing")
    protocol_sha = manifest.get("protocol_sha256")
    if (
        not isinstance(protocol_sha, str)
        or extension.sha256_json(protocol) != protocol_sha
    ):
        raise RuntimeError("nested extension protocol fingerprint is invalid")

    recorded_head = protocol.get("git_head_at_protocol")
    if (
        not isinstance(recorded_head, str)
        or HEX_SHA1.fullmatch(recorded_head) is None
    ):
        raise RuntimeError("frozen protocol Git head is malformed")

    expected_hashes = protocol.get("scientific_sources_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeError("frozen scientific-source hashes are missing")
    current_hashes = extension._current_source_hashes()  # noqa: SLF001
    if set(current_hashes) != set(expected_hashes):
        raise RuntimeError("frozen scientific-source path set is incomplete")
    for relative, expected in expected_hashes.items():
        if current_hashes.get(relative) != expected:
            raise RuntimeError(
                f"scientific source differs from frozen protocol: {relative}"
            )

    driver_relative = "experiments/run_nested_operating_point_extension.py"
    snapshot = (
        extension.SOURCE_SNAPSHOT_DIR
        / "run_nested_operating_point_extension.py"
    )
    snapshot_manifest_path = extension.SOURCE_SNAPSHOT_DIR / "manifest.json"
    if not snapshot.is_file() or not snapshot_manifest_path.is_file():
        raise RuntimeError("nested extension source snapshot is incomplete")
    snapshot_hash = extension.sha256_file(snapshot)
    if snapshot_hash != expected_hashes.get(driver_relative):
        raise RuntimeError("frozen extension source snapshot hash disagrees")
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    expected_snapshot_path = str(snapshot.relative_to(extension.REPO_ROOT))
    if (
        snapshot_manifest.get("artifact_type")
        != "nested_extension_source_snapshot"
        or snapshot_manifest.get("protocol_sha256") != protocol_sha
        or snapshot_manifest.get("path") != expected_snapshot_path
        or snapshot_manifest.get("sha256") != snapshot_hash
    ):
        raise RuntimeError("nested extension source snapshot manifest is invalid")

    # If Git history is available, also prove that the recorded commit contains
    # the exact frozen scientific sources.  Extracted archives remain portable:
    # their protocol and source snapshots are authenticated by package checksums.
    if (
        extension.git_head() is not None
        and _recorded_commit_available(recorded_head)
    ):
        for relative, expected in expected_hashes.items():
            result = subprocess.run(
                ["git", "show", f"{recorded_head}:{relative}"],
                cwd=extension.REPO_ROOT,
                check=False,
                capture_output=True,
            )
            if result.returncode != 0:
                if relative == driver_relative:
                    # The extension driver was frozen as an authenticated
                    # untracked source snapshot at protocol creation.  Every
                    # pre-existing scientific dependency must still resolve
                    # byte-for-byte from the recorded commit.
                    continue
                raise RuntimeError(
                    f"frozen protocol commit lacks source: {relative}"
                )
            if _sha256_bytes(result.stdout) != expected:
                raise RuntimeError(
                    f"frozen protocol commit source disagrees: {relative}"
                )

    extension.validate_source_protocol()
    return manifest, recorded_head


@contextmanager
def authenticated_frozen_protocol() -> Iterator[dict]:
    """Replay validation with the authenticated historical protocol commit."""

    manifest, recorded_head = _authenticated_protocol()
    original_git_head = extension.git_head
    extension.git_head = lambda: recorded_head
    try:
        yield manifest
    finally:
        extension.git_head = original_git_head


def validate_test_rows() -> list[dict]:
    with authenticated_frozen_protocol():
        return extension.validate_test_rows()


def validate_complete_artifacts() -> dict:
    with authenticated_frozen_protocol():
        return extension.validate_complete_artifacts()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=("validate",),
        default="validate",
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    payload = validate_complete_artifacts()
    print(
        "VALID "
        f"{extension.AGGREGATE_PATH} "
        f"protocol_sha256={payload['protocol_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Portably validate the frozen collective-loss input-grid diagnostic.

The generating protocol records Python, NumPy, and SciPy versions as historical
provenance.  Those values must remain authenticated, but they must not make
verification depend on the validator's current runtime.  This companion checks
the frozen protocol hash, exact current scientific-source hashes, and every
runtime-invariant protocol field before replaying the original artifact
validator with the authenticated frozen protocol.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from typing import Iterator

import run_collective_loss_full_input_diagnostic as diagnostic


SOFTWARE_FIELDS = {"python", "numpy", "scipy"}


def _authenticated_protocol() -> dict:
    if not diagnostic.PROTOCOL_PATH.is_file():
        raise RuntimeError("collective-loss diagnostic protocol is missing")
    payload = json.loads(
        diagnostic.PROTOCOL_PATH.read_text(encoding="utf-8")
    )
    if (
        payload.get("artifact_type")
        != "collective_loss_full_input_protocol"
        or payload.get("status") != "frozen_before_diagnostic_rows"
    ):
        raise RuntimeError("unexpected collective-loss protocol envelope")
    frozen = payload.get("protocol")
    if not isinstance(frozen, dict):
        raise RuntimeError("frozen collective-loss protocol is missing")
    protocol_sha = payload.get("protocol_sha256")
    if (
        not isinstance(protocol_sha, str)
        or diagnostic._sha256_json(frozen) != protocol_sha  # noqa: SLF001
    ):
        raise RuntimeError("frozen collective-loss protocol hash is invalid")

    software = frozen.get("software")
    if (
        not isinstance(software, dict)
        or set(software) != SOFTWARE_FIELDS
        or any(
            not isinstance(software[field], str) or not software[field]
            for field in SOFTWARE_FIELDS
        )
    ):
        raise RuntimeError("frozen collective-loss software provenance is invalid")

    expected = diagnostic.protocol_dict()
    expected["software"] = software
    if expected != frozen:
        raise RuntimeError(
            "frozen collective-loss protocol differs in a scientific field"
        )
    return frozen


@contextmanager
def authenticated_frozen_protocol() -> Iterator[dict]:
    frozen = _authenticated_protocol()
    original_protocol_dict = diagnostic.protocol_dict
    diagnostic.protocol_dict = lambda: frozen
    try:
        yield frozen
    finally:
        diagnostic.protocol_dict = original_protocol_dict


def validate_complete_artifacts() -> dict:
    with authenticated_frozen_protocol():
        return diagnostic.verify_artifacts()


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
    aggregate = validate_complete_artifacts()
    print(
        "VALID "
        f"{diagnostic.AGGREGATE_PATH} "
        f"rows={aggregate['row_count']} "
        f"protocol_sha256={aggregate['protocol_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Portably validate the frozen primary readout-regularisation control.

The protocol's Python, NumPy, and SciPy versions describe the environment that
generated the rows.  This validator authenticates those frozen values and the
exact scientific-source hashes, while permitting verification under a
different current runtime.  All invariant protocol fields, raw rows, aggregate
statistics, audits, and report text are rebuilt and compared exactly.
"""

from __future__ import annotations

import argparse
import json
import math

import run_revision_primary_regularization as control


ENVIRONMENT_FIELDS = {"files", "python", "numpy", "scipy"}
CI_ENDPOINT_FIELDS = {"ci95_low", "ci95_high"}
CI_ENDPOINT_ABS_TOL = 1e-15
CI_ENDPOINT_REL_TOL = 1e-15


def authenticated_protocol() -> dict:
    path = control.protocol_path(control.DEFAULT_OUTDIR)
    if not path.is_file():
        raise RuntimeError("primary regularisation protocol is missing")
    frozen = json.loads(path.read_text())
    source = frozen.get("source_environment")
    source_sha = frozen.get("source_environment_sha256")
    if (
        not isinstance(source, dict)
        or set(source) != ENVIRONMENT_FIELDS
        or not isinstance(source.get("files"), dict)
        or not source["files"]
        or any(
            not isinstance(source.get(field), str) or not source[field]
            for field in ("python", "numpy", "scipy")
        )
        or not isinstance(source_sha, str)
        or control.sha256_json(source) != source_sha
    ):
        raise RuntimeError("frozen primary source environment is invalid")

    current_source = control.source_environment_manifest()
    if current_source["files"] != source["files"]:
        raise RuntimeError(
            "primary scientific sources differ from the frozen protocol"
        )

    expected = control.build_protocol()
    expected["source_environment"] = source
    expected["source_environment_sha256"] = source_sha
    if expected != frozen:
        raise RuntimeError(
            "frozen primary protocol differs in a scientific field"
        )
    return frozen


def _assert_portable_aggregate_equal(
    stored: object,
    rebuilt: object,
    path: tuple[str, ...] = (),
) -> None:
    if type(stored) is not type(rebuilt):
        raise RuntimeError(
            "primary aggregate type mismatch at " + ".".join(path)
        )
    if isinstance(stored, dict):
        if set(stored) != set(rebuilt):
            raise RuntimeError(
                "primary aggregate key mismatch at " + ".".join(path)
            )
        for key in sorted(stored):
            _assert_portable_aggregate_equal(
                stored[key],
                rebuilt[key],
                (*path, str(key)),
            )
        return
    if isinstance(stored, list):
        if len(stored) != len(rebuilt):
            raise RuntimeError(
                "primary aggregate length mismatch at " + ".".join(path)
            )
        for index, (left, right) in enumerate(zip(stored, rebuilt)):
            _assert_portable_aggregate_equal(
                left,
                right,
                (*path, f"[{index}]"),
            )
        return
    if stored == rebuilt:
        return
    if (
        path
        and path[-1] in CI_ENDPOINT_FIELDS
        and isinstance(stored, float)
        and isinstance(rebuilt, float)
        and math.isclose(
            stored,
            rebuilt,
            rel_tol=CI_ENDPOINT_REL_TOL,
            abs_tol=CI_ENDPOINT_ABS_TOL,
        )
    ):
        return
    raise RuntimeError("primary aggregate mismatch at " + ".".join(path))


def validate_complete_artifacts() -> dict:
    protocol = authenticated_protocol()
    rows = control.load_rows(
        control.DEFAULT_OUTDIR,
        protocol,
        require_complete=True,
    )
    aggregate_path = control.aggregate_path(control.DEFAULT_OUTDIR)
    if not aggregate_path.is_file():
        raise RuntimeError("primary regularisation aggregate is missing")
    stored = json.loads(aggregate_path.read_text())
    rebuilt = control.build_aggregate(rows, protocol)
    _assert_portable_aggregate_equal(stored, rebuilt)
    if rebuilt.get("status") != "complete":
        raise RuntimeError("primary regularisation aggregate is not complete")
    if not control.DEFAULT_REPORT.is_file():
        raise RuntimeError("primary regularisation report is missing")
    expected_report = control.render_report(stored, protocol)
    if control.DEFAULT_REPORT.read_text() != expected_report:
        raise RuntimeError("primary regularisation report does not match aggregate")
    if control.render_report(rebuilt, protocol) != expected_report:
        raise RuntimeError("runtime CI drift changes the rendered primary report")
    return stored


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
        f"{control.aggregate_path(control.DEFAULT_OUTDIR)} "
        f"jobs={aggregate['n_jobs']} "
        f"protocol_sha256={aggregate['protocol_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

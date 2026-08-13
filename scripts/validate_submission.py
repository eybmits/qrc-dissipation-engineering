#!/usr/bin/env python3
"""Validate the canonical Quantum submission artifact.

This script intentionally uses only the Python standard library and Poppler
command-line tools so it can run in a clean checkout before submission.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TEX = PAPER / "dissipation_qrc.tex"
PDF = PAPER / "dissipation_qrc.pdf"
LOG = PAPER / "dissipation_qrc.log"
FIGURES = PAPER / "figures"
RESET_BUILDER = PAPER / "make_reset_architecture_figure.py"
RESET_SNAPSHOT = PAPER / "data" / "reset_architecture_snapshot.json"
RESET_DRIVER = ROOT / "experiments" / "run_reset_architecture_strict.py"
PHASE_SNAPSHOT = PAPER / "data" / "phase_direction_confirmatory_snapshot.json"
PHASE_BUILDER = PAPER / "make_phase_direction_figure.py"
PHASE_DRIVER = ROOT / "experiments" / "run_phase_direction_confirmatory.py"
PHASE_VALIDATOR = ROOT / "scripts" / "validate_phase_direction_confirmatory.py"
PHASE_EVIDENCE = PAPER / "evidence" / "phase_direction_confirmatory_v1"
PHASE_RESULT_ARCHIVE = (
    ROOT / "results" / "phase_direction_confirmatory_v1_results.tar.gz"
)
PHASE_EVIDENCE_FILE_SHA256 = {
    "protocol.json": (
        "ef12dd27cd78976e644bbaea10a03c694cf34c8f1b287a0029e30625092f52fc"
    ),
    "aggregate.json": (
        "1112b830924c7b43dbb7c16ca4baf9cb1d4ab5205c67c2b01d5ec430aacb9634"
    ),
    "convergence_summary.json": (
        "1402b70561f2c18a545e5636b10b2042cfc01710d8ff275bcadff408d91a9ce1"
    ),
    "validation_amendment.json": (
        "e842c1f6299ec9d839c36825e8736eb1b053388cc9aff48d17ca0a4d0b767b38"
    ),
    "numerical_replay_audit.json": (
        "e401c62cafe9131cb83fe8f6b0af4ef50bb9a9aa13c75d2def47b38179e1a0d1"
    ),
    "validation_report.json": (
        "2379fa37daf108c53088172f96e20e05e96c09493f3c6ebd0f1f2ec19fab1403"
    ),
}
PHASE_RESULT_ARCHIVE_SHA256 = (
    "ec9a7e6aea148fadf59d83a0967a08cdba6436ad8e2fd8e61e81473226f0f394"
)
ORIENTATION_SNAPSHOT = PAPER / "data" / "rank_one_orientation_snapshot.json"
ORIENTATION_EVIDENCE = PAPER / "evidence" / "rank_one_orientation_v1"
ORIENTATION_RESULT_ARCHIVE = (
    ROOT / "results" / "rank_one_orientation_v1_results.tar.gz"
)
ORIENTATION_EVIDENCE_SHA256 = {
    "protocol.json": (
        "23572d7ee36cf63e5560b310f1bea1453a0d5c275e2f578bae3d2b40bf4e68c4"
    ),
    "derived/summary.json": (
        "72dd00e2dcfccdbae5b2a6bc0feede0e7ad340e08b64fea8df757b25ef0a7f9b"
    ),
    "validation_report.json": (
        "4402d851b5418bc56003d7346d095548996330d303f49d1a78973693394c0a83"
    ),
    "provenance.json": (
        "e8a33914e91fa2cdd43c50c7979a4fa7bbc7e7885206c1ffe10b90e2a7ee29bd"
    ),
    "environment.json": (
        "e1abfd119ce581e86027b5e8ba969870d62cba2299503b4b1e5bc0e1317e2a5a"
    ),
}
ORIENTATION_RESULT_ARCHIVE_SHA256 = (
    "e12f0bdd038b8e45ecea247b09d32815de20330e51ab47aecfe1d995fd86f24a"
)
CONTINUOUS_NARMA_EVIDENCE = (
    PAPER / "evidence" / "continuous_drive_narma_washout_v1"
)
CONTINUOUS_NARMA_RESULT_ARCHIVE = (
    ROOT / "results" / "continuous_drive_narma_washout_v1_results.tar.gz"
)
CONTINUOUS_NARMA_EVIDENCE_SHA256 = {
    "protocol.json": "b1b3ef7411c98d493bd4b16c01efc61de3650e57f45d855baaaba384ea73cf0a",
    "aggregate.json": "701746ebdf95c61bb46a78f30a06ae27c89fd589f76aaf9b84288f00e1e1e95e",
    "validation_report.json": "6002923ddb1ab42f2b1a3c91c2a0e390fd74ef63661b17a2bbf4d523fe088376",
}
CONTINUOUS_NARMA_RESULT_ARCHIVE_SHA256 = (
    "64551dae3f31d3e70320c760e9074b6eafe9736656f6cb91428a691b2550a7ab"
)
RESET_EXPECTED_SEEDS = [
    1162690697,
    411886365,
    1080967412,
    1739603920,
    1154959432,
    600439382,
    1254120429,
    1084176823,
    1869730849,
    56490330,
    1779358140,
    216883587,
    1196651361,
    1669520350,
    1902393916,
    724810199,
]
CODE_RENDERED_COMPOSITES = {
    "figures/fig_task_scores.pdf": {
        "size": (481.89, 156.96),
        "creator": "paper/make_figures.py",
    },
    "figures/fig_collective_case.pdf": {
        "size": (481.89, 182.88),
        "creator": "paper/make_forgetting_modes_figure.py",
        "required_text": (
            "(a)",
            "(b)",
            "(c)",
            "(d)",
            "STM",
            "input delay",
        ),
    },
    "figures/fig_profiles.pdf": {
        "size": (230.982, 172.80),
        "creator": "paper/make_figures.py",
    },
    "figures/fig_sampling.pdf": {
        "size": (230.982, 180.00),
        "creator": "paper/make_figures.py",
    },
    "figures/fig_scalar_controls.pdf": {
        "size": (230.982, 172.80),
        "creator": "paper/make_forgetting_modes_figure.py",
    },
    "figures/fig_reset_architecture.pdf": {
        "size": (230.982, 145.44),
        "creator": "paper/make_reset_architecture_figure.py",
    },
    "figures/fig_phase_direction.pdf": {
        "size": (230.982, 144.00),
        "creator": "paper/make_phase_direction_figure.py",
    },
}


def read_tex_tree(path: Path, seen: set[Path] | None = None) -> str:
    r"""Return a manuscript source closure, following local ``\input`` files."""
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    source = path.read_text(encoding="utf-8")
    parts = [source]
    for input_name in re.findall(r"\\input\{([^}]+)\}", source):
        child = (path.parent / input_name)
        if child.suffix == "":
            child = child.with_suffix(".tex")
        if child.is_file():
            parts.append(read_tex_tree(child, seen))
    return "\n".join(parts)


def run(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=PAPER,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode:
        raise RuntimeError(f"{' '.join(args)} failed:\n{completed.stdout}")
    return completed.stdout


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_reset_snapshot(path: Path, failures: list[str]) -> None:
    """Validate the compact, plotted reset-architecture evidence."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        failures.append(f"cannot read reset-architecture snapshot: {error}")
        return

    check(
        payload.get("schema_version") == 1,
        "reset-architecture snapshot has an unsupported schema",
        failures,
    )
    source = payload.get("source")
    check(
        isinstance(source, dict)
        and source.get("path")
        == "results/reset_architecture_replication/strict_washout_arrays.npz"
        and re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
        is not None,
        "reset-architecture snapshot lacks canonical source provenance",
        failures,
    )
    if isinstance(source, dict):
        canonical = ROOT / str(source.get("path", ""))
        if canonical.is_file():
            digest = hashlib.sha256(canonical.read_bytes()).hexdigest()
            check(
                digest == source.get("sha256"),
                "reset-architecture snapshot hash differs from canonical arrays",
                failures,
            )
    protocol = payload.get("protocol")
    expected_protocol = {
        "architecture": "input-by-reset",
        "n_qubits": 5,
        "gamma": 1.0,
        "assigned_frobenius_budget": 80.0,
        "washout": 800,
        "train": 600,
        "test": 400,
        "stm_delays": [1, 20],
        "pairs": 16,
        "master_seed": 2026080603,
    }
    check(
        isinstance(protocol, dict)
        and all(protocol.get(key) == value for key, value in expected_protocol.items()),
        "reset-architecture snapshot protocol differs from the sealed strict run",
        failures,
    )

    arrays = payload.get("arrays")
    required_arrays = {
        "seeds",
        "delays",
        "stm_local",
        "stm_collective",
        "narma_local",
        "narma_collective",
        "lag_local",
        "lag_collective",
    }
    if not isinstance(arrays, dict) or not required_arrays.issubset(arrays):
        failures.append("reset-architecture snapshot lacks required arrays")
        return
    check(
        arrays["seeds"] == RESET_EXPECTED_SEEDS,
        "reset-architecture snapshot strict seed order differs from the audit",
        failures,
    )
    check(
        arrays["delays"] == list(range(1, 21)),
        "reset-architecture snapshot must contain STM delays 1 through 20",
        failures,
    )

    one_dimensional = (
        "stm_local",
        "stm_collective",
        "narma_local",
        "narma_collective",
    )
    two_dimensional = ("lag_local", "lag_collective")
    shapes_ok = all(
        isinstance(arrays[name], list) and len(arrays[name]) == 16
        for name in one_dimensional
    ) and all(
        isinstance(arrays[name], list)
        and len(arrays[name]) == 16
        and all(isinstance(row, list) and len(row) == 20 for row in arrays[name])
        for name in two_dimensional
    )
    check(
        shapes_ok,
        "reset-architecture snapshot array shapes are not 16 paired seeds by 20 lags",
        failures,
    )
    if not shapes_ok:
        return
    numeric_values = [
        value
        for name in one_dimensional
        for value in arrays[name]
    ] + [
        value
        for name in two_dimensional
        for row in arrays[name]
        for value in row
    ]
    finite = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in numeric_values
    )
    check(
        finite,
        "reset-architecture snapshot contains non-finite or non-numeric values",
        failures,
    )
    if not finite:
        return

    for method in ("local", "collective"):
        totals = arrays[f"stm_{method}"]
        lag_rows = arrays[f"lag_{method}"]
        check(
            all(
                abs(total - sum(row)) <= 1e-12
                for total, row in zip(totals, lag_rows, strict=True)
            ),
            f"reset-architecture STM totals do not equal {method} lag sums",
            failures,
        )
    stm_effects = [
        collective - local
        for local, collective in zip(
            arrays["stm_local"],
            arrays["stm_collective"],
            strict=True,
        )
    ]
    narma_effects = [
        local - collective
        for local, collective in zip(
            arrays["narma_local"],
            arrays["narma_collective"],
            strict=True,
        )
    ]
    check(
        all(effect > 0 for effect in stm_effects)
        and all(effect > 0 for effect in narma_effects),
        "reset-architecture strict result does not retain 16/16 favorable pairs",
        failures,
    )
    check(
        abs(sum(stm_effects) / 16 - 1.7734822340766943) <= 5e-7,
        "reset-architecture STM effect differs from the audited result",
        failures,
    )
    check(
        abs(sum(narma_effects) / 16 - 0.18204031742303253) <= 5e-7,
        "reset-architecture NARMA effect differs from the audited result",
        failures,
    )


def validate_phase_direction_snapshot(path: Path, failures: list[str]) -> None:
    """Validate the final N=5 numerical-replay binding used by Fig. 9."""
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        evidence = {
            relative: json.loads(
                (PHASE_EVIDENCE / relative).read_text(encoding="utf-8")
            )
            for relative in PHASE_EVIDENCE_FILE_SHA256
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        failures.append(f"cannot read phase-direction replay evidence: {error}")
        return

    for relative, expected in PHASE_EVIDENCE_FILE_SHA256.items():
        evidence_path = PHASE_EVIDENCE / relative
        check(
            evidence_path.is_file() and sha256_file(evidence_path) == expected,
            f"phase-direction replay evidence hash mismatch: {relative}",
            failures,
        )
    check(
        PHASE_RESULT_ARCHIVE.is_file()
        and sha256_file(PHASE_RESULT_ARCHIVE) == PHASE_RESULT_ARCHIVE_SHA256,
        "phase-direction replay result archive is missing or has the wrong hash",
        failures,
    )

    semantic_hashes = {
        "protocol_sha256": (
            "protocol.json",
            "protocol_sha256",
            "7e47d63012e6ed859214b1440d984dcdf183b2b6e2c4442ad1144da9b487cb53",
        ),
        "aggregate_sha256": (
            "aggregate.json",
            "aggregate_sha256",
            "e185b9e88ef675e037963e916f70c19941b3d737e8ba2780e6bad4afddca68e6",
        ),
        "convergence_summary_sha256": (
            "convergence_summary.json",
            "summary_sha256",
            "61656790f36dccce4bbb115089a7ca62b3c184624d8db52576cc6adf28bf71e2",
        ),
        "validation_amendment_sha256": (
            "validation_amendment.json",
            "amendment_sha256",
            "dd544a57c89967fe5bf0858eefe1c66e28234695cfb5289c10cbd0d6a9ec6ea3",
        ),
        "numerical_replay_audit_sha256": (
            "numerical_replay_audit.json",
            "replay_audit_sha256",
            "271e487d9b7585c5ed5edf9ce252c1879027517f719f4ae032720adc80720659",
        ),
        "validation_report_sha256": (
            "validation_report.json",
            "validation_report_sha256",
            "38e4c076b407ace1959f8b8a7bee12348cab2aafa93f03b0c16348d35842f15d",
        ),
    }
    for snapshot_key, (relative, hash_key, expected) in semantic_hashes.items():
        payload = dict(evidence[relative])
        stored = payload.pop(hash_key, None)
        check(
            snapshot.get(snapshot_key) == stored == expected
            and canonical_json_sha256(payload) == expected,
            f"phase-direction replay semantic hash mismatch: {relative}",
            failures,
        )
        check(
            snapshot.get(f"{snapshot_key.removesuffix('_sha256')}_file_sha256")
            == PHASE_EVIDENCE_FILE_SHA256[relative],
            f"phase-direction replay file binding mismatch: {relative}",
            failures,
        )
    check(
        snapshot.get("full_record_sha256") == PHASE_RESULT_ARCHIVE_SHA256,
        "phase-direction snapshot full-record binding mismatch",
        failures,
    )

    protocol = evidence["protocol.json"]
    aggregate = evidence["aggregate.json"]
    convergence = evidence["convergence_summary.json"]
    amendment = evidence["validation_amendment.json"]
    replay = evidence["numerical_replay_audit.json"]
    report = evidence["validation_report.json"]
    check(
        snapshot.get("schema_version") == 1
        and snapshot.get("protocol_version")
        == "phase-direction-confirmatory-v1-2026-08-12"
        and snapshot.get("n_seeds") == 32
        and snapshot.get("n_conditions") == 9
        and protocol.get("n_seeds") == 32
        and aggregate.get("n_task_checkpoints") == 288,
        "phase-direction snapshot identity mismatch",
        failures,
    )
    check(
        convergence.get("all_gates_passed") is True
        and convergence.get("n_complete") == 72
        and amendment.get("scientific_protocol_changed") is False
        and amendment.get("seeds_conditions_inference_or_scores_changed") is False
        and replay.get("all_gates_passed") is True
        and replay.get("n_complete") == replay.get("n_expected") == 72
        and report.get("status")
        == "validated_confirmatory_result_with_numerical_replay_amendment"
        and report.get("all_convergence_gates_pass") is True
        and report.get("all_numerical_replay_gates_pass") is True
        and report.get("n_numerical_replays") == 72
        and report.get("primary") == aggregate.get("confirmatory_primary")
        and report.get("gated_zero_overlap_generality")
        == aggregate.get("gated_zero_overlap_generality"),
        "phase-direction replay amendment contract mismatch",
        failures,
    )
    replay_snapshot = snapshot.get("convergence_and_replay", {})
    expected_replay = {
        "n_convergence_checkpoints": 72,
        "n_numerical_replays": 72,
        "all_gates_passed": True,
        "worst_trace_distance_at_800": replay.get(
            "maximum_replayed_trace_distance_at_800"
        ),
        "worst_pauli_max_abs_at_800": replay.get(
            "maximum_replayed_pauli_max_abs_at_800"
        ),
        "worst_initial_state_stm_range": replay.get(
            "maximum_replayed_four_rhs_stm_range"
        ),
        "maximum_single_vs_batch_feature_difference": replay.get(
            "maximum_feature_difference"
        ),
        "maximum_single_vs_batch_stm_difference": replay.get(
            "maximum_stm_difference"
        ),
    }
    check(
        replay_snapshot == expected_replay,
        "phase-direction compact replay metrics mismatch",
        failures,
    )


def validate_rank_one_orientation_snapshot(
    path: Path,
    failures: list[str],
) -> None:
    """Validate Fig. 9's compact N=6 record against its hardened evidence."""
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        evidence = {
            relative: json.loads(
                (ORIENTATION_EVIDENCE / relative).read_text(encoding="utf-8")
            )
            for relative in ORIENTATION_EVIDENCE_SHA256
        }
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        failures.append(f"cannot read rank-one orientation evidence: {error}")
        return

    check(
        snapshot.get("schema_version") == 1
        and snapshot.get("experiment_version")
        == "rank-one-orientation-v1-2026-08-12"
        and snapshot.get("n_qubits") == 6
        and snapshot.get("pair_count") == 24
        and snapshot.get("washout") == 800,
        "rank-one orientation snapshot identity mismatch",
        failures,
    )
    for relative, expected in ORIENTATION_EVIDENCE_SHA256.items():
        evidence_path = ORIENTATION_EVIDENCE / relative
        check(
            evidence_path.is_file() and sha256_file(evidence_path) == expected,
            f"rank-one orientation evidence hash mismatch: {relative}",
            failures,
        )


    hardened = snapshot.get("hardened_evidence", {})
    expected_hardened = {
        "root": "paper/evidence/rank_one_orientation_v1",
        "protocol_file_sha256": ORIENTATION_EVIDENCE_SHA256["protocol.json"],
        "protocol_semantic_sha256": (
            "86cfb94729d3ad039e41312f18fd25f001c5c19ef9b40c01458076f3a8ac35c5"
        ),
        "summary_file_sha256": ORIENTATION_EVIDENCE_SHA256["derived/summary.json"],
        "validation_report_file_sha256": ORIENTATION_EVIDENCE_SHA256[
            "validation_report.json"
        ],
        "provenance_file_sha256": ORIENTATION_EVIDENCE_SHA256["provenance.json"],
        "provenance_semantic_sha256": (
            "ca7e32117deb661e8721749593e81660bac6009e068cc8b4fdfb4a8154397be3"
        ),
        "environment_file_sha256": ORIENTATION_EVIDENCE_SHA256["environment.json"],
        "environment_semantic_sha256": (
            "c19a99a02c6d79666e87f511f7d4dd50afdd335289e44a160d4e8a25684a92c1"
        ),
        "raw_checkpoint_set_sha256": (
            "8500af21e29f37e51db1b6eed652894a4b818d6f2e5f6c223a03ce7dedfffdf3"
        ),
        "result_archive": {
            "filename": "rank_one_orientation_v1_results.tar.gz",
            "sha256": ORIENTATION_RESULT_ARCHIVE_SHA256,
            "validation_command": (
                "python3 rank_one_orientation_v1/validate.py "
                "--root rank_one_orientation_v1"
            ),
        },
    }
    check(
        hardened == expected_hardened,
        "rank-one orientation snapshot lacks the sealed hardened-evidence binding",
        failures,
    )

    protocol = evidence["protocol.json"]
    summary = evidence["derived/summary.json"]
    report = evidence["validation_report.json"]
    check(
        protocol.get("experiment_version")
        == "rank-one-orientation-v1-2026-08-12"
        and protocol.get("coherent_processor", {}).get("n_qubits") == 6
        and protocol.get("conditions", {}).get("equal_phase", {}).get(
            "coefficient_vector"
        )
        == [1, 1, 1, 1, 1, 1]
        and protocol.get("conditions", {}).get("drive_orthogonal", {}).get(
            "coefficient_vector"
        )
        == [1, 1, 1, -1, -1, -1]
        and protocol.get("dissipator", {}).get("matched_invariants", {}).get(
            "physical_jump_operator_kernel_dimension"
        )
        == 20,
        "rank-one orientation protocol contract mismatch",
        failures,
    )
    check(
        report.get("status") == "validated"
        and report.get("pair_count") == 24
        and report.get("ground_mixed_audit_pair_count") == 24
        and report.get("additional_four_state_audit_pair_count") == 6
        and report.get("all_convergence_gates_passed") is True
        and report.get("protocol_sha256")
        == expected_hardened["protocol_semantic_sha256"]
        and report.get("provenance_sha256")
        == expected_hardened["provenance_semantic_sha256"]
        and report.get("environment_sha256")
        == expected_hardened["environment_semantic_sha256"]
        and report.get("raw_checkpoint_set_sha256")
        == expected_hardened["raw_checkpoint_set_sha256"],
        "rank-one orientation validation report mismatch",
        failures,
    )

    stm = snapshot.get("stm", {})
    effects = stm.get("paired_values", [])
    finite_effects = (
        isinstance(effects, list)
        and len(effects) == 24
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in effects
        )
    )
    check(finite_effects, "rank-one orientation paired effects are invalid", failures)
    if not finite_effects:
        return
    effect_mean = sum(effects) / len(effects)
    summary_stm = summary.get("stm", {})
    summary_paired = summary_stm.get("paired_equal_minus_orthogonal", {})
    report_paired = report.get("primary_equal_minus_orthogonal", {})
    check(
        all(value > 0 for value in effects)
        and stm.get("wins") == 24
        and stm.get("losses") == 0
        and stm.get("ties") == 0
        and abs(effect_mean - 2.271492573239316) <= 5e-12
        and abs(effect_mean - stm.get("paired_mean", math.nan)) <= 5e-12
        and abs(effect_mean - summary_paired.get("mean", math.nan)) <= 5e-12
        and abs(effect_mean - report_paired.get("mean", math.nan)) <= 5e-12
        and summary_paired.get("wins_positive") == 24
        and report_paired.get("wins_positive") == 24,
        "rank-one orientation primary paired result mismatch",
        failures,
    )
    snapshot_ci = stm.get("paired_ci95_student_t", [])
    report_ci = report_paired.get("ci95", [])
    check(
        snapshot_ci == summary_paired.get("ci95")
        and len(snapshot_ci) == len(report_ci) == 2
        and all(
            abs(left - right) <= 5e-12
            for left, right in zip(snapshot_ci, report_ci)
        )
        and abs(
            stm.get("equal_phase_mean", math.nan)
            - summary_stm.get("equal_phase", {}).get("mean", math.nan)
        )
        <= 5e-12
        and abs(
            stm.get("drive_orthogonal_mean", math.nan)
            - summary_stm.get("drive_orthogonal", {}).get("mean", math.nan)
        )
        <= 5e-12,
        "rank-one orientation condition means or interval mismatch",
        failures,
    )

    lag_rows = summary_stm.get("lag_resolved", [])
    lag_snapshot = stm.get("lag_resolved", {})
    lag_shape_ok = (
        isinstance(lag_rows, list)
        and len(lag_rows) == 20
        and [row.get("delay") for row in lag_rows] == list(range(1, 21))
    )
    check(lag_shape_ok, "rank-one orientation lag record is incomplete", failures)
    if lag_shape_ok:
        check(
            lag_snapshot.get("all_20_mean_differences_positive") is True
            and lag_snapshot.get("all_20_ci95_lower_bounds_positive") is True
            and all(row.get("paired_difference_mean", 0.0) > 0 for row in lag_rows)
            and all(row.get("paired_difference_ci95_low", 0.0) > 0 for row in lag_rows),
            "rank-one orientation all-delay claim mismatch",
            failures,
        )
        lag20 = lag_rows[-1]
        recorded_lag20 = lag_snapshot.get("delay_20", {})
        check(
            recorded_lag20.get("paired_wins") == lag20.get("paired_wins") == 24
            and abs(
                recorded_lag20.get("drive_orthogonal_mean", math.nan)
                - lag20.get("drive_orthogonal_mean", math.nan)
            )
            <= 5e-12
            and abs(
                recorded_lag20.get("equal_phase_mean", math.nan)
                - lag20.get("equal_phase_mean", math.nan)
            )
            <= 5e-12,
            "rank-one orientation delay-20 result mismatch",
            failures,
        )

    convergence = snapshot.get("convergence", {})
    summary_validation = summary.get("validation", {})
    check(
        convergence.get("all_pairs_passed_common_washout") is True
        and convergence.get("ground_mixed_audit_pair_count") == 24
        and convergence.get("additional_four_state_audit_pair_count") == 6
        and convergence.get("selected_common_washout") == 800
        and summary_validation.get("all_selected_washouts") == [800]
        and convergence.get("worst_trace_distance")
        == summary_validation.get("worst_trace_distance")
        and convergence.get("worst_feature_distance")
        == summary_validation.get("worst_feature_distance"),
        "rank-one orientation convergence claim mismatch",
        failures,
    )

    response = snapshot.get("response_diagnostics", {})
    response_contract = (
        (
            "feature_space_effective_rank_equal_minus_orthogonal",
            "feature_space_effective_rank",
            "losses_negative",
        ),
        (
            "leading_singular_energy_fraction_equal_minus_orthogonal",
            "leading_singular_energy_fraction",
            "wins_positive",
        ),
        (
            "long_lag_energy_fraction_equal_minus_orthogonal",
            "long_lag_energy_fraction",
            "losses_negative",
        ),
        (
            "response_lag_centroid_equal_minus_orthogonal",
            "response_lag_centroid",
            "losses_negative",
        ),
    )
    for snapshot_name, summary_name, sign_key in response_contract:
        recorded = response.get(snapshot_name, {})
        source = summary.get("kernel", {}).get(summary_name, {}).get(
            "paired_equal_minus_orthogonal", {}
        )
        check(
            abs(recorded.get("mean", math.nan) - source.get("mean", math.nan))
            <= 5e-12
            and recorded.get("ci95_student_t") == source.get("ci95")
            and recorded.get("same_sign_pairs") == source.get(sign_key) == 24,
            f"rank-one orientation response claim mismatch: {summary_name}",
            failures,
        )

    association = response.get("association_with_stm_change", {})
    association_contract = (
        (
            "feature_space_effective_rank",
            "stm_change_vs_feature_space_effective_rank_change",
        ),
        (
            "leading_singular_energy_fraction",
            "stm_change_vs_leading_singular_energy_fraction_change",
        ),
        ("long_lag_energy_fraction", "stm_change_vs_long_lag_energy_fraction_change"),
        ("response_lag_centroid", "stm_change_vs_response_lag_centroid_change"),
    )
    check(
        association.get("detectable_in_this_sample") is False,
        "rank-one orientation diagnostics are overstated as a detected mediator",
        failures,
    )
    for snapshot_stem, summary_name in association_contract:
        source = summary.get("association", {}).get(summary_name, {})
        check(
            source.get("spearman_p", 0.0) > 0.05
            and abs(
                association.get(f"{snapshot_stem}_spearman_rho", math.nan)
                - source.get("spearman_rho", math.nan)
            )
            <= 5e-12
            and abs(
                association.get(f"{snapshot_stem}_spearman_p", math.nan)
                - source.get("spearman_p", math.nan)
            )
            <= 5e-12,
            f"rank-one orientation diagnostic association mismatch: {summary_name}",
            failures,
        )


def validate_continuous_narma_washout(failures: list[str]) -> None:
    """Validate the compact all-pair continuous-drive W800 NARMA evidence."""
    try:
        protocol = json.loads(
            (CONTINUOUS_NARMA_EVIDENCE / "protocol.json").read_text(
                encoding="utf-8"
            )
        )
        aggregate = json.loads(
            (CONTINUOUS_NARMA_EVIDENCE / "aggregate.json").read_text(
                encoding="utf-8"
            )
        )
        report = json.loads(
            (CONTINUOUS_NARMA_EVIDENCE / "validation_report.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        failures.append(f"cannot read continuous-drive NARMA evidence: {error}")
        return

    for relative, expected in CONTINUOUS_NARMA_EVIDENCE_SHA256.items():
        path = CONTINUOUS_NARMA_EVIDENCE / relative
        check(
            path.is_file() and sha256_file(path) == expected,
            f"continuous-drive NARMA evidence hash mismatch: {relative}",
            failures,
        )
    check(
        CONTINUOUS_NARMA_RESULT_ARCHIVE.is_file()
        and sha256_file(CONTINUOUS_NARMA_RESULT_ARCHIVE)
        == CONTINUOUS_NARMA_RESULT_ARCHIVE_SHA256,
        "continuous-drive NARMA result archive is missing or has the wrong hash",
        failures,
    )

    protocol_unhashed = dict(protocol)
    protocol_sha = protocol_unhashed.pop("protocol_sha256", "")
    aggregate_unhashed = dict(aggregate)
    aggregate_sha = aggregate_unhashed.pop("aggregate_sha256", "")
    check(
        protocol_sha
        == "1db11880bf45bec889299475eb09e8127f8dfbc986395c7f28e70ecb38059864"
        and protocol_sha == canonical_json_sha256(protocol_unhashed),
        "continuous-drive NARMA protocol binding mismatch",
        failures,
    )
    check(
        aggregate_sha
        == "0960a249ce351fe623168fa04f914370d0e26f6f16caa166a8a24deb51106ce5"
        and aggregate_sha == canonical_json_sha256(aggregate_unhashed)
        and aggregate.get("protocol_sha256") == protocol_sha
        and report.get("protocol_sha256") == protocol_sha
        and report.get("aggregate_sha256") == aggregate_sha,
        "continuous-drive NARMA aggregate binding mismatch",
        failures,
    )

    settings = protocol.get("protocol", {})
    check(
        protocol.get("version") == 2
        and len(protocol.get("ordered_seeds", [])) == 32
        and len(set(protocol.get("ordered_seeds", []))) == 32
        and settings.get("primary_washout") == 200
        and settings.get("strict_washout") == 800
        and settings.get("strict_prefix_len") == 600
        and settings.get("train_len") == 600
        and settings.get("test_len") == 400
        and settings.get("initial_state_audit_pairs") == 8
        and settings.get("ridge") == 1e-8,
        "continuous-drive NARMA protocol settings changed",
        failures,
    )
    check(
        aggregate.get("n_pairs") == 32
        and len(aggregate.get("checkpoint_sha256", {})) == 32
        and aggregate.get("baseline_replay_maximum_absolute_error", math.inf)
        <= 1e-12
        and report.get("checkpoint_count") == 32
        and report.get("status") == "pass",
        "continuous-drive NARMA replay or checkpoint contract failed",
        failures,
    )

    scores = aggregate.get("absolute_scores", {}).get("800", {})
    local = scores.get("local", {}).get("ground", {})
    collective = scores.get("collective", {}).get("ground", {})
    primary = aggregate.get("favorable_effects", {}).get("800", {}).get(
        "ground", {}
    )
    change = aggregate.get("ground_effect_change_w800_minus_w200", {})
    audit = aggregate.get("initial_state_audit", {}).get("800", {}).get(
        "collective", {}
    )
    check(
        abs(local.get("mean", math.nan) - 0.3145969497300153) <= 5e-12
        and abs(collective.get("mean", math.nan) - 0.2295183007889942)
        <= 5e-12
        and primary.get("n") == 32
        and primary.get("wins") == 32
        and abs(primary.get("mean_difference", math.nan) - 0.08507864894102105)
        <= 5e-12
        and primary.get("ci95")
        == [0.07159850810793915, 0.09855878977410294],
        "continuous-drive NARMA strict-washout endpoint mismatch",
        failures,
    )
    check(
        abs(change.get("mean_difference", math.nan) + 0.00006989560926137957)
        <= 5e-12
        and change.get("ci95")
        == [-0.0005676275953869268, 0.0004278363768641677]
        and audit.get("maximum_cross_initialization_score_spread", math.inf)
        <= 2.181543589654944e-5 + 5e-12
        and audit.get("maximum_trace_distance_at_washout", math.inf) < 1e-6,
        "continuous-drive NARMA stability or initialization audit mismatch",
        failures,
    )


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    for tool in ("pdfinfo", "pdffonts", "pdftotext"):
        check(shutil.which(tool) is not None, f"missing required tool: {tool}", failures)

    check(TEX.is_file(), f"missing manuscript source: {TEX}", failures)
    check(PDF.is_file(), f"missing manuscript PDF: {PDF}", failures)
    check(LOG.is_file(), f"missing LaTeX log: {LOG}", failures)
    check(
        RESET_BUILDER.is_file(),
        f"missing reset-architecture figure builder: {RESET_BUILDER}",
        failures,
    )
    check(
        RESET_SNAPSHOT.is_file(),
        f"missing reset-architecture snapshot: {RESET_SNAPSHOT}",
        failures,
    )
    check(
        RESET_DRIVER.is_file(),
        f"missing reset-architecture strict driver: {RESET_DRIVER}",
        failures,
    )
    check(
        PHASE_SNAPSHOT.is_file(),
        f"missing phase-direction snapshot: {PHASE_SNAPSHOT}",
        failures,
    )
    check(
        PHASE_BUILDER.is_file(),
        f"missing phase-direction figure builder: {PHASE_BUILDER}",
        failures,
    )
    check(
        PHASE_DRIVER.is_file(),
        f"missing phase-direction confirmatory driver: {PHASE_DRIVER}",
        failures,
    )
    check(
        PHASE_VALIDATOR.is_file(),
        f"missing phase-direction validator: {PHASE_VALIDATOR}",
        failures,
    )
    check(
        PHASE_EVIDENCE.is_dir(),
        f"missing phase-direction replay evidence: {PHASE_EVIDENCE}",
        failures,
    )
    check(
        PHASE_RESULT_ARCHIVE.is_file(),
        f"missing phase-direction result archive: {PHASE_RESULT_ARCHIVE}",
        failures,
    )
    check(
        ORIENTATION_SNAPSHOT.is_file(),
        f"missing rank-one orientation snapshot: {ORIENTATION_SNAPSHOT}",
        failures,
    )
    check(
        ORIENTATION_EVIDENCE.is_dir(),
        f"missing rank-one orientation evidence: {ORIENTATION_EVIDENCE}",
        failures,
    )
    check(
        ORIENTATION_RESULT_ARCHIVE.is_file(),
        f"missing rank-one orientation result archive: {ORIENTATION_RESULT_ARCHIVE}",
        failures,
    )
    check(
        CONTINUOUS_NARMA_EVIDENCE.is_dir(),
        f"missing continuous-drive NARMA evidence: {CONTINUOUS_NARMA_EVIDENCE}",
        failures,
    )
    check(
        CONTINUOUS_NARMA_RESULT_ARCHIVE.is_file(),
        (
            "missing continuous-drive NARMA result archive: "
            f"{CONTINUOUS_NARMA_RESULT_ARCHIVE}"
        ),
        failures,
    )
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        return 1

    validate_reset_snapshot(RESET_SNAPSHOT, failures)
    validate_phase_direction_snapshot(PHASE_SNAPSHOT, failures)
    validate_rank_one_orientation_snapshot(ORIENTATION_SNAPSHOT, failures)
    validate_continuous_narma_washout(failures)
    tex = TEX.read_text(encoding="utf-8")
    manuscript_source = read_tex_tree(TEX)
    log = LOG.read_text(encoding="utf-8", errors="replace")
    pdf_text = run("pdftotext", "-layout", str(PDF), "-")
    pdfinfo = run("pdfinfo", str(PDF))

    page_match = re.search(r"^Pages:\s+(\d+)", pdfinfo, re.MULTILINE)
    pages = int(page_match.group(1)) if page_match else 0
    check(8 <= pages <= 30, f"unexpected manuscript length: {pages} pages", failures)

    hard_log_patterns = {
        "undefined references": r"(undefined references|Reference .* undefined)",
        "undefined citations": r"Citation .* undefined",
        "overfull boxes": r"Overfull \\[hv]box",
        "stuck floats": r"A float is stuck",
        "multiply-defined labels": r"multiply defined",
        "fatal LaTeX errors": r"(^! |Emergency stop|Fatal error)",
    }
    for label, pattern in hard_log_patterns.items():
        check(
            re.search(pattern, log, flags=re.MULTILINE) is None,
            f"LaTeX log contains {label}",
            failures,
        )

    forbidden_source_patterns = {
        "submission TODO": r"TODO before submission",
        "placeholder authorship": r"placeholder",
        "future DOI promise": r"DOI-stamped snapshot will accompany",
        "unstable date command": r"\\date\{\\today\}",
    }
    for label, pattern in forbidden_source_patterns.items():
        check(
            re.search(pattern, manuscript_source, flags=re.IGNORECASE) is None,
            f"source contains {label}",
            failures,
        )

    for token in ("TODO", "PLACEHOLDER", "??"):
        check(token not in pdf_text, f"PDF contains unresolved token {token!r}", failures)

    figure_paths = sorted(
        set(
            re.findall(
                r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
                manuscript_source,
            )
        )
    )
    check(bool(figure_paths), "manuscript includes no figures", failures)
    for figure in figure_paths:
        path = PAPER / figure
        check(path.is_file(), f"missing included figure: {figure}", failures)

    graphics_commands = re.findall(
        r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}",
        manuscript_source,
    )
    for figure, contract in CODE_RENDERED_COMPOSITES.items():
        matching_options = [
            options for options, included_path in graphics_commands if included_path == figure
        ]
        check(
            matching_options == [""],
            (
                f"{figure} must be inserted exactly once at its code-rendered "
                "natural size, without LaTeX scaling or cropping"
            ),
            failures,
        )
        figure_path = PAPER / figure
        if not figure_path.is_file():
            continue
        figure_info = run("pdfinfo", str(figure_path))
        size_match = re.search(
            r"^Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)\s+pts",
            figure_info,
            re.MULTILINE,
        )
        check(size_match is not None, f"could not read page size for {figure}", failures)
        if size_match:
            actual_size = tuple(float(value) for value in size_match.groups())
            expected_size = contract["size"]
            check(
                all(
                    abs(actual - expected) <= 0.05
                    for actual, expected in zip(actual_size, expected_size)
                ),
                (
                    f"{figure} has page size {actual_size}, expected "
                    f"{expected_size}; panel geometry must remain code-owned"
                ),
                failures,
            )
        creator_match = re.search(r"^Creator:\s+(.+)$", figure_info, re.MULTILINE)
        creator = creator_match.group(1).strip() if creator_match else ""
        check(
            creator == contract["creator"],
            (
                f"{figure} creator is {creator!r}, expected "
                f"{contract['creator']!r}"
            ),
            failures,
        )
        required_text = contract.get("required_text", ())
        if required_text:
            figure_text = run("pdftotext", str(figure_path), "-")
            for fragment in required_text:
                check(
                    fragment in figure_text,
                    f"{figure} is missing required text {fragment!r}",
                    failures,
                )

    pdf_targets = [PDF] + sorted(FIGURES.glob("*.pdf"))
    for target in pdf_targets:
        font_table = run("pdffonts", str(target))
        rows = [line.split() for line in font_table.splitlines()[2:] if line.strip()]
        check(rows, f"no fonts detected in {target.relative_to(ROOT)}", failures)
        for row in rows:
            descriptor = " ".join(row)
            check("Type 3" not in descriptor, f"Type 3 font in {target.relative_to(ROOT)}", failures)
            if len(row) >= 6:
                check(
                    row[4].lower() == "yes",
                    f"unembedded font in {target.relative_to(ROOT)}: {row[0]}",
                    failures,
                )

    required_sections = (
        "abstract.tex",
        "introduction.tex",
        "background.tex",
        "related-work.tex",
        "methodology.tex",
        "experimental-section.tex",
        "evaluation.tex",
        "conclusion.tex",
    )
    section_dir = PAPER / "sections"
    for name in required_sections:
        check((section_dir / name).is_file(), f"missing section source: paper/sections/{name}", failures)

    if re.search(
        r"(?:Data\s+and\s+code|source\s+code,\s+numerical\s+data)",
        pdf_text,
        flags=re.IGNORECASE,
    ) is None:
        failures.append("PDF lacks a data-and-code statement")
    if len(re.findall(r"\\bibitem", tex)) < 15 and "\\bibliography{" not in tex:
        warnings.append("bibliography contains fewer than 15 manual entries")
    if re.search(r"Underfull \\[hv]box \(badness 10000\)", log):
        warnings.append("LaTeX log contains maximally underfull boxes")

    for item in warnings:
        print(f"WARN: {item}")
    for item in failures:
        print(f"FAIL: {item}")

    if failures:
        print(f"\nSubmission validation failed: {len(failures)} failure(s).")
        return 1

    print(
        f"Submission validation passed: {pages} pages, "
        f"{len(figure_paths)} included figures, no Type 3 fonts or hard LaTeX defects."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

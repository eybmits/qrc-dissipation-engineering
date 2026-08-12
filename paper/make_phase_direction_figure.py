"""Build the publication direction-intervention figure from frozen evidence.

The experiment driver remains frozen.  This publication-only renderer reads
the complete N=5 aggregate and compact manuscript snapshots for both finite-
reservoir interventions, checks their source bindings and reported statistics,
and then redraws the two panels in the manuscript's shared visual grammar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import l3_style as st


HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / "data" / "phase_direction_confirmatory_snapshot.json"
ORIENTATION_SNAPSHOT = HERE / "data" / "rank_one_orientation_snapshot.json"
ORIENTATION_EVIDENCE = HERE / "evidence" / "rank_one_orientation_v1"
ORIENTATION_RESULT_ARCHIVE = (
    HERE.parent / "results" / "rank_one_orientation_v1_results.tar.gz"
)
PHASE_EVIDENCE = HERE / "evidence" / "phase_direction_confirmatory_v1"
AGGREGATE = PHASE_EVIDENCE / "aggregate.json"
PHASE_RESULT_ARCHIVE = (
    HERE.parent / "results" / "phase_direction_confirmatory_v1_results.tar.gz"
)
DEFAULT_OUTPUT = HERE / "figures" / "fig_phase_direction.pdf"

EXPECTED_PROTOCOL_SHA256 = (
    "7e47d63012e6ed859214b1440d984dcdf183b2b6e2c4442ad1144da9b487cb53"
)
EXPECTED_AGGREGATE_SHA256 = (
    "e185b9e88ef675e037963e916f70c19941b3d737e8ba2780e6bad4afddca68e6"
)
EXPECTED_VALIDATION_REPORT_SHA256 = (
    "38e4c076b407ace1959f8b8a7bee12348cab2aafa93f03b0c16348d35842f15d"
)
EXPECTED_PHASE_EVIDENCE_FILE_SHA256 = {
    "protocol.json": "ef12dd27cd78976e644bbaea10a03c694cf34c8f1b287a0029e30625092f52fc",
    "aggregate.json": "1112b830924c7b43dbb7c16ca4baf9cb1d4ab5205c67c2b01d5ec430aacb9634",
    "convergence_summary.json": "1402b70561f2c18a545e5636b10b2042cfc01710d8ff275bcadff408d91a9ce1",
    "validation_amendment.json": "e842c1f6299ec9d839c36825e8736eb1b053388cc9aff48d17ca0a4d0b767b38",
    "numerical_replay_audit.json": "e401c62cafe9131cb83fe8f6b0af4ef50bb9a9aa13c75d2def47b38179e1a0d1",
    "validation_report.json": "2379fa37daf108c53088172f96e20e05e96c09493f3c6ebd0f1f2ec19fab1403",
}
EXPECTED_CONVERGENCE_SUMMARY_SHA256 = (
    "61656790f36dccce4bbb115089a7ca62b3c184624d8db52576cc6adf28bf71e2"
)
EXPECTED_VALIDATION_AMENDMENT_SHA256 = (
    "dd544a57c89967fe5bf0858eefe1c66e28234695cfb5289c10cbd0d6a9ec6ea3"
)
EXPECTED_NUMERICAL_REPLAY_AUDIT_SHA256 = (
    "271e487d9b7585c5ed5edf9ce252c1879027517f719f4ae032720adc80720659"
)
EXPECTED_PHASE_RESULT_ARCHIVE_SHA256 = (
    "ec9a7e6aea148fadf59d83a0967a08cdba6436ad8e2fd8e61e81473226f0f394"
)
EXPECTED_ORIENTATION_BUNDLE_SHA256 = (
    "daa1296e21430b45d0318078b25d418681bb08c08cf421fe8de41e0924fe0165"
)
EXPECTED_ORIENTATION_MEMBER_SHA256 = {
    "summary.json": "72dd00e2dcfccdbae5b2a6bc0feede0e7ad340e08b64fea8df757b25ef0a7f9b",
    "per_seed.csv": "e765d567fd42bf813beac78e144eb3d072cac6716be9c82016554a49ed815ce2",
    "semantic_validation_manifest.json": "0bbb777ae924127a63583ed788de0b6357b32f66776a3ba886cf880833f29f39",
    "raw_checkpoints.zip": "a30b15cbe1e0be5428ba55881fcc4307484210e6e361374cf78775d8274de6b8",
}
EXPECTED_ORIENTATION_EVIDENCE_SHA256 = {
    "protocol.json": "23572d7ee36cf63e5560b310f1bea1453a0d5c275e2f578bae3d2b40bf4e68c4",
    "derived/summary.json": "72dd00e2dcfccdbae5b2a6bc0feede0e7ad340e08b64fea8df757b25ef0a7f9b",
    "validation_report.json": "4402d851b5418bc56003d7346d095548996330d303f49d1a78973693394c0a83",
    "provenance.json": "e8a33914e91fa2cdd43c50c7979a4fa7bbc7e7885206c1ffe10b90e2a7ee29bd",
    "environment.json": "e1abfd119ce581e86027b5e8ba969870d62cba2299503b4b1e5bc0e1317e2a5a",
}
EXPECTED_ORIENTATION_PROTOCOL_SEMANTIC_SHA256 = (
    "86cfb94729d3ad039e41312f18fd25f001c5c19ef9b40c01458076f3a8ac35c5"
)
EXPECTED_ORIENTATION_PROVENANCE_SEMANTIC_SHA256 = (
    "ca7e32117deb661e8721749593e81660bac6009e068cc8b4fdfb4a8154397be3"
)
EXPECTED_ORIENTATION_ENVIRONMENT_SEMANTIC_SHA256 = (
    "c19a99a02c6d79666e87f511f7d4dd50afdd335289e44a160d4e8a25684a92c1"
)
EXPECTED_ORIENTATION_RAW_SET_SHA256 = (
    "8500af21e29f37e51db1b6eed652894a4b818d6f2e5f6c223a03ce7dedfffdf3"
)
EXPECTED_ORIENTATION_RESULT_ARCHIVE_SHA256 = (
    "e12f0bdd038b8e45ecea247b09d32815de20330e51ab47aecfe1d995fd86f24a"
)
PATH_CONDITIONS = (
    "path_f0",
    "path_f025",
    "path_f05",
    "path_f075",
    "path_f1",
)
PATH_FRACTIONS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
ZERO_OVERLAP_CONDITIONS = (
    "path_f1",
    "scrambled_r1",
    "scrambled_r2",
    "scrambled_r3",
    "scrambled_r4",
)
N_SEEDS = 32
N_ORIENTATION_PAIRS = 24
STAT_TOLERANCE = 5e-12

st.use(times=True)


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_close(observed: float, expected: float, label: str) -> None:
    if not np.isclose(observed, expected, rtol=0, atol=STAT_TOLERANCE):
        raise RuntimeError(
            f"phase-direction evidence mismatch for {label}: "
            f"{observed:.16g} != {expected:.16g}"
        )


def _paired_summary(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    se = float(stats.sem(values))
    half_width = float(stats.t.ppf(0.975, len(values) - 1) * se)
    return mean, mean - half_width, mean + half_width


def _load_phase_replay_evidence(snapshot: dict) -> dict:
    """Authenticate the final N=5 replay amendment and compact result record."""
    expected_bindings = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "protocol_file_sha256": EXPECTED_PHASE_EVIDENCE_FILE_SHA256[
            "protocol.json"
        ],
        "aggregate_sha256": EXPECTED_AGGREGATE_SHA256,
        "aggregate_file_sha256": EXPECTED_PHASE_EVIDENCE_FILE_SHA256[
            "aggregate.json"
        ],
        "convergence_summary_sha256": EXPECTED_CONVERGENCE_SUMMARY_SHA256,
        "convergence_summary_file_sha256": EXPECTED_PHASE_EVIDENCE_FILE_SHA256[
            "convergence_summary.json"
        ],
        "validation_amendment_sha256": EXPECTED_VALIDATION_AMENDMENT_SHA256,
        "validation_amendment_file_sha256": EXPECTED_PHASE_EVIDENCE_FILE_SHA256[
            "validation_amendment.json"
        ],
        "numerical_replay_audit_sha256": EXPECTED_NUMERICAL_REPLAY_AUDIT_SHA256,
        "numerical_replay_audit_file_sha256": EXPECTED_PHASE_EVIDENCE_FILE_SHA256[
            "numerical_replay_audit.json"
        ],
        "validation_report_sha256": EXPECTED_VALIDATION_REPORT_SHA256,
        "validation_report_file_sha256": EXPECTED_PHASE_EVIDENCE_FILE_SHA256[
            "validation_report.json"
        ],
        "full_record_sha256": EXPECTED_PHASE_RESULT_ARCHIVE_SHA256,
    }
    for key, expected in expected_bindings.items():
        if snapshot.get(key) != expected:
            raise RuntimeError(f"phase-direction snapshot {key} mismatch")

    payloads: dict[str, dict] = {}
    for relative, expected_sha256 in EXPECTED_PHASE_EVIDENCE_FILE_SHA256.items():
        path = PHASE_EVIDENCE / relative
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            raise RuntimeError(f"phase-direction evidence hash mismatch: {relative}")
        payloads[relative] = json.loads(path.read_text(encoding="utf-8"))
    if PHASE_RESULT_ARCHIVE.is_file() and (
        _file_sha256(PHASE_RESULT_ARCHIVE) != EXPECTED_PHASE_RESULT_ARCHIVE_SHA256
    ):
        raise RuntimeError("phase-direction result archive hash mismatch")

    self_hash_contract = (
        ("protocol.json", "protocol_sha256", EXPECTED_PROTOCOL_SHA256),
        ("aggregate.json", "aggregate_sha256", EXPECTED_AGGREGATE_SHA256),
        (
            "convergence_summary.json",
            "summary_sha256",
            EXPECTED_CONVERGENCE_SUMMARY_SHA256,
        ),
        (
            "validation_amendment.json",
            "amendment_sha256",
            EXPECTED_VALIDATION_AMENDMENT_SHA256,
        ),
        (
            "numerical_replay_audit.json",
            "replay_audit_sha256",
            EXPECTED_NUMERICAL_REPLAY_AUDIT_SHA256,
        ),
        (
            "validation_report.json",
            "validation_report_sha256",
            EXPECTED_VALIDATION_REPORT_SHA256,
        ),
    )
    for relative, hash_key, expected in self_hash_contract:
        unhashed = dict(payloads[relative])
        stored = unhashed.pop(hash_key, None)
        if stored != expected or _canonical_sha256(unhashed) != expected:
            raise RuntimeError(f"phase-direction self-hash mismatch: {relative}")

    protocol = payloads["protocol.json"]
    aggregate = payloads["aggregate.json"]
    convergence = payloads["convergence_summary.json"]
    amendment = payloads["validation_amendment.json"]
    replay = payloads["numerical_replay_audit.json"]
    report = payloads["validation_report.json"]
    if (
        protocol.get("protocol_version")
        != "phase-direction-confirmatory-v1-2026-08-12"
        or convergence.get("all_gates_passed") is not True
        or convergence.get("n_complete") != 72
        or amendment.get("scientific_protocol_changed") is not False
        or amendment.get("seeds_conditions_inference_or_scores_changed") is not False
        or replay.get("all_gates_passed") is not True
        or replay.get("n_complete") != 72
        or replay.get("n_expected") != 72
        or report.get("status")
        != "validated_confirmatory_result_with_numerical_replay_amendment"
        or report.get("all_convergence_gates_pass") is not True
        or report.get("all_numerical_replay_gates_pass") is not True
        or report.get("n_numerical_replays") != 72
        or report.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or report.get("aggregate_sha256") != EXPECTED_AGGREGATE_SHA256
        or report.get("convergence_summary_sha256")
        != EXPECTED_CONVERGENCE_SUMMARY_SHA256
        or report.get("primary") != aggregate.get("confirmatory_primary")
        or report.get("gated_zero_overlap_generality")
        != aggregate.get("gated_zero_overlap_generality")
    ):
        raise RuntimeError("phase-direction replay amendment contract mismatch")

    replay_snapshot = snapshot.get("convergence_and_replay", {})
    replay_contract = {
        "n_convergence_checkpoints": report.get("n_convergence_checkpoints"),
        "n_numerical_replays": report.get("n_numerical_replays"),
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
    if replay_snapshot != replay_contract:
        raise RuntimeError("phase-direction compact replay summary mismatch")
    return aggregate


def _load_orientation_evidence(snapshot: dict) -> tuple[dict, dict, dict]:
    """Authenticate and return the compact N=6 protocol, summary, and report."""
    hardened = snapshot.get("hardened_evidence", {})
    expected_binding = {
        "root": "paper/evidence/rank_one_orientation_v1",
        "protocol_file_sha256": EXPECTED_ORIENTATION_EVIDENCE_SHA256["protocol.json"],
        "protocol_semantic_sha256": EXPECTED_ORIENTATION_PROTOCOL_SEMANTIC_SHA256,
        "summary_file_sha256": EXPECTED_ORIENTATION_EVIDENCE_SHA256[
            "derived/summary.json"
        ],
        "validation_report_file_sha256": EXPECTED_ORIENTATION_EVIDENCE_SHA256[
            "validation_report.json"
        ],
        "provenance_file_sha256": EXPECTED_ORIENTATION_EVIDENCE_SHA256[
            "provenance.json"
        ],
        "provenance_semantic_sha256": EXPECTED_ORIENTATION_PROVENANCE_SEMANTIC_SHA256,
        "environment_file_sha256": EXPECTED_ORIENTATION_EVIDENCE_SHA256[
            "environment.json"
        ],
        "environment_semantic_sha256": EXPECTED_ORIENTATION_ENVIRONMENT_SEMANTIC_SHA256,
        "raw_checkpoint_set_sha256": EXPECTED_ORIENTATION_RAW_SET_SHA256,
        "result_archive": {
            "filename": "rank_one_orientation_v1_results.tar.gz",
            "sha256": EXPECTED_ORIENTATION_RESULT_ARCHIVE_SHA256,
            "validation_command": (
                "python3 rank_one_orientation_v1/validate.py "
                "--root rank_one_orientation_v1"
            ),
        },
    }
    if hardened != expected_binding:
        raise RuntimeError("N=6 hardened-evidence binding mismatch")

    payloads: dict[str, dict] = {}
    for relative, expected_sha256 in EXPECTED_ORIENTATION_EVIDENCE_SHA256.items():
        path = ORIENTATION_EVIDENCE / relative
        if not path.is_file() or _file_sha256(path) != expected_sha256:
            raise RuntimeError(f"N=6 evidence file hash mismatch: {relative}")
        payloads[relative] = json.loads(path.read_text(encoding="utf-8"))

    # The full result archive is intentionally external to an arXiv source
    # extraction.  When this renderer runs in the repository, authenticate it.
    if ORIENTATION_RESULT_ARCHIVE.is_file() and (
        _file_sha256(ORIENTATION_RESULT_ARCHIVE)
        != EXPECTED_ORIENTATION_RESULT_ARCHIVE_SHA256
    ):
        raise RuntimeError("N=6 full result archive hash mismatch")

    protocol = payloads["protocol.json"]
    summary = payloads["derived/summary.json"]
    report = payloads["validation_report.json"]
    if (
        protocol.get("schema_version") != 1
        or protocol.get("experiment_version")
        != "rank-one-orientation-v1-2026-08-12"
        or protocol.get("coherent_processor", {}).get("n_qubits") != 6
        or protocol.get("task", {}).get("delays") != list(range(1, 21))
        or protocol.get("convergence", {}).get("checkpoints") != [800, 1200, 1600]
        or report.get("schema_version") != 1
        or report.get("status") != "validated"
        or report.get("protocol_sha256")
        != EXPECTED_ORIENTATION_PROTOCOL_SEMANTIC_SHA256
        or report.get("provenance_sha256")
        != EXPECTED_ORIENTATION_PROVENANCE_SEMANTIC_SHA256
        or report.get("environment_sha256")
        != EXPECTED_ORIENTATION_ENVIRONMENT_SEMANTIC_SHA256
        or report.get("raw_checkpoint_set_sha256")
        != EXPECTED_ORIENTATION_RAW_SET_SHA256
        or report.get("pair_count") != N_ORIENTATION_PAIRS
        or report.get("ground_mixed_audit_pair_count") != N_ORIENTATION_PAIRS
        or report.get("additional_four_state_audit_pair_count") != 6
        or report.get("all_convergence_gates_passed") is not True
        or summary.get("experiment", {}).get("pair_count")
        != N_ORIENTATION_PAIRS
        or summary.get("validation", {}).get("all_24_seed_jobs_present") is not True
        or summary.get("validation", {}).get("raw_payload_digests_verified") is not True
        or summary.get("validation", {}).get("all_convergence_gates_passed") is not True
    ):
        raise RuntimeError("N=6 hardened evidence contract mismatch")
    return protocol, summary, report


def _load_orientation_replication() -> np.ndarray:
    """Return the N=6 paired effects after checking the compact source record."""
    snapshot = json.loads(ORIENTATION_SNAPSHOT.read_text(encoding="utf-8"))
    if (
        snapshot.get("schema_version") != 1
        or snapshot.get("experiment_version")
        != "rank-one-orientation-v1-2026-08-12"
        or snapshot.get("n_qubits") != 6
        or snapshot.get("pair_count") != N_ORIENTATION_PAIRS
        or snapshot.get("washout") != 800
    ):
        raise RuntimeError("N=6 orientation snapshot identity mismatch")
    protocol, summary, report = _load_orientation_evidence(snapshot)

    source = snapshot.get("source_bundle", {})
    if (
        source.get("filename") != "rank_one_orientation_complete_bundle.zip"
        or source.get("sha256") != EXPECTED_ORIENTATION_BUNDLE_SHA256
        or source.get("members") != EXPECTED_ORIENTATION_MEMBER_SHA256
    ):
        raise RuntimeError("N=6 orientation source binding mismatch")

    expected_channels = {
        "drive_orthogonal": [1, 1, 1, -1, -1, -1],
        "equal_phase": [1, 1, 1, 1, 1, 1],
    }
    expected_invariants = {
        "operator_sector": "collective lowering",
        "kossakowski_rank": 1,
        "kossakowski_trace": 6.0,
        "kossakowski_nonzero_spectrum": [6.0],
        "kossakowski_diagonal": [1.0] * 6,
        "coefficient_magnitudes": [1.0] * 6,
        "lowering_block_kernel_dimension": 5,
        "assigned_weight_B": 192.0,
        "within_pair_fixed": [
            "Hamiltonian",
            "input stream",
            "Pauli readout",
            "training data",
            "test data",
            "ridge protocol",
        ],
    }
    if (
        snapshot.get("channels") != expected_channels
        or snapshot.get("matched_invariants") != expected_invariants
    ):
        raise RuntimeError("N=6 matched-channel invariant mismatch")
    protocol_invariants = protocol.get("dissipator", {}).get(
        "matched_invariants", {}
    )
    if (
        protocol.get("conditions", {}).get("drive_orthogonal", {}).get(
            "coefficient_vector"
        )
        != expected_channels["drive_orthogonal"]
        or protocol.get("conditions", {}).get("equal_phase", {}).get(
            "coefficient_vector"
        )
        != expected_channels["equal_phase"]
        or protocol_invariants.get("kossakowski_block_rank") != 1
        or protocol_invariants.get("kossakowski_block_trace") != 6.0
        or protocol_invariants.get("kossakowski_block_nonzero_spectrum") != [6.0]
        or protocol_invariants.get("operator_weight_budget_B") != 192.0
        or protocol_invariants.get("kossakowski_block_kernel_dimension") != 5
        or protocol_invariants.get("physical_jump_operator_kernel_dimension") != 20
    ):
        raise RuntimeError("N=6 protocol channel invariant mismatch")

    stm = snapshot.get("stm", {})
    effects = np.asarray(stm.get("paired_values", []), dtype=float)
    if effects.shape != (N_ORIENTATION_PAIRS,) or not np.all(np.isfinite(effects)):
        raise RuntimeError("invalid N=6 paired STM effects")
    effect_mean, effect_low, effect_high = _paired_summary(effects)
    summary_stm = summary.get("stm", {})
    summary_paired = summary_stm.get("paired_equal_minus_orthogonal", {})
    report_paired = report.get("primary_equal_minus_orthogonal", {})
    for label, observed, expected in (
        ("N=6 paired mean", effect_mean, stm.get("paired_mean")),
        ("N=6 paired lower CI", effect_low, stm.get("paired_ci95_student_t", [None])[0]),
        ("N=6 paired upper CI", effect_high, stm.get("paired_ci95_student_t", [None, None])[1]),
        (
            "N=6 condition-mean contrast",
            stm.get("equal_phase_mean") - stm.get("drive_orthogonal_mean"),
            effect_mean,
        ),
        (
            "N=6 relative increase",
            effect_mean / stm.get("drive_orthogonal_mean"),
            stm.get("relative_increase"),
        ),
        (
            "N=6 Cohen dz",
            effect_mean / float(np.std(effects, ddof=1)),
            stm.get("cohens_dz"),
        ),
        (
            "N=6 exact sign p",
            stats.binomtest(
                int(np.count_nonzero(effects > 0)),
                N_ORIENTATION_PAIRS,
                0.5,
                alternative="two-sided",
            ).pvalue,
            stm.get("exact_sign_test_p_two_sided"),
        ),
        ("N=6 hardened paired mean", effect_mean, summary_paired.get("mean")),
        (
            "N=6 hardened paired lower CI",
            effect_low,
            summary_paired.get("ci95", [None])[0],
        ),
        (
            "N=6 hardened paired upper CI",
            effect_high,
            summary_paired.get("ci95", [None, None])[1],
        ),
        ("N=6 validation-report mean", effect_mean, report_paired.get("mean")),
        (
            "N=6 drive-orthogonal mean",
            stm.get("drive_orthogonal_mean"),
            summary_stm.get("drive_orthogonal", {}).get("mean"),
        ),
        (
            "N=6 equal-phase mean",
            stm.get("equal_phase_mean"),
            summary_stm.get("equal_phase", {}).get("mean"),
        ),
    ):
        _assert_close(float(observed), float(expected), label)
    if (
        int(np.count_nonzero(effects > 0)) != N_ORIENTATION_PAIRS
        or stm.get("wins") != N_ORIENTATION_PAIRS
        or stm.get("losses") != 0
        or stm.get("ties") != 0
        or snapshot.get("convergence", {}).get(
            "all_pairs_passed_common_washout"
        )
        is not True
    ):
        raise RuntimeError("N=6 paired-win or convergence gate mismatch")

    lag_rows = summary_stm.get("lag_resolved", [])
    lag_snapshot = stm.get("lag_resolved", {})
    if (
        len(lag_rows) != 20
        or [row.get("delay") for row in lag_rows] != list(range(1, 21))
        or lag_snapshot.get("all_20_mean_differences_positive")
        is not all(row.get("paired_difference_mean", 0.0) > 0 for row in lag_rows)
        or lag_snapshot.get("all_20_ci95_lower_bounds_positive")
        is not all(row.get("paired_difference_ci95_low", 0.0) > 0 for row in lag_rows)
    ):
        raise RuntimeError("N=6 lag-resolved direction claim mismatch")
    lag20 = lag_rows[-1]
    lag20_snapshot = lag_snapshot.get("delay_20", {})
    for label, observed, expected in (
        (
            "N=6 lag-20 drive-orthogonal mean",
            lag20_snapshot.get("drive_orthogonal_mean"),
            lag20.get("drive_orthogonal_mean"),
        ),
        (
            "N=6 lag-20 equal-phase mean",
            lag20_snapshot.get("equal_phase_mean"),
            lag20.get("equal_phase_mean"),
        ),
        (
            "N=6 lag-20 paired mean",
            lag20_snapshot.get("paired_difference_mean"),
            lag20.get("paired_difference_mean"),
        ),
        (
            "N=6 lag-20 paired lower CI",
            lag20_snapshot.get("paired_difference_ci95_student_t", [None])[0],
            lag20.get("paired_difference_ci95_low"),
        ),
        (
            "N=6 lag-20 paired upper CI",
            lag20_snapshot.get("paired_difference_ci95_student_t", [None, None])[1],
            lag20.get("paired_difference_ci95_high"),
        ),
    ):
        _assert_close(float(observed), float(expected), label)
    if lag20_snapshot.get("paired_wins") != lag20.get("paired_wins"):
        raise RuntimeError("N=6 lag-20 paired-win mismatch")

    convergence = snapshot.get("convergence", {})
    summary_validation = summary.get("validation", {})
    if (
        convergence.get("ground_mixed_audit_pair_count")
        != report.get("ground_mixed_audit_pair_count")
        or convergence.get("additional_four_state_audit_pair_count")
        != report.get("additional_four_state_audit_pair_count")
        or convergence.get("selected_common_washout") != 800
        or summary_validation.get("all_selected_washouts") != [800]
        or convergence.get("worst_trace_distance")
        != summary_validation.get("worst_trace_distance")
        or convergence.get("worst_feature_distance")
        != summary_validation.get("worst_feature_distance")
    ):
        raise RuntimeError("N=6 convergence summary mismatch")

    response_snapshot = snapshot.get("response_diagnostics", {})
    response_contract = (
        (
            "feature_space_effective_rank",
            "feature_space_effective_rank_equal_minus_orthogonal",
            "losses_negative",
        ),
        (
            "leading_singular_energy_fraction",
            "leading_singular_energy_fraction_equal_minus_orthogonal",
            "wins_positive",
        ),
        (
            "long_lag_energy_fraction",
            "long_lag_energy_fraction_equal_minus_orthogonal",
            "losses_negative",
        ),
        (
            "response_lag_centroid",
            "response_lag_centroid_equal_minus_orthogonal",
            "losses_negative",
        ),
    )
    for summary_name, snapshot_name, sign_key in response_contract:
        source_row = summary.get("kernel", {}).get(summary_name, {}).get(
            "paired_equal_minus_orthogonal", {}
        )
        recorded = response_snapshot.get(snapshot_name, {})
        for suffix, observed, expected in (
            ("mean", recorded.get("mean"), source_row.get("mean")),
            (
                "lower CI",
                recorded.get("ci95_student_t", [None])[0],
                source_row.get("ci95", [None])[0],
            ),
            (
                "upper CI",
                recorded.get("ci95_student_t", [None, None])[1],
                source_row.get("ci95", [None, None])[1],
            ),
        ):
            _assert_close(float(observed), float(expected), f"{summary_name} {suffix}")
        if recorded.get("same_sign_pairs") != source_row.get(sign_key):
            raise RuntimeError(f"N=6 response sign count mismatch: {summary_name}")

    association = response_snapshot.get("association_with_stm_change", {})
    association_rows = summary.get("association", {})
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
    if association.get("detectable_in_this_sample") is not False:
        raise RuntimeError("N=6 diagnostic-association claim must remain non-causal")
    for snapshot_stem, summary_name in association_contract:
        source_row = association_rows.get(summary_name, {})
        for suffix in ("spearman_rho", "spearman_p"):
            _assert_close(
                float(association.get(f"{snapshot_stem}_{suffix}")),
                float(source_row.get(suffix)),
                f"{summary_name} {suffix}",
            )
        if float(source_row.get("spearman_p")) <= 0.05:
            raise RuntimeError("N=6 diagnostic association unexpectedly crosses 0.05")
    return effects


def load_and_validate() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return both intervention arrays after checking their frozen records."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    if snapshot.get("schema_version") != 1:
        raise RuntimeError("unsupported phase-direction snapshot schema")
    aggregate = _load_phase_replay_evidence(snapshot)

    unhashed = dict(aggregate)
    stored_hash = unhashed.pop("aggregate_sha256", None)
    if stored_hash != EXPECTED_AGGREGATE_SHA256:
        raise RuntimeError("phase-direction aggregate identity mismatch")
    if _canonical_sha256(unhashed) != stored_hash:
        raise RuntimeError("phase-direction aggregate self-hash mismatch")
    if (
        aggregate.get("artifact_type")
        != "phase_direction_confirmatory_aggregate"
        or aggregate.get("status") != "complete"
        or aggregate.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or aggregate.get("pilot_scores_included") is not False
        or aggregate.get("n_seeds") != N_SEEDS
        or aggregate.get("n_conditions") != 9
        or aggregate.get("n_task_checkpoints") != 9 * N_SEEDS
    ):
        raise RuntimeError("phase-direction aggregate violates its frozen contract")

    summaries = aggregate.get("condition_summaries", [])
    by_condition = {row.get("condition"): row for row in summaries}
    expected_conditions = set(PATH_CONDITIONS) | set(ZERO_OVERLAP_CONDITIONS)
    if set(by_condition) != expected_conditions or len(summaries) != 9:
        raise RuntimeError("unexpected phase-direction condition set")

    fixed_values: dict[str, np.ndarray] = {}
    snapshot_means = snapshot.get("fixed_ridge_means", {})
    structural = snapshot.get("structural_invariants", {})
    if structural != {
        "kossakowski_rank": 1,
        "kossakowski_trace": 5.0,
        "kossakowski_spectrum": [5.0, 0.0, 0.0, 0.0, 0.0],
        "kossakowski_diagonal": [1.0, 1.0, 1.0, 1.0, 1.0],
        "coefficient_magnitudes": [1.0, 1.0, 1.0, 1.0, 1.0],
        "magnitude_ipr": 0.2,
        "assigned_weight_B": 80.0,
    }:
        raise RuntimeError("phase-direction snapshot structural contract mismatch")
    for condition, row in by_condition.items():
        values = np.asarray(row.get("fixed_ridge_values", []), dtype=float)
        if values.shape != (N_SEEDS,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid fixed-ridge values for {condition}")
        fixed_values[condition] = values
        _assert_close(float(np.mean(values)), row["fixed_ridge_mean"], f"{condition} mean")
        _assert_close(float(stats.sem(values)), row["fixed_ridge_se"], f"{condition} SE")
        _assert_close(
            float(np.mean(values)),
            snapshot_means[condition],
            f"{condition} snapshot mean",
        )
        eigenvalues = np.asarray(row.get("kossakowski_eigenvalues", []), dtype=float)
        if (
            row.get("kossakowski_rank") != structural.get("kossakowski_rank")
            or not np.isclose(
                row.get("kossakowski_trace"),
                structural.get("kossakowski_trace"),
                rtol=0,
                atol=1e-12,
            )
            or not np.allclose(
                row.get("kossakowski_diagonal", []),
                structural.get("kossakowski_diagonal"),
                rtol=0,
                atol=1e-12,
            )
            or not np.allclose(
                row.get("coefficient_magnitudes", []),
                structural.get("coefficient_magnitudes"),
                rtol=0,
                atol=1e-12,
            )
            or not np.isclose(
                row.get("magnitude_ipr"),
                structural.get("magnitude_ipr"),
                rtol=0,
                atol=1e-12,
            )
            or eigenvalues.shape != (5,)
            or not np.allclose(
                np.sort(eigenvalues),
                [0.0, 0.0, 0.0, 0.0, 5.0],
                rtol=0,
                atol=1e-12,
            )
        ):
            raise RuntimeError(f"matched-channel invariant mismatch for {condition}")

    equal = fixed_values["path_f0"]
    primary_values = equal - fixed_values["path_f1"]
    primary_mean, primary_low, primary_high = _paired_summary(primary_values)
    aggregate_primary = aggregate["confirmatory_primary"]
    snapshot_primary = snapshot["primary"]
    for label, observed, expected in (
        ("primary aggregate mean", primary_mean, aggregate_primary["mean"]),
        ("primary snapshot mean", primary_mean, snapshot_primary["mean"]),
        ("primary aggregate lower CI", primary_low, aggregate_primary["ci95_student_t"][0]),
        ("primary aggregate upper CI", primary_high, aggregate_primary["ci95_student_t"][1]),
        ("primary snapshot lower CI", primary_low, snapshot_primary["ci95_student_t"][0]),
        ("primary snapshot upper CI", primary_high, snapshot_primary["ci95_student_t"][1]),
    ):
        _assert_close(observed, expected, label)
    if (
        int(np.count_nonzero(primary_values > 0)) != N_SEEDS
        or aggregate_primary.get("wins") != N_SEEDS
        or snapshot_primary.get("wins") != N_SEEDS
    ):
        raise RuntimeError("primary paired-win count mismatch")

    zero_matrix = np.column_stack(
        [equal - fixed_values[name] for name in ZERO_OVERLAP_CONDITIONS]
    )
    pooled_values = np.mean(zero_matrix, axis=1)
    pooled_mean, pooled_low, pooled_high = _paired_summary(pooled_values)
    aggregate_pooled = aggregate["gated_zero_overlap_generality"]
    snapshot_pooled = snapshot["gated_zero_overlap_generality"]
    for label, observed, expected in (
        ("pooled aggregate mean", pooled_mean, aggregate_pooled["mean"]),
        ("pooled snapshot mean", pooled_mean, snapshot_pooled["mean"]),
        ("pooled aggregate lower CI", pooled_low, aggregate_pooled["ci95_student_t"][0]),
        ("pooled aggregate upper CI", pooled_high, aggregate_pooled["ci95_student_t"][1]),
        ("pooled snapshot lower CI", pooled_low, snapshot_pooled["ci95_student_t"][0]),
        ("pooled snapshot upper CI", pooled_high, snapshot_pooled["ci95_student_t"][1]),
    ):
        _assert_close(observed, expected, label)
    if (
        int(np.count_nonzero(pooled_values > 0)) != N_SEEDS
        or aggregate_pooled.get("wins") != N_SEEDS
        or snapshot_pooled.get("wins") != N_SEEDS
        or aggregate_pooled.get("tested_by_fixed_sequence_gate") is not True
    ):
        raise RuntimeError("gated zero-overlap result mismatch")

    ordered = aggregate["ordered_path_diagnostic"]
    snapshot_ordered = snapshot["ordered_path_diagnostic"]
    _assert_close(
        ordered["mean_rho"],
        snapshot_ordered["mean_within_seed_spearman_rho"],
        "ordered-path mean rho",
    )
    validation = aggregate["validation_selected_sensitivity"]
    snapshot_validation = snapshot["validation_selected_sensitivity"]
    _assert_close(
        validation["mean"],
        snapshot_validation["mean"],
        "validation-selected mean",
    )
    if snapshot.get("convergence_and_replay", {}).get("all_gates_passed") is not True:
        raise RuntimeError("phase-direction convergence and replay gate did not pass")

    return fixed_values, _load_orientation_replication()


def make_figure(
    fixed_values: dict[str, np.ndarray],
    orientation_effects: np.ndarray,
):
    """Draw the paired path and zero-overlap intervention panels."""
    # Author Figure 9 on a true 1:1 column-width canvas.  The panels remain
    # side by side, while the additional height gives both interventions more
    # vertical resolution without any downstream LaTeX scaling or distortion.
    fig = st.composite_figure("column", height=st.QUANTUM_COLUMN_WIDTH)
    left = st.add_axes_inches(fig, (0.46, 0.68, 1.10, 1.95))
    right = st.add_axes_inches(fig, (1.94, 0.68, 1.17, 1.95))
    axes = np.asarray([left, right])

    path_matrix = np.column_stack(
        [fixed_values[condition] for condition in PATH_CONDITIONS]
    )
    for row in path_matrix:
        left.plot(
            PATH_FRACTIONS,
            row,
            color=st.COLLECTIVE,
            alpha=0.095,
            linewidth=0.54,
            zorder=1,
        )
    path_means = np.mean(path_matrix, axis=0)
    path_ci = stats.sem(path_matrix, axis=0) * stats.t.ppf(0.975, N_SEEDS - 1)
    left.errorbar(
        PATH_FRACTIONS,
        path_means,
        yerr=path_ci,
        color=st.COLLECTIVE,
        marker="o",
        markerfacecolor=st.COLLECTIVE,
        markeredgecolor=st.COLLECTIVE,
        markersize=st.MARKER_SIZE + 0.45,
        linewidth=st.DATA_LINEWIDTH + 0.10,
        elinewidth=st.ERROR_LINEWIDTH,
        capsize=st.ERROR_CAPSIZE,
        zorder=3,
    )
    left.set_xlim(-0.05, 1.05)
    left.set_ylim(8.25, 13.95)
    left.set_xticks([0.0, 0.5, 1.0])
    left.set_yticks([9, 11, 13])
    left.set_xlabel(r"phase fraction $f$")
    left.set_ylabel("STM capacity")
    st.style_axis(left, grid_axis="both")

    equal = fixed_values["path_f0"]
    differences = np.column_stack(
        [equal - fixed_values[name] for name in ZERO_OVERLAP_CONDITIONS]
    )
    pooled = np.mean(differences, axis=1)
    displayed = np.column_stack([differences, pooled])
    x = np.arange(displayed.shape[1])
    # A fixed index permutation supplies display-only jitter and is unrelated
    # to the observed values; the same offsets are used for every condition.
    jitter_order = (np.arange(N_SEEDS) * 13) % N_SEEDS
    jitter = np.linspace(-0.105, 0.105, N_SEEDS)[jitter_order]
    for column in range(displayed.shape[1]):
        color = st.NEUTRAL_DESIGN if column < 5 else st.COLLECTIVE
        right.scatter(
            np.full(N_SEEDS, x[column]) + jitter,
            displayed[:, column],
            s=8.0,
            facecolors=color,
            edgecolors="none",
            alpha=0.24 if column < 5 else 0.28,
            zorder=1,
        )
    effect_means = np.mean(displayed, axis=0)
    effect_ci = stats.sem(displayed, axis=0) * stats.t.ppf(0.975, N_SEEDS - 1)
    right.errorbar(
        x[:-1],
        effect_means[:-1],
        yerr=effect_ci[:-1],
        fmt="o",
        color=st.UNIFORM_LOCAL,
        markerfacecolor="white",
        markeredgecolor=st.UNIFORM_LOCAL,
        markeredgewidth=st.MARKER_EDGEWIDTH + 0.20,
        markersize=st.MARKER_SIZE + 0.55,
        elinewidth=st.ERROR_LINEWIDTH,
        capsize=st.ERROR_CAPSIZE,
        zorder=3,
    )
    right.errorbar(
        x[-1],
        effect_means[-1],
        yerr=effect_ci[-1],
        fmt="D",
        color=st.COLLECTIVE,
        markerfacecolor=st.COLLECTIVE,
        markeredgecolor=st.COLLECTIVE,
        markersize=st.MARKER_SIZE + 0.70,
        elinewidth=st.ERROR_LINEWIDTH,
        capsize=st.ERROR_CAPSIZE,
        zorder=4,
    )
    orientation_x = 6
    orientation_order = (np.arange(N_ORIENTATION_PAIRS) * 11) % N_ORIENTATION_PAIRS
    orientation_jitter = np.linspace(-0.105, 0.105, N_ORIENTATION_PAIRS)[
        orientation_order
    ]
    right.scatter(
        np.full(N_ORIENTATION_PAIRS, orientation_x) + orientation_jitter,
        orientation_effects,
        s=8.0,
        facecolors=st.COLLECTIVE,
        edgecolors="none",
        alpha=0.28,
        zorder=1,
    )
    orientation_mean, orientation_low, orientation_high = _paired_summary(
        orientation_effects
    )
    right.errorbar(
        orientation_x,
        orientation_mean,
        yerr=np.asarray(
            [[orientation_mean - orientation_low], [orientation_high - orientation_mean]]
        ),
        fmt="s",
        color=st.COLLECTIVE,
        markerfacecolor=st.COLLECTIVE,
        markeredgecolor=st.COLLECTIVE,
        markersize=st.MARKER_SIZE + 0.55,
        elinewidth=st.ERROR_LINEWIDTH,
        capsize=st.ERROR_CAPSIZE,
        zorder=4,
    )
    right.axvline(
        5.5,
        color=st.NEUTRAL_DESIGN,
        linestyle=":",
        linewidth=st.REFERENCE_LINEWIDTH,
        zorder=2,
    )
    right.axhline(
        0,
        color=st.INK,
        linestyle="--",
        linewidth=st.REFERENCE_LINEWIDTH,
        zorder=2,
    )
    right.set_xlim(-0.42, 6.42)
    right.set_ylim(-0.18, 4.62)
    right.set_xticks(
        np.arange(7),
        ["F", "A", "B", "C", "D", r"$\mu$", "S"],
        rotation=0,
        ha="center",
    )
    right.set_yticks([0, 2, 4])
    right.set_xlabel("control")
    right.set_ylabel("STM gain")
    st.style_axis(right, grid_axis="y")

    st.panel_labels(fig, axes, labels="ab", y=1.055)
    st.audit_figure(
        fig,
        "fig_phase_direction",
        axes=axes,
        overlap_fraction=0.10,
    )
    return fig


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PDF output path",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="optional PNG preview path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fixed_values, orientation_effects = load_and_validate()
    fig = make_figure(fixed_values, orientation_effects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        format="pdf",
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
        metadata={
            "Creator": "paper/make_phase_direction_figure.py",
            "Title": "Matched rank-one direction interventions",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    if args.preview:
        fig.set_layout_engine("none")
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.preview,
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
            metadata={"Software": "paper/make_phase_direction_figure.py"},
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

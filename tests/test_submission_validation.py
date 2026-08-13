"""Focused checks for the compact reset evidence submission contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_submission.py"
SPEC = importlib.util.spec_from_file_location("submission_validation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
submission_validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submission_validation
SPEC.loader.exec_module(submission_validation)


def test_reset_snapshot_passes_sealed_contract() -> None:
    failures: list[str] = []
    submission_validation.validate_reset_snapshot(
        submission_validation.RESET_SNAPSHOT,
        failures,
    )
    assert failures == []


def test_reset_snapshot_rejects_lost_paired_win(tmp_path: Path) -> None:
    payload = json.loads(
        submission_validation.RESET_SNAPSHOT.read_text(encoding="utf-8")
    )
    payload["arrays"]["stm_collective"][0] = payload["arrays"]["stm_local"][0]
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    failures: list[str] = []
    submission_validation.validate_reset_snapshot(altered, failures)
    assert any("16/16 favorable pairs" in failure for failure in failures)


def test_phase_direction_snapshot_passes_final_replay_contract() -> None:
    failures: list[str] = []
    submission_validation.validate_phase_direction_snapshot(
        submission_validation.PHASE_SNAPSHOT,
        failures,
    )
    assert failures == []


def test_phase_direction_snapshot_rejects_stale_replay_hash(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        submission_validation.PHASE_SNAPSHOT.read_text(encoding="utf-8")
    )
    payload["validation_report_sha256"] = "0" * 64
    altered = tmp_path / "altered-phase.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    failures: list[str] = []
    submission_validation.validate_phase_direction_snapshot(altered, failures)
    assert any("validation_report.json" in failure for failure in failures)


def test_rank_one_orientation_snapshot_passes_hardened_contract() -> None:
    failures: list[str] = []
    submission_validation.validate_rank_one_orientation_snapshot(
        submission_validation.ORIENTATION_SNAPSHOT,
        failures,
    )
    assert failures == []


def test_rank_one_orientation_snapshot_rejects_altered_primary_mean(
    tmp_path: Path,
) -> None:
    payload = json.loads(
        submission_validation.ORIENTATION_SNAPSHOT.read_text(encoding="utf-8")
    )
    payload["stm"]["paired_mean"] += 0.01
    altered = tmp_path / "altered-orientation.json"
    altered.write_text(json.dumps(payload), encoding="utf-8")

    failures: list[str] = []
    submission_validation.validate_rank_one_orientation_snapshot(
        altered,
        failures,
    )
    assert any("primary paired result mismatch" in failure for failure in failures)


def test_continuous_drive_narma_washout_passes_sealed_contract() -> None:
    failures: list[str] = []
    submission_validation.validate_continuous_narma_washout(failures)
    assert failures == []

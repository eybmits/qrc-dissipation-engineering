"""Tests for deterministic result-archive construction."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_result_archives.py"
ROOT = SCRIPT.parents[1]
SPEC = importlib.util.spec_from_file_location("result_archives", SCRIPT)
result_archives = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = result_archives
SPEC.loader.exec_module(result_archives)


def test_tree_archive_is_deterministic_and_excludes_machine_local_resume(tmp_path):
    source = tmp_path / "experiment1_finite_size_v2"
    checkpoints = source / "checkpoints" / "N8"
    checkpoints.mkdir(parents=True)
    (source / "aggregate.json").write_text("{}\n", encoding="utf-8")
    (source / "resume_production.zsh").write_text(
        "#!/bin/zsh\ncd /machine/local/path\n",
        encoding="utf-8",
    )
    (checkpoints / "row.json").write_text('{"status":"complete"}\n', encoding="utf-8")

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    excluded = result_archives.EXCLUDED_TREE_MEMBERS[source.name]
    one = result_archives.build_protocol_archive(
        source, first, source.name, excluded_members=excluded
    )
    two = result_archives.build_protocol_archive(
        source, second, source.name, excluded_members=excluded
    )

    assert one.sha256 == two.sha256
    assert first.read_bytes() == second.read_bytes()
    assert one.file_count == 2
    assert one.excluded_count == 1


def test_published_finite_size_archive_is_complete_and_portable():
    path = ROOT / "results" / "experiment1_finite_size_v2_results.tar.gz"
    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names))
        for member in members:
            pure_name = PurePosixPath(member.name)
            assert not pure_name.is_absolute()
            assert ".." not in pure_name.parts
            assert member.isdir() or member.isfile()
            assert member.mtime == result_archives.NORMALIZED_MTIME
            assert member.uid == result_archives.NORMALIZED_UID
            assert member.gid == result_archives.NORMALIZED_GID
            assert member.uname == member.gname == ""
            expected_mode = (
                result_archives.DIRECTORY_MODE
                if member.isdir()
                else result_archives.FILE_MODE
            )
            assert member.mode == expected_mode
        assert not any("resume_production.zsh" in name for name in names)

        prefix = "experiment1_finite_size_v2/"
        checkpoint_names = [
            name
            for name in names
            if name.startswith(f"{prefix}checkpoints/")
            and name.endswith(".json")
        ]
        assert len(checkpoint_names) == 960
        for n_qubits in range(4, 9):
            assert (
                sum(
                    f"{prefix}checkpoints/N{n_qubits}/" in name
                    for name in checkpoint_names
                )
                == 192
            )

        status_file = bundle.extractfile(f"{prefix}status.json")
        assert status_file is not None
        status = json.load(status_file)
        assert status["complete"] == status["expected"] == 960
        assert status["pending"] == 0


def test_published_operational_activity_archive_preserves_corrected_provenance(
    tmp_path,
):
    path = ROOT / "results" / "operational_activity_ablation_results.tar.gz"
    prefix = "operational_activity_ablation/"
    original_aggregate_sha256 = (
        "94e720aa2ac0f1decbcf73984f8cb396bcf05230aed3feed543ad08d4844dfdb"
    )
    corrected_aggregate_sha256 = (
        "14e38f785b7192e42b240c25f0199358ea6578176060b433f08d7c86b075c28f"
    )

    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names))
        assert names[0] == prefix.rstrip("/")
        for member in members:
            pure_name = PurePosixPath(member.name)
            assert not pure_name.is_absolute()
            assert ".." not in pure_name.parts
            assert member.isdir() or member.isfile()
            assert member.mtime == result_archives.NORMALIZED_MTIME
            assert member.uid == result_archives.NORMALIZED_UID
            assert member.gid == result_archives.NORMALIZED_GID
            assert member.uname == member.gname == ""
            expected_mode = (
                result_archives.DIRECTORY_MODE
                if member.isdir()
                else result_archives.FILE_MODE
            )
            assert member.mode == expected_mode

        file_members = {member.name: member for member in members if member.isfile()}
        provenance_member = file_members[f"{prefix}provenance.json"]
        provenance_file = bundle.extractfile(provenance_member)
        assert provenance_file is not None
        provenance = json.load(provenance_file)

        recorded_hashes = provenance["files_sha256"]
        expected_files = {f"{prefix}provenance.json"} | {
            f"{prefix}{relative}" for relative in recorded_hashes
        }
        assert len(file_members) == 35
        assert set(file_members) == expected_files
        for relative, expected_sha256 in recorded_hashes.items():
            archived_file = bundle.extractfile(file_members[f"{prefix}{relative}"])
            assert archived_file is not None
            assert hashlib.sha256(archived_file.read()).hexdigest() == expected_sha256

        correction = provenance["post_run_inference_correction"]
        assert (
            correction["original_aggregate_sha256"]
            == original_aggregate_sha256
        )
        assert (
            correction["corrected_aggregate_sha256"]
            == corrected_aggregate_sha256
        )
        assert recorded_hashes["aggregate.json"] == corrected_aggregate_sha256
        assert "unchanged per-seed rows" in correction["description"]
        assert "No simulation or task score was rerun" in correction["description"]

        protocol_member = bundle.extractfile(file_members[f"{prefix}protocol.json"])
        assert protocol_member is not None
        protocol = json.load(protocol_member)
        assert protocol["protocol_sha256"] == provenance["protocol_sha256"]
        assert (
            len(
                [
                    name
                    for name in file_members
                    if f"{prefix}calibration/" in name
                ]
            )
            == 8
        )
        assert (
            len([name for name in file_members if f"{prefix}tasks/" in name])
            == 8
        )

        source = tmp_path / prefix.rstrip("/")
        for member in members:
            relative = PurePosixPath(member.name).relative_to(prefix.rstrip("/"))
            target = source.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                archived_file = bundle.extractfile(member)
                assert archived_file is not None
                target.write_bytes(archived_file.read())

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    one = result_archives.build_protocol_archive(
        source, first, prefix.rstrip("/")
    )
    two = result_archives.build_protocol_archive(
        source, second, prefix.rstrip("/")
    )
    assert one.sha256 == two.sha256
    assert first.read_bytes() == second.read_bytes() == path.read_bytes()

    paper_snapshot = json.loads(
        (ROOT / "paper/data/activity_matched_confirmation.json").read_text(
            encoding="utf-8"
        )
    )
    assert (
        paper_snapshot["artifact_aggregate_sha256"]
        == original_aggregate_sha256
    )
    assert (
        paper_snapshot["verified_aggregate_sha256"]
        == corrected_aggregate_sha256
    )
    assert (
        paper_snapshot["github_artifact_sha256"]
        == provenance["github"]["artifact_digest"].removeprefix("sha256:")
    )
    assert paper_snapshot["verified_evidence_commit"].startswith("ed4124f")
    assert not paper_snapshot["post_run_inference_correction"][
        "simulation_or_task_scores_rerun"
    ]


def test_published_reset_architecture_archive_is_exact_and_self_verifying(
    tmp_path,
):
    path = (
        ROOT
        / "results"
        / "reset_architecture_replication_results.tar.gz"
    )
    prefix = f"{result_archives.RESET_ARCHITECTURE_ROOT}/"
    expected_names = [
        result_archives.RESET_ARCHITECTURE_ROOT,
        *[
            f"{prefix}{relative}"
            for relative in result_archives.RESET_ARCHITECTURE_FILES
        ],
    ]

    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert names == expected_names
        assert len(names) == len(set(names))
        for member in members:
            pure_name = PurePosixPath(member.name)
            assert not pure_name.is_absolute()
            assert ".." not in pure_name.parts
            assert member.isdir() or member.isfile()
            assert member.mtime == result_archives.NORMALIZED_MTIME
            assert member.uid == result_archives.NORMALIZED_UID
            assert member.gid == result_archives.NORMALIZED_GID
            assert member.uname == member.gname == ""
            expected_mode = (
                result_archives.DIRECTORY_MODE
                if member.isdir()
                else result_archives.FILE_MODE
            )
            assert member.mode == expected_mode

        file_members = {
            member.name: member
            for member in members
            if member.isfile()
        }
        checksum_name = (
            f"{prefix}{result_archives.RESET_ARCHITECTURE_CHECKSUM_NAME}"
        )
        checksum_file = bundle.extractfile(file_members[checksum_name])
        assert checksum_file is not None
        recorded_hashes = result_archives.parse_sha256_manifest(
            checksum_file.read(),
            checksum_name,
        )
        expected_payloads = set(
            result_archives.RESET_ARCHITECTURE_FILES
        ) - {result_archives.RESET_ARCHITECTURE_CHECKSUM_NAME}
        assert set(recorded_hashes) == expected_payloads
        for relative, expected_sha256 in recorded_hashes.items():
            archived_file = bundle.extractfile(file_members[f"{prefix}{relative}"])
            assert archived_file is not None
            assert (
                hashlib.sha256(archived_file.read()).hexdigest()
                == expected_sha256
            )

        source = tmp_path / result_archives.RESET_ARCHITECTURE_ROOT
        for member in members:
            relative = PurePosixPath(member.name).relative_to(
                result_archives.RESET_ARCHITECTURE_ROOT
            )
            target = source.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                archived_file = bundle.extractfile(member)
                assert archived_file is not None
                target.write_bytes(archived_file.read())

    result_archives.validate_reset_architecture_source(source)
    result_archives.validate_reset_architecture_archive(path)
    first = tmp_path / "reset-first.tar.gz"
    second = tmp_path / "reset-second.tar.gz"
    one = result_archives.build_protocol_archive(
        source,
        first,
        result_archives.RESET_ARCHITECTURE_ROOT,
    )
    two = result_archives.build_protocol_archive(
        source,
        second,
        result_archives.RESET_ARCHITECTURE_ROOT,
    )
    assert one.sha256 == two.sha256 == result_archives.sha256_file(path)
    assert first.read_bytes() == second.read_bytes() == path.read_bytes()

    scores = source / "strict_washout_scores.csv"
    original_scores = scores.read_bytes()
    scores.write_bytes(original_scores + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        result_archives.validate_reset_architecture_source(source)

    scores.write_bytes(original_scores)
    (source / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="membership mismatch"):
        result_archives.validate_reset_architecture_source(source)


def test_published_phase_direction_archive_is_exact_and_self_verifying(
    tmp_path,
):
    path = (
        ROOT
        / "results"
        / result_archives.PHASE_DIRECTION_ARCHIVE_NAME
    )
    validated = result_archives.validate_phase_direction_archive(path)
    assert validated["task_checkpoints"] == 288
    assert validated["convergence_checkpoints"] == 72
    assert validated["primary_mean_difference"] > 0

    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names)) == 374
        assert names[0] == result_archives.PHASE_DIRECTION_ROOT
        for member in members:
            pure_name = PurePosixPath(member.name)
            assert not pure_name.is_absolute()
            assert ".." not in pure_name.parts
            assert member.isdir() or member.isfile()
            assert member.mtime == result_archives.NORMALIZED_MTIME
            assert member.uid == result_archives.NORMALIZED_UID
            assert member.gid == result_archives.NORMALIZED_GID
            assert member.uname == member.gname == ""
            expected_mode = (
                result_archives.DIRECTORY_MODE
                if member.isdir()
                else result_archives.FILE_MODE
            )
            assert member.mode == expected_mode

        source = tmp_path / result_archives.PHASE_DIRECTION_ROOT
        for member in members:
            relative = PurePosixPath(member.name).relative_to(
                result_archives.PHASE_DIRECTION_ROOT
            )
            target = source.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                archived_file = bundle.extractfile(member)
                assert archived_file is not None
                target.write_bytes(archived_file.read())

    source_validation = result_archives.validate_phase_direction_source(source)
    assert source_validation == validated

    first = tmp_path / "phase-first.tar.gz"
    second = tmp_path / "phase-second.tar.gz"
    one = result_archives.build_protocol_archive(
        source,
        first,
        result_archives.PHASE_DIRECTION_ROOT,
    )
    two = result_archives.build_protocol_archive(
        source,
        second,
        result_archives.PHASE_DIRECTION_ROOT,
    )
    assert one.sha256 == two.sha256 == result_archives.sha256_file(path)
    assert first.read_bytes() == second.read_bytes() == path.read_bytes()

    aggregate = source / "aggregate.json"
    original_aggregate = aggregate.read_bytes()
    aggregate.write_bytes(original_aggregate + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        result_archives.validate_phase_direction_source(source)

    aggregate.write_bytes(original_aggregate)
    (source / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum membership mismatch"):
        result_archives.validate_phase_direction_source(source)


def test_published_rank_one_orientation_archive_is_exact_and_self_verifying(
    tmp_path,
):
    path = (
        ROOT
        / "results"
        / result_archives.RANK_ONE_ORIENTATION_ARCHIVE_NAME
    )
    validated = result_archives.validate_rank_one_orientation_archive(path)
    assert validated["file_count"] == 52
    assert validated["pair_count"] == validated["wins"] == 24
    assert validated["primary_mean_difference"] == pytest.approx(
        2.271492573239316, abs=1e-12
    )
    assert validated["primary_ci95"] == pytest.approx(
        [1.8258875618947812, 2.7170975845838505], abs=1e-12
    )

    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names))
        assert names[0] == result_archives.RANK_ONE_ORIENTATION_ROOT
        checkpoint_names = [
            name for name in names if "/checkpoints/seed_" in name
        ]
        assert len(checkpoint_names) == 24
        for member in members:
            pure_name = PurePosixPath(member.name)
            assert not pure_name.is_absolute()
            assert ".." not in pure_name.parts
            assert member.isdir() or member.isfile()
            assert member.mtime == result_archives.NORMALIZED_MTIME
            assert member.uid == result_archives.NORMALIZED_UID
            assert member.gid == result_archives.NORMALIZED_GID
            assert member.uname == member.gname == ""
            expected_mode = (
                result_archives.DIRECTORY_MODE
                if member.isdir()
                else result_archives.FILE_MODE
            )
            assert member.mode == expected_mode

        source = tmp_path / result_archives.RANK_ONE_ORIENTATION_ROOT
        for member in members:
            relative = PurePosixPath(member.name).relative_to(
                result_archives.RANK_ONE_ORIENTATION_ROOT
            )
            target = source.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                archived_file = bundle.extractfile(member)
                assert archived_file is not None
                target.write_bytes(archived_file.read())

    source_validation = result_archives.validate_rank_one_orientation_source(
        source
    )
    assert source_validation == validated

    first = tmp_path / "orientation-first.tar.gz"
    second = tmp_path / "orientation-second.tar.gz"
    one = result_archives.build_protocol_archive(
        source,
        first,
        result_archives.RANK_ONE_ORIENTATION_ROOT,
    )
    two = result_archives.build_protocol_archive(
        source,
        second,
        result_archives.RANK_ONE_ORIENTATION_ROOT,
    )
    assert one.sha256 == two.sha256 == result_archives.sha256_file(path)
    assert first.read_bytes() == second.read_bytes() == path.read_bytes()

    checkpoint = source / "checkpoints" / "seed_00.json"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        result_archives.validate_rank_one_orientation_source(source)

    checkpoint.write_bytes(original)

    summary = source / "derived" / "summary.json"
    ledger = source / result_archives.RANK_ONE_ORIENTATION_CHECKSUM_NAME
    original_summary = summary.read_bytes()
    original_ledger = ledger.read_bytes()
    changed_summary = json.loads(original_summary)
    changed_summary["stm"]["paired_equal_minus_orthogonal"]["mean"] += 0.1
    summary.write_text(
        json.dumps(changed_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    changed_digest = hashlib.sha256(summary.read_bytes()).hexdigest()
    ledger.write_text(
        "\n".join(
            (
                f"{changed_digest}  derived/summary.json"
                if line.endswith("  derived/summary.json")
                else line
            )
            for line in original_ledger.decode("utf-8").splitlines()
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="numeric mismatch"):
        result_archives.validate_rank_one_orientation_source(source)

    summary.write_bytes(original_summary)
    ledger.write_bytes(original_ledger)
    (source / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum membership mismatch"):
        result_archives.validate_rank_one_orientation_source(source)

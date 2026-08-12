"""Tests for the deterministic, self-contained arXiv source archive."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_quantum_source_archive.py"
)
SPEC = importlib.util.spec_from_file_location("quantum_source_archive", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
source_archive = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = source_archive
SPEC.loader.exec_module(source_archive)


def _paper_fixture(root: Path) -> Path:
    paper = root / "paper"
    (paper / "sections").mkdir(parents=True)
    (paper / "figures").mkdir()
    (paper / "dissipation_qrc.tex").write_text(
        "\\input{sections/abstract}\n"
        + "\n".join(
            f"\\includegraphics{{{name}}}"
            for name in source_archive.EXPECTED_FIGURES
        )
        + "\n",
        encoding="utf-8",
    )
    for relative in source_archive.REQUIRED_FILES:
        path = paper / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("fixture\n", encoding="utf-8")
    reference_keys = [
        f"Reference{index:02d}"
        for index in range(source_archive.EXPECTED_REFERENCE_COUNT)
    ]
    (paper / "references.bib").write_text(
        "\n".join(
            "@article{"
            f"{key}, author={{Fixture}}, title={{Fixture}}, "
            f"year={{2026}}, doi={{10.1000/{index:02d}}}"
            "}"
            for index, key in enumerate(reference_keys)
        )
        + "\n",
        encoding="utf-8",
    )
    (paper / "dissipation_qrc.bbl").write_text(
        "\n".join(
            f"\\bibitem{{{key}}} \\href{{https://doi.org/10.1000/{index:02d}}}"
            "{Fixture}."
            for index, key in enumerate(reference_keys)
        )
        + "\n",
        encoding="utf-8",
    )
    for relative in source_archive.EXPECTED_FIGURES:
        (paper / relative).write_bytes(b"%PDF-1.4\nfixture\n")
    return paper


def test_build_is_deterministic_and_complete(tmp_path: Path) -> None:
    paper = _paper_fixture(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    first_result = source_archive.build_archive(first, paper)
    second_result = source_archive.build_archive(second, paper)

    expected_file_count = (
        len(source_archive.REQUIRED_FILES)
        + len(source_archive.EXPECTED_FIGURES)
        + len(source_archive.REPOSITORY_FILES)
        + len(source_archive.GENERATED_FILES)
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_result["sha256"] == second_result["sha256"]
    assert first_result["file_count"] == expected_file_count
    assert first_result["figure_count"] == len(source_archive.EXPECTED_FIGURES)
    assert source_archive.verify_archive(first) == {
        "sha256": first_result["sha256"],
        "file_count": expected_file_count,
        "figure_count": len(source_archive.EXPECTED_FIGURES),
    }


def test_build_rejects_unresolved_manuscript_token(tmp_path: Path) -> None:
    paper = _paper_fixture(tmp_path)
    with (paper / "sections/abstract.tex").open("a", encoding="utf-8") as handle:
        handle.write("PENDING\n")

    with pytest.raises(source_archive.SourceArchiveError, match="unresolved token"):
        source_archive.build_archive(tmp_path / "bad.zip", paper)


def test_build_rejects_uncited_bibliography_entry(tmp_path: Path) -> None:
    paper = _paper_fixture(tmp_path)
    generated = paper / "dissipation_qrc.bbl"
    generated.write_text(
        generated.read_text(encoding="utf-8").replace(
            "\\bibitem{Reference45} "
            "\\href{https://doi.org/10.1000/45}{Fixture}.\n",
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        source_archive.SourceArchiveError,
        match="exactly 46 cited entries",
    ):
        source_archive.build_archive(tmp_path / "bad-references.zip", paper)


def test_build_rejects_unlinked_rendered_reference(tmp_path: Path) -> None:
    paper = _paper_fixture(tmp_path)
    generated = paper / "dissipation_qrc.bbl"
    generated.write_text(
        generated.read_text(encoding="utf-8").replace(
            "\\bibitem{Reference45} "
            "\\href{https://doi.org/10.1000/45}{Fixture}.",
            "\\bibitem{Reference45} Fixture.",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        source_archive.SourceArchiveError,
        match="rendered bibliography entry must contain a hyperlink",
    ):
        source_archive.build_archive(tmp_path / "bad-links.zip", paper)


def test_verify_rejects_tampered_payload(tmp_path: Path) -> None:
    paper = _paper_fixture(tmp_path)
    valid = tmp_path / "valid.zip"
    tampered = tmp_path / "tampered.zip"
    source_archive.build_archive(valid, paper)

    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(tampered, "w") as sink:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "references.bib":
                payload += b"tamper"
            sink.writestr(info, payload)

    with pytest.raises(source_archive.SourceArchiveError, match="checksum mismatch"):
        source_archive.verify_archive(tampered)


def test_verify_rejects_symlink_member(tmp_path: Path) -> None:
    paper = _paper_fixture(tmp_path)
    valid = tmp_path / "valid.zip"
    unsafe = tmp_path / "unsafe.zip"
    source_archive.build_archive(valid, paper)

    with zipfile.ZipFile(valid) as source, zipfile.ZipFile(unsafe, "w") as sink:
        for original in source.infolist():
            info = zipfile.ZipInfo(original.filename, date_time=original.date_time)
            info.compress_type = original.compress_type
            info.create_system = 3
            if original.filename == "references.bib":
                info.external_attr = (0o120777 << 16)
            else:
                info.external_attr = original.external_attr
            sink.writestr(info, source.read(original.filename))

    with pytest.raises(source_archive.SourceArchiveError, match="non-regular"):
        source_archive.verify_archive(unsafe)

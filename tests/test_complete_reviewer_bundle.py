"""Integration checks for the self-contained reviewer bundle."""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_complete_reviewer_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("complete_reviewer_bundle", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bundle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bundle
SPEC.loader.exec_module(bundle)


def test_complete_reviewer_bundle_is_deterministic_and_valid(tmp_path: Path) -> None:
    first = bundle.build(tmp_path / "first.zip")
    second = bundle.build(tmp_path / "second.zip")

    assert first.read_bytes() == second.read_bytes()
    bundle.verify(first)

    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    prefix = bundle.ARCHIVE_ROOT + "/"
    assert prefix + "REPRODUCIBILITY_INDEX.md" in names
    assert prefix + "protocol_manifest.json" in names
    assert prefix + "manuscript_source.zip" in names
    assert prefix + "numerical_evidence.zip" in names
    assert prefix + "phase_direction_confirmatory_v1_results.tar.gz" in names
    assert prefix + "rank_one_orientation_v1_results.tar.gz" in names
    assert prefix + "validate_complete_bundle.py" in names

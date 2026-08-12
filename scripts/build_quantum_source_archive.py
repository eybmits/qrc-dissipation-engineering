#!/usr/bin/env python3
"""Build and verify the self-contained Quantum/arXiv manuscript source ZIP."""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "paper"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "arxiv_submission.zip"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o644

LOCAL_PAIR_EXTENSION_ROOT = "evidence/local_pair_convergence_extension_v1"
LOCAL_PAIR_EXTENSION_SEEDS = (
    181_031_542,
    307_836_921,
    613_734_097,
    818_242_779,
    1_067_952_113,
    1_319_483_087,
    1_673_840_529,
    1_984_601_307,
)
LOCAL_PAIR_EXTENSION_FILES = (
    f"{LOCAL_PAIR_EXTENSION_ROOT}/protocol.json",
    f"{LOCAL_PAIR_EXTENSION_ROOT}/aggregate.json",
    f"{LOCAL_PAIR_EXTENSION_ROOT}/SHA256SUMS",
    *(
        f"{LOCAL_PAIR_EXTENSION_ROOT}/checkpoints/"
        f"principal_{design}_N{size}_seed_{seed}.json"
        for design in ("local", "pair")
        for size in (4, 5, 6)
        for seed in LOCAL_PAIR_EXTENSION_SEEDS
    ),
)

REQUIRED_FILES = (
    "dissipation_qrc.tex",
    "dissipation_qrc.bbl",
    "quantumarticle.cls",
    "quantum.bst",
    "references.bib",
    "fig1_L3.py",
    "l3_style.py",
    "make_figures.py",
    "make_forgetting_modes_figure.py",
    "make_phase_direction_figure.py",
    "make_reset_architecture_figure.py",
    "data/activity_matched_confirmation.json",
    "data/experiment1_finite_size_snapshot.json",
    "data/experiment1_finite_size_seed_values.json",
    "data/experiment1_principal_summary.json",
    "data/experiment1_scalar_control_seed_values.json",
    "data/experiment1_finite_size_lag_snapshot.json",
    "data/experiment1_parity_window_snapshot.json",
    "data/experiment1_robustness_snapshot.json",
    "data/reset_architecture_snapshot.json",
    "data/phase_direction_confirmatory_snapshot.json",
    "data/rank_one_orientation_snapshot.json",
    "data/reproducibility_manifest.json",
    "evidence/switched_input_memory_control_v2/aggregate.json",
    "evidence/phase_direction_confirmatory_v1/protocol.json",
    "evidence/phase_direction_confirmatory_v1/aggregate.json",
    "evidence/phase_direction_confirmatory_v1/convergence_summary.json",
    "evidence/phase_direction_confirmatory_v1/validation_amendment.json",
    "evidence/phase_direction_confirmatory_v1/numerical_replay_audit.json",
    "evidence/phase_direction_confirmatory_v1/validation_report.json",
    "evidence/rank_one_orientation_v1/README.md",
    "evidence/rank_one_orientation_v1/protocol.json",
    "evidence/rank_one_orientation_v1/provenance.json",
    "evidence/rank_one_orientation_v1/environment.json",
    "evidence/rank_one_orientation_v1/validation_report.json",
    "evidence/rank_one_orientation_v1/derived/summary.json",
    "evidence/rank_one_orientation_v1/validate.py",
    "evidence/rank_one_orientation_v1/frozen_source/requirements.txt",
    "evidence/rank_one_orientation_v1/frozen_source/experiments/run_rank_one_orientation.py",
    "evidence/rank_one_orientation_v1/frozen_source/experiments/"
    "aggregate_rank_one_orientation_artifacts.py",
    "evidence/collective_N5_convergence_extension_v1/protocol.json",
    "evidence/collective_N5_convergence_extension_v1/aggregate.json",
    "evidence/collective_N5_convergence_extension_v1/SHA256SUMS",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_1067952113.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_1319483087.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_1673840529.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_181031542.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_1984601307.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_307836921.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_613734097.json",
    "evidence/collective_N5_convergence_extension_v1/checkpoints/seed_818242779.json",
    *LOCAL_PAIR_EXTENSION_FILES,
    "evidence/canonical_gap_control/calibration.csv",
    "evidence/canonical_gap_control/lag_capacities.csv",
    "sections/abstract.tex",
    "sections/background.tex",
    "sections/conclusion.tex",
    "sections/evaluation.tex",
    "sections/experimental-section.tex",
    "sections/introduction.tex",
    "sections/methodology.tex",
    "sections/related-work.tex",
)
EXPECTED_FIGURES = (
    "figures/fig_designspace.pdf",
    "figures/fig_task_scores.pdf",
    "figures/fig_map.pdf",
    "figures/fig_collective_case.pdf",
    "figures/fig_profiles.pdf",
    "figures/fig_sampling.pdf",
    "figures/fig_scalar_controls.pdf",
    "figures/fig_reset_architecture.pdf",
    "figures/fig_phase_direction.pdf",
)
GENERATED_FILES = ("README.txt", "SHA256SUMS.txt")
REPOSITORY_FILES = (
    (
        "evidence/collective_N5_convergence_extension_v1/"
        "run_collective_convergence_extension.py",
        REPO_ROOT / "experiments" / "run_collective_convergence_extension.py",
    ),
    (
        f"{LOCAL_PAIR_EXTENSION_ROOT}/"
        "run_local_pair_convergence_extension.py",
        REPO_ROOT / "experiments" / "run_local_pair_convergence_extension.py",
    ),
    (
        "evidence/reset_architecture_replication/"
        "run_reset_architecture_strict.py",
        REPO_ROOT / "experiments" / "run_reset_architecture_strict.py",
    ),
    (
        "evidence/phase_direction_confirmation/"
        "run_phase_direction_confirmatory.py",
        REPO_ROOT / "experiments" / "run_phase_direction_confirmatory.py",
    ),
    (
        "evidence/phase_direction_confirmation/"
        "validate_phase_direction_confirmatory.py",
        REPO_ROOT / "scripts" / "validate_phase_direction_confirmatory.py",
    ),
)
EXPECTED_REFERENCE_COUNT = 46


class SourceArchiveError(RuntimeError):
    """Raised when a source package is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class SourcePayload:
    name: str
    data: bytes


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or ":" in path.parts[0]
    ):
        raise SourceArchiveError(f"unsafe archive path: {value}")
    return path.as_posix()


def _read_regular(path: Path, paper_root: Path) -> bytes:
    try:
        relative = path.relative_to(paper_root).as_posix()
    except ValueError as error:
        raise SourceArchiveError(f"source escapes paper directory: {path}") from error
    _safe_name(relative)
    if path.is_symlink():
        raise SourceArchiveError(f"symlinks are not allowed: {relative}")
    if not path.is_file():
        raise SourceArchiveError(f"missing source file: {relative}")
    return path.read_bytes()


def _read_repository_regular(path: Path) -> bytes:
    path = path.resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise SourceArchiveError(f"source escapes repository: {path}") from error
    if path.is_symlink():
        raise SourceArchiveError(f"symlinks are not allowed: {path}")
    if not path.is_file():
        raise SourceArchiveError(f"missing repository source file: {path}")
    return path.read_bytes()


def _read_tex_tree(
    path: Path,
    paper_root: Path,
    seen: set[Path] | None = None,
) -> str:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    try:
        path.relative_to(paper_root.resolve())
    except ValueError as error:
        raise SourceArchiveError(f"TeX input escapes paper directory: {path}") from error
    source = path.read_text(encoding="utf-8")
    parts = [source]
    for input_name in re.findall(r"\\input\{([^}]+)\}", source):
        child = path.parent / input_name
        if child.suffix == "":
            child = child.with_suffix(".tex")
        if not child.is_file():
            raise SourceArchiveError(f"missing TeX input: {input_name}")
        parts.append(_read_tex_tree(child, paper_root, seen))
    return "\n".join(parts)


def _readme() -> bytes:
    return (
        "arXiv submission source package\n"
        "===============================\n\n"
        "Canonical source: dissipation_qrc.tex\n\n"
        "Build from this directory with:\n"
        "  latexmk -pdf -interaction=nonstopmode -halt-on-error "
        "dissipation_qrc.tex\n\n"
        "The package contains the generated bibliography, the bundled "
        "quantumarticle class and bibliography style, all section files, "
        "and exactly the nine vector figures used by the manuscript. The "
        "figure sources and compact paper snapshots are included for direct "
        "inspection, including the reset-architecture snapshot and its "
        "repo-native strict driver. The N=6 rank-one orientation snapshot is "
        "bound to its compact validated summary, protocol, provenance, "
        "environment, validator, and frozen execution and aggregation source; "
        "the separately released full record contains all 24 checkpoints. "
        "The final N=5 numerical-replay report and its non-scientific protocol "
        "amendment are also included and hash-bound to the compact snapshot. "
        "The exact protocol manifest, frozen "
        "driver snapshots, and complete N=5 "
        "collective and cross-size local/pair continuation records are included. "
        "The separate numerical-evidence archive contains the full raw "
        "outputs. The complete project repository and numerical evidence "
        "have been released publicly at "
        "https://github.com/eybmits/qrc-dissipation-engineering.\n"
    ).encode("utf-8")


def collect_payloads(paper_root: Path = PAPER_ROOT) -> list[SourcePayload]:
    paper_root = paper_root.resolve()
    main_tex = paper_root / "dissipation_qrc.tex"
    manuscript = _read_tex_tree(main_tex, paper_root)

    unresolved = re.search(
        r"\b(?:PENDING|PLACEHOLDER|TODO)\b|\?\?",
        manuscript,
        flags=re.IGNORECASE,
    )
    if unresolved:
        raise SourceArchiveError(
            f"manuscript contains unresolved token: {unresolved.group(0)!r}"
        )

    observed_figures = {
        _safe_name(name)
        for name in re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",
            manuscript,
        )
    }
    if observed_figures != set(EXPECTED_FIGURES):
        raise SourceArchiveError(
            "included figure set differs from the sealed figure allowlist: "
            f"{sorted(observed_figures)}"
        )

    bibliography = (paper_root / "references.bib").read_text(encoding="utf-8")
    generated_bibliography = (paper_root / "dissipation_qrc.bbl").read_text(
        encoding="utf-8"
    )
    bib_keys = re.findall(
        r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,",
        bibliography,
    )
    bbl_keys = re.findall(
        r"\\bibitem(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}",
        generated_bibliography,
    )
    if len(bib_keys) != len(set(bib_keys)):
        raise SourceArchiveError("references.bib contains duplicate entry keys")
    if len(bbl_keys) != len(set(bbl_keys)):
        raise SourceArchiveError("generated bibliography contains duplicate entries")
    if (
        len(bib_keys) != EXPECTED_REFERENCE_COUNT
        or len(bbl_keys) != EXPECTED_REFERENCE_COUNT
        or set(bib_keys) != set(bbl_keys)
    ):
        raise SourceArchiveError(
            "bibliography must contain exactly "
            f"{EXPECTED_REFERENCE_COUNT} cited entries"
        )
    entry_blocks = re.findall(
        r"(?ms)^@\w+\s*\{.*?(?=^@\w+\s*\{|\Z)",
        bibliography,
    )
    missing_links = [
        key
        for key, block in zip(bib_keys, entry_blocks, strict=True)
        if re.search(r"\b(?:doi|eprint|url)\s*=", block, flags=re.IGNORECASE)
        is None
    ]
    if missing_links:
        raise SourceArchiveError(
            "every bibliography entry must provide a DOI, eprint, or stable URL: "
            f"{missing_links}"
        )
    bbl_blocks = re.findall(
        r"(?ms)\\bibitem(?:\s*\[[^\]]*\])?\s*\{([^}]+)\}"
        r"(.*?)(?=\\bibitem|\\end\{thebibliography\}|\Z)",
        generated_bibliography,
    )
    missing_rendered_links = [
        key
        for key, block in bbl_blocks
        if "\\href{" not in block and "\\url{" not in block
    ]
    if missing_rendered_links:
        raise SourceArchiveError(
            "every rendered bibliography entry must contain a hyperlink: "
            f"{missing_rendered_links}"
        )

    payloads = [
        SourcePayload(relative, _read_regular(paper_root / relative, paper_root))
        for relative in (*REQUIRED_FILES, *EXPECTED_FIGURES)
    ]
    payloads.extend(
        SourcePayload(name, _read_repository_regular(path))
        for name, path in REPOSITORY_FILES
    )
    payloads.append(SourcePayload("README.txt", _readme()))
    payloads.sort(key=lambda payload: payload.name)

    manifest = "".join(
        f"{sha256_bytes(payload.data)}  {payload.name}\n"
        for payload in payloads
    ).encode("utf-8")
    payloads.append(SourcePayload("SHA256SUMS.txt", manifest))
    payloads.sort(key=lambda payload: payload.name)
    return payloads


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_name(name), date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = FILE_MODE << 16
    return info


def build_archive(
    output: Path,
    paper_root: Path = PAPER_ROOT,
) -> dict[str, object]:
    payloads = collect_payloads(paper_root)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as bundle:
            for payload in payloads:
                bundle.writestr(_zip_info(payload.name), payload.data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    verified = verify_archive(output)
    return {
        **verified,
        "path": str(output),
    }


def verify_archive(path: Path) -> dict[str, object]:
    path = path.resolve()
    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise SourceArchiveError("archive contains duplicate members")
            if names != sorted(names):
                raise SourceArchiveError("archive members are not sorted")
            for info in infos:
                _safe_name(info.filename)
                if info.is_dir():
                    raise SourceArchiveError(
                        f"archive contains a directory member: {info.filename}"
                    )
                unix_type = (info.external_attr >> 16) & 0o170000
                if unix_type not in (0, 0o100000):
                    raise SourceArchiveError(
                        f"archive contains a non-regular member: {info.filename}"
                    )
                if info.flag_bits & 0x1:
                    raise SourceArchiveError(
                        f"archive contains an encrypted member: {info.filename}"
                    )
            expected = set(
                (
                    *REQUIRED_FILES,
                    *EXPECTED_FIGURES,
                    *GENERATED_FILES,
                    *(name for name, _ in REPOSITORY_FILES),
                )
            )
            if set(names) != expected:
                raise SourceArchiveError(
                    "archive membership differs from the sealed allowlist"
                )
            members = {name: bundle.read(name) for name in names}
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise SourceArchiveError(f"cannot read source archive: {path}") from error

    try:
        manifest_text = members["SHA256SUMS.txt"].decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceArchiveError("checksum manifest is not UTF-8") from error
    observed: dict[str, str] = {}
    for line in manifest_text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise SourceArchiveError(f"malformed checksum line: {line!r}")
        digest, name = match.groups()
        name = _safe_name(name)
        if name in observed:
            raise SourceArchiveError(f"duplicate checksum entry: {name}")
        observed[name] = digest
    expected_manifest_names = set(members) - {"SHA256SUMS.txt"}
    if set(observed) != expected_manifest_names:
        raise SourceArchiveError("checksum manifest membership mismatch")
    for name, digest in observed.items():
        if sha256_bytes(members[name]) != digest:
            raise SourceArchiveError(f"checksum mismatch: {name}")

    return {
        "sha256": sha256_file(path),
        "file_count": len(members),
        "figure_count": len(EXPECTED_FIGURES),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--paper-root", type=Path, default=PAPER_ROOT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        result = build_archive(args.output, args.paper_root)
    else:
        result = verify_archive(args.archive)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

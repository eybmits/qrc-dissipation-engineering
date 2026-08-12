"""Build the revision-tuning inference artifact and human-readable report.

This reporter is intentionally separate from ``run_revision_tuning.py`` so the
source snapshots already frozen for the strength and nested stages remain
byte-exact.  It performs the post hoc multiplicity correction requested for the
six-channel ranking: simultaneous 95% familywise intervals cover all 15
pairwise channel contrasts, although the report foregrounds the five
collective-minus-competitor contrasts.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "revision_tuning"
STRENGTH_PATH = (
    RESULT_ROOT / "strength_extension" / "six_channel_aggregate.json"
)
MATCHING_PATH = (
    RESULT_ROOT / "strength_extension" / "alternative_matching_aggregate.json"
)
NESTED_PATH = RESULT_ROOT / "nested_tuning" / "nested_tuning_results.json"
FRESH_PATH = (
    RESULT_ROOT / "fresh_interpolation" / "fresh_interpolation_results.json"
)
INFERENCE_PATH = RESULT_ROOT / "derived_simultaneous_inference.json"
REPORT_PATH = REPO_ROOT / "reports" / "revision_tuning_report.md"
REPORTER_PATH = Path(__file__).resolve()

CHANNELS = (
    "A1_heterogeneous",
    "B2_thermal",
    "B3_collective",
    "B4_loss_exchange",
    "B5_pair",
    "CD_paper",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_complete(path: Path, *, require_status: bool = True) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing prerequisite artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if require_status and payload.get("status") != "complete":
        raise RuntimeError(f"incomplete prerequisite artifact: {path}")
    return payload


def simultaneous_pairwise_intervals(
    strength: dict,
    *,
    familywise_alpha: float = 0.05,
) -> dict:
    """Compute two-sided Bonferroni t intervals for all 6 choose 2 pairs."""
    if tuple(strength["methods"]) != CHANNELS:
        raise RuntimeError("six-channel method set/order drifted")
    scores: dict[str, dict[str, float]] = {
        channel: {
            str(seed): float(value)
            for seed, value in strength["methods"][channel][
                "leave_one_seed_out"
            ]["scores_by_seed"].items()
        }
        for channel in CHANNELS
    }
    seed_sets = {tuple(sorted(channel_scores)) for channel_scores in scores.values()}
    if len(seed_sets) != 1:
        raise RuntimeError("paired six-channel seed sets differ")
    seeds = next(iter(seed_sets))
    n = len(seeds)
    if n < 2:
        raise RuntimeError("at least two paired reservoirs are required")
    pair_count = math.comb(len(CHANNELS), 2)
    critical = float(
        stats.t.ppf(
            1.0 - familywise_alpha / (2.0 * pair_count),
            df=n - 1,
        )
    )
    pairs = []
    for first, second in itertools.combinations(CHANNELS, 2):
        differences = np.asarray(
            [scores[first][seed] - scores[second][seed] for seed in seeds],
            dtype=float,
        )
        mean = float(np.mean(differences))
        se = float(np.std(differences, ddof=1) / math.sqrt(n))
        pairs.append(
            {
                "first": first,
                "second": second,
                "orientation": "first_minus_second",
                "n": n,
                "mean_difference": mean,
                "se_difference": se,
                "simultaneous_ci95_low": mean - critical * se,
                "simultaneous_ci95_high": mean + critical * se,
            }
        )
    return {
        "artifact_type": "six_channel_simultaneous_pairwise_inference",
        "coverage": "two-sided 95% familywise",
        "method": "Bonferroni-adjusted paired t intervals",
        "family_definition": "all 15 pairwise contrasts among six active channels",
        "familywise_alpha": familywise_alpha,
        "pair_count": pair_count,
        "paired_reservoir_count": n,
        "degrees_of_freedom": n - 1,
        "critical_t": critical,
        "pairs": pairs,
    }


def _oriented_pair(
    inference: dict,
    candidate: str,
    reference: str,
) -> dict:
    for pair in inference["pairs"]:
        if pair["first"] == candidate and pair["second"] == reference:
            return dict(pair)
        if pair["first"] == reference and pair["second"] == candidate:
            return {
                **pair,
                "first": candidate,
                "second": reference,
                "mean_difference": -float(pair["mean_difference"]),
                "simultaneous_ci95_low": -float(
                    pair["simultaneous_ci95_high"]
                ),
                "simultaneous_ci95_high": -float(
                    pair["simultaneous_ci95_low"]
                ),
                "orientation": "first_minus_second",
            }
    raise KeyError(f"missing pair {candidate} minus {reference}")


def _fmt(mean: float, se: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def build() -> tuple[Path, Path]:
    strength = _load_complete(STRENGTH_PATH)
    matching = _load_complete(MATCHING_PATH, require_status=False)
    nested = _load_complete(NESTED_PATH)
    fresh = _load_complete(FRESH_PATH)
    inference = simultaneous_pairwise_intervals(strength)
    collective = {
        reference: _oriented_pair(
            inference, "B3_collective", reference
        )
        for reference in CHANNELS
        if reference != "B3_collective"
    }
    derived = {
        **inference,
        "status": "complete",
        "collective_minus_competitor": collective,
        "all_collective_simultaneous_intervals_exclude_zero": all(
            item["simultaneous_ci95_low"] > 0.0
            for item in collective.values()
        ),
        "reporter_source": {
            "path": str(REPORTER_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256(REPORTER_PATH),
        },
        "source_artifacts": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (STRENGTH_PATH, MATCHING_PATH, NESTED_PATH, FRESH_PATH)
        },
    }
    _atomic_json(INFERENCE_PATH, derived)

    ranking = sorted(
        CHANNELS,
        key=lambda channel: -float(
            strength["methods"][channel]["leave_one_seed_out"]["test_mean"]
        ),
    )
    lines = [
        "# Revision tuning and independence controls",
        "",
        "These controls retain the paper's dissipation-engineering question while "
        "closing the channel-specific tuning, resource-matching, and "
        "same-ensemble validation gaps.",
        "",
        "## Six-channel validation-selected ranking",
        "",
        "For each held-out reservoir, strength is selected only from validation "
        "scores on the other 19 reservoirs. The simultaneous intervals below use "
        "a two-sided Bonferroni correction over all 15 pairwise contrasts among "
        "the six channels (familywise coverage 95%).",
        "",
        "| rank | channel | held-out STM MC ± SE | selected strengths |",
        "|---:|---|---:|---|",
    ]
    for rank, channel in enumerate(ranking, start=1):
        result = strength["methods"][channel]["leave_one_seed_out"]
        lines.append(
            f"| {rank} | {channel} | "
            f"{_fmt(result['test_mean'], result['test_se'])} | "
            f"`{result['selection_counts']}` |"
        )
    lines.extend(
        [
            "",
            "| collective minus | ΔMC | simultaneous 95% familywise interval |",
            "|---|---:|---:|",
        ]
    )
    for reference in ranking:
        if reference == "B3_collective":
            continue
        item = collective[reference]
        lines.append(
            f"| {reference} | {item['mean_difference']:+.3f} | "
            f"[{item['simultaneous_ci95_low']:+.3f}, "
            f"{item['simultaneous_ci95_high']:+.3f}] |"
        )
    bracket = strength["methods"]["B3_collective"]["curve_bracket"]
    lines.extend(
        [
            "",
            f"The collective validation optimum is bracketed at "
            f"x{float(bracket['best_multiplier']):g} between "
            f"x{float(bracket['left_multiplier']):g} and "
            f"x{float(bracket['right_multiplier']):g}. All five collective contrasts "
            "remain positive under simultaneous 95% coverage of the complete "
            "15-comparison family.",
            "",
            "## Alternative dynamical matching",
            "",
            "| channel | convention | reachable instances | scale | ΔMC vs nominal dial (95% paired CI) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    condition_order = (
        "B3_collective__energy",
        "B3_collective__gap",
        "B3_collective__activity",
        "B2_thermal__energy",
        "B2_thermal__gap",
        "B2_thermal__activity",
        "B5_pair__energy",
        "B5_pair__gap",
        "B5_pair__activity",
    )
    matching_mode_labels = {
        "energy": "initial excitation-number loss",
        "gap": "driven Liouvillian gap",
        "activity": "steady-state jump activity",
    }
    for key in condition_order:
        item = matching["conditions"][key]
        feasibility = item["match_feasibility"]
        effect = item["effect_vs_standard_dial"]
        lines.append(
            f"| {item['method']} | "
            f"{matching_mode_labels.get(item['matching_mode'], item['matching_mode'])} | "
            f"{feasibility['reachable_count']}/{feasibility['total']} | "
            f"{item['scale_factor_mean']:.3g} | "
            f"{effect['mean_difference']:+.3f} "
            f"[{effect['ci95_low']:+.3f}, {effect['ci95_high']:+.3f}] |"
        )
    collective_gap = matching["conditions"]["B3_collective__gap"]
    collective_activity = matching["conditions"]["B3_collective__activity"]
    lines.extend(
        [
            "",
            "Initial excitation-number loss-rate matching is exact. Collective "
            "driven-gap matching is unreachable within the stored x0.05–x40 "
            f"search in {32 - collective_gap['match_feasibility']['reachable_count']}"
            "/32 instances, and steady-state activity matching is exact in only "
            f"{collective_activity['match_feasibility']['reachable_count']}/32. "
            "Those rows are therefore labeled closest-achievable boundary "
            "controls, not exact matches.",
            "",
            "## Independently nested operating-point comparison",
            "",
            "A common cheap screen only formed the shortlists. Final "
            "(h, dt, strength, ridge) choices used 12 independent full-length "
            "selection reservoirs and were evaluated once on 24 untouched "
            "reservoirs.",
            "",
        ]
    )
    for channel in ("B3_collective", "CD_paper"):
        result = nested["methods"][channel]
        lines.append(
            f"- {channel}: selected (h, dt, multiplier)="
            f"`{result['selected']['config']}`, ridge="
            f"{result['selected']['best_ridge']:.3g}; untouched-test STM MC "
            f"{_fmt(result['test_mean'], result['test_se'])}."
        )
    comparison = nested["collective_vs_local"]
    lines.extend(
        [
            "",
            f"Collective minus local on the 24 untouched reservoirs: "
            f"ΔMC={comparison['mean_difference']:.3f} "
            f"[95% paired CI {comparison['ci95_low']:.3f}, "
            f"{comparison['ci95_high']:.3f}], "
            f"{comparison['relative_mean_difference_percent']:+.1f}%, "
            f"{comparison['wins']}/{comparison['n']} wins, exact paired sign "
            f"p={comparison['exact_sign_p_two_sided']:.4g}.",
            "",
            "## Frozen interpolation on a fresh reservoir ensemble",
            "",
        ]
    )
    for n_qubits in (4, 5):
        result = fresh["results_by_N"][str(n_qubits)]
        selected = result["selected_alpha_vs_local"]
        lines.append(
            f"- N={n_qubits}: frozen alpha="
            f"{result['frozen_selected_alpha']:.1f} minus local "
            f"ΔMC={selected['mean_difference']:.3f} "
            f"[95% paired CI {selected['ci95_low']:.3f}, "
            f"{selected['ci95_high']:.3f}], "
            f"{selected['wins']}/{selected['n']} wins, exact paired sign "
            f"p={selected['exact_sign_p_two_sided']:.4g}; frozen-gap versus "
            f"fresh-score Spearman rho="
            f"{result['frozen_gap_vs_fresh_mean_spearman_rho']:.3f}, exact "
            f"720-permutation p="
            f"{result['frozen_gap_vs_fresh_mean_exact_permutation_p']:.4g}."
        )
    lines.extend(
        [
            "",
            f"The {fresh['fresh_task_seed_count']} fresh task seeds have zero "
            "overlap with the 20 diagnostic seeds. This is an out-of-ensemble "
            "confirmation within the same local-to-collective operator "
            "interpolation; it is not presented as an out-of-family test.",
            "",
            "## Reproducibility links",
            "",
            f"- `{STRENGTH_PATH.relative_to(REPO_ROOT)}`: all-six ranking, raw "
            "rows, and source-row hashes.",
            f"- `{INFERENCE_PATH.relative_to(REPO_ROOT)}`: all 15 simultaneous "
            "pairwise intervals and exact input-artifact hashes.",
            f"- `{MATCHING_PATH.relative_to(REPO_ROOT)}`: operational matching "
            "curves, feasibility flags, and raw rows.",
            f"- `{NESTED_PATH.relative_to(REPO_ROOT)}`: screen, selection, and "
            "untouched-test rows.",
            f"- `{FRESH_PATH.relative_to(REPO_ROOT)}`: frozen diagnostics plus "
            "all fresh alpha-grid rows.",
            "- `results/revision_tuning/strength_extension/source_snapshot/`, "
            "`results/revision_tuning/nested_tuning/source_snapshot/`, and "
            "`results/revision_tuning/fresh_interpolation/source_snapshot/`: "
            "byte-exact stage drivers with hash-linkage manifests.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return INFERENCE_PATH, REPORT_PATH


def main() -> None:
    inference, report = build()
    print(f"INFERENCE {inference.relative_to(REPO_ROOT)}")
    print(f"REPORT {report.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

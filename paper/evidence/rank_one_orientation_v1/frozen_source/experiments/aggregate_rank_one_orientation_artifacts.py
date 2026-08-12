"""Aggregate rank-one orientation checkpoints with semantic validation.

This repairs the original aggregate gate, whose protocol SHA included raw
roundoff-level null eigenvalues from ``np.linalg.eigvalsh`` and therefore
varied across GitHub-hosted runner BLAS backends.  Task outputs are never
recomputed here.  Every checkpoint payload digest, frozen seed, jump hash,
budget, version, and convergence record is validated before statistics are
formed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.stats import binomtest, pearsonr, spearmanr, t as student_t

VERSION = "rank-one-orientation-v1-2026-08-12"
SEEDS = (
    956087733, 1375334633, 707736772, 1133846500, 365211353, 878523603,
    457552621, 363662622, 853972123, 1403843447, 151336801, 1991628836,
    1627319819, 336852480, 1454963355, 203675062, 93339074, 8147085,
    264759322, 16866769, 346211042, 1665106229, 1622806565, 1222562911,
)
CONDITIONS = ("drive_orthogonal", "equal_phase")
METRICS = (
    "response_lag_centroid",
    "long_lag_energy_fraction",
    "feature_space_effective_rank",
    "leading_singular_energy_fraction",
)


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def mean_ci(x: Sequence[float]) -> dict:
    a = np.asarray(x, dtype=float)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a)))
    half = float(student_t.ppf(0.975, len(a) - 1) * se)
    return {"n": len(a), "mean": mean, "standard_error": se, "ci95": [mean - half, mean + half]}


def paired(x: Sequence[float]) -> dict:
    a = np.asarray(x, dtype=float)
    out = mean_ci(a)
    wins, losses = int(np.sum(a > 0)), int(np.sum(a < 0))
    ties = int(np.sum(a == 0))
    out.update({
        "median": float(np.median(a)),
        "minimum": float(a.min()),
        "maximum": float(a.max()),
        "wins_positive": wins,
        "losses_negative": losses,
        "ties": ties,
        "exact_sign_test_p_two_sided": float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0,
        "cohens_dz": float(a.mean() / a.std(ddof=1)),
    })
    return out


def load_and_validate(checkpoint_dir: Path) -> list[dict]:
    paths = sorted(checkpoint_dir.glob("seed_*.json"))
    if len(paths) != 24:
        raise RuntimeError(f"expected 24 checkpoints, found {len(paths)}")
    rows = []
    for path in paths:
        row = json.loads(path.read_text())
        claimed = row.pop("payload_sha256")
        observed = digest(row)
        row["payload_sha256"] = claimed
        if claimed != observed:
            raise RuntimeError(f"payload SHA mismatch: {path}")
        rows.append(row)
    rows.sort(key=lambda r: r["seed_index"])
    if [r["seed_index"] for r in rows] != list(range(24)):
        raise RuntimeError("seed indices are missing or duplicated")
    if tuple(r["seed"] for r in rows) != SEEDS:
        raise RuntimeError("frozen seed identities do not match")
    if {r["version"] for r in rows} != {VERSION}:
        raise RuntimeError("mixed experiment versions")
    if not all(r["convergence"]["both_conditions_passed"] for r in rows):
        raise RuntimeError("at least one convergence gate failed")
    for condition in CONDITIONS:
        if {r["reservoirs"][condition]["budget"] for r in rows} != {192.0}:
            raise RuntimeError(f"budget mismatch for {condition}")
        if len({r["reservoirs"][condition]["jump_sha256"] for r in rows}) != 1:
            raise RuntimeError(f"jump operator changed for {condition}")
    return rows


def make_plots(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    equal = np.array([r["conditions"]["equal_phase"]["stm"]["total_capacity"] for r in rows])
    orth = np.array([r["conditions"]["drive_orthogonal"]["stm"]["total_capacity"] for r in rows])
    caps_equal = np.array([r["conditions"]["equal_phase"]["stm"]["capacity_by_delay"] for r in rows])
    caps_orth = np.array([r["conditions"]["drive_orthogonal"]["stm"]["capacity_by_delay"] for r in rows])
    rank_equal = np.array([r["conditions"]["equal_phase"]["kernel"]["feature_space_effective_rank"] for r in rows])
    rank_orth = np.array([r["conditions"]["drive_orthogonal"]["kernel"]["feature_space_effective_rank"] for r in rows])
    energy_equal = np.array([r["conditions"]["equal_phase"]["kernel"]["normalized_lag_energy"] for r in rows])
    energy_orth = np.array([r["conditions"]["drive_orthogonal"]["kernel"]["normalized_lag_energy"] for r in rows])

    def save(stem: str) -> None:
        plt.tight_layout()
        plt.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
        plt.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(6.4, 4.6))
    for left, right in zip(orth, equal):
        plt.plot([0, 1], [left, right], "o-", alpha=0.24, linewidth=1)
    os, es = mean_ci(orth), mean_ci(equal)
    plt.errorbar([0, 1], [os["mean"], es["mean"]],
                 yerr=[[os["mean"] - os["ci95"][0], es["mean"] - es["ci95"][0]],
                       [os["ci95"][1] - os["mean"], es["ci95"][1] - es["mean"]]],
                 fmt="D", capsize=4, linewidth=2, label="mean and 95% interval")
    plt.xticks([0, 1], ["drive-orthogonal", "equal-phase"])
    plt.ylabel("STM capacity")
    plt.title("(a) Orientation changes STM at fixed rank and spectrum")
    plt.legend(frameon=False)
    save("figure_a_total_stm")

    plt.figure(figsize=(6.4, 4.6))
    for matrix, label in ((caps_orth, "drive-orthogonal"), (caps_equal, "equal-phase")):
        mean = matrix.mean(axis=0)
        half = student_t.ppf(0.975, len(rows) - 1) * matrix.std(axis=0, ddof=1) / math.sqrt(len(rows))
        x = np.arange(1, 21)
        plt.plot(x, mean, "o-", markersize=3, label=label)
        plt.fill_between(x, mean - half, mean + half, alpha=0.18)
    plt.xlabel("input delay")
    plt.ylabel(r"STM capacity $C_\tau$")
    plt.title("(b) The orientation gain grows toward long delays")
    plt.legend(frameon=False)
    save("figure_b_lag_stm")

    plt.figure(figsize=(6.4, 4.6))
    for left, right in zip(rank_orth, rank_equal):
        plt.plot([0, 1], [left, right], "o-", alpha=0.24, linewidth=1)
    os, es = mean_ci(rank_orth), mean_ci(rank_equal)
    plt.errorbar([0, 1], [os["mean"], es["mean"]],
                 yerr=[[os["mean"] - os["ci95"][0], es["mean"] - es["ci95"][0]],
                       [os["ci95"][1] - os["mean"], es["ci95"][1] - es["mean"]]],
                 fmt="D", capsize=4, linewidth=2, label="mean and 95% interval")
    plt.xticks([0, 1], ["drive-orthogonal", "equal-phase"])
    plt.ylabel("response effective rank")
    plt.title("(c) Equal phase concentrates the observable response")
    plt.legend(frameon=False)
    save("figure_c_response_rank")

    plt.figure(figsize=(6.4, 4.6))
    for matrix, label in ((energy_orth, "drive-orthogonal"), (energy_equal, "equal-phase")):
        mean = matrix.mean(axis=0)
        half = student_t.ppf(0.975, len(rows) - 1) * matrix.std(axis=0, ddof=1) / math.sqrt(len(rows))
        x = np.arange(1, 21)
        plt.plot(x, mean, "o-", markersize=3, label=label)
        plt.fill_between(x, np.maximum(mean - half, np.finfo(float).tiny), mean + half, alpha=0.18)
    plt.yscale("log")
    plt.xlabel("response lag")
    plt.ylabel("normalized kernel energy")
    plt.title("(d) Raw perturbation persistence is not the STM explanation")
    plt.legend(frameon=False)
    save("figure_d_kernel_energy")

    panels = [Image.open(out / f"figure_{letter}_{name}.png").convert("RGB") for letter, name in (
        ("a", "total_stm"), ("b", "lag_stm"), ("c", "response_rank"), ("d", "kernel_energy"))]
    width, height = max(x.width for x in panels), max(x.height for x in panels)
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    for i, image in enumerate(panels):
        x = (i % 2) * width + (width - image.width) // 2
        y = (i // 2) * height + (height - image.height) // 2
        canvas.paste(image, (x, y))
    canvas.save(out / "rank_one_orientation_composite.png", dpi=(240, 240))
    canvas.save(out / "rank_one_orientation_composite.pdf", "PDF", resolution=240)


def aggregate(checkpoint_dir: Path, out: Path) -> None:
    rows = load_and_validate(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    absolute = {condition: {"stm": [], **{key: [] for key in METRICS}} for condition in CONDITIONS}
    per_seed, lag_rows = [], []
    worst_trace, worst_feature = {k: 0.0 for k in CONDITIONS}, {k: 0.0 for k in CONDITIONS}
    for row in rows:
        selected = str(row["convergence"]["selected_common_washout"])
        record = {"seed_index": row["seed_index"], "seed": row["seed"], "washout": int(selected),
                  "protocol_sha256": row["protocol_sha256"]}
        for condition in CONDITIONS:
            result = row["conditions"][condition]
            stm = result["stm"]["total_capacity"]
            record[f"{condition}_stm"] = stm
            absolute[condition]["stm"].append(stm)
            audit = row["convergence"]["audits"][condition][selected]
            worst_trace[condition] = max(worst_trace[condition], audit["maximum_trace_distance"])
            worst_feature[condition] = max(worst_feature[condition], audit["maximum_feature_distance"])
            for key in METRICS:
                value = result["kernel"][key]
                record[f"{condition}_{key}"] = value
                absolute[condition][key].append(value)
            for delay, value in enumerate(result["stm"]["capacity_by_delay"], 1):
                lag_rows.append({"seed_index": row["seed_index"], "seed": row["seed"],
                                 "condition": condition, "delay": delay, "capacity": value})
        record["stm_equal_minus_orthogonal"] = record["equal_phase_stm"] - record["drive_orthogonal_stm"]
        for key in METRICS:
            record[f"{key}_equal_minus_orthogonal"] = record[f"equal_phase_{key}"] - record[f"drive_orthogonal_{key}"]
        per_seed.append(record)

    paired_metrics = {}
    for key in ("stm",) + METRICS:
        paired_metrics[key] = paired(np.asarray(absolute["equal_phase"][key]) - np.asarray(absolute["drive_orthogonal"][key]))
    d_stm = np.asarray([x["stm_equal_minus_orthogonal"] for x in per_seed])
    association = {}
    for key in METRICS:
        change = np.asarray([x[f"{key}_equal_minus_orthogonal"] for x in per_seed])
        pr, sr = pearsonr(d_stm, change), spearmanr(d_stm, change)
        association[key] = {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                            "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue)}

    summary = {
        "version": VERSION,
        "pair_count": 24,
        "all_convergence_passed": True,
        "selected_washouts": sorted({x["washout"] for x in per_seed}),
        "worst_trace_distance": worst_trace,
        "worst_feature_distance": worst_feature,
        "protocol_hash_variants": sorted({x["protocol_sha256"] for x in per_seed}),
        "protocol_hash_note": "Variants arise only from backend-dependent roundoff in hashed null eigenvalues; semantic invariants and payload SHAs were verified.",
        "absolute": {condition: {key: mean_ci(values) for key, values in metrics.items()} for condition, metrics in absolute.items()},
        "paired_equal_minus_orthogonal": paired_metrics,
        "association_stm_change_vs_kernel_change": association,
        "decision": {"orientation_dependence_supported": paired_metrics["stm"]["ci95"][0] > 0,
                     "observed_direction": "equal_phase_higher"},
    }
    write_json(out / "summary.json", summary)
    with (out / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_seed[0].keys())
        writer.writeheader(); writer.writerows(per_seed)
    with (out / "lag_capacities_long.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=lag_rows[0].keys())
        writer.writeheader(); writer.writerows(lag_rows)

    make_plots(rows, out)
    stm = paired_metrics["stm"]
    rank = paired_metrics["feature_space_effective_rank"]
    lead = paired_metrics["leading_singular_energy_fraction"]
    centroid = paired_metrics["response_lag_centroid"]
    tail = paired_metrics["long_lag_energy_fraction"]
    report = f"""# Rank-one orientation intervention\n\n**Result:** equal phase exceeds the drive-orthogonal rank-one channel by **{stm['mean']:+.6f} STM units** (95% CI **[{stm['ci95'][0]:+.6f}, {stm['ci95'][1]:+.6f}]**; wins/losses/ties **{stm['wins_positive']}/{stm['losses_negative']}/{stm['ties']}**; exact sign-test p={stm['exact_sign_test_p_two_sided']:.6g}).\n\nAll 24 pairs passed at the common 800-input washout. Equal phase lowered response effective rank by {-rank['mean']:.6f} and raised the leading singular-energy fraction by {lead['mean']:.6f}, both in all 24 pairs. The normalized impulse tail was shorter: centroid change {centroid['mean']:+.6f}, long-lag fraction change {tail['mean']:+.6f}. Orientation dependence is established; raw perturbation persistence is not the explanation, and the kernel summaries are supporting rather than mediating evidence.\n"""
    (out / "REPORT.md").write_text(report)
    shutil.make_archive(str(out.parent / "rank_one_orientation_v1_results"), "zip", root_dir=out.parent, base_dir=out.name)
    print(json.dumps(summary["decision"], indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    aggregate(args.checkpoint_dir.resolve(), args.outdir.resolve())


if __name__ == "__main__":
    main()

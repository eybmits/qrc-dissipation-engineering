"""Defensible paired inference for the revised manuscript (review point 5b).

Replaces the Gaussian "sigma" language. For every paired comparison we report:
  - paired mean difference (the effect size, in task units) and Hedges' g;
  - a 95% t-interval on the paired difference;
  - Wilcoxon signed-rank p (two-sided);
  - exact sign-test p (binomial, two-sided) and the win fraction;
  - a sign-flip permutation p (exact for n<=20, else 100k Monte Carlo resamples);
  - Holm-adjusted significance across each declared comparison family.

Usage:
  PYTHONPATH=../src python review_stats.py           # regenerate the map + scaling inference
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL = os.path.join(ROOT, "results", "final_protocol")
REVIEW = os.path.join(ROOT, "results", "review_protocol")


def load(indir):
    rows = []
    for f in glob.glob(os.path.join(indir, "*.json")):
        try:
            rows.append(json.loads(open(f).read()))
        except Exception:
            pass
    return rows


def cell(rows, block, task, method, N=None, **kw):
    out = {}
    for r in rows:
        if r.get("block") != block or r.get("task") != task \
                or r.get("method") != method or r.get("value") is None:
            continue
        if N is not None and r.get("N") != N:
            continue
        if any(r.get(k) != v for k, v in kw.items()):
            continue
        out[r["seed"]] = r["value"]
    return out


def paired_inference(a: dict, b: dict, higher_better=True) -> dict:
    """Full paired inference for 'a - b' on shared seeds (positive favours a)."""
    sh = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in sh], float)
    if not higher_better:
        d = -d
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci = (mean - tcrit * se, mean + tcrit * se)
    g = mean / sd * (1 - 3 / (4 * n - 5)) if sd > 0 else np.inf  # Hedges' g
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    # Wilcoxon signed-rank (drops zeros)
    try:
        w_p = float(stats.wilcoxon(d[d != 0]).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        w_p = 1.0
    sign_p = float(stats.binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0
    # sign-flip permutation on the mean
    if n <= 20:
        signs = np.array(np.meshgrid(*([[1, -1]] * n))).T.reshape(-1, n)
        perm = signs @ d / n
        perm_p = float((np.abs(perm) >= abs(mean) - 1e-12).mean())
    else:
        rng = np.random.default_rng(12345)
        flips = rng.choice([1.0, -1.0], size=(100_000, n))
        perm = flips @ d / n
        perm_p = float(((np.abs(perm) >= abs(mean) - 1e-12).sum() + 1) / (len(perm) + 1))
    return dict(n=n, mean=mean, ci_lo=ci[0], ci_hi=ci[1], hedges_g=g,
                wins=wins, win_frac=wins / n if n else np.nan,
                wilcoxon_p=w_p, sign_p=sign_p, perm_p=perm_p)


def holm(pvals: dict, alpha=0.05) -> dict:
    """Holm-Bonferroni over a family; returns {key: (p_adj, significant)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max = {}, 0.0
    for rank, (k, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, p_adj)
        out[k] = (running_max, running_max < alpha)
    return out


def fmt(r, unit=""):
    return (f"Δ={r['mean']:+.3f}{unit} [95% CI {r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
            f"g={r['hedges_g']:.1f} wins={r['wins']}/{r['n']} "
            f"Wilcoxon p={r['wilcoxon_p']:.2e} perm p={r['perm_p']:.2e}")


HIGHER = {"stm": True, "narma": False, "parity": True, "mg": False}
METHODS = ["FN", "B3_collective", "A1_heterogeneous", "B5_pair", "B2_thermal",
           "B4_loss_exchange", "B1_dephasing"]


def main():
    rows = load(FINAL)
    # forecasting: the leak-free R_mgfix rerun supersedes the original A_table
    # mg cells (normalisation from the train prefix only)
    rows = [r for r in rows
            if not (r.get("block") == "A_table" and r.get("task") == "mg")]
    for r in load(REVIEW):
        if r.get("block") == "R_mgfix":
            r2 = dict(r)
            r2["block"] = "A_table"
            rows.append(r2)
    print(f"final_protocol rows (mg from R_mgfix): {len(rows)}")

    print("\n=== THE MAP (N=5, vs the dial) — full paired inference ===")
    family = {}
    results = {}
    for task in ("stm", "narma", "parity", "mg"):
        dial = cell(rows, "A_table", task, "CD_paper", N=5)
        for m in METHODS:
            d = cell(rows, "A_table", task, m, N=5)
            if not d:
                continue
            r = paired_inference(d, dial, HIGHER[task])
            key = f"{task}:{m}"
            results[key] = r
            family[key] = r["perm_p"]
    adj = holm(family)
    for task in ("stm", "narma", "parity", "mg"):
        print(f"\n  {task.upper()}")
        for m in METHODS:
            key = f"{task}:{m}"
            if key in results:
                pa, sig = adj[key]
                mark = "*" if sig else " "
                print(f"   {mark} {m:18s} {fmt(results[key])}  Holm p={pa:.2e}")

    print("\n=== SCALING (collective vs dial, per N) ===")
    for N in (4, 5, 6, 7, 8):
        a = cell(rows, "B_scale", "stm", "B3_collective", N=N)
        b = cell(rows, "B_scale", "stm", "CD_paper", N=N)
        if a and b:
            print(f"   N={N}: {fmt(paired_inference(a, b, True))}")

    print("\n=== PROFILE LADDER (block F, vs uniform) ===")
    uni = cell(rows, "F_adaptive", "stm", "A0_uniform", N=5)
    for m in ("A1_random", "A3_learned", "A4_adaptive"):
        d = cell(rows, "F_adaptive", "stm", m, N=5)
        if d:
            print(f"   {m:12s} {fmt(paired_inference(d, uni, True))}")

    rrows = load(REVIEW)
    if rrows:
        print(f"\nreview_protocol rows: {len(rrows)}")

    def rcell(block, method, **match):
        out = {}
        for r in rrows:
            if r.get("block") != block or r.get("method") != method:
                continue
            if any(r.get(k) != v for k, v in match.items()):
                continue
            if r.get("value") is None:
                continue
            out[r["seed"]] = r["value"]
        return out

    print("\n=== R_lenfix (train-only length control, N=6, coll vs dial) ===")
    for tr in (150, 300, 600, 1200, 1800):
        a = rcell("R_lenfix", "B3_collective", train_len=tr)
        b = rcell("R_lenfix", "CD_paper", train_len=tr)
        if a and b:
            r = paired_inference(a, b, True)
            gain = 100 * (sum(a.values())/len(a) - sum(b.values())/len(b)) / (sum(b.values())/len(b))
            print(f"   train={tr:5d}: {fmt(r)}  (+{gain:.0f}%)")

    print("\n=== R_shots (finite-shot STM, N=5, coll vs dial) ===")
    for s in (256, 1024, 4096, 0):
        a = rcell("R_shots", "B3_collective", shots=s)
        b = rcell("R_shots", "CD_paper", shots=s)
        if a and b:
            print(f"   shots={s:5d}: {fmt(paired_inference(a, b, True))}")

    print("\n=== R_optfix (equal-budget profiles, vs uniform block-F baseline) ===")
    uniF = cell(load(FINAL), "F_adaptive", "stm", "A0_uniform", N=5)
    for m in ("A3_eq", "A4_eq"):
        d = rcell("R_optfix", m)
        if d and uniF:
            print(f"   {m:8s} {fmt(paired_inference(d, uniF, True))}")

    print("\n=== R_match2 (steady-state-activity-matched budget, vs standard dial) ===")
    dialA = cell(rows, "A_table", "stm", "CD_paper", N=5)
    for m in ("B3_collective", "B5_pair", "B2_thermal"):
        a = rcell("R_match2", m)
        if a and dialA:
            print(f"   {m:16s} {fmt(paired_inference(a, dialA, True))}")


if __name__ == "__main__":
    main()

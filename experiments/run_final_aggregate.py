"""Aggregate results/final_protocol/*.json into the paper's summary tables.

Robust to partial (still-running) results: every cell reports its seed count n.
Paired significance (sigma vs CD_paper) uses only seeds present for BOTH methods.
Run any time:  PYTHONPATH=../src python run_final_aggregate.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

import _paths  # noqa: F401
from _paths import RESULTS_DIR

OUTDIR = os.path.join(RESULTS_DIR, "final_protocol")


def load():
    rows = []
    for f in glob.glob(os.path.join(OUTDIR, "*.json")):
        try:
            rows.append(json.loads(open(f).read()))
        except (json.JSONDecodeError, OSError):
            continue
    return rows


def paired_sigma(vals_m: dict, vals_cd: dict, higher_better: bool):
    """Paired mean-diff / SE over shared seeds. Returns (mean_m, se_m, n, sigma, winrate)."""
    m = np.array([vals_m[s] for s in sorted(vals_m)])
    se = m.std(ddof=1) / np.sqrt(len(m)) if len(m) > 1 else float("nan")
    shared = sorted(set(vals_m) & set(vals_cd))
    if len(shared) < 2:
        return m.mean(), se, len(m), float("nan"), float("nan")
    a = np.array([vals_m[s] for s in shared])
    c = np.array([vals_cd[s] for s in shared])
    d = (a - c) if higher_better else (c - a)
    sig = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
    win = float(np.mean(d > 0))
    return m.mean(), se, len(m), sig, win


def collect(rows, block, task):
    """method -> {seed: value} for a block+task (value scalar)."""
    out = defaultdict(dict)
    for r in rows:
        if r.get("block") == block and r.get("task") == task and r.get("value") is not None:
            out[r["method"]][r["seed"]] = float(r["value"])
    return out


HIGHER = {"stm": True, "narma": False, "parity": True, "mg": False}
UNIT = {"stm": "MC↑", "narma": "NMSE↓", "parity": "cap↑", "mg": "MSE↓"}


def table_block_A(rows):
    print("\n" + "=" * 72)
    print("RUN A — dissociation table (N=5), mean±SE, paired σ vs CD_paper, win%")
    for task in ("stm", "narma", "parity", "mg"):
        data = collect(rows, "A_table", task)
        if not data:
            continue
        cd = data.get("CD_paper", {})
        print(f"\n  {task.upper()} ({UNIT[task]})")
        for m in sorted(data):
            mean, se, n, sig, win = paired_sigma(data[m], cd, HIGHER[task])
            tag = "" if (m == "CD_paper" or np.isnan(sig)) else f"  {sig:+.1f}σ  win={100*win:.0f}%"
            print(f"    {m:18s} {mean:8.3f} ± {se:6.3f}  (n={n}){tag}")


def scaling_block_B(rows):
    print("\n" + "=" * 72)
    print("RUN B — memory scaling, per N: mean±SE, σ vs CD, collective %gain")
    for task in ("stm", "narma"):
        print(f"\n  {task.upper()} ({UNIT[task]})")
        gains = {}
        for N in (4, 5, 6, 7, 8):
            data = {m: {s: v for s, v in d.items()}
                    for m, d in collect(rows, "B_scale", task).items()}
            # restrict to this N
            dN = defaultdict(dict)
            for r in rows:
                if r.get("block") == "B_scale" and r.get("task") == task \
                        and r.get("N") == N and r.get("value") is not None:
                    dN[r["method"]][r["seed"]] = float(r["value"])
            if not dN:
                continue
            cd = dN.get("CD_paper", {})
            print(f"    N={N}")
            for m in sorted(dN):
                mean, se, n, sig, win = paired_sigma(dN[m], cd, HIGHER[task])
                tag = "" if (m == "CD_paper" or np.isnan(sig)) else f"  {sig:+.1f}σ win={100*win:.0f}%"
                if m == "B3_collective" and cd:
                    cmean = np.mean([cd[s] for s in cd])
                    gains[N] = 100 * (mean - cmean) / cmean * (1 if HIGHER[task] else -1)
                print(f"      {m:18s} {mean:8.3f} ± {se:6.3f} (n={n}){tag}")
        if len(gains) >= 2:
            Ns = sorted(gains)
            slope = np.polyfit(Ns, [gains[N] for N in Ns], 1)[0]
            print(f"    collective %gain vs N: {{{', '.join(f'{N}:{gains[N]:+.0f}%' for N in Ns)}}}"
                  f"  slope={slope:+.1f}%/qubit")


def control_block_C(rows):
    print("\n" + "=" * 72)
    print("RUN C — sequence-length control (N=6, STM): collective advantage vs length")
    by_len = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("block") == "C_seqlen" and r.get("value") is not None:
            L = r["wash"] + r["train"] + r["test"]
            by_len[L][r["method"]][r["seed"]] = float(r["value"])
    for L in sorted(by_len):
        cd = by_len[L].get("CD_paper", {})
        b3 = by_len[L].get("B3_collective", {})
        if not cd or not b3:
            continue
        mean, se, n, sig, win = paired_sigma(b3, cd, True)
        cmean = np.mean([cd[s] for s in cd])
        gain = 100 * (mean - cmean) / cmean
        print(f"    L={L:5d}  CD={cmean:6.2f}  B3={mean:6.2f}  Δ={gain:+.0f}%  {sig:+.1f}σ (n={n})")


def robust_block_D(rows):
    print("\n" + "=" * 72)
    print("RUN D — operating-point robustness (N=5): dissociation across (h,dt)")
    pts = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        if r.get("block") == "D_oppoint" and r.get("value") is not None:
            pts[(r["h"], r["dt"])][r["task"]][r["method"]][r["seed"]] = float(r["value"])
    for hp in sorted(pts):
        print(f"    (h,dt)={hp}")
        for task in ("stm", "parity"):
            d = pts[hp].get(task, {})
            cd = d.get("CD_paper", {})
            for m in sorted(d):
                if m == "CD_paper":
                    continue
                mean, se, n, sig, win = paired_sigma(d[m], cd, HIGHER[task])
                print(f"      {task:6s} {m:16s} {mean:7.3f}±{se:5.3f} {sig:+.1f}σ (n={n})")


def diag_block_E(rows):
    print("\n" + "=" * 72)
    print("RUN E — diagnostics vs N (mean over seeds): unitality / gap / mixing / sep01")
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        if r.get("block") == "E_diag" and r.get("diagnostics"):
            for k, v in r["diagnostics"].items():
                if isinstance(v, (int, float)):
                    agg[r["N"]][r["method"]][k].append(v)
    for N in sorted(agg):
        print(f"    N={N}")
        for m in sorted(agg[N]):
            g = agg[N][m]
            def mv(k):
                return f"{np.mean(g[k]):.3g}" if g.get(k) else "  -  "
            print(f"      {m:18s} unit={mv('unitality_defect'):>7} gap={mv('spectral_gap'):>7} "
                  f"tmix={mv('mixing_time'):>7} sep={mv('separability_01'):>7}")


def main():
    rows = load()
    print(f"loaded {len(rows)} result files from {OUTDIR}")
    by_block = defaultdict(int)
    for r in rows:
        by_block[r.get("block")] += 1
    print("by block:", dict(by_block))
    table_block_A(rows)
    scaling_block_B(rows)
    control_block_C(rows)
    robust_block_D(rows)
    diag_block_E(rows)


if __name__ == "__main__":
    main()

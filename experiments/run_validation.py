"""Drive the Phase-0 baseline validation and save raw results to results/.

Run from the experiments/ directory:
    python run_validation.py            # full validation preset
    python run_validation.py --quick    # faster, coarser (smoke test)

This is the reduced-scale replication used in the report.  For the paper-scale
run (N=5, washout/train/test=1000, 100 realizations) pass --preset paper, which
is intended for a cluster.
"""
from __future__ import annotations

import _paths  # noqa: F401
import argparse
import time

import experiments_baseline as eb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="validation")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tasks", nargs="+",
                    default=["stm", "narma", "parity", "mackey_glass"])
    args = ap.parse_args()

    overrides = {}
    if args.quick:
        args.preset = "demo"
    cfg = eb.make_config(args.preset, **overrides)
    print(f"=== baseline validation | preset={args.preset} ===")
    print(cfg)

    t0 = time.time()
    if "stm" in args.tasks:
        print("\n--- STM (gamma sweep) ---")
        eb.run_stm(cfg, gammas=[0.05, 0.1, 0.3, 1.0, 3.0],
                   delays=list(range(1, 16)))
    if "narma" in args.tasks:
        print("\n--- NARMA ---")
        eb.run_narma(cfg, gammas=[0.1, 0.3, 1.0], orders=[2, 5, 10],
                     input_scale=0.2)
    if "parity" in args.tasks:
        print("\n--- parity (V=15) ---")
        eb.run_parity(cfg, gammas=[0.3, 1.0], delays=list(range(1, 7)),
                      n_virtual=15)
    if "mackey_glass" in args.tasks:
        print("\n--- Mackey-Glass autonomous ---")
        eb.run_mackey_glass(cfg, gammas=[0.3, 1.0], horizon=150)
    print(f"\n=== all done in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()

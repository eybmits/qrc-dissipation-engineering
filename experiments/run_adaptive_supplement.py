"""Supplementary protocol block F: adaptive rate engineering at protocol scale.

Upgrades the 3-seed A3/A4 indication (run_track_a_learned.py) to 32 paired seeds,
using the SAME protocol seed set as run_final_scaling.py so results are comparable.
One checkpoint JSON per (method, seed) in results/final_protocol/ (block F_adaptive);
skip-if-exists per seed; a failing seed never kills the run.

Methods (STM, lossy regime gbar=1.5 where rate shaping matters, matched MEAN rate):
  A0_uniform    gamma_i = gbar                        (1 param, the Sannia dial)
  A1_random     log-uniform spread, mean gbar         (N params, blind heterogeneity)
  A3_learned    softplus(theta_i), shape optimised on a validation split (N params)
  A4_adaptive   softplus(a_i s + b_i), input-dependent rates (2N params)

Run:  cd experiments && PYTHONPATH=../src python run_adaptive_supplement.py --workers 8
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import _paths  # noqa: F401
import numpy as np
from scipy.optimize import minimize

from _paths import RESULTS_DIR
from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.reservoirs import (GeneralLindbladReservoir, ising_xx_hamiltonian,
                            transverse_drive)
from run_final_scaling import deterministic_seeds

N = 5
H, DT, GBAR, QUANT = 0.5, 0.5, 1.5, 32
WASH, TRAIN, TEST, VAL = 100, 250, 200, 150   # VAL carved from the tail of TRAIN
DELAYS = list(range(1, 13))
RIDGE = 1e-8
OUTDIR = Path(RESULTS_DIR) / "final_protocol"
METHODS = ("A0_uniform", "A1_random", "A3_learned", "A4_adaptive")

obs = readout.pauli_observables(N, max_weight=2)


def mc_on(X, post, which):
    Xb = readout.add_bias(X)
    tr = slice(0, TRAIN - VAL)
    block = {"val": slice(TRAIN - VAL, TRAIN),
             "test": slice(TRAIN, TRAIN + TEST)}[which]
    mc = 0.0
    for d in DELAYS:
        y = tasks.delayed_target(post, d)
        a = np.zeros(len(y), bool); a[tr] = True; a &= ~np.isnan(y)
        b = np.zeros(len(y), bool); b[block] = True; b &= ~np.isnan(y)
        w = readout.train_readout(Xb[a], y[a], ridge=RIDGE)
        mc += readout.capacity(y[b], readout.predict(Xb[b], w))
    return mc


def static_mc(gammas, J, inputs, post, which):
    r = res.continuous_dissipation(N, J, H, gammas, DT, quantize=QUANT)
    X = r.run(inputs, obs, washout=WASH, n_virtual=1)
    return mc_on(X, post, which)


def adaptive_mc(a, b, J, inputs, post, which):
    H0 = ising_xx_hamiltonian(J, H, N); Hx = transverse_drive(N)
    sgrid = np.linspace(0, 1, 11)
    mbar = np.mean([dsp.adaptive_rates(s, a, b).mean() for s in sgrid])
    k = GBAR / max(mbar, 1e-6)
    r = GeneralLindbladReservoir(
        N,
        lambda s: H0 + H * (s + 1) * Hx,
        lambda s: [(res.sminus(i, N), float(k * dsp.adaptive_rates(s, a, b)[i]))
                   for i in range(N)],
        DT, quantize=QUANT)
    X = r.run(inputs, obs, washout=WASH, n_virtual=1)
    return mc_on(X, post, which)


def out_path(method, seed):
    return OUTDIR / f"F_adaptive__stm_N{N}_{method}_s{seed}.json"


def write_ckpt(method, seed, value, t0):
    payload = {"block": "F_adaptive", "N": N, "method": method, "task": "stm",
               "seed": int(seed), "h": H, "dt": DT, "wash": WASH, "train": TRAIN,
               "test": TEST, "value": float(value), "gbar": GBAR,
               "runtime_s": time.time() - t0, "backend": "dense_quantized"}
    p = out_path(method, seed)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, p)


def run_seed(seed: int) -> str:
    if all(out_path(m, seed).exists() for m in METHODS):
        return f"skip s={seed}"
    t0 = time.time()
    try:
        rng = np.random.default_rng(seed)
        J = res.random_couplings(N, 1.0, rng)
        total = WASH + TRAIN + TEST
        inputs = tasks.stm_inputs(total, rng)
        post = inputs[WASH:]

        g_unif = dsp.uniform_rates(N, GBAR)
        g_rand = dsp.loguniform_rates(N, GBAR / 10, GBAR * 10, rng, mean=GBAR)
        write_ckpt("A0_uniform", seed, static_mc(g_unif, J, inputs, post, "test"), t0)
        write_ckpt("A1_random", seed, static_mc(g_rand, J, inputs, post, "test"), t0)

        def negval_A3(theta):
            g = dsp.normalize_rates(dsp.rates_from_theta(theta), GBAR)
            return -static_mc(g, J, inputs, post, "val")
        starts = [np.zeros(N), np.log(np.expm1(np.clip(g_rand, 1e-3, None)))]
        best = None
        for th0 in starts:
            sol = minimize(negval_A3, th0, method="Nelder-Mead",
                           options=dict(maxfev=35, xatol=1e-2, fatol=1e-3))
            if best is None or sol.fun < best.fun:
                best = sol
        g_learned = dsp.normalize_rates(dsp.rates_from_theta(best.x), GBAR)
        write_ckpt("A3_learned", seed, static_mc(g_learned, J, inputs, post, "test"), t0)

        def negval_A4(ab):
            return -adaptive_mc(ab[:N], ab[N:], J, inputs, post, "val")
        ab0 = np.concatenate([rng.standard_normal(N) * 0.5, np.full(N, 0.5)])
        sol2 = minimize(negval_A4, ab0, method="Nelder-Mead",
                        options=dict(maxfev=50, xatol=1e-2, fatol=1e-3))
        write_ckpt("A4_adaptive", seed,
                   adaptive_mc(sol2.x[:N], sol2.x[N:], J, inputs, post, "test"), t0)
        return f"done s={seed} ({time.time()-t0:.0f}s)"
    except Exception as exc:  # noqa: BLE001
        return f"error s={seed}: {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=32)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = deterministic_seeds(args.seeds)
    print(f"block F_adaptive: {len(seeds)} seeds, workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_seed, int(s)): s for s in seeds}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"worker died s={futs[f]}: {exc}", flush=True)
    print("F_adaptive COMPLETE", flush=True)


if __name__ == "__main__":
    main()

"""Block G: the joint-axis experiment --- do the two engineering axes stack?

The paper studies the two axes separately: loss channels at a uniform profile, and
damping profiles on the local channel. This block runs the 2x2 cross-product to test
whether the two best levers combine:

    channel in {local, collective}  x  profile in {uniform, learned}

plus collective-with-unequal-coefficients as a sanity rung. All at matched total
dissipation, in the same strong-damping regime as block F (where profile-learning is
known to help), using the exact sparse evolver. "Learned" optimises the per-qubit
weights (local rates gamma_i, or collective coefficients c_i in L=sum_i c_i sigma^-_i)
on a validation split; scored on held-out test data.

Question: is (collective + learned) > (collective + uniform) and > (local + learned)?
If not, the collective structure already captures the benefit and the axes do not
stack --- which is itself a clean result and matches the paper's "distinct" framing.

Run:  cd experiments && PYTHONPATH=../src python run_joint_axis.py --workers 8
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
from qrc.operators import sminus
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir
from run_final_scaling import deterministic_seeds

N = 5
H, DT, GBAR = 0.5, 0.5, 1.5          # strong-damping regime (profiles matter here)
WASH, TRAIN, TEST, VAL = 100, 250, 200, 150
DELAYS = list(range(1, 13))
RIDGE = 1e-8
OUTDIR = Path(RESULTS_DIR) / "final_protocol"
# (channel, profile): the 2x2 + one sanity rung
METHODS = ("G_local_uniform", "G_local_learned",
           "G_coll_uniform", "G_coll_learned", "G_coll_unequal")
obs = readout.pauli_observables(N, max_weight=2)


def build_reservoir(channel, weights, J, target):
    H0 = ising_xx_hamiltonian(J, H, N)
    Hx = transverse_drive(N)
    if channel == "local":
        jumps = dsp.normalize_jump_strength(
            [(sminus(i, N), float(weights[i])) for i in range(N)], target)
    else:  # collective: single jump L = sum_i c_i sigma^-_i
        jumps = dsp.normalize_jump_strength(
            dsp.collective_loss(N, 1.0, c=np.asarray(weights, float)), target)
    return SparseLindbladReservoir.from_terms(N, H0 + H * Hx, H * Hx, jumps, DT)


def stm_mc(channel, weights, J, target, inputs, post, which):
    r = build_reservoir(channel, weights, J, target)
    X = r.run(inputs, obs, washout=WASH)
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


def learn_weights(channel, J, target, inputs, post, rng):
    """Optimise per-qubit weights on the validation split (Nelder-Mead, restarts)."""
    def neg_val(theta):
        return -stm_mc(channel, dsp.softplus(theta), J, target, inputs, post, "val")
    g_rand = dsp.loguniform_rates(N, 0.1, 10, rng, mean=1.0)
    starts = [np.zeros(N), np.log(np.expm1(np.clip(g_rand, 1e-3, None)))]
    best = None
    for th0 in starts:
        sol = minimize(neg_val, th0, method="Nelder-Mead",
                       options=dict(maxfev=40, xatol=1e-2, fatol=1e-3))
        if best is None or sol.fun < best.fun:
            best = sol
    return dsp.softplus(best.x)


def out_path(method, seed):
    return OUTDIR / f"G_joint__stm_N{N}_{method}_s{seed}.json"


def write_ckpt(method, seed, value, t0):
    payload = {"block": "G_joint", "N": N, "method": method, "task": "stm",
               "seed": int(seed), "h": H, "dt": DT, "gbar": GBAR,
               "value": float(value), "runtime_s": time.time() - t0,
               "backend": "sparse"}
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
        inputs = tasks.stm_inputs(total, rng); post = inputs[WASH:]
        target = dsp.jump_strength(dsp.local_loss(N, GBAR))
        ones = np.ones(N)
        c_rand = dsp.loguniform_rates(N, 0.3, 3.0, rng, mean=1.0)

        write_ckpt("G_local_uniform", seed,
                   stm_mc("local", ones, J, target, inputs, post, "test"), t0)
        write_ckpt("G_coll_uniform", seed,
                   stm_mc("collective", ones, J, target, inputs, post, "test"), t0)
        write_ckpt("G_coll_unequal", seed,
                   stm_mc("collective", c_rand, J, target, inputs, post, "test"), t0)
        w_loc = learn_weights("local", J, target, inputs, post,
                              np.random.default_rng(seed + 1))
        write_ckpt("G_local_learned", seed,
                   stm_mc("local", w_loc, J, target, inputs, post, "test"), t0)
        w_col = learn_weights("collective", J, target, inputs, post,
                              np.random.default_rng(seed + 2))
        write_ckpt("G_coll_learned", seed,
                   stm_mc("collective", w_col, J, target, inputs, post, "test"), t0)
        return f"done s={seed} ({time.time()-t0:.0f}s)"
    except Exception as exc:  # noqa: BLE001
        return f"error s={seed}: {type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=24)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = deterministic_seeds(args.seeds)
    print(f"block G_joint: {len(seeds)} seeds, workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_seed, int(s)): s for s in seeds}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"worker died s={futs[f]}: {exc}", flush=True)
    print("G_joint COMPLETE", flush=True)


if __name__ == "__main__":
    main()

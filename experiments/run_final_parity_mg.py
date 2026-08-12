"""Extend the robust 16-seed final table to Parity and Mackey-Glass.

Same 16 paired seeds as run_final_table.py (rng 777). STM was a linear task scored
at each method's linear (STM) optimum; Parity and Mackey-Glass are nonlinear, so we
score them at each method's nonlinear (NARMA) optimum from the grid. FN included.

Usage:  python run_final_parity_mg.py parity   |   python run_final_parity_mg.py mg
"""
import _paths  # noqa: F401
import json
import sys
import time
import numpy as np

from qrc import reservoirs as res, readout, tasks, dissipators as dsp
from experiments_track_b import make_reservoir_with_jumps
from run_best_vs_best import methods_at_strength, METHODS

N_REAL = 16
QUANT = 28
WASH, TRAIN, TEST = 120, 380, 250
H = 0.5
PAR_DELAYS = list(range(1, 8))
NVIRT = 15
HORIZON = 150
SAMPLE_EVERY = 3


def _cfg(dt):
    return type("C", (), dict(n_qubits=5, h=H, dt=dt, quantize=QUANT))()


def parity_cap(reservoir, par_in, post, obs):
    X = reservoir.run(par_in, obs, washout=WASH, n_virtual=NVIRT)
    Xb = readout.add_bias(X); tr = slice(0, TRAIN); te = slice(TRAIN, TRAIN + TEST)
    tot = 0.0
    for d in PAR_DELAYS:
        y = tasks.parity_target(post, d)
        a = np.zeros(len(y), bool); a[tr] = True; a &= ~np.isnan(y)
        b = np.zeros(len(y), bool); b[te] = True; b &= ~np.isnan(y)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        tot += readout.capacity(y[b], readout.predict(Xb[b], w))
    return tot


def mg_mse(reservoir, J, obs, series, n_train):
    drive = series[:n_train]
    X = reservoir.run(drive, obs, washout=WASH, n_virtual=1)
    post = drive[WASH:]; y = np.empty(len(post)); y[:-1] = post[1:]; y[-1] = series[n_train]
    w = readout.train_readout(readout.add_bias(X)[:-1], y[:-1], ridge=1e-8)
    rho = reservoir.initial_state()
    for s in drive:
        rho = reservoir.step(rho, float(s))
    preds = []
    for _ in range(HORIZON):
        feat = readout.add_bias(readout.features_from_states([rho], obs))[0]
        cur = float(np.clip(feat @ w, 0, 1)); preds.append(cur); rho = reservoir.step(rho, cur)
    truth = series[n_train:n_train + HORIZON]
    return float(np.mean((truth - np.array(preds)) ** 2))


def main(task):
    bb = json.load(open("../results/best_vs_best.json"))["results"]
    n = 5
    obs = readout.pauli_observables(n, max_weight=2)
    total = WASH + TRAIN + TEST
    seeds = np.random.default_rng(777).integers(0, 2 ** 31 - 1, N_REAL)
    models = ["FN (normal/unitary)"] + METHODS
    val = {m: np.zeros(N_REAL) for m in models}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(n, 1.0, rng)
        if task == "parity":
            par_in = tasks.parity_inputs(total, rng); post = par_in[WASH:]
            fn = res.FujiNakajimaReservoir(n, J, H, 0.5)
            val["FN (normal/unitary)"][r_i] = parity_cap(fn, par_in, post, obs)
        else:
            n_train = WASH + TRAIN
            series = tasks.mackey_glass_series(n_train + HORIZON + 5, tau=17,
                                               sample_every=SAMPLE_EVERY, discard=500, rng=rng)
            series = tasks.rescale_unit(series, 0, 1)
            fn = res.FujiNakajimaReservoir(n, J, H, 0.25)
            val["FN (normal/unitary)"][r_i] = mg_mse(fn, J, obs, series, n_train)
        for m in METHODS:
            ndt, nsg = bb[m]["narma_best_dt_sgamma"]
            fams = methods_at_strength(J, n, dsp.jump_strength(dsp.local_loss(n, nsg)))
            r = make_reservoir_with_jumps(J, _cfg(ndt), fams[m])
            if task == "parity":
                r.cache_propagators = True
                val[m][r_i] = parity_cap(r, par_in, post, obs)
            else:
                val[m][r_i] = mg_mse(r, J, obs, series, n_train)
        print(f"  [{task}] seed {r_i+1}/{N_REAL} done ({time.time()-t0:.0f}s)")

    out = {"task": task, "models": models, "n_real": N_REAL,
           "value": {m: val[m].tolist() for m in models},
           "seeds": seeds.tolist(), "runtime_s": time.time() - t0}
    json.dump(out, open(f"../results/final_{task}.json", "w"), indent=2)
    higher = task == "parity"
    cd = np.array(val["CD_paper (uniform local)"])
    print(f"\n=== final {task} (N={N_REAL}) {'capacity↑' if higher else 'MSE↓'} ===")
    for m in models:
        a = np.array(val[m]); se = a.std(ddof=1) / np.sqrt(N_REAL)
        diff = (a - cd) if higher else (cd - a)
        sig = diff.mean() / (diff.std(ddof=1) / np.sqrt(N_REAL) + 1e-12)
        print(f"  {m:28s} {a.mean():8.3f} ± {se:.3f}   σ/CD={sig:+.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "parity")

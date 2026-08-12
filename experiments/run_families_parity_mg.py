"""Extend Track A/B families to the Parity and Mackey-Glass tasks.

Parity (time-multiplexed, V=15) and Mackey-Glass (autonomous forecast) were only
run for the baseline.  Here we compare the dissipator families (local loss,
collective, heterogeneous rates, loss+exchange) on them, at matched total
dissipative strength.
"""
import _paths  # noqa: F401
import json
import sys
import time
import numpy as np

from qrc import reservoirs as res, readout, tasks
from qrc import dissipators as dsp
from experiments_baseline import make_config, evaluate_delays
from experiments_track_b import make_reservoir_with_jumps


def families(J, n, gamma, include_exchange=False):
    target = dsp.jump_strength(dsp.local_loss(n, gamma))
    rng = np.random.default_rng(int(abs(J).sum() * 1e6) % (2**31))
    fam = {
        "local": dsp.local_loss(n, gamma),
        "collective": dsp.normalize_jump_strength(dsp.collective_loss(n, gamma), target),
        "hetero": [(res.sminus(i, n), float(g)) for i, g in
                   enumerate(dsp.loguniform_rates(n, gamma/10, gamma*10, rng, mean=gamma))],
    }
    if include_exchange:
        edges = dsp.graph_edges(J)
        loss = dsp.normalize_jump_strength(dsp.local_loss(n, 1.0), 0.4 * target)
        ex = dsp.normalize_jump_strength(dsp.exchange(n, 1.0, edges), 0.6 * target)
        fam["loss+exchange"] = loss + ex
    return fam


# ---------------------------------------------------------------- parity
def run_parity():
    cfg = make_config("validation", h=0.5, dt=0.5, quantize=48,
                      n_realizations=4, washout=150, train_len=450, test_len=300)
    gamma = 1.0; nvirt = 15; delays = list(range(1, 8))
    fam_names = ["local", "collective", "hetero"]
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)
    seeds = np.random.default_rng(cfg.seed + 900).integers(0, 2**31-1, cfg.n_realizations)
    cap = {m: np.zeros((cfg.n_realizations, len(delays))) for m in fam_names}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd)); J = res.random_couplings(cfg.n_qubits, 1.0, rng)
        inputs = tasks.parity_inputs(cfg.total_len, rng); post = inputs[cfg.washout:]
        fam = families(J, cfg.n_qubits, gamma)
        for m in fam_names:
            r = make_reservoir_with_jumps(J, cfg, fam[m]); r.cache_propagators = True
            X = r.run(inputs, obs, washout=cfg.washout, n_virtual=nvirt)
            caps = evaluate_delays(X, post, tasks.parity_target, delays, cfg, "capacity")
            cap[m][r_i] = [caps[d] for d in delays]
        print(f"  [parity] realization {r_i+1}/{cfg.n_realizations} ({time.time()-t0:.0f}s)")
    out = {"task": "parity_families", "delays": delays, "n_virtual": nvirt, "gamma": gamma,
           "families": fam_names, "capacity": {m: cap[m].tolist() for m in fam_names},
           "seeds": seeds.tolist(), "config": cfg.__dict__, "runtime_s": time.time()-t0}
    json.dump(out, open("../results/parity_families.json", "w"), indent=2)
    print("=== parity capacity (mean) ===")
    for m in fam_names:
        print(f"  {m:14s}", " ".join(f"{v:.2f}" for v in cap[m].mean(0)))


# ---------------------------------------------------------------- mackey-glass
def run_mg():
    cfg = make_config("validation", h=0.5, dt=0.25, quantize=48,
                      n_realizations=4, washout=150, train_len=450, test_len=0)
    gamma = 0.3; horizon = 150; sample_every = 3
    fam_names = ["local", "collective", "hetero", "loss+exchange"]
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)
    seeds = np.random.default_rng(cfg.seed + 901).integers(0, 2**31-1, cfg.n_realizations)
    n_train = cfg.washout + cfg.train_len
    mse = {m: np.zeros(cfg.n_realizations) for m in fam_names}
    example = {m: None for m in fam_names}; truth_ex = None
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd)); J = res.random_couplings(cfg.n_qubits, 1.0, rng)
        series = tasks.mackey_glass_series(n_train+horizon+5, tau=17,
                                           sample_every=sample_every, discard=500, rng=rng)
        series = tasks.rescale_unit(series, 0, 1); drive = series[:n_train]
        fam = families(J, cfg.n_qubits, gamma, include_exchange=True)
        for m in fam_names:
            r = make_reservoir_with_jumps(J, cfg, fam[m])
            X = r.run(drive, obs, washout=cfg.washout, n_virtual=1)
            post = drive[cfg.washout:]
            y = np.empty(len(post)); y[:-1] = post[1:]; y[-1] = series[n_train]
            Xb = readout.add_bias(X); w = readout.train_readout(Xb[:-1], y[:-1], ridge=cfg.ridge)
            rho = r.initial_state()
            for s in drive: rho = r.step(rho, float(s))
            preds = []
            for _ in range(horizon):
                feat = readout.add_bias(readout.features_from_states([rho], obs))[0]
                cur = float(np.clip(feat @ w, 0, 1)); preds.append(cur); rho = r.step(rho, cur)
            truth = series[n_train:n_train+horizon]
            mse[m][r_i] = float(np.mean((truth-np.array(preds))**2))
            if r_i == 0: example[m] = preds; truth_ex = truth.tolist()
        print(f"  [MG] realization {r_i+1}/{cfg.n_realizations} ({time.time()-t0:.0f}s)")
    out = {"task": "mg_families", "horizon": horizon, "families": fam_names,
           "mse": {m: mse[m].tolist() for m in fam_names},
           "example_pred": {m: example[m] for m in fam_names}, "example_truth": truth_ex,
           "seeds": seeds.tolist(), "config": cfg.__dict__, "runtime_s": time.time()-t0}
    json.dump(out, open("../results/mg_families.json", "w"), indent=2)
    print("=== Mackey-Glass autonomous MSE (mean) ===")
    for m in fam_names:
        print(f"  {m:14s} MSE={mse[m].mean():.4e} ± {mse[m].std():.1e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("parity", "both"): run_parity()
    if which in ("mg", "both"): run_mg()

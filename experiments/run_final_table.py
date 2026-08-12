"""Final robust table: normal model (FN) -> CD (paper) -> our methods.

Evaluates every model at its grid-found best operating point, but now with MANY
paired seeds for tight error bars and significance. Tells the story:
   FN (unitary, "normal")  ->  CD uniform local loss (paper's improvement)
   ->  engineered dissipators (how much further they extend it).

We report STM total memory capacity and NARMA-10 NMSE, each mean ± standard error
over the seeds, with the paired significance (σ) vs FN and vs CD_paper.
"""
import _paths  # noqa: F401
import json
import time
import numpy as np

from qrc import reservoirs as res, readout, tasks, dissipators as dsp
from experiments_track_b import make_reservoir_with_jumps
from run_best_vs_best import methods_at_strength, METHODS

N_REAL = 16
QUANT = 28
WASH, TRAIN, TEST = 150, 450, 300
H = 0.5
DELAYS = list(range(1, 16))
NARMA_ORDER = 10


def _cfg(dt):
    return type("C", (), dict(n_qubits=5, h=H, dt=dt, quantize=QUANT))()


def stm_mc(X, post):
    Xb = readout.add_bias(X); tr = slice(0, TRAIN); te = slice(TRAIN, TRAIN + TEST)
    tot = 0.0
    for d in DELAYS:
        y = tasks.delayed_target(post, d)
        a = np.zeros(len(y), bool); a[tr] = True; a &= ~np.isnan(y)
        b = np.zeros(len(y), bool); b[te] = True; b &= ~np.isnan(y)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        tot += readout.capacity(y[b], readout.predict(Xb[b], w))
    return tot


def narma_nmse(X, post):
    Xb = readout.add_bias(X); tr = slice(0, TRAIN); te = slice(TRAIN, TRAIN + TEST)
    y = tasks.narma_target(post, order=NARMA_ORDER, input_scale=0.2)
    a = np.zeros(len(y), bool); a[tr] = True; a &= ~np.isnan(y)
    b = np.zeros(len(y), bool); b[te] = True; b &= ~np.isnan(y)
    w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
    return readout.nmse(y[b], readout.predict(Xb[b], w))


def main():
    bb = json.load(open("../results/best_vs_best.json"))["results"]
    n = 5
    obs = readout.pauli_observables(n, max_weight=2)
    total = WASH + TRAIN + TEST
    seeds = np.random.default_rng(777).integers(0, 2 ** 31 - 1, N_REAL)
    models = ["FN (normal/unitary)"] + METHODS

    stm = {m: np.zeros(N_REAL) for m in models}
    nar = {m: np.zeros(N_REAL) for m in models}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(n, 1.0, rng)
        stm_in = tasks.stm_inputs(total, rng); ps = stm_in[WASH:]
        nar_in = tasks.narma_inputs(total, rng); pn = nar_in[WASH:]

        # FN at its sensible points (h=0.5; STM dt=0.5, NARMA dt=0.25)
        fn_s = res.FujiNakajimaReservoir(n, J, H, 0.5)
        fn_n = res.FujiNakajimaReservoir(n, J, H, 0.25)
        stm["FN (normal/unitary)"][r_i] = stm_mc(fn_s.run(stm_in, obs, washout=WASH), ps)
        nar["FN (normal/unitary)"][r_i] = narma_nmse(fn_n.run(nar_in, obs, washout=WASH), pn)

        for m in METHODS:
            sdt, ssg = bb[m]["stm_best_dt_sgamma"]
            ndt, nsg = bb[m]["narma_best_dt_sgamma"]
            fams_s = methods_at_strength(J, n, dsp.jump_strength(dsp.local_loss(n, ssg)))
            fams_n = methods_at_strength(J, n, dsp.jump_strength(dsp.local_loss(n, nsg)))
            r_s = make_reservoir_with_jumps(J, _cfg(sdt), fams_s[m])
            r_n = make_reservoir_with_jumps(J, _cfg(ndt), fams_n[m])
            stm[m][r_i] = stm_mc(r_s.run(stm_in, obs, washout=WASH), ps)
            nar[m][r_i] = narma_nmse(r_n.run(nar_in, obs, washout=WASH), pn)
        print(f"  seed {r_i+1}/{N_REAL} done ({time.time()-t0:.0f}s)")

    def stats(arr, ref, higher_better):
        mean = arr.mean(); se = arr.std(ddof=1) / np.sqrt(len(arr))
        diff = (arr - ref) if higher_better else (ref - arr)
        sig = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(arr)) + 1e-12)
        return mean, se, sig

    fn_stm = stm["FN (normal/unitary)"]; cd_stm = stm["CD_paper (uniform local)"]
    fn_nar = nar["FN (normal/unitary)"]; cd_nar = nar["CD_paper (uniform local)"]
    out = {"models": models, "n_real": N_REAL, "seeds": seeds.tolist(),
           "stm": {m: stm[m].tolist() for m in models},
           "narma": {m: nar[m].tolist() for m in models},
           "runtime_s": time.time() - t0}
    json.dump(out, open("../results/final_table.json", "w"), indent=2)

    print(f"\n=== FINAL TABLE (N={N_REAL} seeds, mean ± SE) ===")
    print(f"{'model':28s} {'STM MC':>14s} {'σ/FN':>6s} {'σ/CD':>6s} | "
          f"{'NARMA NMSE':>16s} {'σ/FN':>6s} {'σ/CD':>6s}")
    for m in models:
        ms, ses, _ = stats(stm[m], fn_stm, True)
        _, _, sig_fn_s = stats(stm[m], fn_stm, True)
        _, _, sig_cd_s = stats(stm[m], cd_stm, True)
        mn, sen, _ = stats(nar[m], fn_nar, False)
        _, _, sig_fn_n = stats(nar[m], fn_nar, False)
        _, _, sig_cd_n = stats(nar[m], cd_nar, False)
        print(f"{m:28s} {ms:7.2f}±{ses:4.2f} {sig_fn_s:6.1f} {sig_cd_s:6.1f} | "
              f"{mn:8.3f}±{sen:5.3f} {sig_fn_n:6.1f} {sig_cd_n:6.1f}")


if __name__ == "__main__":
    main()

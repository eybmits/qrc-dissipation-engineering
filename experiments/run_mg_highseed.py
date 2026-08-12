"""Settle the Mackey-Glass question: is B5 pair loss's +25% real or noise?

High-seed (N=48) MG run for the key methods only, to test whether the suggestive
MG improvements (B5 pair, A1 heterogeneous) reach significance vs the paper model.
Same operating points as the final table (each method at its NARMA optimum).
"""
import _paths  # noqa: F401
import json
import time
import numpy as np

from qrc import reservoirs as res, readout, tasks, dissipators as dsp
from experiments_track_b import make_reservoir_with_jumps
from run_best_vs_best import methods_at_strength
from run_final_parity_mg import mg_mse, WASH, TRAIN, TEST, HORIZON, SAMPLE_EVERY, _cfg, H

N_REAL = 48
KEY = ["CD_paper (uniform local)", "B5 pair loss", "A1 heterogeneous rates",
       "A2 inverse-coupling", "B3 collective"]


def main():
    bb = json.load(open("../results/best_vs_best.json"))["results"]
    n = 5
    obs = readout.pauli_observables(n, max_weight=2)
    models = ["FN (normal/unitary)"] + KEY
    seeds = np.random.default_rng(2027).integers(0, 2 ** 31 - 1, N_REAL)
    n_train = WASH + TRAIN
    val = {m: np.zeros(N_REAL) for m in models}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(n, 1.0, rng)
        series = tasks.mackey_glass_series(n_train + HORIZON + 5, tau=17,
                                           sample_every=SAMPLE_EVERY, discard=500, rng=rng)
        series = tasks.rescale_unit(series, 0, 1)
        val["FN (normal/unitary)"][r_i] = mg_mse(
            res.FujiNakajimaReservoir(n, J, H, 0.25), J, obs, series, n_train)
        for m in KEY:
            ndt, nsg = bb[m]["narma_best_dt_sgamma"]
            fams = methods_at_strength(J, n, dsp.jump_strength(dsp.local_loss(n, nsg)))
            r = make_reservoir_with_jumps(J, _cfg(ndt), fams[m])
            val[m][r_i] = mg_mse(r, J, obs, series, n_train)
        print(f"  seed {r_i+1}/{N_REAL} done ({time.time()-t0:.0f}s)")

    json.dump({"task": "mg_highseed", "models": models, "n_real": N_REAL,
               "value": {m: val[m].tolist() for m in models}, "seeds": seeds.tolist(),
               "runtime_s": time.time() - t0}, open("../results/mg_highseed.json", "w"), indent=2)
    cd = np.array(val["CD_paper (uniform local)"])
    print(f"\n=== Mackey-Glass, N={N_REAL} seeds (MSE ↓) ===")
    for m in models:
        a = np.array(val[m]); se = a.std(ddof=1) / np.sqrt(N_REAL)
        d = cd - a  # positive = better than CD
        sig = d.mean() / (d.std(ddof=1) / np.sqrt(N_REAL) + 1e-12)
        imp = d.mean() / cd.mean() * 100
        print(f"  {m:28s} MSE={a.mean():.4f} ± {se:.4f}   {imp:+.0f}% vs CD   {sig:+.1f}σ")


if __name__ == "__main__":
    main()

"""Maximal laptop scaling: N in {4,5,6}, more seeds, to confirm/refute the
'collective advantage grows with N' trend and hunt for any surprising effect.

N=6 is the laptop ceiling (Liouvillian 4096x4096; N=7 would be 4.3 GB + ~18 s per
expm). We use quantize=8 propagator caching: the precompute (8 dense expm) dominates
and is sequence-length-independent, so we can afford proper-length sequences AND
10 seeds. STM total memory capacity, matched budget (strength = local loss at gamma=1).
"""
import _paths  # noqa: F401
import json
import time
import numpy as np

from qrc import reservoirs as res, readout, tasks, dissipators as dsp
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive, LindbladReservoir

NS = [4, 5, 6]
N_REAL = 10
H, DT, GAMMA, QUANT = 0.5, 0.5, 1.0, 8
WASH, TRAIN, TEST = 100, 300, 200
DELAYS = list(range(1, 16))
METHODS = ["CD_paper", "A1_heterogeneous", "B3_collective", "B1_dephasing"]


def jumps_for(name, J, N, s, rng):
    if name == "CD_paper":
        return dsp.normalize_jump_strength(dsp.local_loss(N, 1.0), s)
    if name == "A1_heterogeneous":
        g = dsp.loguniform_rates(N, 0.1, 10, rng, mean=1.0)
        return [(res.sminus(i, N), float(g[i])) for i in range(N)]
    if name == "B3_collective":
        return dsp.normalize_jump_strength(dsp.collective_loss(N, 1.0), s)
    if name == "B1_dephasing":
        return dsp.normalize_jump_strength(dsp.dephasing(N, 1.0), s)
    raise ValueError(name)


def reservoir(name, J, N, s, rng):
    H0 = ising_xx_hamiltonian(J, H, N); Hx = transverse_drive(N)
    return LindbladReservoir.from_terms(N, H0 + H * Hx, H * Hx,
                                        jumps_for(name, J, N, s, rng), DT, quantize=QUANT)


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


def main():
    total = WASH + TRAIN + TEST
    mc = {N: {m: np.zeros(N_REAL) for m in ["FN"] + METHODS} for N in NS}
    t0 = time.time()
    for N in NS:
        obs = readout.pauli_observables(N, max_weight=2)
        s = dsp.jump_strength(dsp.local_loss(N, GAMMA))
        seeds = np.random.default_rng(13).integers(0, 2 ** 31 - 1, N_REAL)
        for r_i, sd in enumerate(seeds):
            rng = np.random.default_rng(int(sd))
            J = res.random_couplings(N, 1.0, rng)
            inp = tasks.stm_inputs(total, rng); post = inp[WASH:]
            mc[N]["FN"][r_i] = stm_mc(res.FujiNakajimaReservoir(N, J, H, DT).run(
                inp, obs, washout=WASH), post)
            for m in METHODS:
                r = reservoir(m, J, N, s, np.random.default_rng(int(sd) + 1))
                mc[N][m][r_i] = stm_mc(r.run(inp, obs, washout=WASH), post)
            print(f"  N={N} seed {r_i+1}/{N_REAL} ({time.time()-t0:.0f}s)")

    out = {"NS": NS, "n_real": N_REAL, "quant": QUANT,
           "mc": {str(N): {m: mc[N][m].tolist() for m in mc[N]} for N in NS}}
    json.dump(out, open("../results/scaling_big.json", "w"), indent=2)

    print("\n=== STM scaling — ΔMC of collective & heterogeneous vs CD_paper ===")
    print(f"{'N':>3} | {'CD MC':>10} | {'collective ΔvsCD':>22} | {'hetero ΔvsCD':>20} | {'deph':>6}")
    for N in NS:
        cd = mc[N]["CD_paper"]
        def line(m):
            d = mc[N][m] - cd; se = d.std(ddof=1) / np.sqrt(N_REAL)
            return f"{d.mean():+.2f}±{se:.2f} ({100*d.mean()/cd.mean():+.0f}%, {d.mean()/(se+1e-9):+.1f}σ)"
        print(f"{N:>3} | {cd.mean():5.2f}±{cd.std(ddof=1)/np.sqrt(N_REAL):.2f} | "
              f"{line('B3_collective'):>22} | {line('A1_heterogeneous'):>20} | "
              f"{mc[N]['B1_dephasing'].mean():.2f}")
    # trend test: is collective %gain increasing with N?
    gains = [100 * (mc[N]['B3_collective'].mean() - mc[N]['CD_paper'].mean()) /
             mc[N]['CD_paper'].mean() for N in NS]
    print(f"\ncollective %gain vs N: {dict(zip(NS, [round(g,1) for g in gains]))}")
    slope = np.polyfit(NS, gains, 1)[0]
    print(f"linear slope of %gain vs N: {slope:+.1f} %/qubit  (positive => advantage GROWS with N)")


if __name__ == "__main__":
    main()

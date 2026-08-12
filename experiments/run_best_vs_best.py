"""Best-vs-best comparison: every dissipation method at its OWN optimum.

Unlike the master comparison (single shared operating point), here each method is
tuned over a grid of (Δt, dissipation-strength) by **validation** score and scored
on a held-out **test** split — the fair "best-vs-best" a referee expects. `h=0.5`
is fixed (Phase 0 found it optimal for both FN and the CD baseline; Δt and the
strength are the dominant knobs).

Tasks: STM (maximise total memory capacity) and NARMA-10 (minimise NMSE).  Each
method's STM-optimum and NARMA-optimum are found independently (they differ).
"""
import _paths  # noqa: F401
import itertools
import json
import time
import numpy as np

from qrc import reservoirs as res, readout, tasks
from qrc import dissipators as dsp
from experiments_track_b import make_reservoir_with_jumps

H = 0.5
DT_GRID = [0.1, 0.25, 0.5, 1.0]
# total dissipative strengths = local loss at gamma in {0.3, 1, 3}
S_GRID_GAMMA = [0.3, 1.0, 3.0]
N_REAL = 3
QUANT = 28
WASH, TRAIN, VAL, TEST = 120, 280, 150, 200
DELAYS = list(range(1, 16))
NARMA_ORDER = 10


def methods_at_strength(J, n, s):
    """Every family normalised to total dissipative strength ``s``."""
    rng = np.random.default_rng(int(abs(J).sum() * 1e6) % (2 ** 31))
    g_rand = dsp.loguniform_rates(n, 0.1, 10, rng, mean=1.0)
    g_cpl = dsp.coupling_rates(J, mean=1.0)
    g_inv = dsp.inverse_coupling_rates(J, mean=1.0)
    c_graph = np.sum(np.abs(J), axis=1) + 1e-3
    edges = dsp.graph_edges(J)
    pair_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-9]
    raw = {
        "CD_paper (uniform local)": dsp.local_loss(n, 1.0),
        "A1 heterogeneous rates": [(res.sminus(i, n), float(g_rand[i])) for i in range(n)],
        "A2 coupling-structured": [(res.sminus(i, n), float(g_cpl[i])) for i in range(n)],
        "A2 inverse-coupling": [(res.sminus(i, n), float(g_inv[i])) for i in range(n)],
        "B1 dephasing": dsp.dephasing(n, 1.0),
        "B2 local gain/loss": dsp.thermal(n, 1.0, 0.3),
        "B3 collective": dsp.collective_loss(n, 1.0),
        "B3 collective (graph c)": dsp.collective_loss(n, 1.0, c=c_graph),
        "B4 loss+exchange": (dsp.normalize_jump_strength(dsp.local_loss(n, 1.0), 0.4)
                             + dsp.normalize_jump_strength(dsp.exchange(n, 1.0, edges), 0.6)),
        "B5 pair loss": dsp.pair_loss(n, 1.0, pair_edges),
    }
    return {k: dsp.normalize_jump_strength(v, s) for k, v in raw.items()}


METHODS = list(methods_at_strength(np.ones((5, 5)) - np.eye(5), 5, 80.0))


def _split_metric(X, post, target_fn, delays, metric, which):
    Xb = readout.add_bias(X)
    tr = slice(0, TRAIN)
    blk = {"val": slice(TRAIN, TRAIN + VAL),
           "test": slice(TRAIN + VAL, TRAIN + VAL + TEST)}[which]
    tot = 0.0
    for d in delays:
        y = target_fn(post, d)
        a = np.zeros(len(y), bool); a[tr] = True; a &= ~np.isnan(y)
        b = np.zeros(len(y), bool); b[blk] = True; b &= ~np.isnan(y)
        r = readout.fit_and_score(Xb[a], y[a], Xb[b], y[b], metric=metric, ridge=1e-8)
        tot += r.test_metric
    return tot


def main():
    n = 5
    total_len = WASH + TRAIN + VAL + TEST
    obs = readout.pauli_observables(n, max_weight=2)
    seeds = np.random.default_rng(31415).integers(0, 2 ** 31 - 1, N_REAL)
    grid = list(itertools.product(DT_GRID, S_GRID_GAMMA))
    narma_tf = lambda s_, k: tasks.narma_target(s_, order=k, input_scale=0.2)

    # val/test scores: [method][(dt,sgamma)] -> arrays over realizations
    stm_val = {m: {g: np.zeros(N_REAL) for g in grid} for m in METHODS}
    stm_test = {m: {g: np.zeros(N_REAL) for g in grid} for m in METHODS}
    nar_val = {m: {g: np.zeros(N_REAL) for g in grid} for m in METHODS}
    nar_test = {m: {g: np.zeros(N_REAL) for g in grid} for m in METHODS}

    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(n, 1.0, rng)
        stm_in = tasks.stm_inputs(total_len, rng)
        nar_in = tasks.narma_inputs(total_len, rng)
        s_targets = {g: dsp.jump_strength(dsp.local_loss(n, g)) for g in S_GRID_GAMMA}
        for (dt, sg) in grid:
            fams = methods_at_strength(J, n, s_targets[sg])
            for m in METHODS:
                cfg = type("C", (), dict(n_qubits=n, h=H, dt=dt, quantize=QUANT,
                                         J_s=1.0, ridge=1e-8, max_weight=2))()
                reservoir = make_reservoir_with_jumps(J, cfg, fams[m])
                Xs = reservoir.run(stm_in, obs, washout=WASH)
                Xn = reservoir.run(nar_in, obs, washout=WASH)
                ps, pn = stm_in[WASH:], nar_in[WASH:]
                stm_val[m][(dt, sg)][r_i] = _split_metric(Xs, ps, tasks.delayed_target, DELAYS, "capacity", "val")
                stm_test[m][(dt, sg)][r_i] = _split_metric(Xs, ps, tasks.delayed_target, DELAYS, "capacity", "test")
                nar_val[m][(dt, sg)][r_i] = _split_metric(Xn, pn, narma_tf, [NARMA_ORDER], "nmse", "val")
                nar_test[m][(dt, sg)][r_i] = _split_metric(Xn, pn, narma_tf, [NARMA_ORDER], "nmse", "test")
        print(f"  realization {r_i+1}/{N_REAL} done ({time.time()-t0:.0f}s)")

    out = {"methods": METHODS, "h": H, "dt_grid": DT_GRID, "s_grid_gamma": S_GRID_GAMMA,
           "results": {}, "grid": {}, "seeds": seeds.tolist(), "runtime_s": time.time() - t0}
    for m in METHODS:
        # full grid (for both the best-vs-best and the matched-budget views)
        out["grid"][m] = {f"{dt}|{sg}": {
            "stm_val": float(stm_val[m][(dt, sg)].mean()),
            "stm_test": float(stm_test[m][(dt, sg)].mean()),
            "stm_test_std": float(stm_test[m][(dt, sg)].std()),
            "nar_val": float(nar_val[m][(dt, sg)].mean()),
            "nar_test": float(nar_test[m][(dt, sg)].mean()),
            "nar_test_std": float(nar_test[m][(dt, sg)].std()),
        } for (dt, sg) in grid}
        # best-vs-best: tune over the WHOLE grid (dt + strength)
        best_stm = max(grid, key=lambda g: stm_val[m][g].mean())
        best_nar = min(grid, key=lambda g: nar_val[m][g].mean())
        # matched-budget: fix strength s=gamma 1.0 (==80), tune only dt
        mb = [(dt, sg) for (dt, sg) in grid if sg == 1.0]
        mb_stm = max(mb, key=lambda g: stm_val[m][g].mean())
        mb_nar = min(mb, key=lambda g: nar_val[m][g].mean())
        out["results"][m] = {
            "stm_best_dt_sgamma": list(best_stm),
            "stm_test_MC": float(stm_test[m][best_stm].mean()),
            "stm_test_MC_std": float(stm_test[m][best_stm].std()),
            "narma_best_dt_sgamma": list(best_nar),
            "narma_test_NMSE": float(nar_test[m][best_nar].mean()),
            "narma_test_NMSE_std": float(nar_test[m][best_nar].std()),
            # matched-budget (fair, s fixed) view:
            "mb_stm_best_dt": mb_stm[0], "mb_stm_test_MC": float(stm_test[m][mb_stm].mean()),
            "mb_narma_best_dt": mb_nar[0], "mb_narma_test_NMSE": float(nar_test[m][mb_nar].mean()),
        }
    json.dump(out, open("../results/best_vs_best.json", "w"), indent=2)

    print("\n=== BEST-vs-BEST (each method at its own (Δt, strength) optimum) ===")
    ref = out["results"]["CD_paper (uniform local)"]
    print(f"{'method':28s} {'STM MC':>8s} {'@(dt,γ)':>10s} {'NARMA':>7s} {'@(dt,γ)':>10s}")
    for m in METHODS:
        r = out["results"][m]
        print(f"{m:28s} {r['stm_test_MC']:8.2f} {str(tuple(r['stm_best_dt_sgamma'])):>10s} "
              f"{r['narma_test_NMSE']:7.3f} {str(tuple(r['narma_best_dt_sgamma'])):>10s}")
    print(f"\n(reference CD_paper: STM MC={ref['stm_test_MC']:.2f}, "
          f"NARMA NMSE={ref['narma_test_NMSE']:.3f})")


if __name__ == "__main__":
    main()

"""Referee-response experiment blocks (Quantum review). Checkpointed, restart-safe.

Writes one JSON per job to results/review_protocol/ and skips existing files.

Blocks
  R_zero   Zero-jump continuous-drive control: SAME Hamiltonian input encoding
           h(1+s)Sum sigma_x and SAME readout as every Lindblad reservoir, with
           jumps = [] (review point 1: a genuine "remove the Lindblad loss"
           control; the reset-based Fujii-Nakajima model is a different input
           map and is relabelled, not used as the lossless control).
  R_sweep  Full damping-strength curves per channel (review point 3i): strength
           multiplier in {0.25,0.5,1,2,4} x 6 channels, STM, N=5. Also yields
           per-channel best-vs-best via validation selection (point 3ii).
  R_match  Alternative normalisations (review point 3iii): rates rescaled so
           that (a) the initial energy-loss rate from the fully excited state,
           or (b) the driven-Liouvillian spectral gap at s=0.5, matches the
           uniform-local dial. Scale factors are computed per seed and stored.
  R_ham    Hamiltonian-ensemble generality (review point 5): 4 ensembles x
           {zero-jump, dial, collective, pair, dephasing} x {stm,narma,parity},
           N=5, 32 paired seeds; plus per-(ensemble,channel) spectral gaps to
           test whether the compass predicts the winner per ensemble.
  R_wash   Initial-state / washout convergence (resubmission condition 3):
           rho0 in {ground, maximally mixed, Haar-random pure} x washout in
           {50,200,800} for {dial, collective, zero-jump}.

Run:  cd experiments && PYTHONPATH=../src python run_review_experiments.py --workers 8
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

from _paths import RESULTS_DIR
from qrc import diagnostics as diag
from qrc import dissipators as dsp
from qrc import liouvillian as dlio
from qrc import readout, reservoirs as res, tasks
from qrc.operators import pauli_op, sminus, two_site_pauli
from qrc.reservoirs import transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir
from run_final_scaling import TCFG, build_jumps, deterministic_seeds

H, DT, GAMMA = 0.5, 0.5, 1.0
WASH, TRAIN, TEST = 200, 600, 400
OUTDIR = Path(RESULTS_DIR) / "review_protocol"

SWEEP_CHANNELS = ("CD_paper", "B3_collective", "A1_heterogeneous",
                  "B2_thermal", "B4_loss_exchange", "B5_pair")
SWEEP_MULT = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SHOTMAP_CHANNELS = ("CD_paper", "B3_collective", "A1_heterogeneous",
                    "B2_thermal", "B4_loss_exchange", "B5_pair")
SHOT_BUDGETS = (64, 256, 1024, 4096, 16384, 65536, 262144, 0)  # 0 = exact
MATCH_CHANNELS = ("B3_collective", "B5_pair", "B2_thermal")
HAM_METHODS = ("ZJ_zerojump", "CD_paper", "B3_collective", "B5_pair", "B1_dephasing")
ENSEMBLES = ("xx_z_x", "zz_x_z", "xy_z_x", "xx_ring")


# ------------------------------------------------------------- Hamiltonians
def field_op(axis, N):
    return sum(pauli_op(axis, i, N) for i in range(N))


def build_hamiltonian(ensemble: str, J: np.ndarray, N: int):
    """Return (H_static_without_drive_offset, H_drive). Total H(s) = H_static
    + (1+s) * h * H_drive_direction, matching the implemented convention."""
    if ensemble == "xx_z_x":
        coup = sum(J[i, j] * two_site_pauli("x", i, j, N)
                   for i in range(N) for j in range(i + 1, N))
        return coup + H * field_op("z", N), field_op("x", N)
    if ensemble == "zz_x_z":
        coup = sum(J[i, j] * two_site_pauli("z", i, j, N)
                   for i in range(N) for j in range(i + 1, N))
        return coup + H * field_op("x", N), field_op("z", N)
    if ensemble == "xy_z_x":
        coup = sum(J[i, j] * (two_site_pauli("x", i, j, N) + two_site_pauli("y", i, j, N)) / 2
                   for i in range(N) for j in range(i + 1, N))
        return coup + H * field_op("z", N), field_op("x", N)
    if ensemble == "xx_ring":
        coup = sum(J[i, (i + 1) % N] * two_site_pauli("x", i, (i + 1) % N, N)
                   for i in range(N))
        return coup + H * field_op("z", N), field_op("x", N)
    raise ValueError(ensemble)


def ring_couplings(N, scale, rng):
    J = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        v = rng.uniform(-scale, scale)
        J[i, j] = J[j, i] = v
    return J


def make_reservoir(ensemble, method, J, N, rng, rate_scale=1.0):
    H0, Hdir = build_hamiltonian(ensemble, J, N)
    base = H0 + H * Hdir           # constant offset: total field h(1+s)
    drive = H * Hdir
    if method == "FN":
        return res.FujiNakajimaReservoir(N, J, H, DT)
    if method == "ZJ_zerojump":
        jumps = []
    else:
        target = dsp.jump_strength(dsp.local_loss(N, GAMMA)) * rate_scale
        jumps = build_jumps(method, J, N, target, rng)
        if method == "A1_heterogeneous" and rate_scale != 1.0:
            jumps = [(L, r * rate_scale) for L, r in jumps]
    return SparseLindbladReservoir.from_terms(N, base, drive, jumps, DT)


# ------------------------------------------------------------- task metrics
def _masks(y, train, test):
    a = np.zeros(len(y), bool); a[:train] = True; a &= ~np.isnan(y)
    b = np.zeros(len(y), bool); b[train:train + test] = True; b &= ~np.isnan(y)
    return a, b


def stm_mc(X, post, train=TRAIN, test=TEST):
    Xb = readout.add_bias(X)
    tot = 0.0
    for d in TCFG.stm_delays:
        y = tasks.delayed_target(post, d)
        a, b = _masks(y, train, test)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        tot += readout.capacity(y[b], readout.predict(Xb[b], w))
    return float(tot)


def narma_nmse(X, post):
    Xb = readout.add_bias(X)
    y = tasks.narma_target(post, order=TCFG.narma_order, input_scale=TCFG.input_scale)
    a, b = _masks(y, TRAIN, TEST)
    w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
    return float(readout.nmse(y[b], readout.predict(Xb[b], w)))


def parity_cap(X, post):
    Xb = readout.add_bias(X)
    tot = 0.0
    for d in TCFG.parity_delays:
        y = tasks.parity_target(post, d)
        a, b = _masks(y, TRAIN, TEST)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        tot += readout.capacity(y[b], readout.predict(Xb[b], w))
    return float(tot)


SWEEP_VAL = 150   # validation window carved from the train tail (R_sweep2)


def sweep2_scores(reservoir, N, seed):
    """STM capacity on a validation split AND the held-out test split.

    Fit always uses the first TRAIN-SWEEP_VAL steps; the validation score is on
    the carved train tail, the test score on the untouched test block. Used for
    honest per-channel strength selection (select on val, report test)."""
    rng = np.random.default_rng(seed)
    _ = res.random_couplings(N, 1.0, rng)
    obs = readout.pauli_observables(N, max_weight=2)
    total = WASH + TRAIN + TEST
    inp = tasks.stm_inputs(total, rng)
    X = reservoir.run(inp, obs, washout=WASH)
    post = inp[WASH:]
    Xb = readout.add_bias(X)
    fit = np.zeros(TRAIN + TEST, bool); fit[:TRAIN - SWEEP_VAL] = True
    val = np.zeros(TRAIN + TEST, bool); val[TRAIN - SWEEP_VAL:TRAIN] = True
    tst = np.zeros(TRAIN + TEST, bool); tst[TRAIN:TRAIN + TEST] = True
    out = []
    for block in (val, tst):
        mc = 0.0
        for d in TCFG.stm_delays:
            y = tasks.delayed_target(post, d)
            a = fit & ~np.isnan(y)
            b = block & ~np.isnan(y)
            w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
            mc += readout.capacity(y[b], readout.predict(Xb[b], w))
        out.append(float(mc))
    return out  # [val_mc, test_mc]


def run_task(reservoir, task, N, seed, washout=WASH, rho0=None):
    rng = np.random.default_rng(seed)
    _ = res.random_couplings(N, 1.0, rng)      # keep the seed stream identical
    obs = readout.pauli_observables(N, max_weight=2)
    total = washout + TRAIN + TEST
    if task == "stm":
        inp = tasks.stm_inputs(total, rng)
        X = reservoir.run(inp, obs, washout=washout, rho0=rho0)
        return stm_mc(X, inp[washout:])
    if task == "narma":
        inp = tasks.narma_inputs(total, rng)
        X = reservoir.run(inp, obs, washout=washout, rho0=rho0)
        return narma_nmse(X, inp[washout:])
    if task == "parity":
        inp = tasks.parity_inputs(total, rng)
        X = reservoir.run(inp, obs, washout=washout,
                          n_virtual=TCFG.parity_n_virtual, rho0=rho0)
        return parity_cap(X, inp[washout:])
    if task == "mg":
        from types import SimpleNamespace
        from run_final_scaling import mg_mse
        n_samples = washout + TRAIN + TCFG.mg_horizon + 5
        series = tasks.mackey_glass_series(n_samples, tau=17,
                                           sample_every=TCFG.mg_sample_every,
                                           discard=500, rng=rng)
        series = tasks.rescale_unit(series, 0.0, 1.0)
        return mg_mse(reservoir, obs, series,
                      SimpleNamespace(wash=washout, train=TRAIN))
    raise ValueError(task)


# ------------------------------------------------------------- normalisation
def energy_rate(jumps, N):
    """Initial loss rate of total excitation number from the fully excited
    state under the dissipator alone (dense, N=5 is small)."""
    d = 2 ** N
    rho = np.zeros((d, d), complex); rho[d - 1, d - 1] = 1.0   # |1..1>
    Nop = sum((np.eye(d) - pauli_op("z", i, N)) / 2 for i in range(N))
    drho = np.zeros_like(rho)
    for L, rate in jumps:
        L = np.asarray(L, complex)
        drho += rate * (L @ rho @ L.conj().T
                        - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))
    return float(-np.real(np.trace(Nop @ drho)))


def driven_gap(ensemble, J, N, jumps):
    H0, Hdir = build_hamiltonian(ensemble, J, N)
    Hs = H0 + H * Hdir + 0.5 * H * Hdir       # s = 0.5
    L = dlio.lindbladian(Hs, jumps)
    return diag.spectral_gap(L)


def match_scale(method, J, N, rng, mode):
    """Scale factor c so the channel at c*standard_target matches the dial."""
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
    dial = build_jumps("CD_paper", J, N, target, rng)
    def with_scale(c):
        return build_jumps(method, J, N, c * target, np.random.default_rng(rng.integers(1 << 31)))
    if mode == "energy":
        ref = energy_rate(dial, N)
        cur = energy_rate(with_scale(1.0), N)
        return ref / cur if cur > 0 else 1.0
    if mode == "gap":
        ref = driven_gap("xx_z_x", J, N, dial)
        lo, hi = 0.05, 40.0
        for _ in range(18):
            mid = np.sqrt(lo * hi)
            g = driven_gap("xx_z_x", J, N, with_scale(mid))
            if g < ref:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))
    raise ValueError(mode)


# ------------------------------------------------------------- checkpointing
def ck_path(block, tag, seed):
    return OUTDIR / f"{block}__{tag}_s{seed}.json"


def write_ck(block, tag, seed, payload):
    p = ck_path(block, tag, seed)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, p)


def run_job(job):
    block = job["block"]
    tag = job["tag"]
    seed = job["seed"]
    if ck_path(block, tag, seed).exists():
        return f"skip {block} {tag} s={seed}"
    t0 = time.time()
    try:
        N = job.get("N", 5)
        ens = job.get("ensemble", "xx_z_x")
        rng = np.random.default_rng(seed)
        J = ring_couplings(N, 1.0, rng) if ens == "xx_ring" \
            else res.random_couplings(N, 1.0, rng)
        method_rng = np.random.default_rng(seed + 1)

        if block == "R_optfix":
            out = optfix_job(job["method"], seed)
            write_ck(block, tag, seed, {
                "block": block, "N": 5, "method": job["method"], "task": "stm",
                "seed": seed, "value": out["test_at_best"], **out,
                "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_gapsweep":
            g = gapsweep_job(job["method"], job["mult"], seed)
            write_ck(block, tag, seed, {
                "block": block, "N": 5, "method": job["method"],
                "task": "gap", "seed": seed, "mult": job["mult"],
                "value": g, "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        extra = {}
        if block == "R_match2":
            c, ratio, reachable = activity_scale(job["method"], J, N,
                                                 np.random.default_rng(seed + 3))
            extra["scale_factor"] = c
            extra["activity_ratio"] = ratio
            extra["reachable"] = reachable
            r = make_reservoir(ens, job["method"], J, N, method_rng, rate_scale=c)
            val = run_task(r, "stm", N, seed)
            write_ck(block, tag, seed, {
                "block": block, "N": N, "method": job["method"], "task": "stm",
                "seed": seed, "mode": "activity", "value": float(val),
                "runtime_s": time.time() - t0, **extra})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_match" and job.get("mode"):
            c = match_scale(job["method"], J, N, np.random.default_rng(seed + 2),
                            job["mode"])
            extra["scale_factor"] = c
            r = make_reservoir(ens, job["method"], J, N, method_rng, rate_scale=c)
        else:
            r = make_reservoir(ens, job["method"], J, N, method_rng,
                               rate_scale=job.get("mult", 1.0))

        if block == "R_sweep2":
            val_mc, test_mc = sweep2_scores(r, N, seed)
            write_ck(block, tag, seed, {
                "block": block, "N": N, "ensemble": ens, "method": job["method"],
                "task": "stm", "seed": seed, "mult": job.get("mult"),
                "val_value": val_mc, "value": test_mc,
                "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_mgfix":
            val = mg_trainonly(r, N, seed)
            write_ck(block, tag, seed, {
                "block": block, "N": N, "method": job["method"], "task": "mg",
                "seed": seed, "value": float(val),
                "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_lenfix":
            val = lenfix_stm(r, N, seed, job["train_len"])
            write_ck(block, tag, seed, {
                "block": block, "N": N, "method": job["method"], "task": "stm",
                "seed": seed, "train_len": job["train_len"],
                "value": float(val), "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_shots":
            val = shots_stm(r, N, seed, job["shots"])
            write_ck(block, tag, seed, {
                "block": block, "N": N, "method": job["method"], "task": "stm",
                "seed": seed, "shots": job["shots"],
                "value": float(val), "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_shotmap":
            out = shotmap_job(job["method"], seed)
            write_ck(block, tag, seed, {
                "block": block, "N": 5, "method": job["method"], "task": "stm",
                "seed": seed, "feat_var": out["feat_var"], "caps": out["caps"],
                "runtime_s": time.time() - t0})
            return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"

        if block == "R_wash":
            d = 2 ** N
            if job["rho0"] == "ground":
                rho0 = None
            elif job["rho0"] == "mixed":
                rho0 = np.eye(d, dtype=complex) / d
            else:
                v = np.random.default_rng(seed + 9).standard_normal(d) \
                    + 1j * np.random.default_rng(seed + 10).standard_normal(d)
                v /= np.linalg.norm(v)
                rho0 = np.outer(v, v.conj())
            val = run_task(r, "stm", N, seed, washout=job["washout"], rho0=rho0)
        else:
            val = run_task(r, job["task"], N, seed)

        write_ck(block, tag, seed, {
            "block": block, "N": N, "ensemble": ens, "method": job["method"],
            "task": job.get("task", "stm"), "seed": seed,
            "mult": job.get("mult"), "mode": job.get("mode"),
            "rho0": job.get("rho0"), "washout": job.get("washout", WASH),
            "value": float(val), "runtime_s": time.time() - t0, **extra})
        return f"done {block} {tag} s={seed} ({time.time()-t0:.0f}s)"
    except Exception as exc:  # noqa: BLE001
        return f"error {block} {tag} s={seed}: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------- referee round 2
def mg_trainonly(reservoir, N, seed, washout=WASH):
    """Mackey--Glass with TRAIN-ONLY normalisation (fixes the leakage found in
    review: min/max are computed on the data available at training time only;
    the drive is clipped to [0,1], targets use the unclipped scaled truth)."""
    from types import SimpleNamespace
    from run_final_scaling import mg_mse
    rng = np.random.default_rng(seed)
    _ = res.random_couplings(N, 1.0, rng)
    n_samples = washout + TRAIN + TCFG.mg_horizon + 5
    series = tasks.mackey_glass_series(n_samples, tau=17,
                                       sample_every=TCFG.mg_sample_every,
                                       discard=500, rng=rng)
    lo = series[:washout + TRAIN].min()
    hi = series[:washout + TRAIN].max()
    series = (series - lo) / max(hi - lo, 1e-12)
    # NOT clipped: the drive prefix lies in [0,1] by construction of lo/hi,
    # closed-loop predictions are clipped inside mg_mse, and the held-out
    # continuation must stay unclipped so the MSE is scored against the true
    # normalised trajectory even where it exceeds the training extrema.
    return mg_mse(reservoir, readout.pauli_observables(N, max_weight=2),
                  series, SimpleNamespace(wash=washout, train=TRAIN))


def lenfix_stm(reservoir, N, seed, train_len):
    """STM with washout and test FIXED (200/400) and only the training length
    varied — isolates training-data scarcity (review: the original control
    scaled washout, train, and test together)."""
    rng = np.random.default_rng(seed)
    _ = res.random_couplings(N, 1.0, rng)
    obs = readout.pauli_observables(N, max_weight=2)
    total = WASH + train_len + TEST
    inp = tasks.stm_inputs(total, rng)
    X = reservoir.run(inp, obs, washout=WASH)
    post = inp[WASH:]
    Xb = readout.add_bias(X)
    mc = 0.0
    for d in TCFG.stm_delays:
        y = tasks.delayed_target(post, d)
        a = np.zeros(len(y), bool); a[:train_len] = True; a &= ~np.isnan(y)
        b = np.zeros(len(y), bool); b[train_len:train_len + TEST] = True
        b &= ~np.isnan(y)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        mc += readout.capacity(y[b], readout.predict(Xb[b], w))
    return float(mc)


def shotmap_job(method, seed):
    """Finite-shot STM for one channel across the FULL shot sweep, plus a static
    feature-variance diagnostic. One reservoir run, reused for every budget:
    tests whether the design map REORDERS with the measurement budget and
    whether the shot-robust winner is predictable a priori from the exact
    input-driven feature variance (larger dynamic range -> more shot-robust)."""
    N = 5
    rng = np.random.default_rng(seed)
    J = res.random_couplings(N, 1.0, rng)
    r = make_reservoir("xx_z_x", method, J, N, np.random.default_rng(seed + 1))
    obs = readout.pauli_observables(N, max_weight=2)
    inp = tasks.stm_inputs(WASH + TRAIN + TEST, rng)
    X = r.run(inp, obs, washout=WASH)
    post = inp[WASH:]
    feat_var = float(np.sum(np.var(X, axis=0)))   # predictive diagnostic
    caps = {}
    for i, shots in enumerate(SHOT_BUDGETS):
        if shots == 0:
            Xn = X
        else:
            nrng = np.random.default_rng(seed + 101 + i)
            Xn = X + nrng.standard_normal(X.shape) * np.sqrt(
                np.clip(1.0 - X ** 2, 0.0, None) / shots)
        caps[str(shots)] = float(stm_mc(Xn, post))
    return dict(feat_var=feat_var, caps=caps)


def shots_stm(reservoir, N, seed, shots):
    """STM with finite-shot feature estimates: each recorded expectation value
    is replaced by a sample mean over `shots` projective measurements,
    emulated as Gaussian noise with variance (1-<O>^2)/shots per feature."""
    rng = np.random.default_rng(seed)
    _ = res.random_couplings(N, 1.0, rng)
    obs = readout.pauli_observables(N, max_weight=2)
    total = WASH + TRAIN + TEST
    inp = tasks.stm_inputs(total, rng)
    X = reservoir.run(inp, obs, washout=WASH)
    if shots > 0:
        nrng = np.random.default_rng(seed + 77)
        X = X + nrng.standard_normal(X.shape) * np.sqrt(
            np.clip(1.0 - X ** 2, 0.0, None) / shots)
    return stm_mc(X, inp[WASH:])


def optfix_job(method, seed):
    """Equal-budget re-optimisation of the learned (A3) and adaptive (A4)
    profiles: BOTH get 2 restarts x 35 Nelder-Mead evaluations, and the full
    per-evaluation validation trace is stored, so overfitting can be
    distinguished from failed optimisation (review round 2, point 1c)."""
    from scipy.optimize import minimize
    from run_adaptive_supplement import (GBAR as PGBAR, adaptive_mc, static_mc)
    N = 5
    rng = np.random.default_rng(seed)
    J = res.random_couplings(N, 1.0, rng)
    total = 100 + 250 + 200          # run_adaptive_supplement protocol lengths
    inputs = tasks.stm_inputs(total, rng)
    post = inputs[100:]
    g_rand = dsp.loguniform_rates(N, PGBAR / 10, PGBAR * 10, rng, mean=PGBAR)
    uni_val = static_mc(dsp.uniform_rates(N, PGBAR), J, inputs, post, "val")
    uni_test = static_mc(dsp.uniform_rates(N, PGBAR), J, inputs, post, "test")

    traces, best_x, best_val = [], None, -np.inf
    if method == "A3_eq":
        def val_of(theta):
            g = dsp.normalize_rates(dsp.rates_from_theta(theta), PGBAR)
            return static_mc(g, J, inputs, post, "val")
        starts = [np.zeros(N), np.log(np.expm1(np.clip(g_rand, 1e-3, None)))]
        dim = N
    else:  # A4_eq
        def val_of(ab):
            return adaptive_mc(ab[:N], ab[N:], J, inputs, post, "val")
        b0 = np.log(np.expm1(PGBAR))
        starts = [np.concatenate([np.zeros(N), np.full(N, b0)]),
                  np.concatenate([np.random.default_rng(seed + 5)
                                  .standard_normal(N) * 0.5, np.full(N, 0.5)])]
        dim = 2 * N
    for x0 in starts:
        tr = []
        def neg(x):
            v = val_of(x)
            tr.append(float(v))
            return -v
        sol = minimize(neg, x0, method="Nelder-Mead",
                       options=dict(maxfev=35, xatol=1e-2, fatol=1e-3))
        traces.append(tr)
        if -sol.fun > best_val:
            best_val, best_x = -sol.fun, sol.x
    if method == "A3_eq":
        g = dsp.normalize_rates(dsp.rates_from_theta(best_x), PGBAR)
        test = static_mc(g, J, inputs, post, "test")
    else:
        test = adaptive_mc(best_x[:N], best_x[N:], J, inputs, post, "test")
    return dict(val_best=float(best_val), test_at_best=float(test),
                val_uniform=float(uni_val), test_uniform=float(uni_test),
                traces=traces, n_evals=int(sum(len(t) for t in traces)))


def activity_scale(method, J, N, rng):
    """Scale factor matching the steady-state jump activity
    sum_k rate_k <L_k^dag L_k>_ss of the driven generator at s=0.5 to the
    dial's (review round 2: an operational 'actual jump rate' convention)."""
    from qrc.liouvillian import steady_state
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))

    def activity(jumps):
        H0, Hdir = build_hamiltonian("xx_z_x", J, N)
        Hs = H0 + H * Hdir + 0.5 * H * Hdir
        rho = steady_state(dlio.lindbladian(Hs, jumps))
        return float(sum(r * np.real(np.trace(
            np.asarray(L).conj().T @ np.asarray(L) @ rho)) for L, r in jumps))

    ref = activity(build_jumps("CD_paper", J, N, target, rng))
    # steady-state activity is NOT monotone in the rate scale (it collapses in
    # the overdamped/Zeno regime), so: scan a log grid, bracket the FIRST
    # upward crossing if one exists and bisect inside that bracket; otherwise
    # report the closest approach (argmax) with its achieved/reference ratio.
    sub = np.random.default_rng(rng.integers(1 << 31))
    grid = np.geomspace(0.02, 50.0, 18)
    acts = [activity(build_jumps(method, J, N, g * target,
                                 np.random.default_rng(sub.integers(1 << 31))))
            for g in grid]
    bracket = None
    for i in range(len(grid) - 1):
        if (acts[i] - ref) < 0 <= (acts[i + 1] - ref):
            bracket = (grid[i], grid[i + 1])
            break
    if acts[0] >= ref:
        bracket = (grid[0] / 4, grid[0])
    if bracket is None:
        k = int(np.argmax(acts))
        return float(grid[k]), float(acts[k] / ref), False
    lo, hi = bracket
    for _ in range(14):
        mid = np.sqrt(lo * hi)
        a = activity(build_jumps(method, J, N, mid * target,
                                 np.random.default_rng(sub.integers(1 << 31))))
        if a < ref:
            lo = mid
        else:
            hi = mid
    c = float(np.sqrt(lo * hi))
    a = activity(build_jumps(method, J, N, c * target,
                             np.random.default_rng(sub.integers(1 << 31))))
    return c, float(a / ref), True


def gapsweep_job(method, mult, seed):
    N = 5
    rng = np.random.default_rng(seed)
    J = res.random_couplings(N, 1.0, rng)
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
    jumps = build_jumps(method, J, N, mult * target,
                        np.random.default_rng(seed + 1))
    return driven_gap("xx_z_x", J, N, jumps)


def diag_job(ens, method, seed):
    tag = f"diag_{ens}_{method}"
    if ck_path("R_hamdiag", tag, seed).exists():
        return f"skip diag {ens} {method} s={seed}"
    t0 = time.time()
    try:
        N = 5
        rng = np.random.default_rng(seed)
        J = ring_couplings(N, 1.0, rng) if ens == "xx_ring" \
            else res.random_couplings(N, 1.0, rng)
        target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
        jumps = [] if method == "ZJ_zerojump" else \
            build_jumps(method, J, N, target, np.random.default_rng(seed + 1))
        gap = None if not jumps else driven_gap(ens, J, N, jumps)
        write_ck("R_hamdiag", tag, seed, {
            "block": "R_hamdiag", "N": N, "ensemble": ens, "method": method,
            "task": "diagnostics", "seed": seed, "spectral_gap": gap,
            "value": gap, "runtime_s": time.time() - t0})
        return f"done diag {ens} {method} s={seed}"
    except Exception as exc:  # noqa: BLE001
        return f"error diag {ens} {method} s={seed}: {type(exc).__name__}: {exc}"


def build_manifest(n_seeds):
    S = deterministic_seeds(max(n_seeds, 32))
    jobs = []

    # R_zero: the genuine lossless control
    for N, ns in ((4, 30), (5, 30), (6, 30), (7, 30), (8, 20)):
        for task in ("stm", "narma"):
            for sd in S[:ns]:
                jobs.append(dict(block="R_zero", tag=f"zj_{task}_N{N}", seed=int(sd),
                                 N=N, method="ZJ_zerojump", task=task))
    for task in ("parity", "mg"):
        for sd in S[:32]:
            jobs.append(dict(block="R_zero", tag=f"zj_{task}_N5", seed=int(sd),
                             N=5, method="ZJ_zerojump", task=task))

    # R_sweep: strength curves
    for m in SWEEP_CHANNELS:
        for mult in SWEEP_MULT:
            for sd in S[:20]:
                jobs.append(dict(block="R_sweep", tag=f"{m}_x{mult:g}", seed=int(sd),
                                 N=5, method=m, task="stm", mult=mult))

    # R_sweep2: same sweep with a genuine validation split (select on val,
    # report test) -- used for the per-channel-optimum comparison
    for m in SWEEP_CHANNELS:
        for mult in SWEEP_MULT:
            for sd in S[:20]:
                jobs.append(dict(block="R_sweep2", tag=f"{m}_x{mult:g}", seed=int(sd),
                                 N=5, method=m, task="stm", mult=mult))

    # R_match: alternative normalisations
    for m in MATCH_CHANNELS:
        for mode in ("energy", "gap"):
            for sd in S[:32]:
                jobs.append(dict(block="R_match", tag=f"{m}_{mode}", seed=int(sd),
                                 N=5, method=m, task="stm", mode=mode))
    for sd in S[:32]:   # dial reference row under the identical pipeline
        jobs.append(dict(block="R_match", tag="CD_paper_ref", seed=int(sd),
                         N=5, method="CD_paper", task="stm", mult=1.0))

    # R_ham: generality ensembles
    for ens in ENSEMBLES:
        for m in HAM_METHODS:
            for task in ("stm", "narma", "parity"):
                for sd in S[:32]:
                    jobs.append(dict(block="R_ham", tag=f"{ens}_{m}_{task}",
                                     seed=int(sd), N=5, ensemble=ens, method=m,
                                     task=task))

    # ---- referee round 2 blocks ----
    # R_mgfix: Mackey--Glass rerun with train-only normalisation
    for m in ("FN", "CD_paper", "A1_heterogeneous", "B1_dephasing", "B2_thermal",
              "B3_collective", "B4_loss_exchange", "B5_pair"):
        for sd in deterministic_seeds(64):
            jobs.append(dict(block="R_mgfix", tag=f"{m}", seed=int(sd),
                             N=5, method=m, task="mg"))
    for sd in S[:32]:
        jobs.append(dict(block="R_mgfix", tag="ZJ_zerojump", seed=int(sd),
                         N=5, method="ZJ_zerojump", task="mg"))

    # R_lenfix: train-only length scaling (washout/test fixed)
    for m in ("CD_paper", "B3_collective"):
        for tl in (150, 300, 600, 1200, 1800):
            for sd in S[:20]:
                jobs.append(dict(block="R_lenfix", tag=f"{m}_t{tl}", seed=int(sd),
                                 N=6, method=m, train_len=tl))

    # R_optfix: equal-budget profile optimisation with stored traces
    for m in ("A3_eq", "A4_eq"):
        for sd in S[:32]:
            jobs.append(dict(block="R_optfix", tag=m, seed=int(sd), method=m))

    # R_match2: steady-state jump-activity matching
    for m in MATCH_CHANNELS:
        for sd in S[:32]:
            jobs.append(dict(block="R_match2", tag=f"{m}_activity", seed=int(sd),
                             N=5, method=m))

    # R_gapsweep: driven gap vs strength (for the compass correlation analysis)
    for m in SWEEP_CHANNELS:
        for mult in SWEEP_MULT:
            for sd in S[:3]:
                jobs.append(dict(block="R_gapsweep", tag=f"{m}_x{mult:g}",
                                 seed=int(sd), method=m, mult=mult))

    # R_shots: finite-shot readout robustness
    for m in ("CD_paper", "B3_collective"):
        for sh in (256, 1024, 4096, 0):
            for sd in S[:20]:
                jobs.append(dict(block="R_shots", tag=f"{m}_s{sh}", seed=int(sd),
                                 N=5, method=m, shots=sh))

    # R_shotmap: full shot sweep x all live channels + feature-variance
    # diagnostic (does the design map reorder with the measurement budget, and
    # is the shot-robust winner predictable a priori?)
    for m in SHOTMAP_CHANNELS:
        for sd in S[:20]:
            jobs.append(dict(block="R_shotmap", tag=m, seed=int(sd),
                             N=5, method=m))

    # R_wash: initial-state / washout convergence
    for m in ("CD_paper", "B3_collective", "ZJ_zerojump"):
        for r0 in ("ground", "mixed", "haar"):
            for w in (50, 200, 800):
                for sd in S[:10]:
                    jobs.append(dict(block="R_wash", tag=f"{m}_{r0}_w{w}",
                                     seed=int(sd), N=5, method=m, rho0=r0,
                                     washout=w))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--blocks", default=None,
                    help="comma filter, e.g. R_zero,R_sweep")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    jobs = build_manifest(32)
    if args.smoke:
        seen, smoke = set(), []
        for j in jobs:
            if j["block"] not in seen:
                seen.add(j["block"]); smoke.append(j)
        jobs = smoke
    if args.blocks:
        keep = set(args.blocks.split(","))
        jobs = [j for j in jobs if j["block"] in keep]
    # cheap blocks first so results land early; R_zero N=7/8 (slow) last
    order = {"R_mgfix": 0, "R_gapsweep": 0, "R_shots": 0, "R_shotmap": 0,
             "R_wash": 0, "R_lenfix": 1,
             "R_sweep": 1, "R_sweep2": 1, "R_match": 2, "R_match2": 2, "R_ham": 3,
             "R_zero": 4, "R_optfix": 5}
    jobs.sort(key=lambda j: (order[j["block"]],
                             j.get("N", 5), j.get("task") == "parity"))
    diag_jobs = [(e, m, int(s)) for e in ENSEMBLES for m in HAM_METHODS
                 for s in deterministic_seeds(3)] if not args.smoke else []

    print(f"review protocol: {len(jobs)} jobs + {len(diag_jobs)} diagnostics, "
          f"workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_job, j): j for j in jobs}
        futs.update({pool.submit(diag_job, *d): d for d in diag_jobs})
        done = 0
        for f in as_completed(futs):
            done += 1
            try:
                msg = f.result()
            except Exception as exc:  # noqa: BLE001
                msg = f"worker died: {exc}"
            if done % 50 == 0 or msg.startswith("error"):
                print(f"[{done}/{len(futs)}] {msg}", flush=True)
    print("REVIEW PROTOCOL COMPLETE", flush=True)


if __name__ == "__main__":
    main()

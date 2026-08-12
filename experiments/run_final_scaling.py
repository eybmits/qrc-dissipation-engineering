"""Final, pre-registered protocol driver (Runs A-E) with exact sparse Lindblad
evolution. One checkpointed, restart-safe (block, N, method, task, seed, h, dt,
sequence-length) job per JSON file; skip-if-exists; atomic writes; a failing job
never kills the run. Launch, and on any crash just re-run the same command.

Protocol (see reports discussion):
  A  Definitive dissociation table  : N=5, all 8 methods, matched budget, exact
     evolver; STM/NARMA/parity @32 seeds, MG @64 seeds.
  B  Memory N-scaling               : N=4..8, {FN,CD,B3,B1,A1}, STM+NARMA,
     30 seeds (20 @ N=8).
  C  Sequence-length control        : N=6, {CD,B3}, STM at 4 lengths, 20 seeds
     (separates the DFS/subradiance mechanism from a memory-limited artifact).
  D  Operating-point robustness      : N=5, {CD,B3,B5,B1}, STM+parity, 3 (h,dt)
     points, 20 seeds (shows the dissociation is not cherry-picked).
  E  Diagnostics vs N (static)       : unitality-defect (all N), spectral gap /
     mixing / stationary separability (N<=6, dense), all methods, 10 seeds.

Matched budget everywhere: strength = local-loss at gamma=1; the operating point
is fixed at (h,dt)=(0.5,0.5) except in Run D. FN is the unitary baseline; every
Lindblad method uses the sparse matrix-free path (validated == dense to 1e-8).
"""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import _paths  # noqa: F401
import numpy as np

from _paths import RESULTS_DIR
from qrc import diagnostics as diag
from qrc import dissipators as dsp
from qrc import liouvillian as dense_lio
from qrc import readout, reservoirs as res, tasks
from qrc.operators import sminus
from qrc.reservoirs import (LindbladReservoir, ising_xx_hamiltonian,
                            transverse_drive)
from qrc.sparse_evolve import SparseLindbladReservoir

GAMMA = 1.0
ALL_METHODS = ("FN", "CD_paper", "A1_heterogeneous", "B1_dephasing",
               "B2_thermal", "B3_collective", "B4_loss_exchange", "B5_pair")
SCALE_METHODS = ("FN", "CD_paper", "A1_heterogeneous", "B1_dephasing",
                 "B3_collective")
DENSE_DIAG_NMAX = 6  # dense Liouvillian eig feasible up to N=6 on 32 GB


# --- task-constant knobs (do NOT vary across the protocol) ----------------
@dataclass(frozen=True)
class TaskCfg:
    stm_delays: tuple = tuple(range(1, 21))
    parity_delays: tuple = tuple(range(1, 8))
    parity_n_virtual: int = 15
    narma_order: int = 10
    input_scale: float = 0.2
    mg_sample_every: int = 3
    mg_horizon: int = 150  # closed-loop forecast length (matches run_final_parity_mg)


TCFG = TaskCfg()


@dataclass(frozen=True)
class Job:
    block: str
    N: int
    method: str
    task: str
    seed: int
    h: float = 0.5
    dt: float = 0.5
    wash: int = 200
    train: int = 600
    test: int = 400

    @property
    def total_len(self) -> int:
        return self.wash + self.train + self.test


def deterministic_seeds(n: int) -> list[int]:
    # nested: seeds(32) == seeds(64)[:32], so higher-seed tasks extend lower ones
    return [int(x) for x in np.random.default_rng(2024).integers(0, 2**31 - 1, n)]


# --- reservoir construction (mirrors experiments/cluster/scaling_job.py) ---
def build_jumps(method: str, J: np.ndarray, N: int, target_strength: float,
                rng: np.random.Generator):
    if method == "CD_paper":
        return dsp.normalize_jump_strength(dsp.local_loss(N, 1.0), target_strength)
    if method == "A1_heterogeneous":
        rates = dsp.loguniform_rates(N, 0.1, 10.0, rng, mean=1.0)
        return [(sminus(i, N), float(rates[i])) for i in range(N)]
    if method == "B1_dephasing":
        return dsp.normalize_jump_strength(dsp.dephasing(N, 1.0), target_strength)
    if method == "B2_thermal":
        return dsp.normalize_jump_strength(dsp.thermal(N, 1.0, 0.3), target_strength)
    if method == "B3_collective":
        return dsp.normalize_jump_strength(dsp.collective_loss(N, 1.0), target_strength)
    if method == "B4_loss_exchange":
        edges = dsp.graph_edges(J)
        return (dsp.normalize_jump_strength(dsp.local_loss(N, 1.0), 0.4 * target_strength)
                + dsp.normalize_jump_strength(dsp.exchange(N, 1.0, edges), 0.6 * target_strength))
    if method == "B5_pair":
        edges = [(i, j) for i in range(N) for j in range(i + 1, N)
                 if abs(J[i, j]) > 1e-9]
        return dsp.normalize_jump_strength(dsp.pair_loss(N, 1.0, edges), target_strength)
    raise ValueError(f"unknown Lindblad method {method}")


def make_reservoir(method: str, J: np.ndarray, N: int, rng: np.random.Generator,
                   h: float, dt: float, dense: bool = False):
    if method == "FN":
        return res.FujiNakajimaReservoir(N, J, h, dt)
    H0 = ising_xx_hamiltonian(J, h, N)
    Hx = transverse_drive(N)
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
    jumps = build_jumps(method, J, N, target, rng)
    if dense:  # only for diagnostics (needs dense liouvillian(s)/steady_state)
        return LindbladReservoir.from_terms(N, H0 + h * Hx, h * Hx, jumps, dt,
                                            cache_propagators=False, quantize=None)
    return SparseLindbladReservoir.from_terms(N, H0 + h * Hx, h * Hx, jumps, dt)


# --- task metrics (unchanged math; take the job for wash/train/test) -------
def _fit_masks(y: np.ndarray, job: Job):
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[:job.train] = True
    test_mask[job.train:job.train + job.test] = True
    valid = ~np.isnan(y)
    return train_mask & valid, test_mask & valid


def stm_mc(X: np.ndarray, post: np.ndarray, job: Job) -> float:
    Xb = readout.add_bias(X)
    total = 0.0
    for delay in TCFG.stm_delays:
        y = tasks.delayed_target(post, delay)
        a, b = _fit_masks(y, job)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        total += readout.capacity(y[b], readout.predict(Xb[b], w))
    return float(total)


def narma_nmse(X: np.ndarray, post: np.ndarray, job: Job) -> float:
    Xb = readout.add_bias(X)
    y = tasks.narma_target(post, order=TCFG.narma_order, input_scale=TCFG.input_scale)
    a, b = _fit_masks(y, job)
    w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
    return float(readout.nmse(y[b], readout.predict(Xb[b], w)))


def parity_capacity(X: np.ndarray, post: np.ndarray, job: Job) -> float:
    Xb = readout.add_bias(X)
    total = 0.0
    for delay in TCFG.parity_delays:
        y = tasks.parity_target(post, delay)
        a, b = _fit_masks(y, job)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        total += readout.capacity(y[b], readout.predict(Xb[b], w))
    return float(total)


def _observable_stack(observables):
    return np.stack([o.matrix for o in observables])


def _feature_row(rho: np.ndarray, Omats: np.ndarray) -> np.ndarray:
    return np.real(np.einsum("kij,ji->k", Omats, rho))


def mg_mse(reservoir, observables, series: np.ndarray, job: Job) -> float:
    n_train = job.wash + job.train
    drive = series[:n_train]
    Omats = _observable_stack(observables)
    rho = reservoir.initial_state()
    rows = []
    for k, s in enumerate(drive):
        rho = reservoir.step(rho, float(s))
        if k >= job.wash:
            rows.append(_feature_row(rho, Omats))
    X = np.asarray(rows)
    post = drive[job.wash:]
    y = np.empty(len(post), dtype=float)
    y[:-1] = post[1:]
    y[-1] = series[n_train]
    w = readout.train_readout(readout.add_bias(X)[:-1], y[:-1], ridge=1e-8)

    preds = []
    for _ in range(TCFG.mg_horizon):
        feat = _feature_row(rho, Omats)
        cur = float(np.clip(np.dot(np.append(feat, 1.0), w), 0.0, 1.0))
        preds.append(cur)
        rho = reservoir.step(rho, cur)
    truth = series[n_train:n_train + TCFG.mg_horizon]
    return float(np.mean((truth - np.asarray(preds)) ** 2))


def compute_diagnostics(method: str, J: np.ndarray, N: int, h: float, dt: float,
                        rng: np.random.Generator) -> dict:
    """Static Liouvillian diagnostics. unitality_defect at any N; the dense-eig
    diagnostics (gap/mixing/separability) only for N<=DENSE_DIAG_NMAX. Each is
    wrapped so a single failure never aborts the run."""
    out: dict = {"unitality_defect": None, "spectral_gap": None,
                 "mixing_time": None, "separability_01": None}
    if method == "FN":
        return out  # unitary baseline has no dissipator diagnostics
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
    jumps = build_jumps(method, J, N, target, rng)  # build ONCE, reuse everywhere
    try:
        out["unitality_defect"] = diag.unitality_defect(jumps)
    except Exception as exc:  # noqa: BLE001
        out["unitality_defect_error"] = f"{type(exc).__name__}: {exc}"
    if N <= DENSE_DIAG_NMAX:
        try:
            H0 = ising_xx_hamiltonian(J, h, N)
            Hx = transverse_drive(N)
            dres = LindbladReservoir.from_terms(N, H0 + h * Hx, h * Hx, jumps, dt,
                                                cache_propagators=False, quantize=None)
            L = dres.liouvillian(0.5)
            out["spectral_gap"] = diag.spectral_gap(L)
            out["mixing_time"] = diag.mixing_time(L)
            sep = diag.stationary_separability(dres, [0.0, 1.0])
            out["separability_01"] = float(sep[0, 1])
        except Exception as exc:  # noqa: BLE001
            out["dense_diag_error"] = f"{type(exc).__name__}: {exc}"
    return out


def value_for_job(job: Job):
    problem_rng = np.random.default_rng(job.seed)
    J = res.random_couplings(job.N, 1.0, problem_rng)
    method_rng = np.random.default_rng(job.seed + 1)

    if job.task == "diagnostics":
        return compute_diagnostics(job.method, J, job.N, job.h, job.dt, method_rng)

    reservoir = make_reservoir(job.method, J, job.N, method_rng, job.h, job.dt)
    observables = readout.pauli_observables(job.N, max_weight=2)

    if job.task == "stm":
        inputs = tasks.stm_inputs(job.total_len, problem_rng)
        X = reservoir.run(inputs, observables, washout=job.wash)
        return stm_mc(X, inputs[job.wash:], job)
    if job.task == "narma":
        inputs = tasks.narma_inputs(job.total_len, problem_rng)
        X = reservoir.run(inputs, observables, washout=job.wash)
        return narma_nmse(X, inputs[job.wash:], job)
    if job.task == "parity":
        inputs = tasks.parity_inputs(job.total_len, problem_rng)
        X = reservoir.run(inputs, observables, washout=job.wash,
                          n_virtual=TCFG.parity_n_virtual)
        return parity_capacity(X, inputs[job.wash:], job)
    if job.task == "mg":
        n_samples = job.wash + job.train + TCFG.mg_horizon + 5
        series = tasks.mackey_glass_series(n_samples, tau=17,
                                           sample_every=TCFG.mg_sample_every,
                                           discard=500, rng=problem_rng)
        series = tasks.rescale_unit(series, 0.0, 1.0)
        return mg_mse(reservoir, observables, series, job)
    raise ValueError(f"unknown task {job.task}")


# --- checkpointing (atomic, restart-safe, crash-tolerant) ------------------
def output_path(outdir: Path, job: Job) -> Path:
    return outdir / (
        f"{job.block}__{job.task}_N{job.N}_{job.method}_s{job.seed}"
        f"_h{job.h:g}_dt{job.dt:g}_L{job.wash}-{job.train}-{job.test}.json"
    )


def _is_done(old: dict) -> bool:
    v = old.get("value")
    if v is not None and np.isfinite(v):
        return True
    return old.get("diagnostics") is not None


def run_one(job: Job, outdir_s: str) -> dict:
    outdir = Path(outdir_s)
    outdir.mkdir(parents=True, exist_ok=True)
    out = output_path(outdir, job)
    if out.exists():
        try:
            old = json.loads(out.read_text())
        except (json.JSONDecodeError, OSError):
            old = {}
        if _is_done(old):
            return {**asdict(job), "status": "skip", "value": old.get("value"),
                    "path": str(out)}
        # corrupt / incomplete checkpoint -> recompute

    t0 = time.time()
    try:
        result = value_for_job(job)
    except Exception as exc:  # keep the unattended run alive; write NO checkpoint
        return {**asdict(job), "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_s": time.time() - t0, "path": str(out)}
    runtime = time.time() - t0

    payload = {**asdict(job), "runtime_s": runtime,
               "backend": "fn_unitary" if job.method == "FN" else "sparse"}
    if isinstance(result, dict):
        payload["value"] = None
        payload["diagnostics"] = result
    else:
        payload["value"] = float(result)

    tmp = out.with_suffix(out.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out)
    return {**asdict(job), "status": "done", "value": payload["value"],
            "runtime_s": runtime, "path": str(out)}


# --- manifest (priority order: complete paper first, N=8 reach last) -------
def build_manifest(seed_pool: int = 64) -> list[Job]:
    S = deterministic_seeds(seed_pool)
    jobs: list[Job] = []

    # A - dissociation table (N=5, all methods)
    for task, ns in (("stm", 32), ("narma", 32), ("parity", 32), ("mg", 64)):
        for m in ALL_METHODS:
            for sd in S[:ns]:
                jobs.append(Job("A_table", 5, m, task, sd))

    # B - memory scaling N<=7 (cheap-to-moderate part of the headline)
    for N in (4, 5, 6, 7):
        for m in SCALE_METHODS:
            for task in ("stm", "narma"):
                for sd in S[:30]:
                    jobs.append(Job("B_scale", N, m, task, sd))

    # D - operating-point robustness (N=5)
    for (h, dt) in ((0.5, 0.5), (1.0, 0.5), (0.5, 1.0)):
        for m in ("CD_paper", "B3_collective", "B5_pair", "B1_dephasing"):
            for task in ("stm", "parity"):
                for sd in S[:20]:
                    jobs.append(Job("D_oppoint", 5, m, task, sd, h=h, dt=dt))

    # C - sequence-length control (N=6, CD vs B3, STM at 4 lengths)
    for (w, tr, te) in ((100, 300, 200), (200, 600, 400),
                        (400, 1200, 800), (600, 1800, 1200)):
        for m in ("CD_paper", "B3_collective"):
            for sd in S[:20]:
                jobs.append(Job("C_seqlen", 6, m, "stm", sd, wash=w, train=tr, test=te))

    # E - diagnostics vs N (static; cheap)
    for N in (4, 5, 6, 7, 8):
        for m in ALL_METHODS:
            for sd in S[:10]:
                jobs.append(Job("E_diag", N, m, "diagnostics", sd))

    # B reach - N=8 (the expensive npj point; last so a crash never costs the core)
    for m in SCALE_METHODS:
        for task in ("stm", "narma"):
            for sd in S[:20]:
                jobs.append(Job("B_scale", 8, m, task, sd))

    return jobs


def build_smoke_manifest() -> list[Job]:
    S = deterministic_seeds(2)
    jobs = []
    for m in ("CD_paper", "B3_collective"):
        for sd in S:
            jobs.append(Job("smoke", 4, m, "stm", sd, wash=5, train=16, test=10))
    jobs.append(Job("smoke", 4, "B3_collective", "parity", S[0], wash=5, train=16, test=10))
    jobs.append(Job("smoke", 4, "CD_paper", "mg", S[0], wash=5, train=16, test=10))
    jobs.append(Job("smoke", 5, "B3_collective", "diagnostics", S[0]))
    jobs.append(Job("smoke", 7, "B3_collective", "diagnostics", S[0]))  # unitality-only path
    return jobs


# --- runner ----------------------------------------------------------------
def parse_csv(value):
    return None if value is None else tuple(x.strip() for x in value.split(",") if x.strip())


def parse_shard(value):
    if value is None:
        return None
    i, k = (int(x) for x in value.split("/", 1))
    if k <= 0 or not (0 <= i < k):
        raise ValueError("--shard must be i/k with 0 <= i < k")
    return i, k


def format_progress(r: dict) -> str:
    v = r.get("value")
    vs = "diag" if (r.get("task") == "diagnostics" and r.get("status") == "done") \
        else ("nan" if v is None else f"{float(v):.5g}")
    rt = r.get("runtime_s")
    rts = "" if rt is None else f" {rt:.1f}s"
    tail = f"  !{r['error']}" if r.get("status") == "error" else ""
    return (f"{r['status']:5s} {r['block']} N={r['N']} {r['method']} {r['task']} "
            f"s={r['seed']} h{r.get('h')} dt{r.get('dt')} L{r.get('wash')}-{r.get('train')}-{r.get('test')} "
            f"={vs}{rts}{tail}")


def run_jobs(jobs: list[Job], outdir: Path, workers: int) -> None:
    if workers <= 1:
        for job in jobs:
            print(format_progress(run_one(job, str(outdir))), flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, job, str(outdir)): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:  # worker died (e.g. OOM) -> record, keep going
                result = {**asdict(job), "status": "error",
                          "error": f"worker died: {type(exc).__name__}: {exc}"}
            print(format_progress(result), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", help="run every k-th job, as i/k (0-based)")
    ap.add_argument("--blocks", help="comma-separated block filter, e.g. A_table,B_scale")
    ap.add_argument("--outdir")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        outdir = Path(args.outdir) if args.outdir else Path(RESULTS_DIR) / "final_protocol_smoke"
        jobs = build_smoke_manifest()
    else:
        outdir = Path(args.outdir) if args.outdir else Path(RESULTS_DIR) / "final_protocol"
        jobs = build_manifest()
        blocks = parse_csv(args.blocks)
        if blocks:
            jobs = [j for j in jobs if j.block in set(blocks)]

    shard = parse_shard(args.shard)
    if shard:
        i, k = shard
        jobs = [j for idx, j in enumerate(jobs) if idx % k == i]

    outdir.mkdir(parents=True, exist_ok=True)
    print(f"outdir={outdir} jobs={len(jobs)} workers={args.workers}", flush=True)
    run_jobs(jobs, outdir, args.workers)
    print("ALL JOBS PROCESSED", flush=True)


if __name__ == "__main__":
    main()

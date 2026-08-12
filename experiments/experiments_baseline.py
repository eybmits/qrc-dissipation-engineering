"""Phase 0 - reproduce the two baseline QRC models from Sannia et al.

Models
------
* FN  - Fuji-Nakajima unitary reservoir (input by reset).
* CD  - continuous-dissipation Lindblad reservoir (uniform local loss).

Tasks
-----
STM, NARMA, parity (with time multiplexing), Mackey-Glass autonomous forecast.

Everything is driven by :class:`BaselineConfig`.  The defaults are *reduced*
(few realizations, moderate sequence lengths) so a validation run finishes in
minutes on a laptop; the ``paper`` preset matches the paper's scale (N=5,
washout/train/test = 1000, 100 realizations) for a cluster run.

Raw results are saved to ``results/`` as ``.npz`` plus a JSON metadata sidecar
(config, seeds, runtime); plots/tables are produced by ``analysis.py``.
"""

from __future__ import annotations

import _paths  # noqa: F401  (sets sys.path + result dirs)

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from qrc import reservoirs as res
from qrc import readout
from qrc import tasks
from qrc import diagnostics as diag
from _paths import RESULTS_DIR


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    n_qubits: int = 5
    J_s: float = 1.0                 # coupling scale; we work in units of J_s
    h: float = 1.0                   # transverse field / drive coefficient
    dt: float = 1.0                  # evolution time per input step
    washout: int = 200
    train_len: int = 600
    test_len: int = 400
    n_realizations: int = 5
    seed: int = 12345
    quantize: Optional[int] = 64     # input quantisation for the CD propagator cache
    ridge: float = 1e-8              # tiny ridge for numerical stability of SVD readout
    max_weight: int = 2              # Pauli-string readout weight (1 and 2)

    @property
    def total_len(self) -> int:
        return self.washout + self.train_len + self.test_len


PRESETS: Dict[str, dict] = {
    "demo": dict(n_realizations=3, washout=100, train_len=300, test_len=200),
    "validation": dict(n_realizations=5, washout=200, train_len=600, test_len=400),
    "paper": dict(n_realizations=100, washout=1000, train_len=1000, test_len=1000),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_reservoir(kind: str, J: np.ndarray, cfg: BaselineConfig, gamma: float = 1.0,
                   cache: bool = False):
    if kind == "FN":
        return res.FujiNakajimaReservoir(cfg.n_qubits, J, cfg.h, cfg.dt)
    if kind == "CD":
        return res.continuous_dissipation(
            cfg.n_qubits, J, cfg.h, gamma, cfg.dt,
            cache_propagators=cache, quantize=cfg.quantize,
        )
    raise ValueError(kind)


def _split(X: np.ndarray, cfg: BaselineConfig):
    """Train/test split of post-washout features (val folded into train here)."""
    tr = slice(0, cfg.train_len)
    te = slice(cfg.train_len, cfg.train_len + cfg.test_len)
    return tr, te


def evaluate_delays(
    X: np.ndarray,
    post_inputs: np.ndarray,
    target_fn,
    delays: List[int],
    cfg: BaselineConfig,
    metric: str = "capacity",
) -> Dict[int, float]:
    """For each delay, build target, drop warm-up NaNs, fit readout on the train
    split and score on the test split."""
    Xb = readout.add_bias(X)
    tr, te = _split(X, cfg)
    out: Dict[int, float] = {}
    for d in delays:
        y = target_fn(post_inputs, d)
        valid = ~np.isnan(y)
        tr_mask = np.zeros(len(y), bool); tr_mask[tr] = True; tr_mask &= valid
        te_mask = np.zeros(len(y), bool); te_mask[te] = True; te_mask &= valid
        r = readout.fit_and_score(
            Xb[tr_mask], y[tr_mask], Xb[te_mask], y[te_mask],
            metric=metric, ridge=cfg.ridge,
        )
        out[d] = r.test_metric
    return out


# --------------------------------------------------------------------------
# STM
# --------------------------------------------------------------------------

def run_stm(cfg: BaselineConfig, gammas: List[float], delays: List[int]) -> dict:
    """Short-term memory: y_k = s_{k-tau}.  Compares FN against CD for several
    gamma.  Returns per-realization capacity[model/gamma][tau]."""
    rng_master = np.random.default_rng(cfg.seed)
    seeds = rng_master.integers(0, 2 ** 31 - 1, size=cfg.n_realizations)
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)

    models = {"FN": None}
    for g in gammas:
        models[f"CD_g{g:g}"] = g

    cap = {m: np.zeros((cfg.n_realizations, len(delays))) for m in models}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(cfg.n_qubits, cfg.J_s, rng)
        inputs = tasks.stm_inputs(cfg.total_len, rng)
        post = inputs[cfg.washout:]
        for m, g in models.items():
            kind = "FN" if m == "FN" else "CD"
            reservoir = make_reservoir(kind, J, cfg, gamma=(g or 1.0))
            X = reservoir.run(inputs, obs, washout=cfg.washout, n_virtual=1)
            caps = evaluate_delays(X, post, tasks.delayed_target, delays, cfg, "capacity")
            cap[m][r_i] = [caps[d] for d in delays]
        print(f"  [STM] realization {r_i + 1}/{cfg.n_realizations} done "
              f"({time.time() - t0:.0f}s)")
    runtime = time.time() - t0

    result = dict(
        task="STM", delays=np.asarray(delays), models=list(models),
        capacity={m: cap[m] for m in models},
        seeds=np.asarray(seeds), config=asdict(cfg), runtime_s=runtime,
    )
    _save("stm", result)
    return result


# --------------------------------------------------------------------------
# NARMA
# --------------------------------------------------------------------------

def run_narma(cfg: BaselineConfig, gammas: List[float], orders: List[int],
              input_scale: float = 0.2) -> dict:
    rng_master = np.random.default_rng(cfg.seed + 1)
    seeds = rng_master.integers(0, 2 ** 31 - 1, size=cfg.n_realizations)
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)

    models = {"FN": None}
    for g in gammas:
        models[f"CD_g{g:g}"] = g

    nmse = {m: np.zeros((cfg.n_realizations, len(orders))) for m in models}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(cfg.n_qubits, cfg.J_s, rng)
        inputs = tasks.narma_inputs(cfg.total_len, rng)
        post = inputs[cfg.washout:]
        target_fn = lambda s, n: tasks.narma_target(s, order=n, input_scale=input_scale)
        for m, g in models.items():
            kind = "FN" if m == "FN" else "CD"
            reservoir = make_reservoir(kind, J, cfg, gamma=(g or 1.0))
            X = reservoir.run(inputs, obs, washout=cfg.washout, n_virtual=1)
            scores = evaluate_delays(X, post, target_fn, orders, cfg, "nmse")
            nmse[m][r_i] = [scores[n] for n in orders]
        print(f"  [NARMA] realization {r_i + 1}/{cfg.n_realizations} done "
              f"({time.time() - t0:.0f}s)")
    runtime = time.time() - t0
    result = dict(
        task="NARMA", orders=np.asarray(orders), models=list(models),
        nmse={m: nmse[m] for m in models}, input_scale=input_scale,
        seeds=np.asarray(seeds), config=asdict(cfg), runtime_s=runtime,
    )
    _save("narma", result)
    return result


# --------------------------------------------------------------------------
# Parity (time multiplexing)
# --------------------------------------------------------------------------

def run_parity(cfg: BaselineConfig, gammas: List[float], delays: List[int],
               n_virtual: int = 15) -> dict:
    rng_master = np.random.default_rng(cfg.seed + 2)
    seeds = rng_master.integers(0, 2 ** 31 - 1, size=cfg.n_realizations)
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)

    models = {"FN": None}
    for g in gammas:
        models[f"CD_g{g:g}"] = g

    cap = {m: np.zeros((cfg.n_realizations, len(delays))) for m in models}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(cfg.n_qubits, cfg.J_s, rng)
        inputs = tasks.parity_inputs(cfg.total_len, rng)
        post = inputs[cfg.washout:]
        for m, g in models.items():
            kind = "FN" if m == "FN" else "CD"
            # binary input -> propagator cache reused across the 2 levels
            reservoir = make_reservoir(kind, J, cfg, gamma=(g or 1.0), cache=True)
            X = reservoir.run(inputs, obs, washout=cfg.washout, n_virtual=n_virtual)
            caps = evaluate_delays(X, post, tasks.parity_target, delays, cfg, "capacity")
            cap[m][r_i] = [caps[d] for d in delays]
        print(f"  [PARITY] realization {r_i + 1}/{cfg.n_realizations} done "
              f"({time.time() - t0:.0f}s)")
    runtime = time.time() - t0
    result = dict(
        task="parity", delays=np.asarray(delays), models=list(models),
        n_virtual=n_virtual, capacity={m: cap[m] for m in models},
        seeds=np.asarray(seeds), config=asdict(cfg), runtime_s=runtime,
    )
    _save("parity", result)
    return result


# --------------------------------------------------------------------------
# Mackey-Glass autonomous forecasting
# --------------------------------------------------------------------------

def run_mackey_glass(cfg: BaselineConfig, gammas: List[float],
                     horizon: int = 150, sample_every: int = 3) -> dict:
    """Train one-step-ahead prediction, then roll out autonomously by feeding the
    model's own prediction back in.  Reports MSE over the first ``horizon``
    autonomous points."""
    rng_master = np.random.default_rng(cfg.seed + 3)
    seeds = rng_master.integers(0, 2 ** 31 - 1, size=cfg.n_realizations)
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)

    models = {"FN": None}
    for g in gammas:
        models[f"CD_g{g:g}"] = g

    mse = {m: np.zeros(cfg.n_realizations) for m in models}
    traces = {m: [] for m in models}
    t0 = time.time()
    n_train = cfg.washout + cfg.train_len
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(cfg.n_qubits, cfg.J_s, rng)
        series = tasks.mackey_glass_series(
            n_train + horizon + 5, tau=17, sample_every=sample_every,
            discard=500, rng=rng)
        series = tasks.rescale_unit(series, 0.0, 1.0)
        drive = series[:n_train]                      # teacher-forced inputs
        for m, g in models.items():
            kind = "FN" if m == "FN" else "CD"
            reservoir = make_reservoir(kind, J, cfg, gamma=(g or 1.0))
            # teacher-forced phase: features for one-step-ahead targets
            X = reservoir.run(drive, obs, washout=cfg.washout, n_virtual=1)
            post = drive[cfg.washout:]
            y = np.empty(len(post)); y[:-1] = post[1:]; y[-1] = series[n_train]
            Xb = readout.add_bias(X)
            w = readout.train_readout(Xb[:-1], y[:-1], ridge=cfg.ridge)
            # autonomous rollout from the last teacher-forced state.
            # Re-evolve over `drive` so rho encodes up to drive[-1]; then read
            # out the one-step-ahead prediction, feed it back, and repeat.
            rho = reservoir.initial_state()
            for s in drive:
                rho = reservoir.step(rho, float(s))
            preds = []
            for _ in range(horizon):
                feat = readout.add_bias(
                    readout.features_from_states([rho], obs))[0]
                cur = float(np.clip(feat @ w, 0.0, 1.0))  # predicts next value
                preds.append(cur)
                rho = reservoir.step(rho, cur)             # feed prediction back
            truth = series[n_train:n_train + horizon]
            preds = np.asarray(preds)
            mse[m][r_i] = float(np.mean((truth - preds) ** 2))
            if r_i == 0:
                traces[m] = preds.tolist()
        print(f"  [MG] realization {r_i + 1}/{cfg.n_realizations} done "
              f"({time.time() - t0:.0f}s)")
    runtime = time.time() - t0
    # store one example truth for plotting
    series_example = series[n_train:n_train + horizon]
    result = dict(
        task="mackey_glass", horizon=horizon, sample_every=sample_every,
        models=list(models), mse={m: mse[m] for m in models},
        example_pred={m: np.asarray(traces[m]) for m in models},
        example_truth=np.asarray(series_example),
        seeds=np.asarray(seeds), config=asdict(cfg), runtime_s=runtime,
    )
    _save("mackey_glass", result)
    return result


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def _save(name: str, result: dict) -> None:
    """Save raw arrays (.npz) + JSON metadata sidecar."""
    flat = {}
    meta = {}
    for k, v in result.items():
        if isinstance(v, dict) and all(isinstance(x, np.ndarray) for x in v.values()):
            for kk, vv in v.items():
                flat[f"{k}__{kk}"] = vv
            meta.setdefault("_dict_keys", {})[k] = list(v.keys())
        elif isinstance(v, np.ndarray):
            flat[k] = v
        else:
            meta[k] = v
    np.savez(f"{RESULTS_DIR}/{name}.npz", **flat)
    with open(f"{RESULTS_DIR}/{name}.json", "w") as f:
        json.dump(_jsonable(meta), f, indent=2)
    print(f"  saved results/{name}.npz + results/{name}.json")


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def make_config(preset: str = "validation", **overrides) -> BaselineConfig:
    params = dict(PRESETS.get(preset, {}))
    params.update(overrides)
    return BaselineConfig(**params)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Baseline (Phase 0) experiments")
    p.add_argument("task", choices=["stm", "narma", "parity", "mackey_glass", "all"])
    p.add_argument("--preset", default="validation", choices=list(PRESETS))
    p.add_argument("--gammas", type=float, nargs="+",
                   default=[0.1, 0.3, 1.0, 3.0])
    p.add_argument("--max-delay", type=int, default=15)
    args = p.parse_args()

    cfg = make_config(args.preset)
    print(f"config: {cfg}")
    delays = list(range(1, args.max_delay + 1))

    if args.task in ("stm", "all"):
        run_stm(cfg, args.gammas, delays)
    if args.task in ("narma", "all"):
        run_narma(cfg, args.gammas, orders=[2, 5, 10])
    if args.task in ("parity", "all"):
        run_parity(cfg, args.gammas, delays=list(range(1, 8)))
    if args.task in ("mackey_glass", "all"):
        run_mackey_glass(cfg, args.gammas)

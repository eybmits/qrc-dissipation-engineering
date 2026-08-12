"""Second-Hamiltonian laptop generality smoke check.

This is a reduced-scale check for whether the dissipator roles survive a
Hamiltonian change. It compares the default all-to-all XX + Z-field reservoir to
an alternative all-to-all ZZ + X-field reservoir with a Z input drive.

The goal is not paper-scale statistics; it is to close the laptop-doable
generality gap before the cluster run.  The check deliberately uses N=4 and short
sequences so it finishes locally.
"""
import _paths  # noqa: F401
import json
import time
from pathlib import Path

import numpy as np

from qrc import dissipators as dsp
from qrc import readout, reservoirs as res, tasks
from qrc.operators import pauli_op, two_site_pauli
from qrc.reservoirs import LindbladReservoir


N = 4
N_REAL = 12
H = 0.5
DT = 0.5
GAMMA = 1.0
QUANTIZE = 16
WASH, TRAIN, TEST = 60, 180, 120
DELAYS_STM = list(range(1, 13))
DELAYS_PARITY = list(range(1, 7))
METHODS = ["CD_paper", "A1_heterogeneous", "B3_collective", "B5_pair", "B1_dephasing"]


def z_drive(n: int) -> np.ndarray:
    d = 2 ** n
    out = np.zeros((d, d), dtype=complex)
    for i in range(n):
        out += pauli_op("z", i, n)
    return out


def x_field(n: int) -> np.ndarray:
    d = 2 ** n
    out = np.zeros((d, d), dtype=complex)
    for i in range(n):
        out += pauli_op("x", i, n)
    return out


def zz_hamiltonian(J: np.ndarray, h: float, n: int) -> np.ndarray:
    d = 2 ** n
    H0 = np.zeros((d, d), dtype=complex)
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0:
                H0 += J[i, j] * two_site_pauli("z", i, j, n)
    H0 += h * x_field(n)
    return H0


def hamiltonian_terms(kind: str, J: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if kind == "xx_zfield_xdrive":
        drive = res.transverse_drive(n)
        return res.ising_xx_hamiltonian(J, H, n) + H * drive, H * drive
    if kind == "zz_xfield_zdrive":
        drive = z_drive(n)
        return zz_hamiltonian(J, H, n) + H * drive, H * drive
    raise ValueError(kind)


def jumps_for(name: str, J: np.ndarray, n: int, target: float, rng: np.random.Generator):
    if name == "CD_paper":
        return dsp.normalize_jump_strength(dsp.local_loss(n, 1.0), target)
    if name == "A1_heterogeneous":
        rates = dsp.loguniform_rates(n, 0.1, 10.0, rng, mean=1.0)
        return [(res.sminus(i, n), float(rates[i])) for i in range(n)]
    if name == "B3_collective":
        return dsp.normalize_jump_strength(dsp.collective_loss(n, 1.0), target)
    if name == "B5_pair":
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(J[i, j]) > 1e-9]
        return dsp.normalize_jump_strength(dsp.pair_loss(n, 1.0, edges), target)
    if name == "B1_dephasing":
        return dsp.normalize_jump_strength(dsp.dephasing(n, 1.0), target)
    raise ValueError(name)


def make_reservoir(kind: str, method: str, J: np.ndarray, seed: int):
    H_static, H_drive = hamiltonian_terms(kind, J, N)
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
    jumps = jumps_for(method, J, N, target, np.random.default_rng(seed + 1))
    return LindbladReservoir.from_terms(N, H_static, H_drive, jumps, DT, quantize=QUANTIZE)


def capacity_sum(X: np.ndarray, post: np.ndarray, delays: list[int], target_fn) -> float:
    Xb = readout.add_bias(X)
    tr = slice(0, TRAIN)
    te = slice(TRAIN, TRAIN + TEST)
    total = 0.0
    for delay in delays:
        y = target_fn(post, delay)
        a = np.zeros(len(y), dtype=bool)
        b = np.zeros(len(y), dtype=bool)
        a[tr] = True
        b[te] = True
        a &= ~np.isnan(y)
        b &= ~np.isnan(y)
        w = readout.train_readout(Xb[a], y[a], ridge=1e-8)
        total += readout.capacity(y[b], readout.predict(Xb[b], w))
    return float(total)


def paired_summary(values: dict[str, list[float]], ref: str = "CD_paper") -> dict[str, dict[str, float]]:
    ref_v = np.asarray(values[ref], dtype=float)
    out = {}
    for method, vals in values.items():
        v = np.asarray(vals, dtype=float)
        diff = v - ref_v
        se = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        out[method] = {
            "mean": float(v.mean()),
            "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
            "diff_vs_cd": float(diff.mean()),
            "diff_se": se,
            "sigma_vs_cd": float(diff.mean() / (se + 1e-12)),
            "pct_vs_cd": float(100.0 * diff.mean() / (ref_v.mean() + 1e-12)),
            "wins_vs_cd": int(np.sum(diff > 0)),
        }
    return out


def main():
    total = WASH + TRAIN + TEST
    obs = readout.pauli_observables(N, max_weight=2)
    rng0 = np.random.default_rng(240628)
    seeds = rng0.integers(0, 2 ** 31 - 1, N_REAL)
    kinds = ["xx_zfield_xdrive", "zz_xfield_zdrive"]
    raw = {
        kind: {
            task: {method: [] for method in METHODS}
            for task in ["STM", "parity"]
        }
        for kind in kinds
    }
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(N, 1.0, rng)
        stm_in = tasks.stm_inputs(total, rng)
        par_in = tasks.parity_inputs(total, rng)
        for kind in kinds:
            for method in METHODS:
                reservoir = make_reservoir(kind, method, J, int(sd))
                X_stm = reservoir.run(stm_in, obs, washout=WASH)
                raw[kind]["STM"][method].append(
                    capacity_sum(X_stm, stm_in[WASH:], DELAYS_STM, tasks.delayed_target)
                )
                X_par = reservoir.run(par_in, obs, washout=WASH, n_virtual=4)
                raw[kind]["parity"][method].append(
                    capacity_sum(X_par, par_in[WASH:], DELAYS_PARITY, tasks.parity_target)
                )
        print(f"  seed {r_i + 1}/{N_REAL} done ({time.time() - t0:.0f}s)")

    summary = {
        kind: {task: paired_summary(raw[kind][task]) for task in raw[kind]}
        for kind in raw
    }
    out = {
        "n": N,
        "n_real": N_REAL,
        "h": H,
        "dt": DT,
        "gamma": GAMMA,
        "quantize": QUANTIZE,
        "wash_train_test": [WASH, TRAIN, TEST],
        "methods": METHODS,
        "hamiltonians": kinds,
        "seeds": seeds.tolist(),
        "raw": raw,
        "summary": summary,
        "runtime_s": time.time() - t0,
    }
    path = Path("../results/generality_hamiltonian.json")
    path.write_text(json.dumps(out, indent=2))
    print("\n=== Second-Hamiltonian generality summary ===")
    for kind in kinds:
        print(f"\n{kind}")
        for task in ["STM", "parity"]:
            print(f"  {task}")
            for method in METHODS:
                s = summary[kind][task][method]
                print(
                    f"    {method:16s} mean={s['mean']:.3f} "
                    f"diff={s['diff_vs_cd']:+.3f} "
                    f"({s['pct_vs_cd']:+.0f}%, {s['sigma_vs_cd']:+.1f}σ, "
                    f"wins={s['wins_vs_cd']}/{N_REAL})"
                )


if __name__ == "__main__":
    main()

"""Phase 3 - Track B: JUMP-OPERATOR engineering.

Keeps local rates comparable but changes the dissipative *process*:

    B0 local loss      L_i = sigma^-_i                     (== CD baseline)
    B1 dephasing       L_i = sigma^z_i
    B2 local gain/loss L_i,down = sigma^-_i, L_i,up = sigma^+_i
    B3 collective      L = sum_i c_i sigma^-_i
    B4 exchange        L_ij = sigma^-_i sigma^+_j  (graph edges)
    B5 pair (optional) L_ij = sigma^-_i sigma^-_j

CONTROL: total dissipative strength is held comparable across families
(collective coefficients normalised to ||c||^2 = N; per-channel rates matched),
so improvements are not just "more dissipation".

>>> GATING <<<  Do not draw Track-B conclusions until the Phase-0 baseline
replication has passed qualitatively.  Wired to the same evaluation code; not run
as part of baseline validation.
"""
from __future__ import annotations

import _paths  # noqa: F401
import time
from typing import Dict, List

import numpy as np

from qrc import reservoirs as res
from qrc import readout, tasks
from qrc import dissipators as dsp
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive, LindbladReservoir
from experiments_baseline import BaselineConfig, evaluate_delays, make_config, _save


def build_jump_families(J: np.ndarray, n: int, gamma: float,
                        match_strength: bool = True) -> Dict[str, list]:
    """All Track-B jump families, each normalised to the SAME total dissipative
    strength as B0 local loss at rate ``gamma`` (``Σ_k rate_k Tr(L_k^dag L_k)``),
    so improvements cannot be attributed to stronger dissipation."""
    edges = dsp.graph_edges(J)
    families = {
        "B0_local": dsp.local_loss(n, gamma),
        "B1_dephasing": dsp.dephasing(n, gamma),
        "B2_thermal": dsp.thermal(n, gamma_down=gamma, gamma_up=0.3 * gamma),
        "B3_collective": dsp.collective_loss(n, gamma),
        "B4_exchange": dsp.exchange(n, gamma, edges),
    }
    if match_strength:
        target = dsp.jump_strength(families["B0_local"])
        families = {k: dsp.normalize_jump_strength(v, target)
                    for k, v in families.items()}
    return families


def make_reservoir_with_jumps(J, cfg: BaselineConfig, jumps):
    """CD-style reservoir (input-driven H) with an arbitrary static jump list."""
    n = cfg.n_qubits
    H0 = ising_xx_hamiltonian(J, cfg.h, n)
    Hx = transverse_drive(n)
    H_static = H0 + cfg.h * Hx
    H_drive = cfg.h * Hx
    return LindbladReservoir.from_terms(
        n, H_static, H_drive, jumps, cfg.dt, quantize=cfg.quantize)


def run_jump_engineering_stm(cfg: BaselineConfig, gamma: float = 1.0,
                             delays: List[int] | None = None) -> dict:
    delays = delays or list(range(1, 16))
    rng_master = np.random.default_rng(cfg.seed + 200)
    seeds = rng_master.integers(0, 2 ** 31 - 1, size=cfg.n_realizations)
    obs = readout.pauli_observables(cfg.n_qubits, max_weight=cfg.max_weight)

    family_names = ["B0_local", "B1_dephasing", "B2_thermal",
                    "B3_collective", "B4_exchange"]
    cap = {m: np.zeros((cfg.n_realizations, len(delays))) for m in family_names}
    t0 = time.time()
    for r_i, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd))
        J = res.random_couplings(cfg.n_qubits, cfg.J_s, rng)
        inputs = tasks.stm_inputs(cfg.total_len, rng)
        post = inputs[cfg.washout:]
        families = build_jump_families(J, cfg.n_qubits, gamma)
        for m in family_names:
            reservoir = make_reservoir_with_jumps(J, cfg, families[m])
            X = reservoir.run(inputs, obs, washout=cfg.washout, n_virtual=1)
            caps = evaluate_delays(X, post, tasks.delayed_target, delays, cfg, "capacity")
            cap[m][r_i] = [caps[d] for d in delays]
        print(f"  [TrackB-STM] realization {r_i + 1}/{cfg.n_realizations} "
              f"({time.time() - t0:.0f}s)")
    result = dict(task="trackB_stm", delays=np.asarray(delays),
                  models=family_names, gamma=gamma, capacity=cap,
                  seeds=np.asarray(seeds), config=cfg.__dict__,
                  runtime_s=time.time() - t0)
    _save("trackB_stm", result)
    return result


# Validated CD operating point from Phase 0 (STM optimum): h=0.5, dt=0.5.
# Track B compares jump-operator FAMILIES at this point, each normalised to the
# same total dissipative strength (Σ rate·Tr(L†L)) as B0 local loss at gamma.
STM_OPTIMUM = dict(h=0.5, dt=0.5, gamma=0.3)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Track B: jump-operator engineering (gate OPEN)")
    ap.add_argument("--preset", default="validation")
    ap.add_argument("--max-delay", type=int, default=15)
    ap.add_argument("--quantize", type=int, default=64)
    args = ap.parse_args()

    cfg = make_config(args.preset, h=STM_OPTIMUM["h"], dt=STM_OPTIMUM["dt"],
                      quantize=args.quantize)
    print(f"=== Track B | STM | operating point {STM_OPTIMUM} | {cfg} ===")
    print("Families matched to B0 total dissipative strength; paired per realization.")
    run_jump_engineering_stm(cfg, gamma=STM_OPTIMUM["gamma"],
                             delays=list(range(1, args.max_delay + 1)))

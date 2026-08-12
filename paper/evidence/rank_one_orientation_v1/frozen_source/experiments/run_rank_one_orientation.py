"""Frozen N=6 rank-one orientation intervention for the QRC manuscript.

The two dissipators are
    L_+    = sqrt(gamma) sum_i sigma_i^-
    L_perp = sqrt(gamma) (sum_{i=1}^3 sigma_i^- - sum_{i=4}^6 sigma_i^-).
They have the same rank, Kossakowski eigenvalues, Frobenius budget, coefficient
magnitudes and dark-subspace dimension.  Only their orientation relative to the
uniform input drive is changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import numpy as np
from scipy.sparse.linalg import expm_multiply
from scipy.stats import binomtest, pearsonr, spearmanr, t as student_t

from qrc import dissipators as dsp
from qrc import readout
from qrc.liouvillian import unvec, vec
from qrc.reservoirs import ising_xx_hamiltonian, transverse_drive
from qrc.sparse_evolve import SparseLindbladReservoir

VERSION = "rank-one-orientation-v1-2026-08-12"
N, H, DT, GAMMA = 6, 0.5, 0.5, 1.0
J_SCALE = math.sqrt(4.0 / (N - 1))
SEEDS = (
    956087733, 1375334633, 707736772, 1133846500, 365211353, 878523603,
    457552621, 363662622, 853972123, 1403843447, 151336801, 1991628836,
    1627319819, 336852480, 1454963355, 203675062, 93339074, 8147085,
    264759322, 16866769, 346211042, 1665106229, 1622806565, 1222562911,
)
COEFF = {
    "equal_phase": np.ones(N),
    "drive_orthogonal": np.array([1, 1, 1, -1, -1, -1], dtype=float),
}
CHECKS = (800, 1200, 1600)
TRACE_GATE, FEATURE_GATE = 1e-8, 2e-8
TRAIN, TEST, DELAYS, RIDGE = 600, 400, tuple(range(1, 21)), 1e-8
DELTA, KLAGS, ANCHORS, TAIL_START = 1e-3, 20, tuple(range(100, 541, 40)), 10
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "rank_one_orientation_v1"


def canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(obj: object) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def array_digest(a: np.ndarray) -> str:
    a = np.asarray(a)
    dt = "<c16" if np.iscomplexobj(a) else "<f8"
    b = np.ascontiguousarray(a, dtype=dt)
    return hashlib.sha256(np.asarray(b.shape, dtype="<i8").tobytes() + b.tobytes()).hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def protocol() -> dict:
    uniform = np.ones(N)
    inv = {}
    for name, c in COEFF.items():
        gram = np.outer(c, c)
        inv[name] = {
            "coefficients": c.tolist(),
            "norm_squared": float(c @ c),
            "coefficient_magnitudes": np.abs(c).tolist(),
            "kossakowski_rank": int(np.linalg.matrix_rank(gram)),
            "kossakowski_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
            "sitewise_diagonal": np.diag(gram).tolist(),
            "squared_overlap_with_uniform_drive": float(abs(uniform @ c) ** 2 / ((uniform @ uniform) * (c @ c))),
        }
    return {
        "version": VERSION,
        "status": "frozen before task trajectories",
        "question": "Does rank-one dissipative orientation change readout-accessible STM?",
        "primary": "paired equal-phase minus drive-orthogonal STM capacity, two-sided 95% t interval",
        "claim_gate": "all convergence audits pass and the primary interval excludes zero",
        "n_qubits": N,
        "h": H,
        "dt": DT,
        "gamma": GAMMA,
        "coupling_distribution": "complete graph U[-1,1], scaled by sqrt(4/5)",
        "coefficients": inv,
        "matched_budget": float(dsp.jump_strength(dsp.local_loss(N, GAMMA))),
        "seeds": list(SEEDS),
        "convergence": {"checkpoints": list(CHECKS), "trace_gate": TRACE_GATE, "feature_gate": FEATURE_GATE,
                        "states_all_pairs": ["ground", "maximally_mixed"],
                        "extra_states_first_six": ["fully_excited", "haar_pure"]},
        "stm": {"train": TRAIN, "test": TEST, "delays": list(DELAYS), "ridge": RIDGE},
        "kernel": {"delta": DELTA, "lags": KLAGS, "anchors": list(ANCHORS), "tail_start": TAIL_START},
    }


def streams(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, d = np.random.SeedSequence([2026081201, seed]).spawn(4)
    rj, rw, rt, rh = map(np.random.default_rng, (a, b, c, d))
    J = np.zeros((N, N))
    ix = np.triu_indices(N, 1)
    J[ix] = rj.uniform(-1, 1, len(ix[0]))
    J = J_SCALE * (J + J.T)
    wash = rw.uniform(0, 1, CHECKS[-1])
    task = rt.uniform(0, 1, TRAIN + TEST)
    psi = rh.normal(size=2**N) + 1j * rh.normal(size=2**N)
    psi /= np.linalg.norm(psi)
    return J, wash, task, psi


def functionals() -> tuple[list[str], np.ndarray]:
    obs = readout.pauli_observables(N, max_weight=2)
    return [o.name for o in obs], np.stack([vec(o.matrix.T) for o in obs])


def build(name: str, J: np.ndarray) -> tuple[SparseLindbladReservoir, dict]:
    jumps = dsp.collective_loss(N, GAMMA, c=COEFF[name])
    target = dsp.jump_strength(dsp.local_loss(N, GAMMA))
    actual = dsp.jump_strength(jumps)
    if not np.isclose(actual, target, rtol=1e-12, atol=1e-12):
        raise RuntimeError("budget mismatch")
    h0, hx = ising_xx_hamiltonian(J, H, N), transverse_drive(N)
    r = SparseLindbladReservoir.from_terms(N, h0 + H * hx, H * hx, jumps, DT)
    return r, {"budget": float(actual), "jump_sha256": array_digest(jumps[0][0])}


def init_states(psi: np.ndarray, full: bool) -> tuple[list[str], np.ndarray]:
    d = 2**N
    ground = np.zeros((d, d), complex); ground[0, 0] = 1
    mixed = np.eye(d) / d
    rows = [("ground", ground), ("maximally_mixed", mixed)]
    if full:
        excited = np.zeros((d, d), complex); excited[-1, -1] = 1
        rows.extend([("fully_excited", excited), ("haar_pure", np.outer(psi, psi.conj()))])
    return [x[0] for x in rows], np.column_stack([vec(x[1]) for x in rows])


def evolve(r: SparseLindbladReservoir, state: np.ndarray, s: float) -> np.ndarray:
    return expm_multiply(r.liouvillian(float(s)) * DT, state)


def feat(F: np.ndarray, state: np.ndarray) -> np.ndarray:
    out = np.real(F @ state)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("non-finite feature")
    return out


def convergence(names: Sequence[str], states: np.ndarray, F: np.ndarray) -> dict:
    d = 2**N
    pairwise, mt, mf = [], 0.0, 0.0
    for i in range(states.shape[1]):
        for j in range(i + 1, states.shape[1]):
            x = unvec(states[:, i] - states[:, j], d)
            x = (x + x.conj().T) / 2
            td = float(np.sum(np.abs(np.linalg.eigvalsh(x))) / 2)
            fd = float(np.max(np.abs(feat(F, states[:, i] - states[:, j]))))
            mt, mf = max(mt, td), max(mf, fd)
            pairwise.append({"left": names[i], "right": names[j], "trace_distance": td, "max_feature_distance": fd})
    return {"maximum_trace_distance": mt, "maximum_feature_distance": mf,
            "passed": bool(mt <= TRACE_GATE and mf <= FEATURE_GATE), "pairwise": pairwise}


def synchronized_wash(reservoirs: dict, wash: np.ndarray, names: list[str], initial: np.ndarray, F: np.ndarray):
    states = {k: initial.copy() for k in COEFF}
    audits = {k: {} for k in COEFF}
    selected = CHECKS[-1]
    for step, s in enumerate(wash, 1):
        for k in COEFF:
            states[k] = evolve(reservoirs[k], states[k], s)
        if step in CHECKS:
            for k in COEFF:
                audits[k][str(step)] = convergence(names, states[k], F)
            if all(audits[k][str(step)]["passed"] for k in COEFF):
                selected = step
                break
    passed = all(audits[k][str(selected)]["passed"] for k in COEFF)
    return selected, states, {"selected_common_washout": selected, "both_conditions_passed": passed, "audits": audits}


def baseline(r: SparseLindbladReservoir, state: np.ndarray, inputs: np.ndarray, F: np.ndarray):
    X = np.empty((len(inputs), F.shape[0]))
    anchor_states = {}
    for t, s in enumerate(inputs):
        if t in ANCHORS:
            anchor_states[t] = state.copy()
        state = evolve(r, state, s)
        X[t] = feat(F, state)
    return X, anchor_states


def stm_score(X: np.ndarray, inputs: np.ndarray) -> dict:
    xtr, xte = readout.add_bias(X[:TRAIN]), readout.add_bias(X[TRAIN:])
    caps = []
    for tau in DELAYS:
        y = np.full(len(inputs), np.nan); y[tau:] = inputs[:-tau]
        ok = np.isfinite(y[:TRAIN])
        w = readout.train_readout(xtr[ok], y[:TRAIN][ok], ridge=RIDGE)
        caps.append(float(readout.capacity(y[TRAIN:], readout.predict(xte, w))))
    return {"capacity_by_delay": caps, "total_capacity": float(sum(caps))}


def response_kernel(r: SparseLindbladReservoir, inputs: np.ndarray, anchors: dict, F: np.ndarray) -> dict:
    R = np.empty((len(ANCHORS), KLAGS, F.shape[0]))
    for arow, a in enumerate(ANCHORS):
        s = float(inputs[a]); lo, hi = max(0, s - DELTA), min(1, s + DELTA); den = hi - lo
        plus, minus = evolve(r, anchors[a], hi), evolve(r, anchors[a], lo)
        R[arow, 0] = (feat(F, plus) - feat(F, minus)) / den
        pair = np.column_stack([plus, minus])
        for lag in range(1, KLAGS):
            pair = evolve(r, pair, inputs[a + lag])
            R[arow, lag] = (feat(F, pair[:, 0]) - feat(F, pair[:, 1])) / den
    energy = np.mean(np.sum(R * R, axis=2), axis=0)
    p = energy / energy.sum()
    s = np.linalg.svd(R.reshape(-1, R.shape[-1]), compute_uv=False)
    q = s * s / np.sum(s * s)
    nz = q > 0
    return {"normalized_lag_energy": p.tolist(),
            "response_lag_centroid": float(np.arange(1, KLAGS + 1) @ p),
            "long_lag_energy_fraction": float(p[TAIL_START - 1:].sum()),
            "feature_space_effective_rank": float(np.exp(-np.sum(q[nz] * np.log(q[nz])))),
            "leading_singular_energy_fraction": float(q[0]),
            "response_sha256": array_digest(R)}


def run_condition(name: str, r: SparseLindbladReservoir, state: np.ndarray, inputs: np.ndarray, F: np.ndarray) -> dict:
    X, anchors = baseline(r, state, inputs, F)
    return {"stm": stm_score(X, inputs), "kernel": response_kernel(r, inputs, anchors, F), "features_sha256": array_digest(X)}


def checkpoint(out: Path, index: int) -> Path:
    return out / "checkpoints" / f"seed_{index:02d}.json"


def run(index: int, out: Path, force: bool = False) -> None:
    if not 0 <= index < len(SEEDS):
        raise ValueError("seed index out of range")
    path = checkpoint(out, index)
    if path.exists() and not force:
        print(f"exists: {path}"); return
    started = time.time(); p = protocol(); psha = digest(p); seed = SEEDS[index]
    J, wash, task, psi = streams(seed); names, F = functionals()
    reservoirs, meta = {}, {}
    for name in COEFF:
        reservoirs[name], meta[name] = build(name, J)
    initial_names, initial = init_states(psi, index < 6)
    wash_len, washed, audit = synchronized_wash(reservoirs, wash, initial_names, initial, F)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut = {name: pool.submit(run_condition, name, reservoirs[name], washed[name][:, 0], task, F) for name in COEFF}
        results = {name: fut[name].result() for name in COEFF}
    de = results["equal_phase"]["stm"]["total_capacity"] - results["drive_orthogonal"]["stm"]["total_capacity"]
    row = {"version": VERSION, "protocol_sha256": psha, "seed_index": index, "seed": seed,
           "coupling_sha256": array_digest(J), "wash_sha256": array_digest(wash), "task_sha256": array_digest(task),
           "full_four_state_audit": index < 6, "reservoirs": meta, "convergence": audit, "conditions": results,
           "stm_equal_minus_orthogonal": float(de), "runtime_seconds": time.time() - started}
    row["payload_sha256"] = digest(row)
    write_json(path, row)
    print(f"seed {index:02d}: wash={wash_len}, dSTM={de:+.6f}, {row['runtime_seconds']:.1f}s", flush=True)


def mean_ci(x: Sequence[float]) -> dict:
    a = np.asarray(x, float); m = float(a.mean()); se = float(a.std(ddof=1) / math.sqrt(len(a)))
    h = float(student_t.ppf(.975, len(a) - 1) * se)
    return {"mean": m, "standard_error": se, "ci95": [m - h, m + h]}


def paired(x: Sequence[float]) -> dict:
    a = np.asarray(x, float); o = mean_ci(a); wins, losses = int((a > 0).sum()), int((a < 0).sum())
    o.update({"median": float(np.median(a)), "wins_positive": wins, "losses_negative": losses,
              "ties": int((a == 0).sum()),
              "sign_test_p": float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.0})
    return o


def corr(x, y) -> dict:
    x, y = np.asarray(x), np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
    a, b = pearsonr(x, y), spearmanr(x, y)
    return {"pearson_r": float(a.statistic), "pearson_p": float(a.pvalue),
            "spearman_rho": float(b.statistic), "spearman_p": float(b.pvalue)}


def load(out: Path) -> list[dict]:
    psha = digest(protocol()); rows = []
    for i, seed in enumerate(SEEDS):
        path = checkpoint(out, i)
        if not path.exists(): raise RuntimeError(f"missing {path}")
        row = json.loads(path.read_text()); h = row.pop("payload_sha256")
        if h != digest(row): raise RuntimeError(f"bad digest {path}")
        row["payload_sha256"] = h
        if row["protocol_sha256"] != psha or row["seed"] != seed: raise RuntimeError(f"identity mismatch {path}")
        rows.append(row)
    return rows


def make_figure(rows: list[dict], out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eq = np.array([r["conditions"]["equal_phase"]["stm"]["total_capacity"] for r in rows])
    ort = np.array([r["conditions"]["drive_orthogonal"]["stm"]["total_capacity"] for r in rows])
    ce = np.array([r["conditions"]["equal_phase"]["stm"]["capacity_by_delay"] for r in rows])
    co = np.array([r["conditions"]["drive_orthogonal"]["stm"]["capacity_by_delay"] for r in rows])
    ke = np.array([r["conditions"]["equal_phase"]["kernel"]["normalized_lag_energy"] for r in rows])
    ko = np.array([r["conditions"]["drive_orthogonal"]["kernel"]["normalized_lag_energy"] for r in rows])
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for a, b in zip(ort, eq): ax[0,0].plot([0,1], [a,b], "o-", alpha=.25)
    ax[0,0].set(xticks=[0,1], xticklabels=["drive-orthogonal","equal-phase"], ylabel="STM capacity", title="(a) Orientation intervention")
    for data, label in ((co,"drive-orthogonal"),(ce,"equal-phase")):
        m=data.mean(0); h=student_t.ppf(.975,len(rows)-1)*data.std(0,ddof=1)/math.sqrt(len(rows)); x=np.array(DELAYS)
        ax[0,1].plot(x,m,label=label); ax[0,1].fill_between(x,m-h,m+h,alpha=.18)
    ax[0,1].set(xlabel="input delay",ylabel="capacity",title="(b) Lag-resolved STM"); ax[0,1].legend(frameon=False)
    for data, label in ((ko,"drive-orthogonal"),(ke,"equal-phase")):
        m=data.mean(0); h=student_t.ppf(.975,len(rows)-1)*data.std(0,ddof=1)/math.sqrt(len(rows)); x=np.arange(1,KLAGS+1)
        ax[1,0].plot(x,m,label=label); ax[1,0].fill_between(x,m-h,m+h,alpha=.18)
    ax[1,0].axvline(TAIL_START,ls="--"); ax[1,0].set(xlabel="response lag",ylabel="kernel-energy fraction",title="(c) Switched response")
    dstm=eq-ort; dtail=np.array([r["conditions"]["equal_phase"]["kernel"]["long_lag_energy_fraction"]-r["conditions"]["drive_orthogonal"]["kernel"]["long_lag_energy_fraction"] for r in rows])
    ax[1,1].scatter(dtail,dstm); ax[1,1].axhline(0,lw=.8); ax[1,1].axvline(0,lw=.8); ax[1,1].set(xlabel="change in long-lag kernel fraction",ylabel="change in STM",title="(d) Dynamics versus task")
    for ext in ("pdf","png"): fig.savefig(out/f"rank_one_orientation.{ext}",dpi=220)
    plt.close(fig)


def aggregate(out: Path) -> None:
    rows=load(out); write_json(out/"protocol.json",protocol())
    metrics={name:{k:[] for k in ("stm","centroid","tail","rank","leading")} for name in COEFF}
    perseed=[]; lagrows=[]
    for r in rows:
        rec={"seed_index":r["seed_index"],"seed":r["seed"],"washout":r["convergence"]["selected_common_washout"],"convergence_passed":r["convergence"]["both_conditions_passed"]}
        for name in COEFF:
            z=r["conditions"][name]; k=z["kernel"]
            vals=(z["stm"]["total_capacity"],k["response_lag_centroid"],k["long_lag_energy_fraction"],k["feature_space_effective_rank"],k["leading_singular_energy_fraction"])
            for key,val in zip(metrics[name],vals): metrics[name][key].append(val)
            for key,val in zip(metrics[name],vals): rec[f"{name}_{key}"]=val
        rec["stm_equal_minus_orthogonal"]=rec["equal_phase_stm"]-rec["drive_orthogonal_stm"]
        rec["tail_equal_minus_orthogonal"]=rec["equal_phase_tail"]-rec["drive_orthogonal_tail"]
        perseed.append(rec)
        for j,tau in enumerate(DELAYS): lagrows.append({"seed":r["seed"],"delay":tau,"equal_phase":r["conditions"]["equal_phase"]["stm"]["capacity_by_delay"][j],"drive_orthogonal":r["conditions"]["drive_orthogonal"]["stm"]["capacity_by_delay"][j]})
    pd={}
    for key in metrics["equal_phase"]: pd[key]=paired(np.array(metrics["equal_phase"][key])-np.array(metrics["drive_orthogonal"][key]))
    allconv=all(x["convergence_passed"] for x in perseed); ci=pd["stm"]["ci95"]; resolved=ci[0]>0 or ci[1]<0
    summary={"version":VERSION,"pair_count":len(rows),"all_convergence_passed":allconv,
             "absolute":{n:{k:mean_ci(v) for k,v in m.items()} for n,m in metrics.items()},
             "paired_equal_minus_orthogonal":pd,
             "association_stm_vs_tail_change":corr([x["stm_equal_minus_orthogonal"] for x in perseed],[x["tail_equal_minus_orthogonal"] for x in perseed]),
             "decision":{"orientation_dependence_supported":bool(allconv and resolved),"primary_ci_excludes_zero":bool(resolved),
                         "observed_direction":"equal_phase_higher" if pd["stm"]["mean"]>0 else "drive_orthogonal_higher"}}
    write_json(out/"summary.json",summary)
    with (out/"per_seed.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=perseed[0].keys()); w.writeheader(); w.writerows(perseed)
    with (out/"lag_capacities.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=lagrows[0].keys()); w.writeheader(); w.writerows(lagrows)
    make_figure(rows,out)
    s=pd["stm"]; report=f"""# Rank-one orientation intervention\n\nEqual-phase minus drive-orthogonal STM: **{s['mean']:+.6f}** (95% CI **[{s['ci95'][0]:+.6f}, {s['ci95'][1]:+.6f}]**; wins/losses/ties **{s['wins_positive']}/{s['losses_negative']}/{s['ties']}**; exact sign-test p={s['sign_test_p']:.6g}).\n\nConvergence passed in **{sum(x['convergence_passed'] for x in perseed)}/{len(perseed)}** pairs.\n\n**Decision:** orientation dependence {'is supported' if summary['decision']['orientation_dependence_supported'] else 'was not established'} under the frozen rule.\n\nSecondary equal-minus-orthogonal changes: lag centroid {pd['centroid']['mean']:+.6f} [{pd['centroid']['ci95'][0]:+.6f}, {pd['centroid']['ci95'][1]:+.6f}]; long-lag fraction {pd['tail']['mean']:+.6f} [{pd['tail']['ci95'][0]:+.6f}, {pd['tail']['ci95'][1]:+.6f}]; effective rank {pd['rank']['mean']:+.6f} [{pd['rank']['ci95'][0]:+.6f}, {pd['rank']['ci95'][1]:+.6f}].\n"""
    (out/"REPORT.md").write_text(report)
    shutil.make_archive(str(out.parent/"rank_one_orientation_v1_results"),"zip",root_dir=out.parent,base_dir=out.name)
    print(json.dumps(summary["decision"],indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--outdir",type=Path,default=DEFAULT_OUT); sub=p.add_subparsers(dest="cmd",required=True)
    r=sub.add_parser("run"); r.add_argument("--seed-index",type=int,required=True); r.add_argument("--force",action="store_true")
    sub.add_parser("aggregate"); sub.add_parser("protocol")
    a=p.parse_args(argv); out=a.outdir.resolve(); out.mkdir(parents=True,exist_ok=True)
    if a.cmd=="run": run(a.seed_index,out,a.force)
    elif a.cmd=="aggregate": aggregate(out)
    else: print(json.dumps(protocol(),indent=2))

if __name__=="__main__": main()

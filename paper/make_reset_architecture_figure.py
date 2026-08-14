"""Build the reset-architecture appendix figure from the sealed strict run.

The canonical input is
``results/reset_architecture_replication/strict_washout_arrays.npz``.  The
small JSON snapshot under ``paper/data`` keeps the arXiv source self-contained.
Refresh that snapshot only after the canonical run has completed:

    python3 paper/make_reset_architecture_figure.py --refresh-snapshot

Every empirical mark is recomputed from the paired arrays.  The declared seeds
and independently audited headline effects below are validation gates, not
plotting inputs: a mismatched canonical result fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats

import l3_style as st


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CANONICAL_ARRAYS = (
    ROOT
    / "results"
    / "reset_architecture_replication"
    / "strict_washout_arrays.npz"
)
SNAPSHOT = HERE / "data" / "reset_architecture_snapshot.json"
DEFAULT_OUTPUT = HERE / "figures" / "fig_reset_architecture.pdf"

EXPECTED_SEEDS = np.asarray(
    [
        1162690697,
        411886365,
        1080967412,
        1739603920,
        1154959432,
        600439382,
        1254120429,
        1084176823,
        1869730849,
        56490330,
        1779358140,
        216883587,
        1196651361,
        1669520350,
        1902393916,
        724810199,
    ],
    dtype=np.int64,
)
EXPECTED_SUMMARY = {
    "stm_local_mean": 4.5959116733872065,
    "stm_collective_mean": 6.369393907463901,
    "stm_favorable_mean": 1.7734822340766943,
    "narma_local_mean": 0.6729731742237757,
    "narma_collective_mean": 0.4909328568007432,
    "narma_favorable_mean": 0.18204031742303253,
}
SUMMARY_TOLERANCE = 5e-7
ARRAY_TOLERANCE = 5e-7
ARRAY_KEYS = (
    "seeds",
    "delays",
    "stm_local",
    "stm_collective",
    "narma_local",
    "narma_collective",
    "lag_local",
    "lag_collective",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(ARRAY_KEYS) - set(archive.files)
        if missing:
            raise RuntimeError(
                f"{path} lacks required arrays: {', '.join(sorted(missing))}"
            )
        return {key: np.asarray(archive[key]) for key in ARRAY_KEYS}


def _load_snapshot(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported reset snapshot schema in {path}")
    arrays = payload.get("arrays", {})
    missing = set(ARRAY_KEYS) - set(arrays)
    if missing:
        raise RuntimeError(
            f"{path} lacks required arrays: {', '.join(sorted(missing))}"
        )
    return {
        key: np.asarray(
            arrays[key],
            dtype=np.int64 if key in {"seeds", "delays"} else float,
        )
        for key in ARRAY_KEYS
    }


def _summary(data: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "stm_local_mean": float(np.mean(data["stm_local"])),
        "stm_collective_mean": float(np.mean(data["stm_collective"])),
        "stm_favorable_mean": float(
            np.mean(data["stm_collective"] - data["stm_local"])
        ),
        "narma_local_mean": float(np.mean(data["narma_local"])),
        "narma_collective_mean": float(np.mean(data["narma_collective"])),
        "narma_favorable_mean": float(
            np.mean(data["narma_local"] - data["narma_collective"])
        ),
    }


def _validate(data: dict[str, np.ndarray], source: str) -> None:
    expected_shapes = {
        "seeds": (16,),
        "delays": (20,),
        "stm_local": (16,),
        "stm_collective": (16,),
        "narma_local": (16,),
        "narma_collective": (16,),
        "lag_local": (16, 20),
        "lag_collective": (16, 20),
    }
    for key, shape in expected_shapes.items():
        if data[key].shape != shape:
            raise RuntimeError(
                f"{source}: {key} has shape {data[key].shape}, expected {shape}"
            )
        if not np.all(np.isfinite(data[key])):
            raise RuntimeError(f"{source}: {key} contains non-finite values")
    if not np.array_equal(data["seeds"], EXPECTED_SEEDS):
        raise RuntimeError(f"{source}: strict seed order does not match the audit")
    if not np.array_equal(data["delays"], np.arange(1, 21)):
        raise RuntimeError(f"{source}: expected STM delays 1 through 20")
    for method in ("local", "collective"):
        if not np.allclose(
            data[f"stm_{method}"],
            np.sum(data[f"lag_{method}"], axis=1),
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{source}: STM totals do not equal lag-capacity sums for {method}"
            )
    observed = _summary(data)
    for key, expected in EXPECTED_SUMMARY.items():
        if abs(observed[key] - expected) > SUMMARY_TOLERANCE:
            raise RuntimeError(
                f"{source}: {key}={observed[key]:.12g} differs from the "
                f"audited value {expected:.12g}"
            )
    if np.any(data["stm_collective"] <= data["stm_local"]):
        raise RuntimeError(f"{source}: strict STM replication is not 16/16")
    if np.any(data["narma_collective"] >= data["narma_local"]):
        raise RuntimeError(f"{source}: strict NARMA replication is not 16/16")


def _write_snapshot(data: dict[str, np.ndarray], source: Path) -> None:
    _validate(data, str(source))
    try:
        source_label = str(source.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        source_label = str(source.resolve())
    payload = {
        "schema_version": 1,
        "source": {
            "path": source_label,
            "sha256": _sha256(source),
        },
        "protocol": {
            "architecture": "input-by-reset",
            "n_qubits": 5,
            "gamma": 1.0,
            "assigned_frobenius_budget": 80.0,
            "washout": 800,
            "train": 600,
            "test": 400,
            "stm_delays": [1, 20],
            "pairs": 16,
            "master_seed": 2026080603,
        },
        "arrays": {key: data[key].tolist() for key in ARRAY_KEYS},
    }
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_checked_data() -> dict[str, np.ndarray]:
    if not SNAPSHOT.exists():
        raise RuntimeError(
            f"missing {SNAPSHOT}; run this script with --refresh-snapshot "
            "after the canonical strict run"
        )
    snapshot_payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    snapshot_data = _load_snapshot(SNAPSHOT)
    _validate(snapshot_data, str(SNAPSHOT))
    if not CANONICAL_ARRAYS.exists():
        return snapshot_data

    canonical_data = _load_npz(CANONICAL_ARRAYS)
    _validate(canonical_data, str(CANONICAL_ARRAYS))
    expected_source = str(CANONICAL_ARRAYS.relative_to(ROOT))
    declared_source = snapshot_payload.get("source", {})
    if (
        declared_source.get("path") != expected_source
        or declared_source.get("sha256") != _sha256(CANONICAL_ARRAYS)
    ):
        raise RuntimeError(
            f"{SNAPSHOT} does not identify the current canonical arrays; "
            "inspect the run and refresh the snapshot"
        )
    for key in ARRAY_KEYS:
        if key in {"seeds", "delays"}:
            matches = np.array_equal(canonical_data[key], snapshot_data[key])
        else:
            matches = np.allclose(
                canonical_data[key],
                snapshot_data[key],
                rtol=0,
                atol=ARRAY_TOLERANCE,
            )
        if not matches:
            raise RuntimeError(
                f"{CANONICAL_ARRAYS} and {SNAPSHOT} disagree for {key}; "
                "inspect the canonical run before refreshing the snapshot"
            )
    return canonical_data


def _mean_ci95(values: np.ndarray, axis: int = 0):
    values = np.asarray(values, dtype=float)
    count = values.shape[axis]
    mean = np.mean(values, axis=axis)
    standard_error = stats.sem(values, axis=axis, ddof=1)
    half_width = stats.t.ppf(0.975, count - 1) * standard_error
    return mean, half_width


def make_figure(data: dict[str, np.ndarray]):
    """Return the exact one-column appendix figure."""
    st.use(times=True)
    figure_height = 2.02
    fig = st.composite_figure("column", figure_height)
    fig.set_layout_engine("none")

    ax_effect = st.add_axes_inches(fig, [0.45, 0.39, 1.04, 1.12])
    ax_lag = st.add_axes_inches(fig, [1.94, 0.39, 1.20, 1.12])
    axes = np.asarray([ax_effect, ax_lag], dtype=object)

    # Panel (a): percentage changes place STM gain and NARMA error reduction
    # on one shared, directionally aligned scale. Exact absolute effects remain
    # in the appendix table.
    favorable = (
        100.0
        * (data["stm_collective"] - data["stm_local"])
        / data["stm_local"],
        100.0
        * (data["narma_local"] - data["narma_collective"])
        / data["narma_local"],
    )
    x_positions = np.asarray([0.0, 1.0])
    random = np.random.default_rng(20260807)
    for x_position, values in zip(x_positions, favorable, strict=True):
        mean, half_width = _mean_ci95(values)
        jitter = random.uniform(-0.085, 0.085, size=len(values))
        ax_effect.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            s=10.0,
            marker="o",
            facecolor="none",
            edgecolor=st.distance_faded_colors(
                st.COLLECTIVE,
                values,
                center=float(mean),
                near_alpha=0.28,
                far_alpha=0.045,
            ),
            linewidth=0.55,
            zorder=2,
        )
        ax_effect.errorbar(
            x_position,
            mean,
            yerr=half_width,
            color=st.COLLECTIVE,
            marker="o",
            markerfacecolor=st.COLLECTIVE,
            markeredgecolor=st.COLLECTIVE,
            markeredgewidth=st.MARKER_EDGEWIDTH,
            markersize=st.MARKER_SIZE,
            linewidth=st.DATA_LINEWIDTH,
            elinewidth=st.ERROR_LINEWIDTH,
            capsize=st.ERROR_CAPSIZE,
            capthick=st.ERROR_LINEWIDTH,
            zorder=4,
        )
    ax_effect.axhline(
        0,
        color=st.INK,
        linewidth=st.REFERENCE_LINEWIDTH,
        zorder=1,
    )
    ax_effect.set_xlim(-0.40, 1.40)
    ax_effect.set_xticks(x_positions)
    ax_effect.set_xticklabels(["STM", "NARMA-10"])
    ax_effect.set_ylim(-4, 100)
    ax_effect.set_yticks([0, 50, 100])
    ax_effect.set_ylabel("favorable change (%)")
    st.style_axis(ax_effect, "y", minor_axis="y")

    # Panel (b): the lag profile exposes where the additional STM capacity
    # survives. Thin trajectories and small points are the 16 paired seeds;
    # thick lines and bands are the pointwise mean and 95% t intervals.
    delays = data["delays"]
    series = (
        (
            "local",
            data["lag_local"],
            st.LOCAL_CONTRAST,
            "D",
            "--",
            "white",
        ),
        (
            "collective",
            data["lag_collective"],
            st.COLLECTIVE,
            "o",
            "-",
            st.COLLECTIVE,
        ),
    )
    method_offsets = {"local": -0.045, "collective": 0.045}
    seed_jitter = np.linspace(-0.025, 0.025, 16)
    for label, values, color, marker, linestyle, marker_face in series:
        totals = np.sum(values, axis=1)
        trace_colors = st.distance_faded_colors(
            color,
            totals,
            center=float(np.mean(totals)),
            near_alpha=0.18,
            far_alpha=0.018,
        )
        for trace, trace_color in zip(values, trace_colors, strict=True):
            ax_lag.plot(
                delays,
                trace,
                color=trace_color,
                linewidth=0.48,
                zorder=1,
            )
        mean, half_width = _mean_ci95(values, axis=0)
        ax_lag.fill_between(
            delays,
            np.maximum(0.0, mean - half_width),
            np.minimum(1.0, mean + half_width),
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=2,
        )
        for delay_index, delay in enumerate(delays):
            seed_values = values[:, delay_index]
            ax_lag.scatter(
                np.full(len(seed_values), delay + method_offsets[label])
                + seed_jitter,
                seed_values,
                s=4.5,
                marker="o",
                color=st.distance_faded_colors(
                    color,
                    seed_values,
                    center=float(mean[delay_index]),
                    near_alpha=0.17,
                    far_alpha=0.022,
                ),
                linewidths=0,
                zorder=3,
            )
        ax_lag.plot(
            delays,
            mean,
            color=color,
            marker=marker,
            markevery=[0, 4, 9, 14, 19],
            linestyle=linestyle,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=st.MARKER_EDGEWIDTH,
            markersize=st.MARKER_SIZE,
            linewidth=st.DATA_LINEWIDTH,
            zorder=4,
        )
    ax_lag.set_xlim(0.75, 20.25)
    ax_lag.set_xticks([1, 10, 20])
    ax_lag.set_ylim(-0.02, 1.03)
    ax_lag.set_yticks([0, 0.5, 1.0])
    ax_lag.set_xlabel(r"input delay $\tau$")
    ax_lag.set_ylabel("STM contribution")
    st.style_axis(ax_lag, "both", minor_axis="both")

    legend_handles = [
        Line2D(
            [],
            [],
            color=st.LOCAL_CONTRAST,
            marker="D",
            linestyle="--",
            markerfacecolor="white",
            markeredgecolor=st.LOCAL_CONTRAST,
            markeredgewidth=st.MARKER_EDGEWIDTH,
            markersize=st.MARKER_SIZE,
            linewidth=st.DATA_LINEWIDTH,
            label="local",
        ),
        Line2D(
            [],
            [],
            color=st.COLLECTIVE,
            marker="o",
            linestyle="-",
            markerfacecolor=st.COLLECTIVE,
            markeredgecolor=st.COLLECTIVE,
            markeredgewidth=st.MARKER_EDGEWIDTH,
            markersize=st.MARKER_SIZE,
            linewidth=st.DATA_LINEWIDTH,
            label="collective",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=2,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#9E9E9E",
        borderpad=0.32,
        labelspacing=0.20,
        handlelength=1.42,
        handletextpad=0.38,
        columnspacing=0.70,
    )
    legend.get_frame().set_linewidth(st.LEGEND_FRAMEWIDTH)
    st.panel_labels(fig, axes, labels="ab", x=0.0, y=1.055)
    st.audit_figure(
        fig,
        "fig_reset_architecture",
        axes=axes,
        overlap_fraction=0.08,
    )
    return fig


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="refresh the arXiv snapshot from the canonical strict arrays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PDF output path",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        help="optional PNG preview path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.refresh_snapshot:
        if not CANONICAL_ARRAYS.exists():
            raise RuntimeError(
                f"cannot refresh snapshot: missing {CANONICAL_ARRAYS}"
            )
        canonical_data = _load_npz(CANONICAL_ARRAYS)
        _write_snapshot(canonical_data, CANONICAL_ARRAYS)

    data = _load_checked_data()
    fig = make_figure(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0,
        metadata={
            "Creator": "paper/make_reset_architecture_figure.py",
            "Title": "Reset-architecture replication",
            "CreationDate": None,
        },
    )
    if args.preview:
        # Matplotlib may restore the rcParams layout engine after a first
        # vector save; keep the manually positioned canvas fixed for preview.
        fig.set_layout_engine("none")
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.preview,
            dpi=300,
            facecolor="white",
            bbox_inches=None,
            pad_inches=0,
            metadata={
                "Software": "paper/make_reset_architecture_figure.py",
            },
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

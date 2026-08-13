"""Regenerate the publication figures from the archived per-instance JSON.

Run from the repository root (or anywhere):

    python3 paper/make_figures.py

The figures use embedded TrueType fonts, a shared color/marker vocabulary, and
explicit paired uncertainty.  No paper number is hard-coded: every empirical
mark is recomputed from the canonical baseline archives or the sealed revision
artifacts under ``results/``.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats

import l3_style as st


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FINAL_DIR = os.path.join(ROOT, "results", "final_protocol")
REVIEW_DIR = os.path.join(ROOT, "results", "review_protocol")
REVISION_TUNING_DIR = os.path.join(ROOT, "results", "revision_tuning")
REVISION_PROSPECTIVE = os.path.join(
    REVISION_TUNING_DIR,
    "fresh_interpolation",
    "fresh_interpolation_results.json",
)
REVISION_MEASUREMENT = os.path.join(
    ROOT, "results", "measurement_full_v3", "measurement_full_aggregate.json"
)
REVISION_PARITY = os.path.join(
    ROOT, "results", "revision_parity_control", "paper_aggregate.json"
)
REVISION_SCALING = os.path.join(
    ROOT,
    "results",
    "revision_normalized_scaling",
    "paper_variance_aggregate.json",
)
FINITE_SIZE_SNAPSHOT = os.path.join(
    HERE,
    "data",
    "experiment1_finite_size_snapshot.json",
)
PARITY_WINDOW_SNAPSHOT = os.path.join(
    HERE,
    "data",
    "experiment1_parity_window_snapshot.json",
)
ACTIVITY_MATCHED_SNAPSHOT = os.path.join(
    HERE,
    "data",
    "activity_matched_confirmation.json",
)
OUTDIR = os.path.join(HERE, "figures")

st.use(times=True)

# Legacy helpers retain the earlier square-panel geometry for non-manuscript
# diagnostics.  The four manuscript figures now use data-dependent rectangular
# panels at their exact final placement width.
PANEL_SIDE = 1.5594
PANEL_GAP = 0.4508
PANEL_MARGIN_X = 0.5041
PANEL_BOTTOM = 0.505
PANEL_CANVAS_HEIGHT = 2.285

INK = st.INK
MUTED = st.GRAY
GRID = st.GRID
MINOR_GRID = "#F1F1F1"
PAPER = "#FFFFFF"
BLUE = st.BLUE
RED = st.RED
GRAY = st.GRAY
GREEN = st.GREEN
PURPLE = st.PURPLE
ORANGE = st.ORANGE
TEAL = st.TEAL

# Reference-derived final-size marks.  These remain strong at 100% PDF zoom
# without overpowering the compact axes.
DATA_LINEWIDTH = st.DATA_LINEWIDTH
MARKER_SIZE = st.MARKER_SIZE
MARKER_EDGEWIDTH = st.MARKER_EDGEWIDTH
ERROR_LINEWIDTH = st.ERROR_LINEWIDTH
ERROR_CAPSIZE = st.ERROR_CAPSIZE

# Stable method semantics across all empirical figures.  Every line also has a
# non-color encoding, so the set remains readable in grayscale.
C = {
    "CD_paper": st.UNIFORM_LOCAL,
    "B3_collective": st.COLLECTIVE,
    "A1_heterogeneous": st.UNEQUAL_LOCAL,
    "B5_pair": st.PAIR_LOSS,
    "B4_loss_exchange": st.EXCHANGE_ASSISTED,
    "B2_thermal": st.GAIN_LOSS,
    "B1_dephasing": st.DEPHASING,
    "FN": INK,
}
MARKER = {
    "CD_paper": "D",
    "B3_collective": "o",
    "A1_heterogeneous": "s",
    "B5_pair": "^",
    "B4_loss_exchange": "v",
    "B2_thermal": "P",
    "B1_dephasing": "X",
    "FN": "^",
}
LINESTYLE = {
    "CD_paper": "-",
    "B3_collective": "-",
    "A1_heterogeneous": "--",
    "B5_pair": "-.",
    "B4_loss_exchange": ":",
    "B2_thermal": (0, (5, 2, 1, 2)),
    "B1_dephasing": ":",
    "FN": "--",
}
LABEL = {
    "CD_paper": "uniform local relaxation",
    "B3_collective": "collective relaxation",
    "A1_heterogeneous": "unequal rates",
    "B5_pair": "pair loss",
    "B4_loss_exchange": "exchange-assisted relaxation",
    "B2_thermal": "local gain/loss",
    "B1_dephasing": "dephasing",
    "FN": "reset-encoded FN",
}
HIGHER = {"stm": True, "narma": False, "parity": True, "mg": False}

_REVIEW_CACHE = None
_PARITY_CACHE = None


def _read_jsons(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as handle:
                rows.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def load() -> list[dict]:
    """Load canonical results, replacing the superseded forecasting cells."""
    rows = _read_jsons(os.path.join(FINAL_DIR, "*.json"))
    rows = [
        row
        for row in rows
        if not (row.get("block") == "A_table" and row.get("task") == "mg")
    ]
    for row in _read_jsons(os.path.join(REVIEW_DIR, "R_mgfix__*.json")):
        row = dict(row)
        row["block"] = "A_table"
        rows.append(row)
    return rows


def load_review() -> list[dict]:
    global _REVIEW_CACHE
    if _REVIEW_CACHE is None:
        _REVIEW_CACHE = _read_jsons(os.path.join(REVIEW_DIR, "*.json"))
    if len(_REVIEW_CACHE) < 3000:
        raise SystemExit(
            "ERROR: results/review_protocol is missing or incomplete "
            f"({len(_REVIEW_CACHE)} rows). Extract the archived raw data first."
        )
    return _REVIEW_CACHE


def load_revision_parity() -> dict[str, dict[int, float]]:
    """Load the leak-free, validation-selected parity comparison.

    The six active channels and the two reference rows are hash-linked but
    deliberately stored as separate protocols because FN is not a Lindblad
    channel and dephasing is a negative control.
    """
    global _PARITY_CACHE
    if _PARITY_CACHE is not None:
        return _PARITY_CACHE

    result_dir = os.path.dirname(REVISION_PARITY)
    reference_aggregate_path = os.path.join(
        result_dir, "paper_reference_aggregate.json"
    )
    for path in (REVISION_PARITY, reference_aggregate_path):
        if not os.path.isfile(path):
            raise SystemExit(f"ERROR: missing corrected parity artifact {path}")
    with open(REVISION_PARITY, encoding="utf-8") as handle:
        active = json.load(handle)
    with open(reference_aggregate_path, encoding="utf-8") as handle:
        references = json.load(handle)

    expected = (
        (active, 96, 96),
        (references, 32, 32),
    )
    for aggregate, checkpoints, raw_rows in expected:
        if (
            aggregate.get("status") != "complete"
            or int(aggregate.get("expected_checkpoints", -1)) != checkpoints
            or int(aggregate.get("complete_checkpoints", -1)) != checkpoints
            or aggregate.get("missing_checkpoints")
            or len(aggregate.get("raw_rows", [])) != raw_rows
            or aggregate.get("ridge_boundary_audit", {}).get(
                "n_unresolved_upper", -1
            )
            != 0
        ):
            raise SystemExit("ERROR: corrected parity aggregate is incomplete")
    if references.get("active_protocol_sha256") != active.get("protocol_sha256"):
        raise SystemExit(
            "ERROR: corrected parity reference rows are not linked to the "
            "active-channel protocol"
        )

    rows = [*active["raw_rows"], *references["raw_rows"]]
    methods = {
        "FN",
        "CD_paper",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    }
    out = {
        method: {
            int(row["seed"]): float(row["selected_test_capacity"])
            for row in rows
            if row.get("method") == method and row.get("status") == "complete"
        }
        for method in methods
    }
    if any(len(values) != 16 for values in out.values()):
        counts = {method: len(values) for method, values in out.items()}
        raise SystemExit(f"ERROR: corrected parity rows are incomplete: {counts}")
    common_seeds = {tuple(sorted(values)) for values in out.values()}
    if len(common_seeds) != 1:
        raise SystemExit("ERROR: corrected parity rows are not fully paired")
    _PARITY_CACHE = out
    return out


def load_finite_size_snapshot() -> dict:
    """Load the validated final finite-size paper snapshot."""
    if not os.path.isfile(FINITE_SIZE_SNAPSHOT):
        raise SystemExit(
            f"ERROR: missing finite-size paper snapshot {FINITE_SIZE_SNAPSHOT}"
        )
    with open(FINITE_SIZE_SNAPSHOT, encoding="utf-8") as handle:
        snapshot = json.load(handle)
    expected_measurements = {
        "4": 30,
        "5": 45,
        "6": 63,
        "7": 84,
        "8": 108,
    }
    observed_measurements = {
        key: int(value["observables"])
        for key, value in snapshot.get("measurement_scaling", {}).items()
    }
    if (
        snapshot.get("status") != "complete"
        or snapshot.get("complete_sizes") != [4, 5, 6, 7, 8]
        or snapshot.get("lineages_per_size") != 24
        or snapshot.get("checkpoint_count") != 960
        or snapshot.get("confirmatory_family_size") != 62
        or observed_measurements != expected_measurements
        or len(snapshot.get("checkpoint_set_sha256", "")) != 64
    ):
        raise SystemExit(
            "ERROR: finite-size paper snapshot violates its bounded contract"
        )
    return snapshot


def load_parity_window_snapshot() -> dict:
    """Load the compact validation-selected parity-window profiles."""
    if not os.path.isfile(PARITY_WINDOW_SNAPSHOT):
        raise SystemExit(
            "ERROR: missing parity-window snapshot "
            f"{PARITY_WINDOW_SNAPSHOT}"
        )
    with open(PARITY_WINDOW_SNAPSHOT, encoding="utf-8") as handle:
        snapshot = json.load(handle)
    profiles = snapshot.get("profiles", {})
    expected_methods = {
        "CD_paper",
        "B3_collective",
        "B4_loss_exchange",
    }
    if (
        snapshot.get("status") != "complete"
        or snapshot.get("seeds_per_method") != 16
        or snapshot.get("delays") != list(range(1, 8))
        or set(profiles) != expected_methods
        or any(len(profiles[key].get("mean", [])) != 7 for key in profiles)
        or any(len(profiles[key].get("se", [])) != 7 for key in profiles)
        or len(snapshot.get("source_sha256", "")) != 64
        or len(snapshot.get("protocol_sha256", "")) != 64
    ):
        raise SystemExit(
            "ERROR: parity-window snapshot violates its bounded contract"
        )
    return snapshot


def load_activity_matched_snapshot() -> dict:
    """Load and revalidate the hash-linked eight-pair activity control."""
    if not os.path.isfile(ACTIVITY_MATCHED_SNAPSHOT):
        raise SystemExit(
            "ERROR: missing activity-matched paper snapshot "
            f"{ACTIVITY_MATCHED_SNAPSHOT}"
        )
    with open(ACTIVITY_MATCHED_SNAPSHOT, encoding="utf-8") as handle:
        snapshot = json.load(handle)

    rows = snapshot.get("rows", [])
    target = float(snapshot.get("target_activity", np.nan))
    summary = snapshot.get("summary", {})
    seeds = [int(row["seed"]) for row in rows]
    expected_provenance = {
        "protocol_sha256": (
            "58294d536fbe29bee0eef9b98128e2f74615027b2588556ca870aaaa5d127fcd"
        ),
        "artifact_aggregate_sha256": (
            "94e720aa2ac0f1decbcf73984f8cb396bcf05230aed3feed543ad08d4844dfdb"
        ),
        "verified_aggregate_sha256": (
            "14e38f785b7192e42b240c25f0199358ea6578176060b433f08d7c86b075c28f"
        ),
        "github_artifact_sha256": (
            "4bf62b88fbd4679c6aca7a5c749ee2e9d06fe49753376b60f0d13a23705ea2c6"
        ),
        "verified_evidence_commit": (
            "ed4124f8ae962e59fa82e609f224233ac6b1a71e"
        ),
    }
    correction = snapshot.get("post_run_inference_correction", {})
    if (
        snapshot.get("status") != "complete"
        or int(snapshot.get("n_pairs", -1)) != 8
        or len(rows) != 8
        or len(set(seeds)) != 8
        or not np.isclose(target, 0.5075, atol=1e-12, rtol=0)
        or any(
            snapshot.get(key) != value
            for key, value in expected_provenance.items()
        )
        or correction.get("simulation_or_task_scores_rerun") is not False
    ):
        raise SystemExit(
            "ERROR: activity-matched snapshot violates its provenance contract"
        )

    calibration_differences = np.asarray(
        [
            float(row["collective_calibration_activity"])
            - float(row["local_calibration_activity"])
            for row in rows
        ],
        dtype=float,
    )
    stm_differences = np.asarray(
        [
            float(row["collective_stm"]) - float(row["local_stm"])
            for row in rows
        ],
        dtype=float,
    )
    calibration_summary = t_summary(calibration_differences)
    stm_summary = t_summary(stm_differences)
    reported_stm_interval = np.asarray(
        summary.get("collective_minus_local_stm_ci95", []),
        dtype=float,
    )
    reported_calibration_interval = np.asarray(
        summary.get("calibration_activity_difference_ci95", []),
        dtype=float,
    )
    heldout_interval = np.asarray(
        summary.get("heldout_stm_activity_difference_ci95", []),
        dtype=float,
    )
    calibration_activities = np.asarray(
        [
            float(row[key])
            for row in rows
            for key in (
                "local_calibration_activity",
                "collective_calibration_activity",
            )
        ],
        dtype=float,
    )
    maximum_target_error = float(
        np.max(np.abs(calibration_activities - target) / target)
    )
    exact_stm_p = signflip_permutation_p(stm_differences)
    if (
        reported_stm_interval.shape != (2,)
        or reported_calibration_interval.shape != (2,)
        or heldout_interval.shape != (2,)
        or not np.all(stm_differences > 0)
        or int(summary.get("wins", -1)) != 8
        or maximum_target_error > 0.015
        or not np.isclose(
            maximum_target_error,
            float(summary.get("maximum_relative_target_error", np.nan)),
            atol=1e-12,
            rtol=0,
        )
        or not np.isclose(
            stm_summary["mean"],
            float(summary.get("collective_minus_local_stm", np.nan)),
            atol=1e-12,
            rtol=0,
        )
        or not np.allclose(
            [stm_summary["lo"], stm_summary["hi"]],
            reported_stm_interval,
            atol=1e-12,
            rtol=0,
        )
        or not np.isclose(
            calibration_summary["mean"],
            float(summary.get("calibration_activity_difference", np.nan)),
            atol=1e-12,
            rtol=0,
        )
        or not np.allclose(
            [calibration_summary["lo"], calibration_summary["hi"]],
            reported_calibration_interval,
            atol=1e-12,
            rtol=0,
        )
        or not np.isclose(exact_stm_p, 0.0078125, atol=1e-12, rtol=0)
        or not np.isclose(
            4.0 * exact_stm_p,
            float(
                summary.get("holm_adjusted_signflip_p_four_tasks", np.nan)
            ),
            atol=1e-12,
            rtol=0,
        )
        or not (heldout_interval[0] < 0 < heldout_interval[1])
    ):
        raise SystemExit(
            "ERROR: activity-matched snapshot fails its numerical contract"
        )

    snapshot["calibration_differences"] = calibration_differences
    snapshot["stm_differences"] = stm_differences
    snapshot["calibration_summary"] = calibration_summary
    snapshot["stm_summary"] = stm_summary
    return snapshot


def map_cell(
    rows: list[dict],
    task: str,
    method: str,
) -> dict[int, float]:
    if task == "parity":
        return load_revision_parity()[method]
    return cell(rows, "A_table", task, method, N=5)


def cell(
    rows: list[dict],
    block: str,
    task: str,
    method: str,
    N: int | None = None,
    **match,
) -> dict[int, float]:
    out = {}
    for row in rows:
        if (
            row.get("block") != block
            or row.get("task") != task
            or row.get("method") != method
            or row.get("value") is None
        ):
            continue
        if N is not None and row.get("N") != N:
            continue
        if any(row.get(key) != value for key, value in match.items()):
            continue
        out[row["seed"]] = float(row["value"])
    return out


def t_summary(values) -> dict:
    values = np.asarray(list(values), dtype=float)
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan}
    mean = float(values.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "lo": mean, "hi": mean}
    half = float(
        stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n)
    )
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half}


def paired_summary(
    a: dict[int, float],
    b: dict[int, float],
    higher_better: bool = True,
    percent: bool = False,
) -> dict:
    """Paired effect ``a-b``; positive always favors ``a``."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {
            "n": 0,
            "mean": np.nan,
            "lo": np.nan,
            "hi": np.nan,
            "raw": np.array([]),
        }
    direction = 1.0 if higher_better else -1.0
    diff = direction * np.array([a[seed] - b[seed] for seed in shared], float)
    scale = 1.0
    if percent:
        baseline_mean = np.mean([b[seed] for seed in shared])
        scale = 100.0 / baseline_mean
    summary = t_summary(diff * scale)
    summary["raw"] = diff
    summary["shared"] = shared
    return summary


def signflip_permutation_p(diff: np.ndarray, n_resamples: int = 100_000) -> float:
    """Two-sided sign-flip test on the paired mean.

    The exact distribution is counted by meet-in-the-middle enumeration for
    n<=32. Larger samples use the deterministic 100,000-resample policy
    documented in the manuscript.
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0 or np.allclose(diff, 0):
        return 1.0
    tolerance = 1e-12
    observed_sum = abs(float(diff.sum()))
    if n <= 32:
        def signed_sums(values: np.ndarray) -> np.ndarray:
            sums = np.array([0.0])
            for value in values:
                sums = np.concatenate((sums + value, sums - value))
            return sums

        split = n // 2
        left = signed_sums(diff[:split])
        right = np.sort(signed_sums(diff[split:]))
        threshold = observed_sum - tolerance * n
        if threshold <= 0:
            return 1.0
        below = np.searchsorted(right, -threshold - left, side="right")
        above = len(right) - np.searchsorted(
            right, threshold - left, side="left"
        )
        exceed = int(np.sum(below, dtype=np.int64) + np.sum(above, dtype=np.int64))
        return exceed / (1 << n)

    observed = observed_sum / n
    batch = 8192
    exceed = 0
    rng = np.random.default_rng(12345)
    for start in range(0, n_resamples, batch):
        size = min(batch, n_resamples - start)
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8)
        signs = 1 - 2 * signs
        permuted = signs @ diff / n
        exceed += int(np.count_nonzero(np.abs(permuted) >= observed - tolerance))
    return (exceed + 1) / (n_resamples + 1)


def holm_adjust(pvalues: dict[tuple, float]) -> dict[tuple, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    m = len(ordered)
    for rank, (key, pvalue) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * pvalue))
        adjusted[key] = running
    return adjusted


def interval_error(summary: dict):
    return np.array(
        [[summary["mean"] - summary["lo"]], [summary["hi"] - summary["mean"]]]
    )


def style_axis(
    ax,
    grid_axis: str = "both",
    *,
    square: bool = False,
    minor_grid: bool = True,
    minor_axis: str | None = None,
):
    """Apply the shared framed-axis and quiet two-level grid."""
    st.style_axis(
        ax,
        grid_axis,
        minor_axis=minor_axis,
        minor_grid=minor_grid,
    )
    if square:
        ax.set_box_aspect(1)


def fixed_panel_figure(
    count: int,
    *,
    sharey: bool = False,
    side: float = PANEL_SIDE,
    gap: float = PANEL_GAP,
    margin_x: float = PANEL_MARGIN_X,
    height: float = PANEL_CANVAS_HEIGHT,
    bottom: float = PANEL_BOTTOM,
):
    """Create aligned axes with gallery-identical physical square plot boxes."""
    width = 2 * margin_x + count * side + (count - 1) * gap
    fig = plt.figure(figsize=(width, height), layout="none")
    axes = []
    for index in range(count):
        left = margin_x + index * (side + gap)
        ax = fig.add_axes(
            [left / width, bottom / height, side / width, side / height],
            sharey=axes[0] if sharey and axes else None,
        )
        axes.append(ax)
    fig._qrc_panel_axes = axes
    fig._qrc_panel_side = side
    return fig, np.asarray(axes, dtype=object)


def method_errorbar(ax, x, y, summary, method, **kwargs):
    return ax.errorbar(
        x,
        y,
        yerr=interval_error(summary),
        color=C[method],
        marker=MARKER[method],
        linestyle=LINESTYLE[method],
        markerfacecolor=PAPER if method in ("CD_paper", "FN") else C[method],
        markeredgecolor=C[method],
        markeredgewidth=st.RC["lines.markeredgewidth"],
        markersize=MARKER_SIZE,
        linewidth=DATA_LINEWIDTH,
        capsize=ERROR_CAPSIZE,
        capthick=ERROR_LINEWIDTH,
        elinewidth=ERROR_LINEWIDTH,
        **kwargs,
    )


def panel_labels(fig, axes, labels: str):
    """Apply the common upper-left panel labels after layout stabilizes."""
    return st.panel_labels(fig, axes, labels=labels)


def save(fig, filename: str):
    """Save one title-free figure and enforce the visual contract."""
    fig.canvas.draw()
    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        raise RuntimeError(
            "figure-level titles are forbidden by the figure contract: "
            f"{fig._suptitle.get_text()!r}"
        )
    titles = [
        title
        for ax in fig.axes
        for title in (ax.get_title(), ax.get_title(loc="left"), ax.get_title(loc="right"))
        if title.strip()
    ]
    if titles:
        raise RuntimeError(f"plot titles are forbidden by the figure contract: {titles}")
    if fig.legends:
        raise RuntimeError("figure-level legends are forbidden; place legends in axes")
    undersized = [
        (artist.get_text(), float(artist.get_fontsize()))
        for artist in fig.findobj(matplotlib.text.Text)
        if artist.get_visible()
        and artist.get_text().strip()
        and float(artist.get_fontsize()) < st.MIN_FONT_SIZE - 1e-9
    ]
    if undersized:
        preview = ", ".join(
            f"{text!r}: {size:.2f} pt" for text, size in undersized[:8]
        )
        raise RuntimeError(
            f"{filename} contains text below the {st.MIN_FONT_SIZE:.2f} pt "
            f"final-size floor: {preview}"
        )
    renderer = fig.canvas.get_renderer()
    panel_axes = getattr(fig, "_qrc_panel_axes", [])
    if panel_axes:
        expected = float(fig._qrc_panel_side)
        extents = [
            ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
            for ax in panel_axes
        ]
        widths = np.asarray([box.width for box in extents])
        heights = np.asarray([box.height for box in extents])
        bottoms = np.asarray([box.y0 for box in extents])
        tolerance_in = 0.006
        if (
            np.max(np.abs(widths - expected)) > tolerance_in
            or np.max(np.abs(heights - expected)) > tolerance_in
            or np.max(np.abs(widths - heights)) > tolerance_in
            or np.ptp(widths) > tolerance_in
            or np.ptp(heights) > tolerance_in
            or np.ptp(bottoms) > tolerance_in
        ):
            raise RuntimeError(
                f"{filename} violates the fixed square-panel contract: "
                f"widths={widths}, heights={heights}, bottoms={bottoms}"
            )
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        legend_box = legend.get_window_extent(renderer)
        axes_box = ax.get_window_extent(renderer)
        tolerance = 2.0
        if (
            legend_box.x0 < axes_box.x0 - tolerance
            or legend_box.x1 > axes_box.x1 + tolerance
            or legend_box.y0 < axes_box.y0 - tolerance
            or legend_box.y1 > axes_box.y1 + tolerance
        ):
            raise RuntimeError(
                f"legend in {filename} is not contained in its plotting axes"
            )
    st.audit_figure(
        fig,
        filename,
        axes=getattr(fig, "_qrc_audit_axes", None),
    )
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(
        os.path.join(OUTDIR, filename),
        facecolor=PAPER,
        bbox_inches=None,
        pad_inches=0,
        metadata={
            "Creator": "paper/make_figures.py",
            # Omitting the default wall-clock timestamp keeps regenerated
            # vector figures byte-for-byte reproducible.
            "CreationDate": None,
        },
    )
    plt.close(fig)


def fig_designspace():
    """Generate the supplied four-panel Figure 1 without altering its layout."""
    source = os.path.join(HERE, "fig1_L3.py")
    os.makedirs(OUTDIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="qrc-fig1-") as temporary:
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = "0"
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            HERE
            if not existing_pythonpath
            else HERE + os.pathsep + existing_pythonpath
        )
        completed = subprocess.run(
            [sys.executable, source],
            cwd=temporary,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.returncode != 0:
            raise RuntimeError(
                "supplied Figure 1 generator failed:\n" + completed.stderr
            )
        if "problems found        : 0" not in completed.stdout:
            raise RuntimeError(
                "supplied Figure 1 collision audit did not pass:\n"
                + completed.stdout
            )
        generated = os.path.join(temporary, "fig1_L3l.pdf")
        if not os.path.isfile(generated):
            raise RuntimeError("supplied Figure 1 generator produced no PDF")
        shutil.copyfile(
            generated,
            os.path.join(OUTDIR, "fig_designspace.pdf"),
        )


def fig_map(rows: list[dict]):
    """Central channel-task map with the documented permutation-test contract."""
    finite_size = load_finite_size_snapshot()
    activity_matched = load_activity_matched_snapshot()
    methods = [
        "FN",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "CD_paper",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    ]
    tasks = ["stm", "narma", "parity", "mg"]
    row_labels = [
        "reset-encoded FN",
        "collective relaxation",
        "unequal local relaxation",
        "pair loss",
        "uniform local relaxation",
        "local gain/loss",
        "exchange-assisted",
        "dephasing",
    ]
    task_labels = [
        "STM\nmemory",
        "NARMA-10\nNMSE",
        "parity\ncapacity",
        "MG\n150-step\nMSE",
    ]

    gain_matrix = np.zeros((len(methods), len(tasks)))
    raw_pvalues = {}
    sample_sizes = {}
    for row_index, method in enumerate(methods):
        for col_index, task in enumerate(tasks):
            if method == "CD_paper":
                continue
            method_values = map_cell(rows, task, method)
            baseline_values = map_cell(rows, task, "CD_paper")
            paired = paired_summary(
                method_values,
                baseline_values,
                higher_better=HIGHER[task],
                percent=True,
            )
            gain_matrix[row_index, col_index] = paired["mean"]
            raw_pvalues[(row_index, col_index)] = signflip_permutation_p(
                paired["raw"]
            )
            sample_sizes[task] = paired["n"]
    adjusted = holm_adjust(raw_pvalues)

    benefit_map = LinearSegmentedColormap.from_list(
        "benefit", [RED, "#FFFFFF", BLUE], N=256
    )
    # The heatmap carries the structural map.  Two wider, landscape panels are
    # stacked at right, mirroring the reference paper's compact top/bottom
    # construction and avoiding the previous sub-one-inch side panels.
    fig_width, fig_height = st.WIDTH_FULL, 3.45
    fig = plt.figure(figsize=(fig_width, fig_height), layout="none")
    ax = fig.add_axes(
        [
            1.46 / fig_width,
            0.50 / fig_height,
            2.82 / fig_width,
            2.64 / fig_height,
        ]
    )
    clipped = np.clip(gain_matrix, -60, 60)
    image = ax.imshow(
        clipped,
        cmap=benefit_map,
        vmin=-60,
        vmax=60,
        aspect="auto",
        interpolation="nearest",
    )
    ax.grid(False)
    reference_row = methods.index("CD_paper")
    ax.add_patch(
        Rectangle(
            (-0.5, reference_row - 0.5),
            len(tasks),
            1,
            facecolor="#F2F3F5",
            edgecolor="none",
            zorder=2,
        )
    )
    ax.vlines(
        np.arange(0.5, len(tasks) - 0.5, 1.0),
        -0.5,
        len(methods) - 0.5,
        colors=PAPER,
        linewidth=0.75,
        zorder=2.5,
    )
    ax.hlines(
        np.arange(0.5, len(methods) - 0.5, 1.0),
        -0.5,
        len(tasks) - 0.5,
        colors=PAPER,
        linewidth=0.90,
        zorder=2.5,
    )

    for row_index, method in enumerate(methods):
        if method == "CD_paper":
            ax.text(
                1.5,
                row_index,
                "reference",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=7.40,
                style="italic",
                zorder=3,
            )
            continue
        for column_index, task in enumerate(tasks):
            value = gain_matrix[row_index, column_index]
            pvalue = adjusted[(row_index, column_index)]
            stars = (
                "***"
                if pvalue < 0.001
                else (
                    "**"
                    if pvalue < 0.01
                    else ("*" if pvalue < 0.05 else "")
                )
            )
            label = f"{value:+.0f}%{stars}"
            if task in ("stm", "parity"):
                raw_mean = np.mean(
                    list(map_cell(rows, task, method).values())
                )
                if raw_mean < 1e-6:
                    label = "ZERO"
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.35,
                color=(
                    PAPER
                    if abs(clipped[row_index, column_index]) >= 39
                    else INK
                ),
                zorder=3,
            )

    ax.axhline(0.5, color=INK, linewidth=1.10)
    ax.axhline(len(methods) - 1.5, color=INK, linewidth=1.10)
    ax.axvline(2.5, color=INK, linewidth=0.90)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_labels)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(row_labels)
    ax.get_yticklabels()[1].set_color(BLUE)
    ax.get_yticklabels()[4].set_color(MUTED)
    ax.get_yticklabels()[7].set_color(MUTED)
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(st.RC["axes.linewidth"])
    ax.tick_params(
        axis="x",
        which="both",
        top=False,
        bottom=False,
        length=0,
    )
    ax.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        length=0,
        pad=5,
    )

    colorbar_ax = fig.add_axes(
        [
            4.34 / fig_width,
            0.50 / fig_height,
            0.075 / fig_width,
            2.64 / fig_height,
        ]
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax, extend="both")
    colorbar.set_ticks([-60, 0, 60])
    colorbar.outline.set_edgecolor(INK)
    colorbar.outline.set_linewidth(st.RC["axes.linewidth"])
    colorbar.ax.tick_params(
        which="both",
        right=True,
        left=False,
        labelsize=st.MIN_FONT_SIZE,
        pad=2.0,
    )

    # Panel b: the paired, variance-normalized finite-size replication.  The
    # artifact stores mean within-lineage relative effects and deterministic
    # percentile-bootstrap intervals from all 24 paired lineages per size.
    sizes = np.arange(4, 9)
    ax_size = fig.add_axes(
        [
            4.82 / fig_width,
            1.94 / fig_height,
            2.03 / fig_width,
            1.20 / fig_height,
        ]
    )
    finite_summaries = finite_size["summaries"]["primary_fixed"]
    endpoints = {}
    for task, color, marker, linestyle in (
        ("stm", BLUE, "o", "-"),
        ("narma10", RED, "s", "--"),
    ):
        summaries = [
            finite_summaries[str(size)][task]["B3_collective"][
                "versus_local"
            ]["relative"]
            for size in sizes
        ]
        means = 100.0 * np.asarray(
            [float(summary["mean"]) for summary in summaries]
        )
        lower = means - 100.0 * np.asarray(
            [float(summary["ci95_percentile"][0]) for summary in summaries]
        )
        upper = 100.0 * np.asarray(
            [float(summary["ci95_percentile"][1]) for summary in summaries]
        ) - means
        endpoints[task] = float(means[-1])
        ax_size.errorbar(
            sizes,
            means,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=color if task == "stm" else PAPER,
            markeredgecolor=color,
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
    ax_size.text(
        7.94,
        endpoints["stm"] + 3.4,
        "STM",
        color=BLUE,
        fontsize=st.MIN_FONT_SIZE,
        ha="right",
        va="bottom",
    )
    ax_size.text(
        7.94,
        endpoints["narma10"] - 4.2,
        "NARMA-10",
        color=RED,
        fontsize=st.MIN_FONT_SIZE,
        ha="right",
        va="top",
    )
    observable_counts = [
        finite_size["measurement_scaling"][str(size)]["observables"]
        for size in sizes
    ]
    ax_size.axhline(0, color=INK, linewidth=0.7)
    ax_size.set_xlim(3.75, 8.25)
    # Reserve the upper-left band for the legend: the N=5 STM interval ends
    # below it, while the higher-N points sit safely to its right.
    ax_size.set_ylim(0, 78)
    ax_size.set_yticks([0, 20, 40, 60])
    ax_size.set_xticks(sizes)
    ax_size.set_xticklabels(
        [f"{size}/{count}" for size, count in zip(sizes, observable_counts)]
    )
    ax_size.set_xlabel("$N$ / observables", labelpad=1.2)
    ax_size.set_ylabel("benefit vs local (%)", labelpad=1.5)
    ax_size.tick_params(
        axis="both",
        which="major",
        labelsize=st.MIN_FONT_SIZE,
        pad=1.5,
    )
    ax_size.text(
        0.025,
        0.965,
        "3 Pauli bases",
        transform=ax_size.transAxes,
        color=MUTED,
        fontsize=st.MIN_FONT_SIZE,
        ha="left",
        va="top",
    )
    style_axis(ax_size, "y", minor_axis="y")

    # Panel c: every one of the eight calibrated pairs moves upward from local
    # to collective relaxation.  Larger diamonds show the design means.  The matched
    # activity estimate and the paired STM interval are printed in the panel.
    ax_match = fig.add_axes(
        [
            4.82 / fig_width,
            0.50 / fig_height,
            2.03 / fig_width,
            1.20 / fig_height,
        ]
    )
    local_stm = np.asarray(
        [float(row["local_stm"]) for row in activity_matched["rows"]],
        dtype=float,
    )
    collective_stm = np.asarray(
        [float(row["collective_stm"]) for row in activity_matched["rows"]],
        dtype=float,
    )
    for local_value, collective_value in zip(local_stm, collective_stm):
        ax_match.plot(
            [0, 1],
            [local_value, collective_value],
            color="#B8BDC5",
            linewidth=0.75,
            zorder=1,
        )
    ax_match.scatter(
        np.zeros(len(local_stm)),
        local_stm,
        s=16,
        marker="o",
        facecolor=PAPER,
        edgecolor=MUTED,
        linewidth=0.75,
        zorder=3,
    )
    ax_match.scatter(
        np.ones(len(collective_stm)),
        collective_stm,
        s=16,
        marker="o",
        facecolor=BLUE,
        edgecolor=BLUE,
        linewidth=0.75,
        zorder=3,
    )
    local_mean = float(np.mean(local_stm))
    collective_mean = float(np.mean(collective_stm))
    ax_match.plot(
        [0, 1],
        [local_mean, collective_mean],
        color=INK,
        linewidth=1.35,
        zorder=4,
    )
    ax_match.scatter(
        [0, 1],
        [local_mean, collective_mean],
        s=28,
        marker="D",
        facecolor=[PAPER, BLUE],
        edgecolor=[MUTED, BLUE],
        linewidth=0.9,
        zorder=5,
    )
    ax_match.text(
        0.035,
        0.965,
        "$\\Delta=+3.31$\n$[1.39,5.22]$; 8/8\n"
        "$\\Delta\\mathcal{J}$ unresolved",
        transform=ax_match.transAxes,
        fontsize=st.MIN_FONT_SIZE,
        color=INK,
        ha="left",
        va="top",
        linespacing=1.04,
    )
    ax_match.set_xlim(-0.25, 1.25)
    ax_match.set_ylim(8.0, 20.5)
    ax_match.set_xticks([0, 1])
    ax_match.set_xticklabels(["local", "collective"])
    ax_match.set_yticks([8, 12, 16, 20])
    ax_match.set_ylabel("STM capacity", labelpad=1.5)
    ax_match.tick_params(
        axis="both",
        which="major",
        labelsize=st.MIN_FONT_SIZE,
        pad=1.5,
    )
    style_axis(ax_match, "y", minor_axis="y")

    panel_labels(fig, [ax, ax_size], "ab")
    ax_match.text(
        0.98,
        0.96,
        "(c)",
        transform=ax_match.transAxes,
        ha="right",
        va="top",
        fontsize=st.PANEL_LABEL_SIZE,
        color=INK,
    )

    fig.canvas.draw()
    plot_box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    row_height = plot_box.height / len(methods)
    size_box = ax_size.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    match_box = ax_match.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    if (
        plot_box.width < 2.80
        or row_height < 0.30
        or size_box.width < 2.00
        or size_box.height < 1.18
        or match_box.width < 2.00
        or match_box.height < 1.18
    ):
        raise RuntimeError(
            "fig_map unified geometry is too compact: "
            f"plot={plot_box.width:.3f} x {plot_box.height:.3f} in, "
            f"row height={row_height:.3f} in; "
            f"side panels={size_box.width:.3f} x {size_box.height:.3f}, "
            f"{match_box.width:.3f} x {match_box.height:.3f} in"
        )
    save(fig, "fig_map.pdf")
    print(
        "  fig_map: paired sign-flip permutation + Holm; "
        f"minimum adjusted p={min(adjusted.values()):.3g}; "
        f"unified plot={plot_box.width:.3f} x {plot_box.height:.3f} in; "
        f"row height={row_height:.3f} in; "
        f"finite-size n={finite_size['lineages_per_size']} per N; "
        f"activity matched n={activity_matched['n_pairs']}"
    )


def fig_scaling(rows: list[dict]):
    if not os.path.isfile(REVISION_SCALING):
        raise SystemExit(
            f"ERROR: missing sealed normalized-scaling artifact {REVISION_SCALING}"
        )
    with open(REVISION_SCALING, encoding="utf-8") as handle:
        normalized = json.load(handle)
    if (
        normalized.get("status") != "complete"
        or normalized.get("complete_checkpoints") != 80
        or normalized.get("expected_checkpoints") != 80
        or normalized.get("ridge_boundary_audit", {}).get(
            "upper_boundary_is_bracketed"
        )
        is not True
        or normalized.get("invariant_audit", {}).get("all_passed") is not True
        or normalized.get("protocol", {}).get("schemes") != ["variance"]
    ):
        raise SystemExit(
            "ERROR: normalized-scaling artifact is incomplete or failed its "
            "production audits"
        )

    sizes = [4, 5, 6, 7, 8]
    normalized_summary = normalized["summary_by_scheme"]["variance"]
    raw_normalized_rows = normalized["raw_rows"]

    def percentile_interval(summary: dict, scale: float = 1.0):
        lo, hi = summary["ci95_percentile"]
        return {
            "mean": scale * float(summary["mean"]),
            "lo": scale * float(lo),
            "hi": scale * float(hi),
        }

    fig, axes = fixed_panel_figure(3)
    ax_original, ax_normalized, ax_absolute = axes

    # Panel a reproduces the original U[-1,1] coupling convention.  Both
    # protocols use the same estimand: the mean within-instance relative
    # improvement.  Student-t intervals here remain distinct from the fresh
    # percentile-bootstrap control in panel b.
    for task, color, marker, linestyle, label in (
        ("stm", BLUE, "o", "-", "STM"),
        ("narma", RED, "s", "--", "NARMA-10"),
    ):
        summaries = []
        for size in sizes:
            collective = cell(
                rows, "B_scale", task, "B3_collective", N=size
            )
            local = cell(rows, "B_scale", task, "CD_paper", N=size)
            shared = sorted(set(collective) & set(local))
            if task == "stm":
                relative = np.asarray(
                    [
                        (collective[seed] - local[seed]) / local[seed]
                        for seed in shared
                    ],
                    dtype=float,
                )
            else:
                relative = np.asarray(
                    [
                        (local[seed] - collective[seed]) / local[seed]
                        for seed in shared
                    ],
                    dtype=float,
                )
            summaries.append(t_summary(100.0 * relative))
        means = np.array([item["mean"] for item in summaries])
        lower = means - np.array([item["lo"] for item in summaries])
        upper = np.array([item["hi"] for item in summaries]) - means
        ax_original.errorbar(
            sizes,
            means,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=color if task == "stm" else PAPER,
            markeredgecolor=color,
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            label=label,
        )
    ax_original.axhline(0, color=INK, linewidth=0.8)
    ax_original.set_xticks(sizes)
    ax_original.set_xlabel("number of qubits $N$")
    ax_original.set_ylabel("collective gain (%)")
    st.legend(ax_original, loc="lower right")
    style_axis(
        ax_original,
        "both",
        square=True,
        minor_axis="y",
    )

    # Panel b is the fresh, fully paired variance-normalized control.  The
    # aggregate stores mean within-seed relative effects and deterministic
    # percentile-bootstrap intervals.
    for key, color, marker, linestyle, label in (
        ("stm_relative_advantage", BLUE, "o", "-", "STM"),
        ("narma_relative_improvement", RED, "s", "--", "NARMA-10"),
    ):
        summaries = [
            percentile_interval(normalized_summary[str(size)][key], 100.0)
            for size in sizes
        ]
        means = np.array([item["mean"] for item in summaries])
        lower = means - np.array([item["lo"] for item in summaries])
        upper = np.array([item["hi"] for item in summaries]) - means
        ax_normalized.errorbar(
            sizes,
            means,
            yerr=np.vstack([lower, upper]),
            color=color,
            marker=marker,
            linestyle=linestyle,
            markerfacecolor=color if key.startswith("stm") else PAPER,
            markeredgecolor=color,
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            label=label,
        )
    ax_normalized.axhline(0, color=INK, linewidth=0.8)
    ax_normalized.set_xticks(sizes)
    ax_normalized.set_xlabel("number of qubits $N$")
    ax_normalized.set_ylabel("collective gain (%)")
    st.legend(ax_normalized, loc="lower left")
    style_axis(
        ax_normalized,
        "both",
        square=True,
        minor_axis="y",
    )

    # Panel c shows the absolute STM scale behind the relative effect in panel b.
    # Student-t intervals are recomputed directly from the eight raw instances.
    for method in ("B3_collective", "CD_paper"):
        summaries = []
        for size in sizes:
            values = [
                float(row["stm"]["selected_test"])
                for row in raw_normalized_rows
                if row["scheme"] == "variance"
                and row["n_qubits"] == size
                and row["method"] == method
            ]
            summaries.append(t_summary(values))
        means = np.array([item["mean"] for item in summaries])
        lower = means - np.array([item["lo"] for item in summaries])
        upper = np.array([item["hi"] for item in summaries]) - means
        ax_absolute.errorbar(
            sizes,
            means,
            yerr=np.vstack([lower, upper]),
            color=C[method],
            marker=MARKER[method],
            linestyle=LINESTYLE[method],
            markerfacecolor=PAPER if method == "CD_paper" else C[method],
            markeredgecolor=C[method],
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            label="collective" if method == "B3_collective" else "uniform local",
        )
    ax_absolute.set_xticks(sizes)
    ax_absolute.set_xlabel("number of qubits $N$")
    ax_absolute.set_ylabel("STM capacity")
    ax_absolute.set_xlim(3.85, 8.25)
    ax_absolute.set_ylim(7.2, 16.8)
    for y_key, method, text_label in (
        (16.20, "B3_collective", "collective"),
        (15.45, "CD_paper", "uniform local"),
    ):
        ax_absolute.plot(
            [4.08, 4.42],
            [y_key, y_key],
            color=C[method],
            linestyle=LINESTYLE[method],
            linewidth=DATA_LINEWIDTH,
            zorder=4,
        )
        ax_absolute.plot(
            [4.25],
            [y_key],
            color=C[method],
            marker=MARKER[method],
            markerfacecolor=PAPER if method == "CD_paper" else C[method],
            markeredgecolor=C[method],
            markersize=MARKER_SIZE,
            linestyle="none",
            zorder=5,
        )
        ax_absolute.text(
            4.52,
            y_key,
            text_label,
            ha="left",
            va="center",
            fontsize=8.0,
            color=INK,
        )
    style_axis(
        ax_absolute,
        "both",
        square=True,
        minor_axis="y",
    )

    shared_lower = min(ax_original.get_ylim()[0], ax_normalized.get_ylim()[0])
    shared_upper = max(ax_original.get_ylim()[1], ax_normalized.get_ylim()[1])
    ax_original.set_ylim(shared_lower, shared_upper)
    ax_normalized.set_ylim(shared_lower, shared_upper)
    panel_labels(fig, axes, "abc")
    save(fig, "fig_scaling.pdf")


def fig_budget():
    strength_path = os.path.join(
        REVISION_TUNING_DIR, "strength_extension", "six_channel_aggregate.json"
    )
    if not os.path.isfile(strength_path):
        raise SystemExit(
            f"ERROR: missing revision tuning artifact {strength_path}"
        )
    with open(strength_path, encoding="utf-8") as handle:
        strength = json.load(handle)

    methods = [
        "CD_paper",
        "B3_collective",
        "A1_heterogeneous",
        "B2_thermal",
        "B4_loss_exchange",
        "B5_pair",
    ]
    if set(strength.get("methods", {})) != set(methods):
        raise SystemExit(
            "ERROR: revision strength aggregate does not contain exactly the "
            "six non-dephasing process families"
        )
    expected_rows = 0
    for method in methods:
        method_result = strength["methods"][method]
        bracket = method_result.get("curve_bracket", {})
        curve = bracket.get("curve", [])
        if not bracket.get("bracketed"):
            raise SystemExit(
                f"ERROR: {method} validation maximum is not bracketed; "
                "refusing to plot an incomplete strength ranking"
            )
        if len(curve) < 7 or any(int(point.get("n", 0)) != 20 for point in curve):
            raise SystemExit(
                f"ERROR: {method} strength curve is incomplete or has an "
                "unexpected seed count"
            )
        multipliers = [float(point["multiplier"]) for point in curve]
        if len(set(multipliers)) != len(multipliers):
            raise SystemExit(f"ERROR: duplicate strength point for {method}")
        expected_rows += 20 * len(curve)
    if len(strength.get("raw_rows", [])) != expected_rows:
        raise SystemExit(
            "ERROR: strength aggregate raw-row count does not match its "
            "complete validation curves"
        )
    if len(strength.get("raw_provenance", [])) != expected_rows:
        raise SystemExit("ERROR: strength aggregate provenance is incomplete")

    fig, axes = fixed_panel_figure(2, gap=1.08)
    ax_curve, ax_rank = axes

    for method in methods:
        curve = strength["methods"][method]["curve_bracket"]["curve"]
        x = np.array([float(item["multiplier"]) for item in curve])
        mean = np.array([float(item["validation_mean"]) for item in curve])
        half = np.array(
            [
                stats.t.ppf(0.975, int(item["n"]) - 1)
                * float(item["validation_se"])
                for item in curve
            ]
        )
        ax_curve.errorbar(
            x,
            mean,
            yerr=half,
            color=C[method],
            marker=MARKER[method],
            linestyle=LINESTYLE[method],
            markerfacecolor=PAPER if method == "CD_paper" else C[method],
            markeredgecolor=C[method],
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            label=LABEL[method],
        )
    ax_curve.set_xscale("log")
    ax_curve.set_xlabel("dissipative-strength multiplier")
    ax_curve.set_ylabel("validation STM capacity")
    style_axis(ax_curve, "both", square=True)
    # Panel (b) supplies the method names, avoiding a dense legend over the
    # six validation curves while preserving the shared color/marker coding.

    ranked = sorted(
        methods,
        key=lambda method: strength["methods"][method]["leave_one_seed_out"][
            "test_mean"
        ],
    )
    for y, method in enumerate(ranked):
        item = strength["methods"][method]["leave_one_seed_out"]
        half = stats.t.ppf(0.975, 19) * float(item["test_se"])
        ax_rank.errorbar(
            float(item["test_mean"]),
            y,
            xerr=half,
            color=C[method],
            marker=MARKER[method],
            markerfacecolor=PAPER if method == "CD_paper" else C[method],
            markeredgecolor=C[method],
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linestyle="none",
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
        )
    ax_rank.set_yticks(range(len(ranked)))
    ax_rank.set_yticklabels(
        [
            {
                "CD_paper": "uniform local",
                "B3_collective": "collective",
                "A1_heterogeneous": "unequal local",
                "B2_thermal": "local gain/loss",
                "B4_loss_exchange": "exchange-assisted",
                "B5_pair": "pair",
            }[method]
            for method in ranked
        ]
    )
    ax_rank.set_xlabel("held-out STM capacity")
    style_axis(ax_rank, "x", square=True)
    panel_labels(fig, axes, "ab")
    save(fig, "fig_budget.pdf")


def diagnostic_values(rows: list[dict], N: int, method: str, key: str) -> list[float]:
    return [
        float(row["diagnostics"][key])
        for row in rows
        if row.get("block") == "E_diag"
        and row.get("N") == N
        and row.get("method") == method
        and row.get("diagnostics")
        and isinstance(row["diagnostics"].get(key), (int, float))
    ]


def fig_predict(rows: list[dict]):
    methods = [
        "B5_pair",
        "A1_heterogeneous",
        "B3_collective",
        "CD_paper",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    ]
    names = {
        "B5_pair": "pair",
        "A1_heterogeneous": "unequal local",
        "B3_collective": "collective",
        "CD_paper": "uniform local",
        "B2_thermal": "local gain/loss",
        "B4_loss_exchange": "exchange-assisted",
        "B1_dephasing": "dephasing",
    }
    fig, axes = fixed_panel_figure(2, gap=1.08, height=2.45, bottom=0.66)
    ax_gate, ax_gap = axes

    # Panel (a): a ranked lollipop plot is easier to compare than unrelated
    # colored bars and keeps attention on one scalar diagnostic.
    defect = {
        method: t_summary(diagnostic_values(rows, 5, method, "unitality_defect"))
        for method in methods
    }
    y_positions = np.arange(len(methods))
    blue_tints = list(reversed(st.tints(BLUE, 7)))
    for y_position, method, color in zip(y_positions, methods, blue_tints):
        summary = defect[method]
        if method == "B1_dephasing":
            ax_gate.scatter(
                [0],
                [y_position],
                marker="X",
                s=42,
                color=INK,
                zorder=4,
            )
            ax_gate.annotate(
                "zero",
                (0, y_position),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7.8,
                color=INK,
            )
            continue
        ax_gate.hlines(
            y_position,
            0,
            summary["mean"],
            color=color,
            linewidth=DATA_LINEWIDTH,
            zorder=2,
        )
        ax_gate.errorbar(
            summary["mean"],
            y_position,
            xerr=interval_error(summary),
            color=color,
            marker="o",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=MARKER_SIZE,
            linestyle="none",
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
    ax_gate.set_yticks(y_positions)
    ax_gate.set_yticklabels([names[method] for method in methods])
    ax_gate.invert_yaxis()
    # Leave a full marker radius to the left of the true zero value.
    ax_gate.set_xlim(-1.6, 28.0)
    ax_gate.set_xlabel("jump-only unitality defect")
    style_axis(ax_gate, "x", square=True, minor_axis="x")

    # Panel (b): color answers only the key comparison. All remaining designs
    # are neutral and named directly, so no rainbow legend is needed.
    scatter_methods = methods[:-1]
    x_means, y_means = [], []
    label_positions = {
        "B3_collective": (0.18, 11.62, "left"),
        "A1_heterogeneous": (0.32, 10.20, "left"),
        "B5_pair": (0.31, 9.52, "left"),
        "CD_paper": (0.39, 8.02, "left"),
        "B2_thermal": (0.71, 8.78, "left"),
        "B4_loss_exchange": (0.60, 7.18, "center"),
    }
    for method in scatter_methods:
        gap = t_summary(diagnostic_values(rows, 5, method, "spectral_gap"))
        memory = t_summary(cell(rows, "A_table", "stm", method, N=5).values())
        x_means.append(gap["mean"])
        y_means.append(memory["mean"])
        color = BLUE if method == "B3_collective" else GRAY
        face = color if method == "B3_collective" else PAPER
        ax_gap.errorbar(
            gap["mean"],
            memory["mean"],
            xerr=interval_error(gap),
            yerr=interval_error(memory),
            marker=MARKER[method],
            markersize=MARKER_SIZE + 0.5,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=ERROR_LINEWIDTH,
            linestyle="none",
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
        label_x, label_y, horizontal = label_positions[method]
        ax_gap.annotate(
            names[method],
            (gap["mean"], memory["mean"]),
            xytext=(label_x, label_y),
            textcoords="data",
            ha=horizontal,
            va="center",
            fontsize=7.4,
            color=BLUE if method == "B3_collective" else INK,
            arrowprops={
                "arrowstyle": "-",
                "color": BLUE if method == "B3_collective" else GRAY,
                "linewidth": 0.55,
            },
        )
    rho, _ = stats.spearmanr(x_means, y_means)
    ax_gap.set_xlim(0.04, 0.98)
    ax_gap.set_ylim(6.75, 12.75)
    ax_gap.set_xlabel("midpoint Liouvillian gap")
    ax_gap.set_ylabel("STM capacity")
    ax_gap.text(
        0.96,
        0.96,
        rf"Spearman $\rho={rho:+.2f}$",
        transform=ax_gap.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=INK,
    )
    style_axis(ax_gap, "both", square=True)

    panel_labels(fig, axes, "ab")
    save(fig, "fig_predict.pdf")


def fig_shotmap():
    if not os.path.isfile(REVISION_MEASUREMENT):
        raise SystemExit(
            "ERROR: complete equal-nominal-preparation aggregate not found at "
            + REVISION_MEASUREMENT
        )
    with open(REVISION_MEASUREMENT, encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("validation", {}).get("status") != "complete":
        raise SystemExit(
            "ERROR: equal-nominal-preparation measurement aggregate is incomplete"
        )

    methods = [
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
    ]
    finite_budgets = sorted(
        {
            int(row["budget"])
            for row in result["paired_vs_local"]
            if row["budget"] != "exact"
        }
    )
    measurement_models = {
        row["measurement_model"]
        for row in result["paired_vs_local"]
        if row["budget"] != "exact"
    }
    all_methods = {
        row["channel"]
        for row in result["paired_vs_local"]
        if row["budget"] != "exact"
    }
    simultaneous_family = (
        len(measurement_models)
        * len(finite_budgets)
        * len(all_methods)
        * (len(all_methods) - 1)
        // 2
    )
    if simultaneous_family != 210:
        raise SystemExit(
            "ERROR: unexpected measurement-inference family size "
            f"{simultaneous_family}, expected 210"
        )
    simultaneous_critical = stats.t.ppf(
        1.0 - 0.05 / (2.0 * simultaneous_family), 19
    )
    exact_position = finite_budgets[-1] * 8
    lookup = {
        (row["measurement_model"], row["budget"], row["channel"]): row
        for row in result["paired_vs_local"]
    }

    # Two landscape panels use the complete full-width canvas.  The earlier
    # 4.1-inch figure left almost three inches unused in a figure* environment.
    fig_width, fig_height = st.WIDTH_FULL, 2.55
    fig = plt.figure(figsize=(fig_width, fig_height), layout="none")
    axes = np.asarray(
        [
            fig.add_axes(
                [0.72 / fig_width, 0.53 / fig_height, 2.72 / fig_width, 1.70 / fig_height]
            ),
            fig.add_axes(
                [4.05 / fig_width, 0.53 / fig_height, 2.72 / fig_width, 1.70 / fig_height],
                sharey=None,
            ),
        ],
        dtype=object,
    )
    axes[1].sharey(axes[0])
    for ax, model in (
        (axes[0], "independent"),
        (axes[1], "grouped"),
    ):
        for method in methods:
            summaries = [
                lookup[(model, str(budget), method)] for budget in finite_budgets
            ]
            means = np.array([item["mean"] for item in summaries])
            simultaneous_half_width = simultaneous_critical * np.array(
                [item["se"] for item in summaries]
            )
            ax.errorbar(
                finite_budgets,
                means,
                yerr=simultaneous_half_width,
                color=C[method],
                marker=MARKER[method],
                linestyle=LINESTYLE[method],
                markerfacecolor=C[method],
                markeredgecolor=C[method],
                markeredgewidth=ERROR_LINEWIDTH,
                markersize=MARKER_SIZE,
                linewidth=DATA_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                elinewidth=ERROR_LINEWIDTH,
                label=LABEL[method],
            )
            exact = lookup[(model, "exact", method)]
            ax.plot(
                [finite_budgets[-1], exact_position],
                [means[-1], exact["mean"]],
                color=C[method],
                linestyle=":",
                linewidth=1.45,
                alpha=0.75,
            )
            ax.errorbar(
                [exact_position],
                [exact["mean"]],
                yerr=np.array(
                    [
                        [exact["mean"] - exact["ci95_low"]],
                        [exact["ci95_high"] - exact["mean"]],
                    ]
                ),
                color=C[method],
                marker=MARKER[method],
                linestyle="none",
                markerfacecolor=PAPER,
                markeredgecolor=C[method],
                markeredgewidth=ERROR_LINEWIDTH,
                markersize=MARKER_SIZE,
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                elinewidth=ERROR_LINEWIDTH,
            )
        ax.axhline(0, color=INK, linewidth=0.9)
        ax.axvline(
            finite_budgets[-1] * 3,
            color=GRID,
            linewidth=0.9,
            linestyle=":",
        )
        ax.set_xscale("log")
        ax.set_xlim(2_000, exact_position * 1.35)
        ax.set_ylim(-3.05, 7.00)
        ax.set_xticks(
            [
                finite_budgets[0],
                finite_budgets[2],
                finite_budgets[4],
                finite_budgets[-1],
                exact_position,
            ]
        )
        ax.set_xticklabels(["2.9k", "46k", "0.74m", "11.8m", "\nexact"])
        ax.set_xticklabels(["2.9k", "46k", "0.74m", "11.8m", "exact"])
        ax.minorticks_off()
        ax.set_xlabel("preparations per time step")
        style_axis(ax, "both", minor_axis="y")
        ax.text(
            0.97,
            0.96,
            "independent observables" if model == "independent" else "grouped Pauli settings",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=st.MIN_FONT_SIZE,
            color=MUTED,
        )

    axes[0].set_ylabel("STM-capacity difference\nfrom uniform local relaxation")
    axes[1].tick_params(axis="y", labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    st.legend(
        axes[0],
        handles=handles,
        labels=[
            "collective",
            "unequal",
            "pair",
            "local gain/loss",
            "exchange",
        ],
        loc="upper left",
        ncol=2,
        fontsize=st.MIN_FONT_SIZE,
        handlelength=1.45,
        borderpad=0.30,
        labelspacing=0.20,
        columnspacing=0.70,
    )
    panel_labels(fig, axes, "ab")

    fig.canvas.draw()
    boxes = [
        axis.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        for axis in axes
    ]
    if any(box.width < 2.70 or box.height < 1.68 for box in boxes):
        raise RuntimeError(
            "fig_shotmap violates the full-width landscape contract: "
            + ", ".join(f"{box.width:.3f} x {box.height:.3f} in" for box in boxes)
        )
    save(fig, "fig_shotmap.pdf")


def fig_adaptive(rows: list[dict]):
    review = load_review()
    uniform = cell(rows, "F_adaptive", "stm", "A0_uniform", N=5)
    unequal = cell(rows, "F_adaptive", "stm", "A1_random", N=5)
    optimized = {
        method: {
            row["seed"]: row["value"]
            for row in review
            if row.get("block") == "R_optfix" and row.get("method") == method
        }
        for method in ("A3_eq", "A4_eq")
    }
    profile_summaries = [
        paired_summary(unequal, uniform),
        paired_summary(optimized["A3_eq"], uniform),
        paired_summary(optimized["A4_eq"], uniform),
    ]
    profile_seed_sets = {
        tuple(sorted(values))
        for values in (uniform, unequal, *optimized.values())
    }
    if (
        any(item["n"] != 32 for item in profile_summaries)
        or len(profile_seed_sets) != 1
    ):
        raise SystemExit(
            "ERROR: profile-ladder data are not a complete 32-seed paired grid"
        )

    joint_rows = [
        row
        for row in rows
        if row.get("block") == "G_joint" and row.get("task") == "stm"
    ]
    joint_cells = {
        method: {
            row["seed"]: row["value"]
            for row in joint_rows
            if row.get("method") == method
        }
        for method in (
            "G_local_uniform",
            "G_local_learned",
            "G_coll_uniform",
            "G_coll_learned",
        )
    }
    local_gain = paired_summary(
        joint_cells["G_local_learned"], joint_cells["G_local_uniform"]
    )
    collective_gain = paired_summary(
        joint_cells["G_coll_learned"], joint_cells["G_coll_uniform"]
    )
    joint_seed_sets = {
        tuple(sorted(values)) for values in joint_cells.values()
    }
    if (
        local_gain["n"] != 24
        or collective_gain["n"] != 24
        or len(joint_seed_sets) != 1
    ):
        raise SystemExit(
            "ERROR: family-profile data are not a complete 24-seed paired grid"
        )

    # One compact forest plot keeps both exploratory protocols visible without
    # giving them a full-width two-panel treatment. The shared quantitative
    # axis is valid because every mark is an STM-capacity difference; the
    # internal group headers keep the two baselines and seed counts explicit.
    fig_width, fig_height = st.WIDTH_COLUMN, 3.12
    fig = plt.figure(figsize=(fig_width, fig_height), layout="none")
    ax = fig.add_axes(
        [
            1.08 / fig_width,
            0.48 / fig_height,
            2.12 / fig_width,
            2.43 / fig_height,
        ]
    )
    header_color = "#F2F3F5"
    ax.axhspan(6.15, 7.05, color=header_color, zorder=0)
    ax.axhspan(1.95, 2.85, color=header_color, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.8, zorder=1)

    profile_y = [5.55, 4.55, 3.55]
    red_shades = st.tints(RED, 6)
    profile_colors = [red_shades[2], RED, ORANGE]
    profile_markers = ["s", "o", "^"]
    for y_position, summary, color, marker in zip(
        profile_y,
        profile_summaries,
        profile_colors,
        profile_markers,
    ):
        ax.errorbar(
            summary["mean"],
            y_position,
            xerr=interval_error(summary),
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linestyle="none",
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
        ax.text(
            summary["hi"] + 0.06,
            y_position,
            f"{summary['mean']:+.2f}",
            ha="left",
            va="center",
            fontsize=7.85,
        )

    joint_y = [1.45, 0.45]
    joint_summaries = [local_gain, collective_gain]
    joint_colors = [MUTED, BLUE]
    joint_markers = ["D", "o"]
    for y_position, summary, color, marker in zip(
        joint_y,
        joint_summaries,
        joint_colors,
        joint_markers,
    ):
        ax.errorbar(
            summary["mean"],
            y_position,
            xerr=interval_error(summary),
            color=color,
            marker=marker,
            markerfacecolor=PAPER if marker == "D" else color,
            markeredgecolor=color,
            markeredgewidth=ERROR_LINEWIDTH,
            markersize=MARKER_SIZE,
            linestyle="none",
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
        if marker == "D":
            label_x, horizontal = summary["lo"] - 0.06, "right"
        else:
            label_x, horizontal = summary["hi"] + 0.06, "left"
        ax.text(
            label_x,
            y_position,
            f"{summary['mean']:+.2f}",
            ha=horizontal,
            va="center",
            fontsize=7.85,
        )

    ax.text(
        0.02,
        6.60,
        "Local profiles vs uniform  ($n=32$)",
        ha="left",
        va="center",
        fontsize=7.75,
        color=INK,
    )
    ax.text(
        0.02,
        2.40,
        "Learned-profile gain  ($n=24$)",
        ha="left",
        va="center",
        fontsize=7.75,
        color=INK,
    )
    ax.set_yticks(profile_y + joint_y)
    ax.set_yticklabels(
        [
            "unequal local",
            "learned static",
            "input-adaptive",
            "local relaxation",
            "collective relaxation",
        ]
    )
    ax.set_xlim(-0.12, 2.05)
    ax.set_ylim(-0.05, 7.10)
    ax.set_xlabel("STM-capacity difference")
    style_axis(ax, "x", minor_axis="x")

    fig.canvas.draw()
    plot_box = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if plot_box.width < 2.10 or fig_width != st.WIDTH_COLUMN:
        raise RuntimeError(
            "fig_adaptive violates the one-column geometry contract: "
            f"canvas={fig_width:.3f} in, plot width={plot_box.width:.3f} in"
        )
    save(fig, "fig_adaptive.pdf")


def fig_prospective():
    """Diagnostic rule fixed before scoring on an independent task ensemble."""
    if not os.path.isfile(REVISION_PROSPECTIVE):
        raise SystemExit(
            f"ERROR: missing sealed fresh interpolation {REVISION_PROSPECTIVE}"
        )
    with open(REVISION_PROSPECTIVE, encoding="utf-8") as handle:
        result = json.load(handle)
    if (
        result.get("status") != "complete"
        or result.get("diagnostic_seed_count") != 20
        or result.get("fresh_task_seed_count") != 24
        or result.get("seed_overlap_with_frozen_diagnostics") != []
        or result.get("expected_checkpoint_count") != 288
        or result.get("complete_checkpoint_count") != 288
        or result.get("ridge_upper_boundary_hits") != 0
        or "not out-of-family" not in result.get("operator_family_status", "")
    ):
        raise SystemExit(
            "ERROR: fresh interpolation is incomplete or violates its "
            "disjoint-ensemble contract"
        )
    diagnostic_rows = result["frozen_diagnostic_rows"]
    task_rows = result["raw_rows"]
    alphas = [
        float(item["alpha"])
        for item in result["results_by_N"]["4"]["summary"]
    ]
    if alphas != [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        raise SystemExit(f"ERROR: unexpected fresh interpolation grid {alphas}")
    selected_by_n = {
        N: float(result["results_by_N"][str(N)]["frozen_selected_alpha"])
        for N in (4, 5)
    }
    if len(set(selected_by_n.values())) != 1:
        raise SystemExit(
            f"ERROR: frozen selection differs across sizes {selected_by_n}"
        )
    selected_alpha = selected_by_n[4]
    if {
        int(row["seed"]) for row in diagnostic_rows
    } & {
        int(row["seed"]) for row in task_rows
    }:
        raise SystemExit(
            "ERROR: plotted diagnostic and task ensembles are not disjoint"
    )

    fig, axes = fixed_panel_figure(2)
    ax_gap, ax_task = axes
    series = {
        4: {
            "color": BLUE,
            "marker": "s",
            "linestyle": "--",
            "fill": PAPER,
        },
        5: {"color": RED, "marker": "o", "linestyle": "-", "fill": RED},
    }
    summaries = {"gap": {}, "task": {}}
    for N in (4, 5):
        summaries["gap"][N] = []
        summaries["task"][N] = []
        for alpha in alphas:
            gap_values = [
                row["spectral_gap"]
                for row in diagnostic_rows
                if row["N"] == N and float(row["alpha"]) == alpha
            ]
            task_values = [
                row["test_mc"]
                for row in task_rows
                if row["N"] == N and float(row["alpha"]) == alpha
            ]
            summaries["gap"][N].append(t_summary(gap_values))
            summaries["task"][N].append(t_summary(task_values))

    for ax in (ax_gap, ax_task):
        ax.axvline(
            selected_alpha,
            color=GRAY,
            linewidth=0.90,
            linestyle=":",
            zorder=1,
        )
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks(alphas)
        ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"])
        ax.set_xlabel(r"local $\leftarrow\ \alpha\ \rightarrow$ collective")
        style_axis(ax, "both", square=True, minor_axis="y")

    for N in (4, 5):
        style = series[N]
        for ax, key in ((ax_gap, "gap"), (ax_task, "task")):
            values = summaries[key][N]
            means = np.array([item["mean"] for item in values])
            lower = np.array([item["lo"] for item in values])
            upper = np.array([item["hi"] for item in values])
            ax.fill_between(
                alphas,
                lower,
                upper,
                color=style["color"],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            ax.plot(
                alphas,
                means,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markerfacecolor=style["fill"],
                markeredgecolor=style["color"],
                markeredgewidth=ERROR_LINEWIDTH,
                markersize=MARKER_SIZE,
                linewidth=DATA_LINEWIDTH,
                label=f"$N={N}$",
                zorder=3,
            )
            selected_index = alphas.index(selected_alpha)
            ax.scatter(
                [selected_alpha],
                [means[selected_index]],
                s=78,
                facecolor="none",
                edgecolor=INK,
                linewidth=1.25,
                zorder=4,
            )

    ax_gap.set_ylabel("midpoint Liouvillian gap\nat $s=0.5$")
    st.legend(ax_gap, loc="lower left", fontsize=8.0)
    ax_gap.text(
        selected_alpha,
        0.97,
        r"selected $.8$",
        transform=ax_gap.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.8,
        color=INK,
    )
    ax_task.set_ylabel("held-out STM capacity")
    rank_four = result["results_by_N"]["4"]
    rank_five = result["results_by_N"]["5"]
    raw_gap_ranks = {}
    for N, stored in ((4, rank_four), (5, rank_five)):
        gap_means = [item["mean"] for item in summaries["gap"][N]]
        task_means = [item["mean"] for item in summaries["task"][N]]
        raw_gap_ranks[N] = float(
            stats.spearmanr(gap_means, task_means).statistic
        )
        if not np.isclose(
            raw_gap_ranks[N],
            -stored["frozen_gap_vs_fresh_mean_spearman_rho"],
        ):
            raise SystemExit(
                "ERROR: raw-gap rank does not negate the stored negative-gap "
                f"rank at N={N}"
            )
    ax_task.text(
        0.04,
        0.95,
        r"raw $\Delta_{\mathcal{L}}$--STM ranks" "\n"
        rf"$N=4:\ \rho={raw_gap_ranks[4]:.2f}$, "
        rf"$p={rank_four['frozen_gap_vs_fresh_mean_exact_permutation_p']:.4f}$"
        "\n"
        rf"$N=5:\ \rho={raw_gap_ranks[5]:.2f}$, "
        rf"$p={rank_five['frozen_gap_vs_fresh_mean_exact_permutation_p']:.4f}$",
        transform=ax_task.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=INK,
    )
    panel_labels(fig, axes, "ab")
    save(fig, "fig_prospective.pdf")


def fig_map_heatmap_legacy(rows: list[dict]):
    """Legacy relative-improvement heatmap retained for reproducibility."""
    map_text_size = 9.20
    map_tick_size = 9.25
    map_axis_size = 10.15
    map_colorbar_size = 9.20
    map_font_floor = 8.60
    plt.rcParams.update(
        {
            "font.size": 9.70,
            "axes.labelsize": map_axis_size,
            "axes.titlesize": 10.35,
            "xtick.labelsize": map_tick_size,
            "ytick.labelsize": map_tick_size,
        }
    )
    map_purple = st.COLLECTIVE
    map_orange = st.ORANGE
    local_gray = st.GRAY
    # Display only genuine structure changes plus the external reset baseline.
    # Keep the original eight-design multiplicity family for the Holm-adjusted
    # cell tests even though unequal local relaxation is now reported only as a
    # profile variation in Section 3.4.
    methods = [
        "FN",
        "B3_collective",
        "B5_pair",
        "CD_paper",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    ]
    tested_methods = [
        "FN",
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "CD_paper",
        "B2_thermal",
        "B4_loss_exchange",
        "B1_dephasing",
    ]
    tasks = ["stm", "narma", "parity", "mg"]
    row_labels = [
        "reset FN",
        "collective",
        "pair loss",
        "uniform local",
        "local gain/loss",
        "exchange",
        "dephasing",
    ]
    task_labels = [
        "STM\nmemory",
        "NARMA-10\nNMSE",
        "parity\ncapacity",
        "MG 150-step\nMSE",
    ]

    gain_matrix = np.zeros((len(methods), len(tasks)))
    raw_pvalues = {}
    for method in tested_methods:
        for task in tasks:
            if method == "CD_paper":
                continue
            summary = paired_summary(
                map_cell(rows, task, method),
                map_cell(rows, task, "CD_paper"),
                higher_better=HIGHER[task],
                percent=True,
            )
            raw_pvalues[(method, task)] = signflip_permutation_p(
                summary["raw"]
            )
    adjusted = holm_adjust(raw_pvalues)
    for row_index, method in enumerate(methods):
        for column_index, task in enumerate(tasks):
            if method == "CD_paper":
                continue
            gain_matrix[row_index, column_index] = paired_summary(
                map_cell(rows, task, method),
                map_cell(rows, task, "CD_paper"),
                higher_better=HIGHER[task],
                percent=True,
            )["mean"]

    # The task map is authored directly at the natural Quantum column width.
    figure_height = 3.22
    fig = st.composite_figure("column", figure_height)
    figure_width = float(fig.get_figwidth())
    map_side = 2.35
    map_bottom_y = 0.38
    map_top_y = map_bottom_y + map_side
    map_right_x = 3.15
    map_left_x = map_right_x - map_side
    map_width = map_side
    map_height = map_side
    ax_map = st.add_axes_inches(
        fig,
        [
            map_left_x,
            map_bottom_y,
            map_width,
            map_height,
        ],
    )
    axes = np.asarray([ax_map], dtype=object)

    # Six continuously driven structures and one external baseline.
    benefit_map = LinearSegmentedColormap.from_list(
        "benefit",
        [map_orange, "#F7F5F1", map_purple],
        N=256,
    )
    clipped = np.clip(gain_matrix, -60, 60)
    image = ax_map.imshow(
        clipped,
        cmap=benefit_map,
        vmin=-60,
        vmax=60,
        aspect="auto",
        interpolation="nearest",
    )
    ax_map.grid(False)
    reference_row = methods.index("CD_paper")
    external_row = methods.index("FN")
    ax_map.add_patch(
        Rectangle(
            (-0.5, reference_row - 0.5),
            len(tasks),
            1,
            facecolor="#F1F2F3",
            edgecolor="none",
            zorder=2,
        )
    )
    ax_map.add_patch(
        Rectangle(
            (-0.5, external_row - 0.5),
            len(tasks),
            1,
            facecolor="none",
            edgecolor=MUTED,
            linestyle="--",
            linewidth=0.90,
            zorder=4,
        )
    )
    ax_map.vlines(
        np.arange(0.5, len(tasks) - 0.5, 1.0),
        -0.5,
        len(methods) - 0.5,
        colors=PAPER,
        linewidth=0.72,
        zorder=2.5,
    )
    ax_map.hlines(
        np.arange(0.5, len(methods) - 0.5, 1.0),
        -0.5,
        len(tasks) - 0.5,
        colors=PAPER,
        linewidth=0.78,
        zorder=2.5,
    )
    for row_index, method in enumerate(methods):
        if method == "CD_paper":
            ax_map.text(
                1.5,
                row_index,
                "reference",
                ha="center",
                va="center",
                color=MUTED,
                fontsize=map_text_size,
                style="italic",
                zorder=3,
            )
            continue
        for column_index, task in enumerate(tasks):
            value = float(gain_matrix[row_index, column_index])
            pvalue = adjusted[(method, task)]
            stars = (
                "***"
                if pvalue < 0.001
                else "**"
                if pvalue < 0.01
                else "*"
                if pvalue < 0.05
                else ""
            )
            label = f"{value:+.0f}%{stars}"
            if task in ("stm", "parity"):
                raw_mean = np.mean(
                    list(map_cell(rows, task, method).values())
                )
                if raw_mean < 1e-6:
                    label = "ZERO"
            rgba = benefit_map(
                (clipped[row_index, column_index] + 60.0) / 120.0
            )
            luminance = (
                0.2126 * rgba[0]
                + 0.7152 * rgba[1]
                + 0.0722 * rgba[2]
            )
            ax_map.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=map_text_size,
                color=PAPER if luminance < 0.52 else INK,
                zorder=3,
            )
    ax_map.axhline(0.5, color=INK, linewidth=0.82)
    ax_map.axhline(len(methods) - 1.5, color=INK, linewidth=0.82)
    ax_map.axvline(2.5, color=INK, linewidth=0.68)
    ax_map.set_xticks(range(len(tasks)))
    ax_map.set_xticklabels(task_labels)
    ax_map.set_yticks(range(len(methods)))
    ax_map.set_yticklabels(row_labels)
    ax_map.get_yticklabels()[0].set_color(MUTED)
    ax_map.get_yticklabels()[0].set_style("italic")
    ax_map.get_yticklabels()[1].set_color(map_purple)
    ax_map.get_yticklabels()[3].set_color(local_gray)
    ax_map.set_xlim(-0.5, len(tasks) - 0.5)
    ax_map.tick_params(
        axis="both",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
        length=0,
        pad=3.0,
        labelsize=map_tick_size,
    )
    for spine in ax_map.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(st.RC["axes.linewidth"])

    colorbar_width = 2.05
    colorbar_ax = st.add_axes_inches(
        fig,
        [
            map_left_x + (map_width - colorbar_width) / 2.0,
            map_top_y + 0.13,
            colorbar_width,
            0.070,
        ],
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    colorbar.set_ticks([-60, 0, 60])
    colorbar.set_ticklabels([r"$-60$", "$0$", r"$+60$"])
    colorbar.outline.set_edgecolor(INK)
    colorbar.outline.set_linewidth(0.60)
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.tick_params(
        which="both",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=0,
        labelsize=map_colorbar_size,
        pad=2.2,
    )
    fig.text(
        (map_left_x + map_width / 2.0) / figure_width,
        (map_top_y + 0.43) / figure_height,
        "relative improvement over uniform local (%)",
        ha="center",
        va="center",
        fontsize=map_colorbar_size,
        color=INK,
    )

    fig.canvas.draw()
    map_box = ax_map.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    geometry_tolerance = 0.005
    if (
        abs(map_box.width - map_side) > geometry_tolerance
        or abs(map_box.height - map_side) > geometry_tolerance
        or abs(map_box.width - map_box.height) > geometry_tolerance
    ):
        raise RuntimeError(
            "fig_map heatmap must be square: "
            f"{map_box.width:.3f} x {map_box.height:.3f} in"
        )
    st.audit_figure(
        fig,
        "fig_map",
        axes=axes,
        overlap_fraction=0.10,
        font_floor=map_font_floor,
    )
    # Exclude the auxiliary colorbar axis from the subplot-to-subplot
    # decoration test in save(); its text and canvas bounds remain covered by
    # the full-figure audit above.
    fig._qrc_audit_axes = axes
    save(fig, "fig_map.pdf")


def fig_task_scores_dense(rows: list[dict]):
    """Full-width absolute task scores with raw seeds and mean intervals."""

    designs = (
        ("collective relaxation", "B3_collective"),
        ("uniform local (reference)", "CD_paper"),
        ("unequal local (profile)", "A1_heterogeneous"),
        ("pair loss", "B5_pair"),
        ("local gain/loss", "B2_thermal"),
        ("exchange-assisted relaxation", "B4_loss_exchange"),
        ("dephasing", "B1_dephasing"),
    )
    tasks = (
        ("stm", "STM capacity", 32),
        ("narma", "NARMA-10 NMSE", 32),
        ("parity", "parity capacity", 16),
        ("mg", "MG-150 MSE", 64),
    )
    # The native envelopes remain a fail-closed data-integrity check.  The
    # displayed limits below focus the structural comparisons and make every
    # omitted observation explicit at the corresponding boundary.
    native_axis_limits = {
        "stm": (-0.45, 13.85),
        "narma": (0.14, 1.16),
        "parity": (-0.15, 6.0),
        "mg": (-0.008, 0.33),
    }
    axis_limits = {
        "stm": (6.5, 13.75),
        "narma": (0.15, 0.575),
        "parity": (3.25, 5.55),
        "mg": (0.0, 0.33),
    }
    axis_ticks = {
        "stm": (7, 9, 11, 13),
        "narma": (0.2, 0.3, 0.4, 0.5),
        "parity": (4.7, 4.9, 5.1),
        "mg": (0.03, 0.04, 0.05),
    }
    expected_offscale = {
        "stm": {"B1_dephasing": (32, 0)},
        "narma": {"B1_dephasing": (0, 32)},
        "parity": {"B1_dephasing": (16, 0)},
        "mg": {},
    }

    cells = {
        task: {
            method: map_cell(rows, task, method)
            for _label, method in designs
        }
        for task, _title, _expected_n in tasks
    }
    for task, _title, expected_n in tasks:
        task_cells = cells[task]
        observed_seed_sets = [set(values) for values in task_cells.values()]
        if (
            any(len(values) != expected_n for values in task_cells.values())
            or not observed_seed_sets
            or any(seeds != observed_seed_sets[0] for seeds in observed_seed_sets[1:])
        ):
            raise RuntimeError(
                f"fig_task_scores requires {expected_n} paired seeds for {task}"
            )
        all_values = np.asarray(
            [value for values in task_cells.values() for value in values.values()],
            dtype=float,
        )
        lower, upper = native_axis_limits[task]
        if (
            not np.all(np.isfinite(all_values))
            or np.min(all_values) < lower - 1e-12
            or np.max(all_values) > upper + 1e-12
        ):
            raise RuntimeError(
                f"fig_task_scores {task} values exceed the declared native scale"
            )
        focus_lower, focus_upper = axis_limits[task]
        actual_offscale = {}
        for _label, method in designs:
            values = np.asarray(list(task_cells[method].values()), dtype=float)
            counts = (
                int(np.count_nonzero(values < focus_lower)),
                int(np.count_nonzero(values > focus_upper)),
            )
            if counts != (0, 0):
                actual_offscale[method] = counts
        if actual_offscale != expected_offscale[task]:
            raise RuntimeError(
                f"fig_task_scores focused-scale contract changed for {task}: "
                f"{actual_offscale}"
            )

    task_winners = {}
    task_better_than_reference = {}
    for task, _title, _expected_n in tasks:
        task_means = {
            method: float(np.mean(list(cells[task][method].values())))
            for _label, method in designs
        }
        task_winners[task] = (
            max(task_means, key=task_means.get)
            if HIGHER[task]
            else min(task_means, key=task_means.get)
        )
        reference_mean = task_means["CD_paper"]
        task_better_than_reference[task] = tuple(
            method
            for _label, method in designs
            if method != "CD_paper"
            and (
                task_means[method] > reference_mean
                if HIGHER[task]
                else task_means[method] < reference_mean
            )
        )
    expected_winners = {
        "stm": "B3_collective",
        "narma": "B3_collective",
        "parity": "B4_loss_exchange",
        "mg": "A1_heterogeneous",
    }
    if task_winners != expected_winners:
        raise RuntimeError(
            "fig_task_scores winner-highlight contract changed: "
            f"{task_winners}"
        )
    expected_better_than_reference = {
        "stm": ("B3_collective", "A1_heterogeneous", "B5_pair"),
        "narma": ("B3_collective", "A1_heterogeneous", "B5_pair"),
        "parity": ("B2_thermal", "B4_loss_exchange"),
        "mg": ("A1_heterogeneous",),
    }
    if task_better_than_reference != expected_better_than_reference:
        raise RuntimeError(
            "fig_task_scores reference-improvement contract changed: "
            f"{task_better_than_reference}"
        )

    figure_width = st.QUANTUM_TEXT_WIDTH
    figure_height = 2.18
    left_margin = 1.66
    right_margin = 0.07
    panel_gap = 0.13
    panel_bottom = 0.42
    panel_height = 1.42
    panel_width = (
        figure_width
        - left_margin
        - right_margin
        - panel_gap * (len(tasks) - 1)
    ) / len(tasks)
    panel_label_size = 10.80
    row_label_size = 9.75
    axis_label_size = 10.25
    tick_size = 9.25
    mean_marker_size = 5.80
    priority_marker_size = 7.00
    mean_linewidth = 1.05
    priority_linewidth = 1.40
    mean_capsize = 3.15
    raw_marker_area = 9.0
    boundary_label_size = st.MIN_FONT_SIZE
    neutral_color = st.NEUTRAL_DESIGN

    fig = st.composite_figure("full", figure_height)
    axes = []

    # Marked piecewise-linear zooms preserve every native value while giving
    # most of each panel to the scientifically discriminating interval.  In
    # parity this separates the pair/reference/gain-loss/exchange cluster; in
    # MG-150 it makes the small unequal-local/reference difference legible.
    # These transforms are deliberately not logarithmic.
    parity_native_knots = np.asarray((3.25, 4.60, 5.15, 5.55), dtype=float)
    parity_display_knots = np.asarray((0.0, 0.10, 0.92, 1.0), dtype=float)
    mg_native_knots = np.asarray((0.0, 0.025, 0.05, 0.11, 0.33), dtype=float)
    mg_display_knots = np.asarray((0.0, 0.07, 0.70, 0.86, 1.0), dtype=float)

    def parity_zoom_forward(values):
        return np.interp(
            np.asarray(values, dtype=float),
            parity_native_knots,
            parity_display_knots,
        )

    def parity_zoom_inverse(values):
        return np.interp(
            np.asarray(values, dtype=float),
            parity_display_knots,
            parity_native_knots,
        )

    def mg_zoom_forward(values):
        return np.interp(
            np.asarray(values, dtype=float),
            mg_native_knots,
            mg_display_knots,
        )

    def mg_zoom_inverse(values):
        return np.interp(
            np.asarray(values, dtype=float),
            mg_display_knots,
            mg_native_knots,
        )

    for task_index, (task, title, _expected_n) in enumerate(tasks):
        axis = st.add_axes_inches(
            fig,
            [
                left_margin + task_index * (panel_width + panel_gap),
                panel_bottom,
                panel_width,
                panel_height,
            ],
        )
        axes.append(axis)
        y_positions = np.arange(len(designs), dtype=float)
        axis.set_ylim(len(designs) - 0.52, -0.52)
        if task == "parity":
            axis.set_xscale(
                "function",
                functions=(parity_zoom_forward, parity_zoom_inverse),
            )
        elif task == "mg":
            axis.set_xscale(
                "function",
                functions=(mg_zoom_forward, mg_zoom_inverse),
            )
        axis.set_xlim(*axis_limits[task])
        axis.set_xticks(axis_ticks[task])
        if task == "mg":
            axis.set_xticklabels((".03", ".04", ".05"))
        axis.set_yticks(y_positions)
        if task_index == 0:
            axis.set_yticklabels(
                [label for label, _method in designs],
                fontsize=row_label_size,
            )
        else:
            axis.set_yticklabels([])

        reference_values = np.asarray(
            list(cells[task]["CD_paper"].values()),
            dtype=float,
        )
        axis.axvline(
            float(np.mean(reference_values)),
            color=st.UNIFORM_LOCAL,
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.10,
            alpha=0.60,
            zorder=1.0,
        )
        for improved_method in task_better_than_reference[task]:
            improved_values = np.asarray(
                list(cells[task][improved_method].values()),
                dtype=float,
            )
            is_winner_guide = improved_method == task_winners[task]
            axis.axvline(
                float(np.mean(improved_values)),
                color=C[improved_method],
                linestyle=(0, (1.0, 1.8)),
                linewidth=1.10 if is_winner_guide else 0.92,
                alpha=0.78 if is_winner_guide else 0.62,
                dash_capstyle="round",
                zorder=1.1,
            )

        for design_index, (_label, method) in enumerate(designs):
            seed_values = cells[task][method]
            ordered_values = np.asarray(
                [seed_values[seed] for seed in sorted(seed_values)],
                dtype=float,
            )
            summary = t_summary(ordered_values)
            jitter_phase = np.sin(
                np.arange(len(ordered_values)) * 2.399963
                + design_index * 0.79
                + task_index * 1.31
            )
            # Keep the raw instances on one narrow horizontal track.  Their
            # x coordinates are the measured values; only a small continuous
            # y jitter is needed to reveal coincident rings.
            y_jitter = 0.065 * jitter_phase
            is_reference = method == "CD_paper"
            is_winner = method == task_winners[task]
            is_better = method in task_better_than_reference[task]
            aggregate_color = (
                st.UNIFORM_LOCAL
                if is_reference
                else C[method]
                if is_better
                else neutral_color
            )
            seed_color = (
                st.UNIFORM_LOCAL
                if is_reference
                else C[method]
                if is_better
                else neutral_color
            )
            seed_near_alpha = (
                0.14
                if is_winner
                else 0.12
                if is_reference or is_better
                else 0.085
            )
            seed_far_alpha = (
                0.025
                if is_winner
                else 0.020
                if is_reference or is_better
                else 0.012
            )
            focus_lower, focus_upper = axis_limits[task]
            focus_span = focus_upper - focus_lower
            below_count = int(np.count_nonzero(ordered_values < focus_lower))
            above_count = int(np.count_nonzero(ordered_values > focus_upper))
            axis.scatter(
                ordered_values,
                np.full(len(ordered_values), y_positions[design_index]) + y_jitter,
                s=raw_marker_area,
                marker="o",
                facecolors="none",
                edgecolors=st.distance_faded_colors(
                    seed_color,
                    ordered_values,
                    center=summary["mean"],
                    near_alpha=seed_near_alpha,
                    far_alpha=seed_far_alpha,
                ),
                linewidths=0.38,
                zorder=2.1,
            )
            axis.errorbar(
                summary["mean"],
                y_positions[design_index],
                xerr=np.asarray(
                    [
                        [summary["mean"] - summary["lo"]],
                        [summary["hi"] - summary["mean"]],
                    ]
                ),
                color=aggregate_color,
                marker=MARKER[method],
                markerfacecolor=(
                    PAPER if is_reference else aggregate_color
                ),
                markeredgecolor=aggregate_color,
                markeredgewidth=(
                    1.05
                    if is_reference or is_winner
                    else 0.92
                    if is_better
                    else 0.78
                ),
                markersize=(
                    priority_marker_size
                    if is_reference or is_winner
                    else 6.25
                    if is_better
                    else mean_marker_size
                ),
                linestyle="none",
                elinewidth=(
                    priority_linewidth
                    if is_reference or is_winner
                    else 1.20
                    if is_better
                    else mean_linewidth
                ),
                capsize=mean_capsize,
                capthick=(
                    priority_linewidth
                    if is_reference or is_winner
                    else 1.20
                    if is_better
                    else mean_linewidth
                ),
                zorder=4.0,
            )
            if summary["mean"] < focus_lower or summary["mean"] > focus_upper:
                is_below = summary["mean"] < focus_lower
                boundary_x = (
                    focus_lower + 0.025 * focus_span
                    if is_below
                    else focus_upper - 0.025 * focus_span
                )
                boundary_text = (
                    r"$\approx 0$"
                    if task in {"stm", "parity"}
                    else r"$1.020$"
                )
                axis.scatter(
                    [boundary_x],
                    [y_positions[design_index]],
                    s=mean_marker_size**2,
                    marker="<" if is_below else ">",
                    color=aggregate_color,
                    edgecolor=PAPER,
                    linewidth=0.55,
                    clip_on=True,
                    zorder=5.0,
                )
                axis.text(
                    boundary_x + (0.055 if is_below else -0.055) * focus_span,
                    y_positions[design_index],
                    boundary_text,
                    ha="left" if is_below else "right",
                    va="center",
                    fontsize=boundary_label_size,
                    color=neutral_color,
                    clip_on=True,
                    zorder=5.0,
                )
            elif above_count or below_count:
                is_below = below_count > 0
                clipped_count = below_count if is_below else above_count
                boundary_x = (
                    focus_lower + 0.025 * focus_span
                    if is_below
                    else focus_upper - 0.025 * focus_span
                )
                axis.scatter(
                    [boundary_x],
                    [y_positions[design_index]],
                    s=mean_marker_size**2,
                    marker="<" if is_below else ">",
                    color=seed_color,
                    edgecolor=PAPER,
                    linewidth=0.55,
                    clip_on=True,
                    zorder=5.0,
                )
                axis.text(
                    boundary_x + (0.055 if is_below else -0.055) * focus_span,
                    y_positions[design_index] + 0.42,
                    f"{clipped_count}/{len(ordered_values)}",
                    ha="left" if is_below else "right",
                    va="center",
                    fontsize=boundary_label_size,
                    color=seed_color,
                    clip_on=True,
                    zorder=5.0,
                )

        st.style_axis(axis, "both", minor_grid=False)
        axis.minorticks_off()
        axis.tick_params(
            axis="x",
            labelsize=tick_size,
            top=True,
            bottom=True,
        )
        axis.tick_params(
            axis="y",
            left=True,
            right=False,
            labelleft=task_index == 0,
            length=2.6,
            pad=3.0,
        )
        axis.get_xticklabels()[0].set_horizontalalignment("left")
        axis.get_xticklabels()[-1].set_horizontalalignment("right")
        axis.set_xlabel(title, fontsize=axis_label_size, labelpad=3.0)
        axis.text(
            0.0,
            1.035,
            f"({chr(97 + task_index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=panel_label_size,
            color=INK,
            clip_on=False,
        )
        if task in {"parity", "mg"}:
            # Conventional diagonal marks show every change in linear scale.
            display_knots = (
                parity_display_knots if task == "parity" else mg_display_knots
            )
            for x_fraction in display_knots[1:-1]:
                for y_fraction in (0.0, 1.0):
                    axis.plot(
                        [x_fraction - 0.012, x_fraction + 0.012],
                        [y_fraction - 0.015, y_fraction + 0.015],
                        transform=axis.transAxes,
                        color=INK,
                        linewidth=0.80,
                        clip_on=False,
                        zorder=8.0,
                    )

    axes_array = np.asarray(axes, dtype=object)
    st.audit_figure(
        fig,
        "fig_task_scores",
        axes=axes_array,
        overlap_fraction=0.08,
        font_floor=st.MIN_FONT_SIZE,
    )
    fig._qrc_audit_axes = axes_array
    save(fig, "fig_task_scores.pdf")


def fig_map_dense(rows: list[dict]):
    """One-column task-rank chart computed from the canonical score rows."""

    designs = (
        ("collective", "B3_collective", "o", False),
        ("unequal local", "A1_heterogeneous", "s", False),
        ("pair loss", "B5_pair", "*", False),
        ("reference model", "CD_paper", "D", False),
        ("gain/loss", "B2_thermal", "P", False),
        ("exchange", "B4_loss_exchange", "v", False),
        ("dephasing", "B1_dephasing", "X", False),
        ("reset FN", "FN", "d", True),
    )
    tasks = (
        ("stm", "STM", True),
        ("narma", "NARMA-10", False),
        ("parity", "parity", True),
        ("mg", "MG-150", False),
    )

    means = {
        method: np.asarray(
            [
                np.mean(list(map_cell(rows, task, method).values()))
                for task, _label, _higher_is_better in tasks
            ],
            dtype=float,
        )
        for _label, method, _marker, _external in designs
    }
    if any(
        values.shape != (len(tasks),) or not np.all(np.isfinite(values))
        for values in means.values()
    ):
        raise RuntimeError("fig_map canonical task means are incomplete")

    ranks = {method: [] for _label, method, *_rest in designs}
    seed_ranks = {
        method: [[] for _task in tasks]
        for _label, method, *_rest in designs
    }
    for task_index, (_task, _label, higher_is_better) in enumerate(tasks):
        ordered = sorted(
            means,
            key=lambda method: (
                -means[method][task_index]
                if higher_is_better
                else means[method][task_index]
            ),
        )
        for rank, method in enumerate(ordered, start=1):
            ranks[method].append(rank)

        # Seed-wise ranks require paired observations.  The reset-FN row is an
        # external, unpaired baseline, so it contributes only its aggregate
        # mean rank and is excluded from the seed-rank calculation.
        task_cells = {
            method: map_cell(rows, _task, method)
            for _design_label, method, _marker, external in designs
            if not external
        }
        shared_seeds = sorted(
            set.intersection(*(set(values) for values in task_cells.values()))
        )
        if len(shared_seeds) < 16:
            raise RuntimeError(
                f"fig_map lacks paired seed ranks for {_task}: "
                f"{len(shared_seeds)}"
            )
        external_mean_key = "__reset_fn_mean__"
        external_mean = float(means["FN"][task_index])
        for seed in shared_seeds:
            def rank_value(method: str) -> float:
                return (
                    external_mean
                    if method == external_mean_key
                    else float(task_cells[method][seed])
                )

            seed_order = sorted(
                (*task_cells, external_mean_key),
                key=lambda method: (
                    -rank_value(method)
                    if higher_is_better
                    else rank_value(method)
                ),
            )
            rank_by_method = {
                method: rank
                for rank, method in enumerate(seed_order, start=1)
            }
            for method in task_cells:
                seed_ranks[method][task_index].append(rank_by_method[method])

    # Unlike the focused absolute-score Figure 2, this broader rank view uses
    # the manuscript-wide dissipator palette for every continuously driven
    # design.  Reset FN remains a neutral, dashed external baseline.
    display_colors = {
        method: (st.NEUTRAL_DESIGN if method == "FN" else C[method])
        for method in ranks
    }

    # Match the supplied rank-chart topology at the manuscript's exact natural
    # column width.  The normalized bounds make the central plot box square.
    left, right = 0.350, 0.980
    top, bottom = 0.945, 0.150
    figure_width = st.QUANTUM_COLUMN_WIDTH
    axes_width = (right - left) * figure_width
    figure_height = axes_width / (top - bottom)
    rank_text_size = 9.00
    rank_tick_size = 8.75
    rank_note_size = 8.60
    grid_color = "#EAEAEA"
    column_guide_color = "#E4E4E4"

    with matplotlib.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "TeX Gyre Heros",
                "FreeSans",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
        }
    ):
        fig = st.composite_figure("column", figure_height)
        axis = fig.add_axes([left, bottom, right - left, top - bottom])
        axes = np.asarray([axis], dtype=object)
        x_positions = np.arange(len(tasks))
        n_designs = len(designs)
        axis.set_xlim(-0.62, len(tasks) - 1 + 0.62)
        axis.set_ylim(n_designs + 0.55, 0.45)

        for rank in range(1, n_designs + 1):
            axis.axhline(rank, color=grid_color, linewidth=0.45, zorder=0.4)
        for x_position in x_positions:
            axis.axvline(
                x_position,
                color=column_guide_color,
                linewidth=0.50,
                zorder=0.4,
            )

        for side in ("top", "right", "bottom"):
            axis.spines[side].set_visible(False)
        axis.spines["left"].set_bounds(1, n_designs)
        axis.spines["left"].set_linewidth(0.50)

        for design_index, (label, method, marker, external) in enumerate(designs):
            color = display_colors[method]
            method_ranks = ranks[method]
            if not external:
                for task_index, x_position in enumerate(x_positions):
                    raw_ranks = np.asarray(
                        seed_ranks[method][task_index],
                        dtype=float,
                    )
                    seed_jitter = np.linspace(-0.095, 0.095, len(raw_ranks))
                    rank_jitter = 0.070 * np.sin(
                        np.arange(len(raw_ranks)) * 2.399963
                        + design_index * 0.73
                        + task_index * 1.37
                    )
                    axis.scatter(
                        np.full(len(raw_ranks), x_position) + seed_jitter,
                        raw_ranks + rank_jitter,
                        s=9.5,
                        marker="o",
                        color=st.distance_faded_colors(
                            color,
                            raw_ranks,
                            center=float(np.mean(raw_ranks)),
                            near_alpha=0.32,
                            far_alpha=0.060,
                        ),
                        linewidths=0,
                        zorder=1.8,
                    )
            axis.plot(
                x_positions,
                method_ranks,
                color=color,
                linewidth=1.00 if external else DATA_LINEWIDTH,
                linestyle=(0, (2.4, 1.6)) if external else "-",
                alpha=0.60 if external else 0.95,
                zorder=3,
                solid_capstyle="round",
            )
            axis.plot(
                x_positions,
                method_ranks,
                marker=marker,
                markersize=(
                    st.MARKER_SIZE * 1.45 if marker == "*" else st.MARKER_SIZE
                ),
                markerfacecolor=color,
                markeredgecolor=PAPER,
                markeredgewidth=st.MARKER_EDGEWIDTH,
                linestyle="none",
                alpha=0.60 if external else 0.95,
                zorder=4,
            )
            axis.text(
                -0.70,
                method_ranks[0],
                label,
                ha="right",
                va="center",
                fontsize=rank_text_size,
                color=color,
                fontweight="normal" if external else "bold",
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [label for _task, label, _higher_is_better in tasks],
            fontsize=rank_tick_size,
        )
        axis.minorticks_off()
        axis.tick_params(
            axis="x",
            which="both",
            top=False,
            bottom=False,
            length=0,
            pad=3.0,
        )
        axis.set_yticks(range(1, n_designs + 1))
        axis.set_yticklabels([])
        axis.tick_params(
            axis="y",
            which="both",
            left=False,
            right=False,
            length=0,
        )
        axis.text(
            -0.62,
            0.28,
            "rank, best at top",
            fontsize=rank_note_size,
            color="#5A5A5A",
            style="italic",
            ha="left",
            va="center",
        )
        axis.set_box_aspect(1.0)

        fig.canvas.draw()
        plot_box = axis.get_window_extent().transformed(
            fig.dpi_scale_trans.inverted()
        )
        if (
            abs(plot_box.width - plot_box.height) > 0.005
            or abs(plot_box.width - axes_width) > 0.005
        ):
            raise RuntimeError(
                "fig_map rank panel must remain square: "
                f"{plot_box.width:.3f} x {plot_box.height:.3f} in"
            )
        st.audit_figure(
            fig,
            "fig_map",
            axes=axes,
            overlap_fraction=0.10,
            font_floor=rank_note_size,
        )
        fig._qrc_audit_axes = axes
        save(fig, "fig_map.pdf")


def fig_profiles_and_sampling_dense(rows: list[dict]):
    """Separate one-column profile and finite-measurement figures."""
    engineering_text_size = 9.20
    engineering_tick_size = 9.25
    engineering_axis_size = 10.15
    engineering_panel_size = 10.70
    plt.rcParams.update(
        {
            "font.size": 9.70,
            "axes.labelsize": engineering_axis_size,
            "axes.titlesize": 10.35,
            "xtick.labelsize": engineering_tick_size,
            "ytick.labelsize": engineering_tick_size,
            "legend.fontsize": engineering_text_size,
        }
    )
    review = load_review()
    uniform = cell(rows, "F_adaptive", "stm", "A0_uniform", N=5)
    unequal = cell(rows, "F_adaptive", "stm", "A1_random", N=5)
    optimized = {
        method: {
            row["seed"]: row["value"]
            for row in review
            if row.get("block") == "R_optfix" and row.get("method") == method
        }
        for method in ("A3_eq", "A4_eq")
    }
    profile_cells = [
        ("unequal", unequal, "s"),
        ("learned", optimized["A3_eq"], "o"),
        ("adaptive", optimized["A4_eq"], "^"),
    ]
    profile_summaries = [
        paired_summary(values, uniform)
        for _, values, _ in profile_cells
    ]
    if any(summary["n"] != 32 for summary in profile_summaries):
        raise RuntimeError("profile comparison is not a complete paired grid")

    joint_rows = [
        row
        for row in rows
        if row.get("block") == "G_joint" and row.get("task") == "stm"
    ]
    joint_cells = {
        method: {
            int(row["seed"]): float(row["value"])
            for row in joint_rows
            if row.get("method") == method
        }
        for method in (
            "G_local_uniform",
            "G_local_learned",
            "G_coll_uniform",
            "G_coll_learned",
        )
    }
    shared = sorted(set.intersection(*(set(values) for values in joint_cells.values())))
    local_uniform = np.asarray(
        [joint_cells["G_local_uniform"][seed] for seed in shared],
        dtype=float,
    )
    local_learned = np.asarray(
        [joint_cells["G_local_learned"][seed] for seed in shared],
        dtype=float,
    )
    collective_uniform = np.asarray(
        [joint_cells["G_coll_uniform"][seed] for seed in shared],
        dtype=float,
    )
    collective_learned = np.asarray(
        [joint_cells["G_coll_learned"][seed] for seed in shared],
        dtype=float,
    )
    interaction_values = (
        local_learned
        - local_uniform
        - (collective_learned - collective_uniform)
    )
    if len(interaction_values) != 24:
        raise RuntimeError("profile interaction is not a complete 24-pair grid")
    interaction = t_summary(interaction_values)
    interaction_critical = stats.t.ppf(1 - 0.05 / (2 * 11), 23)
    interaction_se = float(np.std(interaction_values, ddof=1) / np.sqrt(24))
    interaction["lo"] = interaction["mean"] - interaction_critical * interaction_se
    interaction["hi"] = interaction["mean"] + interaction_critical * interaction_se

    if not os.path.isfile(REVISION_MEASUREMENT):
        raise RuntimeError("finite-measurement aggregate is missing")
    with open(REVISION_MEASUREMENT, encoding="utf-8") as handle:
        measurement = json.load(handle)
    if measurement.get("validation", {}).get("status") != "complete":
        raise RuntimeError("finite-measurement aggregate is incomplete")

    methods = [
        "B3_collective",
        "A1_heterogeneous",
        "B5_pair",
        "B2_thermal",
        "B4_loss_exchange",
    ]
    finite_budgets = sorted(
        {
            int(row["budget"])
            for row in measurement["paired_vs_local"]
            if row["budget"] != "exact"
        }
    )
    models = {
        row["measurement_model"]
        for row in measurement["paired_vs_local"]
        if row["budget"] != "exact"
    }
    channels = {
        row["channel"]
        for row in measurement["paired_vs_local"]
        if row["budget"] != "exact"
    }
    simultaneous_family = (
        len(models)
        * len(finite_budgets)
        * len(channels)
        * (len(channels) - 1)
        // 2
    )
    if simultaneous_family != 210:
        raise RuntimeError("unexpected finite-measurement contrast family")
    simultaneous_critical = stats.t.ppf(
        1 - 0.05 / (2 * simultaneous_family),
        19,
    )
    exact_position = finite_budgets[-1] * 8
    lookup = {
        (row["measurement_model"], row["budget"], row["channel"]): row
        for row in measurement["paired_vs_local"]
    }
    measurement_scores: dict[tuple[str, str, str, int], float] = {}
    for row in measurement["raw_rows"]:
        budget_key = (
            "exact"
            if row["is_exact"]
            else str(int(row["total_shots_per_time_step"]))
        )
        key = (
            row["measurement_model"],
            budget_key,
            row["channel"],
            int(row["seed"]),
        )
        if key in measurement_scores:
            raise RuntimeError(f"duplicate finite-measurement row: {key}")
        measurement_scores[key] = float(row["test_mc"])

    def measurement_differences(
        model: str,
        budget: str,
        method: str,
    ) -> np.ndarray:
        candidate = {
            seed: value
            for (row_model, row_budget, channel, seed), value
            in measurement_scores.items()
            if row_model == model
            and row_budget == budget
            and channel == method
        }
        reference = {
            seed: value
            for (row_model, row_budget, channel, seed), value
            in measurement_scores.items()
            if row_model == model
            and row_budget == budget
            and channel == "CD_paper"
        }
        seeds = sorted(set(candidate) & set(reference))
        if len(seeds) != 20:
            raise RuntimeError(
                f"incomplete finite-measurement seed block: "
                f"{model}/{budget}/{method}"
            )
        return np.asarray(
            [candidate[seed] - reference[seed] for seed in seeds],
            dtype=float,
        )

    # Each output occupies one manuscript column and contains one row of two
    # physically square panels.  The sampling panels use nearly all available
    # column width and add a dedicated shared-key band above them so no data
    # trajectory is hidden by the five-entry legend.
    profile_figure_height = 2.40
    profile_panel_side = 1.24
    profile_left_x = 0.40
    profile_right_x = 1.91
    profile_row_y = 0.49
    sampling_figure_height = 2.50
    sampling_panel_side = 1.28
    sampling_left_x = 0.40
    sampling_right_x = 1.84
    sampling_row_y = 0.49
    profile_figure = st.composite_figure("column", profile_figure_height)
    profile_width = float(profile_figure.get_figwidth())
    ax_profile = st.add_axes_inches(
        profile_figure,
        [
            profile_left_x,
            profile_row_y,
            profile_panel_side,
            profile_panel_side,
        ],
    )
    ax_interaction = st.add_axes_inches(
        profile_figure,
        [
            profile_right_x,
            profile_row_y,
            profile_panel_side,
            profile_panel_side,
        ],
    )
    profile_axes = np.asarray(
        [ax_profile, ax_interaction],
        dtype=object,
    )
    profile_legend_axis = st.add_axes_inches(
        profile_figure,
        [
            profile_left_x,
            1.92,
            profile_right_x + profile_panel_side - profile_left_x,
            0.39,
        ],
    )
    profile_legend_axis.set_axis_off()

    sampling_figure = st.composite_figure("column", sampling_figure_height)
    sampling_width = float(sampling_figure.get_figwidth())
    ax_independent = st.add_axes_inches(
        sampling_figure,
        [
            sampling_left_x,
            sampling_row_y,
            sampling_panel_side,
            sampling_panel_side,
        ],
    )
    ax_grouped = st.add_axes_inches(
        sampling_figure,
        [
            sampling_right_x,
            sampling_row_y,
            sampling_panel_side,
            sampling_panel_side,
        ],
        sharey=ax_independent,
    )
    sampling_axes = np.asarray(
        [ax_independent, ax_grouped],
        dtype=object,
    )
    sampling_legend_axis = st.add_axes_inches(
        sampling_figure,
        [
            sampling_left_x,
            2.02,
            sampling_right_x + sampling_panel_side - sampling_left_x,
            0.39,
        ],
    )
    sampling_legend_axis.set_axis_off()

    # (a) Every paired seed is visible.  The filled marker and bar give the
    # paired mean and its pointwise 95% t interval.
    random = np.random.default_rng(20260804)
    # Slightly wider categorical spacing keeps the final-size tick labels
    # distinct without reducing their type size.
    category_positions = np.asarray([0.0, 1.15, 2.30], dtype=float)
    for index, ((label, _, marker), summary) in enumerate(
        zip(
            profile_cells,
            profile_summaries,
            strict=True,
        )
    ):
        differences = np.asarray(summary["raw"], dtype=float)
        category_x = category_positions[index]
        jitter = random.uniform(-0.105, 0.105, size=len(differences))
        ax_profile.scatter(
            np.full(len(differences), category_x) + jitter,
            differences,
            s=10.5,
            marker=marker,
            facecolor="none",
            edgecolor=st.distance_faded_colors(
                st.LOCAL_CONTRAST,
                differences,
                center=float(summary["mean"]),
                near_alpha=0.24,
                far_alpha=0.035,
            ),
            linewidth=0.55,
            zorder=2,
        )
        ax_profile.errorbar(
            category_x,
            summary["mean"],
            yerr=np.asarray(
                [
                    [summary["mean"] - summary["lo"]],
                    [summary["hi"] - summary["mean"]],
                ]
            ),
            color=st.LOCAL_CONTRAST,
            marker=marker,
            markerfacecolor=st.LOCAL_CONTRAST,
            markeredgecolor=st.LOCAL_CONTRAST,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            zorder=4,
        )
    ax_profile.axhline(
        0,
        color=INK,
        linewidth=st.REFERENCE_LINEWIDTH,
        zorder=1,
    )
    ax_profile.set_xlim(-0.40, 2.65)
    ax_profile.set_ylim(-0.78, 3.10)
    ax_profile.set_xticks(category_positions)
    ax_profile.set_xticklabels(
        [label for label, _, _ in profile_cells]
    )
    ax_profile.set_yticks([0, 1, 2, 3])
    ax_profile.set_ylabel(r"$\Delta$ STM vs uniform")
    st.style_axis(ax_profile, "both", minor_grid=False)

    # (b) Absolute paired trajectories expose both starting levels and gains.
    # Thin lines are the 24 seeds; thick lines and bars are means and pointwise
    # 95% t intervals.  The annotation reports the declared eleven-contrast
    # Bonferroni interval for the difference of the two gains.
    x_values = np.asarray([0.0, 1.0])
    for start_values, end_values, color in (
        (local_uniform, local_learned, st.LOCAL_CONTRAST),
        (collective_uniform, collective_learned, st.COLLECTIVE),
    ):
        start_mean = float(np.mean(start_values))
        end_mean = float(np.mean(end_values))
        start_scale = max(float(np.std(start_values, ddof=1)), 1e-12)
        end_scale = max(float(np.std(end_values, ddof=1)), 1e-12)
        joint_distance = np.hypot(
            (start_values - start_mean) / start_scale,
            (end_values - end_mean) / end_scale,
        )
        line_colors = st.distance_faded_colors(
            color,
            joint_distance,
            center=0.0,
            near_alpha=0.15,
            far_alpha=0.020,
        )
        seed_offsets = np.linspace(-0.015, 0.015, len(start_values))
        for start, end, offset, line_color in zip(
            start_values,
            end_values,
            seed_offsets,
            line_colors,
            strict=True,
        ):
            ax_interaction.plot(
                x_values + offset,
                [start, end],
                color=line_color,
                linewidth=0.62,
                zorder=1,
            )
        ax_interaction.scatter(
            np.full(len(start_values), x_values[0]) + seed_offsets,
            start_values,
            s=6.0,
            color=st.distance_faded_colors(
                color,
                start_values,
                center=start_mean,
                near_alpha=0.20,
                far_alpha=0.025,
            ),
            linewidths=0,
            zorder=2,
        )
        ax_interaction.scatter(
            np.full(len(end_values), x_values[1]) + seed_offsets,
            end_values,
            s=6.0,
            color=st.distance_faded_colors(
                color,
                end_values,
                center=end_mean,
                near_alpha=0.20,
                far_alpha=0.025,
            ),
            linewidths=0,
            zorder=2,
        )
    for values, color, marker, label in (
        (
            np.vstack([local_uniform, local_learned]),
            st.LOCAL_CONTRAST,
            "s",
            "local",
        ),
        (
            np.vstack([collective_uniform, collective_learned]),
            st.COLLECTIVE,
            "o",
            "collective",
        ),
    ):
        means = values.mean(axis=1)
        half_widths = np.asarray(
            [
                stats.t.ppf(0.975, values.shape[1] - 1)
                * row.std(ddof=1)
                / np.sqrt(values.shape[1])
                for row in values
            ],
            dtype=float,
        )
        ax_interaction.errorbar(
            x_values,
            means,
            yerr=half_widths,
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            linewidth=DATA_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            label=label,
            zorder=4,
        )
    ax_interaction.set_xlim(-0.23, 1.23)
    ax_interaction.set_ylim(4.05, 10.25)
    ax_interaction.set_xticks(x_values)
    ax_interaction.set_xticklabels(["uniform", "learned"])
    ax_interaction.set_yticks([4, 6, 8, 10])
    interaction_handles, interaction_labels = (
        ax_interaction.get_legend_handles_labels()
    )
    interaction_legend = st.legend(
        profile_legend_axis,
        lw=st.LEGEND_FRAMEWIDTH,
        handles=interaction_handles,
        labels=interaction_labels,
        loc="center",
        ncol=2,
        frameon=True,
        handlelength=0.85,
        handletextpad=0.25,
        columnspacing=0.45,
        borderpad=0.18,
        fontsize=engineering_text_size,
        framealpha=1.0,
    )
    interaction_legend.set_zorder(6)
    st.style_axis(ax_interaction, "both", minor_grid=False)

    # (c,d) Equal nominal preparation counts.  Their shared key occupies the
    # dedicated band above both panels rather than covering either data field.
    direct_labels = {
        "B3_collective": "collective",
        "A1_heterogeneous": "unequal",
        "B5_pair": "pair",
        "B2_thermal": "local gain/loss",
        "B4_loss_exchange": "exchange",
    }
    for axis, model in (
        (ax_independent, "independent"),
        (ax_grouped, "grouped"),
    ):
        method_offsets = np.linspace(-0.055, 0.055, len(methods))
        for method_index, method in enumerate(methods):
            finite = [
                lookup[(model, str(budget), method)]
                for budget in finite_budgets
            ]
            means = np.asarray([row["mean"] for row in finite], dtype=float)
            half_width = simultaneous_critical * np.asarray(
                [row["se"] for row in finite],
                dtype=float,
            )
            seed_jitter = np.linspace(-0.012, 0.012, 20)
            for budget, row in zip(finite_budgets, finite, strict=True):
                raw_values = measurement_differences(
                    model,
                    str(budget),
                    method,
                )
                if not np.isclose(
                    np.mean(raw_values),
                    row["mean"],
                    rtol=0,
                    atol=5e-12,
                ):
                    raise RuntimeError(
                        "finite-measurement seed mean mismatch: "
                        f"{model}/{budget}/{method}"
                    )
                axis.scatter(
                    budget
                    * np.exp(method_offsets[method_index] + seed_jitter),
                    raw_values,
                    s=4.5,
                    marker="o",
                    color=st.distance_faded_colors(
                        C[method],
                        raw_values,
                        center=float(row["mean"]),
                        near_alpha=0.16,
                        far_alpha=0.020,
                    ),
                    linewidths=0,
                    zorder=1.5,
                )
            axis.errorbar(
                finite_budgets,
                means,
                yerr=half_width,
                color=C[method],
                marker=MARKER[method],
                linestyle=LINESTYLE[method],
                markerfacecolor=C[method],
                markeredgecolor=C[method],
                markeredgewidth=MARKER_EDGEWIDTH,
                markersize=MARKER_SIZE,
                linewidth=DATA_LINEWIDTH,
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                elinewidth=ERROR_LINEWIDTH,
                label=direct_labels[method],
                zorder=2,
            )
            exact = lookup[(model, "exact", method)]
            exact_values = measurement_differences(
                model,
                "exact",
                method,
            )
            if not np.isclose(
                np.mean(exact_values),
                exact["mean"],
                rtol=0,
                atol=5e-12,
            ):
                raise RuntimeError(
                    "exact-measurement seed mean mismatch: "
                    f"{model}/{method}"
                )
            axis.scatter(
                exact_position
                * np.exp(method_offsets[method_index] + seed_jitter),
                exact_values,
                s=4.5,
                marker="o",
                color=st.distance_faded_colors(
                    C[method],
                    exact_values,
                    center=float(exact["mean"]),
                    near_alpha=0.16,
                    far_alpha=0.020,
                ),
                linewidths=0,
                zorder=1.5,
            )
            axis.plot(
                [finite_budgets[-1], exact_position],
                [means[-1], exact["mean"]],
                color=C[method],
                linestyle=":",
                linewidth=DATA_LINEWIDTH,
                alpha=0.82,
            )
            axis.errorbar(
                exact_position,
                exact["mean"],
                yerr=np.asarray(
                    [
                        [exact["mean"] - exact["ci95_low"]],
                        [exact["ci95_high"] - exact["mean"]],
                    ]
                ),
                color=C[method],
                marker=MARKER[method],
                markerfacecolor="white",
                markeredgecolor=C[method],
                markeredgewidth=MARKER_EDGEWIDTH,
                markersize=MARKER_SIZE,
                linestyle="none",
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                elinewidth=ERROR_LINEWIDTH,
                zorder=3,
            )
        axis.axhline(
            0,
            color=INK,
            linewidth=st.REFERENCE_LINEWIDTH,
        )
        axis.set_xscale("log")
        axis.set_xlim(2_000, exact_position * 1.70)
        axis.set_ylim(-3.25, 4.35)
        axis.set_yticks([-2, 0, 2, 4])
        axis.set_xticks(
            [
                finite_budgets[0],
                finite_budgets[4],
                exact_position,
            ]
        )
        axis.set_xticklabels(["2.9k", "0.74m", "exact"])
        # Keep the endpoint labels inside their own panels so the deliberately
        # wider horizontal gutter remains visually empty.
        axis.get_xticklabels()[0].set_ha("left")
        axis.get_xticklabels()[-1].set_ha("right")
        axis.minorticks_off()
        axis.set_xlabel("preparations/step")
        st.style_axis(axis, "both", minor_grid=False)

    ax_independent.set_ylabel("STM difference")
    ax_grouped.tick_params(labelleft=False)
    handles, legend_labels = ax_independent.get_legend_handles_labels()
    legend_labels = [
        "gain/loss" if label == "local gain/loss" else label
        for label in legend_labels
    ]
    # Matplotlib fills multi-column legends column-major.  This order gives a
    # balanced first row (collective, unequal, pair) and second row
    # (gain/loss, exchange) while retaining the manuscript's semantic order.
    legend_order = [0, 3, 1, 4, 2]
    handles = [handles[index] for index in legend_order]
    legend_labels = [legend_labels[index] for index in legend_order]
    measurement_legend = st.legend(
        sampling_legend_axis,
        lw=st.LEGEND_FRAMEWIDTH,
        handles=handles,
        labels=legend_labels,
        loc="center",
        ncol=3,
        borderaxespad=0.0,
        borderpad=0.18,
        labelspacing=0.15,
        columnspacing=0.42,
        handlelength=0.72,
        handletextpad=0.18,
        fontsize=engineering_text_size,
        framealpha=1.0,
    )
    measurement_legend.set_zorder(6)

    # Both independent outputs use their own (a,b) panel sequence.
    for (
        figure,
        figure_width,
        current_height,
        current_left,
        current_right,
        current_row,
        current_side,
    ) in (
        (
            profile_figure,
            profile_width,
            profile_figure_height,
            profile_left_x,
            profile_right_x,
            profile_row_y,
            profile_panel_side,
        ),
        (
            sampling_figure,
            sampling_width,
            sampling_figure_height,
            sampling_left_x,
            sampling_right_x,
            sampling_row_y,
            sampling_panel_side,
        ),
    ):
        label_y = (current_row + current_side + 0.10) / current_height
        for label, x_position in (("a", current_left), ("b", current_right)):
            figure.text(
                x_position / figure_width,
                label_y,
                f"({label})",
                ha="left",
                va="bottom",
                fontsize=engineering_panel_size,
                color=INK,
            )
    for axis in (*profile_axes, *sampling_axes):
        axis.tick_params(labelsize=engineering_tick_size)
        axis.xaxis.label.set_size(engineering_axis_size)
        axis.yaxis.label.set_size(engineering_axis_size)

    profile_figure.canvas.draw()
    profile_renderer = profile_figure.canvas.get_renderer()
    profile_boxes = [
        axis.get_window_extent().transformed(
            profile_figure.dpi_scale_trans.inverted()
        )
        for axis in profile_axes
    ]
    legend_box = interaction_legend.get_window_extent(profile_renderer)
    profile_panel_top = max(
        axis.get_window_extent(profile_renderer).y1
        for axis in profile_axes
    )
    if (
        not np.isclose(profile_width, st.QUANTUM_COLUMN_WIDTH)
        or any(
            abs(box.width - profile_panel_side) > 0.005
            for box in profile_boxes
        )
        or any(
            abs(box.height - profile_panel_side) > 0.005
            for box in profile_boxes
        )
        or any(abs(box.width - box.height) > 0.005 for box in profile_boxes)
        or legend_box.y0 <= profile_panel_top + 4.0
    ):
        raise RuntimeError(
            "fig_profiles violates the square-panel geometry contract: "
            + ", ".join(
                f"{box.width:.3f} x {box.height:.3f} in"
                for box in profile_boxes
            )
            + (
                f"; legend clearance={legend_box.y0 - profile_panel_top:.1f} px"
            )
        )
    st.audit_figure(
        profile_figure,
        "fig_profiles",
        axes=profile_axes,
        overlap_fraction=0.08,
        font_floor=engineering_text_size,
    )
    save(profile_figure, "fig_profiles.pdf")

    sampling_figure.canvas.draw()
    sampling_renderer = sampling_figure.canvas.get_renderer()
    sampling_boxes = [
        axis.get_window_extent().transformed(
            sampling_figure.dpi_scale_trans.inverted()
        )
        for axis in sampling_axes
    ]
    measurement_legend_box = measurement_legend.get_window_extent(
        sampling_renderer
    )
    sampling_panel_top = max(
        axis.get_window_extent(sampling_renderer).y1
        for axis in sampling_axes
    )
    if (
        not np.isclose(sampling_width, st.QUANTUM_COLUMN_WIDTH)
        or any(
            abs(box.width - sampling_panel_side) > 0.005
            for box in sampling_boxes
        )
        or any(
            abs(box.height - sampling_panel_side) > 0.005
            for box in sampling_boxes
        )
        or any(abs(box.width - box.height) > 0.005 for box in sampling_boxes)
        or measurement_legend_box.y0 <= sampling_panel_top + 4.0
    ):
        raise RuntimeError(
            "fig_sampling violates the square-panel geometry contract: "
            + ", ".join(
                f"{box.width:.3f} x {box.height:.3f} in"
                for box in sampling_boxes
            )
            + (
                "; legend clearance="
                f"{measurement_legend_box.y0 - sampling_panel_top:.1f} px"
            )
        )
    st.audit_figure(
        sampling_figure,
        "fig_sampling",
        axes=sampling_axes,
        overlap_fraction=0.08,
        font_floor=engineering_text_size,
    )
    save(sampling_figure, "fig_sampling.pdf")


REQUIRED_BLOCKS = {
    "A_table": 1200,
    "B_scale": 1300,
    "E_diag": 380,
    "F_adaptive": 120,
    "G_joint": 100,
}


def main():
    rows = load()
    counts = {}
    for row in rows:
        block = row.get("block")
        counts[block] = counts.get(block, 0) + 1
    missing = [
        f"  {block}: found {counts.get(block, 0)}, need >= {minimum}"
        for block, minimum in REQUIRED_BLOCKS.items()
        if counts.get(block, 0) < minimum
    ]
    if missing:
        raise SystemExit(
            f"ERROR: {FINAL_DIR} is incomplete:\n"
            + "\n".join(missing)
            + "\nRe-extract results/final_protocol_results.tar.gz before plotting."
        )
    print(f"loaded {len(rows)} canonical result files")
    fig_designspace()
    fig_task_scores_dense(rows)
    fig_map_dense(rows)
    fig_profiles_and_sampling_dense(rows)
    print(
        "wrote 5 primary manuscript figures to",
        OUTDIR,
        "(run make_forgetting_modes_figure.py for the robustness figure)",
    )


if __name__ == "__main__":
    main()

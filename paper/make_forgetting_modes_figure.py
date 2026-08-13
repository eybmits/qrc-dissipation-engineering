"""Build the Experiment 1 case-study and scalar-control figures.

Section 3 uses one full-width 1x4 figure for the targeted controls,
separate STM and NARMA-10 finite-size recurrences, and the lag-resolved STM
profile.  The Section 4 figure retains the detailed scalar-control and
gap-matched comparison.  All outputs are rebuilt from sealed evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
from scipy.stats import t as student_t

import l3_style as st


HERE = Path(__file__).resolve().parent
AGGREGATE = (
    HERE / "evidence" / "switched_input_memory_control_v2" / "aggregate.json"
)
CONVERGENCE_EXTENSION = (
    HERE
    / "evidence"
    / "collective_N5_convergence_extension_v1"
    / "aggregate.json"
)
LOCAL_PAIR_CONVERGENCE_EXTENSION = (
    HERE
    / "evidence"
    / "local_pair_convergence_extension_v1"
    / "aggregate.json"
)
ACTIVITY = HERE / "data" / "activity_matched_confirmation.json"
ROBUSTNESS = HERE / "data" / "experiment1_robustness_snapshot.json"
FINITE_SIZE = HERE / "data" / "experiment1_finite_size_snapshot.json"
FINITE_SIZE_SEEDS = (
    HERE / "data" / "experiment1_finite_size_seed_values.json"
)
SCALAR_CONTROL_SEEDS = (
    HERE / "data" / "experiment1_scalar_control_seed_values.json"
)
PRINCIPAL_SUMMARY = HERE / "data" / "experiment1_principal_summary.json"
RESET_ARCHITECTURE = HERE / "data" / "reset_architecture_snapshot.json"
GAP_CALIBRATION = (
    HERE / "evidence" / "canonical_gap_control" / "calibration.csv"
)
GAP_LAG_CAPACITIES = (
    HERE / "evidence" / "canonical_gap_control" / "lag_capacities.csv"
)
GAP_LAG_CAPACITIES_SHA256 = (
    "f1bd3bafac4c7556f8de2a56ad97b8b261b40c45dd612125afde60cc09738676"
)
COLLECTIVE_CASE_OUTPUT = HERE / "figures" / "fig_collective_case.pdf"
SCALAR_CONTROLS_OUTPUT = HERE / "figures" / "fig_scalar_controls.pdf"

st.use(times=True)

DATA_LINEWIDTH = st.DATA_LINEWIDTH
MARKER_SIZE = st.MARKER_SIZE
MARKER_EDGEWIDTH = st.MARKER_EDGEWIDTH
ERROR_LINEWIDTH = st.ERROR_LINEWIDTH
ERROR_CAPSIZE = st.ERROR_CAPSIZE
FIG_TEXT_SIZE = 9.20
FIG_TICK_SIZE = 9.25
FIG_AXIS_SIZE = 10.15
FIG_PANEL_SIZE = 10.70

# These composites are printed at their natural manuscript widths.  Keep panel
# letters close to body size, axis labels only slightly smaller, and every
# remaining annotation at or above the shared 8.6-pt figure floor.
plt.rcParams.update(
    {
        "font.size": 9.70,
        "axes.labelsize": FIG_AXIS_SIZE,
        "axes.titlesize": 10.35,
        "xtick.labelsize": FIG_TICK_SIZE,
        "ytick.labelsize": FIG_TICK_SIZE,
        "legend.fontsize": FIG_TEXT_SIZE,
    }
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simultaneous_summary(values: list[float], family_size: int = 1) -> dict:
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / np.sqrt(n))
    critical = float(student_t.ppf(1 - 0.05 / (2 * family_size), n - 1))
    return {
        "n": n,
        "mean": mean,
        "ci95_low": mean - critical * standard_error,
        "ci95_high": mean + critical * standard_error,
        "wins": int(np.count_nonzero(array > 0)),
    }


def assert_close(observed: dict, expected: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if not np.isclose(
            float(observed[field]),
            float(expected[field]),
            rtol=0,
            atol=5e-10,
        ):
            raise RuntimeError(
                f"robustness snapshot mismatch for {observed['key']}:{field}"
            )


def load_and_validate() -> tuple[
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
    dict,
    list[dict],
    list[dict],
    list[dict],
]:
    aggregate = json.loads(AGGREGATE.read_text(encoding="utf-8"))
    extension = json.loads(CONVERGENCE_EXTENSION.read_text(encoding="utf-8"))
    local_pair_extension = json.loads(
        LOCAL_PAIR_CONVERGENCE_EXTENSION.read_text(encoding="utf-8")
    )
    activity = json.loads(ACTIVITY.read_text(encoding="utf-8"))
    snapshot = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    finite_size = json.loads(FINITE_SIZE.read_text(encoding="utf-8"))
    finite_size_seeds = json.loads(
        FINITE_SIZE_SEEDS.read_text(encoding="utf-8")
    )
    scalar_control_seeds = json.loads(
        SCALAR_CONTROL_SEEDS.read_text(encoding="utf-8")
    )
    principal = json.loads(PRINCIPAL_SUMMARY.read_text(encoding="utf-8"))
    reset_architecture = json.loads(
        RESET_ARCHITECTURE.read_text(encoding="utf-8")
    )
    if (
        aggregate.get("status") != "complete"
        or extension.get("status") != "complete"
        or local_pair_extension.get("status") != "complete"
        or activity.get("status") != "complete"
        or snapshot.get("status") != "complete"
        or snapshot.get("metric") != "collective minus local STM capacity"
        or finite_size.get("status") != "complete"
    ):
        raise RuntimeError("Experiment 1 robustness inputs are incomplete")

    finite_summaries = finite_size.get("summaries", {}).get("primary_fixed", {})
    expected_sizes = {str(size) for size in range(4, 9)}
    if (
        finite_size.get("complete_sizes") != [4, 5, 6, 7, 8]
        or finite_size.get("lineages_per_size") != 24
        or finite_size.get("checkpoint_count") != 960
        or finite_size.get("confirmatory_family_size") != 62
        or set(finite_summaries) != expected_sizes
    ):
        raise RuntimeError("finite-size snapshot violates its bounded contract")
    for size in expected_sizes:
        for task in ("stm", "narma10"):
            task_rows = finite_summaries[size].get(task, {})
            for method in ("CD_paper", "B3_collective"):
                summary = task_rows.get(method, {})
                values = np.asarray(
                    [summary.get("mean"), summary.get("se")],
                    dtype=float,
                )
                if (
                    values.shape != (2,)
                    or not np.all(np.isfinite(values))
                    or values[1] < 0
                ):
                    raise RuntimeError(
                        "finite-size snapshot contains an invalid "
                        f"{size}/{task}/{method} summary"
                    )

    seed_sizes = finite_size_seeds.get("sizes", {})
    if (
        finite_size_seeds.get("artifact_type")
        != "figure3_finite_size_seed_values"
        or finite_size_seeds.get("status") != "complete"
        or finite_size_seeds.get("readout") != "primary_fixed"
        or finite_size_seeds.get("checkpoint_set_sha256")
        != finite_size.get("checkpoint_set_sha256")
        or finite_size_seeds.get("methods")
        != ["CD_paper", "B3_collective"]
        or finite_size_seeds.get("tasks") != ["stm", "narma10"]
        or set(seed_sizes) != expected_sizes
    ):
        raise RuntimeError("finite-size seed snapshot violates its contract")
    for size in expected_sizes:
        seed_block = seed_sizes[size]
        seeds = np.asarray(seed_block.get("seeds", []), dtype=int)
        values_by_method = seed_block.get("values", {})
        if (
            seeds.shape != (24,)
            or len(np.unique(seeds)) != 24
            or not np.all(seeds[:-1] < seeds[1:])
            or set(values_by_method) != {"CD_paper", "B3_collective"}
        ):
            raise RuntimeError(f"invalid finite-size seed block at N={size}")
        for method, task_values in values_by_method.items():
            if set(task_values) != {"stm", "narma10"}:
                raise RuntimeError(
                    f"invalid finite-size seed tasks at N={size}/{method}"
                )
            for task, raw_values in task_values.items():
                values = np.asarray(raw_values, dtype=float)
                expected = finite_summaries[size][task][method]
                if (
                    values.shape != (24,)
                    or not np.all(np.isfinite(values))
                    or not np.isclose(
                        np.mean(values), expected["mean"], rtol=0, atol=5e-12
                    )
                    or not np.isclose(
                        np.std(values, ddof=1) / np.sqrt(len(values)),
                        expected["se"],
                        rtol=0,
                        atol=5e-12,
                    )
                ):
                    raise RuntimeError(
                        "finite-size seed values disagree with their summary: "
                        f"N={size}/{method}/{task}"
                    )
    finite_size["seed_values"] = seed_sizes

    provenance = snapshot["provenance"]
    if (
        provenance["activity_snapshot_sha256"] != sha256(ACTIVITY)
        or provenance["forgetting_aggregate_sha256"] != sha256(AGGREGATE)
    ):
        raise RuntimeError("Experiment 1 robustness provenance hash mismatch")

    rows = snapshot["rows"]
    by_key = {row["key"]: row for row in rows}
    required = {
        "fixed_b",
        "washout_800",
        "activity_matched",
        "gap_matched",
        "independent_selection",
        "zz_x_z",
        "xy_z_x",
        "xx_ring",
    }
    if set(by_key) != required or len(rows) != len(required):
        raise RuntimeError("unexpected robustness-forest row set")

    principal_metrics = principal.get("metrics", {})
    expected_principal_archive = (
        "e24df615f8762ba9aa950673b5d776eddbc186a9ad277086c2930cde0ea46948"
    )
    if (
        principal.get("artifact_type")
        != "figure3_principal_absolute_summary"
        or principal.get("status") != "complete"
        or principal.get("source_archive_sha256") != expected_principal_archive
        or int(principal.get("n_pairs", -1)) != 32
        or set(principal_metrics) != {"stm", "narma10"}
    ):
        raise RuntimeError("principal Figure 3 summary violates its contract")
    for metric in ("stm", "narma10"):
        if set(principal_metrics[metric]) != {"local", "collective"}:
            raise RuntimeError(f"principal summary is incomplete for {metric}")
        for method in ("local", "collective"):
            summary_values = np.asarray(
                [
                    principal_metrics[metric][method].get("mean"),
                    principal_metrics[metric][method].get("se"),
                ],
                dtype=float,
            )
            if (
                summary_values.shape != (2,)
                or not np.all(np.isfinite(summary_values))
                or summary_values[1] < 0
            ):
                raise RuntimeError(
                    f"principal summary is invalid for {metric}/{method}"
                )
    principal_seeds = np.asarray(principal.get("seeds", []), dtype=int)
    principal_values = principal.get("values", {})
    if (
        principal_seeds.shape != (32,)
        or len(np.unique(principal_seeds)) != 32
        or not np.all(principal_seeds[:-1] < principal_seeds[1:])
        or set(principal_values) != {"stm", "narma10"}
    ):
        raise RuntimeError("principal Figure 3 seed snapshot is incomplete")
    for metric in ("stm", "narma10"):
        if set(principal_values[metric]) != {"local", "collective"}:
            raise RuntimeError(f"principal seed values are incomplete for {metric}")
        for method in ("local", "collective"):
            raw_values = np.asarray(
                principal_values[metric][method],
                dtype=float,
            )
            expected = principal_metrics[metric][method]
            if (
                raw_values.shape != (32,)
                or not np.all(np.isfinite(raw_values))
                or not np.isclose(
                    np.mean(raw_values), expected["mean"], rtol=0, atol=5e-12
                )
                or not np.isclose(
                    np.std(raw_values, ddof=1) / np.sqrt(32),
                    expected["se"],
                    rtol=0,
                    atol=5e-12,
                )
            ):
                raise RuntimeError(
                    f"principal seed values disagree for {metric}/{method}"
                )
    if (
        not np.isclose(
            principal_metrics["stm"]["collective"]["mean"]
            - principal_metrics["stm"]["local"]["mean"],
            by_key["fixed_b"]["mean"],
            rtol=0,
            atol=5e-10,
        )
        or principal_metrics["narma10"]["collective"]["mean"]
        >= principal_metrics["narma10"]["local"]["mean"]
    ):
        raise RuntimeError("principal summary disagrees with the audited effect")

    reset_arrays = reset_architecture.get("arrays", {})
    reset_local = np.asarray(reset_arrays.get("stm_local", []), dtype=float)
    reset_collective = np.asarray(
        reset_arrays.get("stm_collective", []), dtype=float
    )
    if (
        reset_architecture.get("schema_version") != 1
        or reset_architecture.get("protocol", {}).get("architecture")
        != "input-by-reset"
        or reset_architecture.get("protocol", {}).get("pairs") != 16
        or len(reset_architecture.get("source", {}).get("sha256", "")) != 64
        or reset_local.shape != (16,)
        or reset_collective.shape != (16,)
        or not np.all(np.isfinite(reset_local))
        or not np.all(np.isfinite(reset_collective))
    ):
        raise RuntimeError("reset-architecture STM snapshot is incomplete")
    reset_difference = reset_collective - reset_local
    reset_summary = simultaneous_summary(reset_difference.tolist())
    if not np.all(reset_difference > 0):
        raise RuntimeError("reset-architecture STM ordering changed")
    reset_summary["raw"] = reset_difference.tolist()
    reset_summary["key"] = "reset_encoding"

    scalar_seed_rows = scalar_control_seeds.get("rows", {})
    expected_scalar_counts = {
        "fixed_b": 32,
        "washout_800": 10,
        "activity_matched": 8,
        "gap_matched": 24,
        "independent_selection": 24,
        "zz_x_z": 32,
        "xy_z_x": 32,
        "xx_ring": 32,
    }
    scalar_sources = scalar_control_seeds.get("sources", {})
    expected_scalar_sources = {
        key: snapshot["provenance"][key]
        for key in (
            "final_protocol_archive_sha256",
            "review_protocol_archive_sha256",
            "activity_snapshot_sha256",
            "forgetting_aggregate_sha256",
            "independent_selection_aggregate_sha256",
        )
    }
    if (
        scalar_control_seeds.get("artifact_type")
        != "figure3_figure6_forest_seed_values"
        or scalar_control_seeds.get("status") != "complete"
        or scalar_control_seeds.get("metric") != snapshot.get("metric")
        or scalar_sources != expected_scalar_sources
        or set(scalar_seed_rows) != set(expected_scalar_counts)
    ):
        raise RuntimeError("scalar-control seed snapshot violates its contract")
    for key, expected_count in expected_scalar_counts.items():
        seed_rows = scalar_seed_rows[key]
        seeds = [int(row["seed"]) for row in seed_rows]
        values = [float(row["value"]) for row in seed_rows]
        expected = by_key[key]
        observed = simultaneous_summary(
            values,
            family_size=2 if key == "gap_matched" else 1,
        )
        if (
            len(seed_rows) != expected_count
            or len(set(seeds)) != expected_count
            or not np.all(np.isfinite(values))
            or not np.isclose(
                observed["mean"], expected["mean"], rtol=0, atol=5e-10
            )
            or not np.isclose(
                observed["ci95_low"],
                expected["ci95_low"],
                rtol=0,
                atol=5e-10,
            )
            or not np.isclose(
                observed["ci95_high"],
                expected["ci95_high"],
                rtol=0,
                atol=5e-10,
            )
        ):
            raise RuntimeError(
                f"scalar-control seed values disagree with {key}"
            )

    washout = aggregate["long_washout_stm"]["washouts"]["800"][
        "collective_minus_local_ground_initialization"
    ]
    assert_close(
        by_key["washout_800"],
        washout,
        ("mean", "ci95_low", "ci95_high"),
    )
    if (
        int(by_key["washout_800"]["n"]) != int(washout["n"])
        or int(by_key["washout_800"]["wins"]) != int(washout["wins"])
    ):
        raise RuntimeError("strict-washout row count mismatch")

    activity_summary = activity["summary"]
    expected_activity = {
        "mean": activity_summary["collective_minus_local_stm"],
        "ci95_low": activity_summary["collective_minus_local_stm_ci95"][0],
        "ci95_high": activity_summary["collective_minus_local_stm_ci95"][1],
    }
    assert_close(
        by_key["activity_matched"],
        expected_activity,
        ("mean", "ci95_low", "ci95_high"),
    )

    expected_gap = simultaneous_summary(
        aggregate["input_response"]["paired_stm_difference"],
        family_size=2,
    )
    assert_close(
        by_key["gap_matched"],
        expected_gap,
        ("mean", "ci95_low", "ci95_high"),
    )
    if (
        int(by_key["gap_matched"]["n"]) != expected_gap["n"]
        or int(by_key["gap_matched"]["wins"]) != expected_gap["wins"]
    ):
        raise RuntimeError("gap-matched row count mismatch")

    with GAP_CALIBRATION.open(newline="", encoding="utf-8") as stream:
        calibration = list(csv.DictReader(stream))
    if len(calibration) != 24:
        raise RuntimeError("pairwise gap calibration must contain 24 rows")
    relative_errors = np.asarray(
        [float(row["rel_gap_error"]) for row in calibration],
        dtype=float,
    )
    if np.max(np.abs(relative_errors)) > 0.005:
        raise RuntimeError("pairwise gap calibration failed the 0.5% gate")

    if sha256(GAP_LAG_CAPACITIES) != GAP_LAG_CAPACITIES_SHA256:
        raise RuntimeError("gap-control lag-capacity checksum mismatch")
    with GAP_LAG_CAPACITIES.open(newline="", encoding="utf-8") as stream:
        lag_rows = list(csv.DictReader(stream))
    expected_labels = {"collective", "local_fixed", "local_gap_matched"}
    expected_seeds = set(range(64000, 64024))
    expected_delays = set(range(1, 21))
    identities: set[tuple[int, str, int]] = set()
    observed_labels: set[str] = set()
    observed_seeds: set[int] = set()
    observed_delays: set[int] = set()
    for row in lag_rows:
        seed = int(row["seed"])
        label = row["label"]
        delay = int(row["delay"])
        capacity = float(row["capacity"])
        identity = (seed, label, delay)
        if identity in identities or not np.isfinite(capacity) or not 0 <= capacity <= 1:
            raise RuntimeError("gap-control lag-capacity row is invalid")
        identities.add(identity)
        observed_labels.add(label)
        observed_seeds.add(seed)
        observed_delays.add(delay)
    if (
        len(lag_rows) != 24 * 3 * 20
        or observed_labels != expected_labels
        or observed_seeds != expected_seeds
        or observed_delays != expected_delays
    ):
        raise RuntimeError("gap-control lag-capacity coverage is incomplete")

    extension_steps = np.asarray(extension["step"], dtype=int)
    extension_curve = np.asarray(
        extension["maximum_trace_distance_across_seeds"],
        dtype=float,
    )
    base_curve = np.asarray(
        aggregate["convergence"]["cases"]["principal_collective_N5"][
            "maximum_trace_distance_across_seeds"
        ],
        dtype=float,
    )
    if (
        extension.get("artifact_type")
        != "collective_convergence_extension"
        or len(extension_steps) != len(extension_curve)
        or not np.array_equal(extension_steps, np.arange(len(extension_steps)))
        or extension_steps[-1] != 1200
        or np.max(np.abs(extension_curve[: len(base_curve)] - base_curve))
        > 5e-13
        or extension_curve[-1] > 1e-14
    ):
        raise RuntimeError("collective convergence extension is invalid")

    extended_cases = local_pair_extension.get("cases")
    expected_extended_cases = {
        f"principal_{design}_N{size}"
        for design in ("local", "pair")
        for size in (4, 5, 6)
    }
    if (
        local_pair_extension.get("artifact_type")
        != "local_pair_convergence_extension"
        or not isinstance(extended_cases, dict)
        or set(extended_cases) != expected_extended_cases
        or local_pair_extension.get("validation", {}).get(
            "all_convergence_gates_passed"
        )
        is not True
    ):
        raise RuntimeError("local/pair convergence extension is invalid")
    for name in sorted(expected_extended_cases):
        extended_case = extended_cases[name]
        extended_steps = np.asarray(extended_case["step"], dtype=int)
        extended_curve = np.asarray(
            extended_case["maximum_trace_distance_across_seeds"],
            dtype=float,
        )
        base_case = aggregate["convergence"]["cases"][name]
        base_case_curve = np.asarray(
            base_case["maximum_trace_distance_across_seeds"],
            dtype=float,
        )
        if (
            len(extended_steps) != len(extended_curve)
            or not np.array_equal(
                extended_steps,
                np.arange(len(extended_steps)),
            )
            or extended_steps[-1] != 1200
            or np.max(
                np.abs(
                    extended_curve[: len(base_case_curve)]
                    - base_case_curve
                )
            )
            > 5e-13
            or np.max(extended_curve[1100:]) > 1e-14
        ):
            raise RuntimeError(
                f"local/pair convergence extension is invalid: {name}"
            )

    return (
        principal,
        reset_summary,
        aggregate,
        extension,
        local_pair_extension,
        finite_size,
        scalar_seed_rows,
        rows,
        calibration,
        lag_rows,
    )


def draw_forest(
    axis: plt.Axes,
    rows: list[dict],
    *,
    color: str,
    marker: str,
    labels: dict[str, str],
    seed_values: dict[str, list[dict]] | None = None,
) -> None:
    """Draw one square forest facet without duplicating a wins column."""

    y_positions = np.arange(len(rows) - 1, -1, -1, dtype=float)
    for row, y_position in zip(rows, y_positions, strict=True):
        mean = float(row["mean"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        if seed_values is not None:
            raw_values = np.asarray(
                [entry["value"] for entry in seed_values[row["key"]]],
                dtype=float,
            )
            vertical_jitter = np.linspace(-0.055, 0.055, len(raw_values))
            axis.scatter(
                raw_values,
                np.full(len(raw_values), y_position) + vertical_jitter,
                s=7.0,
                marker="o",
                color=st.distance_faded_colors(
                    color,
                    raw_values,
                    center=mean,
                ),
                linewidths=0,
                zorder=2,
            )
        axis.errorbar(
            mean,
            y_position,
            xerr=np.asarray([[mean - low], [high - mean]]),
            color=color,
            marker=marker,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            linestyle="none",
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )
        # Put exact effects on the free side of each interval.  The activity
        # interval is unusually wide, so its label sits inside the interval,
        # immediately to the right of the marker.
        activity_row = row["key"] == "activity_matched"
        label_on_left = high > 4.85 and not activity_row
        label_x = (
            mean + 0.35
            if activity_row
            else low - 0.10
            if label_on_left
            else high + 0.10
        )
        axis.text(
            label_x,
            y_position,
            f"{mean:.2f}",
            ha="right" if label_on_left else "left",
            va="center",
            fontsize=FIG_TEXT_SIZE,
            color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.03),
            zorder=4,
        )

    axis.axvline(
        0,
        color=st.INK,
        linewidth=st.REFERENCE_LINEWIDTH,
        zorder=1,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels([labels[row["key"]] for row in rows])
    axis.set_xlim(-0.25, 6.35)
    axis.set_xticks([0, 3, 6])
    axis.set_ylim(-0.65, len(rows) - 0.35)
    st.style_axis(axis, "x", minor_grid=False)


def draw_size_trend(
    axis: plt.Axes,
    finite_size: dict,
    *,
    task: str,
) -> None:
    """Draw the local--collective finite-size trend for one native metric."""

    sizes = np.arange(4, 9)
    summaries = finite_size["summaries"]["primary_fixed"]
    seed_values = finite_size["seed_values"]
    critical = float(student_t.ppf(0.975, 23))
    for method, label, color, marker, facecolor, seed_offset in (
        ("CD_paper", "local", st.UNIFORM_LOCAL, "D", "white", -0.035),
        (
            "B3_collective",
            "collective",
            st.COLLECTIVE,
            "o",
            st.COLLECTIVE,
            0.035,
        ),
    ):
        means = np.asarray(
            [
                float(summaries[str(size)][task][method]["mean"])
                for size in sizes
            ],
            dtype=float,
        )
        half_widths = critical * np.asarray(
            [
                float(summaries[str(size)][task][method]["se"])
                for size in sizes
            ],
            dtype=float,
        )
        # Individual reservoirs sit behind the aggregate marks.  A fixed,
        # symmetric horizontal spread makes coincident values visible without
        # suggesting additional x-axis resolution.
        seed_jitter = np.linspace(-0.030, 0.030, 24)
        for size in sizes:
            axis.scatter(
                np.full(24, size + seed_offset) + seed_jitter,
                np.asarray(
                    seed_values[str(size)]["values"][method][task],
                    dtype=float,
                ),
                s=7.0,
                marker="o",
                color=st.distance_faded_colors(
                    color,
                    np.asarray(
                        seed_values[str(size)]["values"][method][task],
                        dtype=float,
                    ),
                    center=float(summaries[str(size)][task][method]["mean"]),
                ),
                linewidths=0,
                zorder=2,
            )
        axis.errorbar(
            sizes,
            means,
            yerr=half_widths,
            color=color,
            marker=marker,
            markersize=MARKER_SIZE,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            linestyle="-",
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            label=label,
            zorder=3,
        )

    axis.set_xlim(3.78, 8.22)
    axis.set_xticks([4, 6, 8])
    axis.set_xlabel(r"qubits $N$", labelpad=1.0)
    if task == "stm":
        axis.set_ylim(6.35, 16.80)
        axis.set_yticks([8, 12, 16])
        axis.set_ylabel("STM capacity", labelpad=1.0)
    else:
        axis.set_ylim(0.12, 0.48)
        axis.set_yticks([0.2, 0.3, 0.4])
        axis.set_ylabel("NARMA-10 NMSE", labelpad=1.0)
    st.style_axis(axis, "both", minor_axis="both", minor_grid=False)


def draw_principal_metric(
    axis: plt.Axes,
    principal: dict,
    *,
    task: str,
    color: str,
) -> None:
    """Draw the two-point principal N=5 absolute comparison."""

    summaries = principal["metrics"][task]
    n_pairs = int(principal["n_pairs"])
    critical = float(student_t.ppf(0.975, n_pairs - 1))
    means = np.asarray(
        [summaries["local"]["mean"], summaries["collective"]["mean"]],
        dtype=float,
    )
    half_widths = critical * np.asarray(
        [summaries["local"]["se"], summaries["collective"]["se"]],
        dtype=float,
    )
    markers = ("D", "o" if task == "stm" else "s")
    colors = (st.UNIFORM_LOCAL, color)
    faces = ("white", color)
    axis.plot(
        [0, 1],
        means,
        color=color,
        linestyle="-" if task == "stm" else "--",
        linewidth=1.75,
        zorder=2,
    )
    for x_value, mean, half_width, marker, mark_color, face in zip(
        (0, 1),
        means,
        half_widths,
        markers,
        colors,
        faces,
        strict=True,
    ):
        axis.errorbar(
            x_value,
            mean,
            yerr=half_width,
            color=mark_color,
            marker=marker,
            markerfacecolor=face,
            markeredgecolor=mark_color,
            markeredgewidth=1.05,
            markersize=7.2,
            linestyle="none",
            capsize=3.1,
            capthick=1.35,
            elinewidth=1.35,
            zorder=3,
        )
        if x_value == 0:
            continue
        if task == "narma10":
            label_x = x_value + 0.16
            label_y = mean - 0.006
            label_ha = "center"
            label_va = "top"
        else:
            # The mini-axis is deliberately short at final column size.  Put
            # the value beside the marker so it cannot collide with the top
            # spine while retaining full manuscript-size typography.
            label_x = 0.70
            label_y = mean
            label_ha = "right"
            label_va = "center"
        axis.text(
            label_x,
            label_y,
            f"{mean:.3f}",
            ha=label_ha,
            va=label_va,
            fontsize=FIG_TEXT_SIZE,
            color=mark_color,
            zorder=4,
        )

    if task == "stm":
        percent = 100.0 * (means[1] - means[0]) / means[0]
        axis.set_ylim(7.4, 13.75)
        axis.set_yticks([8, 10, 12])
        axis.set_ylabel("STM", labelpad=1.0)
        annotation_color = st.COLLECTIVE
        text = f"+{percent:.0f}%"
    else:
        percent = 100.0 * (means[0] - means[1]) / means[0]
        axis.set_ylim(0.19, 0.35)
        axis.set_yticks([0.2, 0.3])
        axis.set_ylabel("NMSE", labelpad=1.0)
        annotation_color = st.COLLECTIVE
        text = f"-{percent:.0f}%"
    axis.text(
        0.62,
        float(np.mean(means)),
        text,
        ha="left",
        va="center",
        fontsize=9.55,
        color=annotation_color,
        fontweight="bold",
    )
    axis.set_xlim(-0.30, 1.58)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["uniform local", "collective"])
    st.style_axis(axis, "both", minor_grid=False)
    axis.minorticks_off()


def draw_joint_principal(axis: plt.Axes, principal: dict) -> plt.Axes:
    """Draw the paired principal scores on one square native-unit panel."""

    right_axis = axis.twinx()
    x_values = np.asarray([0.0, 1.0])
    n_pairs = int(principal["n_pairs"])
    critical = float(student_t.ppf(0.975, n_pairs - 1))

    for target, task, linestyle, markers in (
        (
            axis,
            "stm",
            "-",
            ("D", "o"),
        ),
        (
            right_axis,
            "narma10",
            "--",
            ("D", "s"),
        ),
    ):
        raw = principal["values"][task]
        local_values = np.asarray(raw["local"], dtype=float)
        collective_values = np.asarray(raw["collective"], dtype=float)
        summaries = principal["metrics"][task]
        means = np.asarray(
            [summaries["local"]["mean"], summaries["collective"]["mean"]],
            dtype=float,
        )
        half_widths = critical * np.asarray(
            [summaries["local"]["se"], summaries["collective"]["se"]],
            dtype=float,
        )

        # Mirror Figure 4: paired instances remain visible as quiet trajectories
        # behind the thicker aggregate line and pointwise intervals.
        favorable_difference = (
            collective_values - local_values
            if task == "stm"
            else local_values - collective_values
        )
        trajectory_colors = st.distance_faded_colors(
            st.NEUTRAL_DESIGN,
            favorable_difference,
            center=float(np.mean(favorable_difference)),
            near_alpha=0.15,
            far_alpha=0.020,
        )
        seed_offsets = np.linspace(-0.016, 0.016, n_pairs)
        for local, collective, offset, trajectory_color in zip(
            local_values,
            collective_values,
            seed_offsets,
            trajectory_colors,
            strict=True,
        ):
            target.plot(
                x_values + offset,
                [local, collective],
                color=trajectory_color,
                linewidth=0.62,
                zorder=1,
            )
        for x_value, values in zip(
            x_values,
            (local_values, collective_values),
            strict=True,
        ):
            method_color = (
                st.LOCAL_CONTRAST if x_value == 0 else st.COLLECTIVE
            )
            target.scatter(
                np.full(n_pairs, x_value) + seed_offsets,
                values,
                s=6.0,
                marker="o",
                color=st.distance_faded_colors(
                    method_color,
                    values,
                    center=float(np.mean(values)),
                    near_alpha=0.20,
                    far_alpha=0.025,
                ),
                linewidths=0,
                zorder=2,
            )

        target.plot(
            x_values,
            means,
            color=st.NEUTRAL_DESIGN,
            linestyle=linestyle,
            linewidth=DATA_LINEWIDTH,
            zorder=3,
        )
        for x_value, mean, half_width, marker in zip(
            x_values,
            means,
            half_widths,
            markers,
            strict=True,
        ):
            mark_color = (
                st.LOCAL_CONTRAST if x_value == 0 else st.COLLECTIVE
            )
            target.errorbar(
                x_value,
                mean,
                yerr=half_width,
                color=mark_color,
                marker=marker,
                markerfacecolor=("white" if x_value == 0 else mark_color),
                markeredgecolor=mark_color,
                markeredgewidth=MARKER_EDGEWIDTH,
                markersize=MARKER_SIZE,
                linestyle="none",
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                elinewidth=ERROR_LINEWIDTH,
                zorder=4,
            )

    stm_collective = float(principal["metrics"]["stm"]["collective"]["mean"])
    narma_collective = float(
        principal["metrics"]["narma10"]["collective"]["mean"]
    )
    axis.text(
        0.94,
        stm_collective + 0.20,
        f"{stm_collective:.3f}",
        color=st.COLLECTIVE,
        fontsize=FIG_TEXT_SIZE,
        ha="right",
        va="bottom",
        zorder=5,
    )
    right_axis.text(
        1.12,
        narma_collective - 0.006,
        f"{narma_collective:.3f}",
        color=st.COLLECTIVE,
        fontsize=FIG_TEXT_SIZE,
        ha="right",
        va="top",
        zorder=5,
    )

    axis.set_xlim(-0.18, 1.18)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["uniform local", "collective"])
    axis.set_ylim(7.4, 13.35)
    axis.set_yticks([8, 10, 12])
    axis.tick_params(axis="y", colors=st.INK)
    right_axis.set_ylim(0.19, 0.35)
    right_axis.set_yticks([0.2, 0.3])
    right_axis.tick_params(axis="y", colors=st.INK, pad=1.7)
    axis.set_ylabel("")
    right_axis.set_ylabel("")
    st.style_axis(axis, "both", minor_grid=False)
    axis.minorticks_off()
    right_axis.minorticks_off()
    right_axis.grid(False)
    return right_axis


def draw_size_recurrence(
    axis: plt.Axes,
    finite_size: dict,
    task: str,
) -> None:
    """Draw one finite-size recurrence without a twin-axis overlay."""

    if task not in {"stm", "narma10"}:
        raise ValueError(f"unsupported finite-size task: {task}")
    sizes = np.arange(4, 9)
    summaries = finite_size["summaries"]["primary_fixed"]
    seed_values = finite_size["seed_values"]
    critical = float(student_t.ppf(0.975, 23))
    series = (
        (
            "CD_paper",
            st.LOCAL_CONTRAST,
            "D",
            "white",
            "--",
            -0.025,
        ),
        (
            "B3_collective",
            st.COLLECTIVE,
            "o",
            st.COLLECTIVE,
            "-",
            0.025,
        ),
    )
    for method, color, marker, face, linestyle, seed_offset in series:
        means = np.asarray(
            [summaries[str(size)][task][method]["mean"] for size in sizes],
            dtype=float,
        )
        half_widths = critical * np.asarray(
            [summaries[str(size)][task][method]["se"] for size in sizes],
            dtype=float,
        )
        seed_jitter = np.linspace(-0.025, 0.025, 24)
        for size in sizes:
            raw = np.asarray(
                seed_values[str(size)]["values"][method][task],
                dtype=float,
            )
            axis.scatter(
                np.full(24, size + seed_offset) + seed_jitter,
                raw,
                s=5.2,
                marker="o",
                color=st.distance_faded_colors(
                    color,
                    raw,
                    center=float(summaries[str(size)][task][method]["mean"]),
                    near_alpha=0.12,
                    far_alpha=0.018,
                ),
                linewidths=0,
                zorder=1.8,
            )
        axis.errorbar(
            sizes,
            means,
            yerr=half_widths,
            color=color,
            marker=marker,
            markerfacecolor=face,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            linestyle=linestyle,
            linewidth=DATA_LINEWIDTH,
            capsize=ERROR_CAPSIZE,
            capthick=ERROR_LINEWIDTH,
            elinewidth=ERROR_LINEWIDTH,
            zorder=3,
        )

    axis.set_xlim(3.75, 8.25)
    axis.set_xticks([4, 6, 8])
    axis.set_xlabel(r"qubits $N$", labelpad=2.0)
    if task == "stm":
        axis.set_ylim(6.8, 19.2)
        axis.set_yticks([8, 12, 16])
        metric_label = "STM capacity"
    else:
        axis.set_ylim(0.10, 0.45)
        axis.set_yticks([0.1, 0.2, 0.3, 0.4])
        metric_label = "NARMA-10 NMSE"
    axis.set_ylabel("")
    axis.text(
        0.04,
        0.95,
        metric_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FIG_TEXT_SIZE,
        color=st.INK,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "pad": 0.45,
        },
        zorder=6,
    )
    st.style_axis(axis, "both", minor_grid=False)
    axis.minorticks_off()


def case_legend_handles() -> list[Line2D]:
    """Return the shared two-design key for the case figure."""

    return [
        Line2D(
            [0],
            [0],
            color=st.LOCAL_CONTRAST,
            marker="D",
            markerfacecolor="white",
            markeredgewidth=MARKER_EDGEWIDTH,
            linestyle="--",
            linewidth=DATA_LINEWIDTH,
            label="uniform local",
        ),
        Line2D(
            [0],
            [0],
            color=st.COLLECTIVE,
            marker="o",
            markerfacecolor=st.COLLECTIVE,
            markeredgewidth=MARKER_EDGEWIDTH,
            linewidth=DATA_LINEWIDTH,
            label="collective relaxation",
        ),
    ]


def draw_robustness_overview(
    axis: plt.Axes,
    by_key: dict[str, dict],
    scalar_seed_values: dict[str, list[dict]],
    reset_summary: dict,
) -> None:
    """Draw the seven-row robustness overview without pooling ensembles."""

    groups: list[tuple[str, list[dict]]] = [
        (r"fixed $B$", [by_key["fixed_b"]]),
        ("long washout", [by_key["washout_800"]]),
        (
            "H variants",
            [by_key[key] for key in ("zz_x_z", "xx_ring", "xy_z_x")],
        ),
        ("reset encoding", [reset_summary]),
        ("activity match", [by_key["activity_matched"]]),
        ("matched\nrelaxation rate", [by_key["gap_matched"]]),
        ("indep. selection", [by_key["independent_selection"]]),
    ]
    y_positions = np.arange(len(groups) - 1, -1, -1, dtype=float)
    for (label, rows), y_position in zip(groups, y_positions, strict=True):
        offsets = (
            np.linspace(-0.13, 0.13, len(rows))
            if len(rows) > 1
            else np.asarray([0.0])
        )
        for row, offset in zip(rows, offsets, strict=True):
            mean = float(row["mean"])
            low = float(row["ci95_low"])
            high = float(row["ci95_high"])
            if row["key"] == "reset_encoding":
                raw_values = np.asarray(row["raw"], dtype=float)
            else:
                raw_values = np.asarray(
                    [entry["value"] for entry in scalar_seed_values[row["key"]]],
                    dtype=float,
                )
            vertical_jitter = np.linspace(-0.045, 0.045, len(raw_values))
            axis.scatter(
                raw_values,
                np.full(len(raw_values), y_position + offset) + vertical_jitter,
                s=6.0,
                marker="o",
                color=st.distance_faded_colors(
                    st.COLLECTIVE,
                    raw_values,
                    center=mean,
                    near_alpha=0.15,
                    far_alpha=0.025,
                ),
                linewidths=0,
                zorder=2,
            )
            axis.errorbar(
                mean,
                y_position + offset,
                xerr=np.asarray([[mean - low], [high - mean]]),
                color=st.COLLECTIVE,
                marker="o",
                markerfacecolor=st.COLLECTIVE,
                markeredgecolor=st.COLLECTIVE,
                markeredgewidth=MARKER_EDGEWIDTH,
                markersize=MARKER_SIZE,
                linestyle="none",
                capsize=ERROR_CAPSIZE,
                capthick=ERROR_LINEWIDTH,
                elinewidth=ERROR_LINEWIDTH,
                zorder=3,
            )
    axis.axvline(0, color=st.INK, linewidth=st.REFERENCE_LINEWIDTH, zorder=1)
    axis.set_yticks(y_positions)
    axis.set_yticklabels([label for label, _rows in groups])
    for tick_label in axis.get_yticklabels():
        if "\n" in tick_label.get_text():
            tick_label.set_linespacing(0.72)
    axis.set_xlim(-0.30, 6.15)
    axis.set_xticks([0, 3, 6])
    axis.set_ylim(-0.55, len(groups) - 0.45)
    axis.set_xlabel("STM gain", labelpad=2.0)
    st.style_axis(axis, "x", minor_grid=False)
    axis.minorticks_off()


def draw_fixed_lag_memory(axis: plt.Axes, lag_rows: list[dict]) -> None:
    """Draw fixed-local and collective lag capacities from all 24 seeds."""

    lag_values: dict[tuple[str, int], list[float]] = {}
    for row in lag_rows:
        lag_values.setdefault((row["label"], int(row["delay"])), []).append(
            float(row["capacity"])
        )
    delays = np.arange(1, 21)
    for key, label, color, marker, facecolor, seed_offset in (
        (
            "local_fixed",
            "uniform local",
            st.LOCAL_CONTRAST,
            "D",
            "white",
            -0.035,
        ),
        (
            "collective",
            "collective",
            st.COLLECTIVE,
            "o",
            st.COLLECTIVE,
            0.035,
        ),
    ):
        means: list[float] = []
        low: list[float] = []
        high: list[float] = []
        seed_jitter = np.linspace(-0.022, 0.022, 24)
        for delay in delays:
            values = np.asarray(lag_values[(key, int(delay))], dtype=float)
            mean = float(np.mean(values))
            half_width = float(
                student_t.ppf(0.975, len(values) - 1)
                * np.std(values, ddof=1)
                / np.sqrt(len(values))
            )
            axis.scatter(
                np.full(len(values), delay + seed_offset) + seed_jitter,
                values,
                s=5.5,
                marker="o",
                color=st.distance_faded_colors(
                    color,
                    values,
                    center=mean,
                    near_alpha=0.13,
                    far_alpha=0.020,
                ),
                linewidths=0,
                zorder=2,
            )
            means.append(mean)
            low.append(max(0.0, mean - half_width))
            high.append(min(1.0, mean + half_width))
        means_array = np.asarray(means)
        low_array = np.asarray(low)
        high_array = np.asarray(high)
        axis.fill_between(
            delays,
            low_array,
            high_array,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            delays,
            means_array,
            color=color,
            linewidth=DATA_LINEWIDTH,
            marker=marker,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            label=label,
            zorder=3,
        )
    axis.set_xlim(0.65, 20.35)
    # Preserve all capacities while reserving clear space for the in-panel
    # metric description above the delay curves.
    axis.set_ylim(-0.02, 1.18)
    axis.set_xticks([1, 10, 20])
    axis.set_yticks([0, 0.5, 1.0])
    axis.set_xlabel(r"input delay $\tau$", labelpad=2.0)
    axis.set_ylabel("")
    axis.text(
        0.04,
        0.95,
        r"STM contribution $C_\tau$",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=FIG_TEXT_SIZE,
        color=st.INK,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
            "pad": 0.45,
        },
        zorder=6,
    )
    st.style_axis(axis, "both", minor_grid=False)
    axis.minorticks_off()


def draw_convergence(
    axis: plt.Axes,
    extension: dict,
    local_pair_extension: dict,
) -> None:
    """Draw the existing switched-input initial-state convergence diagnostic."""

    extended_convergence = local_pair_extension["cases"]
    local_pair_steps = np.asarray(
        extended_convergence["principal_local_N4"]["step"],
        dtype=int,
    )
    local_envelope = np.maximum.reduce(
        [
            np.asarray(
                extended_convergence[f"principal_local_N{size}"][
                    "maximum_trace_distance_across_seeds"
                ],
                dtype=float,
            )
            for size in (4, 5, 6)
        ]
    )
    pair_envelope = np.maximum.reduce(
        [
            np.asarray(
                extended_convergence[f"principal_pair_N{size}"][
                    "maximum_trace_distance_across_seeds"
                ],
                dtype=float,
            )
            for size in (4, 5, 6)
        ]
    )
    collective_steps = np.asarray(extension["step"], dtype=int)
    collective_n5 = np.asarray(
        extension["maximum_trace_distance_across_seeds"],
        dtype=float,
    )
    axis.axhline(
        1e-14,
        color=st.GRID_MAJOR,
        linewidth=st.GRID_MAJOR_LINEWIDTH,
        linestyle=st.GRID_MAJOR_DASH,
        zorder=0,
    )
    for steps, values, color, linestyle, marker, facecolor, label in (
        (
            local_pair_steps,
            local_envelope,
            st.UNIFORM_LOCAL,
            "-",
            "D",
            "white",
            "local",
        ),
        (
            collective_steps,
            collective_n5,
            st.COLLECTIVE,
            "-",
            "o",
            st.COLLECTIVE,
            "collective",
        ),
        (
            local_pair_steps,
            pair_envelope,
            st.PAIR_LOSS,
            "-.",
            "^",
            st.PAIR_LOSS,
            "pair",
        ),
    ):
        axis.semilogy(
            steps,
            np.maximum(values, 1e-16),
            color=color,
            linestyle=linestyle,
            linewidth=DATA_LINEWIDTH,
            marker=marker,
            markevery=200,
            markersize=MARKER_SIZE,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            label=label,
        )
    for washout in (200, 800):
        axis.axvline(
            washout,
            color=st.GRID_MAJOR,
            linewidth=st.GRID_MAJOR_LINEWIDTH,
            linestyle=st.GRID_MAJOR_DASH,
            zorder=0,
        )
    axis.set_xlim(0, 1225)
    axis.set_ylim(1e-16, 1.30)
    axis.set_xticks([0, 600, 1200])
    axis.set_yticks([1, 1e-8, 1e-16])
    axis.set_xlabel("input steps", labelpad=1)
    axis.set_ylabel("max. trace distance", labelpad=1)
    st.style_axis(axis, "both", minor_grid=False)


def draw_gap_lags(axis: plt.Axes, lag_rows: list[dict]) -> None:
    """Draw the existing midpoint-gap-control STM lag profiles."""

    lag_values: dict[tuple[str, int], list[float]] = {}
    for row in lag_rows:
        lag_values.setdefault(
            (row["label"], int(row["delay"])),
            [],
        ).append(float(row["capacity"]))
    delays = np.arange(1, 21)
    for key, label, color, linestyle, marker, facecolor in (
        (
            "collective",
            "collective",
            st.COLLECTIVE,
            "-",
            "o",
            st.COLLECTIVE,
        ),
        (
            "local_gap_matched",
            "matched rate",
            st.LOCAL_CONTRAST,
            "--",
            "s",
            st.LOCAL_CONTRAST,
        ),
        (
            "local_fixed",
            "fixed",
            st.LOCAL_CONTRAST,
            ":",
            "D",
            "white",
        ),
    ):
        label_offset = {
            "local_fixed": -0.08,
            "collective": 0.0,
            "local_gap_matched": 0.08,
        }[key]
        seed_jitter = np.linspace(-0.035, 0.035, 24)
        means: list[float] = []
        low: list[float] = []
        high: list[float] = []
        for delay in delays:
            values = np.asarray(lag_values[(key, int(delay))], dtype=float)
            axis.scatter(
                np.full(len(values), delay + label_offset) + seed_jitter,
                values,
                s=5.0,
                marker="o",
                color=st.distance_faded_colors(
                    color,
                    values,
                    center=float(np.mean(values)),
                    near_alpha=0.16,
                    far_alpha=0.025,
                ),
                linewidths=0,
                zorder=2,
            )
            mean = float(np.mean(values))
            half_width = float(
                student_t.ppf(0.975, len(values) - 1)
                * np.std(values, ddof=1)
                / np.sqrt(len(values))
            )
            means.append(mean)
            low.append(max(0.0, mean - half_width))
            high.append(min(1.0, mean + half_width))
        means_array = np.asarray(means)
        low_array = np.asarray(low)
        high_array = np.asarray(high)
        smooth_delays = np.linspace(float(delays[0]), float(delays[-1]), 381)
        smooth_means = PchipInterpolator(delays, means_array)(smooth_delays)
        smooth_low = np.minimum(
            PchipInterpolator(delays, low_array)(smooth_delays),
            smooth_means,
        )
        smooth_high = np.maximum(
            PchipInterpolator(delays, high_array)(smooth_delays),
            smooth_means,
        )
        axis.fill_between(
            smooth_delays,
            smooth_low,
            smooth_high,
            color=color,
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            smooth_delays,
            smooth_means,
            color=color,
            linestyle=linestyle,
            linewidth=DATA_LINEWIDTH,
            marker=marker,
            markevery=[0, 180, 380],
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=MARKER_EDGEWIDTH,
            markersize=MARKER_SIZE,
            label=label,
            zorder=3,
        )
    axis.set_xlim(1, 20)
    axis.set_ylim(0, 1.12)
    axis.set_xticks([1, 10, 20])
    axis.set_yticks([0, 0.5, 1.0])
    axis.set_yticklabels(["", "0.5", "1.0"])
    axis.set_xlabel("input delay", labelpad=1)
    axis.set_ylabel("")
    axis.yaxis.tick_left()
    st.style_axis(axis, "both", minor_axis="both", minor_grid=False)


def assert_square_panels(
    figure: plt.Figure,
    axes: np.ndarray,
    *,
    name: str,
) -> None:
    """Fail closed if any authored panel ceases to be physically square."""

    figure_width, figure_height = map(float, figure.get_size_inches())
    for axis in np.ravel(axes):
        box = axis.get_position()
        width_inches = float(box.width * figure_width)
        height_inches = float(box.height * figure_height)
        if not np.isclose(width_inches, height_inches, rtol=0, atol=1e-10):
            raise RuntimeError(
                f"{name} panels must remain physically square: "
                f"{width_inches:.6f} x {height_inches:.6f} in"
            )


def add_panel_letters(
    figure: plt.Figure,
    positions: tuple[tuple[str, float, float], ...],
) -> None:
    figure_width, figure_height = map(float, figure.get_size_inches())
    for letter, x_position, y_position in positions:
        figure.text(
            x_position / figure_width,
            y_position / figure_height,
            f"({letter})",
            ha="left",
            va="bottom",
            fontsize=FIG_PANEL_SIZE,
            color=st.INK,
        )


def finish_figure(
    figure: plt.Figure,
    axes: np.ndarray,
    *,
    name: str,
    output: Path,
) -> None:
    for axis in np.ravel(axes):
        axis.tick_params(
            axis="x",
            pad=getattr(axis, "_qrc_x_tick_pad", 1.7),
            labelsize=FIG_TICK_SIZE,
        )
        axis.tick_params(
            axis="y",
            pad=getattr(axis, "_qrc_y_tick_pad", 1.7),
            labelsize=FIG_TICK_SIZE,
        )
        axis.xaxis.label.set_size(FIG_AXIS_SIZE)
        axis.yaxis.label.set_size(FIG_AXIS_SIZE)
    st.audit_figure(
        figure,
        name,
        axes=axes,
        overlap_fraction=0.10,
        font_floor=FIG_TEXT_SIZE,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches=None,
        pad_inches=0,
        metadata={
            "Creator": "paper/make_forgetting_modes_figure.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)
    print(output)


def main() -> None:
    (
        principal,
        reset_summary,
        _aggregate,
        _extension,
        _local_pair_extension,
        finite_size,
        scalar_seed_values,
        forest_rows,
        _calibration,
        lag_rows,
    ) = load_and_validate()
    by_key = {row["key"]: row for row in forest_rows}

    # Section 3: one full-width 1x4 figure.  The redundant principal N=5 panel
    # is omitted because Figure 2 and the size sweep already report that
    # comparison.  The robustness overview leads the sequence, followed by
    # metric-specific size recurrences and the lag profile.  Separate STM and
    # NARMA-10 panels remove the former twin-axis overlay.
    case_height = 2.54
    case_figure = st.composite_figure("full", case_height)
    # Match the 1.21-inch square data rectangles used in Figure 4.  A fixed
    # quarter-inch gap leaves room for neighboring y labels, while the explicit
    # right margin prevents the final spine and ticks from touching the canvas.
    panel_side = 1.205
    panel_bottom = 0.65
    # Keep all four data rectangles identical to panel (d) and distribute their
    # left edges uniformly.  The larger leading margin accommodates panel (a)'s
    # categorical labels without shrinking its plotting area.
    panel_left = 1.08
    panel_gap = 0.258
    panel_positions = tuple(
        (
            panel_left + panel_index * (panel_side + panel_gap),
            panel_bottom,
        )
        for panel_index in range(4)
    )
    panel_right = panel_positions[-1][0]
    case_axes = np.asarray(
        [
            st.add_axes_inches(
                case_figure,
                [x_position, y_position, panel_side, panel_side],
            )
            for x_position, y_position in panel_positions
        ],
        dtype=object,
    )
    ax_robustness, ax_stm_size, ax_narma_size, ax_lags = case_axes
    panel_group_center = 0.5 * (panel_left + panel_right + panel_side)
    legend_width = 3.00
    legend_axis = st.add_axes_inches(
        case_figure,
        [panel_group_center - 0.5 * legend_width, 2.12, legend_width, 0.29],
    )
    legend_axis.set_axis_off()
    draw_robustness_overview(
        ax_robustness,
        by_key,
        scalar_seed_values,
        reset_summary,
    )
    draw_size_recurrence(ax_stm_size, finite_size, "stm")
    draw_size_recurrence(ax_narma_size, finite_size, "narma10")
    draw_fixed_lag_memory(ax_lags, lag_rows)

    legend_handles = case_legend_handles()
    legend_handles = [legend_handles[1], legend_handles[0]]
    legend_labels = ("collective relaxation", "uniform local")
    case_legend = st.legend(
        legend_axis,
        lw=st.LEGEND_FRAMEWIDTH,
        handles=legend_handles,
        labels=legend_labels,
        loc="center",
        ncol=2,
        frameon=True,
        fontsize=FIG_TEXT_SIZE,
        handlelength=1.10,
        handletextpad=0.36,
        columnspacing=0.90,
        labelspacing=0.05,
        borderpad=0.16,
        framealpha=1.0,
    )
    case_legend.set_zorder(6)

    figure_width, figure_height = map(float, case_figure.get_size_inches())
    for letter, (x_position, y_position) in zip(
        "abcd",
        panel_positions,
        strict=True,
    ):
        case_figure.text(
            (x_position - 0.12) / figure_width,
            (y_position + panel_side + 0.07) / figure_height,
            f"({letter})",
            ha="left",
            va="bottom",
            fontsize=FIG_PANEL_SIZE,
            color=st.INK,
        )

    for size_axis in (ax_stm_size, ax_narma_size):
        size_axis.get_xticklabels()[0].set_horizontalalignment("left")
        size_axis.get_xticklabels()[-1].set_horizontalalignment("right")
    ax_lags.get_xticklabels()[0].set_horizontalalignment("left")
    ax_lags.get_xticklabels()[-1].set_horizontalalignment("right")
    assert_square_panels(case_figure, case_axes, name="fig_collective_case")
    finish_figure(
        case_figure,
        case_axes,
        name="fig_collective_case",
        output=COLLECTIVE_CASE_OUTPUT,
    )

    # Section 4: the scalar protocol controls and the corresponding
    # gap-controlled lag profiles, with no Section 3 validity rows mixed in.
    controls_height = 2.40
    controls_bottom = 0.51
    controls_panel_side = 1.21
    controls_left = 0.51
    controls_right = 1.97
    controls_figure = st.composite_figure("column", controls_height)
    ax_controls = st.add_axes_inches(
        controls_figure,
        [
            controls_left,
            controls_bottom,
            controls_panel_side,
            controls_panel_side,
        ],
    )
    ax_gap_lag = st.add_axes_inches(
        controls_figure,
        [
            controls_right,
            controls_bottom,
            controls_panel_side,
            controls_panel_side,
        ],
    )
    controls_axes = np.asarray([ax_controls, ax_gap_lag], dtype=object)
    controls_legend_axis = st.add_axes_inches(
        controls_figure,
        [
            controls_left,
            1.92,
            controls_right + controls_panel_side - controls_left,
            0.39,
        ],
    )
    controls_legend_axis.set_axis_off()
    assert_square_panels(
        controls_figure,
        controls_axes,
        name="fig_scalar_controls",
    )
    scalar_rows = [
        by_key[key]
        for key in (
            "fixed_b",
            "activity_matched",
            "gap_matched",
            "independent_selection",
        )
    ]
    draw_forest(
        ax_controls,
        scalar_rows,
        color=st.COLLECTIVE,
        marker="o",
        labels={
            "fixed_b": r"fixed $B$",
            "activity_matched": "mean\nactivity",
            "gap_matched": "matched\nrate",
            "independent_selection": "selected\nsettings",
        },
        seed_values=scalar_seed_values,
    )
    for tick_label in ax_controls.get_yticklabels():
        if "\n" in tick_label.get_text():
            tick_label.set_linespacing(0.72)
    ax_controls.set_xlabel("STM gain", labelpad=2.0)
    draw_gap_lags(ax_gap_lag, lag_rows)
    controls_handles, controls_labels = ax_gap_lag.get_legend_handles_labels()
    controls_legend = st.legend(
        controls_legend_axis,
        lw=st.LEGEND_FRAMEWIDTH,
        handles=controls_handles,
        labels=controls_labels,
        loc="center",
        ncol=3,
        fontsize=FIG_TEXT_SIZE,
        handlelength=1.00,
        handletextpad=0.28,
        columnspacing=0.55,
        borderpad=0.18,
        labelspacing=0.08,
        frameon=True,
    )
    controls_legend.set_zorder(6)
    add_panel_letters(
        controls_figure,
        (
            (
                "a",
                controls_left,
                controls_bottom + controls_panel_side + 0.10,
            ),
            (
                "b",
                controls_right,
                controls_bottom + controls_panel_side + 0.10,
            ),
        ),
    )
    ax_controls.get_xticklabels()[-1].set_horizontalalignment("right")
    # Keep the left y-tick labels inside the inter-panel gutter without
    # shifting or shrinking either square panel.
    ax_gap_lag.get_xticklabels()[0].set_horizontalalignment("left")
    ax_gap_lag.get_xticklabels()[-1].set_horizontalalignment("right")
    finish_figure(
        controls_figure,
        np.asarray([*controls_axes, controls_legend_axis], dtype=object),
        name="fig_scalar_controls",
        output=SCALAR_CONTROLS_OUTPUT,
    )


if __name__ == "__main__":
    main()

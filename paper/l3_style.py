"""Shared publication style for every manuscript figure.

The visual grammar follows the supplied Physical Review A reference while the
physical canvases follow the Quantum text block.  Figures are authored at
their final placement width, use one stable semantic palette, and remain
readable at 100% PDF zoom.  Dense figures may simplify labels, but no visible
artist may fall below :data:`MIN_FONT_SIZE`.
"""

from __future__ import annotations

import colorsys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import AutoMinorLocator


# Legacy nominal canvases retained for figures outside the present manuscript
# composite set.
WIDTH_FULL = 7.06
WIDTH_COLUMN = 3.40
HEIGHT_RATIO = 0.357

# Exact natural-size canvases for the two-column Quantum template used by the
# manuscript.  The class sets 20 mm side margins on A4 paper and a 20 TeX-point
# inter-column gap.  Figures 2--4 are authored on these canvases and inserted
# without any LaTeX-side scaling, cropping, or panel assembly.
QUANTUM_TEXT_WIDTH = 170.0 / 25.4
QUANTUM_COLUMN_SEP = 20.0 / 72.27
QUANTUM_COLUMN_WIDTH = (
    QUANTUM_TEXT_WIDTH - QUANTUM_COLUMN_SEP
) / 2.0

# Final-size typography contract.  Quantum's body text is approximately
# 10.95 pt, so primary plot labels sit near 9.5--10 pt and even dense
# annotations stay above 8.5 pt.  This keeps multi-panel figures compact while
# making their typography visibly continuous with the surrounding manuscript.
MIN_FONT_SIZE = 8.60
FONT_SIZE = 9.30
AXIS_LABEL_SIZE = 9.60
TITLE_SIZE = 9.85
TICK_SIZE = 8.75
LEGEND_SIZE = 8.65
PANEL_LABEL_SIZE = 9.85

# Shared final-size stroke and marker system. Figures 2--4 use these exact
# physical point sizes so equal visual roles have equal weight in the PDF.
AXIS_LINEWIDTH = 0.90
DATA_LINEWIDTH = 1.40
MARKER_SIZE = 4.05
MARKER_EDGEWIDTH = 0.70
ERROR_LINEWIDTH = 0.72
ERROR_CAPSIZE = 1.80
REFERENCE_LINEWIDTH = 0.70
LEGEND_FRAMEWIDTH = 0.58

# Permanent dissipator palette.  These semantic colors are the single source
# of truth whenever color identifies a physical design.  Generic plotting
# colors remain available below for metric- or annotation-specific accents.
INK = "#1A1A1A"
GRID = "#ECEAEC"
GRID_MAJOR = "#D9DDE1"
GRID_MINOR = "#EEF0F2"
GRID_MAJOR_DASH = (0, (2.4, 2.0))
GRID_MINOR_DASH = (0, (1.4, 2.2))
GRID_MAJOR_LINEWIDTH = 0.38
GRID_MINOR_LINEWIDTH = 0.24
GRAY = "#808080"
DARK_VIOLET = "#250C50"
COLLECTIVE = "#6C236B"
PLUM = "#933461"
CORAL = "#C35149"
ORANGE = "#E58336"
GOLD = "#F2C34D"

# Dissipator identities used throughout the manuscript.
UNIFORM_LOCAL = "#5A5A5A"
# Direct binary local--collective comparisons use the original orange--purple
# pair.  Broader multi-design views keep UNIFORM_LOCAL gray so Figure 2 can
# retain its explicit neutral-reference contract.
LOCAL_CONTRAST = ORANGE
UNEQUAL_LOCAL = "#7F3B08"
PAIR_LOSS = "#E58336"
GAIN_LOSS = "#933461"
EXCHANGE_ASSISTED = "#F2C34D"
DEPHASING = "#B1B4B8"

# Figure 2 deliberately highlights only the reference and the best mean in
# each panel.  This neutral is reserved for its non-highlighted competitors
# and for genuinely external baselines.
NEUTRAL_DESIGN = "#989898"
PROFILE_ACCENT = UNEQUAL_LOCAL
# One print-safe signal color is reserved for selected optima.  It is not part
# of the categorical palette and must always use the ordinary point size.
OPTIMUM = "#C62828"

BLUE = COLLECTIVE
RED = CORAL
GREEN = GOLD
PURPLE = PLUM
TEAL = DARK_VIOLET
CYCLE = [
    COLLECTIVE,
    UNIFORM_LOCAL,
    UNEQUAL_LOCAL,
    PAIR_LOSS,
    GAIN_LOSS,
    EXCHANGE_ASSISTED,
    DEPHASING,
]
DIVERGING = "coolwarm"


def distance_faded_colors(
    color: str,
    values,
    *,
    center: float | None = None,
    near_alpha: float = 0.22,
    far_alpha: float = 0.035,
    power: float = 0.75,
) -> np.ndarray:
    """Return per-point RGBA colors that fade with distance from a center.

    The mapping is deterministic and monotone: observations closest to the
    displayed mean receive ``near_alpha`` and the most distant observation
    receives ``far_alpha``.  This keeps raw seeds visible without allowing an
    extreme realization to dominate the aggregate curve or interval.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("distance-faded values must be a finite 1D array")
    if not 0 <= far_alpha <= near_alpha <= 1:
        raise ValueError("seed alpha bounds must satisfy 0 <= far <= near <= 1")
    if power <= 0:
        raise ValueError("seed alpha power must be positive")
    reference = float(np.mean(array) if center is None else center)
    distance = np.abs(array - reference)
    maximum = float(np.max(distance)) if len(distance) else 0.0
    normalized = np.zeros_like(distance) if maximum == 0 else distance / maximum
    alphas = far_alpha + (near_alpha - far_alpha) * (1 - normalized) ** power
    colors = np.tile(np.asarray((*to_rgb(color), 1.0)), (len(array), 1))
    colors[:, 3] = alphas
    return colors


RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
    "font.size": FONT_SIZE,
    "mathtext.fontset": "cm",
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": TITLE_SIZE,
    "axes.facecolor": "white",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.linewidth": AXIS_LINEWIDTH,
    "axes.labelpad": 2.5,
    "axes.titlepad": 3.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": mpl.cycler(color=CYCLE),
    "axes.formatter.limits": (-3, 4),
    "axes.formatter.useoffset": False,
    "axes.grid": False,
    "axes.grid.which": "major",
    "grid.color": GRID,
    "grid.linestyle": "--",
    "grid.linewidth": 0.26,
    "grid.alpha": 1.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.86,
    "ytick.major.width": 0.86,
    "xtick.minor.width": 0.54,
    "ytick.minor.width": 0.54,
    "xtick.major.pad": 2.4,
    "ytick.major.pad": 2.4,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "lines.linewidth": DATA_LINEWIDTH,
    "lines.markersize": MARKER_SIZE,
    "lines.markeredgewidth": MARKER_EDGEWIDTH,
    "lines.solid_capstyle": "projecting",
    "lines.dash_capstyle": "butt",
    "lines.dashed_pattern": [3.7, 1.6],
    "lines.dotted_pattern": [1.0, 1.65],
    "lines.dashdot_pattern": [6.4, 1.6, 1.0, 1.6],
    "scatter.marker": "o",
    "errorbar.capsize": ERROR_CAPSIZE,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.facecolor": "white",
    "legend.edgecolor": "#9E9E9E",
    "legend.fancybox": False,
    "legend.borderpad": 0.40,
    "legend.labelspacing": 0.24,
    "legend.handlelength": 1.55,
    "legend.handletextpad": 0.45,
    "legend.borderaxespad": 0.45,
    "legend.columnspacing": 0.9,
    "legend.markerscale": 1.0,
    "figure.dpi": 110,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.02,
    "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.03,
    "figure.constrained_layout.wspace": 0.03,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "savefig.pad_inches": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "pdf.compression": 6,
    "svg.fonttype": "none",
    "path.simplify": True,
    "path.simplify_threshold": 0.111111,
    "agg.path.chunksize": 20000,
}

TIMES = {
    "font.serif": [
        "STIXGeneral",
        "Times New Roman",
        "Nimbus Roman",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
}


def use(times: bool = False) -> None:
    """Apply the L3 style globally."""
    plt.rcParams.update(RC)
    if times:
        plt.rcParams.update(TIMES)


def style_axis(
    ax,
    grid_axis: str = "both",
    *,
    minor_axis: str | None = None,
    minor_grid: bool = False,
) -> None:
    """Apply the common frame and quiet dashed background grid."""
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(
        True,
        which="major",
        axis=grid_axis,
        color=GRID_MAJOR,
        linestyle=GRID_MAJOR_DASH,
        linewidth=GRID_MAJOR_LINEWIDTH,
        alpha=1.0,
    )
    if minor_axis:
        if minor_axis in ("x", "both") and ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if minor_axis in ("y", "both") and ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if minor_grid:
        ax.grid(
            True,
            which="minor",
            axis=minor_axis or grid_axis,
            color=GRID_MINOR,
            linestyle=GRID_MINOR_DASH,
            linewidth=GRID_MINOR_LINEWIDTH,
            alpha=1.0,
        )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(RC["axes.linewidth"])


def figsize(width: str | float = "full", ratio: float = HEIGHT_RATIO):
    """Return a physical canvas size in inches."""
    resolved = {"full": WIDTH_FULL, "column": WIDTH_COLUMN}.get(width, width)
    return (resolved, resolved * ratio)


def composite_figure(placement: str, height: float):
    """Create a final-size manuscript composite with no downstream scaling."""
    widths = {
        "full": QUANTUM_TEXT_WIDTH,
        "column": QUANTUM_COLUMN_WIDTH,
    }
    if placement not in widths:
        raise ValueError(f"unknown composite placement: {placement!r}")
    width = widths[placement]
    fig = plt.figure(
        figsize=(width, height),
        constrained_layout=False,
    )
    fig._qrc_natural_placement = placement
    fig._qrc_expected_canvas = (width, height)
    return fig


def add_axes_inches(fig, rect, **kwargs):
    """Add an axis whose complete geometry is specified in physical inches."""
    figure_width, figure_height = fig.get_size_inches()
    left, bottom, width, height = rect
    return fig.add_axes(
        [
            left / figure_width,
            bottom / figure_height,
            width / figure_width,
            height / figure_height,
        ],
        **kwargs,
    )


def tints(base: str, n: int, reverse: bool = False):
    """Return ``n`` near-white-to-base shades, linear in RGB."""
    r, g, b = to_rgb(base)
    h, _, _ = colorsys.rgb_to_hls(r, g, b)
    low = np.array(colorsys.hls_to_rgb(h, 0.945, 0.073))
    high = np.array((r, g, b))
    colors = low + (high - low) * np.linspace(0, 1, n)[:, None]
    return list(colors[::-1] if reverse else colors)


def cmap_from(base: str, name: str = "tint"):
    """Create a continuous single-hue colormap."""
    return LinearSegmentedColormap.from_list(name, tints(base, 32))


def legend(ax, lw: float = 0.70, **kwargs):
    """Create an in-axes legend with the L3 frame."""
    leg = ax.legend(**kwargs)
    leg.get_frame().set_linewidth(lw)
    leg.get_frame().set_edgecolor("#9E9E9E")
    return leg


def thin_ticks(axis, keep: int = 2, offset: int = 0) -> None:
    """Hide all but every ``keep``-th tick label."""
    for index, tick in enumerate(axis.get_ticklabels()):
        if (index - offset) % keep:
            tick.set_visible(False)


def panel_labels(
    fig,
    axes,
    labels: str = "abc",
    *,
    x: float = 0.0,
    y: float = 1.025,
    fontsize: float | None = None,
):
    """Place consistent panel letters just above the upper-left plot corner."""
    fig.canvas.draw()
    return [
        ax.text(
            x,
            y,
            f"({label})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE if fontsize is None else fontsize,
            color=INK,
            clip_on=False,
        )
        for ax, label in zip(np.ravel(axes), labels)
    ]


def audit_figure(
    fig,
    name: str,
    *,
    axes=None,
    boundary_tolerance_px: float = 1.5,
    overlap_fraction: float = 0.12,
    font_floor: float | None = None,
) -> None:
    """Fail when visible text or adjacent subplot decorations overlap.

    The manuscript figures are small enough that visual collisions can be
    subtle in a standalone PDF and obvious only after two-column placement.
    This audit therefore checks the rendered artist geometry rather than the
    nominal layout parameters:

    * every visible text box must remain inside the authored canvas;
    * no two visible text boxes may overlap materially;
    * complete subplot decoration boxes (ticks, labels, and panel letters)
      may not intrude into one another; and
    * legends must remain inside their owning plotting axes.

    The final manuscript is still rendered and inspected page by page because
    no geometric test can determine whether a legend hides an important data
    feature.
    """

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_box = fig.bbox
    problems: list[str] = []
    required_font_floor = MIN_FONT_SIZE if font_floor is None else font_floor

    expected_canvas = getattr(fig, "_qrc_expected_canvas", None)
    if expected_canvas is not None and not np.allclose(
        fig.get_size_inches(),
        expected_canvas,
        rtol=0,
        atol=1e-9,
    ):
        problems.append(
            "natural-size canvas changed: "
            f"expected {expected_canvas}, found {tuple(fig.get_size_inches())}"
        )

    texts = []
    for artist in fig.findobj(mpl.text.Text):
        if (
            not artist.get_visible()
            or not artist.get_text()
            or not artist.get_text().strip()
        ):
            continue
        box = artist.get_window_extent(renderer)
        if not np.all(np.isfinite(box.extents)) or box.width <= 0 or box.height <= 0:
            continue
        texts.append((artist, box))
        if float(artist.get_fontsize()) < required_font_floor - 1e-9:
            problems.append(
                f"text below font floor: {artist.get_text()!r} uses "
                f"{float(artist.get_fontsize()):.2f} pt "
                f"(minimum {required_font_floor:.2f} pt)"
            )
        if (
            box.x0 < figure_box.x0 - boundary_tolerance_px
            or box.y0 < figure_box.y0 - boundary_tolerance_px
            or box.x1 > figure_box.x1 + boundary_tolerance_px
            or box.y1 > figure_box.y1 + boundary_tolerance_px
        ):
            problems.append(
                f"text outside canvas: {artist.get_text()!r} "
                f"at {tuple(round(value, 1) for value in box.extents)}"
            )

    def intersection(first, second):
        width = min(first.x1, second.x1) - max(first.x0, second.x0)
        height = min(first.y1, second.y1) - max(first.y0, second.y0)
        return max(0.0, width), max(0.0, height)

    for index, (first_artist, first_box) in enumerate(texts):
        first_area = first_box.width * first_box.height
        for second_artist, second_box in texts[index + 1 :]:
            width, height = intersection(first_box, second_box)
            if width <= 1.0 or height <= 1.0:
                continue
            second_area = second_box.width * second_box.height
            fraction = width * height / min(first_area, second_area)
            if fraction > overlap_fraction:
                problems.append(
                    "text overlap: "
                    f"{first_artist.get_text()!r} with "
                    f"{second_artist.get_text()!r} "
                    f"({100 * fraction:.1f}% of smaller box)"
                )

    panel_axes = list(fig.axes if axes is None else np.ravel(axes))
    decoration_boxes = []
    for axis in panel_axes:
        if not axis.get_visible():
            continue
        box = axis.get_tightbbox(renderer)
        if box is not None:
            decoration_boxes.append((axis, box))
        legend_artist = axis.get_legend()
        if legend_artist is None:
            continue
        legend_box = legend_artist.get_window_extent(renderer)
        axis_box = axis.get_window_extent(renderer)
        if (
            legend_box.x0 < axis_box.x0 - boundary_tolerance_px
            or legend_box.y0 < axis_box.y0 - boundary_tolerance_px
            or legend_box.x1 > axis_box.x1 + boundary_tolerance_px
            or legend_box.y1 > axis_box.y1 + boundary_tolerance_px
        ):
            problems.append("legend extends outside its plotting axes")

    for index, (_, first_box) in enumerate(decoration_boxes):
        for _, second_box in decoration_boxes[index + 1 :]:
            width, height = intersection(first_box, second_box)
            if width > 1.0 and height > 1.0:
                problems.append(
                    "subplot decorations overlap by "
                    f"{width:.1f} x {height:.1f} px"
                )

    if problems:
        preview = "\n  - ".join(problems[:16])
        if len(problems) > 16:
            preview += f"\n  - ... and {len(problems) - 16} more"
        raise RuntimeError(f"{name} failed the overlap audit:\n  - {preview}")


def save(fig, stem: str, formats=("pdf",)):
    """Write the requested vector/raster formats."""
    outputs = []
    for file_format in formats:
        output = f"{stem}.{file_format}"
        fig.savefig(
            output,
            format=file_format,
            bbox_inches=None,
            pad_inches=0,
        )
        outputs.append(output)
    return outputs

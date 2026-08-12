#!/usr/bin/env python3
"""
Figure 1 in the L3 house style (l3_style.py).

    python fig1_L3.py    ->  fig1_L3l.pdf / .png   (6.69 x 4.24 in)

Two module defaults are overridden on purpose, both noted inline:
constrained_layout (this figure places its axes explicitly) and
savefig.bbox="tight" (which would crop the authored Quantum-width canvas).

Layout: top row = (a) the prior one-dimensional local-relaxation view, (b) its
extension to alternative dissipative designs, and (c) several rate profiles
within one fixed collective jump family; bottom row = (d) the design space at
full width. Panels (a) and (b) use the same illustrative response function, so
the black local curve in (b) is panel (a). The solid collective curve in (c) is
likewise identical to the highlighted collective curve in (b); three lighter
line styles illustrate profile changes within that family.

A collision check runs at the end and must report zero problems.
"""

import itertools

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox, ScaledTranslation

import l3_style as st

st.use()
plt.rcParams.update({
    "figure.constrained_layout.use": False,   # axes are placed by hand below
    "savefig.bbox": None,                     # keep the authored 7.06 in width
    # the gallery renders in STIXGeneral (cmr10 is absent there); pin it so this
    # figure matches the gallery on any machine
    "font.serif": ["STIXGeneral", "cmr10", "CMU Serif", "DejaVu Serif"],
})

INK = st.INK
MUTED = "#6F7379"

# Figure-specific semantic tones from the shared aubergine-to-gold spectrum.
# Purple carries the process-family/collective axis, orange the rate-profile axis,
# and neutral gray the contextual designs.
BLUE = st.COLLECTIVE                    # collective relaxation / process-family axis
ORANGE = st.ORANGE                      # original rate-profile orange
LOCAL = st.LOCAL_CONTRAST               # local side of local--collective contrast
OPTIMUM = st.OPTIMUM                    # selected optimum, ordinary point size
CONTEXT = st.GRAY                       # secondary process-family curves
CELL = "#E8E9EB"                        # untested family-profile combinations
CELL_E = "#BEC2C7"
GREY_B = "#B1B4B8"                     # bath grey

# Figure 1 is authored at the exact Quantum text width and inserted at natural
# size, like Figures 2 and 3.  The top row therefore uses their final-size
# typography and stroke contract directly rather than compensating for later
# LaTeX scaling.  Panel (d) stays slightly quieter because it is denser.
F_TITLE = 10.40
F_LAB = 10.00
F_TICK = 9.10
F_LEG = 9.20
F_NAME = 9.10
F_SYM = 8.75
F_AX = F_LAB
F_KEY = 8.75
F_COORD = 11.60


def description_box(pad=0.12):
    """Opaque white backing for explanatory text placed over plotted marks."""
    return dict(boxstyle=f"round,pad={pad}", fc="white", ec="none", alpha=1.0)


# --- one strength response per design ---------------------------------------
# These smooth curves are schematic summaries, not fitted response functions.
# Their rounded peaks and heights visually anchor the design-specific strength
# study reported exactly in Fig. 8; the left/right widths are drawing
# parameters. Panel (c) below is a separate categorical experiment.
DENSE_DOTS = (0, (1.0, 1.1))
SPARSE_DOTS = (0, (1.0, 3.0))
DASH_DOT = (0, (4.0, 1.5, 1.0, 1.5))
SHORT_DASH = (0, (3.0, 2.0))
DESIGNS = [
    #  name          peak   top     wl   wr    colour    lw    ls
    ("thermal",      0.10, 11.87, 2.00, 1.50, "gain_loss", 1.10, DENSE_DOTS),
    ("exchange",     0.10, 11.77, 1.60, 1.20, "exchange", 1.10, SPARSE_DOTS),
    ("unequal",      0.25, 11.39, 2.45, 1.95, "unequal", 1.10, DASH_DOT),
    ("pair",         0.25, 11.11, 1.90, 1.45, "pair", 1.10, SHORT_DASH),
    ("local",        0.25, 11.32, 2.20, 1.70, "local",  1.65, "-"),
    ("collective",   8.00, 14.17, 5.50, 3.20, "collective", 1.65, "-"),
]
CYCOL = {
    "local": LOCAL,
    "collective": BLUE,
    "unequal": st.UNEQUAL_LOCAL,
    "pair": st.PAIR_LOSS,
    "gain_loss": st.GAIN_LOSS,
    "exchange": st.EXCHANGE_ASSISTED,
    "gray": CONTEXT,
}
GAMMA = np.logspace(np.log10(0.05), np.log10(32), 400)


def response(g, name, peak, top, wl, wr):
    d = np.log(g) - np.log(peak)
    if name in ("local", "collective"):
        shape = np.exp(-(d / np.where(d < 0, wl, wr)) ** 2)
        return top * (0.10 + 0.90 * shape)

    # The four grey guides are deliberately different schematic response
    # classes, not fits.  Each is normalized to its stated peak and decays at
    # high gamma, while the distinct shoulder and tail laws keep the designs
    # visually separable across the full logarithmic axis.
    left = np.clip(-d, 0.0, None)
    right = np.clip(d, 0.0, None)
    if name == "thermal":                    # broad peak, slow algebraic tail
        rise = np.exp(-(left / wl) ** 1.4)
        fall = 1.0 / (1.0 + (right / (0.75 * wr)) ** 1.35)
        floor = 0.035
    elif name == "exchange":                 # narrow peak, early sharp decay
        rise = np.exp(-(left / (0.75 * wl)) ** 3.0)
        fall = np.exp(-(right / (0.65 * wr)) ** 1.05)
        floor = 0.020
    elif name == "unequal":                  # two-stage shoulder and long tail
        rise = 1.0 / (1.0 + (left / (0.75 * wl)) ** 2.8)
        fall = (
            0.62 * np.exp(-(right / (0.55 * wr)) ** 1.6)
            + 0.38 * np.exp(-(right / (2.30 * wr)) ** 4.0)
        )
        floor = 0.070
    elif name == "pair":                     # flat shoulder, delayed cliff
        rise = np.exp(-(left / (1.10 * wl)) ** 1.2)
        fall = 1.0 / (1.0 + (right / (0.95 * wr)) ** 4.5)
        floor = 0.025
    else:
        raise KeyError(name)

    shape = np.where(d < 0, rise, fall)
    return top * (floor + (1.0 - floor) * shape)


def curve(name):
    for n, peak, top, wl, wr, _c, _lw, _ls in DESIGNS:
        if n == name:
            return response(GAMMA, name, peak, top, wl, wr), peak, top
    raise KeyError(name)


# --- profiles within one fixed jump family ----------------------------------
# Figure 1 is a design-space schematic rather than a data figure.  Panel (c)
# therefore keeps the solid uniform-profile collective curve from panel (b)
# exactly and adds three progressively lighter illustrative response profiles.
# Their line styles, peak heights, and widths are drawing parameters only.
COLLECTIVE_PROFILE_VARIANTS = [
    # label       peak  top    wl    wr    line style
    ("profile 1", 5.20, 12.10, 5.10, 3.00, SHORT_DASH),
    ("profile 2", 3.60, 10.25, 4.65, 2.75, DENSE_DOTS),
    ("profile 3", 2.40,  8.45, 4.15, 2.45, DASH_DOT),
]

FIGH = 4.24
fig = st.composite_figure("full", FIGH)
FIGW = float(fig.get_figwidth())

# Panels (a)--(c) now use the same near-square physical plot boxes as the
# empirical small panels in Figure 2.  Explicit inch geometry makes their
# apparent size independent of the figure canvas and prevents the former
# shallow-strip appearance.
TOP_PANEL_X = (0.40, 2.532, 4.664)
TOP_PANEL_W = 1.68
TOP_PANEL_Y = 3.02
TOP_PANEL_H = 1.02
ax_a = st.add_axes_inches(
    fig, [TOP_PANEL_X[0], TOP_PANEL_Y, TOP_PANEL_W, TOP_PANEL_H]
)
ax_b2 = st.add_axes_inches(
    fig, [TOP_PANEL_X[1], TOP_PANEL_Y, TOP_PANEL_W, TOP_PANEL_H]
)
ax_c3 = st.add_axes_inches(
    fig, [TOP_PANEL_X[2], TOP_PANEL_Y, TOP_PANEL_W, TOP_PANEL_H]
)

# Panel (d) remains full-width and keeps its established internal geometry.
DESIGN_PANEL_X = 0.13
DESIGN_PANEL_W = 6.38
ax_b = st.add_axes_inches(fig, [DESIGN_PANEL_X, 0.05, DESIGN_PANEL_W, 2.30])

TOP_TITLE_Y = 4.08 / FIGH
for index, left in enumerate(TOP_PANEL_X):
    fig.text(
        (left - 0.30) / FIGW,
        TOP_TITLE_Y,
        f"({chr(ord('a') + index)})",
        fontsize=F_TITLE,
        color=INK,
        va="bottom",
        ha="left",
    )
fig.text(
    DESIGN_PANEL_X / FIGW,
    2.42 / FIGH,
    "(d)",
    fontsize=F_TITLE,
    color=INK,
    va="bottom",
    ha="left",
)


# ============================================== (a) the prior scalar scan ====
y_loc, g_loc, top_loc = curve("local")
ax_a.plot(GAMMA, y_loc, lw=1.65, color=LOCAL, zorder=4)
ax_a.plot([g_loc], [top_loc], marker="o", ms=4.45, mfc=OPTIMUM, mec=OPTIMUM,
          mew=0.70, zorder=5)
ax_a.text(0.057, 17.2, "local relaxation", color=LOCAL, fontsize=F_LEG, ha="left",
          va="center", zorder=8, bbox=description_box())
ax_a.set_xscale("log")
ax_a.set_xlim(0.05, 32)
ax_a.set_ylim(0, 19)
ax_a.set_yticks([0, 7, 14])
ax_a.set_xticks([0.1, 1, 10])
ax_a.set_xticklabels(["0.1", "1", "10"])
ax_a.tick_params(labelsize=F_TICK)
ax_a.set_xlabel("damping strength $\\gamma$")
ax_a.set_ylabel("schematic task score", fontsize=F_KEY)
ax_a.set_facecolor("white")
st.style_axis(ax_a, "both", minor_axis="both", minor_grid=True)


# ============================================== (b) the dissipator axis ======
for name, peak, top_, wl, wr, col, lw, ls in DESIGNS:
    c = CYCOL[col]
    ax_b2.plot(GAMMA, response(GAMMA, name, peak, top_, wl, wr),
               lw=lw, color=c,
               ls=ls, alpha=1.0 if name in ("local", "collective") else 0.92,
               zorder=4 if lw > 1.5 else 2)
ax_b2.plot([0.25], [11.32], marker="o", ms=4.45, mfc=OPTIMUM, mec=OPTIMUM, mew=0.70,
           zorder=6)
ax_b2.plot([8.0], [14.17], marker="o", ms=4.45, mfc=OPTIMUM, mec=OPTIMUM, mew=0.70,
           zorder=6)
ax_b2.text(0.055, 20.3, "local relaxation", color=LOCAL, fontsize=F_LEG,
           ha="left", va="center", zorder=8, bbox=description_box())
ax_b2.text(0.94, 17.2, "collective relaxation", transform=ax_b2.get_yaxis_transform(),
           color=BLUE, fontsize=F_LEG, ha="right", va="center", zorder=8,
           bbox=description_box(), clip_on=True)
ax_b2.set_xscale("log")
ax_b2.set_xlim(0.05, 32)
ax_b2.set_ylim(0, 22.5)
ax_b2.set_xticks([0.1, 1, 10])
ax_b2.set_xticklabels(["0.1", "1", "10"])
ax_b2.set_yticks([0, 7, 14])
ax_b2.tick_params(labelsize=F_TICK)
ax_b2.set_xlabel("damping strength $\\gamma$")
ax_b2.set_ylabel("schematic task score", fontsize=F_KEY)
ax_b2.set_facecolor("white")
st.style_axis(ax_b2, "both", minor_axis="both", minor_grid=True)

# ========================================= (c) profiles within collective ===
# Reuse the panel-(b) collective response exactly as the uniform-profile
# reference, then vary only the schematic profile within that family.
y_collective, g_collective, top_collective = curve("collective")
ax_c3.plot(
    GAMMA,
    y_collective,
    lw=1.65,
    color=BLUE,
    ls="-",
    zorder=5,
)
ax_c3.plot(
    [g_collective],
    [top_collective],
    marker="o",
    ms=4.45,
    mfc=OPTIMUM,
    mec=OPTIMUM,
    mew=0.70,
    zorder=7,
)
profile_tones = st.tints(BLUE, 7)
for index, (_label, peak, top_, wl, wr, line_style) in enumerate(
    COLLECTIVE_PROFILE_VARIANTS
):
    tone = profile_tones[5 - index]
    values = response(GAMMA, "collective", peak, top_, wl, wr)
    ax_c3.plot(
        GAMMA,
        values,
        lw=1.42 - 0.08 * index,
        color=tone,
        ls=line_style,
        zorder=4 - index,
    )
    ax_c3.plot(
        [peak],
        [top_],
        marker="o",
        ms=3.90,
        mfc=tone,
        mec=tone,
        mew=0.60,
        zorder=6,
    )

ax_c3.text(0.055, 20.3, "collective relaxation", color=BLUE, fontsize=F_LEG,
           ha="left", va="center", zorder=8, bbox=description_box())
ax_c3.set_xscale("log")
ax_c3.set_xlim(0.05, 32)
ax_c3.set_ylim(0, 22.5)
ax_c3.set_xticks([0.1, 1, 10])
ax_c3.set_xticklabels(["0.1", "1", "10"])
ax_c3.set_yticks([0, 7, 14])
ax_c3.tick_params(labelsize=F_TICK)
ax_c3.set_xlabel("damping strength $\\gamma$")
ax_c3.set_ylabel("schematic task score", fontsize=F_KEY)
ax_c3.set_facecolor("white")
st.style_axis(ax_c3, "both", minor_axis="both", minor_grid=True)

# ==================================================== (d) design space ======
XMIN, XMAX = -2.00, 7.15
YMIN, YMAX = -2.72, 6.20
ax_b.set_xlim(XMIN, XMAX)
ax_b.set_ylim(YMIN, YMAX)
ax_b.axis("off")
ax_b.grid(False)

# aspect correction: vertical distances are given in x-units and converted
pos = ax_b.get_position()
CM_X = pos.width * FIGW * 2.54 / (XMAX - XMIN)
CM_Y = pos.height * FIGH * 2.54 / (YMAX - YMIN)
AR = CM_X / CM_Y


def vy(d):
    """vertical distance d (in x-units) expressed in y-units"""
    return d * AR


def qubit(x, y, ms=4.0):
    ax_b.plot([x], [y], marker="o", ms=ms, mfc="white", mec=INK, mew=0.8,
              zorder=6)


def bath(x, y, w, color=GREY_B):
    h = vy(0.055)
    ax_b.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                  boxstyle="round,pad=0,rounding_size=0.02",
                                  fc=color, ec="none", zorder=3))


def arrow(p0, p1, color=INK, lw=0.8, rad=0.0, scale=4.6, sa=0.8, sb=0.8):
    ax_b.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>",
                                   mutation_scale=scale, lw=lw, color=color,
                                   connectionstyle=f"arc3,rad={rad}",
                                   shrinkA=sa, shrinkB=sb, zorder=5))


cols = [
    ("local",          r"$\sigma_i^-$"),
    ("collective",     r"$\Sigma_i\, c_i\,\sigma_i^-$"),
    ("pair loss",      r"$\sigma_i^-\sigma_j^-$"),
    ("exchange",       r"$\sigma_i^-,\ \sigma_i^-\sigma_j^+$"),
    ("local gain/loss", r"$\sigma_i^-,\ \sigma_i^+$"),
    ("dephasing",      r"$\sigma_i^z$"),
]
rows = [
    ("uniform",        r"$\gamma_i=\gamma$"),
    ("unequal",        r"$\gamma_i\neq\gamma_j$"),
    ("learned",        r"$\gamma_i$ fitted"),
    ("input-adaptive", r"$\gamma_i(s)$"),
]
# Lower the matrix slightly to reserve a clean continuation row below the four
# illustrated profiles.  Together with a matching continuation column after
# dephasing, this makes both design axes visibly open-ended.
GRID_DY = -0.70

# --- broader search space ---------------------------------------------------
# Every neutral cell is a possible family-profile combination outside the two
# highlighted coordinate slices.  Those cells remain deliberately neutral so
# the overview does not promote isolated cross-slice controls to a third slice.
PATHWAY_CELL = st.tints(BLUE, 6)[2]
PROFILE_CELL = st.tints(ORANGE, 6)[2]
CELL_ROUND = 0.055
CELL_X_PAD = 0.01
CELL_WIDTH = 1.0 - 2.0 * CELL_X_PAD
# The citation-bearing prior-work cell needs slightly more horizontal room
# than the other matrix cells.  Borrow that room from the existing label
# gutter instead of shrinking the shared final-size typography.
FIRST_CELL_LEFT = -0.06
FIRST_CELL_RIGHT = 1.0 - CELL_X_PAD
FIRST_CELL_CENTER = 0.5 * (FIRST_CELL_LEFT + FIRST_CELL_RIGHT)
for r in range(4):
    for c in range(6):
        x0 = FIRST_CELL_LEFT if c == 0 else c + CELL_X_PAD
        x1 = c + 1.0 - CELL_X_PAD
        y0, y1 = 3 - r + 0.06 + GRID_DY, 3 - r + 0.94 + GRID_DY
        if r == 0 and c == 0:
            fc, ec, lw = PATHWAY_CELL, LOCAL, 0.85
        elif r == 0:
            fc, ec, lw = PATHWAY_CELL, BLUE, 0.78
        elif c == 0:
            fc, ec, lw = PROFILE_CELL, ORANGE, 0.78
        else:
            fc, ec, lw = to_rgba(CELL, 0.42), to_rgba(CELL_E, 0.58), 0.55
        ax_b.add_patch(FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={CELL_ROUND}",
            fc=fc, ec=ec, lw=lw, zorder=2
        ))

# --- open-ended design space ------------------------------------------------
# Dashed ghost cells extend the matrix by one column and one row without
# enclosing it in a finite outer boundary.  They are orientation cues, not
# completed family-profile evaluations.
GHOST_E = "#9E9D98"
GHOST_DASH = (0, (3.0, 2.5))
ghost_patches = []
for r in range(4):
    y0, y1 = 3 - r + 0.06 + GRID_DY, 3 - r + 0.94 + GRID_DY
    patch = FancyBboxPatch(
        (6.0 + CELL_X_PAD, y0), CELL_WIDTH, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={CELL_ROUND}",
        fc="white", ec=GHOST_E, lw=0.90, ls=GHOST_DASH, zorder=2
    )
    ghost_patches.append(patch)
    ax_b.add_patch(patch)

GHOST_ROW_Y0 = -0.94 + GRID_DY
GHOST_ROW_Y1 = -0.06 + GRID_DY
for c in range(6):
    x0 = FIRST_CELL_LEFT if c == 0 else c + CELL_X_PAD
    x1 = c + 1.0 - CELL_X_PAD
    patch = FancyBboxPatch(
        (x0, GHOST_ROW_Y0), x1 - x0,
        GHOST_ROW_Y1 - GHOST_ROW_Y0,
        boxstyle=f"round,pad=0,rounding_size={CELL_ROUND}",
        fc="white", ec=GHOST_E, lw=0.90, ls=GHOST_DASH, zorder=2
    )
    ghost_patches.append(patch)
    ax_b.add_patch(patch)

family_ellipsis = ax_b.text(
    6.50, 3.50 + GRID_DY, r"$\cdots$",
    ha="center", va="center", fontsize=F_LAB, color=MUTED, zorder=6
)
profile_ellipsis = ax_b.text(
    FIRST_CELL_CENTER, 0.5 * (GHOST_ROW_Y0 + GHOST_ROW_Y1), r"$\vdots$",
    ha="center", va="center", fontsize=F_LAB, color=MUTED, zorder=6
)

# --- swept cross ------------------------------------------------------------
PAD = 0.03                    # margin between a band and the cells it encloses
for xy, wh, col in [
        ((FIRST_CELL_LEFT - PAD, 3.06 - PAD + GRID_DY),
         (6.0 - CELL_X_PAD - FIRST_CELL_LEFT + 2*PAD, 0.88 + 2*PAD), BLUE),
        ((FIRST_CELL_LEFT - PAD, 0.06 - PAD + GRID_DY),
         (FIRST_CELL_RIGHT - FIRST_CELL_LEFT + 2*PAD, 3.88 + 2*PAD), ORANGE)]:
    ax_b.add_patch(FancyBboxPatch(xy, *wh,
                                  boxstyle="round,pad=0,rounding_size=0.06",
                                  fc="none", ec=col, lw=1.20, zorder=4))

# --- prior work -------------------------------------------------------------
QUANTUM_VIOLET = "#53257F"
PRIOR_WORK_FONT_SIZE = 8.70
prior_work_prefix = ax_b.text(
    FIRST_CELL_CENTER, 3.5 + GRID_DY, "prior work ",
    ha="right", va="center", fontsize=PRIOR_WORK_FONT_SIZE,
    color="white", zorder=7,
)
prior_work_citation = ax_b.text(
    FIRST_CELL_CENTER, 3.5 + GRID_DY, "[6]",
    ha="left", va="center", fontsize=PRIOR_WORK_FONT_SIZE,
    color=QUANTUM_VIOLET, zorder=7,
)

# Offset the join between the two text runs so their combined bounding box,
# rather than the join itself, is centred in the cell.
fig.canvas.draw()
_prior_renderer = fig.canvas.get_renderer()
_prior_prefix_width = prior_work_prefix.get_window_extent(
    renderer=_prior_renderer
).width
_prior_citation_width = prior_work_citation.get_window_extent(
    renderer=_prior_renderer
).width
_prior_offset_inches = (
    _prior_prefix_width - _prior_citation_width
) / (2.0 * fig.dpi)
_prior_transform = ax_b.transData + ScaledTranslation(
    _prior_offset_inches, 0.0, fig.dpi_scale_trans
)
prior_work_prefix.set_transform(_prior_transform)
prior_work_citation.set_transform(_prior_transform)
prior_work_texts = (prior_work_prefix, prior_work_citation)

# --- icons ------------------------------------------------------------------
# Keep the family headers visually tied to the matrix rather than floating
# midway between the panel title and the swept row.
FAMILY_HEADER_DY = -0.42
cy = 5.68 + FAMILY_HEADER_DY
ICON_HALF = 0.205          # icon half-height in x-units
for c in range(6):
    cx = FIRST_CELL_CENTER if c == 0 else c + 0.5
    if c == 0:                                        # local
        for dx in (-0.20, 0.20):
            qubit(cx + dx, cy + vy(0.10))
            arrow((cx + dx, cy + vy(0.10)), (cx + dx, cy - vy(0.115)), sa=2.0)
            bath(cx + dx, cy - vy(0.175), 0.26)
    elif c == 1:                                      # collective
        # qubits a little wider apart and shorter arrows, so each arrow leaves
        # its own qubit and both stop symmetrically short of the bath centre
        bath(cx, cy - vy(0.175), 0.70)
        for dx in (-0.24, 0.24):
            qubit(cx + dx, cy + vy(0.10))
            arrow((cx + dx, cy + vy(0.10)),
                  (cx + 0.31 * dx, cy - vy(0.105)), scale=4.2, sa=2.0)
    elif c == 2:                                      # pair
        for dx in (-0.15, 0.15):
            qubit(cx + dx, cy + vy(0.11))
        ax_b.plot([cx - 0.15, cx + 0.15], [cy + vy(0.11)] * 2, lw=0.9,
                  color=INK, zorder=4)
        for dx in (-0.05, 0.05):
            arrow((cx + dx, cy + vy(0.045)), (cx + dx, cy - vy(0.115)), lw=0.7)
        bath(cx, cy - vy(0.175), 0.32)
    elif c == 3:                                      # exchange-assisted
        qubit(cx - 0.22, cy + vy(0.11))
        qubit(cx + 0.22, cy + vy(0.11))
        arrow((cx - 0.22, cy + vy(0.11)), (cx + 0.22, cy + vy(0.11)),
              rad=-0.55, lw=0.7, scale=4.0, sa=2.2, sb=2.2)
        arrow((cx - 0.22, cy + vy(0.11)), (cx - 0.22, cy - vy(0.115)), lw=0.7,
              sa=2.0)
        bath(cx - 0.22, cy - vy(0.175), 0.26)
    elif c == 4:                                      # local gain/loss
        qubit(cx, cy + vy(0.12))
        arrow((cx - 0.08, cy + vy(0.06)), (cx - 0.08, cy - vy(0.115)), lw=0.85)
        arrow((cx + 0.08, cy - vy(0.115)), (cx + 0.08, cy + vy(0.06)), lw=0.55,
              color=MUTED)
        bath(cx, cy - vy(0.175), 0.36)
    else:                                             # dephasing
        qubit(cx, cy + vy(0.10))
        t = np.linspace(0, 1, 90)
        for sgn in (-1, 1):
            ax_b.plot(cx + sgn * (0.10 + 0.20 * t),
                      cy + vy(0.10) + vy(0.030) * np.sin(3 * np.pi * t),
                      lw=0.8, color=INK, zorder=4)
        ax_b.plot([cx, cx], [cy + vy(0.085), cy - vy(0.14)], lw=0.7,
                  color=MUTED, ls=(0, (1.2, 1.2)), zorder=3)
        bath(cx, cy - vy(0.175), 0.36)

# --- headers and labels -----------------------------------------------------
for c, (name, sym) in enumerate(cols):
    cx = FIRST_CELL_CENTER if c == 0 else c + 0.5
    ax_b.text(cx, 4.58 + FAMILY_HEADER_DY, name,
              ha="center", va="bottom", fontsize=F_NAME,
              color=INK)
    ax_b.text(cx, 4.46 + FAMILY_HEADER_DY, sym,
              ha="center", va="top", fontsize=F_SYM,
              color=MUTED)

for r, (name, sym) in enumerate(rows):
    ax_b.text(-0.15, 3 - r + 0.72 + GRID_DY, name, ha="right", va="center",
              fontsize=F_NAME, color=INK)
    ax_b.text(-0.15, 3 - r + 0.22 + GRID_DY, sym, ha="right", va="center",
              fontsize=F_SYM, color=MUTED)

x_axis_title = ax_b.text(
    3.0, -1.76, "jump family",
    ha="center", va="top", fontsize=F_COORD, color=INK
)
x_axis_subtitle = ax_b.text(
    3.0, -2.36, "which combinations are coupled",
    ha="center", va="top", fontsize=F_KEY, color=MUTED
)
ax_b.text(-1.36, 2.0 + GRID_DY, "rate profile", ha="center", va="center",
          fontsize=F_COORD, color=INK, rotation=90)
ax_b.text(-1.84, 2.0 + GRID_DY,
          "how that action is distributed\nacross sites",
          ha="center", va="center", multialignment="center",
          fontsize=F_KEY, color=MUTED, rotation=90)

# --- compact legend inset into the untested lower-right design space --------
# This replaces the former external legend.  It deliberately occupies only
# neutral, untested cells, leaving the purple/orange swept cross and both open-ended
# continuation cues unobstructed.
legend_items = [
    ("family", "jump families"),
    ("profile", "rate profiles"),
    ("other", "other combinations"),
]
# Centre the key on the six tested jump-family columns and keep it slightly
# below the matrix midpoint so the neutral design cells remain visible above.
KEY_X, KEY_W = 1.90, 2.20
KEY_Y, KEY_H = 0.60 + GRID_DY, 1.88
KEY_ITEM_X = KEY_X + 0.14
key_frame = FancyBboxPatch(
    (KEY_X, KEY_Y), KEY_W, KEY_H,
    boxstyle="round,pad=0,rounding_size=0.045",
    fc="white", ec="#CBCBCB", lw=0.82, zorder=5
)
ax_b.add_patch(key_frame)
sw = 0.15
key_texts = []
for k, (kind, label) in enumerate(legend_items):
    yk = 2.08 + GRID_DY - 0.53 * k
    sy0, sy1 = yk - vy(sw) / 2, yk + vy(sw) / 2
    sx0, sx1 = KEY_ITEM_X, KEY_ITEM_X + sw
    if kind == "family":
        ax_b.add_patch(Rectangle((sx0, sy0), sw, sy1 - sy0,
                                 fc=PATHWAY_CELL, ec=BLUE, lw=0.78, zorder=6))
    elif kind == "profile":
        ax_b.add_patch(Rectangle((sx0, sy0), sw, sy1 - sy0,
                                 fc=PROFILE_CELL, ec=ORANGE, lw=0.78, zorder=6))
    else:
        ax_b.add_patch(FancyBboxPatch(
            (sx0, sy0), sw, sy1 - sy0,
            boxstyle="round,pad=0,rounding_size=0.025",
            fc=to_rgba(CELL, 0.42), ec=to_rgba(CELL_E, 0.58),
            lw=0.55, zorder=6
        ))
    key_texts.append(
        ax_b.text(KEY_ITEM_X + sw + 0.08, yk, label, ha="left", va="center",
                  fontsize=F_KEY, color=INK, zorder=7)
    )


# ================================================ self-check and output =====
def disp_rect(ax, x0, y0, x1, y1):
    """data rectangle -> display bbox"""
    from matplotlib.transforms import Bbox
    p0 = ax.transData.transform((x0, y0))
    p1 = ax.transData.transform((x1, y1))
    return Bbox([[min(p0[0], p1[0]), min(p0[1], p1[1])],
                 [max(p0[0], p1[0]), max(p0[1], p1[1])]])


def text_boxes(figure):
    rend = figure.canvas.get_renderer()
    out = []
    for ax in figure.axes:
        if not getattr(ax, "axison", True):
            cands = list(ax.texts)          # axis("off"): only my own labels
        else:
            _lg = ax.get_legend()
            if _lg is not None:
                cands = list(_lg.get_texts())
            else:
                cands = []
            cands += list(ax.texts) + list(ax.get_xticklabels()) + \
                list(ax.get_yticklabels()) + [ax.xaxis.label, ax.yaxis.label,
                                              ax.title]
        for t in cands:
            if t.get_visible() and t.get_text().strip():
                out.append((t.get_text().replace("\n", " / ")[:32],
                            t.get_window_extent(renderer=rend)))
    for t in figure.texts:
        if t.get_visible() and t.get_text().strip():
            out.append((t.get_text().replace("\n", " / ")[:32],
                        t.get_window_extent(renderer=rend)))
    return out


fig.canvas.draw()
boxes = text_boxes(fig)
problems = []

# 0. the three top panels must retain identical physical plot boxes
_top_positions = [ax.get_position() for ax in (ax_a, ax_b2, ax_c3)]
_top_widths = [p.width * FIGW for p in _top_positions]
_top_heights = [p.height * FIGH for p in _top_positions]
if max(_top_widths) - min(_top_widths) > 1e-10 or \
   max(_top_heights) - min(_top_heights) > 1e-10:
    problems.append(
        "top-panel geometry differs: "
        f"widths={_top_widths}, heights={_top_heights}"
    )

# 1. no two pieces of text may touch
for a, b in itertools.combinations(boxes, 2):
    if a[1].padded(-0.6).overlaps(b[1].padded(-0.6)):
        problems.append(f"text/text: '{a[0]}' <-> '{b[0]}'")

# 2. no text may run into the matrix, icon strip, or response curves
keep_clear = [("grid", disp_rect(ax_b, 0, GRID_DY, 6, 4 + GRID_DY))]
keep_clear += [(f"icon {cols[c][0]}",
                disp_rect(ax_b, c + 0.12, cy - vy(ICON_HALF),
                          c + 0.88, cy + vy(ICON_HALF))) for c in range(6)]
keep_clear += [("(a) curve", disp_rect(ax_a, 0.05, 0, 32, 11.4)),
               ("(b) optimum", disp_rect(ax_b2, 6.4, 12.8, 10.0, 15.5))]
_bins = np.logspace(np.log10(0.05), np.log10(32), 33)
for _name, _pk, _tp, _wl, _wr, _cc, _lw, _ls in DESIGNS:
    for _x0, _x1 in zip(_bins[:-1], _bins[1:]):
        _y = response(np.array([_x0, _x1]), _name, _pk, _tp, _wl, _wr)
        keep_clear.append((f"(b) {_name}", disp_rect(ax_b2, _x0, _y.min() - 0.35,
                                                     _x1, _y.max() + 0.35)))

allowed = {"pair", "thermal", "exchange", "prior work ", "[6]",
           "jump families", "rate profiles",
           "other combinations", r"$\cdots$", r"$\vdots$"}
for label, bb in boxes:
    if label in allowed:
        continue
    for name, kc in keep_clear:
        if bb.padded(-0.6).overlaps(kc):
            problems.append(f"text/graphic: '{label}' runs into {name}")

# 3. panel (b) deliberately uses direct labels instead of a legend. Retain
#    this guard in case a later revision adds one.
_lg = ax_b2.get_legend()
if _lg is not None:
    _lb = _lg.get_frame().get_window_extent()
    for _pt, _nm in (((0.25, 11.32), "local optimum"),
                     ((8.0, 14.17), "collective optimum")):
        if _lb.contains(*ax_b2.transData.transform(_pt)):
            problems.append(f"(b) legend covers the {_nm}")

# 4. text placed inside a framed panel must not spill over its frame
for _ax, _nm in ((ax_b2, "(b)"), (ax_a, "(a)"), (ax_c3, "(c)")):
    _fr = _ax.get_window_extent()
    for _t in _ax.texts:
        _tb = _t.get_window_extent(renderer=fig.canvas.get_renderer())
        if not _fr.padded(-0.5).contains(_tb.x0, _tb.y0) or \
           not _fr.padded(-0.5).contains(_tb.x1, _tb.y1):
            problems.append(f"{_nm}: '{_t.get_text()[:24]}' crosses the frame")

# 5. every key label must retain padding inside the manually drawn key frame
_key_frame = key_frame.get_window_extent(renderer=fig.canvas.get_renderer())
for _t in key_texts:
    _tb = _t.get_window_extent(renderer=fig.canvas.get_renderer())
    if not _key_frame.padded(-2.5).contains(_tb.x0, _tb.y0) or \
       not _key_frame.padded(-2.5).contains(_tb.x1, _tb.y1):
        problems.append(f"key: '{_t.get_text()}' crosses the frame padding")

# The prior-work label sits directly on the cell's white fill; an additional
# white text backing would hide the cell border and make the label appear
# clipped.  Require visible breathing room inside the actual cell instead.
_prior_cell = disp_rect(
    ax_b,
    FIRST_CELL_LEFT,
    3.06 + GRID_DY,
    FIRST_CELL_RIGHT,
    3.94 + GRID_DY,
)
_prior_text = Bbox.union([
    text.get_window_extent(renderer=fig.canvas.get_renderer())
    for text in prior_work_texts
])
if not _prior_cell.padded(-2.5).contains(_prior_text.x0, _prior_text.y0) or \
   not _prior_cell.padded(-2.5).contains(_prior_text.x1, _prior_text.y1):
    problems.append("prior-work label lacks 2.5 pt of cell padding")

# 6. the ghost row and column must remain separate from the axis title and key.
_ghost_column_frame = disp_rect(ax_b, 6.05, GRID_DY, 6.95, 4.0 + GRID_DY)
_ghost_row_frame = disp_rect(
    ax_b, 0.05, GHOST_ROW_Y0, 5.95, GHOST_ROW_Y1
)
_x_axis_title_bb = x_axis_title.get_window_extent(renderer=fig.canvas.get_renderer())
if _x_axis_title_bb.y1 >= _ghost_row_frame.y0 - 3.0:
    problems.append("x-axis title touches the ghost profile row")
if _ghost_column_frame.overlaps(_key_frame):
    problems.append("legend overlaps the ghost family column")
for _patch in ghost_patches:
    _pb = _patch.get_window_extent(renderer=fig.canvas.get_renderer())
    if _pb.overlaps(_key_frame):
        problems.append("legend overlaps a ghost continuation cell")

# 7. everything must sit inside the canvas
tb = fig.get_tightbbox(fig.canvas.get_renderer())
if not (tb.x0 >= -0.005 and tb.y0 >= -0.005
        and tb.x1 <= FIGW + 0.005 and tb.y1 <= FIGH + 0.005):
    problems.append(f"canvas: bbox x[{tb.x0:.3f},{tb.x1:.3f}] "
                    f"y[{tb.y0:.3f},{tb.y1:.3f}] exceeds {FIGW} x {FIGH}")

_gap = (6.0 - (KEY_X + KEY_W)) * \
    ax_b.get_position().width * FIGW / (XMAX - XMIN)
if _gap < 0.08:
    problems.append(f"legend sits only {_gap:.3f} in from the matrix edge")

# 8. no displayed text may fall below the final-size PDF readability floor.
for _text in fig.findobj(matplotlib.text.Text):
    if (
        _text.get_visible()
        and _text.get_text().strip()
        and float(_text.get_fontsize()) < st.MIN_FONT_SIZE - 1e-9
    ):
        problems.append(
            f"font floor: '{_text.get_text()[:24]}' is "
            f"{float(_text.get_fontsize()):.2f} pt"
        )
print(f"legend clearance      : {_gap:.3f} in from the matrix edge")
print(f"top-panel plot boxes  : {_top_widths[0]:.3f} x "
      f"{_top_heights[0]:.3f} in each")
print(f"text elements checked : {len(boxes)}")
print(f"problems found        : {len(problems)}")
for pr in problems:
    print("    " + pr)
print(f"tight bbox            : x[{tb.x0:.3f},{tb.x1:.3f}] "
      f"y[{tb.y0:.3f},{tb.y1:.3f}]  canvas {FIGW} x {FIGH} in")

fig.savefig("fig1_L3l.pdf")
fig.savefig("fig1_L3l.png")
print(f"written: fig1_L3l.pdf / .png   ({FIGW:.2f} x {FIGH:.2f} in)")

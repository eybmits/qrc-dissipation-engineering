# Quantum manuscript

Canonical manuscript:

> **The Organization of Environmental Coupling Shapes What Quantum Reservoirs Remember**

- Source: [`dissipation_qrc.tex`](dissipation_qrc.tex)
- Compiled paper: [`dissipation_qrc.pdf`](dissipation_qrc.pdf)
- Bibliography: [`references.bib`](references.bib)

This is the only active manuscript. It uses the official bundled
`quantumarticle` class and the original `quantum` bibliography style. The
compiled paper is 22 pages and contains exactly 46 cited, linked references.

## Scientific structure

The main text uses one naming convention:

1. **Experiment 1: jump-family comparison**
   - **Control 1: matched expected jump activity**
   - **Control 2: matched midpoint Liouvillian gap**
   - **Control 3: independent operating-point selection**
2. **Experiment 2: local rate-profile comparison**
3. **Experiment 3: finite-sampling comparison**

The flagship inference is the initialization-independent \(N=5\)
collective-over-local relaxation STM contrast. A fresh 16-pair replication
recovers the ordering for both STM and NARMA-10 after replacing the
continuous-drive input with input-by-reset. This establishes transfer across
the two tested encodings while retaining the same processor and readout, not
universal architecture independence. The continuously driven rows of the
broader eight-row task-rank chart are explicitly limited to the tested
fixed-Frobenius protocol; reset FN is an external baseline. The
\(N=4,\ldots,8\) fixed-initialization trend is descriptive because the
\(N=4\) continuation contains one nonconverged collective lineage.

The rank-one orientation evidence has two complementary components. The
\(N=5\) phase-direction experiment tests a frozen complex path and four
phase-scrambled zero-overlap directions. A separately generated \(N=6\),
24-pair replication compares equal phase with the real sign-balanced vector
\((1,1,1,-1,-1,-1)\). It matches the number of jumps, Kossakowski rank and
nonzero spectrum, trace, sitewise diagonal, coefficient magnitudes, assigned
weight, processor protocol, input stream, readout, split, and ridge rule. The
replication establishes orientation dependence at a second tested finite size;
it is not an asymptotic scaling analysis, a universal equal-phase optimality
claim, or identification of a unique microscopic mediator.

## Appendix structure

The appendix is deliberately narrative rather than table-driven:

- **A — Shared methods:** reservoir, tasks, process construction, and
  statistics;
- **B — Experiment 1:** jump-family map, coherent-dynamics and reset-encoding
  replications, finite size, and the three targeted controls;
- **C — Experiment 2:** within-family rate profiles and their interaction with
  jump family;
- **D — Experiment 3:** the two finite-sampling estimators and inference;
- **E — Interpretation support and limits:** observable relaxation,
  switched-input response, interpolation, and the dephasing result; and
- **F — Phase direction:** matched rank-one interventions with frozen
  phase-scrambled controls and a second-size real sign-balanced replication.

Exact grids and secondary outputs live in the sealed evidence archive. The
absolute-score matrix now appears beside the main jump-family results; the
estimator comparison remains an appendix table.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error dissipation_qrc.tex
cd ..
python scripts/validate_submission.py
```

## Source structure

```text
sections/abstract.tex
sections/related-work.tex
sections/introduction.tex
sections/background.tex
sections/experimental-section.tex
sections/evaluation.tex
sections/conclusion.tex
sections/methodology.tex
```

## Canonical figure set

| File | Role |
|---|---|
| `fig_designspace.pdf` | Conceptual jump-family and rate-profile design space |
| `fig_task_scores.pdf` | Full-width absolute-score distributions for the four-task comparison |
| `fig_map.pdf` | Appendix task-wise rank view of the broader catalog |
| `fig_collective_case.pdf` | Full-width 1x3 with equal square panels: robustness controls, size recurrence, and lag-resolved local–collective memory |
| `fig_profiles.pdf` | Paired local-profile effects and the profile-by-structure interaction |
| `fig_sampling.pdf` | Independent and grouped finite-sampling estimators |
| `fig_scalar_controls.pdf` | Activity-, gap-, and operating-point controls with gap-resolved STM profiles |
| `fig_reset_architecture.pdf` | Reset-encoded paired effects and lag-resolved STM capacities |
| `fig_phase_direction.pdf` | Matched rank-one phase path, zero-overlap controls, and second-size real sign-balanced replication |

[`make_figures.py`](make_figures.py) builds the conceptual, map, profile, and
finite-sampling figures.
[`make_forgetting_modes_figure.py`](make_forgetting_modes_figure.py) builds
the case-study and scalar-control figures from the compact snapshots and
aggregates under
[`data/`](data/) and [`evidence/`](evidence/).
[`make_reset_architecture_figure.py`](make_reset_architecture_figure.py)
builds the reset-encoding replication from its checksum-linked compact
snapshot. [`make_phase_direction_figure.py`](make_phase_direction_figure.py)
builds both matched rank-one interventions from their checksum-linked compact
snapshots. All canonical PDFs are stored in [`figures/`](figures/).

The initialization audit remains part of the manuscript evidence even though
the consolidated Figure 3 now uses panel (c) for lag-resolved memory. The
collective continuation has eight \(N=5\) lineages across four initial states;
the local/pair continuation contains 48 lineages across \(N=4,5,6\). Its
frozen protocol, checkpoints, aggregate, and checksums live under
[`evidence/local_pair_convergence_extension_v1/`](evidence/local_pair_convergence_extension_v1/).

All nine PDFs share the reference-derived contract in
[`l3_style.py`](l3_style.py): native full- or single-column dimensions,
publication-scale typography, semantic colors with marker/linestyle
redundancy, a final-size text floor, embedded vector fonts, and
generation-time text/decoration collision audits. Figure 2 is a full-width
four-task absolute-score comparison; the compact rank view is retained in the
appendix. All empirical figures use the manuscript's Times/STIX plotting
style.

The permanent dissipator palette is defined once in `l3_style.py`: collective
is purple, uniform local dark gray, unequal local brown, pair loss orange,
gain/loss mauve, exchange-assisted gold, and dephasing light gray. Figure 2 is
the deliberate focus exception: only uniform local and the best mean in each
panel retain their semantic colors, while the other summaries are neutral
gray. Marker and line-style differences remain available independently of
color.

The deterministic source-package builder includes only these nine referenced
figures, the complete TeX closure, the generated `.bbl`, the official class
and style, and the compact inputs needed to regenerate the map and robustness
figures, including both 1200-input convergence continuations and the strict
reset-architecture driver and snapshot, plus the phase-direction driver,
validator, compact snapshots, and frozen second-size orientation record.

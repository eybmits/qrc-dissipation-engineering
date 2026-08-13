# Manuscript revision report

## Governing argument

The revision is organized around one hierarchy:

> At fixed coherent dynamics and readout, the jump family determines the
> reservoir's task-dependent computational profile, the rate profile refines
> that profile, and measurement design determines whether the resulting
> differences can be resolved.

The revision changes the organization, wording, and presentation of the paper
without adding experiments or changing numerical results. Every textual claim
and visual summary is aligned with the plotted estimand and archived records.

## Section-by-section change log

### Abstract

- Compressed the argument into a short problem--concept--evidence--implication
  sequence.
- Presented the problem, the coupling-organization concept, the evidence
  hierarchy, and the design implication without numerical detail.
- Preserved the task-dependent boundary: no tested continuously driven design
  has the best mean on all four tasks.
- Included the collective-memory, within-family rate-profile, and
  finite-sampling conclusions without a long control inventory.

### 1. Introduction

- Reorganized the introduction into five functional paragraphs: temporal
  processing problem, established work and gap, coupling-organization concept,
  main result hierarchy, and explicit contributions.
- Standardized `jump family`, `rate profile`, `uniform local relaxation`, and
  `structural budget B`.
- Scoped the finite-size result to the fixed-initialization size sweep and the
  reset result to two tested input encodings with the same `N=5` `XX+Z`
  processor and Pauli readout.
- Removed the redundant section-by-section roadmap.

### 2. Environmental-coupling organization and controlled comparison

- Added a plain-language definition of the fixed-input generator and explained
  switched-input evolution as a composition of the corresponding propagators.
- Defined the Kossakowski representation before using its rank.
- Promoted the main theoretical anchor: in the one-body lowering block at fixed
  `B`, uniform local relaxation has rank `N` when all local rates are nonzero,
  whereas collective relaxation has rank one for a nonzero shared coefficient
  vector.
- Explicitly states that these ranks count directly environment-coupled
  Kossakowski directions, not decay modes of the full driven Liouvillian.
- Consolidated the structural-budget boundary: `B` is a bookkeeping
  normalization, not a physical activity, thermodynamic, implementation, or
  hardware-cost model.
- Added compact protocol and task summaries while retaining the full appendix
  definitions.

### 3. What changes when coupling organization changes?

- Rewrote the task-map discussion to match Figure 2: absolute scores in
  native metrics, pointwise Student-`t` intervals on cell means, and a dashed
  uniform-local reference. Interval overlap is not used as a paired test.
- Made Figure 3 the central collective-memory result. It now supports the
  principal `N=5` comparison, the fixed-initialization size-sweep ordering, the
  targeted STM controls, and the fixed-rate lag-resolved comparison.
- Marked the four Hamiltonian results as four ensemble means; the displayed
  2.22--4.02 span is not a confidence interval.
- Made the profile-experiment boundary prominent and reports the local-profile
  and family-by-profile experiments only within their respective protocols.
- Distinguished pointwise, Bonferroni-simultaneous, and exact-expectation
  intervals in the profile and finite-sampling captions and methods.
- Closed the section with the three-level hierarchy rather than a repeated
  result inventory.

### 4. Why collective relaxation retains more memory

- Replaced the former figure-led discussion with a concise mechanistic picture:
  fewer directly environment-coupled lowering combinations, Hamiltonian-mediated
  indirect relaxation, potentially slower readout-visible response, and a
  longer accessible input history.
- Consolidated activity matching, midpoint-gap matching, and independent
  operating-point selection.
- Corrected the midpoint-gap comparison to the same 24 control lineages:
  3.757 to 1.424 STM units, a 62.1% attenuation, with a positive residual
  effect.
- States once, at the end, that the diagnostics support a scoped interpretation
  rather than a unique microscopic mechanism.

### 5. Experimental relevance, limitations, and conclusion

- Replaced the broad discussion with four compact paragraphs: principal result,
  design hierarchy, experimental relevance, and limitations.
- Separates the fixed-`B` theoretical normalization from hardware cost.
- Ends with a bounded design implication: coupling organization should be
  co-designed with the coherent processor, readout, and measurement protocol.

### Appendices

- Preserved detailed protocols, seed-level evidence, initialization audits,
  scalar controls, estimator inference, and reproducibility information.
- Replaced internal review and archive-management language with a concise
  publication-style data-and-code statement.
- Standardized optimizer provenance and states explicitly when full optimizer
  trajectories were not retained.

## Figure relocation and interpretation map

| Evidence | Final location | Interpretation |
|---|---|---|
| Four-panel concept map | Main Figure 1 | Jump family and rate profile are complementary environmental-coupling coordinates. |
| Absolute four-task comparison | Main Figure 2 | Absolute native-metric scores and seed-level dispersion on focused, explicitly bounded axes. |
| Collective-memory synthesis | Main Figure 3 | Principal effect, fixed-initialization size sweep, targeted controls, and fixed-rate lag-resolved memory. |
| Rate-profile experiments | Main Figure 4 | Two separate within-family protocols; absolute values are not compared across protocols. |
| Finite-sampling comparison | Main Figure 5 | Frozen multi-design independent/grouped comparison; simultaneous finite-budget inference. |
| Full reset-based replication | Appendix Figure 6 | Seed-level and lag-resolved recurrence under the second tested encoding. |
| Detailed scalar-control analysis | Appendix Figure 7 | Activity, midpoint-gap, and operating-point controls plus detailed dynamical diagnostics. |

The final main-text sequence is therefore Figures 1--5; all detailed controls
remain available as Appendix Figures 6--8.

## Numerical and statistical consistency report

- Principal `N=5` means: STM `8.634 -> 12.206`; NARMA-10 NMSE
  `0.315 -> 0.229`. These ratios correspond to a 41% increase and a 27%
  decrease, respectively.
- Principal paired STM effect: `3.572 [3.365, 3.779]`; all 32 paired reservoirs
  are favorable on both STM and NARMA-10.
- Long-washout STM effect: `3.564 [3.124, 4.004]`, `10/10` wins.
- Four Hamiltonian ensembles are represented: the principal ensemble plus
  three alternatives. Their STM-effect means span `2.22--4.02`; this is a range
  of ensemble means, not an interval.
- Reset-based encoding: STM effect `1.773 [1.312, 2.235]`; favorable NARMA-10
  reduction `0.182 [0.144, 0.220]`; `16/16` favorable pairs on both endpoints.
- Activity-matched STM effect: `3.306 [1.393, 5.220]`, `8/8` wins.
- Midpoint-gap control on the same 24 lineages: fixed-rate effect `3.757`,
  matched effect `1.424 [0.800, 2.049]`, a 62.1% attenuation with `20/24`
  wins. The matched interval is Bonferroni-simultaneous over the two declared
  task endpoints.
- Independent operating-point selection: `3.945 [3.523, 4.368]`, `24/24`
  wins.
- Dedicated local-profile experiment: learned-minus-uniform STM gain
  `1.34 [1.13, 1.55]`, `32/32` wins.
- Family-by-profile interaction: `1.397 [1.078, 1.716]`, `24/24` positive
  paired values; the interval is Bonferroni-simultaneous over 11 declared
  contrasts.
- Finite-budget intervals are Bonferroni-simultaneous over 210 declared
  contrasts. Exact-expectation endpoints use pointwise paired intervals.
  Grouped measurement resolves the collective-over-local STM ordering at
  one-sixteenth the nominal preparation count on the tested grid; this is an
  estimator-level result, not a hardware-speed claim.
- Figure 2 and the absolute means in Figure 3(a,b,d) use pointwise Student-`t`
  intervals on cell means. Their overlap is not interpreted as a paired test.
- Appendix finite-size relative-effect summaries retain their declared paired
  percentile-bootstrap intervals; archived paired continuously driven
  comparisons retain Holm-corrected sign-flip tests.

## Validation

- Manuscript compiles to 21 A4 pages.
- Eight vector PDF figures are included: five in the main text and three in the
  appendix.
- Submission validation passes with no Type 3 fonts or hard LaTeX defects.
- The complete 21-page rendered PDF was inspected for clipping, overlap,
  conspicuous arbitrary whitespace, orphan headings, and float/caption order.
- Numerical evidence validation passes, including outer and nested checksums,
  finite-size evidence, scalar controls, switched-input controls, chronology,
  and zero-jump summaries.
- All figure and table references resolve; no undefined citations, stuck
  floats, or overfull boxes remain. The log retains only benign underfull-box
  warnings.

## Unresolved editorial choice

The displayed manuscript date remains `30 July 2026`. It was preserved rather
than silently changed; the authors should update it only when the intended
submission date is fixed. No scientific or numerical inconsistency remains
open in this revision.

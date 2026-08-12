# Reproducibility index

This is the reader-facing index for the current manuscript,
**“The Architecture of Dissipation Shapes Memory in Quantum Reservoir Computing.”** It maps
scientific results to machine-readable records without requiring knowledge of
the historical panel labels retained inside the sealed evidence archive.

## Start here

1. Download
   [`complete_reviewer_bundle.zip`](complete_reviewer_bundle.zip).
   It contains the current manuscript, its complete source package, the full
   numerical-evidence archive, the reset-architecture replication archive,
   the phase-direction confirmation archive, the second-size rank-one
   orientation archive, this index, and the exact protocol manifest.
2. Extract it and run:

   ```bash
   python3 validate_complete_bundle.py
   ```

   This single command verifies the current manuscript and source checksums,
   runs the full nested numerical-evidence validator, checks the complete
   protocol manifest, and authenticates the eight \(N=5\) collective and 48
   local/pair cross-size switched-input convergence continuations through 1200
   inputs. It also validates all 16 reset-encoded pairs and their four-state
   initialization audit, plus all 288 phase-direction task cells, 72
   convergence cells, and 72 independent numerical replays. Finally, it
   reconstructs the 24-pair real sign-balanced orientation replication from
   its raw checkpoints and checks the matched channel invariants.
3. The nested numerical-evidence archive is independently sealed with
   SHA-256:

   ```text
   876d6db7592a9fec1dc093fac0eeac422a0a6f129f50994fbbb45a360242ed6e
   ```

   It can also be extracted and checked directly with:

   ```bash
   python3 validate_bundle.py
   ```

The validator checks the outer manifest, nested archives, raw checkpoints,
calibration records, frozen code snapshots, and reconstructed aggregates.
The exact ordered seed arrays, collective coefficients, parameter grids,
solver settings, optimizer rules, convergence gates, and tolerances are in
`protocol_manifest.json` at the root of the complete reviewer bundle.

## The collective process is fully specified

For the main comparison and all local-versus-collective controls, the
collective coefficient vector is fixed rather than sampled or fitted:

```text
c_i = 1 for every site i
c = (1, ..., 1), real and equal phase
L_c = sqrt(gamma) * sum_i sigma_i^-
sum_i |c_i|^2 = N
```

The common Frobenius target is the unit-rate local value
`B = N * 2^(N-1)`, hence `B = 80` at `N=5`; the finite-size protocol
rematches that target separately at every size.

Only the separately labelled profile-by-family interaction check fits a
nonuniform collective profile.

## Convergence evidence for the headline memory result

Initialization independence is tested under the actual switched input rather
than inferred from a constant-input spectral gap. For each tested process and
size, eight lineages start from the ground, fully excited, maximally mixed,
and Haar-random pure states while receiving identical inputs. The records
store the maximum trace and Pauli-feature distances over all six state pairs.

For collective relaxation at the principal \(N=5\) setting, the worst trace
distance falls from \(4.78\times10^{-4}\) at 200 inputs to
\(1.32\times10^{-11}\) at 800. The checksum-sealed continuation reaches
\(4.42\times10^{-15}\) at 1200 inputs and passes its declared \(10^{-14}\)
gate in all eight lineages. Separate checksum-sealed continuations follow all
local and pair-loss lineages at \(N=4,5,6\) through the same 1200-input
horizon; their worst trace distance over steps 1100--1200 is
\(6.85\times10^{-15}\). A separate ten-pair task control then repeats the
comparison with an 800-input washout.
Initialization-induced score differences are unresolved, while the collective
STM advantage remains
\(3.564\,[3.124,4.004]\) with 10/10 paired wins.

The complete reviewer bundle contains every protocol, input definition,
ordered seed, checkpoint, strict-washout score, aggregate, checksum, and
verification script used for these statements.

## Cross-encoding replication

The principal local--collective comparison was repeated after replacing the
continuous transverse-drive input with input-by-reset. The \(N=5\) \(XX+Z\)
processor, 45 Pauli features, task definitions, \(\gamma=1\), and assigned
Frobenius budget \(\mathcal B=80\) were retained. Each of 16 fresh pairs shared
its Hamiltonian, iid \(U[0,1]\) input, target, and data split; the protocol used
800 washout, 600 training, and 400 untouched test inputs.

Collective relaxation increased STM from \(4.596\) to \(6.369\), with a paired
gain of \(1.773\,[1.312,2.235]\), and reduced NARMA-10 NMSE from \(0.673\) to
\(0.491\), with a favorable paired reduction of
\(0.182\,[0.144,0.220]\). Both endpoints were favorable in all 16 pairs. A
four-state rerun gave a worst score spread of \(6.82\times10^{-7}\) and a
largest trace distance of \(1.28\times10^{-14}\) immediately after exactly
800 inputs.

The self-verifying record is
`reset_architecture_evidence.tar.gz` inside the complete reviewer bundle. Its
repository copy is
[`reset_architecture_replication_results.tar.gz`](reset_architecture_replication_results.tar.gz),
with SHA-256:

```text
bfec022e3b79991329231b59317ff237d1031258b8488f51fd7e1d1b778ecb54
```

This establishes transfer across the two tested input encodings while
retaining the same processor and readout; it is not a claim of universal
architecture independence.

## Rank-one orientation at a second finite size

The complex phase-direction experiment is complemented by an independently
generated \(N=6\) intervention with 24 paired reservoirs. Within each pair, the
equal-phase vector \((1,1,1,1,1,1)\) is compared with the real sign-balanced
vector \((1,1,1,-1,-1,-1)\), which is orthogonal to the uniform site profile
of the fixed Hamiltonian input drive. The number of jumps, Kossakowski rank,
nonzero Kossakowski spectrum, trace, coefficient magnitudes, sitewise
diagonal, assigned operator-weight budget, Hamiltonian, input stream, readout,
data split, and ridge rule are held fixed.

Mean STM is \(13.626\) for equal phase and \(11.354\) for the orthogonal
direction. The paired equal-phase-minus-orthogonal effect is
\(2.271\,[1.826,2.717]\), with 24/24 positive differences and a two-sided exact
sign-test \(p=1.19\times10^{-7}\). All pairs pass the ground-versus-mixed audit
at the common 800-input washout; the first six also pass the fully excited and
Haar-pure audit. The response-concentration results are supporting diagnostics
and are not asserted to mediate the STM change.

The complete record is the top-level reviewer-bundle member
`rank_one_orientation_v1_results.tar.gz`. Its repository copy is
[`rank_one_orientation_v1_results.tar.gz`](rank_one_orientation_v1_results.tar.gz),
with SHA-256:

```text
e12f0bdd038b8e45ecea247b09d32815de20330e51ab47aecfe1d995fd86f24a
```

This is a second-size finite-reservoir isolation of orientation dependence,
not an asymptotic scaling analysis, a universal equal-phase optimum, or proof
of a unique microscopic mechanism.

## Current manuscript result map

Unless a top-level bundle member is named explicitly, paths in the second
column are relative to the extracted `collective-loss-numerical-evidence/`
directory.

| manuscript result | raw record | reconstruction or check |
|---|---|---|
| Figure 1: dissipative-architecture design space | The panels are schematics; the illustrative profile values come from `base_reproducibility/qrc_dissipation_reproducibility_package.zip`. | Manuscript source archive: `fig1_L3.py`; base package: `make_figures.py`. |
| Figure 2: absolute four-task score distributions for the seven continuously driven designs | Raw paired family-sweep rows in `base_reproducibility/qrc_dissipation_reproducibility_package.zip`; the corrected parity rows are under `results/revision_parity_control/`, and the replacement MG rows are in the reviewer-protocol archive. | `make_figures.py` reconstructs every seed, absolute mean, and pointwise 95% Student-t interval. Table 2 reports the corresponding means and standard errors; Appendix Figure 6 retains the task-wise ranks and external reset-FN baseline. |
| Archived parity-window diagnostic (not plotted in the current manuscript) | Validation-selected per-seed parity rows in the base reproducibility package under `results/revision_parity_control/`. | The three profiles and pointwise uncertainties remain checksum-linked in `paper/data/experiment1_parity_window_snapshot.json`. |
| Figure 3(a): collective-minus-uniform-local STM effects across targeted controls | Main and Hamiltonian rows in the base package; strict-washout rows in `numerical_evidence.zip`; the reset row in `reset_architecture_replication_results.tar.gz`; scalar controls in their declared evidence packages. | The robustness and reset snapshots plus checksum-linked per-seed effects in `paper/data/experiment1_scalar_control_seed_values.json` are validated before rendering. The three Hamiltonian ensembles remain separate and are not pooled. |
| Figure 3(b): local–collective STM and NARMA-10 across sizes | `finite_size/experiment1_finite_size_v2_results.tar.gz`, containing all 960 nested-lineage checkpoints for `N=4,...,8`. | Frozen protocol, aggregate, production log, validator, aggregate snapshot `paper/data/experiment1_finite_size_snapshot.json`, and the checksum-verified plotted values in `paper/data/experiment1_finite_size_seed_values.json`. |
| Figure 3(c): fixed-local and collective lag-resolved STM | `qrc_dissipation_revision_evidence/results/canonical_gap_control/lag_capacities.csv` in the base reproducibility package. | The tracked checksum-locked `paper/evidence/canonical_gap_control/lag_capacities.csv` contains all 24 seeds, two displayed designs, and 20 delays; the builder validates the complete three-design control table before plotting the displayed pair. |
| Initial-state convergence control | `numerical_evidence.zip` contains `switched_input_memory_control_v2/results/convergence/`; `manuscript_source.zip` contains `evidence/collective_N5_convergence_extension_v1/`, `evidence/local_pair_convergence_extension_v1/`, and both frozen driver snapshots. | The complete-bundle validator authenticates the 800-step control, checks every continuation checksum and protocol hash, reconstructs the envelopes, verifies all frozen prefixes, and checks the eight collective \(N=5\) plus all 48 local/pair \(N=4,5,6\) continuation checkpoints through 1200 inputs. |
| Figure 4(a): paired local rate-profile effects | Per-seed held-out profile rows in the base reproducibility package (32 paired reservoirs per comparison). | `make_figures.py` renders every paired effect with distance-aware opacity; base-package deterministic scripts and held-out reports reconstruct the learned profiles. |
| Figure 4(b): paired profile-by-family interaction | Per-seed local/collective and uniform/learned rows in the base reproducibility package (24 paired reservoirs). | The plotted seed endpoints and trajectories use the same distance-aware opacity rule; the base-package interaction report, simultaneous contrasts, and deterministic scripts provide the audit. |
| Figure 5(a,b): independent and grouped finite sampling | Per-seed, per-estimator, per-budget rows under `base_reproducibility/qrc_dissipation_reproducibility_package.zip` → `qrc_dissipation_revision_evidence/results/measurement_full_v3/`; the exact protocol is `protocol.json`. | `make_figures.py` validates every plotted seed difference against the aggregate mean before rendering; the base-package measurement report and tests provide the independent audit. |
| Figure 6: task-wise rank view of the broader catalog | Absolute means and continuously driven paired rows from the same archives used for Figure 2; reset FN is the external unpaired baseline. | `make_figures.py` reconstructs task-wise aggregate and seed ranks without assigning pseudo-paired seed ranks to reset FN. |
| Figure 7: reset-encoded replication and lag-resolved STM | Top-level complete-bundle member `reset_architecture_evidence.tar.gz` (16 paired reservoirs plus 64-row initial-state audit). | The embedded complete-bundle validator checks exact membership and checksums, protocol, 16/16 paired signs, audited means, and initialization gates. The source package contains the strict driver and checksum-linked compact plotting snapshot. |
| Figure 8(a): scalar controls for the principal STM contrast | Main rows in the base package; activity rows in `expected_jump_activity/operational_activity_ablation_results.tar.gz`; gap rows in `midpoint_gap_control/canonical_gap_control/`; independent-selection manifest at `base_reproducibility/qrc_dissipation_reproducibility_package.zip` → `qrc_dissipation_revision_evidence/results/revision_tuning/nested_operating_point_extension/manifest.json`. | Control-specific validators, top-level `validate_bundle.py`, and the checksum-linked plotted effects in `paper/data/experiment1_scalar_control_seed_values.json`. |
| Figure 8(b): gap-controlled STM lag profiles | `midpoint_gap_control/canonical_gap_control/lag_capacities.csv` (24 paired seeds, delays 1--20). | The canonical gap-control validator authenticates the table; the paper source includes the exact checksum-verified CSV used for rendering. |
| Figure 9: orientation interventions within rank-one shared relaxation | Top-level complete-bundle members `phase_direction_confirmatory_v1_results.tar.gz` and `rank_one_orientation_v1_results.tar.gz`. The first contains 32 fresh paired \(N=5\) reservoirs across nine frozen complex directions, 288 task checkpoints, 72 four-state convergence checkpoints, and 72 numerical replays. The second contains 24 separately generated paired \(N=6\) reservoirs for the real sign-balanced control, every raw checkpoint, and frozen source. | The phase validator authenticates both propagation paths and the complex-direction aggregate. The complete-bundle validator independently checks the second archive's safe TAR structure and checksums, reconstructs its 24-row aggregate and lag-resolved effects, and confirms the matched rank, nonzero spectrum, trace, diagonal, coefficient magnitudes, and \(\mathcal B\). |
| Absolute-score and readout-regularization tables | Family-sweep rows in the base package; Hamiltonian-only rows in `zero_jump_control/`. | `zero_jump_control/summary.csv`, base-package reports, and tests. |

## Where to inspect calibrations

### Expected jump activity

The activity comparison uses eight fresh paired reservoirs. Its archive is:

```text
expected_jump_activity/operational_activity_ablation_results.tar.gz
```

It contains the task-independent input stream, the local and collective
calibration histories, the frozen target and brackets, all per-seed task
records, the aggregate, and the exact source snapshot. All 16 calibration
cells had to pass before any task result was accepted.

### Midpoint Liouvillian gap

The canonical gap records are:

```text
midpoint_gap_control/canonical_gap_control/protocol.json
midpoint_gap_control/canonical_gap_control/calibration.csv
midpoint_gap_control/canonical_gap_control/scores.csv
midpoint_gap_control/canonical_gap_control/lag_capacities.csv
midpoint_gap_control/canonical_gap_control/summary.json
```

The fail-closed reconstruction command is documented in the extracted
archive's `README.md`. A separate dense-spectrum and sparse shift-invert spot
check is under `independent_gap_check/`.

## Seeds, optimizers, and tolerances

The primary, parity, finite-size, calibration, selection, and measurement
seed arrays are in `paper/data/reproducibility_manifest.json`, in generation
order and grouped by scientific role. The Hamiltonian replications and local
profile comparison reuse the first 32 primary seeds; the profile-by-family
check reuses the first 24. The reset-architecture block records its 16 fresh
ordered pair seeds and four-state audit. The rank-one orientation block records
the ordered 24-pair seed ledger and its separate `2026081201` stream namespace;
these are described as separately generated rather than globally fresh because
some numeric identifiers occurred in a pilot protocol. The
independent-initial-state and long-washout seed ledgers are in
`switched_input_memory_control_v2/protocol.json`. The machine-readable
manifest also gives:

- the `h`, `Delta t`, strength, ridge, and sampling-budget grids;
- the activity and gap brackets, bisection counts, and acceptance gates;
- Nelder--Mead starts, evaluation limits, `xatol`, and `fatol`;
- washout, fitting, validation, and test lengths for all three experiments;
- the reset input map, autonomous processor, paired task protocol, and scope;
- linear-solver behavior, Frobenius checks, and feature guards; and
- protocol-specific Python, NumPy, and SciPy versions.

Three easy-to-miss settings are explicit there: parity uses the frozen
`2026072301` seed namespace and its own ridge grid; the profile optimization
uses 32-level input quantization and delays 1 through 12 inclusive; and the
Mackey–Glass stream uses Euler step 1, initial history 1.2 with the recorded
random perturbation, 500 discarded samples, and prefix-only normalization.
Closed-loop predictions are clipped before feedback, but held-out targets are
not clipped.

The archived protocols do not all use one software environment. The
revision-control records use Python 3.13.2, NumPy 2.3.3, and SciPy 1.16.2;
`measurement_full_v3/protocol.json` records Python 3.12.13, NumPy 2.5.1, and
SciPy 1.18.0. Other sealed protocols carry their own environment block.

The Experiment 2 package contains optimizer code, fixed settings, seeds, and
held-out scores. The deterministic sealed scripts reconstruct the learned
profiles. Full optimizer histories were not stored and are not claimed as
archived evidence.

## Historical labels

The numerical archive predates the final compact figure layout. Its internal
`EVIDENCE_MAP.md` therefore retains older panel labels and the earlier working
title so that its checksum remains immutable. The data and protocols are
unchanged. This file is the authoritative semantic map for the current
manuscript.

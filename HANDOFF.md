# Handoff — 2026-08-14

## Resume here

This repository is the canonical source for the paper.

Remote: `https://github.com/eybmits/qrc-dissipation-engineering`

The canonical working branch is `main`, tracking `origin/main`. Continue from
this checkout and branch rather than the older standalone or dated worktrees.

The canonical repository is **public**. Anonymous access to the repository,
paper PDF, arXiv package, and checksum-sealed evidence artifacts was verified
as part of the final publication handoff.

## Canonical release

Title: **The Organization of Environmental Coupling Shapes What Quantum Reservoirs Remember**

- manuscript source: `paper/dissipation_qrc.tex`
- compiled 23-page PDF: `paper/dissipation_qrc.pdf`
- complete reviewer reproducibility bundle:
  `results/complete_reviewer_bundle.zip`
- arXiv upload package: `results/arxiv_submission.zip`
- checksum-sealed supporting manuscript evidence:
  `results/manuscript_supporting_evidence.zip`
- complete evidence archive:
  `results/collective_loss_usable_memory_numerical_evidence.zip`
- reset-architecture evidence archive:
  `results/reset_architecture_replication_results.tar.gz`
- phase-direction evidence archive:
  `results/phase_direction_confirmatory_v1_results.tar.gz`
- second-size rank-one orientation evidence archive:
  `results/rank_one_orientation_v1_results.tar.gz`
- continuous-drive strict-washout NARMA-10 evidence archive:
  `results/continuous_drive_narma_washout_v1_results.tar.gz`
- reader-facing evidence map: `results/REPRODUCIBILITY_INDEX.md`
- exact machine-readable protocol ledger:
  `paper/data/reproducibility_manifest.json`

The paper uses the original Quantum bibliography presentation, 46 linked
references, eight consistently styled vector figures, and a five-section
scientific narrative.

## Manuscript organization

The main text follows one causal chain: coupling organization changes the routes
through which information fades, those routes determine which input traces
survive, the readout determines which traces become usable computation, and
finite sampling determines what is statistically visible. Its five sections
are:

1. **Introduction** — motivation, the two-axis design concept, and the
   local--collective intuition;
2. **Reservoir model and controlled environmental coupling** — fixed processor,
   generator-level coupling organization and Kossakowski geometry,
   structural normalization, readout, and tasks;
3. **Computational effects of coupling organization** — jump-family task
   profiles, rate-profile tuning, and finite-sampling visibility;
4. **Dynamical interpretation of the local and collective memory contrast** —
   physical picture, scalar controls, positive diagnostics, and limits; and
5. **Discussion and conclusion** — direct answer, evidence hierarchy,
   claim boundaries, and co-design implication.

The single-column appendix follows the same reader path:

- **A:** reservoir model, dissipator construction, and shared methods;
- **B:** jump-family computation, the reset-encoding replication, and
  local--collective controls;
- **C:** rate-profile optimization and comparison;
- **D:** finite-sampling estimators and inference;
- **E:** interpretation support and limits; and
- **F:** matched rank-one phase-direction interventions, including the frozen
  complex path and a second-size real sign-balanced replication.

The appendix begins with one self-contained verification path. It identifies
the reviewer bundle, gives its single validation command, fixes the collective
coefficients explicitly, and explains where the seeds, grids, calibration
records, tolerances, raw outputs, and convergence checkpoints live. The
reader-facing evidence index maps every current result to the sealed records.

## Scientific boundary to preserve

The strongest result is the initialization-independent \(N=5\)
collective-over-local STM advantage, now reproduced across two tested input
architectures. Switched-input convergence and a strict 800-step washout
validate that the continuously driven effect is usable memory rather than
surviving initialization. An all-32-pair continuous-drive NARMA-10 replay also
retains the ordering after the 800-input washout, with local-minus-collective
NMSE \(0.0851\,[0.0716,0.0986]\), 32/32 wins, and an effect change from the
200-input washout of
\(-0.00007\,[-0.00057,0.00043]\). Its first eight pairs carry a four-state
audit at both washouts. A separate 16-pair reset-encoded replication uses
the same \(XX+Z\) processor and Pauli readout, changes the input mechanism, and
recovers favorable STM and NARMA-10 effects in all 16 pairs after its own
800-input washout and four-state audit.

This is strong cross-encoding evidence: the ordering is not an artifact of the
continuous-drive input map. Preserve the explicit boundary that only two input
encodings were tested while the \(N=5\) processor and readout were retained; it
is not universal architecture independence.

Three separate controls retain the contrast after:

1. matching mean expected jump activity;
2. matching the collective dominant Liouvillian gap at \(s=0.5\); or
3. selecting local and collective operating points independently.

The gap match attenuates the advantage substantially. It rules out that one
matched midpoint gap is the sole explanation; it does not establish complete
operational equivalence or a fully resolved mechanism.

The orientation result is also supported by two complementary interventions.
The frozen \(N=5\) phase experiment follows a complex equal-phase-to-orthogonal
path and tests four phase-scrambled zero-overlap directions. The separately
generated \(N=6\), 24-pair replication uses the real sign-balanced vector
\((1,1,1,-1,-1,-1)\) and gives an equal-phase-minus-orthogonal STM effect of
\(2.271\,[1.826,2.717]\), favorable in all 24 pairs. It fixes the number of
jumps, Kossakowski rank and nonzero spectrum, trace, sitewise diagonal,
coefficient magnitudes, assigned weight, Hamiltonian, input stream, readout,
split, and ridge rule within each pair. Preserve the boundary: this is a
second-size finite-reservoir isolation of orientation dependence, not an
asymptotic scaling law, a universal equal-phase optimum, or a unique
microscopic mechanism.

The continuously driven rows of the eight-row task-rank chart use the
fixed-Frobenius comparison; reset FN is an external baseline. The chart is not
a physically or operationally matched ranking. One collective \(N=4\) lineage
does not forget its initialization after a 20,000-step continuation. Keep the
\(N=4,\ldots,8\) fixed-initialization trend descriptive unless new
initialization-independent scaling evidence is generated.

## Verified release state

The 14 August 2026 handoff passed:

- the full Python test suite passed;
- submission validation at 23 pages and eight vector figures, with no Type 3
  fonts or hard LaTeX defects;
- source-archive verification and fresh-extraction compilation for all eight figures;
- complete reviewer-bundle validation, including the current PDF and source,
  the full numerical archive, the protocol ledger, the eight-lineage \(N=5\)
  collective continuation, and all 48 local/pair cross-size continuations
  through 1200 inputs, the 16-pair reset-encoding replication and four-state
  audit, plus the 32-pair phase-direction intervention, its 72-cell
  convergence audit, and 72 independent numerical replays, and the 24-pair
  real sign-balanced orientation replication at a second finite size, plus
  the all-32-pair continuous-drive strict-washout NARMA-10 confirmation;
- validation of the complete numerical-evidence archive;
- deterministic evidence-archive hygiene: no Python bytecode, cache members,
  or machine-local paths, enforced by the archive validator; and
- all checks in `results/ARCHIVE_SHA256SUMS.txt`.

The arXiv upload-package checksum is recorded in
`results/ARCHIVE_SHA256SUMS.txt`.
The complete reviewer-bundle checksum is recorded in
`results/ARCHIVE_SHA256SUMS.txt`.

## Verification commands

Run from the repository root:

```bash
PYTHONPATH=src:experiments python -m pytest -q

cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error dissipation_qrc.tex
cd ..
python scripts/validate_submission.py

python scripts/build_quantum_source_archive.py verify \
  results/arxiv_submission.zip
python scripts/build_complete_reviewer_bundle.py verify \
  results/complete_reviewer_bundle.zip
python scripts/validate_final_evidence_archive.py
(cd results && shasum -a 256 -c ARCHIVE_SHA256SUMS.txt)
```

## Regeneration order

After changing manuscript or figure source:

```bash
PYTHONPATH=src:experiments python experiments/run_reset_architecture_strict.py
python scripts/build_result_archives.py \
  --bundle reset_architecture_replication
python scripts/build_result_archives.py \
  --bundle rank_one_orientation_v1
tar -xzf results/continuous_drive_narma_washout_v1_results.tar.gz \
  -C results
PYTHONPATH=src:experiments python \
  experiments/run_continuous_drive_narma_washout.py verify \
  --out results/continuous_drive_narma_washout_v1
python scripts/build_result_archives.py \
  --bundle continuous_drive_narma_washout_v1
cd paper
python make_figures.py
python make_forgetting_modes_figure.py
python make_reset_architecture_figure.py --refresh-snapshot
python make_phase_direction_figure.py
latexmk -pdf -interaction=nonstopmode -halt-on-error dissipation_qrc.tex
cd ..
python scripts/build_quantum_source_archive.py build --profile arxiv \
  --output results/arxiv_submission.zip
python scripts/build_quantum_source_archive.py build --profile reviewer \
  --output results/manuscript_supporting_evidence.zip
python scripts/validate_submission.py
python scripts/build_quantum_source_archive.py verify \
  results/arxiv_submission.zip
python scripts/build_complete_reviewer_bundle.py build
```

Update the arXiv ZIP, supporting-source ZIP, and reviewer-bundle entries in
`results/ARCHIVE_SHA256SUMS.txt`, then rerun the full checksum command above.
Commit only the eight canonical PDFs under
`paper/figures/`; local PNG previews, LaTeX intermediates, and caches are
ignored.

## Next session

1. Read this file and `git status --short --branch`.
2. Confirm `git rev-list --left-right --count origin/main...HEAD` reports
   `0 0`.
3. Edit only the canonical source under `paper/` unless an older version is
   explicitly requested.
4. Rebuild the PDF and source ZIP after manuscript changes.
5. Preserve the claim boundary and the negative \(N=4\) convergence result.

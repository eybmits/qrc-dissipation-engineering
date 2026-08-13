# The Organization of Environmental Coupling Shapes What Quantum Reservoirs Remember

Research code, manuscript source, and checksum-sealed numerical artifacts for
the Quantum manuscript **“The Organization of Environmental Coupling Shapes
What Quantum Reservoirs Remember.”**

- [Read the canonical paper](paper/dissipation_qrc.pdf)
- [Download the complete reviewer reproducibility bundle](results/complete_reviewer_bundle.zip)
- [Browse the canonical manuscript source](paper/)
- [Download the arXiv upload package](results/arxiv_submission.zip)
- [Download the complete numerical-evidence archive](results/collective_loss_usable_memory_numerical_evidence.zip)
- [Download the reset-architecture replication](results/reset_architecture_replication_results.tar.gz)
- [Download the phase-direction confirmation](results/phase_direction_confirmatory_v1_results.tar.gz)
- [Download the second-size rank-one orientation replication](results/rank_one_orientation_v1_results.tar.gz)
- [Download the continuous-drive strict-washout NARMA-10 confirmation](results/continuous_drive_narma_washout_v1_results.tar.gz)
- [Open the reader-facing reproducibility index](results/REPRODUCIBILITY_INDEX.md)
- [Inspect the exact machine-readable protocol manifest](paper/data/reproducibility_manifest.json)
- [Resume work from the current handoff](HANDOFF.md)

## Result and scope

The paper asks whether changing the organization of environmental coupling at
fixed assigned jump-operator weight changes what a quantum reservoir can
compute.
Its strongest conclusion is specific and experimentally challenged: at
\(N=5\), collective relaxation retains more short-term memory than uniform
local relaxation after dependence on the tested initial states has vanished,
and the ordering recurs under both continuous-drive and input-by-reset
encoding.

The canonical manuscript follows one five-section causal narrative:

1. **Introduction:** controlled forgetting and coupling organization;
2. **Reservoir model and controlled environmental coupling:** the fixed processor,
   generator-level design space, and paired comparison;
3. **Computational consequences:** jump-family task profiles, within-family
   rate-profile tuning, and finite-sampling visibility;
4. **Dynamical interpretation:** the local--collective memory contrast,
   alternative scalar explanations, and positive diagnostics; and
5. **Discussion, scope, and conclusion:** the evidence hierarchy, claim
   boundaries, and co-design implication.

Within the principal jump-family analysis, switched-input convergence and an
800-step task washout establish usable rather than initialization memory at
the flagship \(N=5\) point. A separate all-32-pair continuous-drive NARMA-10
replay preserves the favorable ordering at W800: local-minus-collective NMSE
is \(0.0851\,[0.0716,0.0986]\), with 32/32 wins, while the effect change from
W200 is \(-0.00007\,[-0.00057,0.00043]\). Section 4 then shows that the
collective-over-local contrast remains after matching mean expected jump
activity, matching the dominant midpoint Liouvillian gap, or selecting the two
operating points independently. The gap match strongly attenuates the effect,
so relaxation is an important part of the explanation but not its sole
measured component.

An independent 16-pair input-by-reset replication preserves the \(N=5\)
\(XX+Z\) processor and Pauli readout while changing the input mechanism. It
recovers the collective advantage on both STM and NARMA-10 in every pair. This
establishes transfer across the two tested encodings, not universal
independence from reservoir architecture.

A separate 32-pair phase-direction intervention holds the rank-one
Kossakowski spectrum, diagonal weights, coefficient magnitudes, assigned
weight, processor, input, and readout fixed. Rotating away from the equal-phase
direction reduces held-out STM in all 32 pairs, including against four frozen
phase-scrambled zero-overlap controls. This identifies direction sensitivity
within the tested environmental-coupling design without claiming a unique
microscopic cause.

An independent real sign-balanced replication repeats the orientation
intervention at a second finite size. Across 24 separately generated paired
reservoirs, changing only the rank-one coefficient direction from equal phase
to a vector orthogonal to the uniform site profile of the fixed input drive
reduces STM in every pair; the equal-phase-minus-orthogonal effect is
\(2.271\,[1.826,2.717]\). The number of jumps, Kossakowski rank and nonzero
spectrum, trace, sitewise diagonal, coefficient magnitudes, assigned weight,
processor, input stream, readout, split, and ridge rule are all matched. This
is a second-size finite-reservoir isolation of orientation dependence, not an
asymptotic scaling result, universal equal-phase optimum, or unique mechanism.

The continuously driven rows in the wider eight-row task-rank chart use the
tested fixed-Frobenius comparison; reset FN is shown separately as an external
baseline. The chart is not an operationally matched ranking. One collective \(N=4\) lineage remains
initialization dependent after a 20,000-step continuation, so the
\(N=4,\ldots,8\) fixed-initialization sweep is reported descriptively rather
than as initialization-independent scaling evidence.

## Verified checkout

The current checkout was checked locally on 13 August 2026:

- all 183 Python tests pass;
- the 22-page PDF contains nine vector figures, no Type 3 fonts, and no hard
  LaTeX defects;
- the arXiv upload package reproduces all nine figures from a fresh extraction;
- the complete reviewer bundle passes its single-command outer and nested
  validation, including the eight-lineage, four-initial-state collective
  continuation and all 48 local/pair cross-size continuations through 1200
  inputs, the 16-pair reset-encoded replication and its four-state audit, and
  the 32-pair phase-direction confirmation with 72 numerical replays, plus the
  independent 24-pair real sign-balanced orientation replication at a second
  finite size and the all-32-pair continuous-drive strict-washout NARMA-10
  confirmation;
- the complete numerical-evidence archive and every entry in
  [`ARCHIVE_SHA256SUMS.txt`](results/ARCHIVE_SHA256SUMS.txt) validate;
- the numerical-evidence archive contains no bytecode, cache members, or
  machine-local paths; and
- the arXiv upload package is compiled from a fresh extraction and its
  checksum is recorded in
  [`ARCHIVE_SHA256SUMS.txt`](results/ARCHIVE_SHA256SUMS.txt).

The complete release is publicly available at
[github.com/eybmits/qrc-dissipation-engineering](https://github.com/eybmits/qrc-dissipation-engineering).
The self-contained reviewer bundle additionally supports
submission-independent validation without repository access.

## Repository map

| Path | Contents |
|---|---|
| [`paper/`](paper/) | Canonical Quantum source, 46 linked references, and nine figures |
| [`src/qrc/`](src/qrc/) | Reservoir, Lindblad, task, and readout implementation |
| [`experiments/`](experiments/) | Primary experiment and analysis entry points |
| [`reports/`](reports/) | Verified analysis reports |
| [`tests/`](tests/) | Numerical, physics, and package-integrity tests |
| [`results/`](results/) | Canonical source/evidence archives and compact historical result bundles |
| [`scripts/`](scripts/) | Submission, source-package, and evidence validators |

Raw checkpoint trees are distributed inside sealed archives instead of being
expanded into Git. Generated caches and LaTeX intermediates are ignored.
The appendix now prints the most important frozen settings directly, while
[`REPRODUCIBILITY_INDEX.md`](results/REPRODUCIBILITY_INDEX.md) maps each
current result to its raw records and validator.

## Build and verify

Python 3.10 or newer and a standard LaTeX installation are required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

The complete numerical archive contains raw per-seed outputs, frozen
protocols, code snapshots, chronology records, the failed predecessor
forgetting protocol, independent gap checks, nested checksum manifests, and
its own standard-library validator.

## Licence

See [LICENSE](LICENSE).

# Project status - bounded practitioner project complete

## Decision

**Complete for the bounded claim and ready for review.**

The project now contains a precise theorem, a QRC-specific task score, explicit
error propagation, an exact physical counterexample, exhaustive candidate
training for the practitioner benchmark, finite-size and finite-shot supporting
validation, a tested implementation, a reproducible result ledger, and an
8-page working manuscript.

## Reviewer-safe claim

> For a fixed QRC architecture, a specified temporal task, and a finite
> hardware-compatible dissipator family, differentiated-channel task scores can
> prescreen candidates before task-specific readout training. A valid uniform
> score-error radius gives deterministic regret, exact-recovery, safe-elimination,
> and top-k guarantees.

## Completed evidence

- finite-family `2 delta` regret theorem and proof;
- exact-winner, safe-shortlist, and top-k certificates;
- regularized score perturbation theorem;
- explicit contraction-based lag-tail bound;
- exact equal-budget CPTP example with incompatible task optima;
- 84 exhaustive candidate conditions and 21 practitioner decisions;
- 18/21 top-1 winner recovery with 75% training-run saving;
- mean top-1 regret `0.0002006`;
- 21/21 empirical coverage for a conservative leave-one-processor-out shortlist;
- candidate-level prediction correlation `0.99711`;
- 14 task-resolved/certificate tests passing together;
- compiled and visually checked Quantum-style manuscript.

## Claim boundary

The project does not claim the optimal Lindbladian among all possible open
systems, asymptotic quantum advantage, or uniform switched-input contractivity
from one midpoint spectral gap. Those are separate follow-up questions and are
not required for the bounded practitioner contribution.

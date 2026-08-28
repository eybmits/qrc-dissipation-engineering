# Task-resolved Volterra project status

## Decision

**Go.** The project has passed the mechanism, implementation, exact-simulation, finite-shot, held-out-selection, and phase-orientation tests.

## Established

- Exact first channel derivative via `scipy.linalg.expm_frechet`.
- Correct first- and second-order temporal kernels.
- Closed population capacity for Walsh delay and delayed-product tasks.
- Independent and commuting-group finite-shot covariance.
- Fixed-trace PSD Kossakowski construction.
- Primitive-gap feasibility checks.
- Strong N=3 exact validation and a finite N=4 replication.
- Theory-only held-out coupling selection.
- Task-specific phase-orientation gains.
- Six unit tests and a deterministic reproduction entrypoint.

## Defensible publication claim

> Under fixed total dissipative strength and a contractivity constraint, differentiated-channel Walsh capacities predict and select task-dependent environmental couplings before task-specific readout training.

## Not yet established

- Generic quantum advantage over echo-state networks.
- A universal collective optimum.
- Asymptotic scaling.
- General optimality over every PSD Kossakowski matrix.
- Uniform echo-state behavior inferred only from one midpoint gap.

## Minimal final-paper extension

1. General PSD projected optimizer.
2. Uniform switched-input contraction bound.
3. Canonical N=5 integration with convergence audit.
4. Commuting-group shot allocation.
5. Exactly solvable incompatibility theorem.
6. Matched classical Volterra/ESN baseline.

# Rank-one dissipator-orientation intervention

This directory is the sealed numerical record for the confirmatory `N=6`
orientation intervention. It contains all 24 byte-preserved paired checkpoints,
the frozen simulation source, the backend-stable semantic aggregator, the exact
workflows and environment record, figure-input tables, and a standalone
validator.

## What was changed and what was fixed

The two real collective-lowering directions were

- equal phase: `c = (1, 1, 1, 1, 1, 1)`;
- drive orthogonal: `c = (1, 1, 1, -1, -1, -1)`.

Only this direction changed within each paired reservoir. The number of jumps,
Kossakowski rank, nonzero Kossakowski spectrum, trace, coefficient magnitudes,
sitewise diagonal weights, operator-weight budget, Hamiltonian, input stream,
readout, training data, test data, and ridge rule were fixed.

The matched five-dimensional object is the kernel of the `6 x 6` Kossakowski
coefficient block. It is not a five-dimensional many-body dark subspace. The
represented `64 x 64` lowering operator has a 20-dimensional kernel for either
direction.

## Result and audit scope

Equal phase exceeded the drive-orthogonal direction by `2.271493` STM units
with a two-sided 95% Student-t interval `[1.825888, 2.717098]`; all 24 paired differences
were positive. Every pair passed the ground-versus-maximally-mixed convergence
gate at the common 800-input washout. The first six pairs additionally passed
the fully-excited and Haar-pure initial-state audit. The response-concentration
summaries are supporting diagnostics, not demonstrated mediators.

This establishes orientation dependence inside the tested finite rank-one
collective-lowering sector. It does not establish universal equal-phase
optimality, an asymptotic law, or a unique microscopic mechanism.

## Validate without repository access

From this directory, run:

```bash
python validate.py
```

The validator checks the complete SHA-256 ledger; authenticates every raw
checkpoint payload; reconstructs the random coupling, washout-input, and
task-input hashes; verifies the matched channel invariants and convergence
scope; and reconstructs every aggregate statistic and CSV table stored here.

## Full numerical rerun

The exact run environment is recorded in `environment.json`. In a compatible
Python environment:

```bash
cd frozen_source
python -m pip install -e .
PYTHONPATH=src:experiments python experiments/run_rank_one_orientation.py \
  --outdir ../rerun run --seed-index 0
```

Repeat seed indices 0 through 23. Then aggregate the checkpoints without
recomputing trajectories:

```bash
PYTHONPATH=src:experiments python \
  experiments/aggregate_rank_one_orientation_artifacts.py \
  --checkpoint-dir ../rerun/checkpoints \
  --outdir ../rerun-aggregate
```

The native aggregation command in the frozen runner is retained for provenance,
but its hash gate serializes backend-dependent roundoff-level null eigenvalues.
The included semantic aggregator is therefore the authoritative cross-backend
aggregation route.

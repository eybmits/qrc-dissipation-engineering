# Certified dissipator prescreening for quantum reservoir computing

## Final bounded claim

For a **fixed QRC architecture**, a **specified temporal task**, and a **finite,
physically feasible family of dissipators**, differentiated-channel
Walsh--Volterra scores can rank the candidates before task-specific readout
training.

If every predicted score is within a uniform error `delta` of its true
population score, then:

1. training only the predicted winner has regret at most `2 delta`;
2. a predicted winner margin greater than `2 delta` certifies the exact winner;
3. interval elimination safely retains every true optimum; and
4. a top-k boundary gap greater than `2 delta` certifies the entire top-k set.

This is deliberately not a claim about the globally best Lindbladian. The
practitioner supplies the architecture and the realizable candidate family; the
method determines which candidates deserve full task-specific training.

## Why the tool is needed

An exact two-mode diagonal CPTP reservoir with fixed total decay budget proves
that no single environment need be best for all temporal tasks. A balanced
decay allocation wins for delay 2, while a heterogeneous allocation wins for
delay 10. Therefore task-dependent dissipator selection is not merely a tuning
convenience.

## Training-free score

At reference input `u0`, let

```math
A_C = Phi_{u0,C},
D_C = partial_u Phi_{u,C}|_{u0},
A_C r_{*,C}=r_{*,C}.
```

The first-order delay direction and second-order cross-delay direction are

```math
h_d(C)=R A_C^d D_C r_{*,C},
```

and

```math
q_{a,b}(C)=R A_C^a D_C A_C^{b-a-1} D_C r_{*,C},
\qquad a<b.
```

For centered binary inputs, these directions form an orthogonal Walsh feature
model. The optimal population capacity for a delayed-product task is computed
from the feature covariance without fitting a task-specific readout.

## Exhaustive practitioner benchmark

The primary decision experiment contains:

- 3 independently drawn three-qubit processors;
- 7 delayed-product tasks;
- 4 matched-gap local-to-collective dissipator candidates;
- 84 fully simulated and fully trained candidate conditions;
- 21 architecture-task selection decisions.

Training only the predicted winner:

- recovers the empirical winner in **18/21 decisions (85.7%)**;
- saves **75% of full candidate-training runs**;
- has mean regret **0.0002006**;
- has maximum regret **0.003204**.

Candidate-level predicted and fully trained capacities correlate at **0.99711**.
A conservative leave-one-processor-out interval shortlist retains the empirical
winner in **21/21** decisions, has zero observed shortlist regret, and saves
**15.5%** of full trainings.

Supporting validations already in the branch include a finite four-qubit
replication, finite-shot covariance, mixed finite-difference derivative audits,
and task-selected collective phase orientations.

## Reproduction

```bash
python -m pip install -r requirements-task-resolved.txt
bash scripts/reproduce_prescreening_claim.sh
```

The command runs the task-resolved and certificate tests, regenerates the
prescreening result ledger, creates all manuscript figures, and compiles the
paper when `latexmk` is installed.

## Main files

- `src/qrc/task_resolved.py`: differentiated-channel kernels and capacities.
- `src/qrc/prescreen.py`: deterministic certificates and error propagation.
- `tests/test_task_resolved.py`: channel and capacity regression tests.
- `tests/test_prescreen.py`: adversarial certificate and perturbation tests.
- `experiments/analyze_dissipator_prescreening.py`: exhaustive ranking analysis.
- `paper/certified_qrc_dissipator_prescreening.tex`: final working manuscript.
- `reports/task_resolved_volterra/BOUNDED_PRACTITIONER_CLAIM.md`: theorem proof,
  exact counterexample, empirical results, and claim boundary.

## Honest scientific boundary

Established: certified and empirically accurate prescreening inside a fixed
finite feasible family.

Not established: a global optimum over all environments, asymptotic scaling,
or generic quantum advantage over classical reservoirs.

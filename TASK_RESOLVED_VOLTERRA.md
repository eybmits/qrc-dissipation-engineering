# Compiling Temporal Tasks into Dissipation

This branch adds a training-free design framework for open quantum reservoir computing:

> Different temporal functions require different environmental coupling geometries, and those couplings can be selected directly from differentiated quantum channels before fitting a readout.

## What is established

- Exact Frechet derivatives of input-affine quantum channels.
- First-order delay kernels and distinct-delay second-order Volterra kernels.
- Population linear-readout capacities, including finite-shot covariance.
- An exact equal-budget CPTP counterexample proving that no environment is optimal for every delay.
- Matched-gap three-qubit validation and independent fixed-trace four-qubit replication.
- Held-out coupling selection and task-specific rank-one bath-phase selection.
- A covariance ablation showing that second-order feature covariance is essential even for nominally linear memory targets.

## Main finite-size results

- Under matched Liouvillian gap, the strongest tested collective mixture increases recent product capacity by approximately 4.1x while decreasing an old product capacity by approximately 93%.
- Recent and old nonlinear tasks select incompatible couplings in 10/10 three-qubit processors.
- Training-free predictions reach Pearson r = 0.9987 over 245 held-out three-qubit nonlinear conditions.
- Four-qubit validation reaches r = 0.9998 for 56 delay conditions and r = 0.9974 for 56 delayed-product conditions.
- Shot-aware predictions reach r = 0.9924 over 96 noisy conditions.
- Task-selected bath phases improve over local loss in 20/20 tested cases and over the equal-phase collective channel in 19/20.

These results establish a strong finite-size proof of concept, not an asymptotic quantum-advantage theorem.

## Reproduction

Install the extension requirements:

```bash
python -m pip install -r requirements-task-resolved.txt
```

Run the full staged workflow:

```bash
bash scripts/reproduce_task_resolved_volterra.sh
```

The detailed numerical report is in `reports/task_resolved_volterra/RESULTS.md`. The manuscript source is `paper/task_resolved_volterra.tex`; the analysis script generates and copies all required figures into `paper/figures/` before compilation.

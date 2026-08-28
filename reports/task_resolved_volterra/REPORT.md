# Compiling Temporal Tasks into Dissipation

## Verdict

The project is established as a strong, reproducible proof of concept.

The defensible central claim is:

> Differentiated quantum channels define training-free Walsh capacities that
> predict and select task-dependent environmental couplings under fixed total
> dissipative strength and a stability constraint.

This is stronger than another Liouvillian spectrum plot. It is a task compiler:

\[
\text{temporal target}
\longrightarrow
\text{channel derivatives}
\longrightarrow
\text{population capacity}
\longrightarrow
\text{environmental coupling}.
\]

## Mathematical result

For

\[
r_t=\Phi_{u_t,C}r_{t-1},\qquad x_t=Rr_t,
\]

at reference input \(u_0\), define

\[
A_C=\Phi_{u_0,C},\qquad
D_C=\left.\partial_u\Phi_{u,C}\right|_{u_0},\qquad
A_Cr_{*,C}=r_{*,C}.
\]

The first-order delay kernel is

\[
h_d(C)=RA_C^dD_Cr_{*,C}.
\]

For distinct delays \(0\le a<b\), the second-order cross-delay kernel is

\[
q_{a,b}(C)=RA_C^aD_CA_C^{b-a-1}D_Cr_{*,C}.
\]

For Rademacher perturbations \(u_t=u_0+\epsilon z_t\),

\[
x_t-\bar x
=
\epsilon\sum_d h_dz_{t-d}
+
\epsilon^2\sum_{a<b}q_{a,b}z_{t-a}z_{t-b}
+O(\epsilon^3).
\]

Walsh orthogonality gives

\[
G_2(C)=\epsilon^2HH^\top+\epsilon^4QQ^\top+\Sigma_{\rm shot}.
\]

Thus delayed-product capacity is predicted before readout training:

\[
\mathcal C_{a,b}(C)
=
(\epsilon^2q_{a,b})^\top
G_2(C)^+
(\epsilon^2q_{a,b}).
\]

## Physical design family

The experiments use

\[
C_{\alpha,c}
=
\gamma[(1-\alpha)I+\alpha cc^\dagger],
\qquad \|c\|^2=N,
\]

so \(\operatorname{Tr}C=N\gamma\) is constant. A primitive-gap condition
excludes dark or non-forgetting candidates.

## Strengthened results

- Exact nonlinear validation, all sizes: 105 conditions, correlation 0.98912,
  MAE 0.00833.
- Primary N=3 validation: 84 conditions, correlation 0.99688, MAE 0.00432.
- Finite N=4 replication: 21 conditions, correlation 0.99089, MAE 0.02437.
- Finite-shot validation: 72 conditions, correlation 0.99542, MAE 0.00313.
- Mixed finite-difference audit: 20 checks, maximum relative error
  \(1.26\times10^{-7}\).
- Per-reservoir theory selection: exact best grid point in 75% of 28 cases,
  mean regret \(1.23\times10^{-4}\).
- Global held-out selection: 21 cases, mean regret 0.00152, mean gain over local
  0.02572, and no tested loss relative to local.
- Phase-orientation optimization: 16 cases, prediction correlation 0.99886,
  mean gain 0.05695 over local and 0.02892 over equal-phase coupling.

The global theory-only choices were:

| task | selected alpha |
|---|---:|
| (1,2) | 0.75 |
| (1,5) | 0.75 |
| (1,10) | 0.00 |
| (5,10) | 0.25 |
| (5,15) | 0.00 |
| (10,20) | 0.00 |
| (15,25) | 0.00 |

Recent nonlinear interactions prefer collective organization; old cross-delay
interactions prefer local or weakly collective coupling.

## What is new

The paper-level contribution is the combination of:

1. exact differentiated-channel kernels;
2. an orthogonal-task population capacity formula;
3. finite-shot covariance inside the design objective;
4. constrained Kossakowski optimization;
5. held-out coupling selection before task training; and
6. phase-direction design rather than only scalar dissipation tuning.

## Claim boundary

This is not yet an asymptotic quantum-advantage result. The strengthened study
covers N=3 and a finite N=4 replication. The theory is perturbative, the phase
search uses a structured rank-one family, and one midpoint Liouvillian gap is
not a proof of uniform switched-input contractivity.

## Minimal final-paper extension

1. Prove a uniform switched-input contraction and truncation bound.
2. Optimize over general PSD Kossakowski matrices with fixed trace.
3. Integrate the compiler with the canonical N=5 convergence-controlled
   evidence pipeline.
4. Use commuting-group shot covariance and explicit shot allocation.
5. Add one exactly solvable family with analytically incompatible task optima.
6. Compare with a matched classical Volterra or echo-state baseline without
   claiming generic quantum advantage.

Recommended title:

**Compiling Temporal Tasks into Dissipation: Task-Resolved Volterra Design of
Quantum Reservoirs**

# Verified expected-jump-activity-matched control

## Scope

This targeted control asks whether the uniform-local versus collective STM
ordering survives when both designs are calibrated to the same time-averaged
expected jump activity

\[
J=\frac{1}{T\Delta t}\sum_t\int_0^{\Delta t}
\sum_k \operatorname{Tr}(L_k^\dagger L_k\rho_t(\tau))\,d\tau.
\]

For the one-body lowering channels used here, this is the mean bath-induced
excitation-removal rate in the declared jump representation. It is not
dissipated power, entropy production, relaxation time, Liouvillian gap,
experimental effort, or hardware cost.

## Frozen protocol

- One reachable target: \(J_\star=0.5075\), the rounded central member of the
  earlier five-target activity grid.
- Eight fresh paired \(N=5\) reservoirs, disjoint from the primary pool.
- \((h,\Delta t)=(0.5,0.5)\).
- Uniform-local and collective rates calibrated separately within every pair
  on the same unlabeled i.i.d. \(U[0,1]\) stream, independent of task streams.
- Hard all-cell gate: all 16 calibrations had to reach the target within
  \(1.5\%\) before any task score.
- The calibrated multipliers were frozen before the four task endpoints were
  evaluated.

The trajectory run completed in GitHub Actions workflow `30268451892` at
commit `ab9a86391da3136fe146784e512a963d61505350`. Artifact `8654879434`
has SHA-256
`4bf62b88fbd4679c6aca7a5c749ee2e9d06fe49753376b60f0d13a23705ea2c6`.
The frozen protocol hash is
`58294d536fbe29bee0eef9b98128e2f74615027b2588556ca870aaaa5d127fcd`.

The workflow artifact's original `aggregate.json` has SHA-256
`94e720aa2ac0f1decbcf73984f8cb396bcf05230aed3feed543ad08d4844dfdb`.
It mislabeled a binomial sign test as an exact paired random-sign test. The
verified aggregate was rebuilt from the unchanged per-seed rows with the
correct random-sign test and four-task Holm adjustment; no simulation or task
score was rerun. That aggregate has SHA-256
`14e38f785b7192e42b240c25f0199358ea6578176060b433f08d7c86b075c28f`
and is preserved at commit
`ed4124f8ae962e59fa82e609f224233ac6b1a71e`. The deterministic local archive
`results/operational_activity_ablation_results.tar.gz` preserves the corrected
aggregate, all 16 per-seed result files, protocol, provenance, and executed
source snapshot; its SHA-256 is
`85a2679a0f9aba300d09a6e91ed8fcf2033c743c04b0568b3c1886c52c2b0745`.

## Calibration

All 16 calibrations passed. The maximum target error was \(0.391\%\).

| target | uniform local | collective | collective-local, 95% CI |
|---:|---:|---:|---:|
| 0.5075 | 0.508163 +/- 0.000396 | 0.507269 +/- 0.000308 | -0.000894 [-0.002353, 0.000564] |

## Task results

Effects are oriented so positive favors collective loss. Error-metric
differences are sign-reversed. Intervals are paired pointwise 95% Student-t
intervals; Holm correction covers exactly four exact paired random-sign tests.

| task | uniform local | collective | oriented effect, 95% CI | wins | Holm p |
|---|---:|---:|---:|---:|---:|
| STM capacity | 10.734850 | 14.041058 | +3.306208 [1.392545, 5.219871] | 8/8 | 0.03125 |
| NARMA-10 NMSE | 0.261032 | 0.218483 | +0.042550 [-0.009693, 0.094792] | 5/8 | 0.21875 |
| parity capacity | 3.289352 | 2.830490 | -0.458862 [-0.699411, -0.218313] | 1/8 | 0.046875 |
| Mackey-Glass MSE | 0.063182 | 0.075458 | -0.012276 [-0.062826, 0.038274] | 4/8 | 0.703125 |

An independently reimplemented augmented-Liouvillian integral reproduced the
archived first-seed calibration values. With the calibrated rates frozen, the
collective-minus-local activity difference on the held-out STM test segment
was \(-0.011467\;[-0.030956,0.008021]\), so no difference was resolved.
These task-stream checks were post-hoc verification intervals, not
preregistered equivalence tests.

## Supported conclusion

The collective STM advantage survives a separately frozen, one-target
expected-jump-activity normalization. Unequal expected jump activity is
therefore not the sole explanation of the 3.31-unit STM contrast in this
protocol.

This is not a complete activity-response curve. The earlier five-target
protocol remains a failed feasibility audit because four high-target
local-loss cells could not pass its all-cell gate. Neither protocol establishes
equal heat flow, entropy production, Liouvillian gap, relaxation time,
experimental effort, or hardware cost.

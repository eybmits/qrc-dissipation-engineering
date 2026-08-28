# Certified training-free prescreening of QRC dissipators

## Final bounded claim

For a **fixed QRC architecture**, a **specified temporal task**, and a **finite,
physically feasible family of dissipators**, differentiated-channel
Walsh--Volterra scores can rank the candidates before task-specific readout
training. If the score approximates each candidate's population performance to
uniform error at most \(\delta\), then:

1. choosing the predicted winner has regret at most \(2\delta\);
2. a predicted margin larger than \(2\delta\) certifies the exact winner;
3. interval elimination produces a shortlist guaranteed to retain every true
   winner; and
4. a boundary gap larger than \(2\delta\) certifies the complete top-\(k\) set.

This is the useful practitioner result. It does **not** claim the globally best
environment among all imaginable open-system processes. It claims safe and
accurate prescreening **inside the hardware-compatible family supplied by the
practitioner**.

---

## 1. Fixed practitioner setting

Fix:

- a reservoir Hilbert space and Hamiltonian;
- an input encoding and time step;
- a measured observable set;
- a temporal target \(Y_T\), normalized to unit variance;
- a finite feasible family
  \[
  \mathfrak C=\{C_1,\ldots,C_M\},
  \]
  where each \(C_i\succeq0\) obeys the same physical budget and a declared
  stability or contractivity requirement.

Let \(S_i\in[0,1]\) be the true population score of candidate \(C_i\) for the
specified task, and let \(\widehat S_i\) be the training-free differentiated-
channel score.

The procedure does not alter the QRC architecture. It only ranks the allowed
environmental couplings.

---

## 2. The deterministic prescreening theorem

### Theorem 1 - finite-family regret and recovery

Assume

\[
\max_{1\le i\le M}|S_i-\widehat S_i|\le\delta.
\]

Let

\[
i^*\in\operatorname*{arg\,max}_i S_i,
\qquad
\widehat i\in\operatorname*{arg\,max}_i\widehat S_i.
\]

Then:

#### A. Regret bound

\[
\boxed{S_{i^*}-S_{\widehat i}\le2\delta.}
\]

#### B. Exact winner certificate

If

\[
\widehat S_{\widehat i}
-
\max_{j\ne\widehat i}\widehat S_j
>
2\delta,
\]

then \(\widehat i\) is the unique true winner.

#### C. Safe elimination

Define the best lower endpoint

\[
L_{\max}=\max_i(\widehat S_i-\delta).
\]

Discard candidate \(j\) whenever

\[
\widehat S_j+\delta<L_{\max}.
\]

Every true maximizer remains in the non-discarded set.

#### D. Exact top-k certificate

Order the predictions as

\[
\widehat S_{(1)}\ge\cdots\ge\widehat S_{(M)}.
\]

If

\[
\widehat S_{(k)}-\widehat S_{(k+1)}>2\delta,
\]

then the predicted top-\(k\) set equals the true top-\(k\) set.

### Proof

For the regret,

\[
\begin{aligned}
S_{i^*}-S_{\widehat i}
&=
(S_{i^*}-\widehat S_{i^*})
+(
\widehat S_{i^*}-\widehat S_{\widehat i})
+(\widehat S_{\widehat i}-S_{\widehat i})\\
&\le\delta+0+\delta=2\delta,
\end{aligned}
\]

because \(\widehat i\) maximizes the predicted score.

For exact recovery, every competitor obeys

\[
S_j\le\widehat S_j+\delta
<\widehat S_{\widehat i}-\delta
\le S_{\widehat i}.
\]

For safe elimination, suppose a true maximizer \(j\) were discarded. Then some
candidate \(i\) satisfies

\[
S_j\le\widehat S_j+\delta
<\widehat S_i-\delta
\le S_i,
\]

contradicting maximality of \(j\). The same interval comparison across the
predicted top-\(k\) boundary proves the final statement. \(\square\)

### Operational consequence

If full task training costs \(T_{\rm train}\) per candidate and the prescreener
retains \(k\) of \(M\) candidates, the number of full task-specific training
runs falls from \(M\) to \(k\):

\[
\boxed{\text{full-training saving}=1-\frac{k}{M}.}
\]

The theorem bounds the loss incurred by the prescreening decision; it does not
pretend that evaluating the response-kernel score is free.

---

## 3. From channel approximation error to score error

The practitioner needs a way to obtain \(\delta\). Use the regularized
population score

\[
s_\lambda(G,g)=g^\top(G+\lambda I)^{-1}g,
\qquad\lambda>0,
\]

where \(G\) is the feature covariance and \(g=\operatorname{Cov}(x,Y_T)\).
The differentiated-channel model produces \(\widehat G\) and \(\widehat g\).

### Theorem 2 - score perturbation bound

Assume

\[
\|G-\widehat G\|_2\le\epsilon_G,
\qquad
\|g-\widehat g\|_2\le\epsilon_g,
\]

with \(G,\widehat G\succeq0\). Then

\[
\boxed{
|s_\lambda(G,g)-s_\lambda(\widehat G,\widehat g)|
\le
\frac{\epsilon_g(2\|\widehat g\|_2+\epsilon_g)}{\lambda}
+
\frac{\|\widehat g\|_2^2\epsilon_G}{\lambda^2}.
}
\]

### Proof

Let \(A=G+\lambda I\) and \(\widehat A=\widehat G+\lambda I\). Since both
covariances are positive semidefinite,

\[
\|A^{-1}\|_2,\|\widehat A^{-1}\|_2\le\lambda^{-1}.
\]

Insert \(\widehat g^\top A^{-1}\widehat g\). The vector perturbation is at most

\[
\|g-\widehat g\|_2\|A^{-1}\|_2
(\|g\|_2+\|\widehat g\|_2)
\le
\frac{\epsilon_g(2\|\widehat g\|_2+\epsilon_g)}{\lambda}.
\]

For the covariance perturbation, the resolvent identity gives

\[
A^{-1}-\widehat A^{-1}
=A^{-1}(\widehat G-G)\widehat A^{-1},
\]

so

\[
|\widehat g^\top(A^{-1}-\widehat A^{-1})\widehat g|
\le
\frac{\|\widehat g\|_2^2\epsilon_G}{\lambda^2}.
\]

Adding the bounds proves the result. \(\square\)

### Feature-level corollary

Suppose the centered true feature is

\[
x=\widehat x+e,
\]

with

\[
\|\widehat x\|_2\le B,
\qquad
\|e\|_2\le r,
\qquad
|Y_T|\le1.
\]

Then

\[
\epsilon_g\le r
\]

and, allowing a shot-covariance approximation error \(\epsilon_\Sigma\),

\[
\epsilon_G\le2Br+r^2+\epsilon_\Sigma.
\]

These expressions can be inserted into Theorem 2 and then into Theorem 1.

---

## 4. Explicit memory-tail control

At reference input, let \(A\) be the channel on the traceless state subspace and
assume a chosen induced norm satisfies

\[
\|A\|\le\eta<1.
\]

For

\[
h_d=RA^dD r_*,
\]

write \(B_1=\|R\|\|Dr_*\|\). Then

\[
\|h_d\|\le B_1\eta^d.
\]

For

\[
q_{a,b}=RA^aDA^{b-a-1}Dr_*,
\qquad a<b,
\]

write \(B_2=\|R\|\|D\|\|Dr_*\|\). Then

\[
\|q_{a,b}\|\le B_2\eta^{b-1}.
\]

Truncating all delays older than \(L\) gives the explicit first- and second-
order lag remainder

\[
\boxed{
r_{L,2}
\le
|\epsilon|B_1\frac{\eta^{L+1}}{1-\eta}
+
|\epsilon|^2B_2
\frac{(L+1)\eta^L-L\eta^{L+1}}{(1-\eta)^2}
+r_{\ge3}(\epsilon).
}
\]

The final term is the higher-order small-signal remainder. It can be bounded
from higher channel derivatives or conservatively calibrated. This exposes the
scope of the guarantee instead of hiding it: finite lag, finite order, declared
input amplitude, declared contraction, and declared measurement model.

---

## 5. Exact equal-budget impossibility of a universal environment

A bounded prescreening tool is useful only if different tasks can genuinely
prefer different feasible environments. This can be proved exactly in a
commuting open-quantum model.

Consider two independently damped computational-basis modes with linear
response

\[
x_t=\Lambda x_{t-1}+b u_t,
\qquad
\Lambda=\operatorname{diag}(e^{-\gamma_1},e^{-\gamma_2}),
\]

and full readout of both modes. Such dynamics is realized by input-dependent
binary CPTP channels and hence is a valid commuting quantum reservoir. For
zero-mean unit-variance i.i.d. inputs, the exact delay-\(d\) capacity is

\[
\mathcal C_d=r_d^\top G^{-1}r_d,
\]

with

\[
(r_d)_i=b_i\lambda_i^d,
\qquad
G_{ij}=\frac{b_i b_j}{1-\lambda_i\lambda_j}.
\]

Compare two environments with the same total decay budget

\[
\gamma_1+\gamma_2=1:
\]

- balanced: \((\gamma_1,\gamma_2)=(0.4,0.6)\);
- heterogeneous: \((\gamma_1,\gamma_2)=(0.05,0.95)\).

The exact capacities are

| delay | balanced | heterogeneous | winner |
|---:|---:|---:|---|
| 2 | 0.265180 | 0.078103 | balanced |
| 10 | 0.003620 | 0.043885 | heterogeneous |

Therefore no one of these equal-budget environments is optimal for both tasks.
The effect has a simple interpretation: balancing decay gives stronger recent
coverage, while concentrating the budget creates one slow mode for old memory.

---

## 6. Exhaustive prescreening experiment

### Protocol

The empirical test used the existing three-qubit differentiated-channel
validation data:

- three independently drawn XX+Z processors;
- seven delayed-product tasks;
- four matched-gap dissipator candidates per processor-task group;
- 21 architecture-task groups;
- 84 candidate conditions, each evaluated by a full switched Lindblad
  simulation, readout training, and held-out test capacity.

The prescreen score was computed without task-specific readout training. The
fully trained test capacity served as the oracle score.

### Top-1 prescreening

Training only the predicted winner instead of all four candidates:

- recovered the exact empirical winner in **18/21 groups = 85.7%**;
- Wilson 95% interval for the hit fraction: **[65.4%, 95.0%]**;
- saved **75% of full candidate-training runs**;
- mean regret: **0.0002006**;
- bootstrap 95% interval for mean regret: **[0, 0.0005298]**;
- maximum regret: **0.003204**.

The mean empirical gain of the selected candidate was:

- **0.02076 over always-local selection**;
- **0.001786 over always choosing the most collective candidate**;
- **0.01445 over a uniformly random candidate in expectation**.

The corresponding bootstrap intervals for the mean gains were strictly
positive in all three comparisons.

### Ranking quality

Across the 21 groups:

- mean Spearman rank correlation: **0.781**;
- median Spearman correlation: **1.0**;
- mean Kendall correlation: **0.762**;
- mean pairwise ordering accuracy: **88.1%**.

### Shortlist trade-off

| shortlist | empirical oracle included | mean regret | max regret | full trainings saved |
|---:|---:|---:|---:|---:|
| top 1 | 85.7% | 0.0002006 | 0.003204 | 75% |
| top 2 | 85.7% | 0.0002006 | 0.003204 | 50% |
| top 3 | 95.2% | 0.0000846 | 0.001776 | 25% |
| all 4 | 100% | 0 | 0 | 0% |

Top-2 does not improve this particular four-candidate grid because each of the
three ranking errors placed the empirical winner below second place. This
negative detail is retained rather than hidden.

### Conservative leave-one-seed-out intervals

For each held-out processor seed, a uniform residual radius was estimated as
the maximum absolute prediction error on the other two seeds. Applying the safe
elimination rule to the held-out seed:

- retained the empirical winner in **21/21 groups**;
- produced zero shortlist regret in all 21 groups;
- retained 3.38 of 4 candidates on average;
- saved 15.5% of full training runs on average.

This conservative variant trades more computation for an observed zero-miss
shortlist. The deterministic theorem applies when the supplied radius is a
valid uniform bound; leave-one-seed-out calibration is an empirical procedure,
not by itself a universal probabilistic guarantee.

---

## 7. Supporting validation already established in the branch

The prescreening result sits on top of the following independent checks:

- 105 exact nonlinear conditions across finite N=3 and N=4 systems:
  prediction correlation 0.9891 and MAE 0.00833;
- N=3 subset: 84 conditions, correlation 0.9969;
- finite N=4 replication: 21 conditions, correlation 0.9909;
- 72 finite-shot conditions: correlation 0.9954 and MAE 0.00313;
- 20 mixed finite-difference kernel audits: maximum relative error
  \(1.26\times10^{-7}\);
- 16 phase-orientation conditions: prediction correlation 0.9989 and mean gain
  0.02892 over the equal-phase collective channel;
- global task-level choices evaluated on held-out processors: mean gain 0.02572
  over local loss and no tested negative gain.

These support a finite-size practitioner tool. They do not establish an
asymptotic or generic quantum advantage.

---

## 8. Practitioner algorithm

1. **Freeze the architecture.** Fix processor, encoding, time step, observables,
   input amplitude, ridge convention, and shot budget.
2. **Declare the feasible family.** Every candidate must satisfy complete
   positivity, the same physical budget, and a stability/convergence screen.
3. **Describe the task.** Use a delay, delayed product, or a finite expansion in
   an orthogonal temporal basis.
4. **Compute response kernels.** Obtain the needed derivatives of the reference
   channel and form \(h_d\), \(q_{a,b}\), and the shot-aware covariance.
5. **Score all candidates without task training.** Compute \(\widehat S_i\).
6. **Attach an error radius.** Use an analytic truncation/moment bound or a
   deliberately conservative calibration protocol.
7. **Eliminate safely.** Discard candidates whose upper score interval is below
   the best lower interval.
8. **Fully train only the survivors.** Select the best survivor on a validation
   split, then open the untouched test set.

---

## 9. Final scientific boundary

### Established claim

> Within a fixed architecture and a declared physically feasible dissipator
> family, differentiated-channel task scores are a useful training-free
> prescreener. Uniform approximation error yields deterministic regret,
> recovery, and safe-elimination guarantees, and exhaustive finite-size tests
> show large reductions in full candidate training with negligible observed
> regret.

### Not claimed

- the optimal environment over every possible Lindbladian;
- generic quantum advantage over all classical reservoirs;
- asymptotic scaling advantage;
- uniform switched-input contractivity from one midpoint spectral gap;
- non-perturbative accuracy for arbitrary input amplitude;
- a zero-cost prescreener.

This boundary is both useful to QRC practitioners and defensible to reviewers.

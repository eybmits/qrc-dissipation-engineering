# Experiment entry points

The manuscript has three headline experiments. Their canonical drivers and
sealed outputs are:

| Experiment | Purpose | Primary entry points |
|---|---|---|
| 1 | Dissipative-process comparison, with matched-activity, matched-gap, and independent-selection controls | [`run_final_table.py`](run_final_table.py), [`run_final_parity_mg.py`](run_final_parity_mg.py), [`run_experiment1_finite_size.py`](run_experiment1_finite_size.py), [`run_revision_tuning.py`](run_revision_tuning.py), [`run_nested_operating_point_extension.py`](run_nested_operating_point_extension.py) |
| 2 | Local rate-profile comparison and process–profile interaction | [`run_adaptive_supplement.py`](run_adaptive_supplement.py), [`run_joint_axis.py`](run_joint_axis.py) |
| 3 | Finite-sampling comparison | [`run_measurement_full.py`](run_measurement_full.py) |

The complete legacy/reviewer evidence is sealed in
[`qrc_dissipation_reproducibility_package.zip`](../results/qrc_dissipation_reproducibility_package.zip).
The completed finite-size checkpoints are in
[`experiment1_finite_size_v2_results.tar.gz`](../results/experiment1_finite_size_v2_results.tar.gz).
The separately frozen activity-matched control, including its executed source
snapshot, is in
[`operational_activity_ablation_results.tar.gz`](../results/operational_activity_ablation_results.tar.gz).
The complete current evidence archive, including the switched-input,
strict-washout, midpoint-gap, negative \(N=4\), independent gap, and
no-dissipation
records, is
[`collective_loss_usable_memory_numerical_evidence.zip`](../results/collective_loss_usable_memory_numerical_evidence.zip).

## Finite-size workflow

The Experiment-1 extension is restart-safe and validates every checkpoint
before reuse. To inspect or reproduce the published run from a fresh clone,
first restore its compact checkpoint tree:

```bash
tar -xzf results/experiment1_finite_size_v2_results.tar.gz -C results
```

Then run:

```bash
python experiments/run_experiment1_finite_size.py freeze
python experiments/run_experiment1_finite_size.py status
python experiments/run_experiment1_finite_size.py run --workers 4
python experiments/run_experiment1_finite_size.py validate
python experiments/build_experiment1_finite_size_paper_snapshot.py
```

The frozen production manifest contains eight designs, five sizes
\(N=4,\ldots,8\), and 24 paired lineages: 960 trajectories in total. Raw
checkpoint trees are distributed as archives rather than expanded in Git.

Historical filenames and result identifiers containing `collective_loss` are
kept stable for archive compatibility; the manuscript term is **collective
relaxation**.

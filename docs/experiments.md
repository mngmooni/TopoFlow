# Research experiment index

Experiments are organized by question rather than calendar labels. The renamed entrypoints preserve the scientific implementation and should be run from the repository root unless otherwise noted. Many training/evaluation scripts require regenerated layout datasets and trained checkpoints, which are not bundled.

| Research question | Directory | Representative entrypoints |
|---|---|---|
| Does route conditioning improve low-NFE path behavior? | `experiments/baselines` | `cfm_oracle_benchmark.py`, `route_conditioning_benchmark.py` |
| Do model size and obstacle attention improve generalization? | `experiments/baselines` | `cfm_capacity_sweep.py`, `obstacle_attention_benchmark.py`, `mlp_training_budget.py` |
| How should route class selection be modeled? | `experiments/topology` | `route_selection_study.py`, `selector_ablation.py`, `cost_regression_selector_eval.py` |
| Why do raw flows fail clearance? | `experiments/analysis` | `d1_fn_probe.py`, `t_flowtime.py`, `ood_quality.py` |
| Does a feasible chart preserve segment clearance? | `experiments/feasible_chart` | `v2_train.py`, `v2_eval.py`, `chart_verify.py`, `chart_coverage.py` |
| How does chart generation compare with correction and reference data? | `experiments/validation` | `chart_soundness.py`, `baseline_paired.py`, `fidelity_collect.py` |

The final paper comparison uses original frozen source data; its published numerical summaries have not been recomputed as part of this code-layout refactor.

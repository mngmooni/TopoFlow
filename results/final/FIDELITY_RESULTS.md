# Experiment C — class-matched conditional distributional fidelity (D2/D3, in distribution)

Locked: `outputs/predictions_final_C.md` (sha256 73c4de90…). Code: `scripts/fidelity_collect.py`,
`scripts/fidelity_boot.py`, `scripts/fig_trajectories.py`. Data: `outputs/final/fidelity/*.npz`; summary
`outputs/final/distribution_fidelity.json`. No training. **The sampling planner is a reference distribution, not
ground truth**: "closer to the reference" means fidelity to that planner, not better paths.

## Matching
| | chart-feasible classes | matched (planner references) | no reference (disclosed, not failures) | references whose class has no feasible chart | refs / class (median) |
|---|---|---|---|---|---|
| D2 | 6,240 | 6,240 (100%) | 0 | 0 (Experiment A) | 4 |
| D3 | 6,152 | 6,147 (99.9%) | 5 | 0 (Experiment A) | 4 |

Generated: 8 samples per class, NFE 20, noise seeds 11/12/13. v2: V2_s0 on D2 and D3, plus V2_s1, V2_s2 on D2
(model-seed analysis; the D2 intervals below pool the 9 runs). P1 + DiRecT-chart: P1_s0 × 3 noise seeds.

## Results (equal-class aggregation; 95% layout-clustered bootstrap, B = 4000, run drawn per replicate)
Ratios are generated / reference of class-averaged statistics; len_ratio and hug are differences (gen − ref).
len_ratio is length / J_c (chart-constrained shortest length of the class).

| statistic | v2 D2 | v2 D3 | P1 + DiRecT-chart D2 | P1 + DiRecT-chart D3 |
|---|---|---|---|---|
| len_ratio gap | −0.003 [−0.005, −0.002] | −0.001 [−0.002, 0.000] | −0.021 [−0.022, −0.019] | −0.019 [−0.020, −0.018] |
| roughness ratio | 0.83 [0.80, 0.85] | 0.83 [0.81, 0.86] | 0.71 [0.70, 0.73] | 0.73 [0.72, 0.75] |
| min-clearance ratio | 0.93 [0.92, 0.94] | 0.94 [0.93, 0.95] | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| hug gap (share with clearance < c + 0.005) | +0.100 [0.085, 0.115] | +0.088 [0.075, 0.100] | +0.064 [0.053, 0.074] | +0.065 [0.055, 0.074] |
| diversity ratio | 0.95 [0.94, 0.96] | 0.95 [0.94, 0.95] | 0.83 [0.83, 0.84] | 0.82 [0.82, 0.83] |

Reference-count-weighted summaries agree with the equal-class ones to the third decimal (refs per class are almost
all 4), so the weighting rule does not change any conclusion. Model-seed spread (D2 v2 roughness ratio, noise-pooled):
0.831 / 0.824 / 0.825. Minimum analytic clearance of every generated sample ≥ c (0.010004).

Secondary (labelled secondary): per-class 1D W1, median over classes, gen–ref vs gen–gen (seed 11 vs 12) floor.
| | len_ratio | roughness | min clearance |
|---|---|---|---|
| v2 D2: gen–ref / floor | 0.031 / 0.024 | 4.5e-4 / 1.7e-4 | 0.0058 / 0.0033 |
| P1 + DiRecT-chart D2 | 0.032 / 0.020 | 5.5e-4 / 1.3e-4 | 0.0051 / 0.0033 |
Caveat: gen–ref compares 8 vs ~4 samples, gen–gen 8 vs 8; W1 between small samples is biased upward for the smaller
pair, so "exceeds the floor" overstates the gap. Use only as a diagnostic.

## Predictions vs observations
| id | prediction | result |
|---|---|---|
| F1 | matched ≥ 99%, 0 references without a feasible chart | **hit** (100% / 99.9%, 0) |
| F2 | v2 roughness ratio CI upper < 1 | **hit** (0.85 / 0.86) |
| F3 | v2 len_ratio gap CI contains 0 or \|gap\| < 0.01 | **hit** (−0.003 / −0.001) |
| F4 | v2 clearance ratio CI within [0.9, 1.1] | **hit** ([0.92, 0.94] / [0.93, 0.95]) |
| F5 | v2 diversity ratio CI within [0.85, 1.15] | **hit** (0.94–0.96) |
| F6 | p1chart diversity ratio < v2's | **hit** (0.83 vs 0.95; 0.82 vs 0.95) |
| F7 | equal-class and ref-weighted agree in direction | **hit** |
| F8 | D2 model-seed range of v2 roughness ratio < half its CI width | **hit** (0.007 vs 0.025) |
| F9 | median W1(len_ratio) gen–ref > gen–gen floor (D2) | **hit** (0.031 > 0.024; see caveat) |
Score 9/9.

## Reading
1. **Length.** v2 reproduces the reference length distribution within a class (gap ≤ 0.3% of J_c). P1 + DiRecT-chart is
   ~2% *shorter than the references*: Experiment B's "chart projection is shorter" is a departure from the reference
   distribution toward the class shortest path, not a fidelity gain. Whether shorter is better depends on the use;
   the planner is not optimal.
2. **Diversity.** v2 keeps 95% of the reference within-class spread; chart projection keeps 82–83%.
3. **Smoothness.** Both are smoother than the planner (planner paths are piecewise-random); chart projection more so.
4. **Clearance — a fidelity gap of v2.** v2's mean minimum clearance is 6–7% below the reference and 9–10 pp more of
   its samples come within c + 0.005 of an obstacle (reference hug rate is the baseline). Chart projection matches the
   reference clearance. v2 never violates c (by construction), but it sits closer to the band edges than the planner.
   Recorded as a limitation; no retraining (stop condition).
5. Within-class fidelity is measured against ~4 references per class, so per-class distributions are coarse; the
   aggregate statements rest on 6,000+ classes and layout-clustered intervals.

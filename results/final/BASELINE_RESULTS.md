# Experiment B — v2 vs P1 + DiRecT-chart: paired uncertainty and standardized cost (D2/D3)

Locked: `outputs/predictions_final_B.md` (sha256 9864be9f…). Code: `scripts/baseline_paired.py`, `scripts/baseline_boot.py`,
`scripts/baseline_latency.py`, `scripts/latency_score.py`. Data: `outputs/final/baseline/*.npz`; summaries
`outputs/final/baseline_paired.json`, `outputs/final/baseline_latency.json` (+ raw `.jsonl`).
Figure: `figs/pareto_time_diversity.png`. No checkpoint, evaluation definition or training changed.

Requests: the pre-validation comparison study request set (T1 selector, 32 codes per unseen layout, 300 layouts, 9,600 samples per run).
Arms: **v2** (chart-space flow, `V2_s{k}`) and **P1 + DiRecT-chart** (code-conditioned P1, `P1_s{k}`, each step
projected onto the same class chart). Noise seeds 777/778/779; D2 model seeds 0/1/2 (paired V2_sk with P1_sk), D3 model
seed 0. Intervals: 95% layout-clustered bootstrap (B = 4,000), each replicate also draws one noise seed.

## Safety (independent checker, `scripts/indep_geom.py`)
0 Euclidean clearance violations, 0 square-inflated-box violations, 0 crossing-word mismatches across all 8 settings × 2
arms × 3 noise seeds (460,800 samples). Smallest analytic clearance 0.010010 (c = 0.01). Both arms decode through the
chart, so this is expected; it credits the **chart representation**, not the chart-space flow specifically.

## Paired differences v2 − P1 + DiRecT-chart (pooled over noise seeds)
| setting | diversity | length ratio | roughness | min clearance | layouts with v2 more diverse |
|---|---|---|---|---|---|
| D2 NFE 5, m0 | +0.0210 [0.0202, 0.0218] | +0.044 [0.042, 0.045] | +1.96e-4 [1.83, 2.10] | −0.0132 [−0.0141, −0.0124] | 100% |
| D2 NFE 5, m1 | +0.0216 [0.0209, 0.0223] | +0.044 [0.042, 0.045] | +1.84e-4 | −0.0135 | 100% |
| D2 NFE 5, m2 | +0.0214 [0.0207, 0.0221] | +0.044 [0.042, 0.045] | +1.89e-4 | −0.0133 | 100% |
| D3 NFE 5, m0 | +0.0213 [0.0207, 0.0220] | +0.044 [0.042, 0.045] | +2.09e-4 [1.92, 2.26] | −0.0119 [−0.0127, −0.0110] | 100% |
| D2 NFE 20, m0 | +0.0082 [0.0072, 0.0092] | +0.020 [0.019, 0.022] | +9.4e-5 [8.1, 10.8] | −0.0055 [−0.0061, −0.0049] | 89% |
| D2 NFE 20, m1 | +0.0088 [0.0078, 0.0098] | +0.021 [0.019, 0.022] | +9.8e-5 | −0.0058 | 91% |
| D2 NFE 20, m2 | +0.0083 [0.0073, 0.0093] | +0.020 [0.019, 0.022] | +8.8e-5 | −0.0056 | 89% |
| D3 NFE 20, m0 | +0.0090 [0.0081, 0.0099] | +0.021 [0.019, 0.023] | +9.4e-5 [7.7, 11.6] | −0.0054 [−0.0060, −0.0048] | 89% |

Absolute diversity (NFE 20, m0): v2 0.061 vs 0.053 (D2), 0.061 vs 0.052 (D3); NFE 5: 0.052 vs 0.030. Relative range of
the difference across the 3 noise seeds: 1–3% (NFE 5), 5–18% (NFE 20). Per-seed values: `baseline_paired.json`.

## Standardized cost (one host, 1 CPU thread, batch 320, subprocess per configuration, median of 5 repeats; 3 if > 60 s)
| ms / sample | NFE 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| v2 D2 / D3 | 0.31 / 0.30 | 0.56 / 0.52 | 1.40 / 1.43 | 2.49 / 2.57 | 5.60 / 4.75 | 12.5 / 12.1 |
| P1 + DiRecT-chart D2 / D3 | – | 0.53 / 0.63 | 1.43 / 1.93 | 3.51 / 3.44 | 7.46 / 7.28 | 18.7 / 18.4 |
| P1 raw D2 / D3 | – | – | 1.12 / 1.22 | – | 4.96 / 4.80 | – |

Free-space projection arms (D2 / D3): P1 + final projection 20 it 15.6 / 14.0 (NFE 5), 19.0 / 18.1 (NFE 20); P1 + DiRecT
free-space 100 it 132 / 133 (NFE 5), 645 / 658 (NFE 20); B0 + free-space 100 it 129 / 132, 708 / 666. Chart construction
is inside the timed region for every chart-based arm. Peak RSS, IQR, warm-up: `baseline_latency.json`.

## Equal wall-clock (v2 at the largest NFE whose standardized time ≤ the baseline's)
Quality at NFE 5/20 is from this experiment (3 seeds); for other NFEs and the free-space arms it is the pre-validation comparison study single
noise seed on the same requests (labelled in `baseline_latency.json`).
- P1 + DiRecT-chart NFE 20 (7.5 / 7.3 ms) ↔ v2 NFE 20 (5.6 / 4.8 ms): diversity 0.061 vs 0.053 / 0.061 vs 0.052.
- P1 + DiRecT-chart NFE 5 ↔ v2 NFE 5; NFE 2 ↔ v2 NFE 1 (D2) / 2 (D3): v2 more diverse in every pair.
- P1 raw NFE 20 ↔ v2 NFE 10 (D2) / 20 (D3): v2 0 violations vs 3.7–3.8% clearance/class violations for P1 raw.
- Every free-space projection configuration (14–708 ms) ↔ v2 NFE 50 (12.1–12.5 ms): v2 exact and more diverse (0.064 vs
  0.031–0.054); B0 + free-space still has 0.03–0.47% violations and the wrong class distribution (pre-validation comparison study).
- **Not won by v2 at any matched budget: length and roughness** (chart projection shorter and smoother, see table above
  and Experiment C, where that "shorter" is shown to be a departure from the reference distribution).

## Predictions vs observations
| id | prediction | result |
|---|---|---|
| B1 | 0 independent-check violations, both arms, all settings | **hit** |
| B2 | NFE 20 diversity difference > 0, CI excludes 0, each noise seed, D2 and D3 | **hit** |
| B3 | NFE 5, same | **hit** |
| B4 | NFE 20 length-ratio difference > 0, CI excludes 0 | **hit** (+0.020 / +0.021) |
| B5 | NFE 20 roughness difference > 0, CI excludes 0 | **hit** |
| B6 | seed range of NFE 20 diversity difference < 25% of mean | **hit** (≤ 9% D2 m0, 6% D3) |
| B7 | D2: positive for all three model-seed pairs | **hit** |
| B8 | time ratio P1 + DiRecT-chart / v2 at NFE 20 in [1, 2] | **hit** (1.33 D2, 1.53 D3) |
| B9 | equal wall-clock: matched v2 diversity ≥ P1 + DiRecT-chart NFE 20 | **hit** (matched v2 NFE 20 on both) |
| B10 | P1 + DiRecT free-space (100 it, NFE 20) ≥ 30× v2 NFE 20 | **hit** (115× D2, 138× D3) |
Score 10/10.

## Reading (interpretation gates fixed in the lock)
1. **Feasibility belongs to the chart.** P1 + DiRecT-chart is as safe as v2 at 1.3–1.5× v2's cost. No claim that
   chart-space flow is uniquely safe.
2. **v2's advantage is a diversity–compute trade-off, not superiority.** v2 keeps a reproducible within-class diversity
   advantage (+15% at NFE 20, +70% at NFE 5; stable over noise and model seeds) and is cheaper at matched NFE, but its
   paths are ~2% longer, rougher, and sit closer to obstacles (−0.005 min clearance at NFE 20).
3. **Free-space projection (DiRecT-style) is 100×+ slower** for the same exactness, and the code-blind variant does not
   reach exactness or class fidelity.

## Caveats
1. Timing is one CPU type, one thread, batch 320, on a cloud container; absolute ms will move on other hardware. Ratios
   between arms measured in the same sweep are the reportable quantity. Earlier pre-validation comparison study times differ (different batch
   chunking and container host); the standardized ones replace them for cost claims.
2. D3 has one model seed. P1 and v2 were trained by us; DiRecT-chart is our adaptation, not the published DiRecT.
3. Diversity is measured among valid samples within class; at NFE 2 the P1-based arms are far from converged.

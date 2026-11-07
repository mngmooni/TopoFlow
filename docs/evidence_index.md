# Paper evidence index

This index connects the study claims with the checked-in **saved artifacts**. A JSON file verifies what a saved evaluation reported; it does not independently rerun the underlying training or sampling. Files under `results/final/` retain the frozen results and historical provenance metadata, including some source strings referring to pre-refactor `outputs/` paths.

## Experiment A — chart construction and soundness

| Question | Evidence | Public verification |
|---|---|---|
| How many route-code instances were audited? | `results/final/chart_coverage_audit.json` | `python -m tools.verify_evidence` |
| How many paths and segments were checked? | `results/final/chart_soundness.json` and `results/final/soundness/*.json` | `python -m tools.verify_evidence` |
| Did the independent checker report any violations? | `results/final/chart_soundness.json` | zero recorded Euclidean and square-inflated violations, subject to the test's conditions |
| Is chart coverage equivalent to neural-model coverage? | `results/final/CHART_SOUNDNESS.md` | **No**: construction coverage also includes stress layouts beyond the trained model's interface |
| Can Figure 1 be regenerated from saved metadata? | `results/final/chart_coverage_audit.json` + `experiments/validation/fig_coverage.py` | `python -m tools.replay_coverage_figure` (non-destructive) |

**Audited totals from the frozen JSON:** 2,100 layouts, 82,224 route-code instances, 695,488 decoded paths, and 42,631,824 complete line segments. The coverage categories sum to 49,138 default-chart successes, 2,624 additional refined successes, and 30,462 certified-infeasible route codes **within the tested restricted path family**. The category totals are derived from the individual family entries rather than copied from an informal summary table.

The empirical checker reported no violations in the audited sample. The theoretical segment-clearance guarantee is conditional on the chart's geometric assumptions. Homotopy identification additionally depends on the validity of the selected cut system; some stacked layouts lack this certificate.

## Experiment B — direct chart learning versus correction

| Question | Evidence | Notes |
|---|---|---|
| Matched comparison of v2 and chart-corrected P1 | `results/final/baseline_paired.json` | eight configuration groups × three sampling-noise seeds × two arms |
| Test count and empirical violations | `results/final/baseline_paired.json` | 460,800 recorded generated paths, zero measured violations for both chart-based arms |
| Inference cost / NFE tradeoff | `results/final/baseline_latency.json` and `.jsonl` | original host and batch configuration recorded; CPU-specific |
| Differences from prior constrained samplers | `results/final/BASELINE_RESULTS.md` | correction arm is an in-house rectified-flow adaptation, not original DiRecT |

Frozen paired and latency JSON can be inspected directly; regenerating their numerical estimates from raw sample archives requires the original `results/final/baseline/*.npz` files and trained checkpoints, which are **not bundled**.

## Experiment C — class-matched distribution fidelity

| Question | Evidence | Notes |
|---|---|---|
| Matching classes to available references | `results/final/distribution_fidelity.json` | D2: 6,240 matched; D3: 6,147 matched plus 5 chart-feasible without reference |
| Class-wise diversity / length / roughness / clearance | `results/final/distribution_fidelity.json` | saved aggregates and bootstrap intervals |
| Obstacle-proximity limitation | `results/final/FIDELITY_RESULTS.md` | observed distributional bias, not an identified cause |
| Supplementary fidelity visualization | `figures/fidelity_distributions.png` | raw class-level `.npz` data are not bundled for figure recomputation |

## Provenance and immutable outputs

`results/final/ARTIFACT_SHA256.txt` records the checksums of the shipped final structured evidence, paper figures, and curated validation reports. Run `python -m tools.verify_evidence` to check that they have not been accidentally edited.

The preserved output files are evidence of the reported studies. The public source refactor is **not** a re-execution of the original experiments and must not be presented as one. The original archive snapshots and any reconstructed Git history should be retained separately with their origin accurately disclosed.

## Additional encoding diagnostics

The original training bookkeeping counter uses nominal waypoint bounds. The effective encoding applies contracted widths and normalization clipping. A diagnostic on 300 training layouts measured nominal/effective affected-path rates of **7.2% / 7.9% on D2** and **56.8% / 57.4% on D4**. These measurements were supplied as a subsequent diagnostic; no corresponding raw data table is included in this candidate, so they are not part of the automated frozen-result check. No causal link between clipping and obstacle hugging is claimed.

### Experiment C metric implementation

`experiments/validation/fidelity_metrics.py` is the import-safe source of the class-level summary functions used by both `fidelity_boot.py` and `fig_fidelity.py`. `tests/test_fidelity_metrics.py` guards the numerical definitions on synthetic inputs. The historical `results/final/distribution_fidelity.json` and curated `figures/fidelity_distributions.png` remain frozen; synthetic-data regression establishes refactor equivalence, not reproduction of the archived study without the missing per-sample `.npz` inputs.

## Bootstrap analysis refactor verification

`experiments/validation/baseline_boot.py` and `fidelity_boot.py` now accept
explicit NPZ input directories and protected reproduction JSON destinations.
The original archived aggregates remain untouched. Legacy and refactored
programs produced byte-identical JSON reports when run against matching
synthetic per-run inputs at equal bootstrap budgets and seeds. This checks the
calculation refactor; it is not an independent replay of the unavailable
original Experiment B/C NPZ inputs.

# Chart soundness and coverage — proof note and audit (Experiment A)

Locked plan: `outputs/predictions_final_A.md` (sha256 0d74a340…). Code under test: `topoflow/chart.py` (unchanged).
Independent checkers: `scripts/indep_geom.py` (no import of chart internals, `Env.collision_mask`,
`Env.segments_hit` or `topoflow/topology.py`), unit tests `tests/test_chart_soundness.py` (7 pass).
Runs: `scripts/chart_soundness.py` → `outputs/final/soundness/{DS}.json`; `scripts/chart_coverage_audit.py` →
`outputs/final/chart_coverage_audit.json`. The earlier note `CHART_PROOF.md` is kept unchanged; this note restates
the claim with the assumptions the audit checks.

## 1. Claim (conditional; not unconditional safety)

**Setting.** Obstacles O_1..O_m are closed disks or closed axis-aligned boxes; clearance c > 0. The *chart geometry*
inflates disks to radius r + c and boxes to the square-cornered rectangle [cx ± (hw + c)] × [cy ± (hh + c)]
(this contains the Euclidean c-neighbourhood of the box, and is exactly the geometry of the M metric).
Start s and goal g are fixed; y is bounded by the chart convention Y_MIN = −1.2, Y_MAX = 1.2.

**Assumptions.**
- (G) *Grid.* Interior x-grid strictly increasing, x_s < x_1 < … < x_n < x_g (checked: `grid_strict_all` true on
  every chart of every dataset, default and bisected grids).
- (B) *Bands.* For each segment interval I_k, L_k (U_k) is an upper (lower) bound of the inflated walls of the
  above-passed (below-passed) obstacles over I_k, with Y_MIN ≤ L_k, U_k ≤ Y_MAX; lo_k = max(L_{k−1}, L_k),
  hi_k = min(U_{k−1}, U_k).
- (F) *Feasible chart.* lo_k < hi_k for every k, L_0 ≤ y_s ≤ U_0 and L_n ≤ y_g ≤ U_n.
- (D) *Decoder.* y_k = m_k + w_k tanh(u_k), w_k = (1 − ε)(hi_k − lo_k)/2, ε = 10⁻³, u_k finite or ±∞.

**Proposition S (clearance).** Under (G), (B), (F), (D), every decoded polyline keeps Euclidean distance ≥ c from
every obstacle; more strongly, it does not meet the open inflated set of any obstacle in the chart geometry.

*Proof.* |tanh u| ≤ 1 and ε > 0 give lo_k < y_k < hi_k (strictly, also for u = ±∞), so both endpoints of segment k
lie in R_k = I_k × [L_k, U_k] (endpoints of the first/last segment by (F)). R_k is convex, so the segment lies in R_k.
By (B), every point of an above-passed inflated obstacle over I_k has y ≤ L_k and every point of a below-passed one
has y ≥ U_k, so R_k meets no open inflated obstacle. An obstacle whose inflated x-range misses I_k does not meet the
strip over I_k. Points outside the open inflated set are at distance ≥ c from the obstacle. ∎

Note: (B) requires that *every* obstacle meeting I_k is constrained on one side. The implementation achieves it by
evaluating each wall's extremum over the closed interval (disk: at the interval point nearest the centre; box:
constant), for every obstacle, including obstacles near the endpoints — so Proposition S does **not** need the
endpoint assumption A3 of CHART_PROOF.md (unit test `test_obstacle_inside_endpoint_range`).

**Proposition W (route code ↔ reduced crossing word).** Additional assumptions:
- (A3) every inflated x-range [cx − R, cx + R] lies strictly inside (x_s, x_g);
- (Cut) cuts are the downward rays x = cx_j, y < (lowest point of O_j); they have distinct x, miss every other
  obstacle, and do not contain s or g (then they are disjoint simple rays to infinity, and the reduced word of signed
  crossings is a homotopy invariant of paths from s to g in the plane minus the obstacles);
- (Grid edges) the grid contains a point within 10⁻⁶ outside each inflated edge (implementation), so X_j is covered by
  whole segment intervals.
Then every decoded polyline (x-monotone by (G)) crosses cut j exactly once, left to right, if h_j = 1 and never if
h_j = 0, so its reduced crossing word is word(h) = (j : h_j = 1 in x order).

*Proof.* Under (A3) the cut x = cx_j lies in the interior of X_j, inside some interval I_k. If h_j = 0, the
polyline at x = cx_j has y > L_k ≥ top_j(cx_j) ≥ lowest point of O_j, so it does not meet the cut. If h_j = 1, it has
y < U_k ≤ bot_j(cx_j) < lowest point of O_j, so it meets the cut; x-monotonicity makes this the only crossing, and
it is left to right. Crossings are ordered by x, i.e. by cut index. ∎

What W does **not** say: the route code is a descriptor; it names a homotopy class only through W, i.e. only when
(Cut) holds, and only for paths of the x-monotone family. Codes that need non-monotone paths are outside the chart.
On layouts with invalid cuts (H1: 184, H4: 171 of 300 layouts — stacked obstacles whose downward cut passes
through a lower obstacle) the crossing word is not a homotopy invariant, even though the computed word still matched
the code on every decoded path there.

**Numerical scope.** The decoded clearance margin is at least ε·(hi − lo)/2 at waypoints; with the measured minimum
band width (D2/D3 ≥ 0.67, D4 ≥ 6.6·10⁻⁴ at n = 60, ≥ 4.4·10⁻⁵ on bisected grids) this is ≫ float64 rounding
(~10⁻¹⁶). Bands narrower than ~10⁻¹² would make the strict inequality depend on rounding; none occurred, and the
checker re-checks any segment with float margin < 10⁻⁹ in exact rational arithmetic (none was needed).
The guarantee covers the polyline only (no continuous tracking), static obstacles, and the latents fed to the
decoder (NaN latents are outside the claim).

## 2. Independent verification (A1)

Analytic segment-to-obstacle distance (disk: point-to-segment − r; box: 0 if meeting, else endpoint/corner
minimum), open square-rectangle test, independent reduced crossing word, exact-rational recheck below 10⁻⁹.
Latent battery per chart: 24 (D2/D3/D4) or 8 (H1–H4) latents incl. ±50 saturation; every chart-feasible code at
n = 60 plus every code recovered by local bisection (≤ 400 points), on its bisected grid.

| data | layouts | codes (all 2^m) | charts checked (bisected) | decoded paths | segments | Euclid. viol. | square viol. | word mismatch | min Euclid. margin | min square margin |
|---|---|---|---|---|---|---|---|---|---|---|
| D2 | 300 | 6,240 | 6,240 (0) | 149,760 | 9.1 M | 0 | 0 | 0 | 1.4e-6 | 1.0e-6 |
| D3 | 300 | 6,152 | 6,152 (0) | 147,648 | 9.0 M | 0 | 0 | 0 | 1.5e-6 | 1.0e-6 |
| D4 | 300 | 7,424 | 5,195 (420) | 124,680 | 7.7 M | 0 | 0 | 0 | 1.2e-6 | 2.3e-8 |
| H1 stacked | 300 | 5,632 | 3,298 (64) | 26,384 | 1.6 M | 0 | 0 | 0 | 9.0e-7 | 9.0e-7 |
| H2 tight gaps | 300 | 7,168 | 5,048 (305) | 40,384 | 2.5 M | 0 | 0 | 0 | 1.0e-6 | 9.9e-7 |
| H3 dense | 300 | 43,904 | 22,197 (1,835) | 177,576 | 10.9 M | 0 | 0 | 0 | 1.1e-6 | 6.6e-8 |
| H4 stacked boxes | 300 | 5,704 | 3,632 (0) | 29,056 | 1.8 M | 0 | 0 | 0 | 1.8e-7 | 1.8e-7 |

Margins are distance − c (Euclidean) and distance to the square-inflated rectangle; both strictly positive
everywhere. "0 violations" is evidence for the implementation of Proposition S on these inputs, not a proof; the
proof is §1, and it is conditional on (G)(B)(F)(D).

Setup checks: A3, disjointness, distinct cut x and cut validity hold on every D2/D3/D4 layout (minimum A3 margin
0.005 on D4); H1/H4 have invalid cuts as stated above; H2/H3 are valid.

## 3. Coverage and failure taxonomy (A2)

Class set: all 2^m codes per layout. δ(h) = min_x U(x) − L(x) of the class's exact pointwise free interval, computed
pairwise with convex 1D minimisation, independent of chart.py. δ < 0 comes with a witness (x*, i, j); every witness
was re-evaluated directly (0 failures). Under δ > 0 a continuous x-monotone path with clearance c exists
(L upper-, U lower-semicontinuous, insertion theorem); under δ < 0 none exists in the x-monotone family.

| data | codes | covered n = 60 | recovered by bisection (max pts) | certified infeasible | conservative excl. | Y-bound | degenerate | unrecovered | n = 60 share of δ > 0 |
|---|---|---|---|---|---|---|---|---|---|
| D2 | 6,240 | 6,240 | 0 | 0 | 0 | 0 | 0 | 0 | 100% |
| D3 | 6,152 | 6,152 | 0 | 0 | 0 | 0 | 0 | 0 | 100% |
| D4 | 7,424 | 4,775 | 420 (108) | 2,229 | 0 | 0 | 0 | 0 | 91.9% |
| H1 | 5,632 | 3,234 | 64 (98) | 2,334 | 0 | 0 | 0 | 0 | 98.1% |
| H2 | 7,168 | 4,743 | 305 (107) | 2,120 | 0 | 0 | 0 | 0 | 94.0% |
| H3 | 43,904 | 20,362 | 1,835 (121) | 21,707 | 0 | 0 | 0 | 0 | 91.7% |
| H4 | 5,704 | 3,632 | 0 | 2,072 | 0 | 0 | 0 | 0 | 100% |

Planner status (sampling planner; `not_found` is not infeasibility):

| data | found | found: covered n60 / bisect / certified infeasible | not found | not found: chart-feasible (planner false negative) / certified infeasible |
|---|---|---|---|---|
| D2 | 6,240 | 6,240 / 0 / 0 | 0 | – |
| D3 | 6,147 | 6,147 / 0 / 0 | 5 | 5 / 0 |
| D4 | 2,394 | 2,275 / 119 / 0 | 5,030 | 2,801 / 2,229 |
| H1–H4 | no planner reference | | | |

Reading:
- Inside the x-monotone family, every code falls in exactly one of two groups: covered (default grid or bisection)
  or certified infeasible. No class was excluded by the chart's conservatism (square box corners, Y bound), none is
  degenerate, and none with δ > 0 was left uncovered by bisection.
- The trained v2 field uses the default 60-point grid only, so the *model's* coverage is the n = 60 column
  (91.7–100% of feasible classes); classes that need bisection are certified but not generated by v2.
- D4: 5,195 classes with δ > 0 (the earlier variable-resolution coverage study count of 5,161 used the VR evaluation's capped refinement; the
  difference is refinement budget, not geometry).
- Planner false negatives: on D4 2,801 of 5,030 not-found codes have a feasible chart (constructive certificate of a
  valid path), 2,229 are certified infeasible.
- Old dense "exact" flags (CHART_PROOF.md §6) agree in sign with δ on every D2/D3 code (12,392 compared).

## 4. Predictions (locked) vs observations
| id | result |
|---|---|
| S1 0 Euclidean violations | **hit** |
| S2 exact recheck of margins < 1e-9 | **hit** (vacuous: no segment below 1e-9) |
| S3 0 square-rectangle meets | **hit** |
| S4 word = word(h) on valid-cut layouts | **hit** (also on invalid-cut layouts) |
| S5 A3 and cut validity on D2/D3/D4 | **hit** |
| S6 some hard layout has invalid cuts | **hit** (H1 184, H4 171) |
| S7 min band ≥ 1e-6 on D2/D3/D4 at n = 60 | **hit** (6.6e-4) |
| C1 no found code certified infeasible | **hit** |
| C2 D4 n = 60 covers 92.5 ± 1 pp of δ > 0 | **hit** (91.9%) |
| C3 every δ > 1e-6 covered (n60 or bisection) | **hit** |
| C4 some conservative-exclusion code in D4/H4 | **miss** (0) |
| C5 sign agreement with old dense flag | **hit** (0 disagreements) |
Score 11/12. The miss: square box corners never decided feasibility here; the square-vs-Euclidean difference only
matters within c of a box's x-edge, and no throat in these layouts was that tight.

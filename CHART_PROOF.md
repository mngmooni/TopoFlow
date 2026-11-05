# TDFM v2 class chart: assumptions, propositions, proofs, coverage

Code: `topoflow/chart.py` (`bounds_xs` = chart on any grid). Independent check: `scripts/chart_verify.py`.
Coverage measurements: `scripts/chart_coverage.py`, `scripts/chart_refine.py` (Day 12).

## 1. Setting and assumptions

**A1 (obstacles).** A finite set of pairwise disjoint closed obstacles O_1..O_m, each a disk (centre (a_j, b_j), radius r_j)
or an axis-aligned box (centre (a_j, b_j), half-sizes w_j, v_j). Obstacles are indexed by centre x.

**A2 (clearance).** c > 0. The inflated obstacle O_j^c is the disk of radius R_j = r_j + c, or the box with half-sizes
w_j + c, v_j + c (square corners; for boxes set R_j = w_j + c). Every point outside O_j^c is at Euclidean distance ≥ c
from O_j. (The square corners make the box case conservative.) X_j = [a_j − R_j, a_j + R_j] is the inflated x-range.
top_j(x), bot_j(x) are the upper and lower boundaries of O_j^c over X_j.

**A3 (endpoints).** Start s = (x_s, y_s) and goal g = (x_g, y_g) satisfy x_s < min_j (a_j − R_j) and
max_j (a_j + R_j) < x_g, and Y_MIN < y_s, y_g < Y_MAX.

**A4 (route family and classes).** Paths are x-monotone polylines from s to g. A route code h ∈ {0,1}^m says pass
O_j above (h_j = 0) or below (h_j = 1). The class of h is identified by its reduced crossing word (cuts = downward rays
from each obstacle; `topoflow/topology.py`). A path is **valid for (h, c)** if it stays outside every O_j^c and its
crossing word is word(h).

**A5 (grid).** An interior grid x_s = x_0 < x_1 < … < x_n < x_{n+1} = x_g that contains every inflated edge
a_j ± R_j. (The implementation puts each edge 10⁻⁶ outside the inflated range; see Remark 3.)

**A6 (decoder).** ε ∈ (0, 1); y_k = m_k + w_k tanh(u_k), m_k = (lo_k + hi_k)/2, w_k = (1 − ε)(hi_k − lo_k)/2.

## 2. Construction

For each segment interval I_k = [x_k, x_{k+1}], k = 0..n:

- L_k = max( Y_MIN, max over {j : h_j = 0, X_j meets I_k} of max_{x ∈ I_k ∩ X_j} top_j(x) )
- U_k = min( Y_MAX, min over {j : h_j = 1, X_j meets I_k} of min_{x ∈ I_k ∩ X_j} bot_j(x) )

(for a disk the extremum is at the point of I_k nearest a_j; for a box it is the constant b_j ± (v_j + c)).
Waypoint chart: lo_k = max(L_{k−1}, L_k), hi_k = min(U_{k−1}, U_k) for k = 1..n.

**Definition (feasible chart).** The chart of (env, h, c, grid) is feasible if lo_k < hi_k for all k = 1..n,
L_0 ≤ y_s ≤ U_0 and L_n ≤ y_g ≤ U_n. Otherwise it is infeasible (see §5 for what happens then).

## 3. Soundness

**Lemma 1 (segments stay in rectangles).** If the chart is feasible, then for every u ∈ ℝⁿ the k-th segment of the
decoded polyline lies in R_k = I_k × [L_k, U_k], for k = 0..n.

*Proof.* |tanh| < 1 and ε > 0 give lo_k < y_k < hi_k, so y_k ∈ [L_{k−1}, U_{k−1}] ∩ [L_k, U_k]. The endpoints
y_0 = y_s and y_{n+1} = y_g are in the first and last bands by feasibility. Both endpoints of segment k are in the
convex set R_k, so the whole segment is. ∎

**Lemma 2 (rectangles are free).** R_k contains no point of the interior of O_j^c for any j; decoded interior
waypoints lie strictly inside the bands.

*Proof.* If X_j does not meet I_k, O_j^c does not meet the strip I_k × ℝ. Otherwise, if h_j = 0, every point of
O_j^c with x ∈ I_k has y ≤ top_j(x) ≤ L_k, so O_j^c ∩ R_k lies on the line y = L_k; the interior of O_j^c misses R_k.
If h_j = 1 the argument is symmetric with U_k. ∎

So a decoded polyline never enters the interior of an inflated obstacle, i.e. its distance to every O_j is ≥ c.
(Touching the boundary at distance exactly c is possible only at the fixed endpoints, which are excluded by A3;
the checker's minimum observed clearance is c + 10⁻⁶.)

**Lemma 3 (side and crossing word).** Over X_j, every point of the decoded polyline is above top_j (h_j = 0) or below
bot_j (h_j = 1).

*Proof.* By A5, X_j is a union of whole segment intervals, and for each of them Lemma 1–2 put the segment in R_k with
L_k ≥ top_j (h_j = 0) or U_k ≤ bot_j (h_j = 1) on I_k. ∎

The polyline is x-monotone (x_k strictly increasing), so it crosses each vertical cut below O_j exactly when it
passes O_j below, once, and the reduced crossing word is word(h).

**Proposition 1 (soundness).** Under A1–A6, if the chart of (env, h, c, grid) is feasible, then for **every**
u ∈ ℝⁿ the decoded polyline is valid for (h, c). This holds for any grid satisfying A5, so refining the grid (§5)
keeps the guarantee.

*Proof.* Lemma 2 gives clearance ≥ c. Lemma 3 and x-monotonicity give the word. ∎

**Corollary 1 (feasibility certificate).** A feasible chart is a constructive proof that class h contains a valid
x-monotone path at clearance c (decode u = 0). This is how §6 finds planner false negatives.

## 4. Class completeness under refinement

Soundness says nothing about classes whose chart is infeasible. The next result says those are a resolution
problem whenever the class has a free tube of positive width.

On each open cell between consecutive inflated edges, the active set {j : x ∈ X_j} is constant, and L(x) =
max(Y_MIN, max_{active, h_j = 0} top_j(x)), U(x) = min(Y_MAX, min_{active, h_j = 1} bot_j(x)) extend continuously to
the cell's closure. At a point x write L⁻, L⁺ (U⁻, U⁺) for the limits from the two adjacent cells (equal away from
edges). Define the **tube width**

  δ(h) = min_x [ min(U⁻(x), U⁺(x)) − max(L⁻(x), L⁺(x)) ].

δ(h) > 0 means there is a free x-monotone tube of class h of width ≥ δ everywhere.

**Lemma 4 (modulus of the walls).** Within one cell, L and U have modulus of continuity ω(t) = √(2 R_max t),
R_max = max_j R_j (disk walls); box walls are constant.

*Proof.* For a disk, |√(R² − p²) − √(R² − q²)| ≤ √|p² − q²| = √(|p − q| |p + q|) ≤ √(2R |p − q|), using
|√α − √β| ≤ √|α − β|. A max (or min) of functions with modulus ω has modulus ω. ∎

**Proposition 2 (refinement completeness).** Under A1–A5 with the grid containing the inflated edges exactly, if
δ(h) > 0 and every waypoint x_k has both adjacent segment lengths < h* = δ(h)² / (8 R_max), the chart is feasible.

*Proof.* Each segment lies in one cell closure (edges are grid points). By Lemma 4, L_{k−1} ≤ L⁻(x_k) + ω(h_{k−1}) and
L_k ≤ L⁺(x_k) + ω(h_k), so lo_k ≤ max(L⁻, L⁺)(x_k) + ω(max(h_{k−1}, h_k)); symmetrically
hi_k ≥ min(U⁻, U⁺)(x_k) − ω(·). Hence hi_k − lo_k ≥ δ − 2ω(·) > 0 because ω(t) < δ/2 ⟺ t < δ²/(8 R_max).
The first and last segments meet no inflated obstacle (A3, A5), so L_0 = L_n = Y_MIN, U_0 = U_n = Y_MAX and the
endpoint condition holds. ∎

The bound is local, so only the waypoints near a narrow throat need short segments. **Local bisection** (bisect both
segments next to each failing waypoint, repeat) therefore terminates for δ > 0: a segment shorter than h* is never
adjacent to a waypoint that fails because of it, and only finitely many segments of length ≥ h*/2 can be created.

*Remarks.*
1. The bound is loose (uniform n = 60 gives mesh 0.03, which the bound certifies only for δ > 0.21), but D1 classes
   with δ ≈ 0.013 are already feasible after adding ≤ 14 points locally (§6).
2. δ(h) = 0 (a throat closing to a point) or < 0 (closed) means no class-h x-monotone path with clearance c
   exists in the open-tube sense; then no grid works, and the class is truly uncovered.
3. The implementation's 10⁻⁶ edge offset adds a sliver segment; it moves the bands by at most ω(10⁻⁶) ≈ 6·10⁻⁴.
   Soundness (Prop. 1) is unaffected; Prop. 2 holds with δ replaced by δ − 1.2·10⁻³.

## 5. What happens when a chart is infeasible (policy fixed before the Day 12 runs)

1. Build the chart on the default grid (n = 60).
2. If infeasible, run local bisection up to a budget (Day 12 measured ≤ 74 points total).
3. If still infeasible, the code is **uncovered**: v2 does not generate it, the selector's support is restricted to
   covered codes and renormalized, and the lost target mass is reported.

Caveat for step 2: the trained v2 field has a fixed number of grid tokens (60). A refined grid with more points needs
either a variable-length token interface or a re-placement of the base points that keeps n = 60 (G2 below).
This was **not** evaluated with a trained model yet; all Day 12 numbers are construction-level.

## 6. Measured coverage (Day 12; 300 unseen + first 300 train layouts per dataset)

"Exact" = δ(h) > 0 checked on a dense grid (4000 points plus both sides of every inflated edge).

| | D1 unseen | D1 train | D2 unseen | D3 unseen |
|---|---|---|---|---|
| clearance c | 0.03 | 0.03 | 0.01 | 0.01 |
| found codes | 4228 | 4329 | 6240 | 6147 |
| found: chart feasible, n = 60 | 98.9% | 98.5% | 100% | 100% |
| found: exact tube nonempty | 100% | 100% | 100% | 100% |
| infeasible found charts fixed by uniform n = 120 / 240 | 16 / 19 of 45 | 39 / 61 of 65 | – | – |
| … by G2 (edges ± 0.005, n = 60) | 27 / 45 | 61 / 65 | – | – |
| … by local bisection | **45 / 45** (≤ 74 pts) | **65 / 65** (≤ 71 pts) | – | – |
| medium p* mass on uncovered codes (mean / max per layout) | 0.6% / 84% | 0.9% / 45% | 0 | 0 |
| not_found codes | 1980 | 2231 | 0 | 5 |
| not_found with feasible chart (planner false negative) | **1147** (+20 after bisection) | 1324 (+25) | – | 5 |
| chart feasible but exact says no | 0 | 0 | 0 | 0 |
| waypoint band / exact free width (median) | 0.987 | 0.988 | 0.989 | 0.989 |
| reference paths entirely inside the chart | 74.8% | 74.3% | 92.5% | 92.4% |

Reading:
- **Class coverage.** Every infeasible chart of a found code in D1 is a resolution artefact: all involve two
  x-adjacent obstacles whose inflated x-ranges overlap and whose code bits differ (a diagonal throat). Their exact
  tube width is small (median ≈ 0.02 for the ones the fixed grids miss), and local bisection fixes all of them.
  With bisection, the chart covers 100% of exactly feasible x-monotone classes in D1, D2 and D3.
- **Mass.** The uncovered mass is small on average but concentrated: in a few layouts the cheapest route goes
  through the throat and carries most of p*. Dropping uncovered codes is not harmless there; bisection is needed.
- **Planner labels.** 58% of D1's not_found codes are certified feasible at c = 0.03 by the chart. Day 9's reading
  ("not_found ≈ gap narrower than 0.03") was wrong for most of them: they are failures of the sampling planner. The
  remaining ~41% have no x-monotone tube at clearance 0.03.
- **Within-class coverage.** Bands are ~99% of the exact free width at grid points, but planner paths hug the wall,
  so 25% (D1) / 7.5% (D2, D3) of them touch the lost 1% and get clipped. Within-class completeness fails; its
  measured cost is in PREDICTIONS.md (Day 11 clipping section).

### Hard layouts (Day 12 step 3; c = 0.01, 300 layouts each; full table in PREDICTIONS.md)
| family | exact-feasible codes | n = 60 covers | + local bisection (max pts) |
|---|---|---|---|
| H1 stacked obstacles | 3298 | 98.1% | 100% (98) |
| H2 tight x-gaps 0–0.04 | 5048 | 94.0% | 100% (107) |
| H3 dense, m = 6–8 | 22197 | 91.7% | 100% (121) |
| H4 stacked boxes | 3632 | 100% | – |
Chart feasible but exact tube empty: 0 everywhere. The fixed-grid limit is resolution at circle throats; the trained v2
(60 tokens) only reaches the n = 60 column.

## 7. Independent numerical check (scripts/chart_verify.py)

The checker uses none of the chart's code and none of `Env.collision_mask` / crossing-word code: 64 points per
segment, exact Euclidean distance to disks and boxes (rounded-corner SDF), side test against each obstacle's centre
over its x-extent; u from N(0, 1), N(0, 9), ±50 (saturated), and ±50/0 mixtures; 24 per chart.

| data | c | split | charts | feasible | decoded paths | clearance violations | side violations | min clearance |
|---|---|---|---|---|---|---|---|---|
| D2 | 0.01 | train (300 layouts) | 6219 | 100% | 149,256 | 0 | 0 | 0.010001 |
| D2 | 0.01 | unseen (300) | 6240 | 100% | 149,760 | 0 | 0 | 0.010001 |
| D1 | 0.03 | train (300) | 4329 | 98.5% | 102,336 | 0 | 0 | 0.030002 |
| D1 | 0.03 | unseen (300) | 4228 | 98.9% | 100,392 | 0 | 0 | 0.030002 |
| D3 | 0.01 | train + unseen | – | 100% | ~305,000 | 0 | 0 | 0.010001 |

## 8. Scope (what the propositions do not cover)

- **Within-class completeness fails** (clipping, §6); Prop. 2 is about classes, not about every path in a class.
- **Route family.** Only x-monotone routes with start left and goal right of every obstacle (A3, A4). Classes that
  need backtracking are outside the chart by construction.
- **Shapes.** Disks and axis-aligned boxes only. Rotated boxes / convex polygons need their own top/bottom maxima
  (concave piecewise linear, so the interval max is still easy) but are not implemented.
- **Disjointness (A1).** Overlapping obstacles merge into one shape; their per-obstacle codes are then not
  independent classes.
- **Polyline only.** A robot tracking the polyline continuously is a separate question.

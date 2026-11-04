# TDFM v2 chart: soundness statement, proof, and what it does not cover

Code: `topoflow/chart.py`. Independent check: `scripts/chart_verify.py`.

## Setting
- Obstacles: disjoint circles (centre `(cx, cy)`, radius `r`) and axis-aligned boxes (centre, half-sizes `hw, hh`).
- Clearance `c > 0`. Inflated obstacle `O^c`: circle of radius `r + c`; box with half-sizes `hw + c, hh + c` (square
  corners). Every point outside `O^c` has Euclidean distance ≥ c from `O` (for boxes the square-corner inflation
  contains the Euclidean `c`-neighbourhood, so it is stricter).
- Route code `h ∈ {0,1}^m`: `h_j = 0` means "pass obstacle j above", `1` means "below".
- Grid: `x_0 = start.x < x_1 < … < x_n < x_{n+1} = goal.x`. The interior grid contains every x-edge of every `O^c_j`
  (the edges sit just outside, by 1e-6).

## Construction
For each segment interval `I_k = [x_k, x_{k+1}]`, `k = 0..n`:
- `L_k = max( Y_MIN, max_{j: h_j = 0, O^c_j overlaps I_k} max_{x ∈ I_k} top_j(x) )`
- `U_k = min( Y_MAX, min_{j: h_j = 1, O^c_j overlaps I_k} min_{x ∈ I_k} bottom_j(x) )`

where `top_j / bottom_j` are the upper / lower boundaries of `O^c_j` (for a circle, max of the top is attained at the
point of `I_k` closest to `cx`; for a box it is the constant `cy ± (hh + c)`).
Waypoint chart: `lo_k = max(L_{k-1}, L_k)`, `hi_k = min(U_{k-1}, U_k)`, `k = 1..n`.
The chart is **feasible** if `lo_k < hi_k` for all k and start/goal y lie in `[L_0, U_0]` / `[L_n, U_n]`.
Decoding: `y_k = m_k + w_k tanh(u_k)`, `m_k = (lo_k + hi_k)/2`, `w_k = (1 − ε)(hi_k − lo_k)/2`, `ε = 10⁻³`.

## Claim (soundness)
If the chart of `(env, h, c)` is feasible, then for every `u ∈ ℝⁿ` the polyline
`(x_0, y_0), (x_1, y_1), …, (x_{n+1}, y_{n+1})` with `y_0 = start.y`, `y_{n+1} = goal.y`:
1. does not enter any `O^c_j` (so it keeps clearance ≥ c from every obstacle), and
2. passes every obstacle j on the side `h_j`, hence its crossing word is the one of code `h`.

## Proof
1. *Each segment stays in a rectangle.* For `k = 1..n`, `y_k ∈ (lo_k, hi_k) ⊆ [L_{k−1}, U_{k−1}] ∩ [L_k, U_k]`
   because `|tanh| < 1`. The endpoints are inside by feasibility. So both endpoints of segment k lie in
   `R_k = I_k × [L_k, U_k]`, and since `R_k` is convex, the whole segment lies in `R_k`.
2. *Each rectangle is free.* Take an obstacle j whose `O^c_j` meets the strip `I_k × ℝ`. If `h_j = 0`, every point
   of `O^c_j` with x ∈ I_k has `y ≤ top_j(x) ≤ L_k`; the rectangle has `y ≥ L_k`; they can meet only on the boundary
   line `y = L_k`, and the decoded points satisfy `y > lo_k ≥ L_k` strictly (ε > 0) except at fixed start/goal,
   which are far from obstacles in our layouts. Symmetric for `h_j = 1`. Obstacles not meeting the strip cannot meet
   `R_k`. So `R_k ∩ O^c_j` contains no decoded point, for every j. → (1).
3. *Side.* Over the x-extent of `O_j`, every segment lies above `top_j` (code 0) or below `bottom_j` (code 1) by
   step 2, so the path passes j on side `h_j`. The path is x-monotone (x_k strictly increasing), so it crosses each
   obstacle's downward cut at most once, and only when it passes below; the reduced crossing word is therefore
   exactly the code's word. → (2). ∎

Why the obstacle x-edges are grid points: a route that switches side between neighbouring obstacles needs a segment
that is unconstrained by both. With the edges on the grid, the interval between `O^c_j`'s right edge and
`O^c_{j+1}`'s left edge is its own segment (or several) and meets neither strip. Without them, 15% of D2 charts were
infeasible at n = 60.

## Independent numerical check (scripts/chart_verify.py)
The checker uses none of the chart's code and none of `Env.collision_mask` / crossing-word code: dense points
(64 per segment), exact Euclidean distance to circles and boxes (rounded-corner SDF), side test against each
obstacle's centre over its x-extent. u from N(0, 1), N(0, 9), ±50 (saturated), and ±50/0 mixtures; 24 per chart.

| data | c | split | charts | feasible | decoded paths | clearance violations | side violations | min clearance |
|---|---|---|---|---|---|---|---|---|
| D2 | 0.01 | train (300 layouts) | 6219 | 100% | 149,256 | 0 | 0 | 0.010001 |
| D2 | 0.01 | unseen (300) | 6240 | 100% | 149,760 | 0 | 0 | 0.010001 |
| D1 | 0.03 | train (300) | 4329 | 98.5% | 102,336 | 0 | 0 | 0.030002 |
| D1 | 0.03 | unseen (300) | 4228 | 98.9% | 100,392 | 0 | 0 | 0.030002 |

The minimum clearance equals c up to the ε shrink, so the bound is tight: saturated u decodes onto the band edge.

## What the claim does not cover
- **Completeness fails.** The bands are conservative (max of the lower wall over a whole segment interval), so some
  valid paths of class h fall outside the chart. On D2, 7.3% of training paths had to be clipped into it; on D1
  (c = 0.03, data hugging the margin), 25%. Some classes have no feasible chart at all (D1: 1–1.5%; D2: 0%).
- **Route family.** Only x-monotone paths with start left of all obstacles and goal right of them. Non-monotone
  routes (e.g. winding around an obstacle) are outside the chart by construction.
- **Obstacle shapes.** Exact bands are implemented for circles and axis-aligned boxes only.
- **Fixed start/goal height inside the first/last band** is assumed; true in D1/D2 (obstacles are in x ∈ [−0.66, 0.66],
  start/goal at x = ±0.9). D3 must re-check it.
- The proof is about the decoded polyline. A robot's continuous trajectory tracking that polyline is a separate question.

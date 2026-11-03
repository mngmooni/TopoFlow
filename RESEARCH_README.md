# topoflow — Day 1–2: environment, route family, topology validator

```
pip install numpy scipy matplotlib pytest
python -m pytest -q            # 18 tests
python scripts/day1_demo.py    # writes outputs/day1/
```

## Modules
- `topoflow/env.py` — disjoint circles / axis-aligned boxes, downward vertical cuts,
  `Env.validate()` (cut hits another obstacle, shared cut x, start/goal on a cut, overlap),
  vectorized segment collision (`path_collides`, `collision_mask` for (N,T,2) batches),
  `random_layout()` for E2 (non-overlapping x-projections ⇒ cuts always valid).
- `topoflow/topology.py` — ordered signed cut crossings → stack free reduction →
  reduced word. Reference path = pass above everything (word `e`). Lowercase = generator,
  uppercase = inverse. `winding_vector` = abelianization (for showing what it loses).
- `topoflow/routes.py` — x-monotone route family per bit code (0 above / 1 below),
  arc-length resampling to T waypoints, validation on the **final T-point polyline**,
  `enumerate_codes` (`found` / `not_found`, never "infeasible"), `make_dataset`
  with class imbalance.

## Data format (`outputs/day1/*.npz`)
`X` (N,T,2) incl. endpoints · `code` (N,m) · `class_id` · `classes` · `words` ·
`start`, `goal` · `obstacles` (m,5) = [kind(0 circle/1 box), cx, cy, half_w, half_h] · `obstacle_mask`.
Model-facing vector: `Dataset.interior()` → (N, (T−2)·2); endpoints are fixed and excluded.

## Known limitations
- The within-class distribution is whatever this sampler produces (via points + low-freq
  wiggle + smoothing). It is a synthetic target, not a planner's distribution.
- Cut convention: points exactly on x = cx count as "right" (half-open).
- In this x-monotone family, code ↔ word agreement holds by construction; the validator
  does its real work on generated (model) paths.

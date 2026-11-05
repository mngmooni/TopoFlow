# Baseline phase results (E0, E2) 
All numbers: collision % / deep-penetration % (> 1/3 of obstacle size) / adherence %.
E2 eval is paired: every model sees identical (layout, code, noise) triples.
Held-out combinations: ABA, BAB on 3-obstacle layouts (never trained on, except *-full models).

## E0 (single obstacle, fixed layout)
- Low NFE failure is a property of the probability path: the exact unconditional oracle
  collides 100% at NFE 1 and 41–43% at NFE 2; the exact class-conditional oracle collides 0%.
- Learned B1 reproduces this (NFE 2: 0–0.05% vs B0 44%).
- High-NFE 2–3% collisions are shallow grazes (data clearance 0.01), present with or without
  conditioning. Deep penetrations (> 1/3 obstacle size) appear only for B0 (~0.5%); this is a geometric severity measure, not a topological label (the word of a colliding path is undefined).
- Earlier "jitter" and "minority collapse" were undertraining (gone at 512 units / 30k steps).

## E2 (3000 random layouts, 1–3 obstacles), NFE 5

| model | params | steps | unseen positions | unseen combinations |
|---|---|---|---|---|
| B1 padded MLP | 0.87M | 30k | 5.0–5.6 / 0.2–0.4 / 99.7 | 19.4–22.6 / 1.2–2.6 / 98.3–98.7 |
| B1 padded MLP | 0.87M | 90k | 1.1 / 0.05 / 100 | 8.3 / 0.3 / 99.6 |
| B1-full (sees ABA/BAB) | 0.87M | 30k | 5.3 / 0.2 / 99.8 | 7.8–8.4 / 0.25 / 99.4–99.8 |
| **P1 obstacle-token attention** | 0.84M | 30k | **0.0–0.1 / 0 / 100** | **0.0–0.1 / 0 / 100** |

(ranges = 2 seeds; B1 90k is 1 seed). NFE 2 and 20 show the same ordering; P1's worst cell is
unseen combinations at NFE 20: 0.7–1.1% collision, 0 deep.

Fidelity vs reference paths (unseen combinations, NFE 5): length ratio P1 0.98, B1 0.96;
nearest-reference RMS P1 0.030, B1 0.046, reference-reference 0.035 → P1 is not detouring.

Diversity (mean pairwise RMS within an item): reference 0.066, P1 0.038–0.039, B1-90k 0.039,
B1 0.049. All learned models under-disperse. At matched diversity (P1 vs B1-90k) collisions
on unseen combinations are 0–0.1% vs 8.3%.

## B0 (environment-only conditioning) on E2
NFE 20: 43–50% collision, 25–32% deep; NFE 50 the same → not integration error.
1 obstacle 27–32%, 3 obstacles 53–61%. Capacity not yet tested (open).

## Caveats
- 2 seeds (1 for B1-90k). P1 used ~2.5x the wall-clock of B1-90k on this CPU.
- P1 was never trained "full", so P1's own combination gap is only bounded (≈0 at NFE 5).
- Under-dispersion of all learned models is unexplained (low-NFE Euler, or undertraining).
- Obstacle-token cross-attention is a standard architecture; the result says the padded MLP
  is a poor structure here, not that this design is novel.

## TDFM controlled comparison (v0.6) 
B0-attn = P1 backbone with route bits zeroed, trained on the cost-aware target marginal (30k steps, seed 0).
B2 = MLP selector (8 logits, structural mask) + the existing P1 checkpoint (2 seeds). 300 unseen layouts x 32
samples, paired noise. Joint TV floor from 32 samples: uniform 0.145 / medium 0.089 / concentrated 0.018.

Collision % / joint TV:

| regime | model | NFE 2 | NFE 5 | NFE 20 |
|---|---|---|---|---|
| uniform | B0-attn | 80.1 / 0.801 | 17.9 / 0.259 | 4.1 / 0.181 |
| uniform | B2 | 0.2 / 0.153 | 0.02–0.09 / 0.153 | 0.3–0.4 / 0.154 |
| medium | B0-attn (seeds 0–1) | 36.2–36.4 / 0.374 | 7.0–7.3 / 0.127–0.129 | 1.8–2.0 / 0.098–0.099 |
| medium | B2 | 0.05–0.09 / 0.105 | 0 / 0.104 | 0.2–0.3 / 0.106 |
| concentrated | B0-attn | 9.5 / 0.104 | 2.0 / 0.046 | 0.85 / 0.041 |
| concentrated | B2 | 0.05–0.12 / 0.066 | 0 / 0.066 | 0.2–0.3 / 0.068 |

Selector TV to target: 0.030 / 0.042 / 0.060 (learned feasibility). Layout-matched B0-attn retrain (medium):
same numbers within noise. Full detail: HANDOFF.md §4.4 and outputs/day7/results.jsonl.

## Day 8 — selector variants, OT-CFM baseline 
Selector TV to p* (learned feasibility): MLP 512x4 0.019 / 0.022 / 0.030; **cost regression
(predict J per code + feasibility head, p = softmax(-beta J_hat), one model for all beta) 0.016 / 0.013 / 0.013**;
compositional set encoder 0.029 / 0.023 / 0.024 (uniform / medium / concentrated).

B2-costreg (P1 seeds 0–1) vs B0-attn, collision % / joint TV:

| regime | model | NFE 2 | NFE 5 | NFE 20 |
|---|---|---|---|---|
| uniform | B0-attn | 80.1 / 0.801 | 17.9 / 0.259 | 4.1 / 0.181 |
| uniform | B2-costreg | 0.25–0.27 / 0.156 | 0.02–0.11 / 0.155 | 0.34–0.42 / 0.157 |
| medium | B0-attn (seeds 0–1) | 36.2–36.4 / 0.374 | 7.0–7.3 / 0.127–0.129 | 1.8–2.0 / 0.098–0.099 |
| medium | B0-attn + OT-CFM | 15.6 / 0.185 | 4.1 / 0.110 | 2.1 / 0.100 |
| medium | B2-costreg | 0.01–0.02 / 0.091 | 0 / 0.091 | 0.2–0.3 / 0.093 |
| concentrated | B0-attn | 9.5 / 0.104 | 2.0 / 0.046 | 0.85 / 0.041 |
| concentrated | B2-costreg | 0.01 / 0.024 | 0 / 0.024 | 0.24–0.33 / 0.026 |

B2-costreg wins all 9 cells on both collision and joint TV (oracle floor 0.145 / 0.089 / 0.018); deep = 0 everywhere.
OT-CFM = within-layout minibatch OT (16 layouts x 16 samples), same backbone, seed 0: halves low-NFE failure,
no gain at NFE 20. Invalid mass of B2 on uniform (~1.5%) comes from feasibility-head errors.

## Day 9 : pre-registered checks
- OT 4x64 per layout: NFE 2/5/20 collision 12.6 / 3.8 / 2.0% (16x16: 15.6 / 4.1 / 2.1).
- Oracle decomposition (E2, 100 layouts): exact code-blind field NFE 2 Euler 39.6% (B0-attn 42.5%); exact code-conditional 0%
  at all NFE; high-NFE plateaus (B0 1.5%, OT 2.6%, P1 ~1%) are learned-field error.
- D1 (3–5 obstacles): B0-attn 55.1 / 20.7 / 5.6%, monotone in obstacle count; B2 0.06 / 0.04 / 0.39%; P1+oracle 0 / 0 / 0.33%.
- not_found codes are ~all realizable at zero clearance, 6.7% at the planner's 0.03 clearance → clearance mismatch between
  target (0.03) and validity check (0) inflates B2's invalid rate in D1.

## Day 11 : TDFM v2 chart flow, seeds, proof, clipping, D3
- D2 flow-time: 93% of P1 failures enter violation at t ≥ 0.7; exact field after 0.75 → 0%. High NFE = diversity.
- Full-path projection: P1 NFE 20 U+M 4.18 → 0.21%, diversity kept. Clearance loss w=100: broken (U 37%).
- v2 (D2, 3 seeds): U = M = T = 0 at NFE 2/5/20; TV 0.160 (projection 0.160); diversity 0.034/0.052/0.062
  vs P1 0.013/0.031/0.052 (data 0.064); params 838,918 vs 836,614; same generation time.
- Independent chart check: 0 violations over ~800k decoded paths (D1, D2, D3).
- Clipping: 7.3% of training paths at n = 60, ~1/n, 99.98% circle flanks; no measurable pull of v2 samples.
- D3: v2 0/0/0, diversity 0.061 vs P1 0.051; B2 0.11/0.06/1.76% vs B0-attn 65.1/27.6/10.2% collision.

## Day 12: chart proof, coverage, hard layouts
See CHART_PROOF.md (Prop. 1–2, §6 coverage), PREDICTIONS.md (Day 12), RELATED.md.
- D1 infeasible charts (1.1–1.5% of found codes): all throats between x-overlapping inflated neighbours; local
  bisection fixes 100% (≤ 74 pts). 58% of D1 not_found codes are chart-certified feasible (planner false negatives).
- Hard layouts, n = 60 coverage of exact-feasible classes: boxes 100%, stacked 98.1%, tight 94.0%, dense 91.7%;
  bisection 100% everywhere (≤ 121 pts). 0 chart-feasible-but-empty.
- OOD (D2-trained): P1 73–82% invalid on H1/H2/H4, v2 0%.

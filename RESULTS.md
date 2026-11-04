# Baseline phase results (E0, E2) — 2026-10-29

All numbers: collision % / deep-penetration % (> 1/3 of obstacle size) / adherence %.
E2 eval is paired: every model sees identical (layout, code, noise) triples.
Held-out combinations: ABA, BAB on 3-obstacle layouts (never trained on, except *-full models).

## E0 (single obstacle, fixed layout)
- Low NFE failure is a property of the probability path: the exact unconditional oracle
  collides 100% at NFE 1 and 41–43% at NFE 2; the exact class-conditional oracle collides 0%.
- Learned B1 reproduces this (NFE 2: 0–0.05% vs B0 44%).
- High-NFE 2–3% collisions are shallow grazes (data clearance 0.01), present with or without
  conditioning. Deep penetrations (only B0, ~0.5%) are the topological failures.
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

# Where TDFM v2 sits: what is topology-specific (Day 12 draft)

Status: positioning draft, not a systematic survey. Items marked (✓) were checked online on 2026-11-02 (abstract or
paper page); the others are from memory and must be checked before citing.

## 1. Four families of "generate inside a constraint set"

| family | examples | how the constraint is met | hard guarantee? | needs |
|---|---|---|---|---|
| post-hoc / in-loop projection or filtering | projected diffusion; DiRecT (Giaretta, Li, Azizan 2026) (✓); safety filters in diffusion planners, e.g. Mao et al. 2025 (✓) | solve an optimization or filter on the predicted clean sample | conditional: holds if the last subproblem is feasible (DiRecT Prop. 1) | a projector / solver per sample; can change class or shrink diversity |
| barrier / CBF guidance | SafeDiffuser (Xiao et al. 2023); PC-Diffuser (Ku, Lyu 2026) (✓) | CBF-QP inside denoising | no in practice (slacks, residual collisions 10% in PC-Diffuser) | differentiable barrier |
| reparameterized domain | reflected diffusion (Lou, Ermon 2023), mirror diffusion (Liu et al. 2023) (✓), log-barrier / reflected FM on constrained manifolds (Fishman et al. 2023) | generate in a space that maps onto the set (mirror map) or reflect at its boundary | yes, by construction | a **convex or simple** domain with a tractable mirror map / reflection |
| corridor planning (no learning of a distribution) | safe flight corridors (Liu et al. 2017), IRIS (Deits, Tedrake 2015), Graphs of Convex Sets (Marcucci et al. 2023) (✓) | choose a sequence of convex free regions, optimize one path inside | yes (convex constraints) | region decomposition; returns one optimal path, not a distribution |

Topology inside generative planners so far is used as a label, a filter or a diversity target, not as a structural
constraint: e.g. Topological Motion Planning Diffusion (Tian et al. 2026) (✓) samples unconditioned candidates and
filters them by winding number, falling back to the best candidate if none passes. GoalFlow / StepNav / MM-FM
(HANDOFF §4.5) choose a discrete mode and then generate, without a feasibility guarantee inside the mode.

## 2. What v2 takes from each, and what it adds

- From **reparameterized domains**: decoding y = m + w tanh(u) is the simplest possible mirror map (a box). Nothing new
  about the map itself.
- From **corridor planning**: the class chart is a chain of axis-aligned free rectangles, i.e. a safe corridor whose
  region sequence is fixed by the route code. Nothing new about corridors either.
- **The topology-specific step is the link between the two.** The feasible set of collision-free trajectories is
  non-convex and disconnected, so no single mirror map or reflection applies to it (the domain methods assume convex
  or simply-shaped sets). Fixing the homotopy class (here: the route code within the x-monotone family) picks one
  component, and on the waypoint grid that component contains a **product of intervals** (Prop. 1) that covers almost
  all of it (band width 99% of the exact free width, Prop. 2 for class completeness under refinement). So:
  *topology is what turns an intractable constrained-generation domain into one where the trivial mirror map is exact.*
  Without the class, there is no chart: the union of corridors of different classes is not a product of intervals.
- What a corridor planner does not give and v2 does: a learned **distribution** over paths inside each class
  (within-class diversity close to the data, D2: 0.062 vs 0.064), and a learned distribution over classes (selector),
  at the generation cost of the unconstrained model (same params, same NFE).
- What projection does not give and v2 does: exactness (0 vs ~0.2% margin violations), no per-sample solver, no
  class switches, and higher diversity at low NFE (D2 NFE 5: 0.052 vs 0.031).

## 3. Claims we can and cannot make

Can (with current evidence):
1. A topology-conditioned flow (P1) learns class membership (T = 0) but not in-class feasibility (U + M ≈ 4% at NFE 20 on
   D2/D3); class conditioning alone is not a feasibility guarantee.
2. Generating in the class chart satisfies both, for every sample and every NFE, inside the stated assumptions
   (CHART_PROOF.md A1–A6), with matched cost, reproducible over 3 seeds.
3. The chart doubles as a feasibility certificate for classes (found 58% of D1 not_found codes feasible).

Cannot (yet):
- Generality beyond x-monotone routes, disks and axis-aligned boxes, 2D, static obstacles.
- That the guarantee survives a continuous controller tracking the polyline.
- Novelty of the chart or of corridor-in-class construction on its own; the contribution is the pairing of a learned
  class distribution with an exact class chart, plus the controlled analysis (entropy × NFE, oracle decomposition,
  clearance taxonomy, flow-time analysis).
- Comparison against DiRecT-style terminal-constraint sampling or reflected/mirror FM with a hand-built global map;
  these are the closest baselines and are not run.

## 4. To check before writing related work
Exact citations for: Lou & Ermon 2023 (ICML); Liu, Chen, Wu, Lipman? → verify author list of Mirror Diffusion (NeurIPS
2023, arXiv 2310.01236); Fishman et al. 2023 (arXiv 2304.05364?); Liu et al. 2017 safe flight corridors (RA-L);
Deits & Tedrake 2015 IRIS; Marcucci et al. 2023 GCS motion planning (Science Robotics); Xiao et al. 2023 SafeDiffuser;
Carvalho et al. 2023 motion planning diffusion; homotopy-class planning (Bhattacharya et al. 2012, h-signature).

Sources checked: arXiv 2606.15359 (DiRecT), 2507.04384 (Mao et al.), 2603.10330 (PC-Diffuser), 2603.26696 (TMPD),
2310.01236 (Mirror Diffusion), 2101.11565 (GCS).

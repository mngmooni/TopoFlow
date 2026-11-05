# Where TDFM v2 sits: what is topology-specific 


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

## 4. Anticipated objections (novelty defence?)

| objection | answer we can give now | what would still weaken us |
|---|---|---|
| "It is a safe corridor plus a tanh." | Yes for each part (§2). The claim is the pairing: a learned class distribution × an exact per-class chart, which no corridor planner (one optimal path) or domain-reparameterized model (convex domain) provides. | A paper that already learns a distribution inside topology-indexed corridors. Not found in the searches so far; not a systematic survey. |
| "Projection after sampling does the same." | Matched evaluation: projection gets U+M 0.21% (not 0), needs a per-sample solver, lowers low-NFE diversity (0.031 vs 0.052 at NFE 5). | Projection onto the *chart* (not the free space) would also be exact; v2's advantage there is cost and diversity only. Not run. |
| "x-monotone, disks/boxes, 2D." | Stated as scope (CHART_PROOF A1–A6, §8). Coverage under harder layouts: 92–100% at n=60, 100% after bisection. | No evidence beyond the family. |
| "The class target is just the planner's output." | Day 12: the cost definition moved p* by TV 0.24 (medium); revised target = chart support + chart-constrained shortest cost J_c, which does not depend on the planner's sampling. J_c is an upper bound on the class optimum, not the optimum. | J_c is chart-dependent; a looser chart would change it. |
| "Topology check is circular." | Independent winding check agrees on ~239k paths; valid only in the x-monotone family (winding does not separate homotopy classes in general). | — |

## 5. Citations 
- Lou, Ermon. Reflected Diffusion Models. ICML 2023 (PMLR 202). ✓
- Liu, Chen, Theodorou, Tao. Mirror Diffusion Models for Constrained and Watermarked Generation. NeurIPS 2023;
  arXiv 2310.01236. ✓
- Fishman, Klarner, De Bortoli, Mathieu, Hutchinson. Diffusion Models for Constrained Domains. TMLR 2023;
  arXiv 2304.05364. ✓ (log-barrier and reflected variants; the "FM on constrained manifolds" wording above is ours.)
- Xiao et al. SafeDiffuser: Safe Planning with Diffusion Probabilistic Models. arXiv 2306.00148 (2023); ICLR 2025. ✓
- Bhattacharya, Likhachev, Kumar. Topological constraints in search-based robot path planning. Autonomous Robots
  2012 (h-signature). ✓ (CMU RI page); also Bhattacharya, Kumar, Likhachev AAAI 2010 for homotopy class constraints.
- Deits, Tedrake. Computing large convex regions of obstacle-free space through semidefinite programming (IRIS).
  WAFR 2014, in Algorithmic Foundations of Robotics XI (Springer 2015). ✓ (MIT copy)
- Marcucci et al. Motion planning around obstacles with convex optimization (GCS). Science Robotics 2023;
  arXiv 2205.04422 / 2101.11565 (GCS theory). ✓ for 2101.11565; check the motion-planning arXiv number.
- Carvalho et al. Motion Planning Diffusion. arXiv 2308.01557 (IROS 2023 — venue from memory); extended version
  arXiv 2412.19948. ✓ arXiv
- Liu et al. Planning dynamically feasible trajectories for quadrotors using safe flight corridors. RA-L 2017.
  **Not verified** (search did not return the paper page).
- DiRecT 2606.15359, Mao et al. 2507.04384, PC-Diffuser 2603.10330, TMPD 2603.26696: ✓ (Day 12).

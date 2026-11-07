# TopoFlow 

# 1. Introduction

Geometric path planning seeks collision-free paths connecting a start and a goal in environments containing obstacles. In such environments, multiple routes may exist that navigate around obstacles in qualitatively different ways. For fixed endpoints, these routes can belong to distinct *homotopy classes*, whose members cannot be continuously deformed into one another while remaining in free space. A generative planner should therefore be able to represent multiple route alternatives and, when desired, generate paths associated with a specified route class. However, specifying an intended route does not by itself ensure that a generated path satisfies the underlying geometric constraints.

Generative models offer a way to learn distributions over path solutions rather than producing only a single deterministic route. Diffusion-based planning methods have demonstrated the potential of learned trajectory priors and cost-guided sampling, while Flow Matching provides a framework for learning continuous vector fields that transport a simple prior distribution toward a target data distribution (Carvalho et al., 2023; Lipman et al., 2023). Applying such models to obstacle-constrained path generation raises two interconnected questions: how can a generator distinguish between alternative topological routes, and how can its outputs be made geometrically feasible? Conditioning a generative model on a route descriptor may address the first question, but does not automatically provide a guarantee for the second. This motivates investigating representations in which route-specific geometric constraints are incorporated into the generative process itself.

Existing approaches address topological diversity and geometric feasibility through different mechanisms. Topology-guided planners explicitly explore alternative route classes, while generative planning methods can incorporate topological information into trajectory generation. Separately, constrained generative approaches use mechanisms such as guidance, barrier functions, and optimization-based corrections to encourage or enforce geometric feasibility. These developments motivate a closer examination of how topological route information and geometric constraints should be represented within a generative model. In particular, enforcing collision avoidance is not sufficient to characterize a generative planner: the range of representable route classes, the fidelity and diversity of generated paths within each class, and the computational cost of constraint enforcement are also important. This raises a central question: **what are the benefits and limitations of learning a generative flow directly within a class-specific feasible representation, rather than generating unconstrained paths and correcting them during inference?**

To investigate this question, we propose **Topology-Aware Flow Matching (TDFM)**, a generative path planning framework that combines explicit route conditioning with class-specific feasible trajectory representations. We consider two-dimensional environments containing static obstacles and focus on a family of x-monotone paths. Each candidate route is represented by a discrete code specifying whether the path passes above or below individual obstacles. Given an obstacle configuration and a route code, we construct a feasible chart that maps unconstrained latent coordinates to waypoint positions within geometrically admissible intervals. The chart is constructed to preserve obstacle clearance along the resulting piecewise-linear path, rather than merely at individual waypoints. TDFM then learns a conditional Flow Matching model in chart coordinates, allowing generated samples to be decoded into geometrically feasible paths without iterative collision correction. Under explicit assumptions on chart construction and obstacle geometry, this representation guarantees geometric validity for decoded paths whenever a valid feasible chart exists. Correspondence between route descriptors and topological path classes requires additional assumptions, including valid crossing cuts. Thus, geometric feasibility is enforced by the representation, while Flow Matching models the distribution of trajectories within that representation.

We investigate the properties of TDFM through controlled experiments evaluating route adherence, geometric validity, class coverage, distributional fidelity, diversity, and computational efficiency. Our results demonstrate that route-conditioned Flow Matching can closely follow prescribed route descriptors while still producing trajectories that intersect obstacles. By contrast, both direct generation in feasible charts and chart-aware projection satisfy the tested geometric constraints, indicating that feasibility is attributable to the representation rather than to Flow Matching alone. Under matched evaluation conditions, TDFM preserves greater within-class trajectory diversity at lower measured inference cost than an adapted chart-aware projection baseline. The projection-based approach, however, tends to produce shorter, smoother paths with greater obstacle clearance. Comparisons with class-matched sampling-planner references further suggest that TDFM better preserves the reference distribution's trajectory variation, although its generated paths tend to pass closer to obstacles. In addition, our feasibility analysis identifies route classes for which a sampling-based planner fails to discover a solution despite the existence of chart-feasible paths. These findings characterize trade-offs among geometric feasibility, generative diversity, computational efficiency, and route-class coverage.

The main contributions of this work are:

1. **Topology-aware generation in feasible trajectory spaces.** We integrate explicit route conditioning with Flow Matching in class-specific feasible charts and establish conditional geometric validity guarantees for decoded paths, together with topological correspondence statements under additional assumptions.

2. **A controlled evaluation of feasibility and generative quality.** We distinguish route adherence from geometric validity and compare direct chart-space generation with chart-aware inference-time projection, examining their effects on within-class diversity, distributional fidelity, obstacle clearance, and computational efficiency.

3. **A chart-based route-class coverage and feasibility audit.** We use feasible-chart construction to assess the representability of candidate route classes and distinguish sampling-planner search failures from infeasibility established under the assumptions of the considered path family.

This study is restricted to static two-dimensional environments with circular and axis-aligned rectangular obstacles and an x-monotone path family. Accordingly, the reported guarantees do not extend directly to arbitrary homotopy classes, dynamic obstacles, robot dynamics, or continuous trajectory tracking.


# 2. Related Work

## 2.1. Topology-Aware Path Planning

In obstacle-constrained environments, multiple collision-free paths may connect the same start and goal configurations while belonging to distinct homotopy classes. Classical topology-aware planning approaches explicitly distinguish these alternatives using topological descriptors, such as winding-based signatures and other path invariants, and incorporate them into search or optimization procedures. Such representations enable planners to reason about route alternatives beyond geometric path length alone. However, the topological information captured by a descriptor depends on its mathematical definition and the assumptions of the planning environment.

Recent work has extended topology-aware reasoning to generative and region-based planning. Tian et al. (2026) introduce Topological Motion Planning Diffusion (TMPD) for tethered robot navigation, where a diffusion-based front-end generates diverse trajectory candidates and a topology-aware back-end evaluates their compatibility with the robot's accumulated tether configuration using generalized winding numbers. Their framework emphasizes topological exploration and candidate filtering under history-dependent entanglement constraints. In a complementary direction, Dong et al. (2026) propose a reference-path-free corridor generation framework based on multi-scale graphs of convex sets. By extending visibility deformation and uniform visibility deformation to sequences of convex sets, their approach identifies and consolidates topologically related corridors while retaining alternative routes.

These studies establish the importance of explicitly accounting for topological structure in path planning, either through path-level descriptors or feasible-region representations. They also illustrate distinct ways of incorporating topology into the planning pipeline: candidate generation followed by topological evaluation, and topology-aware exploration of collision-free corridors. Our work investigates a different design choice within a restricted path family: using an explicit route descriptor to condition Flow Matching on a class-specific feasible trajectory representation. In this setting, we separately examine the geometric validity of decoded paths and the conditions under which route descriptors correspond to topological path classes.

## 2.2. Generative Models for Path Planning

Generative models have emerged as a promising approach to learning distributions over trajectories, enabling the synthesis of multiple candidate solutions from previously observed planning data. Unlike deterministic planning methods that typically return a single solution for a given objective, distribution-based approaches can represent multiple modes of feasible behavior. Janner et al. (2022) introduced *Diffuser*, which formulates planning as an iterative denoising process over trajectories and supports flexible test-time conditioning. Building on diffusion-based trajectory generation, Carvalho et al. (2025) developed *Motion Planning Diffusion (MPD)*, which learns multimodal trajectory priors and combines them with motion-planning cost gradients during sampling. These approaches demonstrate the potential of generative models to capture diverse planning solutions while incorporating task-specific objectives.

Flow Matching (FM) provides an alternative framework for learning continuous generative dynamics. Lipman et al. (2023) formulate Flow Matching as a simulation-free training objective for continuous normalizing flows, in which a parameterized vector field is learned by regressing against conditional probability paths connecting a source distribution to a target distribution. Once trained, samples are generated by integrating the learned vector field. This formulation supports different choices of probability paths, including displacement interpolation, and provides a flexible foundation for conditional generative modeling. However, the standard Flow Matching objective does not inherently enforce geometric constraints on generated trajectories, even when the target distribution consists of feasible examples.

For obstacle-constrained path generation, modeling a multimodal trajectory distribution involves more than recovering visually diverse paths. Generated samples may differ in their route classes or exhibit geometric variation within the same route class, and these two forms of diversity should be distinguished. In addition, successful distribution learning does not necessarily imply that every generated path satisfies the desired obstacle-clearance constraints. These considerations motivate the investigation of generative representations that explicitly incorporate route information and geometric feasibility. TDFM studies this formulation by learning conditional flows in class-specific feasible trajectory coordinates, with separate evaluations of route adherence, geometric validity, and within-class distributional fidelity.

## 2.3. Feasibility-Constrained Generative Modeling

Ensuring geometric feasibility remains a central challenge in generative planning. Although diffusion and flow-based models can learn distributions from feasible demonstrations, their generated outputs are not necessarily guaranteed to satisfy the underlying constraints. Existing methods address this problem through several complementary strategies, including inference-time correction, constraint-preserving generative dynamics, and transformations between constrained and simpler generative spaces. These approaches differ in where constraints are imposed, the assumptions required for feasibility guarantees, and their effects on distributional fidelity and computational efficiency.

**Inference-time constraint enforcement** modifies the generative sampling procedure to encourage or enforce feasibility. Projection-based approaches correct generated states or estimated clean samples through constrained optimization. DiRecT (Giaretta et al., 2026), for example, formulates constrained diffusion planning from a stochastic optimal control perspective and introduces a receding-horizon denoising procedure that applies optimization to predicted clean trajectories. Rather than requiring noisy intermediate samples themselves to remain feasible, DiRecT progressively adjusts the denoising dynamics toward a feasible terminal output. Its formulation demonstrates how constraint enforcement can be incorporated into sampling without retraining the underlying generative model. However, the computational and distributional effects of correction depend on the optimization procedure, constraint geometry, and sampling configuration.

**Constraint-preserving generative dynamics** offer an alternative by embedding feasibility into the model's update mechanism. PolyFlow (Ma et al., 2026) develops a discrete-time Flow Matching formulation for polyhedral constraints and parameterizes each update using a learned direction, a ray-shooting operator, and a bounded gating factor. By constructing updates as convex combinations of feasible points, PolyFlow ensures that successive states remain inside the specified convex feasible set under its assumptions, without iterative projection. For non-convex maze environments, the method employs a decomposition into local convex regions. This line of work illustrates that constraint satisfaction can arise from the structure of the generative dynamics rather than from corrective optimization alone.

**Feasible-space transformations** instead simplify constrained generation through mappings between the original domain and a tractable representation. Gauge Flow Matching (Li et al., 2026) constructs a bijective gauge map between a compact convex feasible set and a unit ball, allowing Flow Matching to operate in the transformed space. Reflection is used to maintain feasibility during sampling in the unit ball, and the resulting samples are mapped back to the original constrained domain. The framework establishes constraint-satisfaction and distribution-approximation results under explicit assumptions and extends its construction to certain non-convex settings. More broadly, such approaches demonstrate that the choice of generative coordinates can play a fundamental role in enforcing constraints and controlling distributional distortion.

These studies establish that feasibility-preserving generation, projection-free updates, and constrained-space mappings are not new in isolation. They also motivate examining how geometric constraints interact with the representation of topologically distinct paths. In obstacle-constrained planning, the feasible path space may contain multiple route classes, while an individual feasible representation may capture only a subset of the available paths. Consequently, constraint satisfaction alone does not characterize the coverage or distributional behavior of a topology-conditioned generator. TDFM investigates these issues in a restricted family of two-dimensional paths by combining explicit route conditioning with class-specific feasible charts and evaluating geometric validity, route-class representability, within-class distributional fidelity, and inference cost.

## 2.4. Positioning of TDFM

The preceding studies demonstrate substantial progress in topology-aware planning, generative trajectory modeling, and constraint-preserving generation. Topology-aware planners distinguish alternative path classes and construct collision-free corridors, while generative approaches learn distributions over candidate trajectories and incorporate constraints through guidance, optimization, or structured sampling dynamics. Feasible-space transformations further establish that constraint satisfaction can be achieved through the choice of generative representation. These developments provide important foundations for our study, rather than representing limitations that TDFM is intended to overcome universally.

Within this broader context, **Topology-Aware Flow Matching (TDFM)** investigates the integration of explicit route conditioning with feasibility-preserving generative coordinates. For a specified obstacle configuration and route descriptor, TDFM constructs a class-specific feasible chart and learns a conditional flow over its latent coordinates. This formulation distinguishes the selection of a route descriptor from the generation of geometric path variations within its associated feasible region. Geometric validity follows from the chart construction under explicit assumptions, while correspondence to a topological path class requires additional conditions on the route representation and crossing cuts. Importantly, the construction is restricted to a particular family of two-dimensional x-monotone paths and is not assumed to represent every feasible path or homotopy class.

Our study also examines properties that are not captured by constraint satisfaction alone. In particular, we evaluate the representability of candidate route classes, distinguish intended-route adherence from geometric feasibility, and assess the distributional fidelity and diversity of generated paths within individual route classes. By comparing direct chart-space generation with an adapted chart-aware correction baseline under matched evaluation conditions, we investigate the practical consequences of incorporating feasibility into the generative representation rather than enforcing it through inference-time correction. The resulting analysis is intended to characterize the capabilities and limitations of topology-conditioned Flow Matching within the specified path family, rather than establish general superiority over existing constrained generative planners.




# 3. Problem Formulation

## 3.1. Planning Environment and Path Representation

We consider geometric path planning in a two-dimensional Euclidean space containing a finite collection of static obstacles. An environment is specified by an obstacle configuration $\mathcal{O}=\{O_1,\ldots,O_M\}$, a start position $s=(x_s,y_s)$, and a goal position $g=(x_g,y_g)$, where $x_s \lt x_g$.

Each obstacle is a closed disk or a closed axis-aligned rectangle. Obstacles are indexed in increasing order of their horizontal center coordinates. The planning problem concerns the geometric properties of paths and does not impose robot dynamics, velocity limits, or trajectory-tracking constraints.

For a clearance requirement $c>0$, let $O_j^{(c)}$ denote a clearance-expanded obstacle. Circular obstacles are expanded by increasing their radii by $c$. Rectangular obstacles are expanded by increasing each half-width and half-height by $c$. The latter expansion conservatively contains the Euclidean distance-$c$ neighborhood of the original rectangle.

The clearance-constrained free space is defined as

$$\mathcal{F}_c=\mathbb{R}^2\setminus\bigcup_{j=1}^{M}\mathrm{int}(O_j^{(c)})$$

A geometrically feasible path connects $s$ to $g$ while remaining in $\mathcal{F}_c$. Under this definition, touching the boundary of a clearance-expanded obstacle is allowed, whereas entering its interior is not.

We restrict the path representation to an $x$-monotone family of piecewise-linear curves. For a selected discretization resolution, let

$$x_s=x_0\lt x_1\lt\cdots\lt x_n\lt x_{n+1}=x_g$$

be a strictly increasing horizontal grid containing $n$ interior positions.

A candidate path is parameterized by its interior vertical coordinates,

$$\mathbf{y}=(y_1,\ldots,y_n)\in\mathbb{R}^n$$

with fixed endpoint coordinates $y_0=y_s$ and $y_{n+1}=y_g$.

The corresponding polygonal path is

$$\tau_{\mathbf{y}}=((x_0,y_0),(x_1,y_1),\ldots,(x_{n+1},y_{n+1}))$$

where consecutive vertices are connected by straight-line interpolation. Since the horizontal coordinates are strictly increasing, the resulting path is $x$-monotone.

Geometric validity is evaluated over entire line segments, not merely at waypoint locations.

For a fixed grid, let the set of geometrically feasible polygonal paths be

$$\mathcal{P}_n(\mathcal{O},c,s,g)=\{\tau_{\mathbf{y}}\mid\mathbf{y}\in\mathbb{R}^n,\;\tau_{\mathbf{y}}\subseteq\mathcal{F}_c\}$$

This set represents the collision-constrained geometric domain of the discretized planning problem. It does not yet distinguish paths according to their intended topological routes.

The feasible-chart representation introduced later will generally parameterize only a subset of this domain.

## 3.2. Topological Route Representation

### 3.2.1. Binary route descriptors

Within the considered path family, we represent an intended route by a binary descriptor

$$h=(h_1,\ldots,h_M)\in\{0,1\}^M$$

The bit $h_j=0$ specifies passage above obstacle $O_j$, whereas $h_j=1$ specifies passage below it.

The descriptor defines a proposed routing pattern rather than a complete path.

A binary code does not, by itself, establish that the corresponding route is geometrically feasible. Its realizability depends on the obstacle configuration, endpoints, clearance requirement, and restrictions of the path representation.

In particular, different codes need not all admit feasible paths, and some geometrically feasible paths may lie outside the restricted $x$-monotone family.

### 3.2.2. Reduced crossing words

To evaluate the relationship between route descriptors and homotopy classes, we use a crossing-word representation based on oriented obstacle cuts.

For each obstacle $O_j$, a downward vertical ray is drawn from its lowest point at the obstacle's horizontal center. The ray serves as the cut associated with generator $a_j$.

As a path crosses the cuts, we record the corresponding generators in traversal order. Crossing cut $j$ from left to right contributes $a_j$, while crossing from right to left contributes its inverse.

The ordered crossings form a word

$$w(\tau)=a_{i_1}^{\sigma_1}a_{i_2}^{\sigma_2}\cdots a_{i_r}^{\sigma_r}$$

where each $\sigma_k$ equals $+1$ or $-1$ according to crossing direction.

The reduced word is obtained by repeatedly canceling adjacent inverse pairs:

$$W(\tau)=\mathrm{Red}(w(\tau))$$

For example, an adjacent pair $a_ja_j^{-1}$ cancels, whereas the commutator $aba^{-1}b^{-1}$ is generally nontrivial in the free group despite having zero total exponent for each generator.

Thus, crossing words retain ordering information that obstacle-wise winding counts can lose.

### 3.2.3. Route-code correspondence

For a binary code $h$, let $j_1,\ldots,j_r$ be the strictly increasing indices of obstacles for which $h_j=1$.

The expected crossing word for the restricted left-to-right route family is

$$W_h=a_{j_1}a_{j_2}\cdots a_{j_r}$$

If every route bit is zero, the expected word is the identity element $e$.

For a generated path, the intended correspondence is evaluated by

$$W(\tau)=W_h$$

However, this equality provides a homotopy-class interpretation only when the underlying cut system is valid.

In particular, the obstacle cuts must have distinct horizontal positions, avoid other obstacles, and avoid the start and goal positions. The relevant obstacle geometry must also satisfy the endpoint and route-family assumptions used in the correspondence argument.

Under such conditions, reduced crossing words provide coordinates for endpoint-fixed homotopy classes relative to a chosen reference path.

When these conditions fail, the computational crossing word can still be evaluated, but it is not treated as an unconditional homotopy certificate.

We therefore distinguish intended route-code adherence, geometric path feasibility, and mathematically justified topological correspondence.

## 3.3. Conditional Generative Planning Objective

### 3.3.1. Desired path distribution

For an environment $\mathcal{O}$ and route descriptor $h$, consider a reference distribution over geometric paths,

$$p_{\mathrm{ref}}(\tau\mid\mathcal{O},h)$$

In this study, the reference distribution is represented by a finite collection of trajectories generated by a sampling-based planner.

The objective is to learn a conditional generative model capable of producing multiple geometric path variations associated with a prescribed route descriptor.

However, the complete reference distribution may not be representable by the feasible chart constructed for a particular environment and route.

This limitation must be distinguished from the approximation error of the learned generative model.

### 3.3.2. Chart-representable paths

Let $\Phi_{\mathcal{O},h}$ denote a decoder associated with a successfully constructed feasible chart:

$$\Phi_{\mathcal{O},h}:\mathbb{R}^n\rightarrow\mathcal{P}_n(\mathcal{O},c,s,g)$$

Its image defines the chart-representable path family,

$$\mathcal{C}_{\mathcal{O},h}=\{\Phi_{\mathcal{O},h}(\mathbf{u})\mid\mathbf{u}\in\mathbb{R}^n\}$$

For a geometrically valid chart, the intended inclusion is

$$\mathcal{C}_{\mathcal{O},h}\subseteq\mathcal{P}_n(\mathcal{O},c,s,g)$$

This inclusion does not imply that the chart covers the entire feasible path space or every path belonging to the associated homotopy class.

The existence of a binary descriptor also does not guarantee that a corresponding feasible chart can be constructed.

### 3.3.3. Encoded reference distribution

The chart-space model is trained using reference trajectories that belong to route codes with successfully constructed charts.

Each reference path is first resampled onto the chart's horizontal grid. Its vertical coordinates are then normalized relative to the chart bounds and encoded through a clipped inverse hyperbolic-tangent transformation.

Let $E_{\mathcal{O},h}$ denote this encoding procedure. For a reference path $\tau$, its encoded coordinates are

$$\mathbf{u}^{\mathrm{ref}}=E_{\mathcal{O},h}(\tau)$$

The resulting empirical latent distribution is denoted by

$$q_{\mathrm{enc}}(\mathbf{u}\mid\mathcal{O},h)$$

The encoding procedure includes clipping. Therefore, a reference path that extends outside the chart-representable waypoint intervals may still produce finite latent training coordinates.

Consequently, $q_{\mathrm{enc}}$ is not necessarily the exact inverse image of the original reference-path distribution.

This distinction is essential: the generative model learns the encoded reference distribution, whereas fidelity to the original geometric reference distribution must be evaluated separately.

### 3.3.4. Conditional Flow Matching

TDFM learns a conditional vector field in the latent chart coordinates.

The source distribution is a standard Gaussian,

$$\mathbf{u}_0\sim\mathcal{N}(\mathbf{0},I)$$

and the target coordinates follow the encoded empirical distribution,

$$\mathbf{u}_1\sim q_{\mathrm{enc}}(\mathbf{u}\mid\mathcal{O},h)$$

For $t\sim\mathrm{Uniform}[0,1]$, define the linear conditional interpolation

$$\mathbf{u}_t=(1-t)\mathbf{u}_0+t\mathbf{u}_1$$

with velocity target

$$\mathbf{v}^{*}=\mathbf{u}_1-\mathbf{u}_0$$

The neural vector field is conditioned on the planning environment, route code, and chart representation:

$$\mathbf{v}_{\theta}=\mathbf{v}_{\theta}(\mathbf{u}_t,t\mid\mathcal{O},h,\mathcal{B})$$

where $\mathcal{B}$ denotes the chart information, including waypoint bounds and horizontal grid coordinates.

The training objective is

$$\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}[\|\mathbf{v}_{\theta}(\mathbf{u}_t,t\mid\mathcal{O},h,\mathcal{B})-(\mathbf{u}_1-\mathbf{u}_0)\|_2^2]$$

The expectation is taken over the training examples, source samples, and interpolation times.

Sampling numerically integrates the learned vector field and decodes the resulting latent coordinate into a geometric path.

Under a valid chart construction, geometric feasibility follows from the decoder rather than from exact recovery of the target latent distribution.

Thus, distribution approximation and geometric validity are distinct properties of the framework.

## 3.4. Research Questions and Evaluation Criteria

The formulation motivates evaluating topology-conditioned generative planning through several separate questions.

**Geometric validity.** Does a successfully constructed feasible chart guarantee that decoded piecewise-linear paths satisfy the prescribed obstacle-clearance constraints? The relevant property concerns complete path segments, not only individual waypoints.

**Topological correspondence.** Under what assumptions does a binary route descriptor correspond to a reduced crossing word, and when does this correspondence provide a valid homotopy-class interpretation? We distinguish this question from geometric collision avoidance.

**Route-class representability.** Which candidate route descriptors admit feasible charts at a given grid resolution? How does local refinement change construction-level coverage, and how does this differ from the route classes supported by a fixed-resolution trained model?

**Distributional fidelity.** How closely does the generated path distribution reproduce the geometric statistics and within-class diversity of the reference planner? We distinguish the encoded latent target distribution from the original geometric reference distribution.

**Generation versus correction.** How does learning directly in feasible coordinates compare with applying chart-aware or free-space corrections during inference? We evaluate within-class diversity, geometric quality, constraint violations, and computational cost under matched experimental conditions.

These questions are investigated within the restricted setting of static two-dimensional environments and $x$-monotone polygonal paths.

The study does not claim completeness over unrestricted homotopy classes, physical robot tracking guarantees, or exact recovery of an unknown distribution of all feasible paths.


# 4. Methodology

Topology-Aware Flow Matching (TDFM) combines explicit route conditioning with a class-specific, feasibility-preserving representation of piecewise-linear paths. The framework separates geometric constraint satisfaction from distribution learning: a deterministic chart construction restricts decoded paths to a certified feasible region, while a conditional Flow Matching model learns variations within the corresponding chart coordinates.

Given an obstacle configuration $\mathcal{O}$, a binary route descriptor $h$, and a clearance margin $c$, TDFM constructs segment-wise feasible bands, converts them into admissible waypoint intervals, and parameterizes these intervals using unconstrained latent coordinates. The model then learns and samples a conditional flow in the resulting coordinate space.

The construction is restricted to static two-dimensional environments with circular or axis-aligned rectangular obstacles and a family of $x$-monotone polygonal paths.

## 4.1. Route-Conditioned Feasible Chart Construction

### 4.1.1. Segment-wise feasible bands

Let $h=(h_1,\ldots,h_M)$ be a prescribed binary route code, where $h_j=0$ denotes passage above obstacle $O_j$ and $h_j=1$ denotes passage below it.

We construct a strictly increasing horizontal grid with $n$ interior points, as defined in Section 3.1. The implementation augments a uniform grid with points placed slightly outside the horizontal boundaries of clearance-expanded obstacles. Specifically, the default grid construction places additional points at offsets of $10^{-6}$ beyond these boundaries whenever they lie between the start and goal.

For each horizontal segment interval, define

$$I_k=[x_k,x_{k+1}],\qquad k=0,\ldots,n$$

The chart assigns a lower bound $L_k$ and an upper bound $U_k$ to this interval. These bounds are constructed from the obstacle walls over the complete interval, rather than from obstacle heights at a single horizontal position.

For an obstacle assigned to the above-passing side, the lower bound must be at least as high as its upper clearance-expanded wall wherever the obstacle overlaps $I_k$. For an obstacle assigned to the below-passing side, the upper bound must be no higher than its lower clearance-expanded wall.

Let $t_j^{(c)}(x)$ and $b_j^{(c)}(x)$ denote the upper and lower boundaries of clearance-expanded obstacle $O_j$ over its horizontal support. The segment bounds are

$$L_k=\max\{Y_{\min},\max_{j:h_j=0}\sup_{x\in I_k\cap X_j^{(c)}}t_j^{(c)}(x)\}$$

$$U_k=\min\{Y_{\max},\min_{j:h_j=1}\inf_{x\in I_k\cap X_j^{(c)}}b_j^{(c)}(x)\}$$

where $X_j^{(c)}$ denotes the horizontal support of the expanded obstacle, and obstacles with no overlap are omitted from the corresponding extrema. In the implementation, $Y_{\min}=-1.2$ and $Y_{\max}=1.2$.

For circular obstacles, the required extremum is evaluated at the horizontal position within the interval closest to the circle center. For axis-aligned rectangular obstacles, the upper and lower walls are constant over the obstacle's horizontal support.

The resulting segment band is

$$R_k=I_k\times[L_k,U_k]$$

Whenever the chart is feasible, these bounds define a rectangular region that avoids the interiors of all relevant clearance-expanded obstacles.

### 4.1.2. Admissible waypoint intervals

Each interior waypoint is shared by two consecutive line segments. Therefore, its vertical coordinate must satisfy the constraints of both adjacent segment bands.

For interior waypoint $i$, define

$$\ell_i=\max(L_{i-1},L_i),\qquad r_i=\min(U_{i-1},U_i)$$

The admissible interval is

$$I_i^{y}=[\ell_i,r_i]$$

A chart is accepted only if all interior waypoint intervals are nondegenerate:

$$\ell_i<r_i,\qquad i=1,\ldots,n$$

In addition, the fixed start and goal positions must satisfy

$$L_0\le y_s\le U_0,\qquad L_n\le y_g\le U_n$$

These conditions are checked explicitly during chart construction.

A successful construction provides a feasible family of polygonal paths. Conversely, failure of the interval test indicates that the current chart construction is unsuccessful; it does not, by itself, prove that the corresponding route is infeasible in the unrestricted free space.

### 4.1.3. Residual coordinate parameterization

For every nondegenerate waypoint interval, define its midpoint

$$m_i=(\ell_i+r_i)/2$$

and a slightly contracted half-width

$$w_i=(1-\varepsilon)(r_i-\ell_i)/2,\qquad\varepsilon=10^{-3}$$

We parameterize the vertical coordinate using an unconstrained latent variable $u_i$:

$$y_i=m_i+w_i\tanh(u_i)$$

For any finite $u_i$, this transformation satisfies

$$\ell_i<y_i<r_i$$

The complete waypoint decoder is

$$\mathbf{y}=\mathbf{m}+\mathbf{w}\odot\tanh(\mathbf{u})$$

where $\odot$ denotes element-wise multiplication.

The horizontal coordinates and fixed endpoints are then appended to obtain the complete piecewise-linear path. We denote this decoding procedure by

$$\tau=\Phi_{\mathcal{O},h}(\mathbf{u})$$

For fixed chart bounds, the coordinate transformation is invertible onto the contracted open product of intervals defined by the decoder. It is not a bijection onto the complete set of geometrically feasible paths.

The contraction factor $\varepsilon$ introduces an additional numerical margin between the decoded waypoint coordinates and the chart boundaries.

### 4.1.4. Encoding reference trajectories

Training trajectories are first resampled onto the fixed horizontal grid by interpolation.

Let $\mathbf{y}^{\mathrm{ref}}$ denote the resampled interior waypoint coordinates of a reference path. The implementation encodes these coordinates by first normalizing them relative to the chart midpoint and contracted half-width:

$$z_i=(y_i^{\mathrm{ref}}-m_i)/w_i$$

The normalized coordinates are clipped before applying the inverse hyperbolic tangent:

$$\widetilde{z}_i=\mathrm{clip}(z_i,-1+10^{-6},1-10^{-6})$$

$$u_i^{\mathrm{ref}}=\operatorname{arctanh}(\widetilde{z}_i)$$

This clipping operation ensures finite training coordinates, including for reference paths whose resampled waypoints fall outside the representable interval.

Importantly, the encoding procedure does not simply discard every reference path that lies outside the chart. Instead, reference paths belonging to a route with a successfully constructed chart are encoded using the clipping rule above. Reference paths associated with unsuccessful chart constructions are excluded from chart-space training.

Therefore, the empirical training distribution is the distribution of encoded, potentially clipped reference trajectories. It is not necessarily the exact inverse image of the original reference-path distribution.

This distinction matters when interpreting distributional fidelity: geometric feasibility of generated paths is guaranteed by the decoder under its assumptions, while faithful recovery of the original reference distribution is a separate empirical question.

## 4.2. Conditional Flow Matching in Chart Coordinates

### 4.2.1. Conditional vector field

For an environment $\mathcal{O}$ and route descriptor $h$, let $q_{\mathrm{enc}}(\mathbf{u}\mid\mathcal{O},h)$ denote the empirical distribution of encoded reference coordinates constructed in Section 4.1.4.

The model learns a conditional flow from a standard Gaussian source distribution to this encoded target distribution.

We sample

$$\mathbf{u}_0\sim\mathcal{N}(\mathbf{0},I)$$

and

$$\mathbf{u}_1\sim q_{\mathrm{enc}}(\mathbf{u}\mid\mathcal{O},h)$$

For an interpolation time $t$ sampled uniformly from $[0,1]$, the conditional interpolation is

$$\mathbf{u}_t=(1-t)\mathbf{u}_0+t\mathbf{u}_1$$

with velocity target

$$\mathbf{v}^{*}=\mathbf{u}_1-\mathbf{u}_0$$

A neural vector field predicts the corresponding velocity conditioned on time, environment geometry, and route information:

$$\mathbf{v}_{\theta}=\mathbf{v}_{\theta}(\mathbf{u}_t,t\mid\mathcal{O},h,\ell,r,x)$$

Here $\ell$, $r$, and $x$ denote the chart's waypoint bounds and horizontal grid coordinates.

The model uses an obstacle-attention architecture. Obstacle features and route bits are represented as conditioning tokens, while waypoint tokens contain chart-coordinate values together with their lower bounds, upper bounds, and horizontal positions. The fixed-grid implementation uses 60 interior waypoint coordinates grouped into 10 waypoint tokens, with six coordinates per token.

The architecture is designed to condition the learned velocity field on both obstacle geometry and the feasible representation associated with the requested route.

### 4.2.2. Training objective

The conditional Flow Matching objective is

$$\mathcal{L}_{\mathrm{CFM}}(\theta)=\mathbb{E}[\|\mathbf{v}_{\theta}(\mathbf{u}_t,t\mid\mathcal{O},h,\ell,r,x)-(\mathbf{u}_1-\mathbf{u}_0)\|_2^2]$$

The expectation is taken over training examples, Gaussian source samples, and interpolation times.

The implementation uses mean squared error regression to train the vector field. No collision penalty or inference-time geometric projection is required by this chart-space training objective.

The learned field determines how probability mass is transported in latent coordinates. The geometric validity of the final decoded path is instead determined by the feasibility-preserving chart construction.

### 4.2.3. Sampling and decoding

At inference time, a valid chart is constructed for the requested environment and route code. An initial latent coordinate is drawn from the Gaussian source distribution.

The learned vector field defines

$$\frac{d\mathbf{u}_t}{dt}=\mathbf{v}_{\theta}(\mathbf{u}_t,t\mid\mathcal{O},h,\ell,r,x)$$

For a sampling budget of $K$ numerical function evaluations, the fixed-grid implementation uses forward Euler integration:

$$\mathbf{u}_{k+1}=\mathbf{u}_k+\frac{1}{K}\mathbf{v}_{\theta}(\mathbf{u}_k,k/K\mid\mathcal{O},h,\ell,r,x)$$

After the final integration step, the resulting latent vector is decoded through the feasible chart:

$$\tau_{\mathrm{gen}}=\Phi_{\mathcal{O},h}(\mathbf{u}_K)$$

Provided that the chart is valid and the latent output is finite, the decoder produces a geometrically feasible piecewise-linear path even when the learned flow is imperfect.

This property concerns the feasibility of the decoded output, not the accuracy of the learned distribution or the validity of a physical robot's continuous execution.

## 4.3. Conditional Validity Guarantees

### 4.3.1. Geometric validity

**Proposition 1 (Geometric soundness of the feasible chart).**

Assume that:

1. The horizontal waypoint grid is strictly increasing.
2. Every segment band is constructed using conservative bounds on all relevant clearance-expanded obstacle walls.
3. All interior waypoint intervals are nondegenerate, and both fixed endpoints satisfy the corresponding terminal band constraints.
4. The decoder uses the interval parameterization defined in Section 4.1.3 with finite latent coordinates.

Then every decoded polygonal path maintains the prescribed obstacle clearance.

**Proof.**

By the definition of the decoder, each interior vertical coordinate lies strictly inside its admissible waypoint interval.

Each admissible waypoint interval is the intersection of the two neighboring segment bands. Therefore, both endpoints of segment $k$ lie within its associated band $R_k$.

Because $R_k$ is a convex rectangle, the straight-line segment connecting its endpoints lies entirely inside $R_k$.

The conservative band construction ensures that $R_k$ does not intersect the interior of any clearance-expanded obstacle. Hence the entire decoded segment satisfies the prescribed obstacle clearance.

Applying this argument to every segment establishes the geometric validity of the complete polygonal path.

For circular obstacles, the expansion by radius $c$ corresponds to Euclidean clearance $c$. For axis-aligned rectangular obstacles, the square-cornered expansion used by the implementation contains the Euclidean $c$-neighborhood of the obstacle. Avoiding its interior therefore also ensures Euclidean distance at least $c$ from the original rectangular obstacle.

Thus, every decoded segment maintains the required Euclidean clearance, completing the proof.

### 4.3.2. Homotopy consistency within a chart

The chart also has a structural topological property.

Consider two paths decoded from the same valid chart, with interior waypoint coordinates $\mathbf{y}^{A}$ and $\mathbf{y}^{B}$. For $\lambda\in[0,1]$, define

$$\mathbf{y}^{\lambda}=(1-\lambda)\mathbf{y}^{A}+\lambda\mathbf{y}^{B}$$

Since every admissible waypoint interval is convex, the interpolated coordinates remain within those intervals.

By Proposition 1, each interpolated polygonal path remains geometrically feasible.

The interpolation therefore defines a continuous endpoint-fixed deformation between the two paths through geometrically feasible paths.

Consequently, all paths decoded from the same valid chart belong to a common endpoint-fixed homotopy class in the obstacle-free space.

This result establishes homotopy consistency within the chart. It does not, by itself, identify that common homotopy class with an externally specified route descriptor.

### 4.3.3. Correspondence to the prescribed crossing word

To connect the binary route descriptor to a reduced crossing word, additional assumptions are required.

In particular, the clearance-expanded horizontal support of every obstacle must lie strictly between the start and goal horizontal coordinates. The downward cut associated with each obstacle must also be valid: the cuts must have distinct horizontal positions, avoid other obstacles, and avoid the start and goal positions.

Under these assumptions, the chart construction forces a decoded path to remain above obstacle $j$ when $h_j=0$ and below it when $h_j=1$.

Because the decoded path is $x$-monotone, a below-passing route crosses the corresponding downward cut exactly once, from left to right. An above-passing route does not cross that cut.

The prescribed crossing word is therefore the sequence of positive obstacle generators corresponding to below-passing route bits, ordered by obstacle horizontal position.

Using one-based obstacle indices, this reference word is

$$W_h=(j\mid h_j=1,\;j=1,\ldots,M)$$

where the notation represents an ordered word rather than an unordered set.

Under the valid-cut assumptions, the decoded path satisfies

$$W(\tau_{\mathrm{gen}})=W_h$$

The geometric guarantee in Proposition 1 does not require these additional topological assumptions. Therefore, geometric validity remains meaningful in environments where the prescribed cut system is unsuitable for homotopy certification.

### 4.3.4. Scope of the guarantees

The guarantees above are conditional.

Geometric soundness applies to the decoded polygonal paths for which a valid feasible chart exists. It does not establish the safety of a robot tracking these paths under dynamics, disturbances, or uncertainty.

Homotopy consistency applies within the chart-representable family. Correspondence between a binary route code and a reduced crossing word additionally requires a valid cut system and the stated geometric assumptions.

Neither result implies that the chart represents every collision-free path within a homotopy class.

Likewise, these guarantees do not establish exact recovery of the original reference-path distribution. Representation limits, clipped training coordinates, vector-field approximation error, and numerical integration may all influence the generated distribution.

## 4.4. Route-Class Representability and Coverage

### 4.4.1. Successful chart construction

For a given environment and clearance requirement, define the chart-representable route-code set as

$$\mathcal{H}_{\mathrm{chart}}(\mathcal{O})=\{h\in\{0,1\}^{M}\mid\text{the chart construction succeeds for }h\}$$

Membership in this set provides a constructive feasibility certificate: decoding the zero latent vector yields a path satisfying the chart's geometric constraints.

However, a route descriptor absent from this set is not necessarily infeasible.

The construction may fail because of insufficient grid resolution, conservative band bounds, or the imposed representational restrictions.

### 4.4.2. Resolution-dependent coverage

The fixed-grid chart-space Flow Matching model uses 60 interior waypoint coordinates. The representability of route codes under this grid is therefore distinct from their representability under arbitrarily refined grids.

The chart construction can also be evaluated on locally refined horizontal grids. When a waypoint interval fails the feasibility test, additional grid points may resolve a narrow passage that was not captured by the original discretization.

Such refinement preserves the geometric soundness argument, provided that the refined grid and bands satisfy the same assumptions.

Nevertheless, a fixed-dimensional model trained on 60 interior waypoint coordinates cannot automatically sample on a larger refined grid. Construction-level coverage and trained-model generation coverage must therefore be reported separately.

### 4.4.3. Feasibility audit

We evaluate representability by enumerating binary route descriptors and distinguishing successful chart construction from unsuccessful construction.

For unsuccessful cases, an independent geometric audit analyzes the pointwise available vertical interval of the prescribed route and searches for evidence of infeasibility within the considered $x$-monotone family.

When a valid geometric infeasibility witness exists, the corresponding route can be classified as infeasible under the stated path-family and clearance assumptions.

When the pointwise route corridor remains feasible but the default chart fails, local grid refinement can be used to investigate resolution-dependent exclusion.

The audit is explicitly restricted to the considered path family. It does not certify infeasibility among arbitrary non-monotone paths.

### 4.4.4. Relationship to the reference planner

We distinguish chart construction success from successful reference trajectory generation.

The reference planner uses a finite sampling budget. Consequently, failure to find a route does not establish that the route is geometrically infeasible.

A successfully constructed chart can provide a constructive counterexample to such a negative planner result.

This distinction motivates reporting route-class representability independently from planner discovery rates and from the distributional performance of the trained generative model.

Together, the chart construction and coverage audit separate three questions: whether a route admits a feasible representation, whether the trained model can generate from that representation, and whether the generated samples reproduce the desired within-class path distribution.


# 5. Experimental Setup

We evaluate Topology-Aware Flow Matching (TDFM) through three complementary experiments. Experiment A examines the geometric soundness and route-class coverage of the feasible chart construction. Experiment B compares direct chart-space Flow Matching against constraint-aware inference procedures under matched evaluation conditions. Experiment C evaluates how closely generated paths reproduce the geometric statistics and within-class diversity of the reference trajectory distribution.

All final-validation experiments use static two-dimensional environments, circular and axis-aligned rectangular obstacles, and a clearance requirement of $c=0.01$. We distinguish theoretical properties of the chart construction from empirical observations of trained generative models.

## 5.1. Planning Environments and Reference Data

### 5.1.1. Environment families

We consider three principal synthetic environment datasets, denoted D2, D3, and D4, together with four additional stress-test families, H1–H4.

**D2: Fixed-endpoint environments.** D2 contains randomly generated layouts with three to five obstacles. The start and goal positions are fixed at $(-0.9,0)$ and $(0.9,0)$, respectively. The layouts contain circular and axis-aligned rectangular obstacles, with an approximate box sampling probability of 0.3. The standard obstacle generator uses radii or size parameters sampled from $[0.08,0.15]$, with horizontal placement structured to separate obstacle groups.

**D3: Variable-endpoint environments.** D3 follows the same general obstacle-generation procedure as D2, but introduces variation in the vertical coordinates of the start and goal positions. Specifically, the endpoint heights are sampled independently from $[-0.4,0.4]$, while their horizontal coordinates remain fixed at $-0.9$ and $0.9$. This setting tests whether the generative model can incorporate changes in endpoint geometry.

**D4: Tightly packed environments.** D4 contains layouts with four or five obstacles arranged with small horizontal gaps. Obstacle sizes are larger than in D2 and D3, and the generator explicitly constructs closely spaced configurations. This setting is used primarily to examine chart representability and the limitations of finite-budget reference-path discovery.

Each principal dataset contains 2,000 training layouts, 300 previously unseen evaluation layouts, and 300 calibration layouts generated with separate random seeds. The synthetic reference-path generator attempts to produce four trajectories per route code, subject to geometric feasibility and a finite search budget. Consequently, some route codes have no reference trajectories.

The reference trajectories are generated by a sampling-based route construction procedure. They provide a finite empirical target distribution, not an exhaustive or ground-truth characterization of all collision-free paths.

### 5.1.2. Stress-test environments

To evaluate geometric robustness beyond the principal layouts, we consider four additional environment families.

- **H1 (stacked obstacles):** Three to five obstacles with spatial arrangements that can create overlapping horizontal supports and vertically stacked configurations.
- **H2 (tight gaps):** Four or five closely spaced obstacles designed to stress narrow-passage representability.
- **H3 (dense layouts):** Six to eight obstacles, increasing the number of possible binary route descriptors and the complexity of the chart construction.
- **H4 (stacked rectangular obstacles):** Three to five axis-aligned rectangular obstacles, emphasizing the effects of rectangular geometry and overlapping horizontal supports.

Each stress-test family contains 300 unseen layouts. These families are used for chart verification and coverage analysis, rather than for the primary in-distribution generative comparison.

Some H1 and H4 environments violate the assumptions required for the downward-cut topology representation. Such environments remain relevant to geometric feasibility tests but are not treated as satisfying the full topological correspondence theorem.

### 5.1.3. Path representations

The raw route-conditioned baseline, P1, represents trajectories using 30 interior two-dimensional waypoints, producing a 60-dimensional free-coordinate state.

The chart-space model, denoted v2, represents paths using 60 interior vertical coordinates on a fixed, strictly increasing horizontal grid. Its decoder reconstructs the complete polygonal path from the chart coordinates and fixed endpoints.

Both models condition on obstacle geometry and an explicit binary route descriptor. The v2 model additionally receives the chart bounds and horizontal grid coordinates. For D3, both architectures incorporate the varying endpoint heights.

The chart-space training set is constructed by resampling reference trajectories onto the chart grid and encoding them using the clipped inverse transformation described in Section 4.1.4.

The fixed-grid generative model is trained only on successfully constructed 60-point charts. Locally refined charts used during the coverage audit are not automatically supported by the fixed-dimensional trained model.

## 5.2. Generative Models and Comparison Methods

### 5.2.1. Chart-space Flow Matching

The principal method is TDFM v2, which trains a conditional vector field directly in the latent coordinates of the class-specific feasible chart.

The implementation uses an obstacle-attention architecture with a hidden dimension of 128 and four attention blocks. For the 60-dimensional latent state, coordinates are grouped into 10 waypoint tokens containing six coordinates each.

Each waypoint token includes its latent coordinates, chart bounds, and horizontal positions. Obstacle tokens encode geometric features and the associated route bits.

Training uses conditional Flow Matching with independent Gaussian source samples and linear interpolation between source and encoded target coordinates.

The principal training configuration consists of 30,000 optimization steps with a batch size of 256. The implementation uses the Adam optimizer, an initial learning rate of $10^{-3}$, and cosine learning-rate scheduling.

At inference time, generated latent samples are integrated using forward Euler steps and decoded through the corresponding feasible chart.

### 5.2.2. Route-conditioned unconstrained Flow Matching

We use P1 as a route-conditioned Flow Matching baseline operating directly in free waypoint coordinates.

P1 uses an obstacle-attention architecture with the same hidden dimension and number of attention blocks as the chart-space model, but it does not restrict its generated waypoint coordinates through a feasible chart during ordinary sampling.

This baseline is important for distinguishing explicit route conditioning from geometric feasibility. A model may follow its prescribed route descriptor while still producing paths that violate obstacle clearance.

### 5.2.3. DiRecT-inspired chart correction

Our principal correction-based comparison uses the trained P1 model with a chart-aware adaptation inspired by DiRecT.

At selected inference steps, the procedure predicts a clean trajectory, projects its resampled vertical coordinates onto the same feasible chart used by v2, and incorporates the resulting correction into the ongoing rectified-flow sampling process.

The implementation applies correction during the later portion of sampling, beginning at normalized flow time $t=0.5$. At the final step, it returns the chart-corrected polygonal path.

This baseline is referred to as **P1 + DiRecT-chart**.

The method is an in-house adaptation of the predicted-clean-trajectory correction principle to rectified Flow Matching. It is not a reproduction of the original diffusion-based DiRecT algorithm.

Importantly, both v2 and P1 + DiRecT-chart use the same feasibility-preserving chart. Therefore, any geometric validity shared by the two methods must be attributed to the chart construction rather than uniquely to chart-space Flow Matching.

### 5.2.4. Additional correction baselines

We also consider secondary comparison methods:

- **P1 raw:** Direct route-conditioned Flow Matching without geometric correction.
- **P1 + final free-space projection:** A correction applied to the predicted final trajectory using a numerical constraint solver.
- **P1 + DiRecT-style free-space correction:** Repeated clean-trajectory correction using a penalty-based free-space solver.
- **B0 + free-space correction:** An environment-conditioned, route-code-blind Flow Matching baseline followed by free-space correction.

The free-space correction solver uses a batched penalty-continuation procedure implemented in PyTorch. The correction constraints are evaluated on densified trajectory samples and are not mathematically identical to the class-specific chart constraints.

These secondary baselines are used to assess the effects of correction strategy and computational budget. They should not be interpreted as exact reproductions of externally published algorithms.

## 5.3. Evaluation Metrics

We evaluate geometric validity, topological correspondence, within-class diversity, geometric quality, distributional fidelity, and inference cost.

### 5.3.1. Geometric validity

For a generated polygonal path $\tau$, define its minimum Euclidean obstacle clearance as

$$d_{\min}(\tau)=\min_j\min_{p\in\tau}\mathrm{dist}(p,O_j)$$

A path satisfies the required clearance if

$$d_{\min}(\tau)\ge c$$

The minimum is evaluated analytically over complete path segments, rather than only at waypoint locations.

For rectangular obstacles, we additionally test intersection with the open axis-aligned rectangle obtained by expanding each half-width and half-height by $c$.

An independent geometric checker implements these tests without relying on the chart's collision-checking routines.

### 5.3.2. Route adherence and sample outcomes

Where the crossing-cut assumptions are valid, we compare the reduced crossing word of each generated path with the word associated with its requested route descriptor.

We also retain the outcome taxonomy used in the evaluation code:

- **U (Unsafe):** The path collides with an obstacle at zero clearance.
- **M (Margin violation):** The path avoids physical collision but violates the required clearance $c$.
- **T (Topology mismatch):** The path satisfies clearance but does not match the requested crossing-word representation.
- **N (Reference-label mismatch):** The path satisfies clearance and the requested word, but its code is absent from the reference planner's found set.
- **V (Valid):** The path satisfies clearance, matches the requested route, and belongs to a reference-planner-found class.

The category N is not, by itself, evidence of geometric or topological failure. It reflects the finite discovery capability of the reference planner.

Where a cut system is invalid, crossing-word agreement is reported as a computational observation rather than an independent homotopy certificate.

### 5.3.3. Within-class diversity

To measure geometric variation within a route class, generated trajectories are interpolated onto a shared 60-point horizontal grid.

For two paths $\tau_a$ and $\tau_b$, with interpolated vertical coordinates $y_a(x_i)$ and $y_b(x_i)$, define

$$D(\tau_a,\tau_b)=\sqrt{\frac{1}{60}\sum_{i=1}^{60}(y_a(x_i)-y_b(x_i))^2}$$

Within-class diversity is measured as the mean pairwise RMS distance between distinct valid samples belonging to the same layout and requested route code.

This metric measures variation within a specified class, not the number of distinct route classes represented by the generator.

### 5.3.4. Path length and smoothness

The geometric length of a polygonal path is

$$L(\tau)=\sum_{i=0}^{n}\|p_{i+1}-p_i\|_2$$

To measure smoothness, trajectories are resampled onto a common horizontal grid. Let $y_i$ denote the resampled vertical coordinate. We define the roughness statistic as

$$R(\tau)=\frac{1}{N-2}\sum_{i=2}^{N-1}(y_{i+1}-2y_i+y_{i-1})^2$$

where $N=60$ for the shared evaluation grid.

Lower roughness indicates smaller discrete second differences in the vertical path coordinates. This is a geometric measure and does not directly measure acceleration under a physical time parameterization.

The length normalization differs between Experiments B and C. Experiment B uses the shortest available planner reference path for a given class. Experiment C uses the chart-constrained shortest length, denoted $J_c$, to define the normalized statistic

$$L_{\mathrm{ratio}}(\tau)=L(\tau)/J_c$$

These two normalization conventions are kept distinct when interpreting the results.

### 5.3.5. Distributional fidelity and obstacle proximity

Distributional fidelity is evaluated through class-matched comparisons between generated trajectories and available planner references.

For each route class, we compare the median normalized path length, mean roughness, mean minimum clearance, and within-class diversity.

We also define an obstacle-proximity indicator:

$$H(\tau)=\mathbb{1}[d_{\min}(\tau)<c+0.005]$$

The corresponding obstacle-hugging rate is the fraction of paths satisfying this indicator.

For roughness, minimum clearance, and diversity, the primary summaries report generated-to-reference ratios. For normalized length and obstacle-hugging rate, we report generated-minus-reference differences.

These comparisons characterize fidelity to the empirical reference planner. They do not establish fidelity to an unknown distribution of all feasible paths or demonstrate that closer agreement necessarily implies higher planning quality.

### 5.3.6. Computational cost

We report the number of neural function evaluations (NFE) and measured inference time per generated sample.

For chart-based methods, the measured inference time includes chart construction. For projection-based methods, it includes the corresponding correction procedure.

We also compare methods at approximately matched wall-clock budgets, selecting the largest tested v2 sampling budget that does not exceed the measured time of a comparator.

The computational comparison is specific to the evaluated implementation and hardware. Absolute times are not treated as hardware-independent complexity guarantees.

## 5.4. Experimental Protocols

### 5.4.1. Experiment A: Chart soundness and coverage

Experiment A evaluates the chart independently of the learned generative model.

For D2, D3, D4, and H1–H4, we enumerate every binary route descriptor for every unseen evaluation layout. Each family contains 300 layouts.

We first construct the default chart using 60 interior waypoints. When construction fails, we attempt local horizontal-grid bisection with a maximum of 400 grid points.

For every successfully constructed chart, we decode a fixed battery of latent samples. The battery includes Gaussian latent vectors with different scales, strongly saturated latent vectors with values of $+50$ or $-50$, and mixed-coordinate saturation patterns.

For D2–D4, 24 latent samples are decoded per successful chart. For H1–H4, eight samples are decoded per successful chart.

All decoded paths are checked using independent analytic segment-to-obstacle distance tests, square-inflated rectangle intersection tests, and an independently implemented reduced crossing-word checker.

Segments with floating-point clearance margins below $10^{-9}$ are designated for exact rational rechecking.

The experiment also verifies the geometric assumptions required by the soundness and topology propositions, including strict horizontal grid ordering, obstacle placement relative to endpoints, and cut validity.

For coverage analysis, route descriptors are classified according to successful default-grid construction, recovery through local refinement, or analytically supported infeasibility within the restricted path family.

The coverage audit distinguishes chart-construction failure from geometric infeasibility and from reference-planner failure.

### 5.4.2. Experiment B: Chart-space generation versus inference-time correction

Experiment B compares TDFM v2 with P1 + DiRecT-chart on the unseen D2 and D3 layouts.

For each dataset, we use 300 evaluation layouts and sample 32 route requests per layout from the same frozen cost-based route selector. The selector request seed is 900.

Both models receive identical layout-code requests and matched Gaussian initial noise.

The primary sampling budgets are

$$K\in\{5,20\}$$

Three sampling-noise seeds, 777, 778, and 779, are evaluated.

For D2, three paired model-seed configurations are used:

$$s_{\mathrm{model}}\in\{0,1,2\}$$

For D3, the available comparison uses model seed 0, together with all three sampling-noise seeds.

All generated samples are retained in the safety evaluation, including unsuccessful or reference-unmatched outcomes.

For each configuration, we compare geometric validity, crossing-word agreement, within-class diversity, path-length ratio, roughness, and minimum clearance.

Uncertainty intervals are estimated using 4,000 layout-clustered bootstrap replicates. The pooled analysis also incorporates variation across the sampling-noise seeds, while D2 model-seed comparisons are reported separately.

Inference-time measurements are conducted in a standardized single-CPU-thread environment using batches of 320 samples. Each configuration is executed in a separate subprocess, with one warm-up run and typically five timed repetitions. The median measured time is used for the primary comparison, with three repetitions for configurations exceeding the prescribed long-run threshold.

Timing evaluations include additional NFE budgets and secondary projection baselines.

### 5.4.3. Experiment C: Class-matched distributional fidelity

Experiment C evaluates the relationship between the generated path distribution and the empirical reference planner distribution.

We use all route descriptors with successful 60-point chart construction on the unseen D2 and D3 layouts.

A route class is included in the primary matched analysis only when planner-generated reference trajectories are available for that same environment, endpoint configuration, and route descriptor.

Classes without reference trajectories are disclosed separately rather than treated as failed generated samples.

For each chart-feasible class, we generate eight paths at NFE 20. Sampling-noise seeds 11, 12, and 13 are used.

The principal v2 evaluation includes model seed 0 on both D2 and D3, with two additional v2 model seeds on D2. P1 + DiRecT-chart is evaluated as a secondary comparator using model seed 0.

Reference trajectories are processed with the same path-quality measurement functions as the generated trajectories.

For each matched class, we compute path-length, roughness, clearance, obstacle-hugging, and diversity statistics. The primary results use equal-class aggregation so that classes with more reference samples do not automatically dominate the summary.

We estimate 95% uncertainty intervals with 4,000 layout-clustered bootstrap replicates, additionally resampling the available generated runs.

As secondary diagnostics, we compute one-dimensional Wasserstein-1 distances for selected path statistics. These estimates are interpreted cautiously because each route class typically contains only approximately four reference trajectories.

## 5.5. Reproducibility and Evaluation Scope

The final experiments use frozen model checkpoints and predefined evaluation procedures. Predictions for Experiments A, B, and C were recorded before execution, and the corresponding result summaries were preserved separately from the evaluation scripts.

The implementation includes independent geometric verification code, explicit random seeds, fixed dataset splits, and scripts for regenerating the principal numerical summaries and figures.

The soundness experiment tests the deterministic chart construction without requiring neural-network accuracy. The generative comparison uses matched route requests and random initial conditions. The fidelity experiment uses exact layout-and-class matching to reduce differences caused by unequal class composition.

Several limitations remain. First, the environments are synthetic and restricted to static two-dimensional geometric planning. Second, the reference trajectory generator provides only a finite, sampling-dependent approximation to the desired trajectory distribution. Third, D3 includes only one model seed in the primary generative comparison. Fourth, the reported computational costs are implementation- and hardware-dependent.

Finally, the mathematical guarantees concern the geometry and topology of decoded polygonal paths under explicit assumptions. They do not establish robot tracking safety, complete representation of unrestricted homotopy classes, or exact recovery of the target trajectory distribution.


# 6. Results

We present the results of three final-validation experiments examining geometric soundness, route-class coverage, generation–correction trade-offs, and conditional distributional fidelity. We distinguish properties established by the chart construction from empirical observations of trained generative models.

## 6.1. Chart Soundness and Route-Class Coverage

### 6.1.1. Independent geometric verification

Experiment A independently verified the feasibility-preserving chart construction across D2, D3, D4, and four additional stress-test environment families.

In total, the audit enumerated 82,224 route-code instances across 2,100 layouts. For every successfully constructed chart, a fixed battery of latent inputs was decoded and evaluated by an independent geometric checker.

The verification covered **695,488 decoded paths and approximately 42.6 million line segments**. No Euclidean clearance violations or intersections with the open clearance-expanded rectangular obstacles were observed.

The smallest measured Euclidean clearance margin above the required clearance $c=0.01$ was approximately $1.8\times10^{-7}$.

The independent crossing-word checker also recorded zero mismatches between decoded paths and their prescribed route words. However, this numerical agreement is not an unconditional homotopy certificate: the required cut-system assumptions were violated in 184 H1 layouts and 171 H4 layouts.

These results support the implementation-level correctness of the geometric chart under the tested conditions. The mathematical guarantee itself follows from Proposition 1 and its assumptions, rather than from the finite number of successful verification cases.

### 6.1.2. Resolution-dependent class coverage

Figure 1 summarizes the relationship between chart construction, local refinement, and route-class infeasibility.

<img width="1424" height="601" alt="chart_coverage" src="https://github.com/user-attachments/assets/65287149-d631-4a4f-9c43-86d21d817bc9" />



**Figure 1.** Route-code coverage of the deterministic feasible-chart construction across seven environment families. Blue indicates route codes admitting a valid chart at the default resolution of 60 interior waypoints, green indicates additional codes recovered through local grid refinement, and gray indicates codes certified infeasible within the specified clearance-constrained, x-monotone path family. Percentages are relative to all enumerated route codes. The figure measures chart-construction coverage, not the tested generative model's ability to sample every displayed environment.

Importantly, default-grid chart representability should not be equated with trained-model coverage. The evaluated v2 architecture accepts at most five obstacle tokens, whereas the H3 stress-test family contains six to eight obstacles. Moreover, Experiments B and C evaluate generative performance on D2 and D3 rather than the H1–H4 stress-test families. The stress-test results therefore establish properties of the deterministic chart construction, not empirical generalization of the trained Flow Matching model to all these environments.


### 6.1.3. Reference-planner failures versus geometric infeasibility

Experiment A also evaluated the relationship between successful reference-path discovery and chart representability.

In D4, the sampling-based reference planner failed to find paths for 5,030 route-code instances. However, 2,801 of these instances admitted successfully constructed feasible charts, while the remaining 2,229 were certified infeasible within the restricted path family.

Thus, a substantial proportion of the planner's negative outcomes were computational discovery failures rather than geometric infeasibility.

This result demonstrates the importance of separating reference-planner availability, chart representability, and geometric infeasibility. A route code without a planner-generated demonstration should not automatically be treated as an impossible route.

## 6.2. Direct Chart Generation versus Inference-Time Correction

### 6.2.1. Geometric validity

Experiment B compares direct chart-space Flow Matching (v2) with P1 + DiRecT-chart, an inference-time correction baseline using the same feasible chart.

Across both models, all tested configurations, and three sampling-noise seeds, the independent checker evaluated **460,800 generated paths**.

Neither model produced any measured Euclidean clearance violations, square-inflated rectangle violations, or crossing-word mismatches in the evaluated valid-cut environments.

This result is consistent with both models using the same feasibility-preserving decoder. It should not be interpreted as evidence that chart-space Flow Matching is uniquely capable of enforcing geometric constraints.

### 6.2.2. Within-class diversity and geometric quality

Despite sharing the same feasible representation, direct chart-space generation and inference-time correction produced different within-class distributions.

At NFE 20, the model-seed-0 within-class diversity on D2 was 0.061 for v2 and 0.053 for P1 + DiRecT-chart. The paired difference was +0.0082, with a 95% layout-clustered bootstrap interval of [0.0072, 0.0092].

On D3, the corresponding diversity difference was +0.0090 [0.0081, 0.0099].

At NFE 5, the diversity differences increased to approximately +0.021 on both datasets. The positive diversity difference was reproduced across all three paired model seeds available on D2.

These results indicate that, within the tested settings, direct learning in chart coordinates preserved more geometric variation within each requested route class than the adapted correction procedure.

However, the chart-space model did not dominate the correction baseline on every geometric quality measure.

At NFE 20, v2 generated paths with slightly greater normalized length and roughness, together with lower minimum obstacle clearance. The correction baseline tended to produce shorter, smoother trajectories with larger clearance.

The observed difference therefore represents a diversity–geometric-quality trade-off rather than unconditional superiority.

### 6.2.3. Computational efficiency

Figure 2 compares within-class diversity against standardized inference time.

<img width="1980" height="903" alt="pareto_time_diversity" src="https://github.com/user-attachments/assets/09ae2c83-0978-46c2-ac0e-7382d2a4f8d3" />


**Figure 2.** Within-class diversity versus standardized inference time on D2 and D3. Filled markers indicate configurations with zero measured clearance/class violations, while hollow markers indicate at least one violation. Numeric labels denote neural function evaluations. Timings were measured on one CPU thread with batch size 320. Primary NFE 5 and 20 comparisons include repeated sampling-noise seeds; secondary operating points and free-space correction baselines contain additional single-seed evaluations.

At NFE 20, v2 required 5.60 ms per sample on D2 and 4.75 ms on D3. The corresponding P1 + DiRecT-chart inference times were 7.46 ms and 7.28 ms.

Thus, under the standardized CPU measurement configuration, the correction baseline required approximately 1.33 times the inference time of v2 on D2 and 1.53 times on D3.

Free-space optimization-based correction procedures were substantially more expensive. At NFE 20, the tested DiRecT-style free-space correction configuration required approximately 115–138 times the inference time of v2.

At comparable measured wall-clock budgets, v2 maintained higher within-class diversity than the evaluated correction baselines while preserving the chart's geometric constraints.

These findings are specific to the implemented models, correction procedures, and measurement environment. In particular, the evaluated DiRecT-chart baseline is an adaptation rather than a direct reproduction of the published DiRecT sampler.

## 6.3. Conditional Distributional Fidelity

### 6.3.1. Reference-class matching

Experiment C evaluates the distribution of generated paths against the finite trajectory distributions produced by the reference planner.

The analysis includes 6,240 matched route classes on D2 and 6,147 on D3. Five additional D3 classes were chart-feasible but lacked planner-generated references and were excluded from the matched fidelity analysis.

Each matched class was evaluated using eight generated paths at NFE 20, with repeated sampling-noise seeds.

Importantly, the reference planner defines the empirical comparison distribution. Agreement with its outputs should not be interpreted as recovery of a uniquely optimal or exhaustive distribution of feasible paths.

### 6.3.2. Qualitative path comparisons

Figure 3 compares reference trajectories with samples from v2 and P1 + DiRecT-chart for identical obstacle layouts and requested route classes.

<img width="1731" height="1421" alt="matched_trajectories" src="https://github.com/user-attachments/assets/48b85495-1d6d-42cc-b874-a9d0ebc1f3d8" />


**Figure 3.** Class-matched trajectory examples from the reference planner, TDFM v2, and P1 + DiRecT-chart. The examples were selected by a fixed rule before inspecting generated outcomes: the first three unseen D2 layouts were used, with the route class containing the largest number of planner references selected in each layout and ties resolved by code order. Each row uses identical environment geometry and route conditioning. The generated columns display eight samples per method at NFE 20. These examples illustrate geometric variation within a route class; the aggregate fidelity results are reported separately.


Both generative methods reproduced the general routing structure of the reference paths. Their samples differed, however, in the extent and distribution of geometric variation within the same class.

The qualitative examples complement the aggregate distributional evaluation; they are not used as evidence of a general quality ranking.

### 6.3.3. Quantitative fidelity

Table 1 summarizes the primary class-matched fidelity measures.

**Table 1.** Distributional fidelity relative to the reference planner. Ratios are generated/reference values; normalized length and obstacle-hugging entries are generated-minus-reference differences. Values are based on equal-class aggregation, with 95% layout-clustered bootstrap intervals.

| Statistic | v2 D2 | v2 D3 | DiRecT-chart D2 | DiRecT-chart D3 |
|---|---:|---:|---:|---:|
| Normalized length gap | -0.003 | -0.001 | -0.021 | -0.019 |
| Roughness ratio | 0.83 | 0.83 | 0.71 | 0.73 |
| Min-clearance ratio | 0.93 | 0.94 | 1.01 | 1.01 |
| Obstacle-hugging gap | +0.100 | +0.088 | +0.064 | +0.065 |
| Within-class diversity ratio | 0.95 | 0.95 | 0.83 | 0.82 |

For normalized length, v2 differed from the reference by -0.003 on D2 and -0.001 on D3, substantially smaller in magnitude than the corresponding differences for P1 + DiRecT-chart.

The v2 model also retained approximately 95% of the reference distribution's within-class diversity on both datasets. The correction baseline retained approximately 82–83%.

Both methods generated smoother paths than the reference planner according to the discrete roughness metric. The correction baseline produced the lowest roughness, although greater smoothness is not necessarily equivalent to greater fidelity to the reference distribution.

These results indicate that the chart-space model more closely preserved the reference planner's length and diversity statistics in the evaluated class-matched settings.

### 6.3.4. Obstacle proximity as a limitation

The fidelity evaluation also identified a consistent limitation of v2.

Although its generated paths satisfied the required clearance, its mean minimum obstacle clearance was lower than that of the reference planner. The generated/reference minimum-clearance ratio was 0.93 on D2 and 0.94 on D3.

In addition, v2 generated an increased proportion of paths within 0.005 of the clearance boundary. The obstacle-hugging rate was 10.0 percentage points higher than the reference on D2 and 8.8 percentage points higher on D3.

The correction baseline more closely matched the reference's mean minimum clearance, although it also exhibited an increased obstacle-hugging rate.

Thus, feasibility preservation alone does not imply faithful recovery of all geometric statistics of the reference distribution.

The observed obstacle-proximity bias remains unresolved in the present study. Its potential relationship to coordinate saturation, chart geometry, reference encoding, and the learned latent distribution requires further investigation.

Detailed class-level fidelity distributions are provided in Supplementary Figure S1.

## 6.4. Summary of Empirical Findings

The experiments support three principal observations.

First, a successfully constructed class-specific chart provides a geometrically valid representation of the considered polygonal path family. Independent evaluation found no clearance violations across approximately 695,000 decoded paths, while the mathematical guarantee remains conditional on the stated chart assumptions.

Second, direct chart-space Flow Matching and inference-time chart correction can both produce geometrically valid paths, but they exhibit different distributional and computational characteristics. In the evaluated settings, direct chart-space generation preserved greater within-class diversity at lower measured inference cost, whereas correction produced shorter and smoother paths with larger clearance.

Third, chart-space generation approximately preserved the reference planner's within-class diversity and normalized path length, but exhibited a measurable tendency toward smaller obstacle clearance and more frequent proximity to the feasible boundary.

Together, these findings characterize the capabilities and limitations of learning a conditional generative flow within a route-specific feasible representation. They do not establish general superiority over constrained generative planners or extend the conditional geometric and topological guarantees beyond the considered path family.


# 7. Discussion

The experiments provide a more precise understanding of the relationship between topology conditioning, geometric feasibility, and distribution learning in generative path planning. The central observation is that these properties should not be treated as interchangeable. Explicit route conditioning specifies the intended geometric alternative, a feasible chart restricts the set of possible decoded outputs, and the learned flow determines the distribution of samples within that representation.

We discuss the implications of this separation, the observed trade-offs between direct chart-space generation and inference-time correction, the limits of chart representability, and the remaining challenges for topology-conditioned generative planning.

## 7.1. Feasibility Comes from the Representation, Not the Learned Flow

A central finding is that route conditioning alone does not guarantee geometric feasibility.

The unconstrained route-conditioned baseline P1 demonstrates this distinction: even when generated paths follow their requested route descriptors, they may violate obstacle clearance. In contrast, both TDFM v2 and P1 + DiRecT-chart produced no measured geometric violations in the matched evaluation.

The two chart-based approaches share the same feasibility-preserving decoder. Therefore, the improvement in geometric validity cannot be attributed uniquely to the learned chart-space vector field.

Rather, geometric feasibility is supplied by a deterministic property of the representation.

This distinction matters when interpreting the methodological contribution. Constraint-preserving maps, safe corridors, and projection-based correction are established ideas in constrained generation and motion planning. TDFM does not introduce these concepts individually. Instead, it combines an explicitly indexed route representation with a feasible coordinate domain used directly for conditional distribution learning.

The class-specific chart restricts the support of decoded trajectories to a subset of geometrically admissible paths. Within this representation, the learned flow does not need to reproduce the obstacle constraints perfectly in order for the decoder to preserve them.

This separation suggests a general design principle: when a feasible generative representation can be constructed, constraint satisfaction and distribution approximation can be addressed by different components.

However, the principle is conditional on the existence and correctness of the representation. It does not imply that an equally tractable chart can be obtained for arbitrary robot configurations, dynamic constraints, or unrestricted path spaces.

## 7.2. What Does Direct Chart-Space Learning Provide Beyond Projection?

Because P1 + DiRecT-chart uses the same feasible chart as v2, the comparison in Experiment B isolates an important practical distinction between two uses of a constrained representation.

The first approach learns a generative flow directly in chart coordinates. The second learns in free path coordinates and introduces chart-aware correction during inference.

Both approaches achieved the same measured geometric validity, but their generated distributions differed.

At NFE 20, v2 exhibited greater within-class diversity, with absolute advantages of 0.0082 on D2 and 0.0090 on D3. At NFE 5, the diversity advantage increased to approximately 0.021. The positive differences persisted across the evaluated sampling-noise seeds and the three paired model seeds available on D2.

The chart-space model also required less inference time under the standardized CPU configuration. At NFE 20, P1 + DiRecT-chart required approximately 1.33 times the inference time of v2 on D2 and 1.53 times on D3.

However, the correction baseline produced shorter and smoother paths with greater obstacle clearance. Consequently, the experiments do not support a claim that direct chart-space generation is universally superior.

Instead, the comparison reveals a trade-off between within-class diversity, geometric quality, and computational cost.

One possible explanation is that repeated correction alters how probability mass is distributed within the feasible chart, potentially concentrating samples in a smaller geometric region. Direct chart-space training, by contrast, attempts to learn the distribution of encoded reference trajectories without introducing the same correction operation during sampling.

This explanation is plausible but has not been causally established. The present comparison does not isolate the contribution of individual correction steps, vector-field architectures, training distributions, or coordinate transformations.

Moreover, a diversity reduction is not an inevitable property of projection-based methods. Its magnitude depends on the correction procedure, the underlying generator, and the target distribution.

The supported conclusion is narrower: for the implemented models and matched evaluation conditions, directly learning in the feasible chart preserved more within-class variation at lower measured inference cost than the adapted chart-correction baseline.

## 7.3. Feasible Representation and Route-Class Coverage Are Different Questions

The chart construction establishes the feasibility of every decoded path when its assumptions hold, but this guarantee says nothing by itself about how many possible routes the representation can express.

The experiments demonstrate why these two questions must be evaluated separately.

On D2 and D3, the default 60-point chart represented every enumerated route code. In more challenging layouts, some geometrically feasible routes required local refinement.

For example, on D4, the default chart represented 4,775 route-code instances. An additional 420 were recovered through grid refinement, while 2,229 were certified infeasible within the prescribed clearance-constrained, x-monotone path family.

The default 60-point chart therefore represented 91.9% of the feasible D4 route-code instances, even though the geometric audit accounted for all enumerated cases through successful construction or verified infeasibility.

This result highlights a distinction between representational coverage and model coverage. Local refinement can establish the existence of a feasible chart without enabling a fixed-dimensional generative model trained on the original grid to sample directly from that refined representation.

Furthermore, a class missing from the reference planner's dataset need not be infeasible.

In D4, 2,801 of the 5,030 route-code instances for which the reference planner found no trajectory were shown to admit feasible charts. These cases demonstrate that finite-budget reference-path discovery can provide incomplete information about route feasibility.

The coverage audit therefore serves a purpose beyond validating the decoder. It provides a way to distinguish limitations of the reference planner, limitations of the fixed-grid representation, and geometric infeasibility within the specified path family.

Nevertheless, the audit does not establish completeness for unrestricted planar path planning. Its infeasibility conclusions depend on the prescribed route interpretation, clearance requirement, and x-monotone family.

Nor do the tested results prove that the local refinement procedure will resolve every feasible route in arbitrary future environments.

## 7.4. Distributional Fidelity and the Obstacle-Proximity Gap

Experiment C indicates that direct chart-space Flow Matching can reproduce important statistics of the reference planner's within-class trajectory distribution.

The generated paths retained approximately 95% of the reference within-class diversity on both D2 and D3. Their normalized path-length statistics were also close to the reference distribution.

These observations suggest that using a restricted feasible representation need not eliminate meaningful geometric variation within a route class.

However, the agreement was not uniform across all path statistics.

The v2 model generated paths with mean minimum clearances approximately 6–7% below the reference values. Its obstacle-hugging rate was 10.0 percentage points higher on D2 and 8.8 percentage points higher on D3.

Thus, a model can generate paths satisfying a hard clearance constraint while still producing a distribution systematically closer to the constraint boundary than its reference distribution.

This is an important limitation of the current implementation.

The chart parameterization offers several possible explanations worth examining.

First, the decoder uses a bounded nonlinear transformation:

$$y_i=m_i+w_i\tanh(u_i)$$

Its derivative is

$$\frac{dy_i}{du_i}=w_i(1-\tanh^2(u_i))$$

As the magnitude of the latent coordinate increases, the derivative approaches zero. Consequently, large differences in latent coordinates can correspond to relatively small differences in decoded coordinates near interval boundaries.

Second, reference trajectories are encoded using a clipped inverse transformation. Reference waypoints outside the contracted chart interval are mapped to its numerical boundary before applying the inverse hyperbolic tangent.

This operation changes the empirical target distribution relative to the original reference trajectories and may influence how probability mass is represented near chart boundaries.

Third, the chart itself imposes a geometric restriction. Even when a route is feasible, its representable path family may not include all geometric variations present in the corresponding reference distribution.

Finally, approximation error in the learned vector field and numerical integration may contribute to deviations between the encoded target distribution and generated samples.

These mechanisms are mathematically or algorithmically plausible, but the present experiments do not establish which mechanism causes the observed obstacle-proximity gap.

In particular, the obstacle-hugging result should not be attributed to hyperbolic-tangent saturation without a controlled comparison against alternative coordinate parameterizations and encoding procedures.

A useful follow-up would examine latent-coordinate distributions, encoding-clipping frequency, chart-boundary proximity, and decoder sensitivity. Such diagnostics could help determine whether the gap originates in representation geometry, training-data encoding, or generative approximation.

## 7.5. Relationship to Existing Constrained Generative Planning

The results should be interpreted within the broader literature on topology-aware planning and constrained generative modeling.

Existing constrained generative methods already demonstrate how feasible outputs can be obtained through projection, constrained dynamics, reflection, or mappings between constrained and simpler spaces. Likewise, classical corridor-based planning provides established techniques for representing collision-free regions.

TDFM builds on these ideas rather than replacing them.

Its distinguishing experimental setting combines an explicit route descriptor, a class-specific feasible path representation, and conditional Flow Matching within that representation.

The experiments investigate how this choice affects geometric validity, route-class representability, within-class diversity, reference fidelity, and computation.

This perspective differs from approaches focused primarily on finding a single low-cost path, discovering multiple corridors, or enforcing constraints on a generated trajectory after sampling.

Nevertheless, the present work does not directly benchmark against PolyFlow, Gauge Flow Matching, Topological Motion Planning Diffusion, or the homotopy-aware corridor method discussed in Section 2. Their constraint domains, planning objectives, and evaluation settings differ from those considered here.

The reported performance comparisons are therefore limited to the implemented baselines, especially the matched P1 + DiRecT-chart adaptation.

The results should not be interpreted as evidence of general performance superiority over those external methods.

More broadly, the novelty of combining route-class conditioning with a feasible generative representation remains dependent on the scope of related prior work. The present study contributes a concrete formulation and controlled empirical analysis within the specified setting, without claiming that feasible charts, conditional generation, or topology-aware planning are individually new.

## 7.6. Limitations

Several limitations constrain the interpretation and generalizability of the results.

**Restricted path family.** The method considers two-dimensional, x-monotone, piecewise-linear paths around static circular and axis-aligned rectangular obstacles. It does not represent arbitrary non-monotone routes, unrestricted homotopy classes, or higher-dimensional configuration spaces. The binary above/below descriptor is specific to the considered route family.

**Conditional topological correspondence.** A feasible chart guarantees geometric validity under its assumptions, but identifying its outputs with a prescribed reduced crossing word requires additional cut-system conditions. Some stacked-obstacle layouts violate those conditions. Although the numerical word checker recorded agreement on the tested paths, such agreement is not an unconditional homotopy certificate.

**Conservative geometric representation.** The chart uses rectangular segment bands and conservative obstacle expansion. These choices simplify the geometric guarantee but restrict the set of representable paths. A valid chart is not a complete parameterization of the corresponding feasible homotopy class.

**Fixed generative resolution.** The trained v2 model uses 60 interior waypoint coordinates. Although the geometric construction can be refined locally, the trained model cannot directly generate paths on an arbitrarily refined grid without an additional representation or model adaptation.

**Synthetic and finite reference distributions.** All evaluation environments are synthetic. Reference trajectories are generated by a sampling-based planner and typically provide only approximately four paths per class. The fidelity results characterize agreement with this finite empirical distribution rather than with an exhaustive distribution of feasible trajectories.

**Limited model-seed coverage.** Experiment B evaluates three paired model seeds on D2 but only one model seed on D3. Repeated sampling-noise seeds and layout-clustered bootstrap intervals reduce some sources of uncertainty but do not replace broader independent training replication.

**Baseline and computational scope.** The principal correction baseline is an adaptation inspired by DiRecT, not a reproduction of its original diffusion sampler. Timing results were measured on a standardized single-CPU-thread configuration and should not be generalized to different hardware, batching strategies, or independently optimized implementations without further evaluation.

**No physical execution guarantee.** The geometric proof concerns decoded polygonal paths. It does not account for robot dynamics, control error, moving obstacles, localization uncertainty, or collision avoidance during continuous tracking.

**Unresolved distributional effects.** The current study does not establish the causes of the observed obstacle-proximity bias or the reduced diversity relative to the reference distribution. No representation-level ablation isolates the effects of latent saturation, coordinate clipping, and chart conservatism.

These limitations are not merely implementation details. They determine the scope within which the theoretical claims and empirical comparisons can be interpreted.

## 7.7. Future Directions

The results suggest several directions for extending the present investigation.

One direction is to examine alternative feasible-coordinate parameterizations and encoding procedures. Such comparisons could test whether changes in boundary sensitivity or reference-data clipping improve minimum clearance and within-class distributional fidelity without sacrificing geometric validity.

A second direction is to develop variable-resolution generative representations. The current coverage audit shows that local grid refinement can recover feasible route classes excluded by the fixed grid. A generative model capable of operating on variable-length or adaptively refined path representations could potentially reduce this gap between construction-level and model-level coverage.

A third direction is to extend the treatment of topology beyond binary x-monotone route descriptors. More general planning environments may require multiple charts, explicit transitions between local representations, or richer topological encodings.

Finally, the relationship between learned route selection and conditional path generation deserves separate investigation. A generative model that faithfully samples within each represented class still depends on the mechanism used to select classes. Errors in class selection, incomplete class coverage, and within-class distribution mismatch contribute different forms of failure and should be evaluated independently.

These extensions would require new representations, theoretical assumptions, and experimental validation. They are not established by the present results.

---

# 8. Conclusion

This work investigated Topology-Aware Flow Matching (TDFM) for route-conditioned generative path planning in static two-dimensional obstacle environments.

The proposed framework combines explicit binary route descriptors with class-specific feasible trajectory charts and learns conditional flows in the corresponding latent coordinates. By constructing conservative segment-wise bands and admissible waypoint intervals, the decoder restricts generated polygonal paths to a geometrically feasible subset of the considered path family.

We established conditional geometric validity for successfully constructed charts and identified additional assumptions required for correspondence between route descriptors and reduced crossing words. Independent verification found no geometric violations across approximately 695,000 decoded paths and 42.6 million segments.

A separate route-class coverage audit distinguished successful chart construction, resolution-dependent exclusions, certified infeasibility within the restricted path family, and failures of finite-budget reference-path discovery.

Controlled comparisons demonstrated that both direct chart-space generation and chart-aware inference-time correction can achieve the same measured geometric validity. Within the evaluated configurations, direct chart-space Flow Matching preserved greater within-class diversity at lower inference cost, whereas correction produced shorter, smoother paths with greater obstacle clearance.

The generated distribution also approximated several reference-planner statistics, retaining approximately 95% of the reference within-class diversity. Nevertheless, the model exhibited a measurable tendency to generate paths closer to obstacle boundaries, highlighting a distributional limitation that geometric feasibility guarantees alone do not resolve.

Overall, the study illustrates the value of distinguishing route specification, feasible representation, and conditional distribution learning. Its findings support the use of class-specific feasible coordinates as one approach to constrained path generation while emphasizing the limits imposed by representation coverage, topological assumptions, and distributional approximation.

The conclusions remain specific to the evaluated two-dimensional x-monotone path family. Extending the framework to more general topology, adaptable path representations, and dynamically feasible robot motion remains an open direction.



### Encoding and Clipping Diagnostics

The clipping statistics recorded during training require a distinction between nominal chart-bound violations and effective clipping during reference encoding. The bookkeeping counter in `training_set()` identifies reference paths containing waypoints outside the nominal admissible intervals. In contrast, `chart.encode()` normalizes waypoint coordinates using an EPS-contracted interval width and subsequently clips the normalized values to $[-1+10^{-6},1-10^{-6}]$ before applying the inverse hyperbolic tangent.

Consequently, effective encoding may clip waypoints that lie inside the nominal admissible intervals but sufficiently close to their boundaries.

In a diagnostic using 300 training layouts at clearance $c=0.01$, the nominal and effective affected-path rates were 7.2% and 7.9% on D2, respectively, and 56.8% and 57.4% on D4. At the waypoint level, the corresponding rates increased from 0.18% to 0.20% on D2 and from 1.64% to 1.68% on D4.

The original counter therefore underestimates effective clipping. This discrepancy affects the reported diagnostic statistics but does not change the training targets, which were generated using the actual encoding procedure.

Clipped normalized coordinates are mapped to latent values with absolute magnitude approximately 7.254. Such values motivate further investigation of encoding saturation and boundary-sensitive distribution learning. However, the present diagnostic does not establish a causal relationship between clipping and the observed obstacle-proximity bias.


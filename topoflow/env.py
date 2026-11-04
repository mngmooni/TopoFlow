"""2D static-obstacle environment with a cut system for crossing-word topology.

Conventions
-----------
- Workspace is nominally [-1, 1]^2, but paths are allowed anywhere in R^2.
- Obstacles are disjoint circles or axis-aligned boxes.
- Each obstacle j gets one cut: the vertical ray x = cx_j, y < y_bottom_j,
  going straight down from the obstacle's lowest point at x = cx_j.
- With all cuts pointing down, two cuts never intersect as long as their
  x-coordinates differ. A cut is invalid if it hits another obstacle or if
  the start/goal lies on it. `Env.validate()` checks all of this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float
    kind: str = field(default="circle", init=False)

    @property
    def xrange(self):
        return self.cx - self.r, self.cx + self.r

    @property
    def yrange(self):
        return self.cy - self.r, self.cy + self.r

    @property
    def half_height(self):
        return self.r

    def cut_top(self):
        """y-coordinate where the downward cut leaves the obstacle."""
        return self.cy - self.r

    def contains(self, p, margin=0.0):
        p = np.asarray(p, dtype=float)
        return np.linalg.norm(p - [self.cx, self.cy], axis=-1) < self.r + margin

    def segments_hit(self, P, Q, margin=0.0):
        """Vectorized: does segment P[k]->Q[k] come within r+margin of the center?"""
        c = np.array([self.cx, self.cy])
        d = Q - P
        dd = np.einsum("ij,ij->i", d, d)
        s = np.where(dd > 0, np.einsum("ij,ij->i", c - P, d) / np.maximum(dd, 1e-300), 0.0)
        s = np.clip(s, 0.0, 1.0)
        closest = P + s[:, None] * d
        return np.linalg.norm(closest - c, axis=1) < self.r + margin

    def as_vector(self):
        # [kind_id, cx, cy, half_w, half_h]; circle uses r for both
        return np.array([0.0, self.cx, self.cy, self.r, self.r])


@dataclass(frozen=True)
class Box:
    cx: float
    cy: float
    hw: float  # half width
    hh: float  # half height
    kind: str = field(default="box", init=False)

    @property
    def xrange(self):
        return self.cx - self.hw, self.cx + self.hw

    @property
    def yrange(self):
        return self.cy - self.hh, self.cy + self.hh

    @property
    def half_height(self):
        return self.hh

    def cut_top(self):
        return self.cy - self.hh

    def contains(self, p, margin=0.0):
        p = np.asarray(p, dtype=float)
        return (np.abs(p[..., 0] - self.cx) < self.hw + margin) & (
            np.abs(p[..., 1] - self.cy) < self.hh + margin
        )

    def segments_hit(self, P, Q, margin=0.0):
        """Vectorized slab test against the box inflated by margin."""
        lo = np.array([self.cx - self.hw - margin, self.cy - self.hh - margin])
        hi = np.array([self.cx + self.hw + margin, self.cy + self.hh + margin])
        d = Q - P
        t0 = np.zeros(len(P))
        t1 = np.ones(len(P))
        for k in range(2):
            dk = d[:, k]
            par = np.abs(dk) < 1e-15
            # parallel segments: outside the slab -> no hit
            outside = par & ((P[:, k] < lo[k]) | (P[:, k] > hi[k]))
            t1 = np.where(outside, -1.0, t1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (lo[k] - P[:, k]) / dk
                tb = (hi[k] - P[:, k]) / dk
            tmin = np.where(par, -np.inf, np.minimum(ta, tb))
            tmax = np.where(par, np.inf, np.maximum(ta, tb))
            t0 = np.maximum(t0, tmin)
            t1 = np.minimum(t1, tmax)
        return t0 < t1

    def as_vector(self):
        return np.array([1.0, self.cx, self.cy, self.hw, self.hh])


@dataclass
class Env:
    obstacles: list
    start: tuple = (-0.9, 0.0)
    goal: tuple = (0.9, 0.0)

    def __post_init__(self):
        # Fixed ordering by cut x-coordinate. Letters a, b, c, ... follow this order.
        self.obstacles = sorted(self.obstacles, key=lambda o: o.cx)
        self.start = np.asarray(self.start, dtype=float)
        self.goal = np.asarray(self.goal, dtype=float)

    @property
    def m(self):
        return len(self.obstacles)

    # ---------------------------------------------------------------- checks
    def validate(self, eps=1e-6):
        """Raise ValueError if the cut system is not usable for crossing words."""
        obs = self.obstacles
        cxs = [o.cx for o in obs]
        if len(set(np.round(cxs, 12))) != len(cxs):
            raise ValueError("two cuts share the same x; cuts would overlap")
        for i, a in enumerate(obs):
            for b in obs[i + 1 :]:
                if _obstacles_overlap(a, b):
                    raise ValueError(f"obstacles overlap: {a} {b}")
        for i, o in enumerate(obs):
            # cut i must not pass through any other obstacle
            for j, other in enumerate(obs):
                if i == j:
                    continue
                x0, x1 = other.xrange
                if x0 - eps <= o.cx <= x1 + eps and other.yrange[0] < o.cut_top():
                    raise ValueError(f"cut of obstacle {i} hits obstacle {j}")
            for name, p in (("start", self.start), ("goal", self.goal)):
                if o.contains(p):
                    raise ValueError(f"{name} is inside obstacle {i}")
                if abs(p[0] - o.cx) < eps and p[1] <= o.cut_top():
                    raise ValueError(f"{name} lies on the cut of obstacle {i}")
        return True

    # ------------------------------------------------------------- collision
    def path_collides(self, path, margin=0.0):
        """True if any segment of the polyline intersects an obstacle (inflated by margin)."""
        path = np.asarray(path, dtype=float)
        P, Q = path[:-1], path[1:]
        return any(bool(o.segments_hit(P, Q, margin).any()) for o in self.obstacles)

    def collision_mask(self, paths, margin=0.0):
        """Batched version: paths (N, T, 2) -> bool (N,)."""
        paths = np.asarray(paths, dtype=float)
        N, T, _ = paths.shape
        P = paths[:, :-1].reshape(-1, 2)
        Q = paths[:, 1:].reshape(-1, 2)
        hit = np.zeros(len(P), dtype=bool)
        for o in self.obstacles:
            hit |= o.segments_hit(P, Q, margin)
        return hit.reshape(N, T - 1).any(axis=1)

    def penetration(self, paths, n_sub=12, relative=False):
        """Max depth inside any obstacle along each polyline, (N,T,2) -> (N,).

        Depth is the distance to the obstacle boundary for points inside
        (<= 0 everywhere means collision-free). Segments are subsampled.
        relative=True divides by the obstacle's inner radius (r, or min(hw, hh)),
        so 1.0 means the path reaches the obstacle's center.
        """
        paths = np.asarray(paths, dtype=float)
        s = np.linspace(0.0, 1.0, n_sub)
        pts = (paths[:, :-1, None, :] * (1 - s)[None, None, :, None]
               + paths[:, 1:, None, :] * s[None, None, :, None]).reshape(len(paths), -1, 2)
        depth = np.full(len(paths), -np.inf)
        for o in self.obstacles:
            dx = pts[..., 0] - o.cx
            dy = pts[..., 1] - o.cy
            if isinstance(o, Circle):
                d = o.r - np.hypot(dx, dy)
                size = o.r
            else:
                d = np.minimum(o.hw - np.abs(dx), o.hh - np.abs(dy))
                size = min(o.hw, o.hh)
            if relative:
                d = d / size
            depth = np.maximum(depth, d.max(axis=1))
        return depth

    # ----------------------------------------------------------- features
    def obstacle_array(self, max_obs=None):
        """(max_obs, 5) array [kind, cx, cy, half_w, half_h] plus a validity mask."""
        max_obs = max_obs or self.m
        arr = np.zeros((max_obs, 5))
        mask = np.zeros(max_obs, dtype=bool)
        for i, o in enumerate(self.obstacles):
            arr[i] = o.as_vector()
            mask[i] = True
        return arr, mask


def _obstacles_overlap(a, b):
    ax0, ax1 = a.xrange
    bx0, bx1 = b.xrange
    ay0, ay1 = a.yrange
    by0, by1 = b.yrange
    if ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0:
        return False
    if isinstance(a, Circle) and isinstance(b, Circle):
        return np.hypot(a.cx - b.cx, a.cy - b.cy) < a.r + b.r
    return True  # conservative for boxes


# ------------------------------------------------------------- layouts
def e0_env():
    """E0: single circular obstacle between fixed start and goal."""
    return Env([Circle(0.0, 0.0, 0.3)])


def random_layout(rng, n_obs, start=(-0.9, 0.0), goal=(0.9, 0.0), x_band=(-0.6, 0.6),
                  y_band=(-0.35, 0.35), r_range=(0.12, 0.22), box_prob=0.0, min_gap=0.06,
                  max_tries=200):
    """Random layout of n_obs obstacles with non-overlapping x-projections.

    Non-overlapping x-projections guarantee that downward cuts never hit another
    obstacle, so validate() always passes for these layouts.
    """
    for _ in range(max_tries):
        obs = []
        for _ in range(n_obs):
            size = rng.uniform(*r_range)
            cx = rng.uniform(*x_band)
            cy = rng.uniform(*y_band)
            if rng.random() < box_prob:
                obs.append(Box(cx, cy, size * rng.uniform(0.7, 1.0), size * rng.uniform(0.7, 1.3)))
            else:
                obs.append(Circle(cx, cy, size))
        obs.sort(key=lambda o: o.cx)
        ok = all(obs[i].xrange[1] + min_gap < obs[i + 1].xrange[0] for i in range(n_obs - 1))
        if not ok:
            continue
        env = Env(obs, start, goal)
        try:
            env.validate()
        except ValueError:
            continue
        return env
    raise RuntimeError("could not sample a valid layout; loosen the parameters")

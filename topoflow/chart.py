"""Class-specific chart for x-monotone routes (TDFM v2: Topology-Constrained Residual Flow).

A path is y-values on a fixed x-grid (N interior points between start and goal). For route code h and clearance c,
every grid segment [x_k, x_{k+1}] gets a conservative free band
    L_k = max over x in [x_k, x_{k+1}] of the lower walls (tops of inflated obstacles the route passes ABOVE),
    U_k = min over the same interval of the upper walls (bottoms of inflated obstacles it passes BELOW),
with outer limits [Y_MIN, Y_MAX]. A waypoint sits on two segments, so its chart is
    lo_k = max(L_{k-1}, L_k),  hi_k = min(U_{k-1}, U_k).
If every waypoint is inside [lo_k, hi_k], every segment lies inside the rectangle [x_k, x_{k+1}] x [L_k, U_k], which
is free of the c-inflated obstacles (inflated circles and boxes, matching Env.collision_mask(margin=c)). The
crossing word equals h because over each obstacle's x-range the path is on the side the code says.
Decoding: y = m + w * tanh(u), m = (lo + hi) / 2, w = (hi - lo) / 2 * (1 - EPS). u = 0 is the corridor centre line,
so u is the residual of the path around the class centre line; any u gives a valid path of class h.
"""
import numpy as np

Y_MIN, Y_MAX = -1.2, 1.2
EPS = 1e-3


def grid(env, n, c):
    """n interior x-positions: the x-edges of every c-inflated obstacle (so side switches between neighbouring
    obstacles happen inside an unconstrained segment), filled up with a uniform grid."""
    sx, gx = env.start[0], env.goal[0]
    edges = []
    for o in env.obstacles:
        R = (o.r if o.kind == "circle" else o.hw) + c
        edges += [o.cx - R - 1e-6, o.cx + R + 1e-6]
    edges = [e for e in edges if sx < e < gx]
    base = np.linspace(sx, gx, n - len(edges) + 2)[1:-1]
    return np.sort(np.concatenate([base, edges]))


def _wall(o, xa, xb, c, top):
    """max (top=True) of the inflated obstacle's upper boundary over [xa, xb], or min of its lower boundary;
    None if the inflated obstacle does not overlap the interval."""
    if o.kind == "circle":
        R = o.r + c
        lo, hi = o.cx - R, o.cx + R
        if xb <= lo or xa >= hi:
            return None
        xc = min(max(o.cx, xa), xb)                    # point of the interval closest to the centre
        h = np.sqrt(max(R * R - (xc - o.cx) ** 2, 0.0))
    else:
        lo, hi = o.cx - o.hw - c, o.cx + o.hw + c
        if xb <= lo or xa >= hi:
            return None
        h = o.hh + c
    return o.cy + h if top else o.cy - h


def walls(o, xa, xb, c, top):
    """Vectorized _wall over interval arrays xa, xb: wall height, NaN where the inflated obstacle misses."""
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    R = (o.r if o.kind == "circle" else o.hw) + c
    miss = (xb <= o.cx - R) | (xa >= o.cx + R)
    if o.kind == "circle":
        xc = np.clip(o.cx, xa, xb)
        h = np.sqrt(np.maximum(R * R - (xc - o.cx) ** 2, 0.0))
    else:
        h = np.full(xa.shape, o.hh + c)
    w = o.cy + h if top else o.cy - h
    return np.where(miss, np.nan, w)


def bounds_xs(env, code, c, xs_int):
    """Chart on an arbitrary strictly increasing interior grid xs_int. Returns (lo, hi, ok, L, U)."""
    xs = np.concatenate([[env.start[0]], xs_int, [env.goal[0]]])
    L = np.full(len(xs) - 1, Y_MIN)
    U = np.full(len(xs) - 1, Y_MAX)
    for o, b in zip(env.obstacles, code):
        w = walls(o, xs[:-1], xs[1:], c, top=(b == 0))
        if b == 0:
            L = np.where(np.isnan(w), L, np.maximum(L, w))
        else:
            U = np.where(np.isnan(w), U, np.minimum(U, w))
    lo = np.maximum(L[:-1], L[1:])
    hi = np.minimum(U[:-1], U[1:])
    # start/goal must sit inside the first/last segment band
    ok = bool(np.all(hi > lo) and L[0] <= env.start[1] <= U[0] and L[-1] <= env.goal[1] <= U[-1])
    return lo, hi, ok, L, U


def bounds(env, code, c, n):
    """(lo, hi) arrays of length n for the interior grid points, plus feasibility flag (all lo < hi)."""
    lo, hi, ok, _, _ = bounds_xs(env, code, c, grid(env, n, c))
    return lo, hi, ok


def encode(y, lo, hi):
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - EPS)
    s = np.clip((y - m) / w, -1 + 1e-6, 1 - 1e-6)
    return np.arctanh(s)


def decode(u, lo, hi):
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - EPS)
    return m + w * np.tanh(u)


def to_grid(env, path, n, c):
    """Resample an x-monotone polyline (T, 2) to y-values on the chart grid."""
    return np.interp(grid(env, n, c), path[:, 0], path[:, 1])


def to_path(env, y, c):
    """y (..., n) -> full polyline (..., n + 2, 2) with start and goal."""
    n = y.shape[-1]
    xs = grid(env, n, c)
    P = np.empty(y.shape + (2,))
    P[..., 0] = xs
    P[..., 1] = y
    S = np.broadcast_to(np.array(env.start, float), y.shape[:-1] + (1, 2))
    G = np.broadcast_to(np.array(env.goal, float), y.shape[:-1] + (1, 2))
    return np.concatenate([S, P, G], -2)

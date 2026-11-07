"""Independent geometry for the final validation (Experiment A/B/C checkers).

Written without importing topoflow.chart, Env.collision_mask, Env.segments_hit or topoflow.topology.
Obstacles are read only through their public parameters (kind, cx, cy, r / hw, hh).

- seg_obstacle_dist:  exact Euclidean distance from segments to a disk or an axis-aligned box (float64).
- seg_meets_open_rect: does a segment meet the OPEN rectangle (square-inflated box) — separating-interval test.
- exact_*:            the same quantities in exact rational arithmetic (Fraction of the float inputs).
- crossing_word:      reduced crossing word with downward cuts (signed ray crossings, free reduction).
- setup_checks:       A1/A3/cut-validity checks of a layout.
"""
from fractions import Fraction as F

import numpy as np


# ---------------------------------------------------------------- float64 geometry
def _pt_seg_d2(px, py, ax, ay, bx, by):
    """squared distance from points (px,py) to segments a-b (broadcast)."""
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    t = np.where(dd > 0, ((px - ax) * dx + (py - ay) * dy) / np.where(dd > 0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * dx - px, ay + t * dy - py
    return qx * qx + qy * qy


def _pt_rect_d2(px, py, x0, x1, y0, y1):
    ex = np.maximum(np.maximum(x0 - px, px - x1), 0.0)
    ey = np.maximum(np.maximum(y0 - py, py - y1), 0.0)
    return ex * ex + ey * ey


def seg_meets_rect(ax, ay, bx, by, x0, x1, y0, y1, open_=False):
    """Liang–Barsky style: parameter interval where the point is inside the (closed / open) rectangle."""
    dx, dy = bx - ax, by - ay
    t0 = np.zeros(np.broadcast(ax, x0).shape)
    t1 = np.ones_like(t0)
    ok = np.ones(t0.shape, bool)
    for p, d, lo, hi in ((ax, dx, x0, x1), (ay, dy, y0, y1)):
        par = d == 0
        inside = (p > lo) & (p < hi) if open_ else (p >= lo) & (p <= hi)
        ok &= ~par | inside
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (lo - p) / d, (hi - p) / d
        tmin = np.where(par, -np.inf, np.minimum(ta, tb))
        tmax = np.where(par, np.inf, np.maximum(ta, tb))
        t0, t1 = np.maximum(t0, tmin), np.minimum(t1, tmax)
    return ok & ((t0 < t1) if open_ else (t0 <= t1))


def seg_obstacle_dist(P, Q, o):
    """P, Q (..., 2) segment endpoints -> Euclidean distance to obstacle o (0 if they meet)."""
    ax, ay, bx, by = P[..., 0], P[..., 1], Q[..., 0], Q[..., 1]
    if o.kind == "circle":
        return np.maximum(np.sqrt(_pt_seg_d2(o.cx, o.cy, ax, ay, bx, by)) - o.r, 0.0)
    x0, x1, y0, y1 = o.cx - o.hw, o.cx + o.hw, o.cy - o.hh, o.cy + o.hh
    d2 = np.minimum(_pt_rect_d2(ax, ay, x0, x1, y0, y1), _pt_rect_d2(bx, by, x0, x1, y0, y1))
    for cx_, cy_ in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
        d2 = np.minimum(d2, _pt_seg_d2(cx_, cy_, ax, ay, bx, by))
    meet = seg_meets_rect(ax, ay, bx, by, x0, x1, y0, y1)
    return np.where(meet, 0.0, np.sqrt(d2))


def seg_inflated_rect_dist(P, Q, o, c):
    """distance from segments to the square-inflated rectangle of o (boxes); for disks, distance to the c-disk."""
    if o.kind == "circle":
        return seg_obstacle_dist(P, Q, o) - c
    class _B:  # noqa: N801
        kind, cx, cy, hw, hh = "box", o.cx, o.cy, o.hw + c, o.hh + c
    return seg_obstacle_dist(P, Q, _B)


# ---------------------------------------------------------------- exact rational geometry
def _f(v):
    return F(float(v))


def exact_pt_seg_d2(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    dd = dx * dx + dy * dy
    t = F(0) if dd == 0 else min(max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / dd, F(0)), F(1))
    qx, qy = a[0] + t * dx - p[0], a[1] + t * dy - p[1]
    return qx * qx + qy * qy


def exact_seg_meets_rect(a, b, x0, x1, y0, y1, open_):
    t0, t1 = F(0), F(1)
    for p, d, lo, hi in ((a[0], b[0] - a[0], x0, x1), (a[1], b[1] - a[1], y0, y1)):
        if d == 0:
            if not ((lo < p < hi) if open_ else (lo <= p <= hi)):
                return False
            continue
        ta, tb = (lo - p) / d, (hi - p) / d
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    return t0 < t1 if open_ else t0 <= t1


def exact_clearance_ok(P, Q, o, c):
    """Exact: Euclidean distance(segment, o) >= c ? Returns (ok, dist^2 - c^2 as Fraction)."""
    a, b = (_f(P[0]), _f(P[1])), (_f(Q[0]), _f(Q[1]))
    cc = _f(c)
    if o.kind == "circle":
        d2 = exact_pt_seg_d2((_f(o.cx), _f(o.cy)), a, b)
        R = _f(o.r) + cc
        return d2 >= R * R, d2 - R * R            # dist(centre, seg) >= r + c  <=>  dist(disk, seg) >= c
    x0, x1 = _f(o.cx) - _f(o.hw), _f(o.cx) + _f(o.hw)
    y0, y1 = _f(o.cy) - _f(o.hh), _f(o.cy) + _f(o.hh)
    if exact_seg_meets_rect(a, b, x0, x1, y0, y1, open_=False):
        return False, -cc * cc

    def prd(p):
        ex = max(x0 - p[0], p[0] - x1, F(0))
        ey = max(y0 - p[1], p[1] - y1, F(0))
        return ex * ex + ey * ey
    d2 = min(prd(a), prd(b), *(exact_pt_seg_d2(q, a, b) for q in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))))
    return d2 >= cc * cc, d2 - cc * cc


def exact_open_rect_meet(P, Q, o, c):
    a, b = (_f(P[0]), _f(P[1])), (_f(Q[0]), _f(Q[1]))
    cc = _f(c)
    if o.kind == "circle":
        d2 = exact_pt_seg_d2((_f(o.cx), _f(o.cy)), a, b)
        R = _f(o.r) + cc
        return d2 < R * R
    return exact_seg_meets_rect(a, b, _f(o.cx) - _f(o.hw) - cc, _f(o.cx) + _f(o.hw) + cc,
                                _f(o.cy) - _f(o.hh) - cc, _f(o.cy) + _f(o.hh) + cc, open_=True)


# ---------------------------------------------------------------- topology
def lowest_y(o):
    return o.cy - (o.r if o.kind == "circle" else o.hh)


def crossing_word(path, obstacles):
    """Reduced word of signed crossings of the polyline with the downward cuts x = cx_j, y < lowest_y(o_j).
    Convention: a crossing left->right is (j, +1); half-open rule P.x < cx <= Q.x avoids double counting at vertices."""
    P, Q = path[:-1], path[1:]
    ev = []
    for j, o in enumerate(obstacles):
        x, yb = o.cx, lowest_y(o)
        right = (P[:, 0] < x) & (x <= Q[:, 0])
        left = (Q[:, 0] < x) & (x <= P[:, 0])
        for k in np.where(right | left)[0]:
            t = (x - P[k, 0]) / (Q[k, 0] - P[k, 0])
            y = P[k, 1] + t * (Q[k, 1] - P[k, 1])
            if y < yb:
                ev.append((k + t, j, 1 if right[k] else -1))
    ev.sort()
    stack = []
    for _, j, s in ev:
        if stack and stack[-1] == (j, -s):
            stack.pop()
        else:
            stack.append((j, s))
    return tuple(stack)


def word_of_code(code):
    return tuple((j, 1) for j, b in enumerate(code) if b == 1)


# ---------------------------------------------------------------- setup
def inflated_half_width(o, c):
    return (o.r if o.kind == "circle" else o.hw) + c


def _obst_dist(a, b):
    """Euclidean distance between two obstacles (0 if they meet)."""
    if a.kind == "circle" and b.kind == "circle":
        return max(np.hypot(a.cx - b.cx, a.cy - b.cy) - a.r - b.r, 0.0)
    if a.kind == "box" and b.kind == "box":
        ex = max(abs(a.cx - b.cx) - a.hw - b.hw, 0.0)
        ey = max(abs(a.cy - b.cy) - a.hh - b.hh, 0.0)
        return float(np.hypot(ex, ey))
    c_, bx = (a, b) if a.kind == "circle" else (b, a)
    ex = max(abs(c_.cx - bx.cx) - bx.hw, 0.0)
    ey = max(abs(c_.cy - bx.cy) - bx.hh, 0.0)
    return max(float(np.hypot(ex, ey)) - c_.r, 0.0)


def setup_checks(env, c):
    obs = env.obstacles
    xs, xg = float(env.start[0]), float(env.goal[0])
    a3 = all(o.cx - inflated_half_width(o, c) > xs and o.cx + inflated_half_width(o, c) < xg for o in obs)
    a3_margin = min(min(o.cx - inflated_half_width(o, c) - xs, xg - o.cx - inflated_half_width(o, c)) for o in obs)
    disjoint = all(_obst_dist(a, b) > 0 for i, a in enumerate(obs) for b in obs[i + 1:])
    distinct_x = len({round(float(o.cx), 12) for o in obs}) == len(obs)
    sorted_x = all(obs[i].cx < obs[i + 1].cx for i in range(len(obs) - 1))
    cuts_ok = True
    for i, o in enumerate(obs):
        yb = lowest_y(o)
        for j, q in enumerate(obs):
            if i == j:
                continue
            dx = abs(o.cx - q.cx)
            if q.kind == "circle":
                if dx < q.r and q.cy - np.sqrt(q.r ** 2 - dx ** 2) < yb:
                    cuts_ok = False
            elif dx <= q.hw and q.cy - q.hh < yb:
                cuts_ok = False
        for p in (env.start, env.goal):
            if abs(p[0] - o.cx) < 1e-12 and p[1] < yb:
                cuts_ok = False
    return dict(a3=a3, a3_margin=float(a3_margin), disjoint=disjoint, distinct_x=distinct_x, sorted_x=sorted_x,
                cuts_ok=cuts_ok)

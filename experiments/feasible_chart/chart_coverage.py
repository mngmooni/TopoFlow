"""research phase: chart success rate and coverage (locked: the archived prediction record).

For every (layout, code) of a dataset split:
  found / not_found (planner), chart feasible at n = 60 (default grid), uniform n = 120 / 240, edge-dense G2 (n = 60),
  exact x-monotone tube feasible (dense pointwise free interval), waypoint-band ratio (chart width / exact width at
  grid points where some obstacle wall is active), reference paths inside the chart, and for infeasible charts the
  binding obstacles and cause.

  TOPO_MAX_OBS=5 DATA=results/d1 python experiments/feasible_chart/chart_coverage.py
Writes {DATA}/chart_coverage.json (summary) and {DATA}/chart_coverage_rows.pkl (per chart).
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402
from topoflow.tdfm import boltzmann  # noqa: E402

DATA = os.environ.get("DATA", "results/d1")
SPLITS = os.environ.get("SPLITS", "unseen,train").split(",")
N_TRAIN = int(os.environ.get("N_TRAIN", "300"))
meta = json.load(open(f"{DATA}/betas.json"))
C = float(meta.get("_margin", 0.03))
BETA = meta["medium"]["beta"]
N = 60
DENSE = 4000


def R_of(o):
    return (o.r if o.kind == "circle" else o.hw) + C


def edges(env):
    e = []
    for o in env.obstacles:
        for x in (o.cx - R_of(o), o.cx + R_of(o)):
            e += [x - 1e-6, x + 1e-6]  # both sides of every inflated edge (thin x-overlap strips must be sampled)
    return [x for x in e if env.start[0] < x < env.goal[0]]


def grid_uniform(env, n):
    return chart.grid(env, n, C)


def grid_g2(env, n=N, d=0.005):
    sx, gx = env.start[0], env.goal[0]
    pts = []
    for o in env.obstacles:
        a, b = o.cx - R_of(o), o.cx + R_of(o)
        pts += [a - 1e-6, b + 1e-6, a - d, a + d, b - d, b + d]
    pts = sorted({p for p in pts if sx < p < gx})
    base = np.linspace(sx, gx, n - len(pts) + 2)[1:-1]
    return np.sort(np.concatenate([base, pts]))


def pointwise(env, code, xs):
    """Exact free interval [L(x), U(x)] of the class at points xs (c-inflated obstacles)."""
    L = np.full(len(xs), chart.Y_MIN)
    U = np.full(len(xs), chart.Y_MAX)
    active = np.zeros(len(xs), bool)
    for o, b in zip(env.obstacles, code):
        R = R_of(o)
        inside = np.abs(xs - o.cx) < R
        if o.kind == "circle":
            h = np.sqrt(np.maximum(R * R - (xs - o.cx) ** 2, 0.0))
        else:
            h = np.full(len(xs), o.hh + C)
        if b == 0:
            L = np.where(inside, np.maximum(L, o.cy + h), L)
        else:
            U = np.where(inside, np.minimum(U, o.cy - h), U)
        active |= inside
    return L, U, active


def exact_feasible(env, code):
    xs = np.sort(np.concatenate([np.linspace(env.start[0], env.goal[0], DENSE), edges(env)]))
    L, U, _ = pointwise(env, code, xs)
    return bool(np.all(L < U) and L[0] <= env.start[1] <= U[0] and L[-1] <= env.goal[1] <= U[-1]), \
        float(np.min(U - L))


def cause(env, code, xs_int, lo, hi, L, U):
    """Binding obstacles at the first failing waypoint; category string."""
    if not (L[0] <= env.start[1] <= U[0] and L[-1] <= env.goal[1] <= U[-1]):
        return "start_goal", None, None
    k = int(np.argmax(hi <= lo))
    x = xs_int[k]
    xs = np.concatenate([[env.start[0]], xs_int, [env.goal[0]]])
    segs = [k, k + 1]  # segments adjacent to waypoint k (segment index = left node index)
    best_lo, j_lo, best_hi, j_hi = -np.inf, None, np.inf, None
    for j, (o, b) in enumerate(zip(env.obstacles, code)):
        w = chart.walls(o, xs[segs], xs[[s + 1 for s in segs]], C, top=(b == 0))
        if np.all(np.isnan(w)):
            continue
        if b == 0 and np.nanmax(w) > best_lo:
            best_lo, j_lo = np.nanmax(w), j
        if b == 1 and np.nanmin(w) < best_hi:
            best_hi, j_hi = np.nanmin(w), j
    if j_lo is None or j_hi is None:
        return "y_bounds", j_lo, j_hi
    a, b = env.obstacles[j_lo], env.obstacles[j_hi]
    gap = max(a.cx - R_of(a), b.cx - R_of(b)) - min(a.cx + R_of(a), b.cx + R_of(b))  # < 0: inflated x-overlap
    adjacent = abs(j_lo - j_hi) == 1
    if gap < 0:
        return ("pair_overlap" if adjacent else "pair_overlap_nonadj"), j_lo, j_hi
    return "pair_separated", j_lo, j_hi


def overlap_switch(env, code):
    """Some x-adjacent pair with overlapping c-inflated x-ranges and different bits."""
    obs = env.obstacles
    order = np.argsort([o.cx for o in obs])
    for p, q in zip(order[:-1], order[1:]):
        a, b = obs[p], obs[q]
        if a.cx + R_of(a) > b.cx - R_of(b) and code[p] != code[q]:
            return True
    return False


rows = []
summary = {}
for split in SPLITS:
    with open(f"{DATA}/layouts.pkl", "rb") as f:
        sets = pickle.load(f)
    lays = sets[1] if split == "unseen" else sets[0][:N_TRAIN]
    with open(f"{DATA}/costs.pkl", "rb") as f:
        costs = pickle.load(f)[split][: len(lays)]
    import itertools
    for i, lay in enumerate(lays):
        env = lay["env"]
        p = boltzmann(costs[i], BETA)
        for code in itertools.product([0, 1], repeat=env.m):
            r = dict(split=split, i=i, m=env.m, code=code, found=code in lay["found"], p=float(p.get(code, 0.0)))
            xs60 = grid_uniform(env, N)
            lo, hi, ok, L, U = chart.bounds_xs(env, code, C, xs60)
            r["ok60"] = ok
            for n in (120, 240):
                r[f"ok{n}"] = chart.bounds_xs(env, code, C, grid_uniform(env, n))[2]
            r["okG2"] = chart.bounds_xs(env, code, C, grid_g2(env))[2]
            r["exact"], r["exact_mingap"] = exact_feasible(env, code)
            r["overlap_switch"] = overlap_switch(env, code)
            if ok:
                Le, Ue, act = pointwise(env, code, xs60)
                den = np.clip(Ue - Le, 0, None)[act].sum()
                r["band_ratio"] = float(np.clip(hi - lo, 0, None)[act].sum() / den) if den > 0 else None
                if r["found"]:
                    Y = np.stack([chart.to_grid(env, q, N, C) for q in lay["found"][code]])
                    r["ref_inside"] = float(np.mean(np.all((Y >= lo) & (Y <= hi), 1)))
            else:
                r["cause"], r["j_lo"], r["j_hi"] = cause(env, code, xs60, lo, hi, L, U)
            rows.append(r)

    R = [r for r in rows if r["split"] == split]
    F = [r for r in R if r["found"]]
    bad = [r for r in F if not r["ok60"]]
    nf = [r for r in R if not r["found"]]
    by_layout = {}
    for r in R:
        by_layout.setdefault(r["i"], 0.0)
        if not r["ok60"]:
            by_layout[r["i"]] += r["p"]
    s = dict(
        charts_all=len(R), found=len(F), found_ok60=float(np.mean([r["ok60"] for r in F])),
        found_ok120=float(np.mean([r["ok120"] for r in F])), found_ok240=float(np.mean([r["ok240"] for r in F])),
        found_okG2=float(np.mean([r["okG2"] for r in F])),
        found_exact=float(np.mean([r["exact"] for r in F])),
        n_bad60=len(bad),
        bad_recovered={k: int(sum(r[k] for r in bad)) for k in ("ok120", "ok240", "okG2", "exact")},
        bad_overlap_switch=int(sum(r["overlap_switch"] for r in bad)),
        bad_causes={c: sum(r["cause"] == c for r in bad) for c in sorted({r["cause"] for r in bad})},
        bad_by_m={m: [sum(1 for r in bad if r["m"] == m), sum(1 for r in F if r["m"] == m)] for m in (3, 4, 5)},
        pmass_uncovered_mean=float(np.mean(list(by_layout.values()))),
        pmass_uncovered_max=float(np.max(list(by_layout.values()))),
        pmass_bad_found_mean=float(np.sum([r["p"] for r in bad]) / len(by_layout)),
        layouts_with_uncovered_found=len({r["i"] for r in bad}),
        notfound=len(nf), notfound_ok60=int(sum(r["ok60"] for r in nf)), notfound_exact=int(sum(r["exact"] for r in nf)),
        exact_all=int(sum(r["exact"] for r in R)), exact_ok60=int(sum(r["exact"] and r["ok60"] for r in R)),
        ok60_not_exact=int(sum(r["ok60"] and not r["exact"] for r in R)),
        band_ratio_median=float(np.median([r["band_ratio"] for r in F if r.get("band_ratio") is not None])),
        band_ratio_p10=float(np.percentile([r["band_ratio"] for r in F if r.get("band_ratio") is not None], 10)),
        ref_inside_mean=float(np.mean([r["ref_inside"] for r in F if "ref_inside" in r])),
        found_overlap_switch=float(np.mean([r["overlap_switch"] for r in F])),
    )
    summary[split] = s
    print(split, json.dumps(s), flush=True)

json.dump(dict(c=C, summary=summary), open(f"{DATA}/chart_coverage.json", "w"), indent=1)
with open(f"{DATA}/chart_coverage_rows.pkl", "wb") as f:
    pickle.dump(rows, f)

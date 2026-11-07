"""Experiment A2 — chart coverage and failure taxonomy (locked: the frozen prediction record (private archive)).

Per (layout, route code), all 2^m codes: planner status x taxonomy. The infeasibility certificate is independent of
topoflow/chart.py: delta(h) = min_x U(x) - L(x) of the class's pointwise free interval, computed pairwise, since
U - L = min over (j below-passed, i above-passed, both active at x) of bot_j(x) - top_i(x), and each pair term is convex
in x (tops are concave, bottoms convex) -> golden-section minimisation on the common inflated x-range. A negative
pair minimum comes with a witness x* that can be re-evaluated directly. Square-corner box inflation = the chart's
geometry (and the M metric's); rounded-corner inflation = Euclidean clearance c.

  python experiments/validation/chart_coverage_audit.py     (after experiments/validation/chart_soundness.py for every dataset)
Writes results/final/chart_coverage_audit.json.
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import indep_geom as G  # noqa: E402

C, TOL, YMIN, YMAX = 0.01, 1e-9, -1.2, 1.2
DSETS = ["d2", "d3", "d4", "H1", "H2", "H3", "H4"]
GR = (np.sqrt(5) - 1) / 2


def wall(o, x, top, rounded):
    """upper (top) / lower boundary of the c-inflated obstacle at x (x inside its inflated x-range)."""
    if o.kind == "circle":
        h = np.sqrt(np.maximum((o.r + C) ** 2 - (x - o.cx) ** 2, 0.0))
    elif rounded:
        h = o.hh + np.sqrt(np.maximum(C * C - np.maximum(np.abs(x - o.cx) - o.hw, 0.0) ** 2, 0.0))
    else:
        h = o.hh + C + 0.0 * x
    return o.cy + h if top else o.cy - h


def pair_min(a, b, rounded):
    """min over the common inflated x-range of bot_b(x) - top_a(x) (a passed above, b passed below)."""
    Ra, Rb = G.inflated_half_width(a, C), G.inflated_half_width(b, C)
    lo, hi = max(a.cx - Ra, b.cx - Rb), min(a.cx + Ra, b.cx + Rb)
    if lo > hi:
        return np.inf, None
    f = lambda x: float(wall(b, x, False, rounded) - wall(a, x, True, rounded))  # noqa: E731
    x0, x1 = lo, hi
    for _ in range(200):
        m1, m2 = x1 - GR * (x1 - x0), x0 + GR * (x1 - x0)
        if f(m1) <= f(m2):
            x1 = m2
        else:
            x0 = m1
        if x1 - x0 < 1e-15:
            break
    cands = [lo, hi, (x0 + x1) / 2]
    v = [f(x) for x in cands]
    k = int(np.argmin(v))
    return v[k], cands[k]


def layout_terms(env, rounded):
    obs = env.obstacles
    P = {(i, j): pair_min(obs[i], obs[j], rounded) for i in range(len(obs)) for j in range(len(obs)) if i != j}
    yt = [YMAX - float(wall(o, o.cx, True, rounded)) for o in obs]      # top is maximal at cx
    yb = [float(wall(o, o.cx, False, rounded)) - YMIN for o in obs]
    return P, yt, yb


def delta(code, P, yt, yb):
    A = [i for i, b in enumerate(code) if b == 0]
    B = [j for j, b in enumerate(code) if b == 1]
    d_obs, wit = np.inf, None
    for i in A:
        for j in B:
            v, x = P[(i, j)]
            if v < d_obs:
                d_obs, wit = v, (i, j, x)
    d_y = min([yt[i] for i in A] + [yb[j] for j in B] + [YMAX - YMIN])
    return d_obs, d_y, wit


def endpoint_ok(env, code, rounded):
    """endpoints inside the class's free interval (only binding when A3 fails)."""
    for p in (env.start, env.goal):
        for o, b in zip(env.obstacles, code):
            if abs(p[0] - o.cx) <= G.inflated_half_width(o, C):
                w = float(wall(o, p[0], b == 0, rounded))
                if (b == 0 and p[1] <= w) or (b == 1 and p[1] >= w):
                    return False
    return True


def classify(r, setup):
    if r["ok60"]:
        return "covered_n60"
    if r.get("ok_bisect"):
        return "resolution_recovered"
    dsq = min(r["d_obs_sq"], r["d_y_sq"])
    if r["d_obs_sq"] < -TOL or not r["endpoint_sq"]:
        if r["d_obs_rd"] < -TOL or not r["endpoint_rd"]:
            return "certified_infeasible"
        if r["d_obs_rd"] > TOL:
            return "conservative_exclusion"
        return "degenerate"
    if r["d_y_sq"] <= TOL < r["d_obs_sq"]:
        return "y_bound_exclusion"
    if abs(dsq) <= TOL:
        return "degenerate"
    return "resolution_unrecovered"


out = {}
for ds in DSETS:
    sd = pickle.load(open(f"results/final/soundness/{ds}_codes.pkl", "rb"))
    lays = pickle.load(open(f"results/hard/{ds}_layouts.pkl", "rb")) if ds.startswith("H") else \
        pickle.load(open(f"results/{ds}/layouts.pkl", "rb"))[1]
    terms = {}
    rows = []
    for r in sd["rows"]:
        i = r["i"]
        env = lays[i]["env"]
        if i not in terms:
            terms[i] = (layout_terms(env, False), layout_terms(env, True))
        (Ps, yts, ybs), (Pr, ytr, ybr) = terms[i]
        q = dict(r)
        q["d_obs_sq"], q["d_y_sq"], q["witness_sq"] = delta(r["code"], Ps, yts, ybs)
        q["d_obs_rd"], q["d_y_rd"], _ = delta(r["code"], Pr, ytr, ybr)
        q["endpoint_sq"], q["endpoint_rd"] = endpoint_ok(env, r["code"], False), endpoint_ok(env, r["code"], True)
        su = sd["setups"][i]
        q["setup_ok"] = su["a3"] and su["disjoint"] and su["cuts_ok"] and su["distinct_x"]
        q["cat"] = classify(q, su)
        # witness re-evaluation (certificate check, direct formula)
        if q["witness_sq"] is not None and q["d_obs_sq"] < -TOL:
            a, b, x = q["witness_sq"]
            oa, ob = env.obstacles[a], env.obstacles[b]
            act = abs(x - oa.cx) <= G.inflated_half_width(oa, C) + 1e-12 and abs(x - ob.cx) <= G.inflated_half_width(ob, C) + 1e-12
            q["witness_ok"] = bool(act and wall(ob, x, False, False) - wall(oa, x, True, False) < -TOL)
        rows.append(q)
    planner = "none" if ds.startswith("H") else "sampling planner"
    cats = ["covered_n60", "resolution_recovered", "certified_infeasible", "conservative_exclusion",
            "y_bound_exclusion", "degenerate", "resolution_unrecovered"]

    def tab(R):
        return {c: sum(q["cat"] == c for q in R) for c in cats}
    pos = [q for q in rows if min(q["d_obs_sq"], q["d_y_sq"]) > TOL and q["endpoint_sq"]]
    s = dict(planner=planner, layouts=len(lays), codes=len(rows), taxonomy_all=tab(rows),
             invalid_setup_layouts=sum(not (su["a3"] and su["disjoint"] and su["cuts_ok"] and su["distinct_x"])
                                       for su in sd["setups"]),
             codes_on_invalid_setup=sum(not q["setup_ok"] for q in rows),
             taxonomy_on_invalid_setup=tab([q for q in rows if not q["setup_ok"]]),
             delta_pos_codes=len(pos), delta_pos_ok60=sum(q["ok60"] for q in pos),
             delta_pos_ok60_frac=float(np.mean([q["ok60"] for q in pos])) if pos else None,
             delta_pos_covered_after_bisect=sum(q["ok60"] or q.get("ok_bisect", False) for q in pos),
             delta_gt_1e6_uncovered=sum(not (q["ok60"] or q.get("ok_bisect", False)) for q in pos if
                                       min(q["d_obs_sq"], q["d_y_sq"]) > 1e-6),
             max_bisect_points=max([q.get("n_bisect", 0) for q in rows] or [0]),
             covered_but_certified_infeasible=sum((q["ok60"] or q.get("ok_bisect", False)) and q["d_obs_sq"] < -TOL for q in rows),
             witnesses_checked=sum("witness_ok" in q for q in rows),
             witnesses_failed=sum(q.get("witness_ok") is False for q in rows),
             certified_min_clearance_of_checked=min([q["m_eu"] for q in rows if "m_eu" in q] or [None]) + C
             if any("m_eu" in q for q in rows) else None)
    if planner != "none":
        F_ = [q for q in rows if q["found"]]
        NF = [q for q in rows if not q["found"]]
        s.update(found=len(F_), not_found=len(NF), taxonomy_found=tab(F_), taxonomy_not_found=tab(NF),
                 found_certified_infeasible=sum(q["cat"] == "certified_infeasible" for q in F_),
                 found_conservative=sum(q["cat"] == "conservative_exclusion" for q in F_),
                 not_found_but_chart_feasible=sum(q["ok60"] or q.get("ok_bisect", False) for q in NF))
    if ds in ("d2", "d3"):
        old = pickle.load(open(f"results/{ds}/chart_coverage_rows.pkl", "rb"))
        old = {(o["i"], tuple(o["code"])): o for o in old if o["split"] == "unseen"}
        dis, comp = [], 0
        for q in rows:
            o = old.get((q["i"], tuple(q["code"])))
            if o is None:
                continue
            d = min(q["d_obs_sq"], q["d_y_sq"])
            if abs(d) > 1e-3:
                comp += 1
                if (d > 0) != bool(o["exact"]):
                    dis.append(dict(i=q["i"], code=q["code"], delta=d, old=o["exact"], old_gap=o["exact_mingap"]))
        s.update(old_dense_compared=comp, old_dense_disagree=len(dis), old_dense_disagree_rows=dis[:20])
    out[ds] = s
    print(ds, json.dumps({k: v for k, v in s.items() if "rows" not in k}), flush=True)
    pickle.dump(rows, open(f"results/final/soundness/{ds}_audit_rows.pkl", "wb"))
json.dump(out, open("results/final/chart_coverage_audit.json", "w"), indent=1, default=str)

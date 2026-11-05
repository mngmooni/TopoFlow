"""Day 12 step 2: chart-based feasibility labels and their effect on the selector target (D1).
(locked: outputs/predictions_day12_labels.md)

  TOPO_MAX_OBS=5 DATA=outputs/d1 python scripts/relabel.py   -> {DATA}/relabel.json, {DATA}/chart_labels.pkl
"""
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow import chart  # noqa: E402
from topoflow.selectors import CostRegSelector, cost_targets, probs_from_cost, train_costlike  # noqa: E402
from topoflow.tdfm import (boltzmann, calibrate_beta, code_index, mean_norm_entropy, selector_inputs,  # noqa: E402
                           structural_mask, target_vector)

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d1")
meta = json.load(open(f"{DATA}/betas.json"))
C = float(meta.get("_margin", 0.03))
N = 60
SG = os.environ.get("SG", "0") == "1"  # D3: selector also takes (start_y, goal_y)
REGIMES = {"uniform": 1.0, "medium": 0.5, "concentrated": 0.1}
sets = dict(zip(["train", "unseen", "calib"], pickle.load(open(f"{DATA}/layouts.pkl", "rb"))))
Crep = pickle.load(open(f"{DATA}/costs.pkl", "rb"))


def refined_grid(env, code, maxp=400):
    xs = chart.grid(env, N, C)
    for _ in range(300):
        lo, hi, ok, L, U = chart.bounds_xs(env, code, C, xs)
        if ok:
            return xs, lo, hi
        if len(xs) >= maxp:
            return None
        full = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
        new = set()
        for k in np.where(hi <= lo)[0] + 1:
            new |= {(full[k - 1] + full[k]) / 2, (full[k] + full[k + 1]) / 2}
        xs = np.unique(np.concatenate([xs, list(new)]))
    return None


def chart_shortest(env, xs, lo, hi):
    """Shortest polyline start -> (xs, y) -> goal with lo <= y <= hi. Convex; L-BFGS-B."""
    X = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
    dx = np.diff(X)
    sy, gy = env.start[1], env.goal[1]
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - chart.EPS)

    def f(y):
        Y = np.concatenate([[sy], y, [gy]])
        dy = np.diff(Y)
        s = np.sqrt(dx ** 2 + dy ** 2)
        g = dy / s
        return s.sum(), g[:-1] - g[1:]

    y0 = np.clip(np.interp(xs, [X[0], X[-1]], [sy, gy]), m - w, m + w)
    r = minimize(f, y0, jac=True, method="L-BFGS-B", bounds=list(zip(m - w, m + w)), options=dict(maxiter=500))
    return float(r.fun)


t0 = time.time()
labels_path = f"{DATA}/chart_labels.pkl"
if os.path.exists(labels_path):
    LAB = pickle.load(open(labels_path, "rb"))
else:
    LAB = {}
    for name, lays in sets.items():
        out = []
        for lay in lays:
            env = lay["env"]
            d = {}
            for code in itertools.product([0, 1], repeat=env.m):
                g = refined_grid(env, code)
                if g is not None:
                    d[code] = chart_shortest(env, *g)
            out.append(d)
        LAB[name] = out
        print(name, "labels done", round(time.time() - t0), flush=True)
    pickle.dump(LAB, open(labels_path, "wb"))

# cost dicts: T0 rep/found, T0' chart/found, T1 chart/chart-feasible
COST = {}
for name, lays in sets.items():
    COST[("T0", name)] = Crep[name]
    COST[("T0p", name)] = [{c: LAB[name][i][c] for c in lay["found"] if c in LAB[name][i]}
                           for i, lay in enumerate(lays)]
    COST[("T1", name)] = LAB[name]
res = dict(c=C)
# coverage sanity: found codes without a chart cost (should be 0)
res["found_without_chart"] = int(sum(c not in LAB["unseen"][i] for i, l in enumerate(sets["unseen"]) for c in l["found"]))
res["n_found_unseen"] = int(sum(len(l["found"]) for l in sets["unseen"]))
res["n_chart_unseen"] = int(sum(len(d) for d in LAB["unseen"]))
BETA = {T: {r: (0.0 if h >= 1.0 else calibrate_beta(COST[(T, "calib")], h)) for r, h in REGIMES.items()}
        for T in ("T0", "T0p", "T1")}
res["beta"] = BETA
res["entropy_unseen"] = {T: {r: mean_norm_entropy(COST[(T, "unseen")], b) for r, b in BETA[T].items()} for T in BETA}


def tvec(T, r, name="unseen"):
    return np.array([target_vector(boltzmann(c, BETA[T][r]), None) for c in COST[(T, name)]])


def tv(P, Q):
    return 0.5 * np.abs(P - Q).sum(1)


res["tv_targets"] = {r: dict(T0_T0p=float(tv(tvec("T0", r), tvec("T0p", r)).mean()),
                             T0p_T1=float(tv(tvec("T0p", r), tvec("T1", r)).mean()),
                             T0_T1=float(tv(tvec("T0", r), tvec("T1", r)).mean()),
                             T0p_T1_p90=float(np.percentile(tv(tvec("T0p", r), tvec("T1", r)), 90)),
                             T1_mass_on_new=float(np.mean([sum(p for c, p in boltzmann(LAB["unseen"][i], BETA["T1"][r]).items()
                                                               if c not in l["found"]) for i, l in enumerate(sets["unseen"])])))
                     for r in REGIMES}
cheapest_new = [min(d, key=d.get) not in l["found"] for d, l in zip(LAB["unseen"], sets["unseen"])]
res["frac_layouts_cheapest_is_new"] = float(np.mean(cheapest_new))
print(json.dumps(res, indent=1), flush=True)

# ---------------------------------------------------------------- selectors
Xtr, Str = selector_inputs(sets["train"], sg=SG)
Xun, Sun = selector_inputs(sets["unseen"], sg=SG)
Sun_np = Sun.numpy()
Fch_un = cost_targets(LAB["unseen"], sets["unseen"])[1]
Ffd_un = cost_targets(Crep["unseen"], sets["unseen"])[1]
sel_res = {}
for lab, T in (("T0", "T0"), ("T0p", "T0p"), ("T1", "T1")):
    path = f"{DATA}/models/sel_relabel_{lab}.pt"
    sel = CostRegSelector(512, 4, extra=2 if SG else 0)
    if os.path.exists(path):
        sel.load_state_dict(torch.load(path))
    else:
        J, F = cost_targets(COST[(T, "train")], sets["train"])
        sel = train_costlike(sel, Xtr, Str, J, F, steps=12000)
        torch.save(sel.state_dict(), path)
    sel.eval()
    with torch.no_grad():
        jh, fl = (t.numpy() for t in sel(Xun))
    feas = (fl > 0) & Sun_np
    feas = np.where(feas.any(1, keepdims=True), feas, Sun_np)
    own_F = Fch_un if T == "T1" else Ffd_un
    s = dict(feas_acc_own=float(((fl > 0) == own_F)[Sun_np].mean()),
             feas_acc_chart=float(((fl > 0) == Fch_un)[Sun_np].mean()))
    for r in REGIMES:
        p = probs_from_cost(jh, feas, BETA[T][r])
        s[r] = dict(tv_vs_own=float(tv(p, tvec(T, r)).mean()), tv_vs_T1=float(tv(p, tvec("T1", r)).mean()),
                    mass_on_chart_infeasible=float((p * ~Fch_un).sum(1).mean()))
    sel_res[lab] = s
    print(lab, json.dumps(s), flush=True)
# Day 12b: the ORIGINAL selector already used in earlier results (sel_costreg.pt, trained on T0), scored
# against both the original target (T0) and the revised target (T1). Nothing is retrained here.
orig = f"{DATA}/models/sel_costreg.pt"
if os.path.exists(orig):
    sel = CostRegSelector(512, 4, extra=2 if SG else 0)
    try:
        sel.load_state_dict(torch.load(orig))
    except RuntimeError as e:
        sel = None
        print("orig selector not loadable with plain inputs:", str(e)[:120], flush=True)
    if sel is not None:
        sel.eval()
        with torch.no_grad():
            jh, fl = (t.numpy() for t in sel(Xun))
        feas = (fl > 0) & Sun_np
        feas = np.where(feas.any(1, keepdims=True), feas, Sun_np)
        s = dict(feas_acc_vs_found=float(((fl > 0) == Ffd_un)[Sun_np].mean()),
                 feas_acc_chart=float(((fl > 0) == Fch_un)[Sun_np].mean()))
        for r in REGIMES:
            # original target uses the selector's own beta (T0); revised uses T1's
            s[r] = dict(tv_vs_T0=float(tv(probs_from_cost(jh, feas, BETA["T0"][r]), tvec("T0", r)).mean()),
                        tv_vs_T1=float(tv(probs_from_cost(jh, feas, BETA["T1"][r]), tvec("T1", r)).mean()))
        sel_res["orig_sel_costreg"] = s
        print("orig", json.dumps(s), flush=True)
res["selectors"] = sel_res
res["sec"] = round(time.time() - t0)
json.dump(res, open(f"{DATA}/relabel.json", "w"), indent=1, default=str)
print("done", res["sec"])

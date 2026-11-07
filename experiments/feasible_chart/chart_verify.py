"""Independent check of the chart's soundness claim (topoflow/chart.py).

Claim: for a feasible chart of (env, code, c), every u in R^n decodes to a polyline that (i) keeps Euclidean distance
>= c from every obstacle and (ii) passes each obstacle on the side given by the code.

This checker does NOT use Env.collision_mask, the chart's own wall function or the crossing-word code:
- distance: dense points on every segment (SUB per segment), exact Euclidean distance to circles (|p - c| - r) and to
  boxes (standard box SDF, rounded corners). The chart inflates boxes with square corners, which is stricter.
- side: for every obstacle, every dense point whose x lies inside the obstacle's x-extent must lie above its centre
  (code 0) or below it (code 1); given the clearance check, that is the same as passing on that side.
u is drawn from: N(0, 1), N(0, 9), saturating values (+-50), and per-coordinate mixtures of +-50 and 0.

  TOPO_MAX_OBS=5 python experiments/feasible_chart/chart_verify.py      (DATA=results/d2, SPLITS=train,unseen)
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402

DATA = os.environ.get("DATA", "results/d2")
SPLITS = os.environ.get("SPLITS", "train,unseen").split(",")
N_LAY = int(os.environ.get("N_LAY", "300"))
N = 60
SUB = 64
meta = json.load(open(f"{DATA}/betas.json"))
C_STAR = float(meta.get("_margin", 0.03))  # D1 predates the _margin field (0.03)
with open(f"{DATA}/layouts.pkl", "rb") as f:
    sets = dict(zip(["train", "unseen", "calib"], pickle.load(f)))
rng = np.random.default_rng(0)


def euclid_clearance(env, P):
    """min distance from dense points of polyline P (..., T, 2) to any obstacle."""
    a = np.linspace(0, 1, SUB + 1)[:-1]
    S = (P[..., :-1, None, :] * (1 - a)[:, None] + P[..., 1:, None, :] * a[:, None]).reshape(*P.shape[:-2], -1, 2)
    S = np.concatenate([S, P[..., -1:, :]], -2)
    best = np.full(S.shape[:-1], np.inf)
    for o in env.obstacles:
        dx, dy = S[..., 0] - o.cx, S[..., 1] - o.cy
        if o.kind == "circle":
            d = np.sqrt(dx ** 2 + dy ** 2) - o.r
        else:
            qx, qy = np.abs(dx) - o.hw, np.abs(dy) - o.hh
            d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
        best = np.minimum(best, d)
    return best.min(-1), S


def side_ok(env, S, code):
    ok = np.ones(S.shape[0], bool)
    # with clearance already checked, a point over the obstacle's x-extent is above iff y > cy
    for o, b in zip(env.obstacles, code):
        x0, x1 = o.xrange
        inx = (S[..., 0] >= x0) & (S[..., 0] <= x1)
        good = (S[..., 1] > o.cy) if b == 0 else (S[..., 1] < o.cy)
        ok &= np.all(~inx | good, -1)
    return ok


def u_batch(n):
    return np.concatenate([rng.normal(0, 1, (8, n)), rng.normal(0, 3, (8, n)),
                           np.full((1, n), 50.0), np.full((1, n), -50.0),
                           rng.choice([-50.0, 0.0, 50.0], size=(6, n))])


res = {}
for split in SPLITS:
    L = sets[split][:N_LAY]
    n_chart = n_feas = n_paths = 0
    worst_clear = np.inf
    viol_clear = viol_side = 0
    for lay in L:
        env = lay["env"]
        for code in lay["found"]:
            n_chart += 1
            lo, hi, ok = chart.bounds(env, code, C_STAR, N)
            if not ok:
                continue
            n_feas += 1
            U = u_batch(N)
            P = chart.to_path(env, chart.decode(U, lo, hi), C_STAR)
            cl, S = euclid_clearance(env, P)
            n_paths += len(P)
            worst_clear = min(worst_clear, float(cl.min()))
            viol_clear += int((cl < C_STAR - 1e-9).sum())
            viol_side += int((~side_ok(env, S, code)).sum())
    res[split] = dict(layouts=len(L), charts=n_chart, feasible=n_feas, feasible_frac=n_feas / n_chart,
                      decoded_paths=n_paths, clearance_violations=viol_clear, side_violations=viol_side,
                      min_clearance=worst_clear, c_star=C_STAR, dense_points_per_segment=SUB)
    print(split, json.dumps(res[split]), flush=True)
json.dump(res, open(f"{DATA}/chart_verify.json", "w"), indent=1)
print("done")

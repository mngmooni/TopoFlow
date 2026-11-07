"""Exploratory (not pre-registered): does the data itself get longer on the refined grid?
Reference paths resampled at base vs base+40 grid (same rng as vr_eval), length ratio."""
import itertools, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart, vr
DATA = "results/d4"; C = float(json.load(open(f"{DATA}/betas.json"))["_margin"])
lays = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:300]
rng = np.random.default_rng(123); r = []
L = lambda X, Y: np.sum(np.hypot(np.diff(X), np.diff(Y)))
for lay in lays:
    env = lay["env"]
    for code in itertools.product([0, 1], repeat=env.m):
        base = vr.refined_grid(env, code, C)
        if base is None: continue
        refined = np.unique(np.concatenate([base, rng.uniform(env.start[0], env.goal[0], min(40, vr.NMAX - len(base)))]))
        if not chart.bounds(env, code, C, 60)[2] or code not in lay["found"]: continue
        for p in lay["found"][code]:
            out = []
            for xs in (base, refined):
                X = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
                out.append(L(X, np.interp(X, p[:, 0], p[:, 1])))
            r.append(out[1] / out[0] - 1)
print(json.dumps(dict(n=len(r), ref_len_rel_diff_mean=float(np.mean(r)), p90=float(np.percentile(r, 90)))))

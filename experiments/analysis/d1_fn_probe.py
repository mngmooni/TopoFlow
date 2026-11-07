"""Planner false-negative probe on D1 unseen layouts.

For every structurally valid code that the reference planner marked not_found, ask P1 for N samples
conditioned on that code and check whether any sample is collision-free AND has exactly that crossing word.
Such a sample is a constructive witness that the code is feasible (at zero clearance; the planner used
margin 0.03, so a witness may still be infeasible at the planner's margin).
Control: the same procedure on found codes (P1's hit rate when the code is known feasible).

  TOPO_MAX_OBS=5 python experiments/analysis/d1_fn_probe.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import MAX_OBS, classify  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import code_index, generate, index_code, structural_mask  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "results/d1")
N = 16
KS = [int(k) for k in os.environ.get("KS", "5,20").split(",")]
MARGINS = [0.0, 0.01, 0.02, 0.03]
with open(f"{DATA}/layouts.pkl", "rb") as f:
    unseen = pickle.load(f)[1]
model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True)
model.load_state_dict(torch.load(f"{DATA}/models/P1_s0.pt"))
model.eval()

pairs = []  # (layout idx, code idx, found?)
for i, lay in enumerate(unseen):
    found = {code_index(c) for c in lay["found"]}
    for k in np.where(structural_mask(lay["env"].m))[0]:
        pairs.append((i, int(k), int(k) in found))
lays = [unseen[i] for i, _, _ in pairs]
codes = np.array([[k] * N for _, k, _ in pairs])
x0 = torch.randn(len(pairs) * N, 60, generator=torch.Generator().manual_seed(4242))
out = {}
for K in KS:
    G = generate(model, lays, codes, K, x0)  # (P, N, 30, 2)
    rows = []
    for p, (i, k, fd) in enumerate(pairs):
        env = unseen[i]["env"]
        P = np.empty((N, 32, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = G[p]
        coll, deep, words = classify(env, P)
        want = index_code(k, env.m)
        ok = [(not c) and (w == want) for c, w in zip(coll, words)]
        r = dict(m=env.m, found=fd, witness=any(ok), hit_rate=float(np.mean(ok)), coll=float(np.mean(coll)))
        for mg in MARGINS[1:]:
            cm = env.collision_mask(P, margin=mg)
            r[f"witness_{mg}"] = any((not c) and (w == want) for c, w in zip(cm, words))
        rows.append(r)
    res = {}
    for fd in (True, False):
        for m in (3, 4, 5, "all"):
            R = [r for r in rows if r["found"] == fd and (m == "all" or r["m"] == m)]
            if not R:
                continue
            res[f"{'found' if fd else 'notfound'}_m{m}"] = dict(
                n_codes=len(R), witness_frac=float(np.mean([r["witness"] for r in R])),
                mean_hit=float(np.mean([r["hit_rate"] for r in R])), coll=float(np.mean([r["coll"] for r in R])),
                **{f"witness_m{mg}": float(np.mean([r[f"witness_{mg}"] for r in R])) for mg in MARGINS[1:]})
    out[K] = res
    print(K, json.dumps(res, indent=1), flush=True)
json.dump(out, open(f"{DATA}/fn_probe_margins.json", "w"), indent=1)
print("done")

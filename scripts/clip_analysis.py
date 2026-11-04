"""Clipping / distortion analysis of the v2 chart on D2 (no training).

(a) where training paths are clipped: grid points outside [lo, hi], amount, binding wall type (circle / box / outer)
(b) resolution sweep: clip rate for n = 30, 60, 120, 240
(c) v2 samples (unseen, medium, oracle codes, NFE 20): RMS to the nearest original vs nearest clipped reference path
(d) per (layout, code): clip rate vs v2 diversity / data diversity (Spearman)

  TOPO_MAX_OBS=5 python scripts/clip_analysis.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow import chart  # noqa: E402
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.tdfm import boltzmann, index_code, sample_codes, target_vector  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample as v2_sample  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d2")
meta = json.load(open(f"{DATA}/betas.json"))
C_STAR = float(meta["_margin"])
with open(f"{DATA}/layouts.pkl", "rb") as f:
    train, unseen = pickle.load(f)[:2]
with open(f"{DATA}/costs.pkl", "rb") as f:
    C = pickle.load(f)
out = {}


def binding(env, code, n, k, below):
    """Which wall binds grid point k on the violated side: 'circle', 'box' or 'outer'."""
    xs = np.concatenate([[env.start[0]], chart.grid(env, n, C_STAR), [env.goal[0]]])
    best, kind = (chart.Y_MIN if below else chart.Y_MAX), "outer"
    for seg in (k, k + 1):  # the two segments touching interior point k (index shift: interior k -> xs[k+1])
        for o, b in zip(env.obstacles, code):
            if below and b == 0:
                w = chart._wall(o, xs[seg], xs[seg + 1], C_STAR, top=True)
                if w is not None and w > best:
                    best, kind = w, o.kind
            if not below and b == 1:
                w = chart._wall(o, xs[seg], xs[seg + 1], C_STAR, top=False)
                if w is not None and w < best:
                    best, kind = w, o.kind
    return kind


# (a) + (b)
for n in (30, 60, 120, 240):
    paths = clipped = pts = 0
    amounts, kinds = [], {"circle": 0, "box": 0, "outer": 0}
    for lay in train[:600]:
        env = lay["env"]
        for code, P in lay["found"].items():
            lo, hi, ok = chart.bounds(env, code, C_STAR, n)
            if not ok:
                continue
            for p in P:
                y = chart.to_grid(env, p, n, C_STAR)
                below, above = y < lo, y > hi
                paths += 1
                pts += n
                if below.any() or above.any():
                    clipped += 1
                    amounts.append(max(np.max(lo - y, initial=0), np.max(y - hi, initial=0)))
                    if n == 60:
                        for k in np.where(below)[0]:
                            kinds[binding(env, code, n, k, True)] += 1
                        for k in np.where(above)[0]:
                            kinds[binding(env, code, n, k, False)] += 1
    rec = dict(n=n, clip_rate=clipped / paths, clip_amount_median=float(np.median(amounts)) if amounts else 0.0,
               clip_amount_p90=float(np.percentile(amounts, 90)) if amounts else 0.0)
    if n == 60:
        tot = sum(kinds.values())
        rec["binding_share"] = {k: v / tot for k, v in kinds.items()}
        nc = sum(o.kind == "circle" for l in train[:600] for o in l["env"].obstacles)
        no = sum(len(l["env"].obstacles) for l in train[:600])
        rec["circle_share_of_obstacles"] = nc / no
    out[f"res_n{n}"] = rec
    print(json.dumps(rec), flush=True)

# (c) + (d)
N_PER = 32
L = len(unseen)
targets = [boltzmann(c, meta["medium"]["beta"]) for c in C["unseen"]]
codes = sample_codes(np.array([target_vector(t, None) for t in targets]), N_PER, seed=900)
x0 = torch.randn(L * N_PER, 60, generator=torch.Generator().manual_seed(777))
req = [(i, index_code(int(codes[i, j]), unseen[i]["env"].m)) for i in range(L) for j in range(N_PER)]
m = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
m.load_state_dict(torch.load(f"{DATA}/models/V2_s0.pt"))
m.eval()
cache = ChartCache(unseen, C_STAR, 60)
P = v2_sample(m, cache, req, 20, x0)
d_orig, d_clip, rows = [], [], []
for i, lay in enumerate(unseen):
    env = lay["env"]
    for code in set(c for (ii, c) in req[i * N_PER:(i + 1) * N_PER]):
        idx = [j for j in range(N_PER) if req[i * N_PER + j][1] == code]
        lo, hi, _, _ = cache.get(i, code)
        refs = lay["found"][code]
        R = np.array([chart.to_grid(env, r, 60, C_STAR) for r in refs])
        Rc = np.clip(R, lo, hi)
        Y = P[[i * N_PER + j for j in idx], 1:-1, 1]
        for y in Y:
            d_orig.append(np.sqrt(((R - y) ** 2).mean(1)).min())
            d_clip.append(np.sqrt(((Rc - y) ** 2).mean(1)).min())
        clip_rate = float(np.mean([np.any((r < lo) | (r > hi)) for r in R]))
        if len(Y) >= 2 and len(R) >= 2:
            dv = np.sqrt(((Y[:, None] - Y[None]) ** 2).mean(-1))[np.triu_indices(len(Y), 1)].mean()
            dr = np.sqrt(((R[:, None] - R[None]) ** 2).mean(-1))[np.triu_indices(len(R), 1)].mean()
            if dr > 0:
                rows.append((clip_rate, dv / dr))
rows = np.array(rows)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


out["v2_nearest"] = dict(rms_to_original=float(np.mean(d_orig)), rms_to_clipped=float(np.mean(d_clip)),
                         frac_closer_to_clipped=float(np.mean(np.array(d_clip) < np.array(d_orig))))
out["div_vs_clip"] = dict(n_groups=len(rows), spearman=spearman(rows[:, 0], rows[:, 1]),
                          div_ratio_clip0=float(rows[rows[:, 0] == 0, 1].mean()),
                          div_ratio_clipped=float(rows[rows[:, 0] > 0, 1].mean()) if (rows[:, 0] > 0).any() else None,
                          frac_groups_with_clipping=float((rows[:, 0] > 0).mean()))
print(json.dumps(out["v2_nearest"]), json.dumps(out["div_vs_clip"]), flush=True)
json.dump(out, open(f"{DATA}/clip_analysis.json", "w"), indent=1)
print("done")

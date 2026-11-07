"""research phase step 4: is v2 safe but meaningless out of distribution? (locked: the archived prediction record)

  TOPO_MAX_OBS=5 python experiments/analysis/ood_quality.py            (FAMS=D2,H1,H2,H4; D4 when results/d4 exists)
Writes results/hard/quality.json.
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample  # noqa: E402

from scipy.optimize import minimize  # noqa: E402

torch.set_num_threads(1)
C, N, N_EVAL, PER, K = 0.01, 60, int(os.environ.get("N_EVAL", "150")), 8, int(os.environ.get("K", "20"))
FAMS = os.environ.get("FAMS", "D2,H1,H2,H4" + (",D4" if os.path.exists("results/d4/layouts.pkl") else "")).split(",")
XU = np.linspace(-0.9, 0.9, 62)[1:-1]
# research phase (locked: the archived prediction record): MODELS=d2,d4 scores several v2 models on the same
# classes/noise; OUT names the json. Also scores the chart-shortest path itself (structure baseline).
MODELS = os.environ.get("MODELS", "d2").split(",")
OUT = os.environ.get("OUT", "results/hard/quality.json")


def shortest(env, xs, lo, hi):
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
    r = minimize(f, y0, jac=True, method="L-BFGS-B", bounds=list(zip(m - w, m + w)))
    return float(r.fun), np.stack([X, np.concatenate([[sy], r.x, [gy]])], 1)


def length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def min_clear(env, P, sub=16):
    a = np.linspace(0, 1, sub + 1)[:-1]
    S = (P[:-1, None] * (1 - a)[:, None] + P[1:, None] * a[:, None]).reshape(-1, 2)
    best = np.inf
    for o in env.obstacles:
        dx, dy = S[:, 0] - o.cx, S[:, 1] - o.cy
        if o.kind == "circle":
            d = np.sqrt(dx ** 2 + dy ** 2) - o.r
        else:
            qx, qy = np.abs(dx) - o.hw, np.abs(dy) - o.hh
            d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
        best = min(best, d.min())
    return best


def metrics(env, paths, lo, hi, xs, short):
    out = []
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - chart.EPS)
    for P in paths:
        y = np.interp(xs, P[:, 0], P[:, 1])
        sat = float(np.mean(np.abs((y - m) / w) > 0.99))
        yu = np.interp(XU, P[:, 0], P[:, 1])
        mc = min_clear(env, P)
        out.append(dict(len_ratio=length(P) / short, sat=sat, rough=float(np.mean(np.diff(yu, 2) ** 2)),
                        hug=float(mc < C + 0.005), clear=float(mc)))
    return out


KEYS = ("len_ratio", "sat", "rough", "hug", "clear")
nets = {}
for mname in MODELS:
    net = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
    net.load_state_dict(torch.load(f"results/{mname}/models/V2_s0.pt"))
    nets[mname] = net.eval()
agg = lambda rs, k: (float(np.median([r[k] for r in rs])) if k == "len_ratio" else float(np.mean([r[k] for r in rs])))  # noqa: E731


def diversity(paths):
    Y = np.stack([np.interp(XU, p[:, 0], p[:, 1]) for p in paths])
    return float(np.mean([np.sqrt(np.mean((Y[a] - Y[b]) ** 2)) for a, b in itertools.combinations(range(len(Y)), 2)]))


res = {}
for fam in FAMS:
    if fam in ("D2", "D4"):
        lays = pickle.load(open(f"results/{fam.lower()}/layouts.pkl", "rb"))[1][:N_EVAL]
    else:
        lays = pickle.load(open(f"results/hard/{fam}_layouts.pkl", "rb"))[:N_EVAL]
    cache = ChartCache(lays, C, N)
    req = []
    for i, lay in enumerate(lays):
        for code in itertools.product([0, 1], repeat=lay["env"].m):
            if cache.get(i, code)[3]:
                req += [(i, code)] * PER
    noise = torch.randn(len(req), N, generator=torch.Generator().manual_seed(11))
    res[fam] = {}
    shorts = {}
    for mname, net in nets.items():
        pre = "v2" if mname == "d2" else f"v2{mname}"
        G = sample(net, cache, req, K, noise)
        rows, ref_rows, sp_rows, nn_rms, divs, ref_divs = [], [], [], [], [], []
        groups = {}
        for j, key in enumerate(req):
            groups.setdefault(key, []).append(G[j])
        for (i, code), paths in groups.items():
            env = lays[i]["env"]
            lo, hi, xs, _ = cache.get(i, code)
            if (i, code) not in shorts:
                shorts[(i, code)] = shortest(env, xs, lo, hi)
            short, SP = shorts[(i, code)]
            rows += metrics(env, paths, lo, hi, xs, short)
            sp_rows += metrics(env, [SP], lo, hi, xs, short)
            divs.append(diversity(paths))
            refs = lays[i].get("found", {}).get(code)
            if refs is not None and len(refs):
                ref_rows += metrics(env, refs, lo, hi, xs, short)
                if len(refs) > 1:
                    ref_divs.append(diversity(refs))
                R = np.stack([np.interp(XU, p[:, 0], p[:, 1]) for p in refs])
                for P in paths:
                    yu = np.interp(XU, P[:, 0], P[:, 1])
                    nn_rms.append(float(np.sqrt(((R - yu) ** 2).mean(1)).min()))
        r = res[fam]
        r["n"] = len(rows)
        r.update({f"{pre}_{k}": agg(rows, k) for k in KEYS})
        r[f"{pre}_len_ratio_p90"] = float(np.percentile([x["len_ratio"] for x in rows], 90))
        r[f"{pre}_div"] = float(np.mean(divs))
        r.update({f"short_{k}": agg(sp_rows, k) for k in ("sat", "rough", "hug", "clear")})
        if ref_rows:
            r.update({f"ref_{k}": agg(ref_rows, k) for k in KEYS})
            r["ref_div"] = float(np.mean(ref_divs)) if ref_divs else None
            r[f"{pre}_nn_ref_rms"] = float(np.mean(nn_rms))
    print(fam, json.dumps(res[fam]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("done")

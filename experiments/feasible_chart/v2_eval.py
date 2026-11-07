"""Compare TDFM v2 (chart flow) with P1 (raw / full-path projection / clearance loss) on D2.

Same layouts, codes (seed 900) and noise (seed 777) for every method. Diversity is computed on a common
representation for all methods: y-values of each polyline on a uniform 60-point x-grid; mean pairwise RMS between
valid samples of the same code in a layout. Reference paths, same metric, give the data value.

  TOPO_MAX_OBS=5 python experiments/feasible_chart/v2_eval.py           (METHODS=p1,p1proj,v2,p1clr)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.taxonomy import classify_samples  # noqa: E402
from topoflow.tdfm import boltzmann, generate, index_code, sample_codes, target_vector  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample as v2_sample  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
SG = os.environ.get("SG", "0") == "1"  # D3: models take (start_y, goal_y)
MSEED = int(os.environ.get("MSEED", "0"))  # which training seed of P1 / V2 to evaluate
DATA = os.environ.get("DATA", "results/d2")
METHODS = os.environ.get("METHODS", "p1,p1proj,v2").split(",")
KS = [int(k) for k in os.environ.get("KS", "2,5,20").split(",")]
N_PER = 32
meta = json.load(open(f"{DATA}/betas.json"))
CSTAR = float(meta["_margin"])
with open(f"{DATA}/layouts.pkl", "rb") as f:
    unseen = pickle.load(f)[1]
with open(f"{DATA}/costs.pkl", "rb") as f:
    C = pickle.load(f)
L = len(unseen)
targets = [boltzmann(c, meta["medium"]["beta"]) for c in C["unseen"]]
codes = sample_codes(np.array([target_vector(t, None) for t in targets]), N_PER, seed=900)
x0 = torch.randn(L * N_PER, 60, generator=torch.Generator().manual_seed(777))
req = [(i, index_code(int(codes[i, j]), unseen[i]["env"].m)) for i in range(L) for j in range(N_PER)]
XG = np.linspace(-0.9, 0.9, 62)[1:-1]

src = open(os.path.join(os.path.dirname(__file__), "v_project.py")).read()
ns = {"np": np}
exec(src[src.index("def depth_and_normal"):src.index("for stem in MODELS:")], ns)
project_full = ns["project_full"]


def p1_paths(stem, K, proj=False):
    m = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True, sg_input=SG)
    m.load_state_dict(torch.load(f"{DATA}/models/{stem}.pt"))
    m.eval()
    t0 = time.time()
    G = generate(m, unseen, codes, K, x0, sg=SG)
    if proj:
        G = np.stack([project_full(l["env"], G[i], CSTAR + 0.003) for i, l in enumerate(unseen)])
    dt = time.time() - t0  # includes projection time
    out = []
    for i, l in enumerate(unseen):
        P = np.empty((N_PER, 32, 2))
        P[:, 0], P[:, -1] = l["env"].start, l["env"].goal
        P[:, 1:-1] = G[i]
        out.append(P)
    return out, dt


def v2_paths(K):
    m = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG)
    m.load_state_dict(torch.load(f"{DATA}/models/V2_s{MSEED}.pt"))
    m.eval()
    cache = ChartCache(unseen, CSTAR, 60, sg=SG)
    t0 = time.time()  # chart construction happens lazily inside, so it is part of the generation cost
    P = v2_sample(m, cache, req, K, x0)
    dt = time.time() - t0
    return [P[i * N_PER:(i + 1) * N_PER] for i in range(L)], dt


def evaluate(paths):
    cnt = {k: 0 for k in "UMTNV"}
    tvs, divs, lens = [], [], []
    for i, lay in enumerate(unseen):
        env, P = lay["env"], paths[i]
        rq = [index_code(int(k), env.m) for k in codes[i]]
        cats, words = classify_samples(env, P, rq, set(lay["found"]), CSTAR)
        for c in cats:
            cnt[c] += 1
        tgt = targets[i]
        hist = {c: 0 for c in tgt}
        for c, w in zip(cats, words):
            if c == "V" and w in hist:
                hist[w] += 1
        tvs.append(0.5 * (sum(abs(hist[c] / N_PER - tgt[c]) for c in tgt) + (N_PER - sum(hist.values())) / N_PER))
        Y = np.array([np.interp(XG, p[:, 0], p[:, 1]) for p in P])
        for code in set(rq):
            idx = [j for j in range(N_PER) if rq[j] == code and cats[j] == "V"]
            if len(idx) >= 2:
                d = np.sqrt(((Y[idx][:, None] - Y[idx][None]) ** 2).mean(-1))
                divs.append(d[np.triu_indices(len(idx), 1)].mean())
            ref = lay["found"].get(code)
            if ref is not None:
                rl = min(np.linalg.norm(np.diff(r, axis=0), axis=-1).sum() for r in ref)
                lens += [np.linalg.norm(np.diff(P[j], axis=0), axis=-1).sum() / rl for j in idx]
    tot = L * N_PER
    return dict(**{k: cnt[k] / tot for k in cnt}, tv=float(np.mean(tvs)), diversity=float(np.mean(divs)),
                length_ratio=float(np.mean(lens)))


dd = []
for lay in unseen:
    for code, ref in lay["found"].items():
        Y = np.array([np.interp(XG, r[:, 0], r[:, 1]) for r in ref])
        if len(Y) >= 2:
            d = np.sqrt(((Y[:, None] - Y[None]) ** 2).mean(-1))
            dd.append(d[np.triu_indices(len(Y), 1)].mean())
print(json.dumps(dict(data_diversity=float(np.mean(dd)))), flush=True)

for meth in METHODS:
    for K in KS:
        if meth == "p1":
            paths, dt = p1_paths(f"P1_s{MSEED}", K)
        elif meth == "p1proj":
            paths, dt = p1_paths(f"P1_s{MSEED}", K, proj=True)
        elif meth == "p1clr":
            paths, dt = p1_paths("P1_s0_clr100", K)
        elif meth == "v2":
            paths, dt = v2_paths(K)
        rec = dict(method=meth, seed=MSEED, K=K, gen_sec=round(dt, 2), **evaluate(paths))
        print(json.dumps(rec), flush=True)
        with open(f"{DATA}/v2_eval.jsonl", "a") as fo:
            fo.write(json.dumps(rec) + "\n")
print("done")

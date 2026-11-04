"""Day 11 step 2: clearance fixes compared on D2 (medium, oracle codes, same x0/codes).

Variants
  raw        : Euler NFE K
  trunc      : Euler with K steps up to t = T_JUMP, then jump to x1_hat (NFE K + 1)
  fullproj   : raw + whole-path projection (scripts/v_project.py: project_full)
  model stems: P1_s0, P1_s0_clr100, ... (any D2 P1 checkpoint)
Metrics: taxonomy at c*, joint TV, within-class diversity (mean pairwise RMS between samples of the same code in a
layout, over samples with clearance >= c*), length ratio vs the nearest reference path of that code.

  TOPO_MAX_OBS=5 MODELS=P1_s0,P1_s0_clr100 python scripts/s2_compare.py
"""
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.e2 import MAX_OBS, env_features, pad_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.taxonomy import classify_samples, tally  # noqa: E402
from topoflow.tdfm import boltzmann, index_code, sample_codes, target_vector  # noqa: E402

os.environ.setdefault("MODELS", "P1_s0")
spec = importlib.util.spec_from_file_location("vp", os.path.join(os.path.dirname(__file__), "v_project.py"))
torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d2")
MODELS = os.environ["MODELS"].split(",")
VARIANTS = os.environ.get("VARIANTS", "raw,trunc,fullproj").split(",")
KS = [int(k) for k in os.environ.get("KS", "5,20").split(",")]
T_JUMP = float(os.environ.get("T_JUMP", "0.75"))
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
conds = torch.tensor(np.array([np.concatenate([*env_features(unseen[i]["env"]),
                                               pad_code(index_code(int(codes[i, j]), unseen[i]["env"].m))])
                               for i in range(L) for j in range(N_PER)]), dtype=torch.float32)

# project_full from v_project.py without running its main loop
src = open(os.path.join(os.path.dirname(__file__), "v_project.py")).read()
ns = {"np": np}
exec(src[src.index("def depth_and_normal"):src.index("for stem in MODELS:")], ns)
project_full = ns["project_full"]


@torch.no_grad()
def sample(model, K, jump=None):
    outs = []
    for s in range(0, len(x0), 4096):
        x, c = x0[s:s + 4096].clone(), conds[s:s + 4096]
        if jump is None:
            for k in range(K):
                x = x + model(x, torch.full((len(x),), k / K), c) / K
        else:
            h = jump / K
            for k in range(K):
                x = x + h * model(x, torch.full((len(x),), k * h), c)
            x = x + (1 - jump) * model(x, torch.full((len(x),), jump), c)
        outs.append(x)
    return torch.cat(outs).numpy().reshape(L, N_PER, 30, 2)


def extras(G):
    divs, lens = [], []
    for i, lay in enumerate(unseen):
        env = lay["env"]
        P = np.empty((N_PER, 32, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = G[i]
        req = [index_code(int(k), env.m) for k in codes[i]]
        cats, _ = classify_samples(env, P, req, set(lay["found"]), CSTAR)
        ok = np.array([c == "V" for c in cats])
        for code in set(req):
            idx = [j for j in range(N_PER) if req[j] == code and ok[j]]
            if len(idx) >= 2:
                X = G[i, idx].reshape(len(idx), -1)
                d = np.sqrt(((X[:, None] - X[None]) ** 2).reshape(len(idx), len(idx), -1, 2).sum(-1).mean(-1))
                divs.append(d[np.triu_indices(len(idx), 1)].mean())
            ref = lay["found"].get(code)
            if ref is not None:
                rl = min(np.linalg.norm(np.diff(r, axis=0), axis=-1).sum() for r in ref)
                for j in idx:
                    lens.append(np.linalg.norm(np.diff(P[j], axis=0), axis=-1).sum() / rl)
    return float(np.mean(divs)), float(np.mean(lens))


# data diversity, same definition, for reference
dd = []
for lay in unseen:
    for code, ref in lay["found"].items():
        X = np.array([r[1:-1] for r in ref])
        if len(X) >= 2:
            d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1).mean(-1))
            dd.append(d[np.triu_indices(len(X), 1)].mean())
print(json.dumps(dict(data_diversity=float(np.mean(dd)))), flush=True)

for stem in MODELS:
    model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True)
    model.load_state_dict(torch.load(f"{DATA}/models/{stem}.pt"))
    model.eval()
    for K in KS:
        G = sample(model, K)
        for var in VARIANTS:
            if var == "raw":
                H, nfe = G, K
            elif var == "trunc":
                H, nfe = sample(model, K, jump=T_JUMP), K + 1
            elif var == "fullproj":
                H, nfe = np.stack([project_full(l["env"], G[i], CSTAR + 0.003) for i, l in enumerate(unseen)]), K
            r = tally(unseen, H, codes, CSTAR, targets, index_code)
            div, lr = extras(H)
            rec = dict(model=stem, variant=var, K=K, nfe=nfe, **{k: r[k] for k in ("U", "M", "T", "N", "V", "tv_V")},
                       diversity=div, length_ratio=lr)
            print(json.dumps(rec), flush=True)
            with open(f"{DATA}/s2_compare.jsonl", "a") as fo:
                fo.write(json.dumps(rec) + "\n")
print("done")

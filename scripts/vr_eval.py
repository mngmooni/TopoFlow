"""Evaluate v2-D4 and v2-VR on D4 unseen. Locked: outputs/predictions_day12_vr.md

  TOPO_MAX_OBS=5 DATA=outputs/d4 python scripts/vr_eval.py   -> {DATA}/vr_eval.json
"""
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow import chart, vr  # noqa: E402
from topoflow.e2 import MAX_OBS, word_to_code  # noqa: E402
from topoflow.topology import crossing_word  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample as sample_v2  # noqa: E402

torch.set_num_threads(1)
DATA = os.environ.get("DATA", "outputs/d4")
N_EVAL = int(os.environ.get("N_EVAL", "300"))
PER = 8
C = float(json.load(open(f"{DATA}/betas.json"))["_margin"])
lays = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:N_EVAL]
XU = np.linspace(-0.9, 0.9, 62)[1:-1]
rng = np.random.default_rng(123)

# classes: every chart-certified code (after bisection)
classes = []
for i, lay in enumerate(lays):
    env = lay["env"]
    for code in itertools.product([0, 1], repeat=env.m):
        base = vr.refined_grid(env, code, C)
        if base is None:
            continue
        ok60 = chart.bounds(env, code, C, 60)[2]
        # deviation (Day 12b, found at eval): bisection can overshoot NMAX-40, so cap the extra points at NMAX - len(base)
        refined = np.unique(np.concatenate([base, rng.uniform(env.start[0], env.goal[0], min(40, vr.NMAX - len(base)))]))
        classes.append(dict(i=i, code=code, base=base, refined=refined, ok60=ok60, found=code in lay["found"]))
print(json.dumps(dict(capped=sum(len(c["base"]) > vr.NMAX - 40 for c in classes), base_max=max(len(c["base"]) for c in classes),
                      classes=len(classes), ok60=sum(c["ok60"] for c in classes),
                      refine_only=sum(not c["ok60"] for c in classes),
                      refine_only_found=sum((not c["ok60"]) and c["found"] for c in classes))), flush=True)


def indep_ok(env, P, code, sub=64):
    a = np.linspace(0, 1, sub + 1)[:-1]
    S = (P[:-1, None] * (1 - a)[:, None] + P[1:, None] * a[:, None]).reshape(-1, 2)
    for o, b in zip(env.obstacles, code):
        dx, dy = S[:, 0] - o.cx, S[:, 1] - o.cy
        if o.kind == "circle":
            d = np.sqrt(dx ** 2 + dy ** 2) - o.r
        else:
            qx, qy = np.abs(dx) - o.hw, np.abs(dy) - o.hh
            d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
        if d.min() < C - 1e-9:
            return False
        inx = (S[:, 0] >= o.xrange[0]) & (S[:, 0] <= o.xrange[1])
        if np.any(inx & ((S[:, 1] <= o.cy) if b == 0 else (S[:, 1] >= o.cy))):
            return False
    return True


def score(groups):
    """groups: list of (cls, paths). Returns aggregate metrics + per-class diversity/length dict."""
    n = U = M = T = viol = 0
    divs, lens, nn = {}, {}, []
    for cls, paths in groups:
        env = lays[cls["i"]]["env"]
        P = np.stack(paths)
        u0 = env.collision_mask(P)
        uc = env.collision_mask(P, margin=C)
        wrong = np.array([word_to_code(crossing_word(env, p), env.m) != cls["code"] for p in P])
        n += len(P)
        U += int(u0.sum())
        M += int((uc & ~u0).sum())
        T += int((wrong & ~uc).sum())
        viol += sum(not indep_ok(env, p, cls["code"]) for p in P)
        Y = np.stack([np.interp(XU, p[:, 0], p[:, 1]) for p in P])
        key = (cls["i"], cls["code"])
        divs[key] = float(np.mean([np.sqrt(np.mean((Y[a] - Y[b]) ** 2)) for a, b in itertools.combinations(range(len(Y)), 2)]))
        lens[key] = float(np.mean([np.linalg.norm(np.diff(p, axis=0), axis=1).sum() for p in P]))
        if cls["found"]:
            R = np.stack([np.interp(XU, q[:, 0], q[:, 1]) for q in lays[cls["i"]]["found"][cls["code"]]])
            nn += [float(np.sqrt(((R - y) ** 2).mean(1)).min()) for y in Y]
    return dict(n=n, U=U / n, M=M / n, T=T / n, indep_viol=viol / n, diversity=float(np.mean(list(divs.values()))),
                length=float(np.mean(list(lens.values()))), nn_ref=float(np.mean(nn)) if nn else None), divs, lens


ONLY = os.environ.get("ONLY", "v2,VRou,VRiid").split(",")   # run models in separate launches (2 h cap)
OUT = f"{DATA}/vr_eval.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
models = {}
if os.path.exists(f"{DATA}/models/V2_s0.pt"):
    m = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
    m.load_state_dict(torch.load(f"{DATA}/models/V2_s0.pt"))
    models["v2"] = m.eval()
for pk in ("ou", "iid"):
    if os.path.exists(f"{DATA}/models/VR{pk}_s0.pt"):
        m = vr.PerceiverVR()
        m.load_state_dict(torch.load(f"{DATA}/models/VR{pk}_s0.pt"))
        models[f"VR{pk}"] = m.eval()

models = {k: v for k, v in models.items() if k in ONLY}
for name, model in models.items():
    for K in (5, 20):
        t0 = time.time()
        if name == "v2":
            cs = [c for c in classes if c["ok60"]]
            req = [(c["i"], c["code"]) for c in cs for _ in range(PER)]
            G = sample_v2(model, ChartCache(lays, C, 60), req, K,
                          torch.randn(len(req), 60, generator=torch.Generator().manual_seed(7)))
            agg, _, _ = score([(c, list(G[k * PER:(k + 1) * PER])) for k, c in enumerate(cs)])
            res[f"v2_K{K}_ok60"] = dict(**agg, sec=round(time.time() - t0))
            print(name, K, "ok60", json.dumps(res[f"v2_K{K}_ok60"]), flush=True)
            json.dump(res, open(OUT, "w"), indent=1)
            continue
        pk = name[2:]
        out = {}
        for grid in ("base", "refined"):
            rows = [(lays[c["i"]]["env"], c["code"], C, c[grid]) for c in classes for _ in range(PER)]
            G = vr.sample(model, rows, K, prior_kind=pk, seed=7)
            groups = [(c, G[k * PER:(k + 1) * PER]) for k, c in enumerate(classes)]
            for subset, sel in (("ok60", lambda c: c["ok60"]), ("refonly", lambda c: not c["ok60"])):
                agg, divs, lens = score([(c, g) for c, g in groups if sel(c)])
                res[f"{name}_K{K}_{grid}_{subset}"] = agg
                out[(grid, subset)] = (divs, lens)
                print(name, K, grid, subset, json.dumps(agg), flush=True)
        # resolution consistency on ok60 classes
        dB, lB = out[("base", "ok60")]
        dR, lR = out[("refined", "ok60")]
        keys = list(dB)
        res[f"{name}_K{K}_consistency"] = dict(
            div_ratio=float(np.mean([dR[k] for k in keys]) / np.mean([dB[k] for k in keys])),
            len_rel_diff=float(np.mean([(lR[k] - lB[k]) / lB[k] for k in keys])),
            sec=round(time.time() - t0))
        print(name, K, "consistency", json.dumps(res[f"{name}_K{K}_consistency"]), flush=True)
        json.dump(res, open(OUT, "w"), indent=1)
json.dump(res, open(OUT, "w"), indent=1)
print("done")

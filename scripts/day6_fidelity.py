"""Day 6b: is P1's near-zero collision real, or is it just generating conservative detours?

For each (layout, code) item, compare generated paths with that item's reference paths
(the generator's own validated samples) on:
  clearance   : min distance to any obstacle surface (relative to obstacle size, like `deep`)
  length ratio: generated length / mean reference length
  nn_rms      : RMS waypoint distance to the nearest reference path
Reference-vs-reference nn_rms gives the within-class spread scale.

Run from the repo root:  python scripts/day6_fidelity.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import VelocityMLP  # noqa: E402
from topoflow.e2 import HELD_OUT, env_features, pad_code  # noqa: E402
from topoflow.e2eval import build_items  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402

torch.set_num_threads(1)
T, K, PER, SEED = 32, 5, 8, 0
with open("outputs/day5/layouts.pkl", "rb") as f:
    _, unseen = pickle.load(f)
held = [l for l in unseen if any(c in HELD_OUT for c in l["found"])]
splits = {"unseen_pos": (unseen, build_items(unseen, "trainable")[:400]),
          "unseen_combo": (held, build_items(held, "held_out"))}


@torch.no_grad()
def generate(model, lays, items):
    cond = torch.tensor(np.repeat(np.array([np.concatenate([*env_features(lays[li]["env"]), pad_code(c)])
                                            for li, c in items]), PER, 0), dtype=torch.float32)
    x = torch.randn(len(cond), 60, generator=torch.Generator().manual_seed(SEED))
    for k in range(K):
        x = x + model(x, torch.full((len(x),), k / K), cond) / K
    return x.numpy().reshape(len(items), PER, T - 2, 2)


def full(env, s):
    P = np.empty((len(s), T, 2))
    P[:, 0], P[:, -1] = env.start, env.goal
    P[:, 1:-1] = s
    return P


def length(P):
    return np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)


def spread(A):
    """Mean pairwise RMS waypoint distance within a set of paths (diversity)."""
    d = np.sqrt(((A[:, None] - A[None]) ** 2).sum(-1).mean(-1))
    n = len(A)
    return d.sum() / (n * (n - 1))


def nn_rms(A, B):
    d = ((A[:, None] - B[None]) ** 2).sum(-1).mean(-1)  # (nA, nB) mean squared waypoint dist
    return np.sqrt(d.min(1))


models = {}
for name, path, ctor in [
    ("B1", "outputs/day5/models/B1_s0.pt", lambda: VelocityMLP(60, cond_dim=21, hidden=512, depth=4)),
    ("B1full", "outputs/day5/models/B1full_s0.pt", lambda: VelocityMLP(60, cond_dim=21, hidden=512, depth=4)),
    ("P1", "outputs/day6/models/P1_s0.pt", lambda: ObstacleAttnField(30, 3, d=128, depth=4, chunk=3)),
    ("P1_s1", "outputs/day6/models/P1_s1.pt", lambda: ObstacleAttnField(30, 3, d=128, depth=4, chunk=3)),
    ("B1long", "outputs/day6/models/B1long_s0.pt", lambda: VelocityMLP(60, cond_dim=21, hidden=512, depth=4)),
]:
    if not os.path.exists(path):
        continue
    m = ctor()
    m.load_state_dict(torch.load(path))
    models[name] = m.eval()

out = {}
for split, (lays, items) in splits.items():
    ref_stats = dict(clear=[], nn=[], spread=[])
    for li, code in items:
        env, R = lays[li]["env"], lays[li]["found"][code]
        ref_stats["clear"] += list(-env.penetration(R, relative=True))
        if len(R) > 1:
            ref_stats["spread"].append(spread(R))
            d = np.sqrt(((R[:, None] - R[None]) ** 2).sum(-1).mean(-1))
            np.fill_diagonal(d, np.inf)
            ref_stats["nn"] += list(d.min(1))
    out[f"{split}|reference"] = dict(clear_med=float(np.median(ref_stats["clear"])),
                                     nn_rms_med=float(np.median(ref_stats["nn"])),
                                     spread_med=float(np.median(ref_stats["spread"])))
    for name, m in models.items():
        G = generate(m, lays, items)
        cl, lr, nn, sp = [], [], [], []
        for (li, code), g in zip(items, G):
            env, R = lays[li]["env"], lays[li]["found"][code]
            P = full(env, g)
            cl += list(-env.penetration(P, relative=True))
            lr += list(length(P) / length(R).mean())
            nn += list(nn_rms(P, R))
            sp.append(spread(P))
        cl = np.array(cl)
        out[f"{split}|{name}"] = dict(clear_med=float(np.median(cl)), clear_p10=float(np.percentile(cl, 10)),
                                      len_ratio_med=float(np.median(lr)), len_ratio_p90=float(np.percentile(lr, 90)),
                                      nn_rms_med=float(np.median(nn)), nn_rms_p90=float(np.percentile(nn, 90)),
                                      spread_med=float(np.median(sp)))
    for k, v in out.items():
        if k.startswith(split):
            print(k, {a: round(b, 4) for a, b in v.items()}, flush=True)
with open("outputs/day6/fidelity.json", "w") as f:
    json.dump(out, f, indent=1)

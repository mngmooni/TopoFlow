"""Training-free baseline for the v2 question: post-hoc clearance projection of generated waypoints.

Each interior waypoint that lies inside an obstacle inflated by (c* + delta) is pushed to the inflated boundary
(circle: radially; box: nearest face). Repeated a few times so pushes out of one obstacle into another settle.
Segments between waypoints are not projected, so segment clipping can remain; it is measured, not hidden.
Any change of crossing word caused by the projection shows up as T.

  TOPO_MAX_OBS=5 DATA=outputs/d2 python scripts/v_project.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.taxonomy import tally  # noqa: E402
from topoflow.tdfm import boltzmann, generate, index_code, sample_codes, target_vector  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d2")
MODELS = os.environ.get("MODELS", "P1_s0,P1_s0_60k").split(",")
KS = [int(k) for k in os.environ.get("KS", "2,5,20").split(",")]
DELTA = float(os.environ.get("DELTA", "0.003"))
FULL = os.environ.get("FULL", "0") == "1"  # whole-path projection (segments checked densely)
N_PER = 32
meta = json.load(open(f"{DATA}/betas.json"))
CSTAR = float(meta.get("_margin", 0.03))
with open(f"{DATA}/layouts.pkl", "rb") as f:
    unseen = pickle.load(f)[1]
with open(f"{DATA}/costs.pkl", "rb") as f:
    C = pickle.load(f)
targets = [boltzmann(c, meta["medium"]["beta"]) for c in C["unseen"]]
codes = sample_codes(np.array([target_vector(t, None) for t in targets]), N_PER, seed=900)
x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))


def project(env, W, c, iters=10):
    """W: (n, k, 2) interior waypoints. Returns projected copy."""
    W = W.copy()
    for _ in range(iters):
        moved = False
        for o in env.obstacles:
            if o.kind == "circle":
                ctr = np.array([o.cx, o.cy])
                d = W - ctr
                dist = np.linalg.norm(d, axis=-1)
                inside = dist < o.r + c
                if inside.any():
                    u = d[inside] / np.maximum(dist[inside, None], 1e-9)
                    W[inside] = ctr + u * (o.r + c)
                    moved = True
            else:
                lo = np.array([o.cx - o.hw - c, o.cy - o.hh - c])
                hi = np.array([o.cx + o.hw + c, o.cy + o.hh + c])
                inside = np.all((W > lo) & (W < hi), axis=-1)
                if inside.any():
                    P = W[inside]
                    gaps = np.stack([P[:, 0] - lo[0], hi[0] - P[:, 0], P[:, 1] - lo[1], hi[1] - P[:, 1]], 1)
                    side = gaps.argmin(1)
                    P[side == 0, 0] = lo[0]
                    P[side == 1, 0] = hi[0]
                    P[side == 2, 1] = lo[1]
                    P[side == 3, 1] = hi[1]
                    W[inside] = P
                    moved = True
        if not moved:
            break
    return W


def depth_and_normal(env, X, c):
    """X: (..., 2) points. Returns max violation depth over obstacles and the outward normal of the worst one."""
    best = np.zeros(X.shape[:-1])
    nrm = np.zeros(X.shape)
    for o in env.obstacles:
        dx, dy = X[..., 0] - o.cx, X[..., 1] - o.cy
        if o.kind == "circle":
            dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-12
            d = o.r + c - dist
            n = np.stack([dx / dist, dy / dist], -1)
        else:
            gx, gy = o.hw + c - np.abs(dx), o.hh + c - np.abs(dy)
            d = np.where((gx > 0) & (gy > 0), np.minimum(gx, gy), -1.0)
            n = np.where((gx < gy)[..., None], np.stack([np.sign(dx), 0 * dx], -1), np.stack([0 * dy, np.sign(dy)], -1))
        upd = d > best
        best = np.where(upd, d, best)
        nrm = np.where(upd[..., None], n, nrm)
    return best, nrm


def project_full(env, W, c, rounds=30, sub=8):
    """Push both endpoints of every violating segment along the worst obstacle normal, until the densified path
    clears c everywhere (or rounds run out). Start/goal stay fixed."""
    n = len(W)
    P = np.empty((n, W.shape[1] + 2, 2))
    P[:, 0], P[:, -1] = env.start, env.goal
    P[:, 1:-1] = W
    a = np.linspace(0, 1, sub + 1)[None, None, :, None]
    for _ in range(rounds):
        S = P[:, :-1, None] * (1 - a) + P[:, 1:, None] * a               # (n, T-1, sub+1, 2)
        d, nr = depth_and_normal(env, S, c)
        k = d.argmax(-1)                                                   # worst point per segment
        dseg = np.take_along_axis(d, k[..., None], -1)[..., 0]
        if (dseg <= 0).all():
            break
        nseg = np.take_along_axis(nr, k[..., None, None].repeat(2, -1), 2)[:, :, 0]
        push = (np.maximum(dseg, 0) + 1e-4)[..., None] * nseg * (dseg > 0)[..., None]
        delta = np.zeros_like(P)
        delta[:, :-1] += push
        delta[:, 1:] += push
        delta[:, 0] = delta[:, -1] = 0
        P = P + delta
    return P[:, 1:-1]


for stem in MODELS:
    model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True)
    model.load_state_dict(torch.load(f"{DATA}/models/{stem}.pt"))
    model.eval()
    for K in KS:
        G = generate(model, unseen, codes, K, x0)
        proj = project_full if FULL else project
        Gp = np.stack([proj(l["env"], G[i], CSTAR + DELTA) for i, l in enumerate(unseen)])
        rec = dict(model=stem, K=K, cstar=CSTAR, delta=DELTA, full=FULL,
                   raw=tally(unseen, G, codes, CSTAR, targets, index_code),
                   proj=tally(unseen, Gp, codes, CSTAR, targets, index_code),
                   proj_c0=tally(unseen, Gp, codes, 0.0, targets, index_code),
                   mean_shift=float(np.sqrt(((Gp - G) ** 2).sum(-1)).mean()),
                   moved_frac=float((np.abs(Gp - G).sum(-1) > 0).any(-1).mean()))
        print(json.dumps(rec), flush=True)
        with open(f"{DATA}/projection{'_full' if FULL else ''}.jsonl", "a") as fo:
            fo.write(json.dumps(rec) + "\n")
print("done")

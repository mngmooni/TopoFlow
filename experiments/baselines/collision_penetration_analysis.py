"""research phase: split collisions into deep (crosses into the obstacle core) vs shallow grazes.

Uses models saved by route_conditioning_benchmark.py. E0 only (single circle).
Run from the repo root:  python experiments/baselines/collision_penetration_analysis.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.cfm import VelocityMLP, euler_sample  # noqa: E402
from topoflow.env import e0_env  # noqa: E402

env = e0_env()
o = env.obstacles[0]


def penetration(P, n_sub=20):
    """Max depth inside the circle along the polyline (<= 0 means no collision)."""
    s = np.linspace(0, 1, n_sub)
    pts = P[:, :-1, None, :] * (1 - s)[None, None, :, None] + P[:, 1:, None, :] * s[None, None, :, None]
    return o.r - np.linalg.norm(pts - [o.cx, o.cy], axis=-1).min(axis=(1, 2))


d = np.load("results/baselines/environment_topology/e0_balanced.npz")
X = d["X"]
T = X.shape[1]
probs = np.bincount(d["class_id"]) / len(X)
print("training data min clearance:", round(float(-penetration(X).max()), 4))
for kind, cd in [("B0", 0), ("B1", 2)]:
    m = VelocityMLP(60, cond_dim=cd, hidden=512, depth=4)
    m.load_state_dict(torch.load(f"results/baselines/route_conditioning/models/e0_balanced_{kind}_s0.pt"))
    m.eval()
    for K in [2, 10, 50, 200]:
        cid = np.random.default_rng(5).choice(2, 4000, p=probs)
        c = torch.nn.functional.one_hot(torch.tensor(cid), 2).float()
        s = euler_sample(lambda x, tt, cc: m(x, tt, cc) if cd else m(x, tt), 4000, 60, K, seed=9, cond=c).numpy()
        P = np.empty((4000, T, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = s.reshape(4000, T - 2, 2)
        p = penetration(P)
        hit = p > 0
        q = np.percentile(p[hit], [50, 90, 100]).round(3) if hit.any() else None
        print(f"{kind} NFE={K:3d}  collision {hit.mean():.4f}  depth p50/p90/max {q}  deep(>0.1) {(p > 0.1).mean():.4f}")

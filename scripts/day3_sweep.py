"""Day 3b: does B0's collision floor survive more capacity / training?

Writes one JSON line per run to outputs/day3/sweep.jsonl (partial results usable).
Run from the repo root:  python scripts/day3_sweep.py
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import VelocityMLP, euler_sample, n_params, oracle_velocity, train_cfm  # noqa: E402
from topoflow.env import e0_env  # noqa: E402
from topoflow.topology import crossing_word, word_to_str  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
OUT = "outputs/day3"
os.makedirs(OUT, exist_ok=True)
KS = [10, 50, 200]
N_SAMPLES = 2000
env = e0_env()

# (dataset, hidden, depth, steps, seeds)
CONFIGS = [
    ("e0_balanced", 256, 4, 8000, [0, 1, 2]),
    ("e0_balanced", 256, 4, 30000, [0, 1, 2]),
    ("e0_balanced", 512, 4, 30000, [0, 1, 2]),
    ("e0_balanced", 1024, 4, 20000, [0]),
    ("e0_imbalanced_90_10", 512, 4, 30000, [0, 1]),
]


def to_paths(interior, T):
    n = len(interior)
    P = np.empty((n, T, 2))
    P[:, 0], P[:, -1] = env.start, env.goal
    P[:, 1:-1] = interior.reshape(n, T - 2, 2)
    return P


def roughness(P):
    """Mean squared second difference per waypoint (lower = smoother)."""
    return float((np.diff(P, n=2, axis=1) ** 2).sum(-1).mean())


def evaluate(P, Xtrain):
    coll = env.collision_mask(P)
    words = np.array([word_to_str(crossing_word(env, p)) for p in P])
    valid = ~coll
    cnt = Counter(words[valid])
    nv = max(valid.sum(), 1)
    by_word = {w: float(coll[words == w].mean()) for w in ("e", "a") if (words == w).any()}
    a = torch.tensor(P.reshape(len(P), -1))
    b = torch.tensor(Xtrain.reshape(len(Xtrain), -1))
    nn_d = (torch.cdist(a, b).min(1).values / np.sqrt(P.shape[1])).numpy()
    return dict(collision=float(coll.mean()), frac_a=cnt.get("a", 0) / nv,
                collision_by_word=by_word, n_other=int((~np.isin(words, ["e", "a"])).sum()),
                roughness=roughness(P), nn_rms_median=float(np.median(nn_d)))


data_cache = {}
for name in sorted({c[0] for c in CONFIGS}):
    d = np.load(f"outputs/day1/{name}.npz")
    X = d["X"]
    data = torch.tensor(X[:, 1:-1].reshape(len(X), -1), dtype=torch.float32)
    data_cache[name] = (X, data)
    # reference numbers: training data and oracle with more samples
    ref = dict(kind="reference", dataset=name, train_roughness=roughness(X),
               train_frac_a=float(np.mean([w == "a" for w in d["words"]])))
    s = euler_sample(lambda x, tt, c=None: oracle_velocity(x, float(tt[0]), data),
                     5000, data.shape[1], 200, seed=7).numpy()
    ref["oracle_K200_n5000"] = evaluate(to_paths(s, X.shape[1]), X)
    print(json.dumps(ref))
    with open(f"{OUT}/sweep.jsonl", "a") as f:
        f.write(json.dumps(ref) + "\n")

for name, hidden, depth, steps, seeds in CONFIGS:
    X, data = data_cache[name]
    for seed in seeds:
        torch.manual_seed(seed)
        model = VelocityMLP(data.shape[1], hidden=hidden, depth=depth)
        t0 = time.time()
        losses = train_cfm(model, data, steps=steps, seed=seed)
        model.eval()
        rec = dict(kind="run", dataset=name, hidden=hidden, depth=depth, steps=steps, seed=seed,
                   params=n_params(model), train_sec=round(time.time() - t0),
                   final_loss=float(np.mean(losses[-500:])), per_k={})
        for K in KS:
            s = euler_sample(lambda x, tt, c=None: model(x, tt), N_SAMPLES, data.shape[1], K,
                             seed=1000 + seed).numpy()
            rec["per_k"][K] = evaluate(to_paths(s, X.shape[1]), X)
        print(json.dumps(rec), flush=True)
        with open(f"{OUT}/sweep.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
print("done")

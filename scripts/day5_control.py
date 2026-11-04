"""Day 5b controls.

(1) B1-full: same as B1 but trained on ALL codes (no held-out). Evaluating it on the held-out
    patterns separates "never saw this combination" from "alternating patterns are just
    geometrically harder".
(2) B0 at NFE 50 from saved models: is B0's E2 failure integration error?

Run from the repo root after day5_e2.py:  python scripts/day5_control.py
"""
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import VelocityMLP, euler_sample, n_params, train_cfm  # noqa: E402
from topoflow.e2 import HELD_OUT, classify, code_ok_for_train, env_features, pad_code, stack_samples  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
OUT = "outputs/day5"
T = 32
KS = [2, 5, 20]
HIDDEN, DEPTH, STEPS = 512, 4, 30000
SEEDS = [0, 1]

with open(f"{OUT}/layouts.pkl", "rb") as f:
    train_layouts, unseen_layouts = pickle.load(f)
seen_eval = train_layouts[:200]
held_layouts = [l for l in unseen_layouts if any(c in HELD_OUT for c in l["found"])]


def to_paths(env, s):
    P = np.empty((len(s), T, 2))
    P[:, 0], P[:, -1] = env.start, env.goal
    P[:, 1:-1] = s.reshape(len(s), T - 2, 2)
    return P


def gen(model, env, n, K, seed, code=None, dim=60):
    f, m = env_features(env)
    cond = np.concatenate([f, m] + ([pad_code(code)] if code is not None else []))
    c = torch.tensor(np.repeat(cond[None], n, 0), dtype=torch.float32)
    s = euler_sample(lambda xx, tt, cc: model(xx, tt, cc), n, dim, K, seed=seed, cond=c).numpy()
    return to_paths(env, s)


def eval_b1(model, layouts, K, seed, which):
    n = adh = vadh = coll = deep = 0
    for i, lay in enumerate(layouts):
        env = lay["env"]
        for j, code in enumerate(lay["found"]):
            if which == "trainable" and not code_ok_for_train(code):
                continue
            if which == "held_out" and code_ok_for_train(code):
                continue
            P = gen(model, env, 8, K, seed * 100000 + i * 16 + j, code=code)
            cl, dp, codes = classify(env, P)
            match = np.array([c == code for c in codes])
            n += len(P)
            adh += match.sum()
            vadh += (match & ~cl).sum()
            coll += cl.sum()
            deep += dp.sum()
    return dict(n=int(n), adherence=adh / n, valid_adherence=vadh / n, collision=coll / n, deep=deep / n)


def eval_b0(model, layouts, K, seed):
    coll, deep, cov = [], [], []
    for i, lay in enumerate(layouts):
        env = lay["env"]
        target = [c for c in lay["found"] if code_ok_for_train(c)]
        P = gen(model, env, 32, K, seed * 100000 + i)
        cl, dp, codes = classify(env, P)
        coll.append(cl.mean())
        deep.append(dp.mean())
        valid = {c for c, h in zip(codes, cl) if not h and c is not None}
        cov.append(len(valid & set(target)) / len(target))
    return dict(collision=float(np.mean(coll)), deep=float(np.mean(deep)), coverage=float(np.mean(cov)))


results = {}
# ---- (2) B0 at NFE 50, and per obstacle count at NFE 20
for seed in SEEDS:
    m = VelocityMLP(60, cond_dim=18, hidden=HIDDEN, depth=DEPTH)
    m.load_state_dict(torch.load(f"{OUT}/models/B0_s{seed}.pt"))
    m.eval()
    r = {"K50_unseen_pos": eval_b0(m, unseen_layouts, 50, seed)}
    for k in (1, 2, 3):
        r[f"K20_unseen_pos_{k}obs"] = eval_b0(m, [l for l in unseen_layouts if l["env"].m == k], 20, seed)
    results[f"B0_s{seed}_extra"] = r
    print(f"B0 s{seed}: {json.dumps(r)}", flush=True)

# ---- (1) B1-full
train = stack_samples(train_layouts)  # all codes
x = torch.tensor(train["x"])
cond = torch.tensor(np.concatenate([train["feat"], train["mask"], train["code"]], 1))
print(f"B1-full train samples {len(x)}", flush=True)
for seed in SEEDS:
    torch.manual_seed(seed)
    model = VelocityMLP(60, cond_dim=cond.shape[1], hidden=HIDDEN, depth=DEPTH)
    t0 = time.time()
    losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=seed)
    model.eval()
    torch.save(model.state_dict(), f"{OUT}/models/B1full_s{seed}.pt")
    rec = dict(params=n_params(model), final_loss=float(np.mean(losses[-500:])),
               train_sec=round(time.time() - t0), per_k={})
    for K in KS:
        rec["per_k"][K] = dict(unseen_pos=eval_b1(model, unseen_layouts, K, seed, "trainable"),
                               held_out_patterns=eval_b1(model, held_layouts, K, seed, "held_out"))
        print(f"B1full s{seed} K={K}: {json.dumps(rec['per_k'][K], default=float)}", flush=True)
    results[f"B1full_s{seed}"] = rec
    with open(f"{OUT}/control_results.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
print("done")

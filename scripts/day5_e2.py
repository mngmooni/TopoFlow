"""Day 5: E2 (variable layouts, 1-3 obstacles). B0 = env-conditioned CFM, B1 = env + route code (padded MLP).

Run from the repo root:  python scripts/day5_e2.py
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
from topoflow.e2 import (HELD_OUT, MAX_OBS, build_layouts, classify, code_ok_for_train, env_features,  # noqa: E402
                         pad_code, stack_samples)

torch.set_num_threads(max(1, os.cpu_count() or 1))
OUT = "outputs/day5"
os.makedirs(f"{OUT}/models", exist_ok=True)
T = 32
N_TRAIN_LAYOUTS, N_SEEN_EVAL, N_UNSEEN = 3000, 200, 300
PER_CODE = 6
KS = [2, 5, 20]
HIDDEN, DEPTH, STEPS = 512, 4, 30000
SEEDS = [0, 1]
B0_SAMPLES_PER_ENV = 32
B1_SAMPLES_PER_CODE = 8

# ------------------------------------------------------------------ data
cache = f"{OUT}/layouts.pkl"
if os.path.exists(cache):
    with open(cache, "rb") as f:
        train_layouts, unseen_layouts = pickle.load(f)
else:
    t0 = time.time()
    train_layouts = build_layouts(np.random.default_rng(100), N_TRAIN_LAYOUTS, PER_CODE, T=T)
    unseen_layouts = build_layouts(np.random.default_rng(200), N_UNSEEN, PER_CODE, T=T)
    with open(cache, "wb") as f:
        pickle.dump((train_layouts, unseen_layouts), f)
    print(f"built layouts in {time.time() - t0:.0f}s")

train = stack_samples(train_layouts, code_filter=code_ok_for_train)
x = torch.tensor(train["x"])
env_c = torch.tensor(np.concatenate([train["feat"], train["mask"]], 1))
code_c = torch.tensor(train["code"])
print(f"train samples {len(x)} from {len(train_layouts)} layouts; "
      f"n_obs hist {Counter(l['env'].m for l in train_layouts)}")
n3 = [l for l in unseen_layouts if l["env"].m == 3]
print(f"unseen layouts {len(unseen_layouts)}, 3-obs {len(n3)}, "
      f"with a held-out code found {sum(any(c in HELD_OUT for c in l['found']) for l in n3)}")

seen_eval = train_layouts[:N_SEEN_EVAL]


def to_paths(env, s):
    P = np.empty((len(s), T, 2))
    P[:, 0], P[:, -1] = env.start, env.goal
    P[:, 1:-1] = s.reshape(len(s), T - 2, 2)
    return P


def gen(model, env, n, K, seed, code=None):
    f, m = env_features(env)
    cond = np.concatenate([f, m])
    if code is not None:
        cond = np.concatenate([cond, pad_code(code)])
    c = torch.tensor(np.repeat(cond[None], n, 0), dtype=torch.float32)
    s = euler_sample(lambda xx, tt, cc: model(xx, tt, cc), n, x.shape[1], K, seed=seed, cond=c).numpy()
    return to_paths(env, s)


def eval_b0(model, layouts, K, seed):
    coll, deep, cov, tv, held_gen, other = [], [], [], [], 0, 0
    for i, lay in enumerate(layouts):
        env = lay["env"]
        target = [c for c in lay["found"] if code_ok_for_train(c)]
        if not target:
            continue
        P = gen(model, env, B0_SAMPLES_PER_ENV, K, seed * 100000 + i)
        cl, dp, codes = classify(env, P)
        coll.append(cl.mean())
        deep.append(dp.mean())
        valid_codes = [c for c, h in zip(codes, cl) if not h and c is not None]
        other += sum(1 for c, h in zip(codes, cl) if not h and c is None)
        held_gen += sum(1 for c in valid_codes if not code_ok_for_train(c))
        cov.append(len(set(valid_codes) & set(target)) / len(target))
        cnt = Counter(c for c in valid_codes if c in target)
        tot = sum(cnt.values())
        if tot:
            tv.append(0.5 * sum(abs(cnt.get(c, 0) / tot - 1 / len(target)) for c in target))
    return dict(collision=float(np.mean(coll)), deep=float(np.mean(deep)), coverage=float(np.mean(cov)),
                tv_to_uniform=float(np.mean(tv)) if tv else float("nan"), held_out_codes_generated=int(held_gen),
                valid_outside_family=int(other))


def eval_b1(model, layouts, K, seed, which):
    n = adh = vadh = coll = deep = 0
    for i, lay in enumerate(layouts):
        env = lay["env"]
        for j, code in enumerate(lay["found"]):
            if which == "trainable" and not code_ok_for_train(code):
                continue
            if which == "held_out" and code_ok_for_train(code):
                continue
            P = gen(model, env, B1_SAMPLES_PER_CODE, K, seed * 100000 + i * 16 + j, code=code)
            cl, dp, codes = classify(env, P)
            match = np.array([c == code for c in codes])
            n += len(P)
            adh += match.sum()
            vadh += (match & ~cl).sum()
            coll += cl.sum()
            deep += dp.sum()
    return dict(n=int(n), adherence=adh / n, valid_adherence=vadh / n, collision=coll / n, deep=deep / n)


results = {}
held_layouts = [l for l in unseen_layouts if any(c in HELD_OUT for c in l["found"])]
for seed in SEEDS:
    for kind in ["B0", "B1"]:
        cond = env_c if kind == "B0" else torch.cat([env_c, code_c], 1)
        torch.manual_seed(seed)
        model = VelocityMLP(x.shape[1], cond_dim=cond.shape[1], hidden=HIDDEN, depth=DEPTH)
        t0 = time.time()
        losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=seed)
        model.eval()
        torch.save(model.state_dict(), f"{OUT}/models/{kind}_s{seed}.pt")
        rec = dict(params=n_params(model), final_loss=float(np.mean(losses[-500:])),
                   train_sec=round(time.time() - t0), per_k={})
        for K in KS:
            if kind == "B0":
                rec["per_k"][K] = dict(seen=eval_b0(model, seen_eval, K, seed),
                                       unseen_pos=eval_b0(model, unseen_layouts, K, seed))
            else:
                rec["per_k"][K] = dict(seen=eval_b1(model, seen_eval, K, seed, "trainable"),
                                       unseen_pos=eval_b1(model, unseen_layouts, K, seed, "trainable"),
                                       unseen_combo=eval_b1(model, held_layouts, K, seed, "held_out"))
            print(f"{kind} s{seed} K={K}: {json.dumps(rec['per_k'][K], default=float)}", flush=True)
        results[f"{kind}_s{seed}"] = rec
        print(f"{kind} s{seed} done ({rec['train_sec']}s, loss {rec['final_loss']:.4f})", flush=True)
        with open(f"{OUT}/results.json", "w") as f:
            json.dump(results, f, indent=1, default=float)
print("done")

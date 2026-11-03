"""Day 4: does conditioning on the route class change low-NFE behaviour, even for exact fields?

Compares, on E0, at matched budget (512x4 MLP, 30k steps):
  oracle-U : exact marginal field of the full (two-mode) empirical data
  oracle-C : exact field of each class's empirical data, class drawn at training frequency
  B0       : learned unconditional CFM
  B1       : learned class-conditioned CFM (one-hot route code), class drawn at training frequency

Run from the repo root:  python scripts/day4_b1_conditional.py
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import VelocityMLP, euler_sample, n_params, oracle_velocity, train_cfm  # noqa: E402
from topoflow.env import e0_env  # noqa: E402
from topoflow.topology import crossing_word, expected_word_for_code  # noqa: E402
from topoflow.viz import draw_env  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
OUT = "outputs/day4"
os.makedirs(f"{OUT}/models", exist_ok=True)
KS = [1, 2, 5, 10, 50]
N = 2000
HIDDEN, DEPTH, STEPS = 512, 4, 30000
RUNS = [("e0_balanced", [0, 1]), ("e0_imbalanced_90_10", [0])]
env = e0_env()


def to_paths(interior, T):
    n = len(interior)
    P = np.empty((n, T, 2))
    P[:, 0], P[:, -1] = env.start, env.goal
    P[:, 1:-1] = interior.reshape(n, T - 2, 2)
    return P


def evaluate(P, target_codes=None):
    coll = env.collision_mask(P)
    words = [crossing_word(env, p) for p in P]
    valid = ~coll
    below = np.array([w == (1,) for w in words])
    out = dict(collision=float(coll.mean()),
               frac_a=float(below[valid].mean()) if valid.any() else float("nan"),
               roughness=float((np.diff(P, n=2, axis=1) ** 2).sum(-1).mean()))
    if target_codes is not None:
        match = np.array([w == expected_word_for_code(tuple(c)) for w, c in zip(words, target_codes)])
        out["adherence"] = float(match.mean())
        out["valid_adherence"] = float((match & valid).mean())
    return out


def sample_classes(rng_seed, probs, n):
    g = np.random.default_rng(rng_seed)
    return g.choice(len(probs), size=n, p=probs)


results = {}
store = {}
for name, seeds in RUNS:
    d = np.load(f"outputs/day1/{name}.npz")
    X = d["X"]
    T = X.shape[1]
    data = torch.tensor(X[:, 1:-1].reshape(len(X), -1), dtype=torch.float32)
    dim = data.shape[1]
    cls = torch.tensor(d["class_id"])
    classes = [tuple(c) for c in d["classes"]]
    n_cls = len(classes)
    probs = np.bincount(d["class_id"], minlength=n_cls) / len(X)
    onehot = torch.nn.functional.one_hot(cls, n_cls).float()
    per_class_data = [data[cls == i] for i in range(n_cls)]
    res = {"classes": [list(c) for c in classes], "train_probs": probs.tolist(),
           "oracle_U": {}, "oracle_C": {}, "B0": {}, "B1": {}}
    print(f"\n=== {name}  classes={classes}  probs={probs.round(3)}")

    # -------- oracles
    for K in KS:
        s = euler_sample(lambda x, tt, c=None: oracle_velocity(x, float(tt[0]), data), N, dim, K, seed=11).numpy()
        P = to_paths(s, T)
        res["oracle_U"][K] = evaluate(P)
        store[(name, "oracle_U", K)] = P

        cid = sample_classes(12, probs, N)
        parts, codes = [], []
        for i in range(n_cls):
            m = int((cid == i).sum())
            if m == 0:
                continue
            di = per_class_data[i]
            s = euler_sample(lambda x, tt, c=None: oracle_velocity(x, float(tt[0]), di), m, dim, K, seed=13 + i).numpy()
            parts.append(s)
            codes += [classes[i]] * m
        P = to_paths(np.concatenate(parts), T)
        res["oracle_C"][K] = evaluate(P, codes)
        store[(name, "oracle_C", K)] = P
        print(f"K={K:3d} oracle_U {res['oracle_U'][K]}")
        print(f"        oracle_C {res['oracle_C'][K]}")

    # -------- learned
    for seed in seeds:
        for kind in ["B0", "B1"]:
            torch.manual_seed(seed)
            cond_dim = n_cls if kind == "B1" else 0
            model = VelocityMLP(dim, cond_dim=cond_dim, hidden=HIDDEN, depth=DEPTH)
            t0 = time.time()
            losses = train_cfm(model, data, cond=onehot if kind == "B1" else None, steps=STEPS, seed=seed)
            model.eval()
            torch.save(model.state_dict(), f"{OUT}/models/{name}_{kind}_s{seed}.pt")
            per_k = {}
            for K in KS:
                if kind == "B0":
                    s = euler_sample(lambda x, tt, c=None: model(x, tt), N, dim, K, seed=2000 + seed).numpy()
                    P = to_paths(s, T)
                    per_k[K] = evaluate(P)
                else:
                    cid = sample_classes(3000 + seed, probs, N)
                    c = torch.nn.functional.one_hot(torch.tensor(cid), n_cls).float()
                    s = euler_sample(lambda x, tt, cc: model(x, tt, cc), N, dim, K, seed=2000 + seed, cond=c).numpy()
                    P = to_paths(s, T)
                    per_k[K] = evaluate(P, [classes[i] for i in cid])
                if seed == seeds[0]:
                    store[(name, kind, K)] = P
            res[kind][seed] = dict(per_k=per_k, params=n_params(model),
                                   final_loss=float(np.mean(losses[-500:])), train_sec=round(time.time() - t0))
            print(f"{kind} seed={seed} ({time.time() - t0:.0f}s, params {n_params(model)}, loss {res[kind][seed]['final_loss']:.4f})")
            for K in KS:
                print(f"   K={K:3d} {per_k[K]}")
    results[name] = res
    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1, default=str)

# ---------------------------------------------------------------- plots
for name, _ in RUNS:
    show = [1, 2, 5]
    rows = ["oracle_U", "oracle_C", "B0", "B1"]
    fig, axes = plt.subplots(len(rows), len(show), figsize=(3.6 * len(show), 3.6 * len(rows)))
    for r, kind in enumerate(rows):
        for c, K in enumerate(show):
            ax = axes[r, c]
            draw_env(ax, env, show_cuts=False)
            P = store[(name, kind, K)]
            sel = np.random.default_rng(0).choice(len(P), 150, replace=False)
            coll = env.collision_mask(P[sel])
            for p, hit in zip(P[sel], coll):
                w = crossing_word(env, p)
                col = "tab:red" if hit else ("tab:blue" if w == () else ("tab:orange" if w == (1,) else "tab:purple"))
                ax.plot(p[:, 0], p[:, 1], color=col, lw=0.7, alpha=0.5)
            m = results[name][kind][K] if kind.startswith("oracle") else results[name][kind][RUNS[[n for n, _ in RUNS].index(name)][1][0]]["per_k"][K]
            ax.set_title(f"{kind}  NFE={K}\ncollision {m['collision']:.3f}", fontsize=10)
    fig.suptitle(f"{name}   blue=above  orange=below  red=collision", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}_lowNFE.png", dpi=120)
    plt.close(fig)

fig, axes = plt.subplots(1, len(RUNS), figsize=(5.5 * len(RUNS), 4))
for ax, (name, seeds) in zip(np.atleast_1d(axes), RUNS):
    r = results[name]
    ax.plot(KS, [r["oracle_U"][K]["collision"] for K in KS], "o--", color="k", label="oracle uncond.")
    ax.plot(KS, [r["oracle_C"][K]["collision"] for K in KS], "o-", color="0.5", label="oracle cond.")
    for kind, col in [("B0", "tab:blue"), ("B1", "tab:orange")]:
        v = np.array([[r[kind][s]["per_k"][K]["collision"] for K in KS] for s in seeds])
        ax.errorbar(KS, v.mean(0), yerr=v.std(0) if len(seeds) > 1 else None, fmt="s-", color=col,
                    label=f"{kind} learned ({len(seeds)} seed)")
    ax.set_xscale("log")
    ax.set_xlabel("NFE")
    ax.set_ylabel("collision rate")
    ax.set_title(name)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/collision_vs_nfe_cond.png", dpi=130)
plt.close(fig)
print("done")

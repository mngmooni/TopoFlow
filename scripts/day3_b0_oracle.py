"""Day 3: B0 (vanilla CFM) vs exact oracle field on E0, across NFE.

Separates three error sources (v0.4 plan):
  oracle, low vs high NFE        -> numerical integration (path geometry) error
  learned vs oracle, same NFE    -> velocity approximation error
  class distribution at high NFE -> learned distribution error

Run from the repo root:  python scripts/day3_b0_oracle.py
Requires outputs/day1/e0_*.npz (python scripts/day1_demo.py).
"""
import json
import os
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import VelocityMLP, euler_sample, n_params, oracle_velocity, train_cfm  # noqa: E402
from topoflow.env import e0_env  # noqa: E402
from topoflow.topology import crossing_word, word_to_str  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
OUT = "outputs/day3"
os.makedirs(OUT, exist_ok=True)
KS = [1, 2, 5, 10, 50, 200]
SEEDS = [0, 1, 2]
N_SAMPLES = 1000
STEPS = 8000

env = e0_env()


def to_paths(interior, T):
    n = len(interior)
    P = np.empty((n, T, 2))
    P[:, 0] = env.start
    P[:, -1] = env.goal
    P[:, 1:-1] = interior.reshape(n, T - 2, 2)
    return P


def evaluate(paths, train_paths):
    coll = env.collision_mask(paths)
    words = [word_to_str(crossing_word(env, p)) for p in paths]
    valid_words = [w for w, c in zip(words, coll) if not c]
    cnt = Counter(valid_words)
    n_valid = len(valid_words)
    # memorization: RMS per-waypoint distance to the nearest training path
    a = torch.tensor(paths.reshape(len(paths), -1))
    b = torch.tensor(train_paths.reshape(len(train_paths), -1))
    nn_d = (torch.cdist(a, b).min(dim=1).values / np.sqrt(paths.shape[1])).numpy()
    return dict(
        collision=float(coll.mean()),
        frac_e=cnt.get("e", 0) / max(n_valid, 1),       # among valid: above
        frac_a=cnt.get("a", 0) / max(n_valid, 1),       # among valid: below
        other_words=dict(Counter(w for w in words if w not in ("e", "a"))),
        nn_rms_median=float(np.median(nn_d)),
    )


@torch.no_grad()
def velocity_error(model, data, ts, seed=0, n=2000):
    """E ||v_theta - v*||^2 / d on the true marginal x_t, per t."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for t in ts:
        idx = torch.randint(0, len(data), (n,), generator=g)
        x0 = torch.randn(n, data.shape[1], generator=g)
        xt = (1 - t) * x0 + t * data[idx]
        v_star = oracle_velocity(xt, t, data)
        v = model(xt, torch.full((n,), t))
        out.append(float(((v - v_star) ** 2).mean()))
    return out


results = {}
sample_store = {}
for name in ["e0_balanced", "e0_imbalanced_90_10"]:
    d = np.load(f"outputs/day1/{name}.npz")
    X = d["X"]
    T = X.shape[1]
    data = torch.tensor(X[:, 1:-1].reshape(len(X), -1), dtype=torch.float32)
    dim = data.shape[1]
    train_frac_a = float(np.mean([w == "a" for w in d["words"]]))
    res = {"train_frac_a": train_frac_a, "oracle": {}, "learned": {}}
    print(f"\n=== {name}: n={len(X)}, dim={dim}, train frac below(a)={train_frac_a:.3f}")

    # ---- oracle
    def oracle_field(x, tt, cond=None):
        return oracle_velocity(x, float(tt[0]), data)

    for K in KS:
        s = euler_sample(oracle_field, N_SAMPLES, dim, K, seed=123).numpy()
        P = to_paths(s, T)
        res["oracle"][K] = evaluate(P, X)
        sample_store[(name, "oracle", K)] = P
        print(f"oracle  K={K:3d}  {res['oracle'][K]}")

    # ---- learned B0, several seeds
    ts = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.97]
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = VelocityMLP(dim, hidden=256, depth=4)
        t0 = time.time()
        losses = train_cfm(model, data, steps=STEPS, seed=seed)
        model.eval()
        field = lambda x, tt, cond=None: model(x, tt)  # noqa: E731
        per_k = {}
        for K in KS:
            s = euler_sample(field, N_SAMPLES, dim, K, seed=1000 + seed).numpy()
            P = to_paths(s, T)
            per_k[K] = evaluate(P, X)
            if seed == 0:
                sample_store[(name, "learned", K)] = P
        verr = velocity_error(model, data, ts, seed=seed)
        res["learned"][seed] = dict(per_k=per_k, final_loss=float(np.mean(losses[-500:])),
                                    vel_err={str(t): e for t, e in zip(ts, verr)},
                                    params=n_params(model), train_sec=time.time() - t0)
        print(f"learned seed={seed}  ({time.time() - t0:.0f}s, loss {res['learned'][seed]['final_loss']:.4f})")
        for K in KS:
            print(f"   K={K:3d}  {per_k[K]}")
        print("   vel err by t:", {t: round(e, 4) for t, e in zip(ts, verr)})
    results[name] = res

with open(f"{OUT}/results.json", "w") as f:
    json.dump(results, f, indent=1, default=str)

# ---------------------------------------------------------------- plots
from topoflow.viz import draw_env  # noqa: E402

for name in ["e0_balanced", "e0_imbalanced_90_10"]:
    show = [1, 2, 5, 50]
    fig, axes = plt.subplots(2, len(show), figsize=(4 * len(show), 8))
    for r, kind in enumerate(["oracle", "learned"]):
        for c, K in enumerate(show):
            ax = axes[r, c]
            draw_env(ax, env, show_cuts=False)
            P = sample_store[(name, kind, K)][:150]
            coll = env.collision_mask(P)
            for p, hit in zip(P, coll):
                w = crossing_word(env, p)
                col = "tab:red" if hit else ("tab:blue" if w == () else ("tab:orange" if w == (1,) else "tab:purple"))
                ax.plot(p[:, 0], p[:, 1], color=col, lw=0.7, alpha=0.5)
            m = results[name][kind][K] if kind == "oracle" else results[name][kind][0]["per_k"][K]
            ax.set_title(f"{kind}  NFE={K}\ncollision {m['collision']:.2f}  below {m['frac_a']:.2f}", fontsize=10)
    fig.suptitle(f"{name}   (blue=above, orange=below, red=collision)", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}_samples.png", dpi=130)
    plt.close(fig)

# summary chart: collision vs NFE
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, name in zip(axes, ["e0_balanced", "e0_imbalanced_90_10"]):
    r = results[name]
    ax.plot(KS, [r["oracle"][K]["collision"] for K in KS], "o-", label="oracle", color="k")
    lc = np.array([[r["learned"][s]["per_k"][K]["collision"] for K in KS] for s in SEEDS])
    ax.errorbar(KS, lc.mean(0), yerr=lc.std(0), fmt="s-", label=f"B0 learned ({len(SEEDS)} seeds)", color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("NFE (Euler steps)")
    ax.set_ylabel("collision rate")
    ax.set_title(name)
    ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/collision_vs_nfe.png", dpi=130)
plt.close(fig)
print("\nsaved to", OUT)

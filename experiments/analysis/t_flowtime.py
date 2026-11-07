"""research phase step 1: where in flow time does P1's high-NFE clearance failure arise? (D2)

(a) Track the predicted endpoint x1_hat(t) = x_t + (1-t) v along the Euler NFE-20 trajectory; record per step the
    fraction of samples whose x1_hat is U / M at c*, and for samples that end in U/M, the step of the last entry.
(b) Field swap: learned field before t*, exact code-conditional field after (and the reverse), NFE 20.
(c) Distance between learned and exact x1_hat along the learned trajectory, per t.

  TOPO_MAX_OBS=5 python experiments/analysis/t_flowtime.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import MAX_OBS, env_features, pad_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.taxonomy import tally  # noqa: E402
from topoflow.tdfm import boltzmann, code_index, index_code, sample_codes, target_vector  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "results/d2")
STEM = os.environ.get("STEM", "P1_s0")
OUT = f"{DATA}/flowtime_{STEM}.json"
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
lay_of = torch.arange(L).repeat_interleave(N_PER)

conds = torch.tensor(np.array([np.concatenate([*env_features(unseen[i]["env"]),
                                               pad_code(index_code(int(codes[i, j]), unseen[i]["env"].m))])
                               for i in range(L) for j in range(N_PER)]), dtype=torch.float32)
model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True)
model.load_state_dict(torch.load(f"{DATA}/models/{STEM}.pt"))
model.eval()

# exact code-conditional field (reference paths of the requested code only)
nmax = max(sum(len(P) for P in l["found"].values()) for l in unseen)
D = torch.zeros(L, nmax, 60)
CODE = torch.full((L, nmax), -1, dtype=torch.long)
for i, lay in enumerate(unseen):
    j = 0
    for c, P in lay["found"].items():
        for p in P:
            D[i, j] = torch.tensor(p[1:-1].reshape(-1), dtype=torch.float32)
            CODE[i, j] = code_index(c)
            j += 1
LW = torch.where(CODE[lay_of] == torch.tensor(codes.reshape(-1))[:, None], 0.0, -torch.inf)


def v_learned(x, t):
    out = []
    for s in range(0, len(x), 4096):
        out.append(model(x[s:s + 4096], torch.full((len(x[s:s + 4096]),), t), conds[s:s + 4096]))
    return torch.cat(out)


def v_oracle(x, t):
    s1 = 1.0 - t
    out = []
    for s in range(0, len(x), 2048):
        Dx = D[lay_of[s:s + 2048]]
        d2 = ((x[s:s + 2048, None, :] - t * Dx) ** 2).sum(-1)
        w = torch.softmax(-d2 / (2 * s1 * s1) + LW[s:s + 2048], dim=1)
        out.append(((w[..., None] * Dx).sum(1) - x[s:s + 2048]) / s1)
    return torch.cat(out)


def cats_of(X):
    """per-sample category at c* for flat samples X (L*N, 60)."""
    G = X.numpy().reshape(L, N_PER, 30, 2)
    from topoflow.taxonomy import classify_samples
    out = np.empty((L, N_PER), dtype="<U1")
    for i, lay in enumerate(unseen):
        env = lay["env"]
        P = np.empty((N_PER, 32, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = G[i]
        req = [index_code(int(k), env.m) for k in codes[i]]
        out[i] = classify_samples(env, P, req, set(lay["found"]), CSTAR)[0]
    return out.reshape(-1)


def solve(K, field_of_t):
    x = x0.clone()
    hats = []
    for k in range(K):
        t = k / K
        v = field_of_t(t)(x, t)
        hats.append((t, x + (1 - t) * v))
        x = x + v / K
    return x, hats


res = {}
with torch.no_grad():
    # (a) predicted-endpoint tracking, learned field
    for K in (20, 5):
        xf, hats = solve(K, lambda t: v_learned)
        final = cats_of(xf)
        bad = np.isin(final, ["U", "M"])
        per_t, hat_cats = [], []
        for t, h in hats:
            c = cats_of(h)
            hat_cats.append(np.isin(c, ["U", "M"]))
            per_t.append(dict(t=t, U=float((c == "U").mean()), M=float((c == "M").mean())))
        H = np.stack(hat_cats)  # (K, N)
        # last entry: first step after which x1_hat is in violation at every later step (for bad finals)
        last_entry = []
        for j in np.where(bad)[0]:
            col = H[:, j]
            k = K
            while k > 0 and col[k - 1]:
                k -= 1
            last_entry.append(k / K if k < K else 1.0)  # 1.0 = never in violation by the last x1_hat
        res[f"track_K{K}"] = dict(final_U=float((final == "U").mean()), final_M=float((final == "M").mean()),
                                  per_t=per_t, n_bad=int(bad.sum()),
                                  last_entry_hist={str(round(t, 2)): int(np.sum(np.isclose(last_entry, t)))
                                                   for t in sorted(set(np.round(last_entry, 2)))})
        print(K, json.dumps(res[f"track_K{K}"])[:600], flush=True)

    # (b) field swaps at NFE 20
    swaps = {}
    for tstar in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        for mode in ("learned_then_oracle", "oracle_then_learned"):
            if mode == "learned_then_oracle":
                f = lambda t, ts=tstar: v_learned if t < ts else v_oracle  # noqa: E731
            else:
                f = lambda t, ts=tstar: v_oracle if t < ts else v_learned  # noqa: E731
            xf, _ = solve(20, f)
            c = cats_of(xf)
            swaps[f"{mode}@{tstar}"] = dict(U=float((c == "U").mean()), M=float((c == "M").mean()),
                                            T=float((c == "T").mean()))
            print(mode, tstar, swaps[f"{mode}@{tstar}"], flush=True)
    res["swaps_K20"] = swaps

    # (c) learned vs exact predicted endpoint along the learned trajectory
    x = x0.clone()
    gap = []
    for k in range(20):
        t = k / 20
        vl, vo = v_learned(x, t), v_oracle(x, t)
        gap.append(dict(t=t, rms_x1hat=float(((1 - t) * (vl - vo)).pow(2).mean().sqrt())))
        x = x + vl / 20
    res["x1hat_gap"] = gap
    print(json.dumps(gap), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
print("done")

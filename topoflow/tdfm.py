"""TDFM pieces: representative route cost, Boltzmann target over codes, selector, end-to-end eval.

Code index convention: a route code is padded to MAX_OBS bits (empty slots 0) and indexed as
an integer with bit j = obstacle j (obstacles sorted by cut x). With MAX_OBS = 3 there are 8 indices.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn as nn

from .e2 import MAX_OBS, classify, code_ok_for_train, env_features, pad_code
from .routes import sample_route

N_CODES = 2 ** MAX_OBS


def code_index(code):
    return int(sum(int(b) << j for j, b in enumerate(code)))


def index_code(i, m):
    return tuple((i >> j) & 1 for j in range(m))


def structural_mask(m):
    """Codes that are well-formed for m obstacles (no bits set on empty slots). Not oracle info."""
    mask = np.zeros(N_CODES, dtype=bool)
    for c in itertools.product([0, 1], repeat=m):
        mask[code_index(c)] = True
    return mask


def path_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=-1).sum())


def representative_cost(env, code, samples, T=32):
    """Length of the minimum-offset route for `code`; falls back to the shortest found sample.

    Returns (cost, source) where source is 'rep' or 'fallback'.
    """
    rng = np.random.default_rng(0)
    for clearance in (0.04, 0.06, 0.08, 0.12):
        P = sample_route(env, code, rng, T=T, clearance=clearance, spread=0.0, wiggle=0.0)
        if not env.path_collides(P):
            return path_length(P), "rep"
    return min(path_length(p) for p in samples), "fallback"


def layout_costs(lay):
    """dict code -> cost for every found code; also counts of fallbacks."""
    env = lay["env"]
    costs, n_fb = {}, 0
    for code, P in lay["found"].items():
        c, src = representative_cost(env, code, P)
        costs[code] = c
        n_fb += src == "fallback"
    return costs, n_fb


def boltzmann(costs, beta):
    codes = list(costs)
    J = np.array([costs[c] for c in codes])
    z = -beta * (J - J.min())
    p = np.exp(z)
    p /= p.sum()
    return dict(zip(codes, p))


def normalized_entropy(p):
    v = np.array(list(p.values()))
    if len(v) < 2:
        return None
    v = v[v > 0]
    return float(-(v * np.log(v)).sum() / np.log(len(p)))


def mean_norm_entropy(cost_list, beta):
    hs = [normalized_entropy(boltzmann(c, beta)) for c in cost_list]
    hs = [h for h in hs if h is not None]
    return float(np.mean(hs))


def calibrate_beta(cost_list, target, lo=0.0, hi=2000.0, iters=60):
    """Bisection on the (monotone decreasing) mean normalized entropy."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mean_norm_entropy(cost_list, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def target_vector(p, m):
    v = np.zeros(N_CODES)
    for c, q in p.items():
        v[code_index(c)] = q
    return v


# ---------------------------------------------------------------- selector
class Selector(nn.Module):
    """MLP over padded obstacle features + mask -> 8 logits. Start/goal are fixed in E2 and omitted."""

    def __init__(self, in_dim=6 * MAX_OBS, hidden=256, depth=3):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, N_CODES))
        self.net = nn.Sequential(*layers)

    def forward(self, env_cond, allow_mask):
        logits = self.net(env_cond)
        return logits.masked_fill(~allow_mask, -1e9)


def selector_inputs(layouts):
    X, S = [], []
    for lay in layouts:
        f, m = env_features(lay["env"])
        X.append(np.concatenate([f, m]))
        S.append(structural_mask(lay["env"].m))
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(S))


def found_mask(layouts):
    M = np.zeros((len(layouts), N_CODES), dtype=bool)
    for i, lay in enumerate(layouts):
        for c in lay["found"]:
            M[i, code_index(c)] = True
    return torch.tensor(M)


def train_selector(X, S, Ptarget, steps=4000, batch=256, lr=1e-3, seed=0):
    """Soft-target cross-entropy (= KL up to a constant). Trained with structural mask only."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Selector()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        logp = torch.log_softmax(model(X[idx], S[idx]), -1)
        loss = -(Ptarget[idx] * logp.clamp_min(-1e4)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


# ---------------------------------------------------------------- end-to-end evaluation
def sample_codes(prob_rows, n, seed):
    """prob_rows: (L, 8) numpy; returns (L, n) code indices."""
    g = np.random.default_rng(seed)
    return np.stack([g.choice(N_CODES, size=n, p=p / p.sum()) for p in prob_rows])


@torch.no_grad()
def generate(model, layouts, code_idx, K, x0, T=32, batch=4096):
    """code_idx: (L, n) indices (ignored by code-blind models). x0: (L*n, 60) paired noise."""
    L, n = code_idx.shape
    conds = []
    for i, lay in enumerate(layouts):
        f, m = env_features(lay["env"])
        for j in range(n):
            conds.append(np.concatenate([f, m, pad_code(index_code(int(code_idx[i, j]), MAX_OBS))]))
    cond = torch.tensor(np.array(conds), dtype=torch.float32)
    outs = []
    for s in range(0, len(cond), batch):
        xb, cb = x0[s : s + batch].clone(), cond[s : s + batch]
        for k in range(K):
            xb = xb + model(xb, torch.full((len(xb),), k / K), cb) / K
        outs.append(xb)
    return torch.cat(outs).numpy().reshape(L, n, T - 2, 2)


def evaluate_samples(layouts, G, targets, costs, code_idx=None, T=32):
    """Metrics for generated samples G (L, n, T-2, 2) against per-layout targets (dict code->p).

    Outcome space per layout: found codes + 'invalid' (collision, or valid path whose word is
    outside the code family, or a code not in the found set).
    """
    out = dict(collision=[], deep=[], invalid=[], tv_joint=[], tv_valid=[], cost=[], coverage=[],
               held_frac=[], held_coll=[], notfound_sel=[])
    for i, lay in enumerate(layouts):
        env = lay["env"]
        P = np.empty((G.shape[1], T, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = G[i]
        cl, dp, codes = classify(env, P)
        tgt = targets[i]
        valid = [(not h) and (c is not None) and (c in tgt) for c, h in zip(codes, cl)]
        n = len(P)
        cnt = {c: 0 for c in tgt}
        for c, v in zip(codes, valid):
            if v:
                cnt[c] += 1
        inv = n - sum(cnt.values())
        out["collision"].append(cl.mean())
        out["deep"].append(dp.mean())
        out["invalid"].append(inv / n)
        out["tv_joint"].append(0.5 * (sum(abs(cnt[c] / n - tgt[c]) for c in tgt) + inv / n))
        nv = sum(cnt.values())
        if nv:
            out["tv_valid"].append(0.5 * sum(abs(cnt[c] / nv - tgt[c]) for c in tgt))
            out["cost"].append(sum(cnt[c] * costs[i][c] for c in tgt) / nv)
        support = [c for c in tgt if tgt[c] > 1e-3]
        out["coverage"].append(sum(cnt[c] > 0 for c in support) / len(support))
        if code_idx is not None:
            req = [index_code(int(k), env.m) for k in code_idx[i]]
            held = np.array([not code_ok_for_train(c) for c in req])
            out["held_frac"].append(held.mean())
            if held.any():
                out["held_coll"].append(cl[held].mean())
            out["notfound_sel"].append(np.mean([c not in lay["found"] for c in req]))
    return {k: (float(np.mean(v)) if len(v) else float("nan")) for k, v in out.items()}

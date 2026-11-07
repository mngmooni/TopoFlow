"""Batched evaluation of route-conditioned generators on E2 splits.

All models are evaluated on exactly the same (layout, code, noise) triples, so differences
between models are paired comparisons.
"""
from __future__ import annotations

import numpy as np
import torch

from .e2 import classify, code_ok_for_train, env_features, pad_code


def build_items(layouts, which):
    """which: 'trainable' (codes allowed in training) or 'held_out'."""
    items = []
    for li, lay in enumerate(layouts):
        for code in lay["found"]:
            ok = code_ok_for_train(code)
            if (which == "trainable" and ok) or (which == "held_out" and not ok):
                items.append((li, code))
    return items


@torch.no_grad()
def eval_conditioned(model, layouts, items, K, per_item=8, seed=0, T=32, batch=4096):
    conds = []
    for li, code in items:
        f, m = env_features(layouts[li]["env"])
        conds.append(np.concatenate([f, m, pad_code(code)]))
    cond = torch.tensor(np.repeat(np.array(conds), per_item, 0), dtype=torch.float32)
    n, dim = len(cond), (T - 2) * 2
    g = torch.Generator().manual_seed(seed)
    x0 = torch.randn(n, dim, generator=g)
    outs = []
    for s in range(0, n, batch):
        xb, cb = x0[s : s + batch].clone(), cond[s : s + batch]
        for k in range(K):
            tt = torch.full((len(xb),), k / K)
            xb = xb + model(xb, tt, cb) / K
        outs.append(xb)
    X = torch.cat(outs).numpy().reshape(len(items), per_item, T - 2, 2)

    rec = dict(n=0, match=0, valid_match=0, coll=0, deep=0)
    per_item_coll = []
    for (li, code), s in zip(items, X):
        env = layouts[li]["env"]
        P = np.empty((per_item, T, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = s
        cl, dp, codes = classify(env, P)
        match = np.array([c == code for c in codes])
        rec["n"] += per_item
        rec["match"] += int(match.sum())
        rec["valid_match"] += int((match & ~cl).sum())
        rec["coll"] += int(cl.sum())
        rec["deep"] += int(dp.sum())
        per_item_coll.append(cl.mean())
    n = rec["n"]
    return dict(n=n, adherence=rec["match"] / n, valid_adherence=rec["valid_match"] / n,
                collision=rec["coll"] / n, deep=rec["deep"] / n), np.array(per_item_coll)

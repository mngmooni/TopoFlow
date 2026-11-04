"""Selector variants for B2 (Day 8).

All variants take the same inputs (padded obstacle features + mask) and produce a distribution over
the 8 padded route codes. Feasibility is either the structural mask (no oracle info) combined with a
learned feasibility head, or the oracle found-mask.

  mlp      : logits head, soft-target cross-entropy to p*_beta (one model per beta)
  costreg  : predicts J(h) per code + feasibility logit; p = softmax(-beta * J_hat) over codes judged
             feasible. One model serves every beta.
  comp     : compositional. For each candidate code, obstacle tokens (features + that code's bit) go
             through a small set transformer and are pooled to a scalar; outputs J_hat and a feasibility
             logit per code, used like costreg.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .e2 import MAX_OBS
from .tdfm import N_CODES, code_index

ALL_BITS = torch.tensor([[(i >> j) & 1 for j in range(MAX_OBS)] for i in range(N_CODES)], dtype=torch.float32)


def mlp(in_dim, out_dim, hidden, depth):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class LogitSelector(nn.Module):
    def __init__(self, hidden=512, depth=4):
        super().__init__()
        self.net = mlp(6 * MAX_OBS, N_CODES, hidden, depth)

    def forward(self, x):
        return self.net(x)


class CostRegSelector(nn.Module):
    def __init__(self, hidden=512, depth=4, extra=0):
        super().__init__()
        self.net = mlp(6 * MAX_OBS + extra, 2 * N_CODES, hidden, depth)  # extra = 2 for (start_y, goal_y) in D3

    def forward(self, x):
        out = self.net(x)
        return out[:, :N_CODES], out[:, N_CODES:]  # J_hat (offset from a learned base), feas logit


class CompSelector(nn.Module):
    """Scores each candidate code with a set encoder over (obstacle feature, bit) tokens."""

    def __init__(self, d=128, depth=2, heads=4):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = mlp(d, 2, d, 2)

    def forward(self, x):
        B = x.shape[0]
        feat = x[:, : 5 * MAX_OBS].view(B, 1, MAX_OBS, 5).expand(B, N_CODES, MAX_OBS, 5)
        mask = x[:, 5 * MAX_OBS :].view(B, 1, MAX_OBS).expand(B, N_CODES, MAX_OBS)
        bits = ALL_BITS.view(1, N_CODES, MAX_OBS).expand(B, -1, -1)
        tok = self.tok(torch.cat([feat, bits[..., None]], -1)).reshape(B * N_CODES, MAX_OBS, -1)
        pad = (mask < 0.5).reshape(B * N_CODES, MAX_OBS)
        pad = pad.clone()
        pad[pad.all(1), 0] = False  # never fully masked (not hit in E2: >= 1 obstacle)
        h = self.enc(tok, src_key_padding_mask=pad)
        keep = (~pad).float()[..., None]
        pooled = (h * keep).sum(1) / keep.sum(1)
        out = self.head(pooled).view(B, N_CODES, 2)
        return out[..., 0], out[..., 1]


# ------------------------------------------------------------------ data
def cost_targets(cost_list, layouts):
    """J (L, 8) with NaN where not found; found mask (L, 8)."""
    J = np.full((len(cost_list), N_CODES), np.nan)
    for i, c in enumerate(cost_list):
        for code, v in c.items():
            J[i, code_index(code)] = v
    return J, ~np.isnan(J)


def probs_from_cost(J_hat, allow, beta):
    z = -beta * J_hat
    z = np.where(allow, z, -np.inf)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def train_costlike(model, X, S, J, F, steps=6000, batch=256, lr=1e-3, seed=0):
    """Masked regression on J (found codes only) + BCE feasibility (structurally valid codes only)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    Jt = torch.tensor(np.nan_to_num(J, nan=0.0), dtype=torch.float32)
    Ft = torch.tensor(F, dtype=torch.float32)
    St = S.float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        jh, fl = model(X[idx])
        reg = (((jh - Jt[idx]) ** 2) * Ft[idx]).sum() / Ft[idx].sum()
        bce = (nn.functional.binary_cross_entropy_with_logits(fl, Ft[idx], reduction="none") * St[idx]).sum() / St[idx].sum()
        loss = 100.0 * reg + bce
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()


def train_logits(model, X, S, P, steps=6000, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=g)
        logp = torch.log_softmax(model(X[idx]).masked_fill(~S[idx], -1e9), -1)
        loss = -(P[idx] * logp.clamp_min(-1e4)).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()

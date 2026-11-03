"""From-scratch Conditional Flow Matching (linear path, Gaussian prior) + exact oracle field.

Path:      X_t = (1 - t) X_0 + t X_1,   X_0 ~ N(0, I),  X_1 ~ data
Target:    u_t(x | x_1) = x_1 - x_0
Oracle:    v*(x, t) = sum_i w_i(x, t) (x_i - x) / (1 - t),
           w_i ∝ exp(-||x - t x_i||^2 / (2 (1 - t)^2))      (empirical data, t < 1)
Sampler:   Euler on t_k = k / K, k = 0..K-1 (the field is never evaluated at t = 1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TimeEmbed(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t.device))
        ang = t[:, None] * freqs[None] * 2 * math.pi
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class VelocityMLP(nn.Module):
    def __init__(self, dim, cond_dim=0, hidden=256, depth=4, t_dim=32):
        super().__init__()
        self.temb = TimeEmbed(t_dim)
        layers, d = [], dim + t_dim + cond_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, dim))
        self.net = nn.Sequential(*layers)
        self.cond_dim = cond_dim

    def forward(self, x, t, cond=None):
        h = [x, self.temb(t)]
        if self.cond_dim:
            h.append(cond)
        return self.net(torch.cat(h, dim=1))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def cfm_loss(model, x1, cond=None):
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), device=x1.device)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1
    return ((model(xt, t, cond) - (x1 - x0)) ** 2).mean()


def train_cfm(model, data, cond=None, steps=15000, batch=256, lr=1e-3, seed=0, log_every=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(data), (batch,), generator=g)
        loss = cfm_loss(model, data[idx], None if cond is None else cond[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}: loss {sum(losses[-log_every:]) / log_every:.4f}")
    return losses


@torch.no_grad()
def oracle_velocity(x, t, data, chunk=4096):
    """Exact marginal velocity for the empirical distribution of `data` (n, d). t: float < 1."""
    s = 1.0 - t
    out = []
    for xb in x.split(chunk):
        d2 = torch.cdist(xb, t * data) ** 2                 # (B, n)
        w = torch.softmax(-d2 / (2 * s * s), dim=1)          # log-space normalization
        out.append((w @ data - xb) / s)
    return torch.cat(out)


@torch.no_grad()
def euler_sample(field, n, dim, K, seed=0, cond=None):
    """field(x, t_tensor, cond) -> velocity. Returns final samples (n, dim)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    for k in range(K):
        t = k / K
        tt = torch.full((n,), t)
        x = x + field(x, tt, cond) / K
    return x

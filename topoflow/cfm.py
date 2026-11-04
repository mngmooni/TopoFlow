"""From-scratch Conditional Flow Matching (linear path, Gaussian prior) + exact oracle field.

Path:      X_t = (1 - t) X_0 + t X_1,   X_0 ~ N(0, I),  X_1 ~ data
Target:    u_t(x | x_1) = x_1 - x_0
Oracle:    v*(x, t) = sum_i w_i(x, t) (x_i - x) / (1 - t),
           w_i ∝ exp(-||x - t x_i||^2 / (2 (1 - t)^2))      (empirical data, t < 1)
Sampler:   Euler on t_k = k / K, k = 0..K-1 (the field is never evaluated at t = 1).
"""
from __future__ import annotations

import math

import numpy as np

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


CKPT_EVERY = 2000
# TRAIN_CHUNK=n: stop after n steps of this process (checkpoint saved), exit code 3, so long runs can be
# split into launches that each fit a wall-clock limit. Re-running the same command resumes.
TRAIN_CHUNK = int(__import__("os").environ.get("TRAIN_CHUNK", "0"))
CHUNK_EXIT = 3


def _ckpt_load(path, model, opt, sched, g, rng=None):
    """Resume state if a checkpoint exists. Returns (start_step, losses)."""
    import os
    if not path or not os.path.exists(path):
        return 0, []
    st = torch.load(path, weights_only=False)
    model.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    sched.load_state_dict(st["sched"])
    g.set_state(st["g"])
    torch.set_rng_state(st["torch_rng"])
    if rng is not None:
        rng.bit_generator.state = st["np_rng"]
    print(f"  resumed from step {st['step']}", flush=True)
    return st["step"], st["losses"]


def _ckpt_save(path, step, model, opt, sched, g, losses, rng=None):
    import os
    st = dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
              g=g.get_state(), torch_rng=torch.get_rng_state(), losses=losses,
              np_rng=None if rng is None else rng.bit_generator.state)
    torch.save(st, path + ".tmp")
    os.replace(path + ".tmp", path)


def train_cfm(model, data, cond=None, steps=15000, batch=256, lr=1e-3, seed=0, log_every=0, weights=None,
              ckpt=None):
    """weights: optional per-sample sampling weights (need not be normalized)."""
    g = torch.Generator().manual_seed(seed)
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=torch.float64)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start, losses = _ckpt_load(ckpt, model, opt, sched, g)
    for step in range(start, steps):
        if ckpt and TRAIN_CHUNK and step - start >= TRAIN_CHUNK:
            _ckpt_save(ckpt, step, model, opt, sched, g, losses)
            print(f"  chunk done at step {step}/{steps}; rerun to resume", flush=True)
            raise SystemExit(CHUNK_EXIT)
        if ckpt and step > start and step % CKPT_EVERY == 0:
            _ckpt_save(ckpt, step, model, opt, sched, g, losses)
        if weights is None:
            idx = torch.randint(0, len(data), (batch,), generator=g)
        else:
            idx = torch.multinomial(weights, batch, replacement=True, generator=g)
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


def train_cfm_ot(model, data, cond, layout_ids, weights, steps=30000, groups=16, per_group=16,
                 lr=1e-3, seed=0, log_every=0, ckpt=None):
    """Conditional minibatch OT-CFM.

    Each step draws `groups` layouts uniformly, then `per_group` samples per layout with the given
    within-layout weights. Noise and data are paired by an exact assignment (squared Euclidean)
    **within each layout only**, so the coupling never mixes conditions. Batch = groups * per_group.
    """
    from scipy.optimize import linear_sum_assignment

    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    layout_ids = np.asarray(layout_ids)
    weights = np.asarray(weights, dtype=np.float64)
    uniq = np.unique(layout_ids)
    members = {l: np.where(layout_ids == l)[0] for l in uniq}
    probs = {l: weights[idx] / weights[idx].sum() for l, idx in members.items()}
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start, losses = _ckpt_load(ckpt, model, opt, sched, g, rng)
    for step in range(start, steps):
        if ckpt and TRAIN_CHUNK and step - start >= TRAIN_CHUNK:
            _ckpt_save(ckpt, step, model, opt, sched, g, losses, rng)
            print(f"  chunk done at step {step}/{steps}; rerun to resume", flush=True)
            raise SystemExit(CHUNK_EXIT)
        if ckpt and step > start and step % CKPT_EVERY == 0:
            _ckpt_save(ckpt, step, model, opt, sched, g, losses, rng)
        idx_all, x0_all = [], []
        for l in rng.choice(uniq, size=groups, replace=True):
            idx = rng.choice(members[l], size=per_group, replace=True, p=probs[l])
            x1 = data[idx]
            x0 = torch.randn(per_group, data.shape[1], generator=g)
            cost = torch.cdist(x0, x1).pow(2).numpy()
            r, c = linear_sum_assignment(cost)
            idx_all.append(idx[c[np.argsort(r)]])
            x0_all.append(x0)
        idx = np.concatenate(idx_all)
        x0 = torch.cat(x0_all)
        x1 = data[idx]
        t = torch.rand(len(x1), generator=g)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((model(xt, t, cond[idx]) - (x1 - x0)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}: loss {sum(losses[-log_every:]) / log_every:.4f}", flush=True)
    return losses

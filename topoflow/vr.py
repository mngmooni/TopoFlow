"""TDFM v2-VR: the class-chart flow on a variable-resolution grid .

Same chart and decoder as v2 (topoflow/chart.py, Prop. 1 holds for any grid containing the inflated edges).
Changes from v2 (ChartField):
  * one token per grid point (v2: 6 grid values per token, fixed 10 tokens) with features (u, lo, hi, x, dx_left,
    dx_right); position enters through Fourier features of x instead of a token-index embedding;
  * key-padding mask in self-attention, so a batch can mix grids of different length (<= NMAX);
  * loss masked to real grid points;
  * optional resolution-consistent prior: x0 = Ornstein-Uhlenbeck process sampled at the grid's x
    (cov exp(-|x - x'| / ELL)), which has the same law at every resolution; "iid" = v2's N(0, I).
cond layout (flat): [feat 5M, mask M, code M, lo NMAX, hi NMAX, xs NMAX, valid NMAX]
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn as nn

from . import chart
from .cfm import CHUNK_EXIT, TRAIN_CHUNK, TimeEmbed, _ckpt_load, _ckpt_save
from .e2 import env_features, pad_code
from .p1 import Block

NMAX = 120
ELL = 0.03


class MaskedBlock(Block):
    def forward(self, h, obs, obs_pad, self_pad=None):
        a = self.n1(h)
        h = h + self.self_attn(a, a, a, key_padding_mask=self_pad, need_weights=False)[0]
        a = self.n2(h)
        h = h + self.cross_attn(a, obs, obs, key_padding_mask=obs_pad, need_weights=False)[0]
        return h + self.mlp(self.n3(h))


class ChartFieldVR(nn.Module):
    def __init__(self, max_obs=5, d=128, depth=4, heads=4, mlp_ratio=2, t_dim=32, n_freq=16, nmax=NMAX):
        super().__init__()
        self.max_obs, self.nmax = max_obs, nmax
        self.temb = TimeEmbed(t_dim)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(200.0), n_freq)))
        self.pt_in = nn.Linear(6 + 2 * n_freq, d)
        self.obs_in = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.null_obs = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([MaskedBlock(d, heads, mlp_ratio) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def split(self, cond):
        B, M = cond.shape[0], self.max_obs
        n = (cond.shape[1] - 7 * M) // 4  # grid width of this batch (<= nmax; batches are trimmed)
        feat = cond[:, : 5 * M].view(B, M, 5)
        mask = cond[:, 5 * M: 6 * M]
        code = cond[:, 6 * M: 7 * M]
        lo, hi, xs, valid = (cond[:, 7 * M + k * n: 7 * M + (k + 1) * n] for k in range(4))
        return feat, mask, code, lo, hi, xs, valid

    def forward(self, u, t, cond):
        feat, mask, code, lo, hi, xs, valid = self.split(cond)
        B = u.shape[0]
        obs = torch.cat([self.null_obs.expand(B, 1, -1), self.obs_in(torch.cat([feat, code[..., None]], -1))], 1)
        opad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask < 0.5], 1)
        dxl = torch.diff(xs, dim=1, prepend=xs[:, :1])
        dxr = torch.diff(xs, dim=1, append=xs[:, -1:])
        ang = xs[..., None] * self.freqs * math.pi
        tok = torch.cat([torch.stack([u, lo, hi, xs, dxl * 30, dxr * 30], -1), torch.sin(ang), torch.cos(ang)], -1)
        h = self.pt_in(tok) + self.t_proj(self.temb(t))[:, None]
        spad = valid < 0.5
        for blk in self.blocks:
            h = blk(h, obs, opad, spad)
        return self.out(h).squeeze(-1) * valid


class PerceiverVR(nn.Module):
    """Resolution-independent chart field, Perceiver-IO style: grid points -> L latent tokens (cross-attention),
    latent blocks (self-attention + obstacle cross-attention, as v2/P1), latents -> each grid point (cross-attention
    + MLP) -> scalar velocity. Compute per point is O(L), so cost grows linearly with the grid."""

    def __init__(self, max_obs=5, d=128, depth=3, heads=4, mlp_ratio=2, t_dim=32, n_freq=16, n_lat=10, nmax=NMAX):
        super().__init__()
        self.max_obs, self.nmax = max_obs, nmax
        self.temb = TimeEmbed(t_dim)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer("freqs", torch.exp(torch.linspace(0, math.log(200.0), n_freq)))
        self.pt_in = nn.Linear(6 + 2 * n_freq, d)
        self.lat = nn.Parameter(torch.randn(n_lat, d) * 0.02)
        self.enc_n = nn.LayerNorm(d)
        self.enc = nn.MultiheadAttention(d, heads, batch_first=True)
        self.obs_in = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.null_obs = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.dec_nq, self.dec_nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dec = nn.MultiheadAttention(d, heads, batch_first=True)
        self.dec_n2 = nn.LayerNorm(d)
        self.dec_mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.SiLU(), nn.Linear(mlp_ratio * d, d))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    split = ChartFieldVR.split

    def forward(self, u, t, cond):
        feat, mask, code, lo, hi, xs, valid = self.split(cond)
        B = u.shape[0]
        obs = torch.cat([self.null_obs.expand(B, 1, -1), self.obs_in(torch.cat([feat, code[..., None]], -1))], 1)
        opad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask < 0.5], 1)
        dxl = torch.diff(xs, dim=1, prepend=xs[:, :1])
        dxr = torch.diff(xs, dim=1, append=xs[:, -1:])
        ang = xs[..., None] * self.freqs * math.pi
        tok = torch.cat([torch.stack([u, lo, hi, xs, dxl * 30, dxr * 30], -1), torch.sin(ang), torch.cos(ang)], -1)
        te = self.t_proj(self.temb(t))[:, None]
        p = self.pt_in(tok) + te
        spad = valid < 0.5
        h = self.lat[None].expand(B, -1, -1) + te
        h = h + self.enc(self.enc_n(h), p, p, key_padding_mask=spad, need_weights=False)[0]
        for blk in self.blocks:
            h = blk(h, obs, opad)
        k = self.dec_nk(h)
        q = p + self.dec(self.dec_nq(p), k, k, need_weights=False)[0]
        q = q + self.dec_mlp(self.dec_n2(q))
        return self.out(q).squeeze(-1) * valid


# ------------------------------------------------------------------ prior
def prior(xs, valid, kind, gen=None):
    """x0 for a batch of grids. kind: 'iid' or 'ou' (exp(-|dx|/ELL) covariance at the grid points)."""
    z = torch.randn(xs.shape, generator=gen)
    if kind == "iid":
        return z * valid
    # OU process sampled at increasing x: exact AR(1) recursion with step-dependent coefficient
    rho = torch.exp(-torch.diff(xs, dim=1).clamp_min(0) / ELL)
    x0 = torch.empty_like(z)
    x0[:, 0] = z[:, 0]
    for k in range(1, xs.shape[1]):
        r = rho[:, k - 1]
        x0[:, k] = r * x0[:, k - 1] + torch.sqrt(1 - r * r) * z[:, k]
    return x0 * valid


# ------------------------------------------------------------------ data
def random_grid(env, c, rng, extra_max=40, base=None):
    xs = chart.grid(env, 60, c) if base is None else base
    k = int(rng.integers(1, extra_max + 1))
    new = rng.uniform(env.start[0], env.goal[0], size=k)
    return np.unique(np.concatenate([xs, new]))


def refined_grid(env, code, c, maxp=NMAX - 40):
    xs = chart.grid(env, 60, c)
    for _ in range(300):
        lo, hi, ok, _, _ = chart.bounds_xs(env, code, c, xs)
        if ok:
            return xs
        if len(xs) >= maxp:
            return None
        full = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
        new = set()
        for k in np.where(hi <= lo)[0] + 1:
            new |= {(full[k - 1] + full[k]) / 2, (full[k] + full[k + 1]) / 2}
        xs = np.unique(np.concatenate([xs, list(new)]))
    return None


def cond_row(env, code, c, xs, nmax=NMAX):
    lo, hi, ok, _, _ = chart.bounds_xs(env, code, c, xs)
    n = len(xs)
    f, m = env_features(env)
    pad = lambda a, v: np.concatenate([a, np.full(nmax - n, v)])  # noqa: E731
    row = np.concatenate([f, m, pad_code(code), pad(lo, -1.0), pad(hi, 1.0), pad(xs, xs[-1]), pad(np.ones(n), 0.0)])
    return row.astype(np.float32), lo, hi, ok


def training_set(layouts, c, rng, p_refine=0.5, nmax=NMAX):
    """One grid per reference path: default n = 60 (prob 1 - p_refine) or default + 1..40 random points; classes
    infeasible at n = 60 use their bisection grid (+ random points with prob p_refine)."""
    U, Cnd, info = [], [], dict(paths=0, refine_only=0, clipped=0, skipped=0)
    for lay in layouts:
        env = lay["env"]
        for code, P in lay["found"].items():
            base = refined_grid(env, code, c)
            if base is None:
                info["skipped"] += len(P)
                continue
            only_ref = len(base) != len(chart.grid(env, 60, c))
            for p in P:
                xs = random_grid(env, c, rng, base=base) if rng.random() < p_refine else base
                if len(xs) > nmax:
                    xs = base
                row, lo, hi, ok = cond_row(env, code, c, xs, nmax)
                if not ok:  # random extra points cannot break feasibility (refinement only shrinks bands)
                    info["skipped"] += 1
                    continue
                y = np.interp(xs, p[:, 0], p[:, 1])
                info["clipped"] += not np.all((y >= lo) & (y <= hi))
                u = np.zeros(nmax, np.float32)
                u[: len(xs)] = chart.encode(y, lo, hi)
                U.append(u)
                Cnd.append(row)
                info["paths"] += 1
                info["refine_only"] += only_ref
    return np.array(U), np.array(Cnd), info


# ------------------------------------------------------------------ training / sampling
def trim(u, cond, M, n):
    """Cut a batch to its longest real grid (saves compute; padding beyond it is masked anyway)."""
    valid = cond[:, 7 * M + 3 * n: 7 * M + 4 * n]
    nb = int(valid.sum(1).max().item())
    parts = [cond[:, : 7 * M]] + [cond[:, 7 * M + k * n: 7 * M + k * n + nb] for k in range(4)]
    return u[:, :nb], torch.cat(parts, 1), nb


def train_vr(model, U, Cnd, steps=30000, batch=256, lr=1e-3, seed=0, prior_kind="iid", ckpt=None, log_every=10000):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    start, losses = _ckpt_load(ckpt, model, opt, sched, g)
    M, n = model.max_obs, model.nmax
    for step in range(start, steps):
        if ckpt and TRAIN_CHUNK and step - start >= TRAIN_CHUNK:
            _ckpt_save(ckpt, step, model, opt, sched, g, losses)
            print(f"  chunk done at step {step}/{steps}; rerun to resume", flush=True)
            raise SystemExit(CHUNK_EXIT)
        if ckpt and step > start and step % 2000 == 0:
            _ckpt_save(ckpt, step, model, opt, sched, g, losses)
        idx = torch.randint(0, len(U), (batch,), generator=g)
        x1, cond, nb = trim(U[idx], Cnd[idx], M, n)
        xs = cond[:, 7 * M + 2 * nb: 7 * M + 3 * nb]
        valid = cond[:, 7 * M + 3 * nb: 7 * M + 4 * nb]
        x0 = prior(xs, valid, prior_kind, g)
        t = torch.rand(batch, generator=g)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = model(xt, t, cond)
        loss = (((v - (x1 - x0)) ** 2) * valid).sum() / valid.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  step {step + 1}: loss {np.mean(losses[-log_every:]):.4f}", flush=True)
    return losses


@torch.no_grad()
def sample(model, rows, K, prior_kind="iid", seed=0, batch=2048):
    """rows: list of (env, code, c, xs). Returns list of decoded polylines (len(xs) + 2, 2)."""
    M, n = model.max_obs, model.nmax
    out = []
    g = torch.Generator().manual_seed(seed)
    for s in range(0, len(rows), batch):
        chunk = rows[s:s + batch]
        built = [cond_row(env, code, c, xs, n) for env, code, c, xs in chunk]
        cond = torch.tensor(np.stack([b[0] for b in built]))
        _, cond, nb = trim(torch.zeros(len(cond), n), cond, M, n)
        xs_t = cond[:, 7 * M + 2 * nb: 7 * M + 3 * nb]
        valid = cond[:, 7 * M + 3 * nb: 7 * M + 4 * nb]
        u = prior(xs_t, valid, prior_kind, g)
        for k in range(K):
            u = u + model(u, torch.full((len(u),), k / K), cond) / K
        u = u.numpy()
        for j, ((env, code, c, xs), (_, lo, hi, ok)) in enumerate(zip(chunk, built)):
            y = chart.decode(u[j, : len(xs)], lo, hi)
            P = np.concatenate([[env.start], np.stack([xs, y], 1), [env.goal]])
            out.append(P)
    return out

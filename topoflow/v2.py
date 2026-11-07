"""TDFM v2: Topology-Constrained Residual Flow.

The flow lives in the class chart (topoflow/chart.py): x = u in R^n, decoded as y = m + w tanh(u) inside the free band
of the requested route code. The backbone is P1's (same blocks, obstacle tokens with route bits, 10 waypoint
tokens); each waypoint token holds `chunk` grid values and, per value, (u, lo, hi, x).

cond layout: [feat (5M), mask (M), code (M), lo (n), hi (n), xs (n), (start_y, goal_y) if sg_input]
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import chart
from .cfm import TimeEmbed
from .e2 import env_features, pad_code
from .p1 import Block


class ChartField(nn.Module):
    def __init__(self, n=60, max_obs=5, d=128, depth=4, heads=4, mlp_ratio=2, t_dim=32, chunk=6, sg_input=False):
        super().__init__()
        self.sg_input = sg_input
        if sg_input:  # same start/goal embedding as P1's sg_input (D3)
            self.sg_proj = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        assert n % chunk == 0
        self.n, self.max_obs, self.chunk = n, max_obs, chunk
        self.n_tok = n // chunk
        self.temb = TimeEmbed(t_dim)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.idx_emb = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        self.way_in = nn.Linear(4 * chunk, d)
        self.obs_in = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.null_obs = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, chunk))

    def forward(self, u, t, cond):
        B, M, n = u.shape[0], self.max_obs, self.n
        feat = cond[:, : 5 * M].view(B, M, 5)
        mask = cond[:, 5 * M: 6 * M]
        code = cond[:, 6 * M: 7 * M]
        lo, hi, xs = (cond[:, 7 * M + k * n: 7 * M + (k + 1) * n] for k in range(3))
        obs = self.obs_in(torch.cat([feat, code[..., None]], -1))
        obs = torch.cat([self.null_obs.expand(B, 1, -1), obs], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool), mask < 0.5], 1)
        tok = torch.stack([u, lo, hi, xs], -1).view(B, self.n_tok, self.chunk * 4)
        h = self.way_in(tok) + self.idx_emb[None] + self.t_proj(self.temb(t))[:, None]
        if self.sg_input:
            h = h + self.sg_proj(cond[:, 7 * M + 3 * n: 7 * M + 3 * n + 2])[:, None]
        for blk in self.blocks:
            h = blk(h, obs, pad)
        return self.out(h).reshape(B, n)


class ChartCache:
    """Per-(layout, code) chart bounds and grid, computed once."""

    def __init__(self, layouts, c, n=60, sg=False):
        self.layouts, self.c, self.n, self.sg = layouts, c, n, sg
        self._b = {}

    def get(self, i, code):
        key = (i, tuple(code))
        if key not in self._b:
            env = self.layouts[i]["env"]
            lo, hi, ok = chart.bounds(env, code, self.c, self.n)
            self._b[key] = (lo, hi, chart.grid(env, self.n, self.c), ok)
        return self._b[key]

    def cond(self, i, code):
        env = self.layouts[i]["env"]
        f, m = env_features(env)
        lo, hi, xs, _ = self.get(i, code)
        extra = [env.start[1:2], env.goal[1:2]] if self.sg else []
        return np.concatenate([f, m, pad_code(code), lo, hi, xs, *extra]).astype(np.float32)


def training_set(layouts, c, n=60, sg=False):
    """u targets and cond rows for every reference path; also returns the clipped fraction."""
    cache = ChartCache(layouts, c, n, sg)
    U, Cnd, clipped = [], [], 0
    for i, lay in enumerate(layouts):
        env = lay["env"]
        for code, P in lay["found"].items():
            lo, hi, xs, ok = cache.get(i, code)
            if not ok:
                continue
            row = cache.cond(i, code)
            for p in P:
                y = chart.to_grid(env, p, n, c)
                clipped += not np.all((y >= lo) & (y <= hi))
                U.append(chart.encode(y, lo, hi))
                Cnd.append(row)
    return np.array(U, np.float32), np.array(Cnd, np.float32), clipped / max(len(U), 1)


@torch.no_grad()
def sample(model, cache, req, K, x0, batch=4096):
    """req: list of (layout index, code) per sample; x0: (N, n) noise. Returns decoded polylines (N, n + 2, 2)."""
    cond = torch.tensor(np.stack([cache.cond(i, c) for i, c in req]))
    us = []
    for s in range(0, len(cond), batch):
        u, cb = x0[s:s + batch].clone(), cond[s:s + batch]
        for k in range(K):
            u = u + model(u, torch.full((len(u),), k / K), cb) / K
        us.append(u)
    u = torch.cat(us).numpy()
    out = []
    for j, (i, c) in enumerate(req):
        lo, hi, xs, _ = cache.get(i, c)
        out.append(chart.to_path(cache.layouts[i]["env"], chart.decode(u[j], lo, hi), cache.c))
    return np.stack(out)

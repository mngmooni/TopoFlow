"""P1: obstacle-token cross-attention velocity field.

Same input interface as the padded-MLP B1: cond = [feat (MAX_OBS*5), mask (MAX_OBS), code (MAX_OBS)].
Only the way that information is processed differs:

  waypoint tokens   `chunk` consecutive waypoints (x, y, ...) + index emb + time emb -> d
  obstacle tokens   (kind, cx, cy, half_w, half_h, route bit)        -> d   (no order embedding)
  L x [ self-attention over waypoints
        cross-attention waypoints -> obstacle tokens (empty slots masked)
        MLP ]
  per-token linear head -> 2D velocity for each waypoint in the chunk

Chunking (default 3 waypoints/token, 10 tokens) is purely for CPU cost.

Obstacle tokens carry no positional/slot embedding, so the field is invariant to the
order of obstacles (tested in tests/test_p1.py). Inductive bias: each path segment can
read the bits of the obstacles it is near, so a new *combination* of bits is locally a
familiar situation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cfm import TimeEmbed


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.SiLU(), nn.Linear(mlp_ratio * d, d))

    def forward(self, h, obs, obs_pad):
        a = self.n1(h)
        h = h + self.self_attn(a, a, a, need_weights=False)[0]
        a = self.n2(h)
        h = h + self.cross_attn(a, obs, obs, key_padding_mask=obs_pad, need_weights=False)[0]
        return h + self.mlp(self.n3(h))


class ObstacleAttnField(nn.Module):
    def __init__(self, n_way, max_obs, d=128, depth=4, heads=4, mlp_ratio=2, t_dim=32, chunk=3, use_code=True):
        super().__init__()
        self.use_code = use_code  # False -> B0-attn: route bits are zeroed, identical backbone
        assert n_way % chunk == 0
        self.n_way, self.max_obs, self.chunk = n_way, max_obs, chunk
        self.n_tok = n_way // chunk
        self.temb = TimeEmbed(t_dim)
        self.t_proj = nn.Sequential(nn.Linear(t_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.idx_emb = nn.Parameter(torch.randn(self.n_tok, d) * 0.02)
        self.way_in = nn.Linear(2 * chunk, d)
        self.obs_in = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.null_obs = nn.Parameter(torch.zeros(1, 1, d))  # always-present token, so 0-obstacle rows stay defined
        self.blocks = nn.ModuleList([Block(d, heads, mlp_ratio) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * chunk))

    def forward(self, x, t, cond):
        B = x.shape[0]
        M = self.max_obs
        feat = cond[:, : 5 * M].view(B, M, 5)
        mask = cond[:, 5 * M : 6 * M]
        code = cond[:, 6 * M : 7 * M]
        if not self.use_code:
            code = torch.zeros_like(code)
        obs = self.obs_in(torch.cat([feat, code[..., None]], -1))
        obs = torch.cat([self.null_obs.expand(B, 1, -1), obs], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), mask < 0.5], 1)
        h = self.way_in(x.view(B, self.n_tok, 2 * self.chunk)) + self.idx_emb[None] + self.t_proj(self.temb(t))[:, None]
        for blk in self.blocks:
            h = blk(h, obs, pad)
        return self.out(h).reshape(B, -1)

"""Packed (tight x-gap) layouts used by Day 12 hard family H2 and the D4 dataset."""
import itertools

import numpy as np

from .env import Box, Circle, Env


def boundary(o, k=96):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    if o.kind == "circle":
        return np.stack([o.cx + o.r * np.cos(t), o.cy + o.r * np.sin(t)], 1)
    s = np.linspace(-1, 1, k // 4)
    return np.concatenate([np.stack([o.cx + o.hw * s, np.full_like(s, o.cy + sg * o.hh)], 1) for sg in (-1, 1)] +
                          [np.stack([np.full_like(s, o.cx + sg * o.hw), o.cy + o.hh * s], 1) for sg in (-1, 1)])


def apart(a, b, d):
    return not (any(b.contains(p, margin=d) for p in boundary(a)) or any(a.contains(p, margin=d) for p in boundary(b))
                or a.contains(np.array([b.cx, b.cy])) or b.contains(np.array([a.cx, a.cy])))


def packed_layout(rng, m, r=(0.12, 0.2), box=0.3, y=(-0.35, 0.35), c=0.01, gap=(0.0, 0.04), min_dist=0.02,
                  start=(-0.9, 0.0), goal=(0.9, 0.0)):
    """Obstacles left to right with x-gaps ~ U[gap]; inflated ranges stay inside the start/goal x-span."""
    half = 0.9 - c - 0.005
    for _ in range(20000):
        sizes = rng.uniform(*r, size=m)
        kinds = rng.random(m) < box
        hws = [s * rng.uniform(0.7, 1.0) if k else s for s, k in zip(sizes, kinds)]
        gaps = rng.uniform(*gap, size=m - 1)
        room = 2 * half - (2 * sum(hws) + gaps.sum())
        if room <= 0:
            continue
        x = -half + rng.uniform(0, room)
        obs = []
        for k in range(m):
            cx = x + hws[k]
            cy = rng.uniform(*y)
            obs.append(Box(cx, cy, hws[k], sizes[k] * rng.uniform(0.7, 1.3)) if kinds[k] else Circle(cx, cy, hws[k]))
            x = cx + hws[k] + (gaps[k] if k < m - 1 else 0)
        env = Env(obs, start, goal)
        if all(apart(a, b, min_dist) for a, b in itertools.combinations(env.obstacles, 2)):
            return env
    raise RuntimeError("no layout")

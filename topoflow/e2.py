"""E2: many random layouts (1-3 obstacles), fixed start/goal, padded features.

Per-sample model inputs (MAX_OBS = 3, obstacles sorted by cut x):
  env_feat : (MAX_OBS * 5,)  [kind, cx, cy, half_w, half_h] per slot, zeros if empty
  env_mask : (MAX_OBS,)      1 if the slot holds an obstacle
  code     : (MAX_OBS,)      route bits (0 above / 1 below), 0 in empty slots
Target: interior waypoints (T-2)*2.

Splits
  train            : train layouts, codes except HELD_OUT on 3-obstacle layouts
  seen             : same train layouts, fresh samples (reproduction)
  unseen_pos       : new layouts, codes that were trainable
  unseen_combo     : new 3-obstacle layouts, HELD_OUT codes only
"""
from __future__ import annotations

import itertools
import os

import numpy as np

from .env import random_layout
from .routes import generate_routes
from .topology import crossing_word

MAX_OBS = int(os.environ.get("TOPO_MAX_OBS", "3"))  # D1+ experiments use 5
HELD_OUT = {(0, 1, 0), (1, 0, 1)}  # alternating patterns never trained on 3-obstacle layouts


def code_ok_for_train(code):
    return not (len(code) == 3 and tuple(code) in HELD_OUT)


def sample_layout(rng, box_prob=0.3):
    n_obs = int(rng.choice([1, 2, 3], p=[0.2, 0.35, 0.45]))
    return random_layout(rng, n_obs, box_prob=box_prob)


def env_features(env):
    arr, mask = env.obstacle_array(MAX_OBS)
    return arr.reshape(-1), mask.astype(np.float32)


def pad_code(code):
    c = np.zeros(MAX_OBS, dtype=np.float32)
    c[: len(code)] = code
    return c


def word_to_code(word, m):
    """Map an x-monotone-family word back to a code; None if the word is outside the family."""
    if any(x < 0 for x in word) or list(word) != sorted(set(word)) or any(x > m for x in word):
        return None
    c = [0] * m
    for x in word:
        c[x - 1] = 1
    return tuple(c)


def build_layouts(rng, n_layouts, per_code, T=32, margin=0.03, max_tries_per_path=40, sampler=None):
    """Returns list of dicts: env, found codes, paths per code."""
    out = []
    for _ in range(n_layouts):
        env = (sampler or sample_layout)(rng)
        found = {}
        for code in itertools.product([0, 1], repeat=env.m):
            P, _, _ = generate_routes(env, code, per_code, rng, T=T, margin=margin,
                                      max_tries_per_path=max_tries_per_path)
            if len(P):
                found[code] = P
        out.append(dict(env=env, found=found))
    return out


def stack_samples(layouts, code_filter=None):
    """Flatten layouts into arrays. code_filter(code) -> bool selects which codes to include."""
    X, F, M, C, L = [], [], [], [], []
    for li, lay in enumerate(layouts):
        f, m = env_features(lay["env"])
        for code, P in lay["found"].items():
            if code_filter is not None and not code_filter(code):
                continue
            for p in P:
                X.append(p[1:-1].reshape(-1))
                F.append(f)
                M.append(m)
                C.append(pad_code(code))
                L.append(li)
    return dict(x=np.array(X, np.float32), feat=np.array(F, np.float32), mask=np.array(M, np.float32),
                code=np.array(C, np.float32), layout=np.array(L))


def classify(env, P, deep_thresh=1 / 3):
    """Per path: valid flag, recovered code (or None), collision flag, deep-collision flag."""
    depth = env.penetration(P, relative=True)
    coll = env.collision_mask(P)
    deep = depth > deep_thresh
    codes = [word_to_code(crossing_word(env, p), env.m) for p in P]
    return coll, deep, codes

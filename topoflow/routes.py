"""Route-family generation, fixed-length resampling, and dataset building.

Route family for this sprint: x-monotone paths from start to goal that pass
each obstacle either above (code 0) or below (code 1). Every generated path
is checked on its *final* T-waypoint representation (the thing the model will
see) for collision and for crossing-word agreement with its route code.

A code for which no valid path is found within the try budget is reported as
'not_found', never as 'infeasible'.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .topology import crossing_word, expected_word_for_code, word_to_str, code_to_str


def resample_arclength(path, T):
    """Resample a polyline to T points equally spaced in arc length (endpoints kept)."""
    path = np.asarray(path, dtype=float)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], T)
    return np.stack([np.interp(target, s, path[:, k]) for k in range(2)], axis=1)


def sample_route(env, code, rng, T=32, clearance=0.04, spread=0.15, wiggle=0.05,
                 n_dense=400, smooth_sigma=3.0):
    """One candidate x-monotone path for a route code (not yet validated)."""
    sx, sy = env.start
    gx, gy = env.goal
    kx = [sx]
    ky = [sy]
    obs = env.obstacles
    # x-limits of the free gaps around each obstacle (start / neighbours / goal)
    lefts = [sx] + [o.xrange[1] for o in obs[:-1]]
    rights = [o.xrange[0] for o in obs[1:]] + [gx]
    for o, b, lft, rgt in zip(obs, code, lefts, rights):
        side = -1.0 if b else 1.0
        offset = o.half_height + clearance + rng.uniform(0.0, spread)
        y = o.cy + side * offset
        x0, x1 = o.xrange
        pad_l = min(clearance, (x0 - lft) / 4)
        pad_r = min(clearance, (rgt - x1) / 4)
        # hold the chosen side over the obstacle's whole x-extent, so side switches
        # happen only in the free gaps between obstacles
        kx += [x0 - pad_l, o.cx, x1 + pad_r]
        ky += [y, y, y]
    kx.append(gx)
    ky.append(gy)
    xs = np.linspace(sx, gx, n_dense)
    ys = np.interp(xs, kx, ky)
    # smooth low-frequency wiggle that vanishes at the endpoints
    u = (xs - sx) / (gx - sx)
    for k in range(1, 4):
        ys += rng.normal(0.0, wiggle / k) * np.sin(np.pi * k * u)
    # smooth, then pin endpoints back exactly
    ys = gaussian_filter1d(ys, smooth_sigma, mode="nearest")
    ys = ys + (sy - ys[0]) * (1 - u) + (gy - ys[-1]) * u
    return resample_arclength(np.stack([xs, ys], axis=1), T)


def validate_path(env, path, code, margin=0.0):
    word = crossing_word(env, path)
    collides = env.path_collides(path, margin)
    word_ok = word == expected_word_for_code(code)
    return (not collides) and word_ok, collides, word


def generate_routes(env, code, n, rng, T=32, max_tries_per_path=50, margin=0.01, **kw):
    """Up to n validated paths for `code`. Returns (paths, n_tries, n_mismatch)."""
    paths = []
    tries = 0
    mismatches = 0  # collision-free but word != code (should be 0 in this family)
    budget = n * max_tries_per_path
    while len(paths) < n and tries < budget:
        tries += 1
        p = sample_route(env, code, rng, T=T, **kw)
        ok, collides, word = validate_path(env, p, code, margin)
        if not collides and not ok:
            mismatches += 1
        if ok:
            paths.append(p)
    return np.array(paths).reshape(-1, T, 2), tries, mismatches


def enumerate_codes(env, rng, T=32, probe=5, max_tries=500, **kw):
    """Probe every bit code. Status is 'found' or 'not_found' (never 'infeasible')."""
    out = {}
    for code in itertools.product([0, 1], repeat=env.m):
        paths, tries, _ = generate_routes(env, code, probe, rng, T=T,
                                          max_tries_per_path=max_tries // probe, **kw)
        out[code] = "found" if len(paths) > 0 else "not_found"
    return out


@dataclass
class Dataset:
    X: np.ndarray            # (N, T, 2) full trajectories including endpoints
    code: np.ndarray         # (N, m) route code bits
    word: list               # reduced crossing word per sample (tuple of ints)
    class_id: np.ndarray     # (N,) index into `classes`
    classes: list            # list of codes, index -> code
    stats: dict

    def interior(self):
        """Model-facing representation: interior waypoints only, flattened (N, (T-2)*2)."""
        return self.X[:, 1:-1].reshape(len(self.X), -1)

    def save(self, path, env):
        obs, mask = env.obstacle_array()
        np.savez(path, X=self.X, code=self.code, class_id=self.class_id,
                 classes=np.array(self.classes), words=np.array([word_to_str(w) for w in self.word]),
                 start=env.start, goal=env.goal, obstacles=obs, obstacle_mask=mask)


def make_dataset(env, n, rng, T=32, class_probs=None, codes=None, **kw):
    """Sample n paths for a single environment.

    class_probs: dict code->prob, or None for uniform over `codes`.
    codes: which codes to include (default: all codes found by enumerate_codes).
    """
    env.validate()
    if codes is None:
        status = enumerate_codes(env, rng, T=T, **kw)
        codes = [c for c, s in status.items() if s == "found"]
    else:
        status = None
    codes = [tuple(c) for c in codes]
    if class_probs is None:
        probs = np.full(len(codes), 1.0 / len(codes))
    else:
        probs = np.array([class_probs.get(c, 0.0) for c in codes], dtype=float)
        probs /= probs.sum()
    counts = rng.multinomial(n, probs)
    X, C, W, ids = [], [], [], []
    stats = {"status": status, "per_class": {}}
    for i, (code, k) in enumerate(zip(codes, counts)):
        if k == 0:
            stats["per_class"][code_to_str(code)] = dict(requested=0, got=0, tries=0, mismatch=0)
            continue
        paths, tries, mism = generate_routes(env, code, k, rng, T=T, **kw)
        stats["per_class"][code_to_str(code)] = dict(requested=int(k), got=len(paths),
                                                     tries=tries, mismatch=mism)
        for p in paths:
            X.append(p)
            C.append(code)
            W.append(crossing_word(env, p))
            ids.append(i)
    perm = rng.permutation(len(X))
    X = np.array(X)[perm]
    C = np.array(C).reshape(len(perm), env.m)[perm]
    W = [W[i] for i in perm]
    ids = np.array(ids)[perm]
    return Dataset(X, C, W, ids, codes, stats)

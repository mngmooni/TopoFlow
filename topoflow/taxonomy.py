"""Per-sample outcome taxonomy that separates physical danger from evaluation-level invalidity.

For a clearance c* (the clearance the target / found set was defined with):
  U  unsafe:          collides at clearance 0
  M  margin:          no collision at 0, but violates clearance c*
  T  topology miss:   clearance >= c*, crossing word != requested code (or undefined / outside the family)
  N  label issue:     clearance >= c*, word == requested, but the code is not in the layout's found set
  V  valid:           clearance >= c*, word == requested, code found
For code-blind samples (requested None): T = word undefined or not in the found set; N does not occur.
"""
import numpy as np

from .e2 import word_to_code
from .topology import crossing_word

CATS = ("U", "M", "T", "N", "V")


def classify_samples(env, P, requested, found, cstar):
    """P: (n, T, 2) full paths. requested: list of code tuples or None. found: set of code tuples.
    Returns (cats list, words list)."""
    c0 = env.collision_mask(P, margin=0.0)
    cs = env.collision_mask(P, margin=cstar) if cstar > 0 else c0
    words = [word_to_code(crossing_word(env, p), env.m) for p in P]
    cats = []
    for j in range(len(P)):
        if c0[j]:
            cats.append("U")
        elif cs[j]:
            cats.append("M")
        else:
            w, r = words[j], (requested[j] if requested is not None else None)
            if requested is None:
                cats.append("V" if (w is not None and w in found) else "T")
            elif w != r:
                cats.append("T")
            elif w not in found:
                cats.append("N")
            else:
                cats.append("V")
    return cats, words


def tally(layouts, G, requested_idx, cstar, targets=None, index_code=None):
    """G: (L, n, T-2, 2). requested_idx: (L, n) code indices or None (code-blind).
    Returns fractions per category, plus joint TV where only V counts as a valid outcome (needs targets)."""
    L, n = G.shape[:2]
    cnt = {c: 0 for c in CATS}
    tvs = []
    for i, lay in enumerate(layouts):
        env = lay["env"]
        P = np.empty((n, G.shape[2] + 2, 2))
        P[:, 0], P[:, -1] = env.start, env.goal
        P[:, 1:-1] = G[i]
        req = None if requested_idx is None else [index_code(int(k), env.m) for k in requested_idx[i]]
        found = set(lay["found"])
        cats, words = classify_samples(env, P, req, found, cstar)
        for c in cats:
            cnt[c] += 1
        if targets is not None:
            tgt = targets[i]
            hist = {c: 0 for c in tgt}
            for c, w in zip(cats, words):
                if c == "V" and w in hist:
                    hist[w] += 1
            inv = n - sum(hist.values())
            tvs.append(0.5 * (sum(abs(hist[c] / n - tgt[c]) for c in tgt) + inv / n))
    tot = L * n
    out = {c: cnt[c] / tot for c in CATS}
    if tvs:
        out["tv_V"] = float(np.mean(tvs))
    return out

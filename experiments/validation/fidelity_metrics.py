"""Import-safe, shared Experiment C class-level fidelity statistics.

This module contains the original metric definitions from ``fidelity_boot.py``.
No result files are loaded or written at import time. Regressions should compare
statistics and reports against the frozen implementation before release.
"""
import numpy as np

C = 0.01
HUG = C + 0.005
STATS = ["len_ratio", "rough", "clear", "hug", "div"]
RATIO = {"rough", "clear", "div"}


def class_table(r, ncls):
    """Per-class [median len_ratio, mean rough, mean min clearance, hug fraction, diversity]; NaN if empty."""
    T = np.full((ncls, 5), np.nan)
    cls = r["cls"].astype(int)
    if len(cls):
        order = np.argsort(cls, kind="stable"); cs = cls[order]
        starts = np.searchsorted(cs, np.arange(ncls)); ends = np.searchsorted(cs, np.arange(ncls), side="right")
        lr, ro, mc = r["len_ratio"][order], r["rough"][order], r["minclear"][order]
        for j in np.where(ends > starts)[0]:
            s = slice(starts[j], ends[j])
            T[j, :4] = [np.median(lr[s]), ro[s].mean(), mc[s].mean(), np.mean(mc[s] < HUG)]
    T[:, 4] = r["div"]
    return T


def per_class_samples(r, ncls, key):
    cls = r["cls"].astype(int); out = [[] for _ in range(ncls)]
    for j, v in zip(cls, r[key]):
        out[j].append(v)
    return out


def agg(G, R, lid, w_cls, wl):
    """Weighted class means -> gen/ref ratios or differences; wl: layout bootstrap weights."""
    w = w_cls * wl[lid]
    out = {}
    for k, s in enumerate(STATS):
        ok = ~np.isnan(G[:, k]) & ~np.isnan(R[:, k]) & (w > 0)
        g, r_ = (G[ok, k] * w[ok]).sum() / w[ok].sum(), (R[ok, k] * w[ok]).sum() / w[ok].sum()
        out[s] = g / r_ if s in RATIO else g - r_
    return out

"""Day 12 step 3': does the crossing-word topology still hold on hard (stacked) layouts?
(locked: outputs/predictions_day12_words.md)

Independent invariant: winding number around each obstacle centre of (path + reference path above everything,
reversed). For x-monotone classes, code_j = winding_j.

  TOPO_MAX_OBS=5 python scripts/word_check.py       -> outputs/hard/words.json
"""
import itertools
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow import chart  # noqa: E402
from topoflow.e2 import MAX_OBS, word_to_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import code_index, generate  # noqa: E402
from topoflow.topology import crossing_word  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample  # noqa: E402

torch.set_num_threads(1)
C, N, N_EVAL, PER = 0.01, 60, 150, 8
rng = np.random.default_rng(5)


def winding_code(env, path):
    ref = np.array([env.goal, [env.goal[0], 1.5], [env.start[0], 1.5], env.start])
    loop = np.concatenate([path, ref[1:]])
    out = []
    for o in env.obstacles:
        v = loop - np.array([o.cx, o.cy])
        a, b = v[:-1], v[1:]
        ang = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1))
        out.append(int(round(ang.sum() / (2 * np.pi))))
    return tuple(out)


def check(env, paths, codes, coll=None):
    """returns counts: n, wind_ok (winding == requested), word_eq (cut word code == winding code, collision-free)"""
    n = wind_ok = word_eq = nfree = 0
    bad = []
    cm = env.collision_mask(paths) if coll is None else coll
    for p, c, col in zip(paths, codes, cm):
        n += 1
        w = winding_code(env, p)
        wind_ok += w == tuple(c)
        if not col:
            nfree += 1
            wc = word_to_code(crossing_word(env, p), env.m)
            word_eq += wc == w
            if wc != w and len(bad) < 5:
                bad.append(dict(word=list(crossing_word(env, p)), wind=list(w), code=list(c)))
    return n, wind_ok, nfree, word_eq, bad


res = {}
for fam in ("H1", "H2", "H3", "H4"):
    lays = pickle.load(open(f"outputs/hard/{fam}_layouts.pkl", "rb"))
    tot = dict(chart_n=0, chart_wind_ok=0, chart_free=0, chart_word_eq=0, w3_codes=0, w3_empty=0, bad=[])
    for lay in lays:
        env = lay["env"]
        for code in itertools.product([0, 1], repeat=env.m):
            lo, hi, ok = chart.bounds(env, code, C, N)
            # W3: above A, below B, B entirely below A over a shared x-range
            for (i, a), (j, b) in itertools.permutations(enumerate(env.obstacles), 2):
                if code[i] == 0 and code[j] == 1 and max(a.xrange[0], b.xrange[0]) < min(a.xrange[1], b.xrange[1]) \
                        and b.yrange[1] < a.yrange[0]:
                    tot["w3_codes"] += 1
                    tot["w3_empty"] += not ok and not chart.bounds(env, code, C, 400)[2]
                    break
            if not ok:
                continue
            U = np.concatenate([rng.normal(0, 1, (2, N)), rng.normal(0, 3, (2, N)), rng.choice([-50, 50], (2, N))])
            P = np.stack([chart.to_path(env, chart.decode(u, lo, hi), C) for u in U])
            n, wo, nf, we, bad = check(env, P, [code] * len(P))
            tot["chart_n"] += n
            tot["chart_wind_ok"] += wo
            tot["chart_free"] += nf
            tot["chart_word_eq"] += we
            tot["bad"] += bad[: max(0, 5 - len(tot["bad"]))]
    res[fam] = tot
    print(fam, json.dumps({k: v for k, v in tot.items() if k != "bad"}), tot["bad"][:2], flush=True)

v2 = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
v2.load_state_dict(torch.load("outputs/d2/models/V2_s0.pt"))
v2.eval()
p1 = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True)
p1.load_state_dict(torch.load("outputs/d2/models/P1_s0.pt"))
p1.eval()
for fam in ("H1", "H2", "H4"):
    ev = pickle.load(open(f"outputs/hard/{fam}_layouts.pkl", "rb"))[:N_EVAL]
    req = []
    for i, lay in enumerate(ev):
        for code in itertools.product([0, 1], repeat=lay["env"].m):
            if chart.bounds(lay["env"], code, C, N)[2]:
                req += [(i, code)] * PER
    G = sample(v2, ChartCache(ev, C, N), req, 5, torch.randn(len(req), N, generator=torch.Generator().manual_seed(9)))
    per = {}
    for j, (i, c) in enumerate(req):
        per.setdefault(i, []).append((c, G[j]))
    a = dict(n=0, wind_ok=0, free=0, word_eq=0)
    for i, items in per.items():
        n, wo, nf, we, _ = check(ev[i]["env"], np.stack([p for _, p in items]), [c for c, _ in items])
        a["n"] += n; a["wind_ok"] += wo; a["free"] += nf; a["word_eq"] += we
    res[fam]["v2"] = a
    # P1, same requests
    width = max(len(v) for v in per.values())
    idx = np.zeros((len(ev), width), int)
    for i, items in per.items():
        idx[i, :len(items)] = [code_index(c) for c, _ in items]
    Gp = generate(p1, ev, idx, 5, torch.randn(len(ev) * width, 60, generator=torch.Generator().manual_seed(9)))
    b = dict(n=0, free=0, wind_ok_free=0, word_eq=0)
    for i, items in per.items():
        env = ev[i]["env"]
        P = np.stack([np.concatenate([env.start[None], Gp[i, j], env.goal[None]]) for j in range(len(items))])
        codes = [c for c, _ in items]
        coll = env.collision_mask(P)
        for p, c, col in zip(P, codes, coll):
            b["n"] += 1
            if col:
                continue
            b["free"] += 1
            w = winding_code(env, p)
            b["wind_ok_free"] += w == tuple(c)
            b["word_eq"] += word_to_code(crossing_word(env, p), env.m) == w
    res[fam]["p1"] = b
    print(fam, "v2", a, "p1", b, flush=True)
json.dump(res, open("outputs/hard/words.json", "w"), indent=1, default=str)
print("done")

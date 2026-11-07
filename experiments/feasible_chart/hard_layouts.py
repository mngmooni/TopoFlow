"""research phase step 3: hard layouts that stress the chart construction (locked: the archived prediction record).

  TOPO_MAX_OBS=5 FAM=H1 python experiments/feasible_chart/hard_layouts.py        (FAM in H1 H2 H3 H4; MODELS=0 skips model eval)
Writes results/hard/{FAM}.json (summary) and results/hard/{FAM}_layouts.pkl.
"""
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402
from topoflow.e2 import MAX_OBS, word_to_code  # noqa: E402
from topoflow.env import Box, Circle, Env  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import code_index, generate  # noqa: E402
from topoflow.topology import crossing_word  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
FAM = os.environ.get("FAM", "H1")
MODELS = os.environ.get("MODELS", "1") == "1"
N_LAY = int(os.environ.get("N_LAY", "300"))
N_EVAL = int(os.environ.get("N_EVAL", "150"))
PER = 8
C = 0.01
N = 60
MIN_DIST = 0.02
os.makedirs("results/hard", exist_ok=True)
FAMS = {
    "H1": dict(m=(3, 5), strat=False, r=(0.08, 0.2), box=0.3, y=(-0.5, 0.5)),
    "H2": dict(m=(4, 5), strat=True, r=(0.12, 0.2), box=0.3, y=(-0.35, 0.35)),
    "H3": dict(m=(6, 8), strat=True, r=(0.05, 0.1), box=0.3, y=(-0.35, 0.35)),
    "H4": dict(m=(3, 5), strat=False, r=(0.08, 0.2), box=1.0, y=(-0.5, 0.5)),
}
FAMS["D2CHECK"] = FAMS["H1"]
P = FAMS[FAM]


def boundary(o, k=96):
    t = np.linspace(0, 2 * np.pi, k, endpoint=False)
    if o.kind == "circle":
        return np.stack([o.cx + o.r * np.cos(t), o.cy + o.r * np.sin(t)], 1)
    s = np.linspace(-1, 1, k // 4)
    return np.concatenate([np.stack([o.cx + o.hw * s, np.full_like(s, o.cy + sg * o.hh)], 1) for sg in (-1, 1)] +
                          [np.stack([np.full_like(s, o.cx + sg * o.hw), o.cy + o.hh * s], 1) for sg in (-1, 1)])


def apart(a, b):
    return not (any(b.contains(p, margin=MIN_DIST) for p in boundary(a)) or
                any(a.contains(p, margin=MIN_DIST) for p in boundary(b)) or
                a.contains(np.array([b.cx, b.cy])) or b.contains(np.array([a.cx, a.cy])))


def packed_layout(rng, m):
    """H2/H3: obstacles placed left to right with x-gaps ~ U[0, 0.04] (deviation from the locked stratified spec,
    which cannot fit these radii; recorded in PREDICTIONS.md before running)."""
    for _ in range(20000):
        sizes = rng.uniform(*P["r"], size=m)
        kinds = rng.random(m) < P["box"]
        hws = [s * rng.uniform(0.7, 1.0) if k else s for s, k in zip(sizes, kinds)]
        gaps = rng.uniform(0.0, 0.04, size=m - 1)
        span = 2 * sum(hws) + gaps.sum()
        room = 2 * (0.9 - C - 0.005) - span
        if room <= 0:
            continue
        x = -(0.9 - C - 0.005) + rng.uniform(0, room)
        obs = []
        for k in range(m):
            cx = x + hws[k]
            cy = rng.uniform(*P["y"])
            obs.append(Box(cx, cy, hws[k], sizes[k] * rng.uniform(0.7, 1.3)) if kinds[k] else Circle(cx, cy, hws[k]))
            x = cx + hws[k] + (gaps[k] if k < m - 1 else 0)
        env = Env(obs, (-0.9, 0.0), (0.9, 0.0))
        if all(apart(a, b) for a, b in itertools.combinations(env.obstacles, 2)):
            return env
    raise RuntimeError("no layout")


def sample_layout(rng):
    m = int(rng.integers(P["m"][0], P["m"][1] + 1))
    if P["strat"]:
        return packed_layout(rng, m)
    for _ in range(5000):
        obs = []
        for k in range(m):
            size = rng.uniform(*P["r"])
            cx = -0.6 + 1.2 * (k + rng.uniform()) / m if P["strat"] else rng.uniform(-0.6, 0.6)
            cy = rng.uniform(*P["y"])
            if rng.random() < P["box"]:
                obs.append(Box(cx, cy, size * rng.uniform(0.7, 1.0), size * rng.uniform(0.7, 1.3)))
            else:
                obs.append(Circle(cx, cy, size))
        obs.sort(key=lambda o: o.cx)
        if len({round(o.cx, 6) for o in obs}) < m:
            continue
        if P["strat"] and not all(obs[i].xrange[1] <= obs[i + 1].xrange[0] for i in range(m - 1)):
            continue
        if not all(apart(a, b) for a, b in itertools.combinations(obs, 2)):
            continue
        if any((o.r if o.kind == "circle" else o.hw) + C + abs(o.cx) >= 0.9 for o in obs):
            continue
        return Env(obs, (-0.9, 0.0), (0.9, 0.0))
    raise RuntimeError("no layout")


def R_of(o):
    return (o.r if o.kind == "circle" else o.hw) + C


def pointwise(env, code, xs):
    L = np.full(len(xs), chart.Y_MIN)
    U = np.full(len(xs), chart.Y_MAX)
    act = np.zeros(len(xs), bool)
    for o, b in zip(env.obstacles, code):
        R = R_of(o)
        inside = np.abs(xs - o.cx) < R
        h = np.sqrt(np.maximum(R * R - (xs - o.cx) ** 2, 0.0)) if o.kind == "circle" else np.full(len(xs), o.hh + C)
        if b == 0:
            L = np.where(inside, np.maximum(L, o.cy + h), L)
        else:
            U = np.where(inside, np.minimum(U, o.cy - h), U)
        act |= inside
    return L, U, act


def exact(env, code):
    e = []
    for o in env.obstacles:
        for x in (o.cx - R_of(o), o.cx + R_of(o)):
            e += [x - 1e-6, x + 1e-6]
    xs = np.sort(np.concatenate([np.linspace(-0.9, 0.9, 4000), e]))
    L, U, _ = pointwise(env, code, xs)
    return bool(np.all(L < U)), float(np.min(U - L))


def refine(env, code, xs, maxp=400):
    for _ in range(300):
        lo, hi, ok, L, U = chart.bounds_xs(env, code, C, xs)
        if ok or len(xs) >= maxp:
            return ok, len(xs)
        full = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
        new = set()
        for k in np.where(hi <= lo)[0] + 1:
            new |= {(full[k - 1] + full[k]) / 2, (full[k] + full[k + 1]) / 2}
        xs = np.unique(np.concatenate([xs, list(new)]))
    return False, len(xs)


t0 = time.time()
rng = np.random.default_rng({"H1": 101, "H2": 102, "H3": 103, "H4": 104, "D2CHECK": 0}[FAM])
if FAM == "D2CHECK":  # sanity: same eval code on in-distribution D2 unseen layouts
    with open("results/d2/layouts.pkl", "rb") as f:
        lays = [dict(env=l["env"]) for l in pickle.load(f)[1][:N_LAY]]
else:
    lays = [dict(env=sample_layout(rng)) for _ in range(N_LAY)]
rows = []
for i, lay in enumerate(lays):
    env = lay["env"]
    for code in itertools.product([0, 1], repeat=env.m):
        lo, hi, ok, _, _ = chart.bounds_xs(env, code, C, chart.grid(env, N, C))
        ex, gap = exact(env, code)
        r = dict(i=i, m=env.m, code=code, ok60=ok, exact=ex, gap=gap)
        if ex and not ok:
            r["ref_ok"], r["ref_pts"] = refine(env, code, chart.grid(env, N, C))
        if ok:
            Le, Ue, act = pointwise(env, code, chart.grid(env, N, C))
            den = np.clip(Ue - Le, 0, None)[act].sum()
            r["band"] = float(np.clip(hi - lo, 0, None)[act].sum() / den) if den > 0 else None
        rows.append(r)
ex = [r for r in rows if r["exact"]]
miss = [r for r in ex if not r["ok60"]]
xo = [sum(a.xrange[1] > b.xrange[0] for a, b in zip(l["env"].obstacles[:-1], l["env"].obstacles[1:])) for l in lays]
S = dict(fam=FAM, layouts=N_LAY, codes=len(rows), exact=len(ex), exact_frac=len(ex) / len(rows),
         ok60=sum(r["ok60"] for r in rows), cover60=sum(r["ok60"] for r in ex) / max(len(ex), 1),
         ok_not_exact=sum(r["ok60"] and not r["exact"] for r in rows),
         miss=len(miss), miss_fixed=sum(r["ref_ok"] for r in miss),
         miss_pts_max=max([r["ref_pts"] for r in miss if r["ref_ok"]] or [0]),
         miss_unfixed_gaps=sorted(round(r["gap"], 5) for r in miss if not r["ref_ok"])[:20],
         miss_gap_median=float(np.median([r["gap"] for r in miss])) if miss else None,
         cover_refined=(sum(r["ok60"] for r in ex) + sum(r["ref_ok"] for r in miss)) / max(len(ex), 1),
         band_median=float(np.median([r["band"] for r in rows if r.get("band") is not None])),
         layouts_with_x_overlap=int(sum(x > 0 for x in xo)),
         cover60_by_m={m: [sum(r["ok60"] for r in ex if r["m"] == m), sum(1 for r in ex if r["m"] == m)]
                       for m in range(P["m"][0], P["m"][1] + 1)})
print(json.dumps(S), flush=True)
with open(f"results/hard/{FAM}_layouts.pkl", "wb") as f:
    pickle.dump(lays, f)

# ---------------------------------------------------------------- models (D2-trained, out of distribution)
SUB = 64


def clearance_indep(env, Pth):
    a = np.linspace(0, 1, SUB + 1)[:-1]
    Sd = (Pth[:, :-1, None, :] * (1 - a)[:, None] + Pth[:, 1:, None, :] * a[:, None]).reshape(len(Pth), -1, 2)
    best = np.full(Sd.shape[:-1], np.inf)
    for o in env.obstacles:
        dx, dy = Sd[..., 0] - o.cx, Sd[..., 1] - o.cy
        if o.kind == "circle":
            d = np.sqrt(dx ** 2 + dy ** 2) - o.r
        else:
            qx, qy = np.abs(dx) - o.hw, np.abs(dy) - o.hh
            d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
        best = np.minimum(best, d)
    return best.min(-1), Sd


def side_indep(env, Sd, code):
    ok = np.ones(len(Sd), bool)
    for o, b in zip(env.obstacles, code):
        x0, x1 = o.xrange
        inx = (Sd[..., 0] >= x0) & (Sd[..., 0] <= x1)
        good = (Sd[..., 1] > o.cy) if b == 0 else (Sd[..., 1] < o.cy)
        ok &= ~np.any(inx & ~good, -1)
    return ok


XU = np.linspace(-0.9, 0.9, 62)[1:-1]


def diversity(paths):
    Y = np.stack([np.interp(XU, p[:, 0], p[:, 1]) for p in paths])
    if len(Y) < 2:
        return None
    d = [np.sqrt(np.mean((Y[a] - Y[b]) ** 2)) for a, b in itertools.combinations(range(len(Y)), 2)]
    return float(np.mean(d))


if MODELS and FAM != "H3":
    ev = lays[:N_EVAL]
    req = [(r["i"], r["code"]) for r in rows if r["i"] < N_EVAL and r["exact"] and r["ok60"] for _ in range(PER)]
    x0 = torch.randn(len(req), N, generator=torch.Generator().manual_seed(777))
    v2 = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
    v2.load_state_dict(torch.load("results/d2/models/V2_s0.pt"))
    v2.eval()
    p1 = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True)
    p1.load_state_dict(torch.load("results/d2/models/P1_s0.pt"))
    p1.eval()
    cache = ChartCache(ev, C, N)
    # P1: group requests per layout into (L, n) code index arrays
    per_lay = {}
    for i, code in req:
        per_lay.setdefault(i, []).append(code)
    width = max(len(v) for v in per_lay.values())
    idx = np.zeros((len(ev), width), int)
    valid = np.zeros((len(ev), width), bool)
    for i, cs in per_lay.items():
        idx[i, :len(cs)] = [code_index(c) for c in cs]
        valid[i, :len(cs)] = True
    x0p = torch.randn(len(ev) * width, 60, generator=torch.Generator().manual_seed(777))
    S["models"] = {}
    for K in (5, 20):
        for name in ("v2", "p1"):
            t1 = time.time()
            if name == "v2":
                G = sample(v2, cache, req, K, x0)
                groups = [(i, c, G[j]) for j, (i, c) in enumerate(req)]
            else:
                Gp = generate(p1, ev, idx, K, x0p)
                groups = []
                for i in range(len(ev)):
                    env = ev[i]["env"]
                    for j in np.where(valid[i])[0]:
                        Pth = np.concatenate([env.start[None], Gp[i, j], env.goal[None]])
                        groups.append((i, per_lay[i][j], Pth))
            U = Mg = T = viol = 0
            div = {}
            for i in range(len(ev)):
                env = ev[i]["env"]
                gi = [(c, p) for (ii, c, p) in groups if ii == i]
                if not gi:
                    continue
                Pth = np.stack([p for _, p in gi])
                codes = [c for c, _ in gi]
                u0 = env.collision_mask(Pth)
                uc = env.collision_mask(Pth, margin=C)
                wrong = np.array([word_to_code(crossing_word(env, p), env.m) != c for c, p in gi])
                U += int(u0.sum())
                Mg += int((uc & ~u0).sum())
                T += int((wrong & ~uc).sum())
                cl, Sd = clearance_indep(env, Pth)
                sd = np.array([side_indep(env, Sd[k:k + 1], c)[0] for k, c in enumerate(codes)])
                viol += int(((cl < C - 1e-9) | ~sd).sum())
                good = ~uc & ~wrong
                for c in set(codes):
                    sel = [k for k, cc in enumerate(codes) if cc == c and good[k]]
                    d = diversity(Pth[sel]) if len(sel) >= 2 else None
                    if d is not None:
                        div[(i, c)] = d
            n = len(groups)
            S["models"][f"{name}_K{K}"] = dict(n=n, U=U / n, M=Mg / n, T=T / n, indep_viol=viol / n,
                                               diversity=float(np.mean(list(div.values()))),
                                               sec=round(time.time() - t1))
            print(name, K, json.dumps(S["models"][f"{name}_K{K}"]), flush=True)
S["sec"] = round(time.time() - t0)
json.dump(S, open(f"results/hard/{FAM}.json", "w"), indent=1)
print("done", S["sec"])

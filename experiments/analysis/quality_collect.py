"""research phase: per-class quality rows for clustered bootstrap (locked: the archived prediction record). No training.
One run = one model checkpoint x one sampling-noise seed; sampling identical to ood_quality_matched.py otherwise.
Saves results/hard/boot/{MODEL}_m{MSEED}_n{NSEED}.npz (and refs.npz once).
  TOPO_MAX_OBS=5 MODEL=d2 MSEED=0 NSEEDS=11,12,13 python experiments/analysis/quality_collect.py
"""
import itertools, json, os, pickle, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

torch.set_num_threads(1)
C, N, PER, K = 0.01, 60, 8, 20
MODEL = os.environ.get("MODEL", "d2"); MSEED = int(os.environ.get("MSEED", "0"))
NSEEDS = [int(s) for s in os.environ.get("NSEEDS", "11").split(",")]
N_EVAL = int(os.environ.get("N_EVAL", "300")); OUTD = os.environ.get("OUTD", "results/hard/boot")
XU = np.linspace(-0.9, 0.9, 62)[1:-1]
lays = pickle.load(open("results/d4/layouts.pkl", "rb"))[1][:N_EVAL]
cache = ChartCache(lays, C, N)


def shortest(env, xs, lo, hi):
    X = np.concatenate([[env.start[0]], xs, [env.goal[0]]]); dx = np.diff(X); sy, gy = env.start[1], env.goal[1]
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - chart.EPS)

    def f(y):
        Y = np.concatenate([[sy], y, [gy]]); dy = np.diff(Y); s = np.sqrt(dx ** 2 + dy ** 2); g = dy / s
        return s.sum(), g[:-1] - g[1:]
    y0 = np.clip(np.interp(xs, [X[0], X[-1]], [sy, gy]), m - w, m + w)
    return float(minimize(f, y0, jac=True, method="L-BFGS-B", bounds=list(zip(m - w, m + w))).fun)


def min_clear(env, P, sub=16):
    a = np.linspace(0, 1, sub + 1)[:-1]
    S = (P[:-1, None] * (1 - a)[:, None] + P[1:, None] * a[:, None]).reshape(-1, 2); best = np.inf
    for o in env.obstacles:
        dx, dy = S[:, 0] - o.cx, S[:, 1] - o.cy
        if o.kind == "circle":
            d = np.sqrt(dx ** 2 + dy ** 2) - o.r
        else:
            qx, qy = np.abs(dx) - o.hw, np.abs(dy) - o.hh
            d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
        best = min(best, d.min())
    return best


def stats(env, paths, short):
    """class row: mean len, median len ratio, mean rough, mean clear, mean hug, diversity (nan if <2)."""
    L = [float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) for P in paths]
    Y = np.stack([np.interp(XU, P[:, 0], P[:, 1]) for P in paths])
    mc = [min_clear(env, P) for P in paths]
    div = (float(np.mean([np.sqrt(np.mean((Y[a] - Y[b]) ** 2)) for a, b in itertools.combinations(range(len(Y)), 2)]))
           if len(Y) > 1 else np.nan)
    return [np.mean(L), np.median(np.array(L) / short), np.mean(np.diff(Y, 2, axis=1) ** 2), np.mean(mc),
            np.mean(np.array(mc) < C + 0.005), div]


COLS = ["len", "len_ratio", "rough", "clear", "hug", "div"]
req = []
for i, lay in enumerate(lays):
    for code in itertools.product([0, 1], repeat=lay["env"].m):
        if cache.get(i, code)[3]:
            req += [(i, code)] * PER
keys = sorted(set(req))
shorts = {k: shortest(lays[k[0]]["env"], *cache.get(*k)[2:3], *cache.get(*k)[:2]) if False else None for k in []}
sh = {}
for k in keys:
    lo, hi, xs, _ = cache.get(*k); sh[k] = shortest(lays[k[0]]["env"], xs, lo, hi)
lid = np.array([k[0] for k in keys]); matched = np.array([len(lays[i]["found"].get(c, [])) > 0 for i, c in keys])
os.makedirs(OUTD, exist_ok=True)
if not os.path.exists(f"{OUTD}/refs.npz"):
    R = np.full((len(keys), len(COLS)), np.nan)
    for j, (i, c) in enumerate(keys):
        if matched[j]:
            R[j] = stats(lays[i]["env"], lays[i]["found"][c], sh[(i, c)])
    np.savez(f"{OUTD}/refs.npz", R=R, lid=lid, matched=matched, cols=COLS)
net = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
net.load_state_dict(torch.load(f"results/{MODEL}/models/V2_s{MSEED}.pt")); net.eval()
for ns in NSEEDS:
    G = sample(net, cache, req, K, torch.randn(len(req), N, generator=torch.Generator().manual_seed(ns)))
    groups = {}
    for j, k in enumerate(req):
        groups.setdefault(k, []).append(G[j])
    M = np.array([stats(lays[k[0]]["env"], groups[k], sh[k]) for k in keys])
    np.savez(f"{OUTD}/{MODEL}_m{MSEED}_n{ns}.npz", M=M, lid=lid, matched=matched, cols=COLS)
    print(MODEL, MSEED, ns, "saved", flush=True)

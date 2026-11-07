"""Experiment C — class-matched conditional fidelity, data collection (locked: the frozen prediction record (private archive)).

D2/D3 unseen, every code with a feasible n = 60 chart, 8 generated samples per class, NFE 20.
Per sample: len_ratio (length / J_c, the chart-constrained shortest length of the class), roughness (x-grid),
analytic min clearance (experiments/validation/indep_geom.py), hug (min clearance < c + 0.005). Same functions for references.
Saves results/final/fidelity/{ds}_{arm}_m{ms}_n{ns}.npz (per-sample values + class index) and {ds}_refs.npz.
Resumable: existing files are skipped.

  TOPO_MAX_OBS=5 DATA=results/d2 RUNS=v2:0:11,p1chart:0:11 python experiments/validation/fidelity_collect.py
"""
import itertools, json, os, pickle, sys, time  # noqa: E401
import numpy as np, torch  # noqa: E401
from scipy.optimize import minimize
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
import indep_geom as G  # noqa: E402
from topoflow import chart, direct  # noqa: E402
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample  # noqa: E402

torch.set_num_threads(1)
DATA = os.environ["DATA"]; DS = DATA[-2:]; SG = DS == "d3"
C, N, PER, K = 0.01, 60, 8, 20
OUTD = "results/final/fidelity"; os.makedirs(OUTD, exist_ok=True)
XU = np.linspace(-0.9, 0.9, 62)[1:-1]
lays = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:300]
cache = ChartCache(lays, C, N, sg=SG)


def shortest(env, xs, lo, hi):
    X = np.concatenate([[env.start[0]], xs, [env.goal[0]]]); dx = np.diff(X); sy, gy = env.start[1], env.goal[1]
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - chart.EPS)

    def f(y):
        Y = np.concatenate([[sy], y, [gy]]); dy = np.diff(Y); s = np.sqrt(dx ** 2 + dy ** 2); g = dy / s
        return s.sum(), g[:-1] - g[1:]
    y0 = np.clip(np.interp(xs, [X[0], X[-1]], [sy, gy]), m - w, m + w)
    return float(minimize(f, y0, jac=True, method="L-BFGS-B", bounds=list(zip(m - w, m + w))).fun)


keys = [(i, code) for i, lay in enumerate(lays) for code in itertools.product([0, 1], repeat=lay["env"].m)
        if cache.get(i, code)[3]]
kid = {k: j for j, k in enumerate(keys)}
req = [k for k in keys for _ in range(PER)]
JC = os.path.join(OUTD, f"{DS}_jc.npy")
if os.path.exists(JC):
    jc = np.load(JC)
else:
    jc = np.array([shortest(lays[i]["env"], cache.get(i, c)[2], *cache.get(i, c)[:2]) for i, c in keys]); np.save(JC, jc)


def sample_stats(env, P, j):
    """P (S, T, 2) paths of class j -> per-sample arrays."""
    ln = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
    Y = np.stack([np.interp(XU, p[:, 0], p[:, 1]) for p in P])
    A, B = P[:, :-1], P[:, 1:]
    mc = np.full(len(P), np.inf)
    for o in env.obstacles:
        mc = np.minimum(mc, G.seg_obstacle_dist(A, B, o).min(1))
    return ln / jc[j], np.mean(np.diff(Y, 2, axis=1) ** 2, 1), mc, Y


def collect(groups):
    """groups: list over classes of path arrays -> dict of flat per-sample arrays + per-class diversity."""
    out = {k: [] for k in ("cls", "len_ratio", "rough", "minclear")}
    div = np.full(len(keys), np.nan)
    for j, P in enumerate(groups):
        if P is None or len(P) == 0:
            continue
        lr, ro, mc, Y = sample_stats(lays[keys[j][0]]["env"], np.asarray(P), j)
        out["cls"] += [j] * len(lr); out["len_ratio"] += list(lr); out["rough"] += list(ro); out["minclear"] += list(mc)
        if len(Y) > 1:
            d = np.sqrt(((Y[:, None] - Y[None]) ** 2).mean(-1)); div[j] = d[np.triu_indices(len(Y), 1)].mean()
    return {k: np.array(v) for k, v in out.items()}, div


lid = np.array([i for i, _ in keys])
REF = f"{OUTD}/{DS}_refs.npz"
if not os.path.exists(REF):
    r, d = collect([lays[i]["found"].get(c) for i, c in keys])
    nref = np.array([len(lays[i]["found"].get(c, [])) for i, c in keys])
    np.savez(REF, **r, div=d, lid=lid, nref=nref)
    print(DS, "refs saved", flush=True)

for spec in os.environ.get("RUNS", "").split(","):
    if not spec:
        continue
    arm, ms, ns = spec.split(":"); ms, ns = int(ms), int(ns)
    f = f"{OUTD}/{DS}_{arm}_m{ms}_n{ns}.npz"
    if os.path.exists(f):
        continue
    x0 = torch.randn(len(req), N, generator=torch.Generator().manual_seed(ns))
    t0 = time.perf_counter()
    if arm == "v2":
        net = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG)
        net.load_state_dict(torch.load(f"{DATA}/models/V2_s{ms}.pt")); net.eval()
        paths = sample(net, cache, req, K, x0)
    else:
        m = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True, sg_input=SG)
        m.load_state_dict(torch.load(f"{DATA}/models/P1_s{ms}.pt")); m.eval()
        paths, _ = direct.sample(m, lays, req, direct.build_cond(lays, req, True, sg=SG), x0, K, "chart", C)
    sec = time.perf_counter() - t0
    paths = np.asarray(paths)
    r, d = collect([paths[j * PER:(j + 1) * PER] for j in range(len(keys))])
    np.savez(f, **r, div=d, lid=lid, sec=sec)
    print(spec, DS, "sec", round(sec), "min clear", float(r["minclear"].min()), flush=True)

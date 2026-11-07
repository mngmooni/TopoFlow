"""research phase exp 2: does chart clipping of training targets explain v2's roughness / hugging on D4?
(locked: the archived prediction record). No training.

Per n=60-feasible planner-found class on D4 unseen: refs are checked against the n=60 chart (clipped = any waypoint
outside [lo, hi]); v2-D4 samples (8 per class, K=20, noise seed 11 as in ood_quality) are compared with the refs of
the same class (excess = model − ref), and groups are compared within band-width strata (difference-in-differences).
  TOPO_MAX_OBS=5 N_EVAL=300 python experiments/feasible_chart/clip_diag.py   -> results/hard/clip_diag.json
"""
import itertools, json, os, pickle, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample  # noqa: E402

torch.set_num_threads(1)
C, N, PER, K = 0.01, 60, 8, 20
N_EVAL = int(os.environ.get("N_EVAL", "300"))
MODEL = os.environ.get("MODEL", "d4")
XU = np.linspace(-0.9, 0.9, 62)[1:-1]
lays = pickle.load(open("results/d4/layouts.pkl", "rb"))[1][:N_EVAL]
cache = ChartCache(lays, C, N)
net = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
net.load_state_dict(torch.load(f"results/{MODEL}/models/V2_s0.pt")); net.eval()
req = []
for i, lay in enumerate(lays):
    for code in itertools.product([0, 1], repeat=lay["env"].m):
        if cache.get(i, code)[3]:
            req += [(i, code)] * PER
G = sample(net, cache, req, K, torch.randn(len(req), N, generator=torch.Generator().manual_seed(11)))


def min_clear(env, P, sub=16):
    a = np.linspace(0, 1, sub + 1)[:-1]
    S = (P[:-1, None] * (1 - a)[:, None] + P[1:, None] * a[:, None]).reshape(-1, 2)
    best = np.inf
    for o in env.obstacles:
        dx, dy = S[:, 0] - o.cx, S[:, 1] - o.cy
        if o.kind == "circle":
            d = np.sqrt(dx ** 2 + dy ** 2) - o.r
        else:
            qx, qy = np.abs(dx) - o.hw, np.abs(dy) - o.hh
            d = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0)
        best = min(best, d.min())
    return best


rough = lambda P: float(np.mean(np.diff(np.interp(XU, P[:, 0], P[:, 1]), 2) ** 2))  # noqa: E731
groups = {}
for j, key in enumerate(req):
    groups.setdefault(key, []).append(G[j])
cls, wp = [], []   # per-class rows, per-waypoint rows
mix = {True: [], False: []}   # class-mix check: model rough/hug on found vs not-found classes
for (i, code), paths in groups.items():
    env = lays[i]["env"]
    refs = lays[i]["found"].get(code)
    isf = refs is not None and len(refs) > 0
    mix[isf] += [(rough(P), float(min_clear(env, P) < C + 0.005)) for P in paths]
    if not isf:
        continue
    lo, hi, xs, _ = cache.get(i, code)
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - chart.EPS)
    Yr = np.stack([np.interp(xs, q[:, 0], q[:, 1]) for q in refs])
    out_r = (Yr < lo) | (Yr > hi)                         # ref waypoint clipped
    clipdist = np.maximum(lo - Yr, 0) + np.maximum(Yr - hi, 0)
    Ym = np.stack([np.interp(xs, P[:, 0], P[:, 1]) for P in paths])
    sat_m = np.abs((Ym - m) / w) > 0.99                   # model waypoint saturated
    cls.append(dict(clip_frac=float(out_r.any(1).mean()), clip_rms=float(np.sqrt((clipdist ** 2).mean())),
                    width=float(np.median(hi - lo)),
                    rough_ex=float(np.mean([rough(P) for P in paths]) - np.mean([rough(q) for q in refs])),
                    hug_ex=float(np.mean([min_clear(env, P) < C + 0.005 for P in paths])
                                 - np.mean([min_clear(env, q) < C + 0.005 for q in refs])),
                    rough_m=float(np.mean([rough(P) for P in paths])), rough_r=float(np.mean([rough(q) for q in refs]))))
    wp.append(np.stack([out_r.mean(0), sat_m.mean(0), hi - lo], 1))   # per waypoint: ref clip rate, model sat rate, width
wp = np.concatenate(wp)
cf = np.array([c["clip_frac"] for c in cls]); W = np.array([c["width"] for c in cls])
q = np.quantile(W, [0.2, 0.4, 0.6, 0.8]); strata = np.digitize(W, q)


def did(key):
    """Within width quintiles: mean(excess | clipped class) − mean(excess | unclipped), weighted by stratum size."""
    v = np.array([c[key] for c in cls]); num = den = 0.0; per = []
    for s in range(5):
        a, b = (strata == s) & (cf > 0), (strata == s) & (cf == 0)
        if a.sum() >= 10 and b.sum() >= 10:
            d = v[a].mean() - v[b].mean(); n = a.sum() + b.sum()
            num += d * n; den += n; per.append(round(float(d), 5))
    return float(num / den) if den else None, per


wq = np.digitize(wp[:, 2], np.quantile(wp[:, 2], [0.2, 0.4, 0.6, 0.8]))
sat_ratio = []
for s in range(5):
    sel = wq == s
    a, b = sel & (wp[:, 0] > 0), sel & (wp[:, 0] == 0)
    if a.sum() >= 20 and b.sum() >= 20:
        sat_ratio.append(float(wp[a, 1].mean() / max(wp[b, 1].mean(), 1e-9)))
uncl = cf == 0
res = dict(model=MODEL, classes=len(cls), frac_classes_any_clip=float((cf > 0).mean()),
           frac_waypoints_clipped=float((wp[:, 0] > 0).mean()),
           rough_ex_all=float(np.mean([c["rough_ex"] for c in cls])), hug_ex_all=float(np.mean([c["hug_ex"] for c in cls])),
           did_rough=did("rough_ex"), did_hug=did("hug_ex"),
           unclipped_rough_ratio=float(np.mean([c["rough_m"] for c, u in zip(cls, uncl) if u])
                                       / np.mean([c["rough_r"] for c, u in zip(cls, uncl) if u])),
           clipped_rough_ratio=float(np.mean([c["rough_m"] for c, u in zip(cls, uncl) if not u])
                                     / np.mean([c["rough_r"] for c, u in zip(cls, uncl) if not u])),
           sat_ratio_by_width=sat_ratio,
           sat_at_clipped=float(wp[wp[:, 0] > 0, 1].mean()), sat_at_unclipped=float(wp[wp[:, 0] == 0, 1].mean()),
           clip_rms_clipped_classes=float(np.mean([c["clip_rms"] for c in cls if c["clip_frac"] > 0])))
for f in (True, False):
    a = np.array(mix[f])
    res[f"model_{'found' if f else 'notfound'}"] = dict(n=len(a), rough=float(a[:, 0].mean()), hug=float(a[:, 1].mean()))
res["matched_rough_ratio"] = float(np.mean([c["rough_m"] for c in cls]) / np.mean([c["rough_r"] for c in cls]))
res["did_rough_share"] = res["did_rough"][0] / res["rough_ex_all"] if res["did_rough"][0] is not None else None
print(json.dumps(res), flush=True)
json.dump(res, open(os.environ.get("OUT", "results/hard/clip_diag.json"), "w"), indent=1)

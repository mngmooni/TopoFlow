"""research phase: class-matched OOD quality on D4 (locked: the archived prediction record). No training.

Same sampling as experiments/analysis/ood_quality.py (MODELS=d2,d4; D4 unseen first 300 layouts; every n=60 chart-feasible class
x 8; K=20; noise seed 11), so the unmatched rows reproduce results/hard/quality_d4.json (validation gate).
Matched = classes (same layout, same start/goal, same route code) that have both model samples and planner refs.
Writes results/hard/quality_d4_matched.json only; existing files are not touched.
  TOPO_MAX_OBS=5 python experiments/analysis/ood_quality_matched.py
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
N_EVAL = int(os.environ.get("N_EVAL", "300"))
OUT = os.environ.get("OUT", "results/hard/quality_d4_matched.json")
QUIET = os.environ.get("QUIET", "0") == "1"
XU = np.linspace(-0.9, 0.9, 62)[1:-1]
lays = pickle.load(open("results/d4/layouts.pkl", "rb"))[1][:N_EVAL]
cache = ChartCache(lays, C, N)


def shortest(env, xs, lo, hi):   # identical to ood_quality.shortest
    X = np.concatenate([[env.start[0]], xs, [env.goal[0]]]); dx = np.diff(X); sy, gy = env.start[1], env.goal[1]
    m, w = (lo + hi) / 2, (hi - lo) / 2 * (1 - chart.EPS)

    def f(y):
        Y = np.concatenate([[sy], y, [gy]]); dy = np.diff(Y); s = np.sqrt(dx ** 2 + dy ** 2); g = dy / s
        return s.sum(), g[:-1] - g[1:]
    y0 = np.clip(np.interp(xs, [X[0], X[-1]], [sy, gy]), m - w, m + w)
    return float(minimize(f, y0, jac=True, method="L-BFGS-B", bounds=list(zip(m - w, m + w))).fun)


def min_clear(env, P, sub=16):   # identical to ood_quality.min_clear
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


def per_path(env, P, short):
    yu = np.interp(XU, P[:, 0], P[:, 1]); mc = min_clear(env, P); ln = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    return dict(len=ln, len_ratio=ln / short, rough=float(np.mean(np.diff(yu, 2) ** 2)), clear=float(mc), hug=float(mc < C + 0.005))


def diversity(paths):
    Y = np.stack([np.interp(XU, p[:, 0], p[:, 1]) for p in paths])
    return float(np.mean([np.sqrt(np.mean((Y[a] - Y[b]) ** 2)) for a, b in itertools.combinations(range(len(Y)), 2)]))


KEYS = ("len", "len_ratio", "rough", "clear", "hug")
req = []
for i, lay in enumerate(lays):
    for code in itertools.product([0, 1], repeat=lay["env"].m):
        if cache.get(i, code)[3]:
            req += [(i, code)] * PER
noise = torch.randn(len(req), N, generator=torch.Generator().manual_seed(11))
# class inventory
n60 = sorted(set(req))
found_all = [(i, c) for i, l in enumerate(lays) for c in l["found"] if len(l["found"][c])]
matched = [k for k in n60 if k in set(found_all)]
inv = dict(layouts=len(lays), classes_n60=len(n60), classes_found=len(found_all), classes_matched=len(matched),
           excluded_n60_no_ref=len(n60) - len(matched), excluded_found_not_n60=len(set(found_all) - set(n60)),
           frac_n60_excluded=1 - len(matched) / len(n60), samples_total=len(req), samples_matched=len(matched) * PER,
           refs_matched=int(sum(len(lays[i]["found"][c]) for i, c in matched)))
shorts = {}
res = dict(inventory=inv)
refrows, refdiv = {}, {}
for mname in ("d2", "d4"):
    net = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6)
    net.load_state_dict(torch.load(f"results/{mname}/models/V2_s0.pt")); net.eval()
    G = sample(net, cache, req, K, noise)
    groups = {}
    for j, key in enumerate(req):
        groups.setdefault(key, []).append(G[j])
    rows = {}
    for key, paths in groups.items():
        i, code = key; env = lays[i]["env"]; lo, hi, xs, _ = cache.get(i, code)
        if key not in shorts:
            shorts[key] = shortest(env, xs, lo, hi)
        rows[key] = [per_path(env, P, shorts[key]) for P in paths]
        if key in matched and key not in refrows:
            refs = lays[i]["found"][code]
            refrows[key] = [per_path(env, q, shorts[key]) for q in refs]
            refdiv[key] = diversity(refs) if len(refs) > 1 else None
        rows[key + ("div",)] = diversity(paths)
    agg_s = lambda rs, k: float(np.median([r[k] for r in rs])) if k == "len_ratio" else float(np.mean([r[k] for r in rs]))  # noqa: E731
    # unmatched (as research phase): model over all n60 classes, sample-weighted; refs over matched classes
    allm = [r for k in n60 for r in rows[k]]
    res[f"{mname}_unmatched"] = {k: agg_s(allm, k) for k in KEYS} | dict(div=float(np.mean([rows[k + ("div",)] for k in n60])))
    # matched, sample-weighted
    mm = [r for k in matched for r in rows[k]]
    res[f"{mname}_matched_sample"] = {k: agg_s(mm, k) for k in KEYS}
    # matched, class-equal (primary): per-class mean, then mean over classes (len_ratio: median of class medians)
    def cls_eq(R):
        out = {}
        for k in KEYS:
            v = [np.median([r[k] for r in R[c]]) if k == "len_ratio" else np.mean([r[k] for r in R[c]]) for c in matched]
            out[k] = float(np.median(v)) if k == "len_ratio" else float(np.mean(v))
        return out
    res[f"{mname}_matched_class"] = cls_eq(rows)
    dk = [c for c in matched if refdiv[c] is not None]
    res[f"{mname}_matched_class"]["div"] = float(np.mean([rows[c + ("div",)] for c in dk]))
    res[f"{mname}_notmatched"] = {k: agg_s([r for k2 in n60 if k2 not in set(matched) for r in rows[k2]], k) for k in KEYS}
    if "ref_matched_class" not in res:
        allr = [r for c in matched for r in refrows[c]]
        res["ref_unmatched"] = {k: agg_s(allr, k) for k in KEYS} | dict(div=float(np.mean([v for v in refdiv.values() if v is not None])))
        res["ref_matched_sample"] = {k: agg_s(allr, k) for k in KEYS}
        res["ref_matched_class"] = cls_eq(refrows) | dict(div=float(np.mean([refdiv[c] for c in dk])))
        res["inventory"]["classes_div"] = len(dk)
    if not QUIET:
        print(mname, "done", flush=True)
# validation gate against the research phase file (unmatched rows must reproduce it)
old = json.load(open("results/hard/quality_d4.json"))["D4"] if os.path.exists("results/hard/quality_d4.json") else None
if old and N_EVAL == 300:
    res["gate"] = dict(d2_rough_diff=abs(res["d2_unmatched"]["rough"] - old["v2_rough"]),
                       d4_rough_diff=abs(res["d4_unmatched"]["rough"] - old["v2d4_rough"]),
                       ref_rough_diff=abs(res["ref_unmatched"]["rough"] - old["ref_rough"]),
                       d2_hug_diff=abs(res["d2_unmatched"]["hug"] - old["v2_hug"]))
json.dump(res, open(OUT, "w"), indent=1)
print("written", OUT, json.dumps(res["inventory"]) if not QUIET else "", flush=True)

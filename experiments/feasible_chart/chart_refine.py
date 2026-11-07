"""research phase (post-hoc, not pre-registered): local refinement of infeasible charts.
Bisect the two segments next to every failing waypoint until the chart is feasible or the grid has MAXP points.
  TOPO_MAX_OBS=5 DATA=results/d1 python experiments/feasible_chart/chart_refine.py
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart  # noqa: E402

DATA = os.environ.get("DATA", "results/d1")
MAXP = int(os.environ.get("MAXP", "2000"))
C = float(json.load(open(f"{DATA}/betas.json")).get("_margin", 0.03))
rows = pickle.load(open(f"{DATA}/chart_coverage_rows.pkl", "rb"))
sets = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))


def refine(env, code, xs):
    for it in range(200):
        lo, hi, ok, L, U = chart.bounds_xs(env, code, C, xs)
        if ok:
            return True, len(xs), it
        if len(xs) >= MAXP:
            return False, len(xs), it
        full = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
        bad = np.where(hi <= lo)[0] + 1  # index in full
        if not (L[0] <= env.start[1] <= U[0] and L[-1] <= env.goal[1] <= U[-1]):
            bad = np.concatenate([bad, [1, len(full) - 2]])
        new = set()
        for k in bad:
            new.add((full[k - 1] + full[k]) / 2)
            new.add((full[k] + full[k + 1]) / 2)
        xs = np.unique(np.concatenate([xs, list(new)]))
    return False, len(xs), 200


out = []
for r in rows:
    if r["found"] and not r["ok60"] or (not r["found"] and r["exact"] and not r["ok60"]):
        lay = (sets[1] if r["split"] == "unseen" else sets[0])[r["i"]]
        ok, npts, it = refine(lay["env"], r["code"], chart.grid(lay["env"], 60, C))
        out.append(dict(split=r["split"], found=r["found"], ok=ok, npts=npts, it=it, mingap=r["exact_mingap"]))
for sp in ("unseen", "train"):
    for f in (True, False):
        o = [x for x in out if x["split"] == sp and x["found"] == f]
        if o:
            print(sp, "found" if f else "notfound_exact", len(o), "recovered", sum(x["ok"] for x in o),
                  "points median/max", int(np.median([x["npts"] for x in o if x["ok"]] or [0])),
                  max([x["npts"] for x in o if x["ok"]] or [0]))
json.dump(out, open(f"{DATA}/chart_refine.json", "w"))

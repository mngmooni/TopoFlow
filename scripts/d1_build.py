"""D1: 3-5 obstacles, fixed start/goal, margin 0.03 (same as E2). Builds layouts, costs and betas.

  TOPO_MAX_OBS=5 python scripts/d1_build.py
  TOPO_MAX_OBS=5 OUT=outputs/d2 MARGIN=0.01 python scripts/d1_build.py   # D2 narrow passages
  TOPO_MAX_OBS=5 OUT=outputs/d3 MARGIN=0.01 SG_Y=0.4 python scripts/d1_build.py   # D3 start/goal variation
"""
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.e2 import MAX_OBS, build_layouts  # noqa: E402
from topoflow.env import random_layout  # noqa: E402
from topoflow.tdfm import calibrate_beta, layout_costs, mean_norm_entropy  # noqa: E402

assert MAX_OBS == 5, "run with TOPO_MAX_OBS=5"
OUT = os.environ.get("OUT", "outputs/d1")
MARGIN = float(os.environ.get("MARGIN", "0.03"))  # D2: 0.01
SG_Y = float(os.environ.get("SG_Y", "0"))  # D3: start/goal y ~ U[-SG_Y, SG_Y] per layout (x fixed at -0.9 / 0.9)
os.makedirs(OUT, exist_ok=True)
N_TRAIN, N_UNSEEN, N_CALIB, PER_CODE = 2000, 300, 300, 4
REGIMES = {"uniform": 1.0, "medium": 0.5, "concentrated": 0.1}


def sampler(rng):
    n = int(rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
    sg = {}
    if SG_Y > 0:
        sg = dict(start=(-0.9, float(rng.uniform(-SG_Y, SG_Y))), goal=(0.9, float(rng.uniform(-SG_Y, SG_Y))))
    return random_layout(rng, n, x_band=(-0.66, 0.66), r_range=(0.08, 0.15), box_prob=0.3,
                         min_gap=0.05, stratified=True, max_tries=2000, **sg)


t0 = time.time()
sets = {}
for name, n, seed in [("train", N_TRAIN, 10), ("unseen", N_UNSEEN, 11), ("calib", N_CALIB, 12)]:
    sets[name] = build_layouts(np.random.default_rng(seed), n, PER_CODE, margin=MARGIN, sampler=sampler)
    print(f"{name}: {n} layouts in {time.time() - t0:.0f}s", flush=True)
with open(f"{OUT}/layouts.pkl", "wb") as f:
    pickle.dump((sets["train"], sets["unseen"], sets["calib"]), f)

C = {}
for name, lays in sets.items():
    cs, fb = zip(*[layout_costs(l) for l in lays])
    C[name] = list(cs)
    print(f"costs {name}: fallback codes {sum(fb)}", flush=True)
with open(f"{OUT}/costs.pkl", "wb") as f:
    pickle.dump(C, f)

stats = {}
for name, lays in sets.items():
    by_m = {}
    for l in lays:
        by_m.setdefault(l["env"].m, []).append(len(l["found"]) / 2 ** l["env"].m)
    stats[name] = {m: dict(n=len(v), found_frac=float(np.mean(v))) for m, v in sorted(by_m.items())}
betas = {r: (0.0 if h >= 1.0 else calibrate_beta(C["calib"], h)) for r, h in REGIMES.items()}
info = {r: dict(beta=b, entropy_calib=mean_norm_entropy(C["calib"], b),
                entropy_unseen=mean_norm_entropy(C["unseen"], b)) for r, b in betas.items()}
info["_stats"] = stats
info["_margin"] = MARGIN
info["_sg_y"] = SG_Y
info["_n_train_samples"] = int(sum(len(P) for l in sets["train"] for P in l["found"].values()))
print(json.dumps(info, indent=1), flush=True)
with open(f"{OUT}/betas.json", "w") as f:
    json.dump(info, f, indent=1)
print(f"done in {time.time() - t0:.0f}s")

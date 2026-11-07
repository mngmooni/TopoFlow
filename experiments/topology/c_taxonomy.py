"""C: per-sample taxonomy (U/M/T/N/V) for existing models on D1 (or D2 via DATA=...).

Regenerates exactly the samples of d1_eval (x0 seed 777, codes seed 900) and classifies each sample at
clearance 0, 0.01 and 0.03. The headline is c* = the planner clearance of the dataset (betas.json _margin).

  TOPO_MAX_OBS=5 python experiments/topology/c_taxonomy.py
  TOPO_MAX_OBS=5 DATA=results/d2 python experiments/topology/c_taxonomy.py
  REGIMES=medium,uniform KS=2,5,20 ONLY=P1_s0,B0attn_medium_s0
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.selectors import CostRegSelector, probs_from_cost  # noqa: E402
from topoflow.taxonomy import tally  # noqa: E402
from topoflow.tdfm import boltzmann, generate, index_code, sample_codes, selector_inputs, target_vector  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
SG = os.environ.get("SG", "0") == "1"  # D3: models and selector take (start_y, goal_y)
DATA = os.environ.get("DATA", "results/d1")
KS = [int(k) for k in os.environ.get("KS", "2,5,20").split(",")]
REGIMES = os.environ.get("REGIMES", "medium,uniform").split(",")
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
N_PER = 32
meta = json.load(open(f"{DATA}/betas.json"))
CSTAR = float(meta.get("_margin", 0.03))
MARGINS = sorted({0.0, 0.01, 0.03, CSTAR})
betas = {k: v["beta"] for k, v in meta.items() if not k.startswith("_")}
with open(f"{DATA}/layouts.pkl", "rb") as f:
    unseen = pickle.load(f)[1]
with open(f"{DATA}/costs.pkl", "rb") as f:
    C = pickle.load(f)
OUTF = f"{DATA}/taxonomy.jsonl"

sel = CostRegSelector(512, 4, extra=2 if SG else 0)
sel.load_state_dict(torch.load(f"{DATA}/models/sel_costreg.pt"))
sel.eval()
Xun, Sun = selector_inputs(unseen, sg=SG)
with torch.no_grad():
    jh, fl = (t.numpy() for t in sel(Xun))
feas = (fl > 0) & Sun.numpy()
feas = np.where(feas.any(1, keepdims=True), feas, Sun.numpy())
x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))
ms = np.array([l["env"].m for l in unseen])


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUTF, "a") as f:
        f.write(json.dumps(rec) + "\n")


def run(model, label, regime, codes):
    targets = [boltzmann(c, betas[regime]) for c in C["unseen"]]
    for K in KS:
        G = generate(model, unseen, codes if codes is not None else np.zeros((len(unseen), N_PER), int), K, x0, sg=SG)
        rec = dict(model=label, regime=regime, K=K, cstar=CSTAR)
        for mg in MARGINS:
            rec[f"c{mg}"] = tally(unseen, G, codes, mg, targets, index_code)
        # headline split by obstacle count at c*
        for m in sorted(set(ms)):
            idx = np.where(ms == m)[0]
            rec[f"m{m}"] = tally([unseen[i] for i in idx], G[idx], None if codes is None else codes[idx], CSTAR,
                                 [targets[i] for i in idx], index_code)
        if codes is not None:  # requested code found vs not_found, at c*
            fmask = np.array([[index_code(int(k), unseen[i]["env"].m) in unseen[i]["found"] for k in codes[i]]
                              for i in range(len(unseen))])
            rec["req_notfound_frac"] = float(1 - fmask.mean())
        log(rec)


for path in sorted(glob.glob(f"{DATA}/models/*.pt")):
    stem = os.path.basename(path)[:-3]
    if stem.startswith("sel_") or (ONLY and stem not in ONLY) or not (stem.startswith("P1") or stem.startswith("B0attn")):
        continue
    is_p1 = stem.startswith("P1")
    model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=is_p1, sg_input=SG)
    model.load_state_dict(torch.load(path))
    model.eval()
    if is_p1:
        for r in REGIMES:
            p_or = np.array([target_vector(boltzmann(c, betas[r]), None) for c in C["unseen"]])
            run(model, f"{stem}+oracle", r, sample_codes(p_or, N_PER, seed=900))
            run(model, f"{stem}+B2costreg", r, sample_codes(probs_from_cost(jh, feas, betas[r]), N_PER, seed=900))
    else:
        run(model, stem, stem.split("_")[1], None)
print("done")

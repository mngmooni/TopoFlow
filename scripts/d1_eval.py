"""Evaluate D1-style environments: cost-regression selector, P1 + {oracle p*, B2} codes, B0-attn variants.
Results are stratified by obstacle count. Appends to {DATA}/results.jsonl.

  TOPO_MAX_OBS=5 python scripts/d1_eval.py
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.selectors import CostRegSelector, cost_targets, probs_from_cost, train_costlike  # noqa: E402
from topoflow.tdfm import (boltzmann, evaluate_samples, found_mask, generate, sample_codes,  # noqa: E402
                           selector_inputs, target_vector)

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
SG = os.environ.get("SG", "0") == "1"  # D3: models and selector take (start_y, goal_y)
DATA = os.environ.get("DATA", "outputs/d1")
KS = [2, 5, 20]
N_PER = 32
with open(f"{DATA}/layouts.pkl", "rb") as f:
    train_layouts, unseen = pickle.load(f)[:2]
with open(f"{DATA}/costs.pkl", "rb") as f:
    C = pickle.load(f)
betas = {k: v["beta"] for k, v in json.load(open(f"{DATA}/betas.json")).items() if not k.startswith("_")}
OUTF = f"{DATA}/results.jsonl"


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUTF, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- selector
Xtr, Str = selector_inputs(train_layouts, sg=SG)
Xun, Sun = selector_inputs(unseen, sg=SG)
Fun = found_mask(unseen).numpy()
sel_path = f"{DATA}/models/sel_costreg.pt"
sel = CostRegSelector(512, 4, extra=2 if SG else 0)
if os.path.exists(sel_path):
    sel.load_state_dict(torch.load(sel_path))
else:
    Jtr, Ftr = cost_targets(C["train"], train_layouts)
    sel = train_costlike(sel, Xtr, Str, Jtr, Ftr, steps=12000)
    torch.save(sel.state_dict(), sel_path)
sel.eval()
with torch.no_grad():
    jh, fl = (t.numpy() for t in sel(Xun))
feas = (fl > 0) & Sun.numpy()
feas = np.where(feas.any(1, keepdims=True), feas, Sun.numpy())
Jun, _ = cost_targets(C["unseen"], unseen)
sel_info = dict(kind="selector", J_mae=float(np.nanmean(np.abs(jh - Jun))),
                feas_acc=float(((fl > 0) == Fun)[Sun.numpy()].mean()))
for r, b in betas.items():
    T = np.array([target_vector(boltzmann(c, b), None) for c in C["unseen"]])
    tv = lambda q: float(np.mean(0.5 * np.abs(q - T).sum(1)))  # noqa: E731
    sel_info[r] = dict(tv_learned=tv(probs_from_cost(jh, feas, b)), tv_oracle_mask=tv(probs_from_cost(jh, Fun, b)))
log(sel_info)

# ---------------------------------------------------------------- generation
x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))
ms = np.array([l["env"].m for l in unseen])
dummy = np.zeros((len(unseen), N_PER), dtype=int)


def run(model, label, regime, codes, extra=None):
    targets = [boltzmann(c, betas[regime]) for c in C["unseen"]]
    for K in KS:
        G = generate(model, unseen, codes, K, x0, sg=SG)
        rec = dict(model=label, regime=regime, K=K, **(extra or {}))
        rec["all"] = evaluate_samples(unseen, G, targets, C["unseen"], code_idx=None if codes is dummy else codes)
        for m in sorted(set(ms)):
            idx = np.where(ms == m)[0]
            rec[f"m{m}"] = {k: v for k, v in evaluate_samples([unseen[i] for i in idx], G[idx],
                            [targets[i] for i in idx], [C["unseen"][i] for i in idx]).items()
                            if k in ("collision", "deep", "invalid", "tv_joint")}
        log(rec)


only = os.environ.get("ONLY", "")  # comma list of model-file stems to evaluate; empty = all
for path in sorted(glob.glob(f"{DATA}/models/*.pt")):
    stem = os.path.basename(path)[:-3]
    if stem.startswith("sel_") or (only and stem not in only.split(",")):
        continue
    is_p1 = stem.startswith("P1")
    model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=is_p1, sg_input=SG)
    model.load_state_dict(torch.load(path))
    model.eval()
    if is_p1:
        for r, b in betas.items():
            p_or = np.array([target_vector(boltzmann(c, b), None) for c in C["unseen"]])
            run(model, f"{stem}+oracle", r, sample_codes(p_or, N_PER, seed=900))
            run(model, f"{stem}+B2costreg", r, sample_codes(probs_from_cost(jh, feas, b), N_PER, seed=900))
    else:
        regime = stem.split("_")[1]
        run(model, stem, regime, dummy)
print("done")

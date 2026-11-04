"""Day 8: selector variants. Reports layout-wise TV(selector, p*_beta) on the 300 unseen layouts.

Run from the repo root (after day7_selector.py):  python scripts/day8_selectors.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.selectors import (CompSelector, CostRegSelector, LogitSelector, cost_targets,  # noqa: E402
                                probs_from_cost, train_costlike, train_logits)
from topoflow.tdfm import boltzmann, found_mask, selector_inputs, target_vector  # noqa: E402

torch.set_num_threads(1)
OUT = "outputs/day8"
os.makedirs(f"{OUT}/models", exist_ok=True)
with open("outputs/day5/layouts.pkl", "rb") as f:
    train_layouts, unseen = pickle.load(f)
with open("outputs/day7/costs.pkl", "rb") as f:
    C = pickle.load(f)
betas = {k: v["beta"] for k, v in json.load(open("outputs/day7/betas.json")).items()}

Xtr, Str = selector_inputs(train_layouts)
Xun, Sun = selector_inputs(unseen)
Fun = found_mask(unseen).numpy()
Jtr, Ftr = cost_targets(C["train"], train_layouts)
Jun, _ = cost_targets(C["unseen"], unseen)
targets = {r: np.array([target_vector(boltzmann(c, b), None) for c in C["unseen"]]) for r, b in betas.items()}


def tv(q, r):
    return float(np.mean(0.5 * np.abs(q - targets[r]).sum(1)))


results = {}


def report(name, per_regime):
    results[name] = per_regime
    print(name, json.dumps({r: {k: round(v, 4) for k, v in d.items()} for r, d in per_regime.items()}), flush=True)


# ---- (a) bigger logit MLP, one per regime
per = {}
for r, b in betas.items():
    P = torch.tensor(np.array([target_vector(boltzmann(c, b), None) for c in C["train"]]), dtype=torch.float32)
    t0 = time.time()
    m = train_logits(LogitSelector(512, 4), Xtr, Str, P, steps=12000)
    with torch.no_grad():
        z = m(Xun)
        q_s = torch.softmax(z.masked_fill(~Sun, -1e9), -1).numpy()
        q_o = torch.softmax(z.masked_fill(~torch.tensor(Fun), -1e9), -1).numpy()
    per[r] = dict(tv_learned=tv(q_s, r), tv_oracle=tv(q_o, r), nf_mass=float((q_s * ~Fun).sum(1).mean()),
                  sec=time.time() - t0)
    torch.save(m.state_dict(), f"{OUT}/models/sel_mlp512_{r}.pt")
report("mlp512x4_12k", per)

# ---- (b) cost regression, (c) compositional: one model for all betas
for name, ctor, steps in [("costreg512x4", lambda: CostRegSelector(512, 4), 12000),
                          ("comp_d128", lambda: CompSelector(128, 2), 6000)]:
    t0 = time.time()
    m = train_costlike(ctor(), Xtr, Str, Jtr, Ftr, steps=steps)
    sec = time.time() - t0
    torch.save(m.state_dict(), f"{OUT}/models/sel_{name}.pt")
    with torch.no_grad():
        jh, fl = m(Xun)
    jh, fl = jh.numpy(), fl.numpy()
    feas = (fl > 0) & Sun.numpy()
    feas = np.where(feas.any(1, keepdims=True), feas, Sun.numpy())
    mae = float(np.nanmean(np.abs(jh - Jun)))
    feas_acc = float(((fl > 0) == Fun)[Sun.numpy()].mean())
    per = {}
    for r, b in betas.items():
        q_s = probs_from_cost(jh, feas, b)
        q_o = probs_from_cost(jh, Fun, b)
        per[r] = dict(tv_learned=tv(q_s, r), tv_oracle=tv(q_o, r), nf_mass=float((q_s * ~Fun).sum(1).mean()))
    per["_diag"] = dict(J_mae=mae, feas_acc=feas_acc, sec=sec)
    report(name, per)

with open(f"{OUT}/selector_variants.json", "w") as f:
    json.dump(results, f, indent=1)
print("done")

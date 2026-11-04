"""Day 8b: end-to-end B2 with the cost-regression selector (learned feasibility) + existing P1.

Same unseen layouts, noise (777) and code-sampling seed (900) as day7_selector.py.
Run from the repo root:  python scripts/day8_b2costreg.py
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.selectors import CostRegSelector, probs_from_cost  # noqa: E402
from topoflow.tdfm import boltzmann, evaluate_samples, found_mask, generate, sample_codes, selector_inputs  # noqa: E402

torch.set_num_threads(1)
N_PER = 32
with open("outputs/day5/layouts.pkl", "rb") as f:
    _, unseen = pickle.load(f)
with open("outputs/day7/costs.pkl", "rb") as f:
    C = pickle.load(f)
betas = {k: v["beta"] for k, v in json.load(open("outputs/day7/betas.json")).items()}

Xun, Sun = selector_inputs(unseen)
sel = CostRegSelector(512, 4)
sel.load_state_dict(torch.load("outputs/day8/models/sel_costreg512x4.pt"))
sel.eval()
with torch.no_grad():
    jh, fl = (t.numpy() for t in sel(Xun))
feas = (fl > 0) & Sun.numpy()
feas = np.where(feas.any(1, keepdims=True), feas, Sun.numpy())

x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))
for seed in [0, 1]:
    p1 = ObstacleAttnField(30, 3, d=128, depth=4, chunk=3)
    p1.load_state_dict(torch.load(f"outputs/day6/models/P1_s{seed}.pt"))
    p1.eval()
    for r, b in betas.items():
        targets = [boltzmann(c, b) for c in C["unseen"]]
        codes = sample_codes(probs_from_cost(jh, feas, b), N_PER, seed=900)
        for K in [2, 5, 20]:
            G = generate(p1, unseen, codes, K, x0)
            rec = dict(model="P1+B2_costreg", p1_seed=seed, regime=r, K=K,
                       **evaluate_samples(unseen, G, targets, C["unseen"], code_idx=codes))
            print(json.dumps(rec), flush=True)
            with open("outputs/day7/results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
print("done")

"""research phase: fairness check. Is B1 (padded MLP) just undertrained on E2? Train 3x longer (90k steps).

Run from the repo root:  python experiments/baselines/mlp_training_budget.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.cfm import VelocityMLP, n_params, train_cfm  # noqa: E402
from topoflow.e2 import HELD_OUT, code_ok_for_train, stack_samples  # noqa: E402
from topoflow.e2eval import build_items, eval_conditioned  # noqa: E402

torch.set_num_threads(1)
OUT = "results/baselines/obstacle_attention"
STEPS = 90000
seed = 0
with open("results/baselines/variable_layout/layouts.pkl", "rb") as f:
    train_layouts, unseen = pickle.load(f)
held = [l for l in unseen if any(c in HELD_OUT for c in l["found"])]
splits = {"unseen_pos": (unseen, build_items(unseen, "trainable")),
          "unseen_combo": (held, build_items(held, "held_out"))}

tr = stack_samples(train_layouts, code_filter=code_ok_for_train)
x = torch.tensor(tr["x"])
cond = torch.tensor(np.concatenate([tr["feat"], tr["mask"], tr["code"]], 1))
torch.manual_seed(seed)
m = VelocityMLP(60, cond_dim=21, hidden=512, depth=4)
t0 = time.time()
losses = train_cfm(m, x, cond=cond, steps=STEPS, seed=seed, log_every=15000)
torch.save(m.state_dict(), f"{OUT}/models/B1long_s{seed}.pt")
print(json.dumps(dict(model="B1long", seed=seed, params=n_params(m), train_sec=round(time.time() - t0),
                      final_loss=float(np.mean(losses[-500:])))), flush=True)
m.eval()
for split, (lays, items) in splits.items():
    for K in [2, 5, 20]:
        r, _ = eval_conditioned(m, lays, items, K, seed=10_000 + seed)
        rec = dict(model="B1long", seed=seed, split=split, K=K, **r)
        print(json.dumps(rec), flush=True)
        with open(f"{OUT}/results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

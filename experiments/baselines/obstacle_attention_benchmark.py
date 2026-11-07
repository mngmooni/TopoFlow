"""research phase: P1 (obstacle-token cross-attention) vs B1 (padded MLP) vs B1-full on E2.

Same data, same conditioning info, same steps (30k) and batch (256), similar params (~0.84M vs 0.87M).
All models are evaluated on identical (layout, code, noise) triples.

Run from the repo root after variable_layout_baselines.py and layout_generalization_controls.py:  python experiments/baselines/obstacle_attention_benchmark.py
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
from topoflow.p1 import ObstacleAttnField  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))
D5, OUT = "results/baselines/variable_layout", "results/baselines/obstacle_attention"
os.makedirs(f"{OUT}/models", exist_ok=True)
KS = [2, 5, 20]
STEPS = 30000
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]

with open(f"{D5}/layouts.pkl", "rb") as f:
    train_layouts, unseen_layouts = pickle.load(f)
held_layouts = [l for l in unseen_layouts if any(c in HELD_OUT for c in l["found"])]
splits = {
    "unseen_pos": (unseen_layouts, build_items(unseen_layouts, "trainable")),
    "unseen_combo": (held_layouts, build_items(held_layouts, "held_out")),
}
print({k: len(v[1]) for k, v in splits.items()}, flush=True)

LOG = f"{OUT}/results.jsonl"
per_item = {}


def evaluate(name, model, seed):
    model.eval()
    for split, (lays, items) in splits.items():
        for K in KS:
            t0 = time.time()
            m, pi = eval_conditioned(model, lays, items, K, seed=10_000 + seed)
            per_item[f"{name}|s{seed}|{split}|K{K}"] = pi
            rec = dict(model=name, seed=seed, split=split, K=K, **m)
            print(json.dumps(rec), f"({time.time() - t0:.0f}s)", flush=True)
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
    np.savez(f"{OUT}/per_item_s{SEEDS[0]}{'_p1' if os.environ.get('SKIP_REFS') else ''}.npz", **per_item)


# ---- references on the identical evaluation set (SKIP_REFS=1 to skip when already logged)
for seed in ([] if os.environ.get("SKIP_REFS") else SEEDS):
    for name, path in [("B1", f"{D5}/models/B1_s{seed}.pt"), ("B1full", f"{D5}/models/B1full_s{seed}.pt")]:
        m = VelocityMLP(60, cond_dim=21, hidden=512, depth=4)
        m.load_state_dict(torch.load(path))
        evaluate(name, m, seed)

# ---- P1, trained without the held-out combinations (same filter as B1)
train = stack_samples(train_layouts, code_filter=code_ok_for_train)
x = torch.tensor(train["x"])
cond = torch.tensor(np.concatenate([train["feat"], train["mask"], train["code"]], 1))
for seed in SEEDS:
    torch.manual_seed(seed)
    model = ObstacleAttnField(30, 3, d=128, depth=4, chunk=3)
    t0 = time.time()
    losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=seed, log_every=5000)
    torch.save(model.state_dict(), f"{OUT}/models/P1_s{seed}.pt")
    info = dict(model="P1", seed=seed, params=n_params(model), train_sec=round(time.time() - t0),
                final_loss=float(np.mean(losses[-500:])))
    print(json.dumps(info), flush=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(info) + "\n")
    evaluate("P1", model, seed)
print("done")

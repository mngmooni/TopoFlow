"""research phase: train B0 (no route code) on the cost-aware target marginal, then evaluate on the same
unseen layouts and paired noise as route_selection_study.py.

  REGIME=medium ARCH=attn python experiments/baselines/unconditional_attention_baseline.py     # B0-attn (P1 backbone, bits zeroed)
  REGIME=medium ARCH=mlp  python experiments/baselines/unconditional_attention_baseline.py     # B0-MLP reference

Sampling weights: layout uniform, then class h ~ p*(h|O), then a trajectory uniform within the class:
  w_i = p*(h_i | O_i) / n_{h_i}(O_i)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.cfm import VelocityMLP, n_params, train_cfm, train_cfm_ot  # noqa: E402
from topoflow.e2 import stack_samples  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import boltzmann, evaluate_samples, generate  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", max(1, os.cpu_count() or 1))))
OUT = "results/topology/route_selection"
REGIME = os.environ.get("REGIME", "medium")
ARCH = os.environ.get("ARCH", "attn")
SEED = int(os.environ.get("SEED", "0"))
# LAYOUT_W=uniform: every training layout gets equal total weight (matches the evaluation average).
# LAYOUT_W=codes:   layout weight proportional to its number of found codes, which is the effective
#                   layout distribution P1 saw under plain per-sample sampling.
LAYOUT_W = os.environ.get("LAYOUT_W", "uniform")
OT = os.environ.get("OT", "0") == "1"  # within-layout minibatch OT coupling (train_cfm_ot)
OT_GROUPS = int(os.environ.get("OT_GROUPS", "16"))
OT_PER = int(os.environ.get("OT_PER", "16"))
STEPS = 30000
KS = [2, 5, 20]
N_PER = 32

with open("results/baselines/variable_layout/layouts.pkl", "rb") as f:
    train_layouts, unseen = pickle.load(f)
with open(f"{OUT}/costs.pkl", "rb") as f:
    C = pickle.load(f)
beta = json.load(open(f"{OUT}/betas.json"))[REGIME]["beta"]

tr = stack_samples(train_layouts)  # all codes: B0 learns the full target marginal
x = torch.tensor(tr["x"])
# recover each sample's code tuple (unpadded) to look up p*
w = np.empty(len(x))
targets_tr = [boltzmann(c, beta) for c in C["train"]]
counts = [{c: len(P) for c, P in lay["found"].items()} for lay in train_layouts]
for i, (li, code) in enumerate(zip(tr["layout"], tr["code"])):
    m = train_layouts[li]["env"].m
    c = tuple(int(b) for b in code[:m])
    w[i] = targets_tr[li][c] / counts[li][c]
    if LAYOUT_W == "codes":
        w[i] *= len(counts[li])
print(f"{ARCH} {REGIME} beta={beta:.2f} samples={len(x)} weight sum per layout ~1: "
      f"{np.bincount(tr['layout'], weights=w)[:5].round(3)}", flush=True)

if ARCH == "attn":
    cond = torch.tensor(np.concatenate([tr["feat"], tr["mask"], tr["code"]], 1))
    model = ObstacleAttnField(30, 3, d=128, depth=4, chunk=3, use_code=False)
else:
    cond = torch.tensor(np.concatenate([tr["feat"], tr["mask"]], 1))
    model = VelocityMLP(60, cond_dim=18, hidden=512, depth=4)
torch.manual_seed(SEED)
suffix = ("_lwcodes" if LAYOUT_W == "codes" else "") + (("_ot" if OT_PER == 16 else f"_ot{OT_GROUPS}x{OT_PER}") if OT else "")
name = f"B0{ARCH}_{REGIME}_s{SEED}" + suffix
CKPT = f"{OUT}/models/{name}.ckpt"
t0 = time.time()
if OT:
    losses = train_cfm_ot(model, x, cond, tr["layout"], w, steps=STEPS, groups=OT_GROUPS,
                          per_group=OT_PER, seed=SEED, log_every=10000, ckpt=CKPT)
else:
    losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=SEED, weights=w, log_every=10000, ckpt=CKPT)
torch.save(model.state_dict(), f"{OUT}/models/{name}.pt")
if os.path.exists(CKPT):
    os.remove(CKPT)
model.eval()
info = dict(model=f"B0-{ARCH}" + suffix.replace("_", "-"), regime=REGIME, seed=SEED, params=n_params(model),
            train_sec=round(time.time() - t0), final_loss=float(np.mean(losses[-500:])))
print(json.dumps(info), flush=True)

if ARCH == "mlp":  # MLP takes 18-dim cond: wrap so generate() can pass 21-dim
    base = model
    model = lambda xx, tt, cc: base(xx, tt, cc[:, :18])  # noqa: E731

x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))
targets = [boltzmann(c, beta) for c in C["unseen"]]
dummy = np.zeros((len(unseen), N_PER), dtype=int)
for K in KS:
    G = generate(model, unseen, dummy, K, x0)
    rec = dict(model=info["model"], seed=SEED, regime=REGIME, K=K, train_sec=info["train_sec"],
               **evaluate_samples(unseen, G, targets, C["unseen"]))
    print(json.dumps(rec), flush=True)
    with open(f"{OUT}/results.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")

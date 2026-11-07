"""Train P1 (code-conditioned) or B0-attn (code-blind, cost-aware target) on D1 / D2 / D3 data.

  TOPO_MAX_OBS=5 MODEL=p1 python experiments/analysis/d1_train.py
  TOPO_MAX_OBS=5 MODEL=b0attn REGIME=medium [OT=1 OT_GROUPS=16 OT_PER=16] python experiments/analysis/d1_train.py
  DATA=results/d2 ... (same layout of files)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.cfm import n_params, train_cfm, train_cfm_ot  # noqa: E402
from topoflow.e2 import MAX_OBS, stack_samples  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import boltzmann  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "results/d1")
MODEL = os.environ.get("MODEL", "p1")
REGIME = os.environ.get("REGIME", "medium")
SEED = int(os.environ.get("SEED", "0"))
OT = os.environ.get("OT", "0") == "1"
OT_GROUPS = int(os.environ.get("OT_GROUPS", "16"))
OT_PER = int(os.environ.get("OT_PER", "16"))
STEPS = int(os.environ.get("STEPS", "30000"))
SG = os.environ.get("SG", "0") == "1"  # D3: start/goal y as extra input (P1 and B0-attn alike)
CLEAR_W = float(os.environ.get("CLEAR_W", "0"))  # clearance-aware loss weight (P1 only); 0 = plain CFM
os.makedirs(f"{DATA}/models", exist_ok=True)

with open(f"{DATA}/layouts.pkl", "rb") as f:
    train_layouts = pickle.load(f)[0]
tr = stack_samples(train_layouts)
x = torch.tensor(tr["x"])
cols = [tr["feat"], tr["mask"], tr["code"]]
if SG:
    cols.append(np.array([[train_layouts[li]["env"].start[1], train_layouts[li]["env"].goal[1]] for li in tr["layout"]],
                         np.float32))
cond = torch.tensor(np.concatenate(cols, 1))
model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=(MODEL == "p1"), sg_input=SG)
torch.manual_seed(SEED)
SUF = ("" if STEPS == 30000 else f"_{STEPS // 1000}k") + (f"_clr{CLEAR_W:g}" if CLEAR_W > 0 else "")
name = f"P1_s{SEED}{SUF}"
t0 = time.time()
if MODEL == "p1":
    aux = None
    if CLEAR_W > 0:
        from topoflow.clearance import endpoint_penalty
        cstar = float(json.load(open(f"{DATA}/betas.json")).get("_margin", 0.03))
        env0 = train_layouts[0]["env"]
        aux = endpoint_penalty(env0.start, env0.goal, MAX_OBS, cstar + 0.002, CLEAR_W, sub=4, t_power=2.0)
    losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=SEED, log_every=10000,
                       ckpt=f"{DATA}/models/{name}.ckpt", aux=aux)
else:
    with open(f"{DATA}/costs.pkl", "rb") as f:
        C = pickle.load(f)
    beta = json.load(open(f"{DATA}/betas.json"))[REGIME]["beta"]
    targets = [boltzmann(c, beta) for c in C["train"]]
    counts = [{c: len(P) for c, P in lay["found"].items()} for lay in train_layouts]
    w = np.empty(len(x))
    for i, (li, code) in enumerate(zip(tr["layout"], tr["code"])):
        c = tuple(int(b) for b in code[: train_layouts[li]["env"].m])
        w[i] = targets[li][c] / counts[li][c]
    name = f"B0attn_{REGIME}_s{SEED}" + (f"_ot{OT_GROUPS}x{OT_PER}" if OT else "") + SUF
    if OT:
        losses = train_cfm_ot(model, x, cond, tr["layout"], w, steps=STEPS, groups=OT_GROUPS,
                              per_group=OT_PER, seed=SEED, log_every=10000, ckpt=f"{DATA}/models/{name}.ckpt")
    else:
        losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=SEED, weights=w, log_every=10000,
                           ckpt=f"{DATA}/models/{name}.ckpt")
torch.save(model.state_dict(), f"{DATA}/models/{name}.pt")
print(json.dumps(dict(name=name, samples=len(x), params=n_params(model), train_sec=round(time.time() - t0),
                      final_loss=float(np.mean(losses[-500:])))), flush=True)
if os.path.exists(f"{DATA}/models/{name}.ckpt"):
    os.remove(f"{DATA}/models/{name}.ckpt")

"""Train TDFM v2 (chart flow) on D2-style data.

  TOPO_MAX_OBS=5 DATA=outputs/d2 TRAIN_CHUNK=12000 python scripts/v2_train.py   (exit 3 = rerun to resume)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import n_params, train_cfm  # noqa: E402
from topoflow.e2 import MAX_OBS  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.v2 import ChartField, training_set  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d2")
STEPS = int(os.environ.get("STEPS", "30000"))
SEED = int(os.environ.get("SEED", "0"))
SG = os.environ.get("SG", "0") == "1"  # D3: start/goal y as extra input
N = 60
CSTAR = float(json.load(open(f"{DATA}/betas.json"))["_margin"])
name = f"V2_s{SEED}"
cache = f"{DATA}/v2_train.npz"
if os.path.exists(cache):
    z = np.load(cache)
    U, Cnd, clip = z["U"], z["C"], float(z["clip"])
else:
    with open(f"{DATA}/layouts.pkl", "rb") as f:
        train_layouts = pickle.load(f)[0]
    U, Cnd, clip = training_set(train_layouts, CSTAR, N, sg=SG)
    np.savez(cache, U=U, C=Cnd, clip=clip)
model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True, sg_input=SG)
p1_params = n_params(model)
model = ChartField(n=N, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG)
print(json.dumps(dict(samples=len(U), clipped_frac=clip, params=n_params(model), p1_params=p1_params,
                      u_abs_p99=float(np.percentile(np.abs(U), 99)))), flush=True)
torch.manual_seed(SEED)
t0 = time.time()
losses = train_cfm(model, torch.tensor(U), cond=torch.tensor(Cnd), steps=STEPS, seed=SEED, log_every=10000,
                   ckpt=f"{DATA}/models/{name}.ckpt")
torch.save(model.state_dict(), f"{DATA}/models/{name}.pt")
print(json.dumps(dict(name=name, train_sec=round(time.time() - t0), final_loss=float(np.mean(losses[-500:])))),
      flush=True)
if os.path.exists(f"{DATA}/models/{name}.ckpt"):
    os.remove(f"{DATA}/models/{name}.ckpt")

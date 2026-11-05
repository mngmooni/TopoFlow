"""Train v2-VR (variable-resolution chart flow). Locked: outputs/predictions_day12_vr.md

  TOPO_MAX_OBS=5 DATA=outputs/d4 PRIOR=ou TRAIN_CHUNK=8000 python scripts/vr_train.py   (exit 3 = rerun to resume)
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from topoflow.cfm import n_params  # noqa: E402
from topoflow.vr import PerceiverVR, training_set, train_vr  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d4")
PRIOR = os.environ.get("PRIOR", "ou")
STEPS = int(os.environ.get("STEPS", "30000"))
SEED = int(os.environ.get("SEED", "0"))
C = float(json.load(open(f"{DATA}/betas.json"))["_margin"])
name = f"VR{PRIOR}_s{SEED}"
cache = f"{DATA}/vr_train.npz"
if os.path.exists(cache):
    z = np.load(cache)
    U, Cnd, info = z["U"], z["C"], json.loads(str(z["info"]))
else:
    train_layouts = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[0]
    U, Cnd, info = training_set(train_layouts, C, np.random.default_rng(0))
    np.savez(cache, U=U, C=Cnd, info=json.dumps(info))
model = PerceiverVR()
lens = Cnd[:, 7 * 5 + 3 * 120:].sum(1)
print(json.dumps(dict(**info, params=n_params(model), n_mean=float(lens.mean()), n_max=float(lens.max()))), flush=True)
torch.manual_seed(SEED)
t0 = time.time()
losses = train_vr(model, torch.tensor(U), torch.tensor(Cnd), steps=STEPS, seed=SEED, prior_kind=PRIOR,
                  ckpt=f"{DATA}/models/{name}.ckpt")
torch.save(model.state_dict(), f"{DATA}/models/{name}.pt")
print(json.dumps(dict(name=name, train_sec_this_run=round(time.time() - t0),
                      final_loss=float(np.mean(losses[-500:])))), flush=True)
if os.path.exists(f"{DATA}/models/{name}.ckpt"):
    os.remove(f"{DATA}/models/{name}.ckpt")

"""Bit-exact resume test: N steps straight vs k steps + resume (both trainers used for D4)."""
import os, sys, tempfile
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import cfm, vr
from topoflow.v2 import ChartField
from topoflow.e2 import MAX_OBS
torch.set_num_threads(1)
D = "results/d4"; N, K = 12, 5
z = np.load(f"{D}/vr_train.npz"); Uv, Cv = torch.tensor(z["U"][:2000]), torch.tensor(z["C"][:2000])
z = np.load(f"{D}/v2_train.npz") if os.path.exists(f"{D}/v2_train.npz") else None

def run(make, train, split):
    tmp = tempfile.mkdtemp(); ck = f"{tmp}/m.ckpt"
    torch.manual_seed(0); m = make()
    if split:
        vr.TRAIN_CHUNK = cfm.TRAIN_CHUNK = K
        try: train(m, ck)
        except SystemExit as e: assert e.code == 3
        else: raise RuntimeError('chunk exit not triggered')
        vr.TRAIN_CHUNK = cfm.TRAIN_CHUNK = 0
        torch.manual_seed(123); torch.randn(7)   # perturb global RNG, as a fresh process would
        m = make()
    l = train(m, ck)
    return torch.cat([p.detach().flatten() for p in m.state_dict().values()]), l

mk_vr = lambda: vr.PerceiverVR(d=32, depth=1)
tr_vr = lambda m, ck: vr.train_vr(m, Uv, Cv, steps=N, batch=32, prior_kind="ou", ckpt=ck, log_every=0)
a, la = run(mk_vr, tr_vr, False); b, lb = run(mk_vr, tr_vr, True)
print("VR   max|dW|", float((a - b).abs().max()), "loss equal", la == lb, len(la), len(lb))
if z is not None:
    Uc, Cc = torch.tensor(z["U"][:2000]), torch.tensor(z["C"][:2000])
    mk = lambda: ChartField(n=60, max_obs=MAX_OBS, d=32, depth=1, chunk=6)
    tr = lambda m, ck: cfm.train_cfm(m, Uc, cond=Cc, steps=N, batch=32, ckpt=ck)
    a, la = run(mk, tr, False); b, lb = run(mk, tr, True)
    print("v2   max|dW|", float((a - b).abs().max()), "loss equal", la == lb)

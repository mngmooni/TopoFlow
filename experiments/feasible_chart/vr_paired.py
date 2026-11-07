"""research phase exp 3: paired coarse/fine grid consistency for VR (locked: the archived prediction record).

Noise is drawn once on the fine grid (base + extra points); the coarse noise is its restriction to the base points.
For OU this restriction is exactly OU on the coarse grid (Markov); for iid it is iid. So each (coarse, fine) pair
shares one noise realization.
  TOPO_MAX_OBS=5 DATA=results/d4 PRIOR=ou N_EVAL=150 K=20 python experiments/feasible_chart/vr_paired.py
"""
import itertools, json, os, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import chart, vr  # noqa: E402

torch.set_num_threads(1)
DATA = os.environ.get("DATA", "results/d4"); PK = os.environ.get("PRIOR", "ou")
N_EVAL = int(os.environ.get("N_EVAL", "150")); K = int(os.environ.get("K", "20")); PER = 8
C = float(json.load(open(f"{DATA}/betas.json"))["_margin"])
lays = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:N_EVAL]
rng = np.random.default_rng(123)
gen = torch.Generator().manual_seed(7)
classes = []
for i, lay in enumerate(lays):
    env = lay["env"]
    for code in itertools.product([0, 1], repeat=env.m):
        base = vr.refined_grid(env, code, C)
        if base is None or not chart.bounds(env, code, C, 60)[2]:
            continue
        fine = np.unique(np.concatenate([base, rng.uniform(env.start[0], env.goal[0], min(40, vr.NMAX - len(base)))]))
        classes.append(dict(i=i, code=code, base=base, fine=fine, found=code in lay["found"]))
model = vr.PerceiverVR(); model.load_state_dict(torch.load(f"{DATA}/models/VR{PK}_s0.pt")); model.eval()

rows_c, rows_f, z_c, z_f, rows_i, z_i = [], [], [], [], [], []
for cl in classes:
    env = lays[cl["i"]]["env"]
    idx = np.searchsorted(cl["fine"], cl["base"])
    assert np.allclose(cl["fine"][idx], cl["base"])
    xf = torch.tensor(cl["fine"][None], dtype=torch.float32).repeat(PER, 1)
    zf = vr.prior(xf, torch.ones_like(xf), PK, gen).numpy()
    xb = torch.tensor(cl["base"][None], dtype=torch.float32).repeat(PER, 1)
    zi = vr.prior(xb, torch.ones_like(xb), PK, gen).numpy()   # independent coarse noise (reference scale)
    for p in range(PER):
        rows_f.append((env, cl["code"], C, cl["fine"])); z_f.append(zf[p])
        rows_c.append((env, cl["code"], C, cl["base"])); z_c.append(zf[p, idx])
        rows_i.append((env, cl["code"], C, cl["base"])); z_i.append(zi[p])
t0 = time.time()
Gf = vr.sample(model, rows_f, K, prior_kind=PK, x0=z_f)
Gc = vr.sample(model, rows_c, K, prior_kind=PK, x0=z_c)
Gi = vr.sample(model, rows_i, K, prior_kind=PK, x0=z_i)
sec = round(time.time() - t0)

L = lambda P: float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())  # noqa: E731


def newpt_dev(X, Y, is_new):
    """|y - linear interpolation between neighbouring shared points| at the new points."""
    Xs, Ys = X[~is_new], Y[~is_new]
    return np.abs(Y[is_new] - np.interp(X[is_new], Xs, Ys))


shared, indep, dlen, dlen_glob, dlen_loc, dev_m, dev_r, dlen_ref = [], [], [], [], [], [], [], []
for k, (Pf, Pc, Pi) in enumerate(zip(Gf, Gc, Gi)):
    cl = classes[k // PER]
    is_new = ~np.isin(Pf[:, 0], np.concatenate([[Pf[0, 0]], cl["base"], [Pf[-1, 0]]]))
    Pf_sh = Pf[~is_new]                         # fine path read at the shared points only
    shared.append(float(np.sqrt(np.mean((Pf_sh[1:-1, 1] - Pc[1:-1, 1]) ** 2))))
    indep.append(float(np.sqrt(np.mean((Pi[1:-1, 1] - Pc[1:-1, 1]) ** 2))))
    lc, lf, lfs = L(Pc), L(Pf), L(Pf_sh)
    dlen.append(lf / lc - 1); dlen_glob.append(lfs / lc - 1); dlen_loc.append((lf - lfs) / lc)
    dev_m += list(newpt_dev(Pf[:, 0], Pf[:, 1], is_new))
    if k % PER == 0 and cl["found"]:
        for q in lays[cl["i"]]["found"][cl["code"]]:
            X = Pf[:, 0]; Y = np.interp(X, q[:, 0], q[:, 1])
            dev_r += list(newpt_dev(X, Y, is_new))
            Xc = Pc[:, 0]; dlen_ref.append(L(np.stack([X, Y], 1)) / L(np.stack([Xc, np.interp(Xc, q[:, 0], q[:, 1])], 1)) - 1)
res = dict(prior=PK, K=K, classes=len(classes), n=len(Gf), sec=sec,
           shared_rms=float(np.mean(shared)), indep_rms=float(np.mean(indep)),
           shared_over_indep=float(np.mean(shared) / np.mean(indep)),
           len_rel=float(np.mean(dlen)), len_rel_global=float(np.mean(dlen_glob)), len_rel_local=float(np.mean(dlen_loc)),
           ref_len_rel=float(np.mean(dlen_ref)), newpt_dev_model=float(np.mean(dev_m)), newpt_dev_ref=float(np.mean(dev_r)),
           newpt_dev_ratio=float(np.mean(dev_m) / np.mean(dev_r)))
res["local_share_of_excess"] = (res["len_rel_local"] - res["ref_len_rel"]) / (res["len_rel"] - res["ref_len_rel"])
print(json.dumps(res), flush=True)
out = f"{DATA}/vr_paired.json"
allr = json.load(open(out)) if os.path.exists(out) else {}
allr[f"{PK}_K{K}"] = res
json.dump(allr, open(out, "w"), indent=1)

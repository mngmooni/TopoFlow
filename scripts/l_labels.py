"""L: does the conditioning label have to be topological? (D1, P1 backbone)

Label types (per layout, k = number of found codes):
  topo    : route code (existing P1_s0, not retrained here)
  cluster : k-means++ on the layout's 60-d paths, cluster ids ranked by mean waypoint y (descending)
  random  : the layout's paths split at random into k equal-size groups
Ids are written in binary into the same code slots P1 uses (slot j sits on obstacle token j, so for
cluster/random the bit pattern is an arbitrary id; for topo it means "obstacle j, below").

  TOPO_MAX_OBS=5 STAGE=train LABEL=cluster TRAIN_CHUNK=12000 python scripts/l_labels.py   (exit 3 = rerun)
  TOPO_MAX_OBS=5 STAGE=eval python scripts/l_labels.py
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
from topoflow.e2 import MAX_OBS, env_features, pad_code, stack_samples  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.taxonomy import tally  # noqa: E402
from topoflow.tdfm import boltzmann, code_index, index_code  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "outputs/d1")
OUT = os.environ.get("OUT", "outputs/l")
STAGE = os.environ.get("STAGE", "train")
LABEL = os.environ.get("LABEL", "cluster")
GLOBAL = os.environ.get("GLOBAL", "0") == "1"  # label as a global embedding instead of per-obstacle bits
STEPS = int(os.environ.get("STEPS", "30000"))
os.makedirs(f"{OUT}/models", exist_ok=True)
with open(f"{DATA}/layouts.pkl", "rb") as f:
    train_layouts, unseen = pickle.load(f)[:2]


def kmeans(X, k, rng, iters=20):
    c = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([((X - ci) ** 2).sum(1) for ci in c], axis=0)
        c.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
    c = np.array(c)
    for _ in range(iters):
        a = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([X[a == j].mean(0) if (a == j).any() else c[j] for j in range(k)])
    return ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1), c


def layout_labels(lay, kind, seed):
    """Returns per-path (code, path, label rank) in stack_samples order."""
    items = [(code, p) for code, P in lay["found"].items() for p in P]
    k = len(lay["found"])
    rng = np.random.default_rng(seed)
    if kind == "topo":
        return [(c, p, None) for c, p in items]
    X = np.array([p[1:-1].reshape(-1) for _, p in items])
    if kind == "cluster":
        a, cent = kmeans(X, k, rng)
        ymean = cent.reshape(k, -1, 2)[:, :, 1].mean(1)
        rank = np.empty(k, int)
        rank[np.argsort(-ymean)] = np.arange(k)
        lab = rank[a]
    else:
        perm = rng.permutation(len(items))
        lab = np.empty(len(items), int)
        for j, part in enumerate(np.array_split(perm, k)):
            lab[part] = j
    return [(c, p, int(l)) for (c, p), l in zip(items, lab)]


def label_sets(kind):
    path = f"{OUT}/labels_{kind}.pkl"
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    res = {s: [layout_labels(l, kind, 1000 * (s == "unseen") + i) for i, l in enumerate(L)]
           for s, L in [("train", train_layouts), ("unseen", unseen)]}
    pickle.dump(res, open(path, "wb"))
    return res


def purity(sets):
    """Share of paths whose label's majority crossing-word code equals their own code."""
    tot = ok = 0
    for rows in sets:
        by = {}
        for c, _, l in rows:
            by.setdefault(l, []).append(c)
        for cs in by.values():
            vals, cnt = np.unique([code_index(c) for c in cs], return_counts=True)
            ok += cnt.max()
            tot += len(cs)
    return ok / tot


if STAGE == "train":
    S = label_sets(LABEL)
    tr = stack_samples(train_layouts)
    lab = np.concatenate([[code_index(c) if LABEL == "topo" else l for c, _, l in rows] for rows in S["train"]])
    assert len(lab) == len(tr["x"])
    ms = np.array([train_layouts[li]["env"].m for li in tr["layout"]])
    code = np.array([pad_code(index_code(int(l), int(m))) for l, m in zip(lab, ms)], np.float32)
    x = torch.tensor(tr["x"])
    cond = torch.tensor(np.concatenate([tr["feat"], tr["mask"], code], 1))
    model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True, label_global=GLOBAL)
    torch.manual_seed(0)
    name = f"P1{LABEL}{'G' if GLOBAL else ''}_s0"
    t0 = time.time()
    losses = train_cfm(model, x, cond=cond, steps=STEPS, seed=0, log_every=10000, ckpt=f"{OUT}/models/{name}.ckpt")
    torch.save(model.state_dict(), f"{OUT}/models/{name}.pt")
    print(json.dumps(dict(name=name, samples=len(x), params=n_params(model), train_sec=round(time.time() - t0),
                          final_loss=float(np.mean(losses[-500:])),
                          purity_train=None if LABEL == "topo" else purity(S["train"]),
                          purity_unseen=None if LABEL == "topo" else purity(S["unseen"]))), flush=True)
    os.remove(f"{OUT}/models/{name}.ckpt")

else:  # eval: oracle label distribution on unseen layouts, same x0 for every label type
    with open(f"{DATA}/costs.pkl", "rb") as f:
        C = pickle.load(f)
    meta = json.load(open(f"{DATA}/betas.json"))
    REGIME = os.environ.get("REGIME", "medium")
    SPLIT = os.environ.get("SPLIT", "unseen")  # "train": first 300 training layouts (memorization check)
    if SPLIT == "train":
        unseen = train_layouts[:300]
        C["unseen"] = C["train"][:300]
    targets = [boltzmann(c, meta[REGIME]["beta"]) for c in C["unseen"]]
    CSTAR = float(meta.get("_margin", 0.03))
    N_PER, KS = 32, [int(k) for k in os.environ.get("KS", "2,5,20").split(",")]
    x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))
    g = np.random.default_rng(900)
    runs = [("topo", f"{DATA}/models/P1_s0.pt"), ("cluster", f"{OUT}/models/P1cluster_s0.pt"),
            ("random", f"{OUT}/models/P1random_s0.pt"), ("none (B0-attn)", f"{DATA}/models/B0attn_{REGIME}_s0.pt"),
            ("clusterG", f"{OUT}/models/P1clusterG_s0.pt"), ("topoG", f"{OUT}/models/P1topoG_s0.pt")]
    ONLYK = [s for s in os.environ.get("ONLY", "").split(",") if s]
    runs = [r for r in runs if (not ONLYK or r[0] in ONLYK) and os.path.exists(r[1])]
    for kind, mpath in runs:
        blind = kind.startswith("none")
        model = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=not blind,
                                  label_global=kind.endswith("G"))
        model.load_state_dict(torch.load(mpath))
        model.eval()
        lk = "topo" if blind else kind.rstrip("G")
        S = label_sets(lk)[SPLIT][:len(unseen)]
        conds, req = [], []
        g = np.random.default_rng(900)  # same draws per kind
        for i, (lay, rows) in enumerate(zip(unseen, S)):
            f, m = env_features(lay["env"])
            ncode = {c: len(P) for c, P in lay["found"].items()}
            labs = [c if kind == "topo" or blind else l for c, _, l in rows]
            if kind == "topoG":  # topology class as a global id: binary of the code itself
                labs = [code_index(c) for c, _, _ in rows]
            keys = sorted(set(labs), key=lambda z: code_index(z) if isinstance(z, tuple) else z)
            q = np.zeros(len(keys))
            maj = {}
            for (c, _, l), lab in zip(rows, labs):
                q[keys.index(lab)] += targets[i][c] / ncode[c]
                maj.setdefault(lab, []).append(c)
            maj = {lab: max(set(cs), key=cs.count) for lab, cs in maj.items()}
            draw = g.choice(len(keys), size=N_PER, p=q / q.sum())
            for d in draw:
                lab = keys[d]
                bits = pad_code(lab) if isinstance(lab, tuple) else pad_code(index_code(lab, lay["env"].m))
                conds.append(np.concatenate([f, m, bits]))
                req.append(code_index(maj[lab]))  # topology the label "stands for" = its majority code
        cond = torch.tensor(np.array(conds), dtype=torch.float32)
        req = np.array(req).reshape(len(unseen), N_PER)
        for K in KS:
            outs = []
            with torch.no_grad():
                for s in range(0, len(cond), 4096):
                    xb, cb = x0[s:s + 4096].clone(), cond[s:s + 4096]
                    for k in range(K):
                        xb = xb + model(xb, torch.full((len(xb),), k / K), cb) / K
                    outs.append(xb)
            G = torch.cat(outs).numpy().reshape(len(unseen), N_PER, 30, 2)
            rec = dict(label=kind, split=SPLIT, regime=REGIME, K=K, cstar=CSTAR,
                       purity_unseen=None if (blind or kind == "topo") else purity(S),
                       c0=tally(unseen, G, None if blind else req, 0.0, targets, index_code),
                       cstar_res=tally(unseen, G, None if blind else req, CSTAR, targets, index_code),
                       # label-agnostic view: any sample with clearance >= c* and a found word counts as valid
                       blind_c0=tally(unseen, G, None, 0.0, targets, index_code),
                       blind_cstar=tally(unseen, G, None, CSTAR, targets, index_code))
            ms = np.array([l["env"].m for l in unseen])
            for mm in sorted(set(ms)):
                idx = np.where(ms == mm)[0]
                rec[f"m{mm}"] = tally([unseen[i] for i in idx], G[idx], None if blind else req[idx], CSTAR,
                                      [targets[i] for i in idx], index_code)
            print(json.dumps(rec), flush=True)
            with open(f"{OUT}/results{'' if SPLIT == 'unseen' else '_' + SPLIT}.jsonl", "a") as fo:
                fo.write(json.dumps(rec) + "\n")
    print("done")

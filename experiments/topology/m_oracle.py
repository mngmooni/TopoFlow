"""M: oracle-based error decomposition on E2 (medium regime, unseen layouts).

Fields
  oracle-marg : exact marginal velocity of the p*-weighted reference paths of each layout (code-blind)
  oracle-cond : exact velocity of the reference paths of the sampled code only
  B0-attn     : learned, code-blind (research phase checkpoint)
  B0-attn-ot  : learned, code-blind, OT-CFM (research phase checkpoint)
  P1          : learned, code-conditioned, codes ~ p* (research phase checkpoint)
Solvers at matched NFE: Euler (K steps) and midpoint (K/2 steps, 2 evals each). Neither ever evaluates
the field at t = 1, where the exact field is singular.

Decomposition (per field, NFE n)
  integration error  : metric(n) - metric(n_ref), n_ref = 200 midpoint
  field error        : learned(n_ref) vs oracle(n_ref) on the same x0, as final-sample RMS
                       (includes generalization, because the oracle reproduces the reference set)

  python experiments/topology/m_oracle.py         (N_LAYOUTS=100 by default)
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import env_features, pad_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import boltzmann, code_index, evaluate_samples, index_code, sample_codes, target_vector  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
N_LAY = int(os.environ.get("N_LAYOUTS", "100"))
N_PER = 32
NFES = [2, 4, 8, 20, 50, 100]
N_REF = 200
OUT = "results/m"
os.makedirs(OUT, exist_ok=True)

with open("results/baselines/variable_layout/layouts.pkl", "rb") as f:
    unseen = pickle.load(f)[1][:N_LAY]
with open("results/topology/route_selection/costs.pkl", "rb") as f:
    costs = pickle.load(f)["unseen"][:N_LAY]
beta = json.load(open("results/topology/route_selection/betas.json"))["medium"]["beta"]
targets = [boltzmann(c, beta) for c in costs]
L = len(unseen)
codes = sample_codes(np.array([target_vector(t, None) for t in targets]), N_PER, seed=900)  # (L, N_PER)
x0 = torch.randn(L * N_PER, 60, generator=torch.Generator().manual_seed(777))

# ---- per-sample oracle data: padded reference set + log weights (marginal and conditional)
nmax = max(sum(len(P) for P in l["found"].values()) for l in unseen)
D = torch.zeros(L, nmax, 60)
LW_marg = torch.full((L, nmax), -torch.inf)
CODE = torch.full((L, nmax), -1, dtype=torch.long)
for i, lay in enumerate(unseen):
    j = 0
    for c, P in lay["found"].items():
        for p in P:
            D[i, j] = torch.tensor(p[1:-1].reshape(-1), dtype=torch.float32)
            LW_marg[i, j] = float(np.log(max(targets[i][c], 1e-300) / len(P)))
            CODE[i, j] = code_index(c)
            j += 1
lay_of = torch.arange(L).repeat_interleave(N_PER)
code_of = torch.tensor(codes.reshape(-1))
LW_cond = torch.where(CODE[lay_of] == code_of[:, None], 0.0, -torch.inf)  # (L*N_PER, nmax)


def oracle_field(lw_rows):
    def f(x, t):
        s = 1.0 - t
        Dx = D[lay_of]                                            # (N, nmax, 60)
        d2 = ((x[:, None, :] - t * Dx) ** 2).sum(-1)
        w = torch.softmax(-d2 / (2 * s * s) + lw_rows, dim=1)
        return ((w[..., None] * Dx).sum(1) - x) / s
    return f


conds = torch.tensor(np.array([np.concatenate([*env_features(unseen[i]["env"]),
                                               pad_code(index_code(int(codes[i, j]), 3))])
                               for i in range(L) for j in range(N_PER)]), dtype=torch.float32)


def learned_field(model):
    return lambda x, t: model(x, torch.full((len(x),), t), conds)


@torch.no_grad()
def solve(f, nfe, method):
    x = x0.clone()
    if method == "euler":
        for k in range(nfe):
            x = x + f(x, k / nfe) / nfe
    else:  # midpoint, nfe/2 steps
        K = nfe // 2
        h = 1.0 / K
        for k in range(K):
            t = k * h
            xm = x + 0.5 * h * f(x, t)
            x = x + h * f(xm, t + 0.5 * h)
    return x


def metrics(G):
    r = evaluate_samples(unseen, G.numpy().reshape(L, N_PER, 30, 2), targets, costs)
    return {k: r[k] for k in ("collision", "deep", "invalid", "tv_joint")}


fields = {"oracle-marg": oracle_field(LW_marg[lay_of]), "oracle-cond": oracle_field(LW_cond)}
for name, path, use_code in [("B0-attn", "results/topology/route_selection/models/B0attn_medium_s0.pt", False),
                             ("B0-attn-ot", "results/topology/route_selection/models/B0attn_medium_s0_ot.pt", False),
                             ("P1", "results/baselines/obstacle_attention/models/P1_s0.pt", True)]:
    m = ObstacleAttnField(30, 3, d=128, depth=4, chunk=3, use_code=use_code)
    m.load_state_dict(torch.load(path))
    fields[name] = learned_field(m.eval())

ref = {}
# resumable per field: a finished field leaves {OUT}/ref_{name}.pt; its rows are already in results.jsonl
with open(f"{OUT}/results.jsonl", "a") as fo:
    for name, f in fields.items():
        rp = f"{OUT}/ref_{name}.pt"
        if os.path.exists(rp):
            ref[name] = torch.load(rp)
            print(f"skip {name} (done)", flush=True)
            continue
        ref[name] = solve(f, N_REF, "midpoint")
        rec = dict(field=name, method="midpoint", nfe=N_REF, **metrics(ref[name]))
        print(json.dumps(rec), flush=True)
        fo.write(json.dumps(rec) + "\n")
        for method in ["euler", "midpoint"]:
            for n in NFES:
                G = solve(f, n, method)
                rec = dict(field=name, method=method, nfe=n, **metrics(G),
                           rms_to_own_ref=float((G - ref[name]).pow(2).mean().sqrt()))
                print(json.dumps(rec), flush=True)
                fo.write(json.dumps(rec) + "\n")
        fo.flush()
        torch.save(ref[name], rp)
    # field (approximation + generalization) error at the reference solve, same x0
    for learned, oracle in [("B0-attn", "oracle-marg"), ("B0-attn-ot", "oracle-marg"), ("P1", "oracle-cond")]:
        rec = dict(kind="field_gap", learned=learned, oracle=oracle,
                   rms=float((ref[learned] - ref[oracle]).pow(2).mean().sqrt()))
        print(json.dumps(rec), flush=True)
        fo.write(json.dumps(rec) + "\n")
print("done")

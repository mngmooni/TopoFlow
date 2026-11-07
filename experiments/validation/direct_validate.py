"""Validate topoflow/direct.py before any comparison run (D2, few layouts). Timing here is not used."""
import json, os, pickle, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import direct  # noqa: E402
from topoflow.e2 import MAX_OBS, word_to_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import generate, index_code, sample_codes  # noqa: E402
from topoflow.topology import crossing_word  # noqa: E402

torch.set_num_threads(1)
DATA = "results/d2"; NL = int(os.environ.get("NL", "20")); N_PER = 32
C = float(json.load(open(f"{DATA}/betas.json"))["_margin"])
lays = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:NL]
rng = np.random.default_rng(0)
codes = np.stack([rng.choice([i for i in range(32) if all(((i >> j) & 1) == 0 for j in range(l["env"].m, 5))], N_PER) for l in lays])
x0 = torch.randn(NL * N_PER, 60, generator=torch.Generator().manual_seed(777))
p1 = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True); p1.load_state_dict(torch.load(f"{DATA}/models/P1_s0.pt")); p1.eval()
req = [(i, index_code(int(codes[i, j]), lays[i]["env"].m)) for i in range(NL) for j in range(N_PER)]
cond = direct.build_cond(lays, req, True)


def check(paths):
    U = M = T = 0
    for (i, code), P in zip(req, paths):
        env = lays[i]["env"]; P = P[None]
        u = env.collision_mask(P)[0]; m = env.collision_mask(P, margin=C)[0]
        U += u; M += (m and not u)
        T += (not m) and word_to_code(crossing_word(env, P[0]), env.m) != code
    n = len(paths); return dict(U=U / n, M=M / n, T=T / n)


out = {}
for K in (5, 20):
    G = generate(p1, lays, codes, K, x0)
    ref = [np.concatenate([[l["env"].start], G[i, j], [l["env"].goal]]) for i, l in enumerate(lays) for j in range(N_PER)]
    P0, _ = direct.sample(p1, lays, req, cond, x0, K, "none", C)
    out[f"K{K}_none_maxdiff"] = float(max(np.abs(a - b).max() for a, b in zip(P0, ref)))
    out[f"K{K}_raw"] = check(P0)
    for it in (20, 100, 300):
        Pf, info = direct.sample(p1, lays, req, cond, x0, K, "final", C, iters=it)
        out[f"K{K}_final_it{it}"] = dict(**check(Pf), res_pos=info["residual_pos_frac"], res_max=info["residual_max"],
                                         shift=float(np.mean([np.abs(a - b).max() for a, b in zip(Pf, P0)])))
    Pc, info = direct.sample(p1, lays, req, cond, x0, K, "chart", C)
    out[f"K{K}_chart"] = dict(**check(Pc), chart_infeasible=info["chart_infeasible"])
    print(K, json.dumps({k: v for k, v in out.items() if k.startswith(f"K{K}")}), flush=True)

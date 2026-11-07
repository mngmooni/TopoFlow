"""research phase exp 1: end-to-end B2 under the original vs revised target (locked: the archived prediction record).

Selectors: orig (sel_costreg.pt, trained on T0 = found support + representative cost; used in all earlier results)
and T1 (sel_relabel_T1.pt, trained on T1 = chart support + chart-constrained cost J_c), plus oracle codes drawn from
each target (finite-sample floor). Generators: P1_s0 and V2_s0 (n = 60 chart). Medium regime, NFE 2/5/20,
32 samples per layout, codes seed 900, noise seed 777 (as research phase–11). No training.
Validity: collision-free at clearance c (primary) and at 0, and realized code in the target's support.
  TOPO_MAX_OBS=5 DATA=results/d2 python experiments/topology/b2_revised.py ;  DATA=results/d3 SG=1 ...
"""
import json, os, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import MAX_OBS, word_to_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.selectors import CostRegSelector, probs_from_cost  # noqa: E402
from topoflow.tdfm import boltzmann, generate, index_code, sample_codes, selector_inputs, target_vector  # noqa: E402
from topoflow.topology import crossing_word  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample as v2_sample  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "1")))
DATA = os.environ.get("DATA", "results/d2"); SG = os.environ.get("SG", "0") == "1"
N_PER, KS = 32, [int(k) for k in os.environ.get("KS", "2,5,20").split(",")]
N_EVAL = int(os.environ.get("N_EVAL", "300"))
meta = json.load(open(f"{DATA}/betas.json")); C = float(meta["_margin"])
unseen = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:N_EVAL]
Crep = pickle.load(open(f"{DATA}/costs.pkl", "rb"))["unseen"][:N_EVAL]
LAB = pickle.load(open(f"{DATA}/chart_labels.pkl", "rb"))["unseen"][:N_EVAL]
B = json.load(open(f"{DATA}/relabel.json"))["beta"]
L = len(unseen)
TGT = {"T0": [boltzmann(c, B["T0"]["medium"]) for c in Crep], "T1": [boltzmann(c, B["T1"]["medium"]) for c in LAB]}

X, S = selector_inputs(unseen, sg=SG); S = S.numpy()
probs = {}
for name, path, T in (("orig", "sel_costreg.pt", "T0"), ("T1sel", "sel_relabel_T1.pt", "T1")):
    sel = CostRegSelector(512, 4, extra=2 if SG else 0); sel.load_state_dict(torch.load(f"{DATA}/models/{path}")); sel.eval()
    with torch.no_grad():
        jh, fl = (t.numpy() for t in sel(X))
    feas = (fl > 0) & S; feas = np.where(feas.any(1, keepdims=True), feas, S)
    probs[name] = probs_from_cost(jh, feas, B[T]["medium"])
for T in ("T0", "T1"):
    probs[f"oracle{T}"] = np.array([target_vector(t, None) for t in TGT[T]])
codes = {k: sample_codes(p, N_PER, seed=900) for k, p in probs.items()}
x0 = torch.randn(L * N_PER, 60, generator=torch.Generator().manual_seed(777))

p1 = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True, sg_input=SG)
p1.load_state_dict(torch.load(f"{DATA}/models/P1_s0.pt")); p1.eval()
v2 = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG)
v2.load_state_dict(torch.load(f"{DATA}/models/V2_s0.pt")); v2.eval()
cache = ChartCache(unseen, C, 60, sg=SG)


def paths_for(gen, cidx, K):
    if gen == "P1":
        G = generate(p1, unseen, cidx, K, x0, sg=SG)
        out = []
        for i, l in enumerate(unseen):
            for j in range(N_PER):
                out.append(np.concatenate([[l["env"].start], G[i, j], [l["env"].goal]]))
        return out, np.ones(L * N_PER, bool)
    req = [(i, index_code(int(cidx[i, j]), unseen[i]["env"].m)) for i in range(L) for j in range(N_PER)]
    okc = np.array([cache.get(i, c)[3] for i, c in req])
    return list(v2_sample(v2, cache, req, K, x0)), okc


def evaluate(paths, okc):
    rec = {k: [] for k in ("coll0", "collc", "chart_infeasible", "tvT0_c", "tvT1_c", "tvT1_0", "invT1_c", "covmassT1", "covmassT0")}
    for i, l in enumerate(unseen):
        env = l["env"]; sl = slice(i * N_PER, (i + 1) * N_PER)
        P = np.stack(paths[sl]); ok = okc[sl]
        u0 = env.collision_mask(P); uc = env.collision_mask(P, margin=C)
        rc = [word_to_code(crossing_word(env, p), env.m) for p in P]
        rec["coll0"].append(u0.mean()); rec["collc"].append(uc.mean()); rec["chart_infeasible"].append((~ok).mean())
        for T, bad, key in (("T0", uc, "tvT0_c"), ("T1", uc, "tvT1_c"), ("T1", u0, "tvT1_0")):
            tgt = TGT[T][i]
            valid = [(not b) and o and (c in tgt) for b, o, c in zip(bad, ok, rc)]
            cnt = {c: 0 for c in tgt}
            for c, v in zip(rc, valid):
                if v:
                    cnt[c] += 1
            inv = N_PER - sum(cnt.values())
            rec[key].append(0.5 * (sum(abs(cnt[c] / N_PER - tgt[c]) for c in tgt) + inv / N_PER))
            if key == "tvT1_c":
                rec["invT1_c"].append(inv / N_PER)
            if key in ("tvT0_c", "tvT1_c"):
                rec[f"covmass{T}"].append(sum(tgt[c] for c in tgt if cnt[c] > 0))
    return {k: float(np.mean(v)) for k, v in rec.items()}


OUT = f"{DATA}/b2_revised.jsonl"
for gen in ("P1", "v2"):
    for src in ("orig", "T1sel", "oracleT0", "oracleT1"):
        for K in KS:
            t0 = time.time()
            r = dict(data=DATA, gen=gen, codes=src, K=K, **evaluate(*paths_for(gen, codes[src], K)), sec=round(time.time() - t0))
            print(json.dumps(r), flush=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(r) + "\n")
print("done")

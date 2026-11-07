"""research phase: DiRecT-style external baseline vs v2 (locked: the archived prediction record). No training.

Arms: v2 | P1 raw | P1 + final projection | P1 + DiRecT-free | P1 + DiRecT-chart | B0 + DiRecT-free.
Same layouts (unseen), codes from the T1 selector (seed 900), noise seed 777, 32 per layout, clearance c, NFE 2/5/20.
Checkpoints: V2_s0, P1_s0, B0attn_medium_s0, sel_relabel_T1 (unchanged). Appends to {DATA}/direct.jsonl.
  TOPO_MAX_OBS=5 DATA=results/d2 ARMS=... KS=2,5,20 ITERS=100 python experiments/validation/direct_eval.py
"""
import itertools, json, os, pickle, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow import direct  # noqa: E402
from topoflow.e2 import MAX_OBS, word_to_code  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.selectors import CostRegSelector, probs_from_cost  # noqa: E402
from topoflow.taxonomy import classify_samples  # noqa: E402
from topoflow.tdfm import boltzmann, index_code, sample_codes, selector_inputs  # noqa: E402
from topoflow.v2 import ChartCache, ChartField, sample as v2_sample  # noqa: E402

torch.set_num_threads(1)
DATA = os.environ.get("DATA", "results/d2"); SG = os.environ.get("SG", "0") == "1"
ARMS = os.environ.get("ARMS", "v2,p1,p1final,p1free,p1chart,b0free").split(",")
KS = [int(k) for k in os.environ.get("KS", "2,5,20").split(",")]
ITERS = int(os.environ.get("ITERS", "100")); SUB = int(os.environ.get("SUB", "16"))
N_EVAL = int(os.environ.get("N_EVAL", "300")); N_PER = 32
OUT = os.environ.get("OUT", f"{DATA}/direct.jsonl")
meta = json.load(open(f"{DATA}/betas.json")); C = float(meta["_margin"])
unseen = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:N_EVAL]
LAB = pickle.load(open(f"{DATA}/chart_labels.pkl", "rb"))["unseen"][:N_EVAL]
B = json.load(open(f"{DATA}/relabel.json"))["beta"]
L = len(unseen)
TGT = [boltzmann(c, B["T1"]["medium"]) for c in LAB]
X, S = selector_inputs(unseen, sg=SG); S = S.numpy()
sel = CostRegSelector(512, 4, extra=2 if SG else 0); sel.load_state_dict(torch.load(f"{DATA}/models/sel_relabel_T1.pt")); sel.eval()
with torch.no_grad():
    jh, fl = (t.numpy() for t in sel(X))
feas = (fl > 0) & S; feas = np.where(feas.any(1, keepdims=True), feas, S)
codes = sample_codes(probs_from_cost(jh, feas, B["T1"]["medium"]), N_PER, seed=900)
x0 = torch.randn(L * N_PER, 60, generator=torch.Generator().manual_seed(777))
req = [(i, index_code(int(codes[i, j]), unseen[i]["env"].m)) for i in range(L) for j in range(N_PER)]
XG = np.linspace(-0.9, 0.9, 62)[1:-1]


def load(stem, use_code):
    m = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=use_code, sg_input=SG)
    m.load_state_dict(torch.load(f"{DATA}/models/{stem}.pt")); return m.eval()


def evaluate(paths, code_blind=False):
    cnt = {k: 0 for k in "UMTNV"}; tvs, divs, lens, rough = [], [], [], []
    for i, lay in enumerate(unseen):
        env = lay["env"]; P = np.stack(paths[i * N_PER:(i + 1) * N_PER])
        rq = [req[i * N_PER + j][1] for j in range(N_PER)]
        cats, words = classify_samples(env, P, None if code_blind else rq, set(lay["found"]), C)
        for c in cats:
            cnt[c] += 1
        tgt = TGT[i]; hist = {c: 0 for c in tgt}
        uc = env.collision_mask(P, margin=C)
        for b, w in zip(uc, words):
            if (not b) and w in hist:
                hist[w] += 1
        tvs.append(0.5 * (sum(abs(hist[c] / N_PER - tgt[c]) for c in tgt) + (N_PER - sum(hist.values())) / N_PER))
        Y = np.array([np.interp(XG, p[:, 0], p[:, 1]) for p in P])
        rough += list(np.mean(np.diff(Y, 2, axis=1) ** 2, 1))
        grp = words if code_blind else rq
        for code in set(g for g in grp if g is not None):
            idx = [j for j in range(N_PER) if grp[j] == code and cats[j] == "V"]
            if len(idx) >= 2:
                d = np.sqrt(((Y[idx][:, None] - Y[idx][None]) ** 2).mean(-1)); divs.append(d[np.triu_indices(len(idx), 1)].mean())
            ref = lay["found"].get(code)
            if ref is not None:
                rl = min(np.linalg.norm(np.diff(r, axis=0), axis=-1).sum() for r in ref)
                lens += [np.linalg.norm(np.diff(P[j], axis=0), axis=-1).sum() / rl for j in idx]
    tot = L * N_PER
    return dict(**{k: cnt[k] / tot for k in cnt}, tv_T1=float(np.mean(tvs)), diversity=float(np.mean(divs)),
                length_ratio=float(np.mean(lens)), roughness=float(np.mean(rough)))


def log(r):
    print(json.dumps(r), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(r) + "\n")


# Chunked execution (added before any DiRecT-free/B0 result): CHUNK=a:b samples layouts a..b-1 with the SAME
# codes and noise rows as the full run and pickles the paths; MERGE=1 evaluates the concatenated chunks.
CHUNK = os.environ.get("CHUNK"); MERGE = os.environ.get("MERGE") == "1"
CHD = os.environ.get("CHUNKDIR", f"{DATA}/direct_chunks")
PKL = lambda arm, K, a: f"{CHD}/{arm}_K{K}_it{ITERS}_{a}.pkl"  # noqa: E731
if CHUNK or MERGE:
    os.makedirs(CHD, exist_ok=True)
cache_models = {}
for arm in ARMS:
    if CHUNK or MERGE:
        blind = arm.startswith("b0")
        mode = {"p1": "none", "p1final": "final", "p1free": "free", "p1chart": "chart", "b0free": "free"}[arm]
        for K in KS:
            if CHUNK:
                a, b = (int(v) for v in CHUNK.split(":"))
                sl = slice(a * N_PER, b * N_PER)
                m = load("B0attn_medium_s0" if blind else "P1_s0", not blind)
                cond = direct.build_cond(unseen, req, not blind, sg=SG)[sl]
                t0 = time.perf_counter()
                paths, info = direct.sample(m, unseen, req[sl], cond, x0[sl], K, mode, C, iters=ITERS, sub=SUB)
                info["sec"] = time.perf_counter() - t0; info["n"] = len(paths)
                pickle.dump((paths, info), open(PKL(arm, K, a) + ".tmp", "wb")); os.replace(PKL(arm, K, a) + ".tmp", PKL(arm, K, a))  # atomic
                print("chunk", arm, K, a, b, round(info["sec"]), flush=True)
            else:
                import glob as _g
                parts = sorted(_g.glob(PKL(arm, K, "*")), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
                paths, infos = [], []
                for f in parts:
                    p, inf = pickle.load(open(f, "rb")); paths += p; infos.append(inf)
                assert len(paths) == len(req), (len(paths), len(req))
                n = sum(i["n"] for i in infos); sec = sum(i["sec"] for i in infos)
                info = dict(t_field=sum(i["t_field"] for i in infos), t_proj=sum(i["t_proj"] for i in infos),
                            residual_mean=sum(i["residual_mean"] * i["n"] for i in infos) / n,
                            residual_max=max(i["residual_max"] for i in infos),
                            residual_pos_frac=sum(i["residual_pos_frac"] * i["n"] for i in infos) / n,
                            chart_infeasible=sum(i["chart_infeasible"] * i["n"] for i in infos) / n, chunks=len(parts))
                log(dict(data=DATA, arm=arm, K=K, iters=ITERS, sub=SUB, sec=round(sec, 2),
                         ms_per_sample=round(1000 * sec / n, 3), **info, **evaluate(paths, blind)))
        continue
    for K in KS:
        t0 = time.perf_counter(); info = {}
        if arm == "v2":
            m = cache_models.setdefault("v2", ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG))
            m.load_state_dict(torch.load(f"{DATA}/models/V2_s0.pt")); m.eval()
            cache = ChartCache(unseen, C, 60, sg=SG)          # chart construction counted in the time
            paths = list(v2_sample(m, cache, req, K, x0)); blind = False
        else:
            blind = arm.startswith("b0")
            m = load("B0attn_medium_s0" if blind else "P1_s0", not blind)
            mode = {"p1": "none", "p1final": "final", "p1free": "free", "p1chart": "chart", "b0free": "free"}[arm]
            cond = direct.build_cond(unseen, req, not blind, sg=SG)
            paths, info = direct.sample(m, unseen, req, cond, x0, K, mode, C, iters=ITERS, sub=SUB)
        sec = time.perf_counter() - t0
        r = dict(data=DATA, arm=arm, K=K, iters=ITERS if arm not in ("v2", "p1", "p1chart") else None, sub=SUB,
                 sec=round(sec, 2), ms_per_sample=round(1000 * sec / len(req), 3), **info, **evaluate(paths, blind))
        log(r)
print("done")

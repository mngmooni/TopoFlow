"""Experiment B — standardized latency (locked: the frozen prediction record (private archive)).

Same host, 1 CPU thread, batch = 320 samples (first 10 unseen layouts x 32 research phase requests), each configuration in
its own subprocess (peak RSS = ru_maxrss of that process), 1 warm-up + R timed repeats, median and IQR.
Timed region: everything per batch after model loading, incl. chart construction (ChartCache / ChartProj).

  python experiments/validation/baseline_latency.py            -> results/final/baseline_latency.json (resumable, appends .jsonl)
"""
import json, os, subprocess, sys, time  # noqa: E401

OUT = "results/final/baseline_latency.jsonl"
CFGS = []
for ds in ("d2", "d3"):
    CFGS += [dict(ds=ds, arm="v2", K=k) for k in (1, 2, 5, 10, 20, 50)]
    CFGS += [dict(ds=ds, arm="p1chart", K=k) for k in (2, 5, 10, 20, 50)]
    CFGS += [dict(ds=ds, arm="p1", K=k) for k in (5, 20)]
    CFGS += [dict(ds=ds, arm=a, K=k, iters=it) for a in ("p1final", "p1free") for it in (20, 100) for k in (5, 20)]
    CFGS += [dict(ds=ds, arm="b0free", K=k, iters=100) for k in (5, 20)]


def worker(cfg):
    import resource
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    torch.set_num_threads(1)
    os.environ["DATA"] = f"results/{cfg['ds']}"
    import pickle
    from topoflow import direct
    from topoflow.e2 import MAX_OBS
    from topoflow.p1 import ObstacleAttnField
    from topoflow.selectors import CostRegSelector, probs_from_cost
    from topoflow.tdfm import index_code, sample_codes, selector_inputs
    from topoflow.v2 import ChartCache, ChartField, sample as v2_sample
    DATA = os.environ["DATA"]; SG = cfg["ds"] == "d3"; C = 0.01; NL, NP = 10, 32
    unseen = pickle.load(open(f"{DATA}/layouts.pkl", "rb"))[1][:300]
    B = json.load(open(f"{DATA}/relabel.json"))["beta"]
    X, S = selector_inputs(unseen, sg=SG); S = S.numpy()
    sel = CostRegSelector(512, 4, extra=2 if SG else 0); sel.load_state_dict(torch.load(f"{DATA}/models/sel_relabel_T1.pt")); sel.eval()
    with torch.no_grad():
        jh, fl = (t.numpy() for t in sel(X))
    feas = (fl > 0) & S; feas = np.where(feas.any(1, keepdims=True), feas, S)
    codes = sample_codes(probs_from_cost(jh, feas, B["T1"]["medium"]), NP, seed=900)
    lays = unseen[:NL]
    req = [(i, index_code(int(codes[i, j]), lays[i]["env"].m)) for i in range(NL) for j in range(NP)]
    x0 = torch.randn(300 * NP, 60, generator=torch.Generator().manual_seed(777))[:NL * NP]
    arm, K = cfg["arm"], cfg["K"]
    if arm == "v2":
        m = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG)
        m.load_state_dict(torch.load(f"{DATA}/models/V2_s0.pt")); m.eval()

        def run():
            cache = ChartCache(lays, C, 60, sg=SG)
            return v2_sample(m, cache, req, K, x0)
    else:
        blind = arm.startswith("b0")
        m = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=not blind, sg_input=SG)
        m.load_state_dict(torch.load(f"{DATA}/models/{'B0attn_medium_s0' if blind else 'P1_s0'}.pt")); m.eval()
        mode = {"p1": "none", "p1final": "final", "p1free": "free", "p1chart": "chart", "b0free": "free"}[arm]

        def run():
            cond = direct.build_cond(lays, req, not blind, sg=SG)
            return direct.sample(m, lays, req, cond, x0, K, mode, C, iters=cfg.get("iters", 100), sub=32, batch=1024)
    t = time.perf_counter(); run(); warm = time.perf_counter() - t
    R = 3 if warm > 60 else 5
    ts = []
    for _ in range(R):
        t = time.perf_counter(); run(); ts.append(time.perf_counter() - t)
    ts = np.array(ts) / len(req) * 1000
    opt = [k for k in range(K) if (k / K >= 0.5 - 1e-12 and k < K - 1)] if arm in ("p1chart", "p1free", "b0free") else []
    field_evals = K + len(opt)
    res = dict(cfg, batch=len(req), repeats=R, warmup_ms=warm / len(req) * 1000, ms_median=float(np.median(ts)),
               ms_q25=float(np.percentile(ts, 25)), ms_q75=float(np.percentile(ts, 75)), ms_all=ts.tolist(),
               field_evals=field_evals, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               threads=torch.get_num_threads(), host=os.uname().nodename, cpu=open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "))
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(json.loads(sys.argv[1])); sys.exit(0)
    done = set()
    if os.path.exists(OUT):
        done = {json.dumps({k: r[k] for k in ("ds", "arm", "K", "iters") if k in r}, sort_keys=True)
                for r in map(json.loads, open(OUT))}
    T0 = time.time()
    for cfg in CFGS:
        if time.time() - T0 > 95 * 60:
            print('budget reached; relaunch to continue', flush=True); break
        key = json.dumps(cfg, sort_keys=True)
        if key in done:
            continue
        p = subprocess.run([sys.executable, __file__, json.dumps(cfg)], capture_output=True, text=True,
                           env=dict(os.environ, TOPO_MAX_OBS="5", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if not line:
            print("FAILED", cfg, p.stderr[-800:], flush=True); continue
        open(OUT, "a").write(line[-1] + "\n")
        r = json.loads(line[-1])
        print(cfg, round(r["ms_median"], 3), "ms", r["repeats"], "reps", flush=True)

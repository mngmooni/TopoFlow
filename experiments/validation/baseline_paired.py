"""Experiment B: route-matched v2 versus P1 + chart-correction inference.

Safe entrypoint, without training. Imports do not load datasets or checkpoints.
Original single-run outputs are not shipped with this public repository; they
are archived separately. Generated NPZ files default to ``reproductions/baseline``.

After restoring original data and checkpoints, for example:
    python -m experiments.validation.baseline_paired --data results/d2 \
        --runs v2:0:777:20,p1chart:0:777:20

Each requested run retains the historic layout/code/noise pairing and numeric
algorithms. Existing reproduction files are skipped unless --overwrite is used.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .output_guard import ROOT

N_PER = 32
MAX_LAYOUTS = 300
VALID_ARMS = ("v2", "p1chart")


def parse_runs(specs: str):
    """Four colon-delimited values per run, as in the archived runner."""
    result = []
    for raw in specs.split(","):
        parts = raw.strip().split(":")
        if len(parts) != 4 or parts[0] not in VALID_ARMS:
            raise ValueError(f"Invalid run {raw!r}; use arm:model_seed:noise_seed:NFE and arm in {VALID_ARMS}")
        arm, ms, ns, K = parts
        try:
            ms, ns, K = int(ms), int(ns), int(K)
        except ValueError as exc:
            raise ValueError(f"Invalid integers in run {raw!r}") from exc
        if ms < 0 or ns < 0 or K < 1:
            raise ValueError(f"Nonnegative seeds and positive NFE required: {raw!r}")
        item = (arm, ms, ns, K)
        if item not in result:
            result.append(item)
    return result


def expected_inputs(data: Path, runs):
    paths = [data / "betas.json", data / "layouts.pkl", data / "relabel.json",
             data / "models/sel_relabel_T1.pt"]
    for arm, ms, _, _ in runs:
        paths.append(data / "models" / f"{'V2' if arm == 'v2' else 'P1'}_s{ms}.pt")
    return list(dict.fromkeys(paths))


def run(data: Path, runs, output_dir: Path, *, overwrite: bool = False, max_obs: int = 5):
    """Execute the original model inference/statistics; only output paths changed."""
    import os
    os.environ["TOPO_MAX_OBS"] = str(max_obs)
    import torch
    from . import indep_geom as G
    from topoflow import direct
    from topoflow.e2 import MAX_OBS
    from topoflow.p1 import ObstacleAttnField
    from topoflow.selectors import CostRegSelector, probs_from_cost
    from topoflow.taxonomy import classify_samples
    from topoflow.tdfm import index_code, sample_codes, selector_inputs
    from topoflow.v2 import ChartCache, ChartField, sample as v2_sample

    torch.set_num_threads(1)
    data = Path(data).resolve()
    output_dir = Path(output_dir).resolve()
    SG = data.name == "d3"
    with (data / "betas.json").open(encoding="utf-8") as fp:
        meta = json.load(fp)
    C = float(meta["_margin"])
    with (data / "layouts.pkl").open("rb") as fp:
        unseen = pickle.load(fp)[1][:MAX_LAYOUTS]
    with (data / "relabel.json").open(encoding="utf-8") as fp:
        B = json.load(fp)["beta"]
    L = len(unseen)
    X, S = selector_inputs(unseen, sg=SG)
    S = S.numpy()
    sel = CostRegSelector(512, 4, extra=2 if SG else 0)
    sel.load_state_dict(torch.load(data / "models/sel_relabel_T1.pt", weights_only=True))
    sel.eval()
    with torch.no_grad():
        jh, fl = (t.numpy() for t in sel(X))
    feas = (fl > 0) & S
    feas = np.where(feas.any(1, keepdims=True), feas, S)
    codes = sample_codes(probs_from_cost(jh, feas, B["T1"]["medium"]), N_PER, seed=900)
    req = [(i, index_code(int(codes[i, j]), unseen[i]["env"].m)) for i in range(L) for j in range(N_PER)]
    XG = np.linspace(-0.9, 0.9, 62)[1:-1]
    REFLEN = {}
    for i, lay in enumerate(unseen):
        for code, ref in lay["found"].items():
            REFLEN[(i, code)] = min(np.linalg.norm(np.diff(r, axis=0), axis=-1).sum() for r in ref)

    def per_sample(paths):
        n = len(paths)
        out = {k: np.zeros(n) for k in ("eu_viol", "sq_viol", "word_ok", "minclear", "length", "rough", "len_ratio")}
        cat = np.empty(n, dtype="<U1")
        for i, lay in enumerate(unseen):
            env = lay["env"]
            sl = slice(i * N_PER, (i + 1) * N_PER)
            P = np.stack(paths[sl])
            rq = [req[i * N_PER + j][1] for j in range(N_PER)]
            cats, _ = classify_samples(env, P, rq, set(lay["found"]), C)
            cat[sl] = cats
            A, Bq = P[:, :-1], P[:, 1:]
            dmin = np.full(N_PER, np.inf)
            sq = np.zeros(N_PER, bool)
            for o in env.obstacles:
                d = G.seg_obstacle_dist(A, Bq, o)
                dmin = np.minimum(dmin, d.min(1))
                sq |= (G.seg_inflated_rect_dist(A, Bq, o, C) <= 0).any(1)
            out["minclear"][sl] = dmin
            out["eu_viol"][sl] = dmin < C
            out["sq_viol"][sl] = sq
            out["word_ok"][sl] = [G.crossing_word(P[j], env.obstacles) == G.word_of_code(rq[j]) for j in range(N_PER)]
            ln = np.linalg.norm(np.diff(P, axis=1), axis=-1).sum(1)
            out["length"][sl] = ln
            out["len_ratio"][sl] = [ln[j] / REFLEN[(i, rq[j])] if (i, rq[j]) in REFLEN else np.nan for j in range(N_PER)]
            Y = np.array([np.interp(XG, p[:, 0], p[:, 1]) for p in P])
            out["rough"][sl] = np.mean(np.diff(Y, 2, axis=1) ** 2, 1)
        return out, cat

    def per_class(paths, cat):
        rows = []
        for i in range(L):
            P = np.stack(paths[i * N_PER:(i + 1) * N_PER])
            rq = [req[i * N_PER + j][1] for j in range(N_PER)]
            Y = np.array([np.interp(XG, p[:, 0], p[:, 1]) for p in P])
            for code in sorted(set(rq)):
                idx = [j for j in range(N_PER) if rq[j] == code and cat[i * N_PER + j] == "V"]
                if len(idx) >= 2:
                    d = np.sqrt(((Y[idx][:, None] - Y[idx][None]) ** 2).mean(-1))
                    rows.append((i, len(idx), d[np.triu_indices(len(idx), 1)].mean()))
        return np.array(rows)

    def load_p1(ms):
        m = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True, sg_input=SG)
        m.load_state_dict(torch.load(data / f"models/P1_s{ms}.pt", weights_only=True))
        return m.eval()

    outputs = []
    for arm, ms, ns, K in runs:
        spec = f"{arm}:{ms}:{ns}:{K}"
        f = output_dir / f"{data.name}_{arm}_m{ms}_n{ns}_K{K}.npz"
        if f.exists() and not overwrite:
            print(f"skip existing reproduction: {f}", flush=True)
            continue
        x0 = torch.randn(L * N_PER, 60, generator=torch.Generator().manual_seed(ns))
        t0 = time.perf_counter()
        info = {}
        if arm == "v2":
            m = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6, sg_input=SG)
            m.load_state_dict(torch.load(data / f"models/V2_s{ms}.pt", weights_only=True))
            m.eval()
            cache = ChartCache(unseen, C, 60, sg=SG)
            paths = list(v2_sample(m, cache, req, K, x0))
            chart_inf = float(np.mean([not cache.get(i, c)[3] for i, c in req]))
        else:
            m = load_p1(ms)
            cond = direct.build_cond(unseen, req, True, sg=SG)
            paths, info = direct.sample(m, unseen, req, cond, x0, K, "chart", C)
            chart_inf = info["chart_infeasible"]
        sec = time.perf_counter() - t0
        ps, cat = per_sample(paths)
        pc = per_class(paths, cat)
        np.savez(f, **ps, cat=cat, lid=np.repeat(np.arange(L), N_PER), cls=pc,
                 paths=np.stack(paths).astype(np.float32), sec=sec, chart_infeasible=chart_inf)
        outputs.append(f)
        print(spec, data.name, "sec", round(sec, 1), "V", float(np.mean(cat == "V")),
              "eu", int(ps["eu_viol"].sum()), "sq", int(ps["sq_viol"].sum()),
              "word_bad", int((ps["word_ok"] == 0).sum()), "div", float(pc[:, 2].mean()),
              "chart_inf", chart_inf, flush=True)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Restored original results/d2 or results/d3")
    parser.add_argument("--runs", required=True, help="Comma-separated arm:model_seed:noise_seed:NFE")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproductions/baseline")
    parser.add_argument("--max-obs", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly overwrite selected reproduction NPZ files")
    args = parser.parse_args(argv)
    try:
        runs = parse_runs(args.runs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.data.name not in ("d2", "d3"):
        parser.error("--data must be a d2 or d3 dataset directory")
    if args.max_obs < 1:
        parser.error("--max-obs must be positive")
    outdir = args.output_dir.expanduser().resolve()
    if outdir.is_relative_to((ROOT / "results").resolve()):
        parser.error("Reproductions must not write into the archived results/ directory")
    data = args.data if args.data.is_absolute() else ROOT / args.data
    missing = [str(p) for p in expected_inputs(data, runs) if not p.is_file()]
    if missing:
        parser.error("Missing original prerequisites (not bundled): " + ", ".join(missing))
    if outdir.is_file():
        parser.error(f"Output directory path is a file: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    run(data, runs, outdir, overwrite=args.overwrite, max_obs=args.max_obs)


if __name__ == "__main__":
    main()

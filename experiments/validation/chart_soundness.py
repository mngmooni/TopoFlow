"""Experiment A: independent soundness verification of decoded feasible charts.

Safe public replay: imports do not load data or perform validation. The original
frozen validation remains at ``results/final/soundness/``; new outputs default to
``reproductions/soundness/``. Original splits are not distributed in this repo.

From the repository root, after restoring the original inputs:
    python -m experiments.validation.chart_soundness --dataset d2
For a limited diagnostic (not a full scientific reproduction):
    python -m experiments.validation.chart_soundness --dataset d2 --limit 1
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np

from .output_guard import ROOT, prepare_outputs

C = 0.01
N = 60
MAXP = 400
TOL = 1e-9
DATASETS = ("d2", "d3", "d4", "H1", "H2", "H3", "H4")


def battery(rng, n: int, hard: bool):
    """Retain the original test samples and RNG consumption order."""
    if hard:
        return np.concatenate([rng.normal(0, 1, (2, n)), rng.normal(0, 3, (2, n)),
                               np.full((1, n), 50.0), np.full((1, n), -50.0),
                               rng.choice([-50.0, 0.0, 50.0], size=(2, n))])
    return np.concatenate([rng.normal(0, 1, (8, n)), rng.normal(0, 3, (8, n)),
                           np.full((1, n), 50.0), np.full((1, n), -50.0),
                           rng.choice([-50.0, 0.0, 50.0], size=(6, n))])


def refine(env, code, xs, chart):
    """Original local-bisection rule, with at most MAXP interior grid points."""
    for _ in range(200):
        lo, hi, ok, L, U = chart.bounds_xs(env, code, C, xs)
        if ok:
            return xs, lo, hi
        if len(xs) >= MAXP:
            return None
        full = np.concatenate([[env.start[0]], xs, [env.goal[0]]])
        bad = np.where(hi <= lo)[0] + 1
        if not (L[0] <= env.start[1] <= U[0] and L[-1] <= env.goal[1] <= U[-1]):
            bad = np.concatenate([bad, [1, len(full) - 2]])
        new = {(full[k - 1] + full[k]) / 2 for k in bad} | {(full[k] + full[k + 1]) / 2 for k in bad}
        xs = np.unique(np.concatenate([xs, list(new)]))
    return None


def paths_on(env, xs, lo, hi, rng, hard, chart):
    Y = chart.decode(battery(rng, len(xs), hard), lo, hi)
    P = np.empty(Y.shape + (2,))
    P[..., 0], P[..., 1] = xs, Y
    S = np.broadcast_to(np.asarray(env.start, float), (len(Y), 1, 2))
    Gl = np.broadcast_to(np.asarray(env.goal, float), (len(Y), 1, 2))
    return np.concatenate([S, P, Gl], 1)


def check(env, code, paths, G):
    A, B = paths[:, :-1], paths[:, 1:]
    m_eu, m_sq = np.inf, np.inf
    n_eu = n_sq = n_exact = n_exact_bad = 0
    for o in env.obstacles:
        d = G.seg_obstacle_dist(A, B, o) - C
        s = G.seg_inflated_rect_dist(A, B, o, C)
        m_eu, m_sq = min(m_eu, float(d.min())), min(m_sq, float(s.min()))
        for idx in zip(*np.where((d < TOL) | (s < TOL))):
            n_exact += 1
            ok, _ = G.exact_clearance_ok(A[idx], B[idx], o, C)
            meet = G.exact_open_rect_meet(A[idx], B[idx], o, C)
            n_exact_bad += int((not ok) or meet)
            n_eu += int(not ok)
            n_sq += int(meet)
    want = G.word_of_code(code)
    words_bad = sum(G.crossing_word(p, env.obstacles) != want for p in paths)
    return dict(m_eu=m_eu, m_sq=m_sq, viol_eu=n_eu, viol_sq=n_sq, exact_checked=n_exact,
                exact_bad=n_exact_bad, words_bad=int(words_bad), n_paths=len(paths),
                n_seg=int(A.shape[0] * A.shape[1]))


def load_layouts(dataset: str, data_root: Path):
    """Read the historic dataset schema; never generate or modify the source."""
    hard = dataset.startswith("H")
    if hard:
        path = data_root / "hard" / f"{dataset}_layouts.pkl"
    else:
        folder = data_root / dataset
        path = folder / "layouts.pkl"
        beta_path = folder / "betas.json"
        if not beta_path.is_file():
            raise FileNotFoundError(f"Missing original margin metadata: {beta_path}")
        with beta_path.open(encoding="utf-8") as fp:
            assert abs(json.load(fp)["_margin"] - C) < 1e-12
    if not path.is_file():
        raise FileNotFoundError(f"Missing original validation layouts: {path}")
    with path.open("rb") as fp:
        layouts = pickle.load(fp)
    return layouts if hard else layouts[1]


def evaluate(dataset: str, layouts, *, limit: int | None = None, progress: bool = True):
    """Run the unchanged geometric checks on supplied layout objects; no file writes."""
    from topoflow import chart
    from . import indep_geom as G

    hard = dataset.startswith("H")
    lays = layouts[:limit] if limit is not None else layouts
    rng = np.random.default_rng(2026)
    t0 = time.time()
    rows, setups, viol = [], [], []
    for i, lay in enumerate(lays):
        env = lay["env"]
        su = G.setup_checks(env, C)
        su["i"] = i
        setups.append(su)
        for code in itertools.product([0, 1], repeat=env.m):
            xs = chart.grid(env, N, C)
            lo, hi, ok, _, _ = chart.bounds_xs(env, code, C, xs)
            r = dict(i=i, code=code, ok60=bool(ok), found=(not hard) and code in lay["found"])
            grids = []
            if ok:
                grids.append(("n60", xs, lo, hi))
            else:
                ref = refine(env, code, xs, chart)
                r["ok_bisect"] = ref is not None
                if ref is not None:
                    r["n_bisect"] = len(ref[0])
                    grids.append(("bisect", *ref))
            for name, xg, l_, h_ in grids:
                r["band_min"] = float(np.min(h_ - l_))
                r["grid_strict"] = bool(np.all(np.diff(
                    np.concatenate([[env.start[0]], xg, [env.goal[0]]])) > 0))
                res = check(env, code, paths_on(env, xg, l_, h_, rng, hard, chart), G)
                r.update({f"{k}": v for k, v in res.items()}, grid=name)
                if res["viol_eu"] or res["viol_sq"] or res["words_bad"]:
                    viol.append(dict(i=i, code=code, grid=name, setup=su, **res))
            rows.append(r)
        if progress and i % 50 == 0:
            print(dataset, i, round(time.time() - t0), flush=True)

    chk = [r for r in rows if "m_eu" in r]
    if not chk:
        raise ValueError("No charts checked; the selected layouts do not support a validation summary")
    summ = dict(
        dataset=dataset, c=C, layouts=len(lays), codes=len(rows), ok60=sum(r["ok60"] for r in rows),
        bisect_recovered=sum(r.get("ok_bisect", False) for r in rows),
        checked_charts=len(chk), checked_charts_bisect=sum(r["grid"] == "bisect" for r in chk),
        decoded_paths=sum(r["n_paths"] for r in chk), segments=sum(r["n_seg"] for r in chk),
        violations_euclid=sum(r["viol_eu"] for r in chk), violations_square=sum(r["viol_sq"] for r in chk),
        word_mismatch_paths=sum(r["words_bad"] for r in chk),
        min_margin_euclid=min(r["m_eu"] for r in chk), min_margin_square=min(r["m_sq"] for r in chk),
        exact_rechecked_segments=sum(r["exact_checked"] for r in chk), exact_failed=sum(r["exact_bad"] for r in chk),
        min_band_width=min(r["band_min"] for r in chk),
        min_band_width_n60=min((r["band_min"] for r in chk if r["grid"] == "n60"), default=None),
        grid_strict_all=all(r["grid_strict"] for r in chk),
        setup=dict(a3_fail=sum(not s["a3"] for s in setups), a3_min_margin=min(s["a3_margin"] for s in setups),
                   not_disjoint=sum(not s["disjoint"] for s in setups),
                   cuts_invalid=sum(not s["cuts_ok"] for s in setups),
                   cx_not_distinct=sum(not s["distinct_x"] for s in setups)),
        word_mismatch_on_invalid_cut_layouts=sum(r["words_bad"] for r in chk if not setups[r["i"]]["cuts_ok"]),
        violation_records=viol[:50], seconds=round(time.time() - t0, 1),
        latents_per_chart=len(battery(rng, N, hard)),
    )
    return summ, dict(rows=rows, setups=setups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="d2")
    parser.add_argument("--data-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproductions/soundness")
    parser.add_argument("--limit", type=int, help="Diagnostic only: evaluate the first N layouts")
    parser.add_argument("--overwrite", action="store_true", help="Replace only selected reproduction outputs")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    outdir = args.output_dir.expanduser().resolve()
    if outdir.is_relative_to((ROOT / "results").resolve()):
        parser.error("A reproduction must not modify the archived results/ tree")
    json_out = outdir / f"{args.dataset}.json"
    codes_out = outdir / f"{args.dataset}_codes.pkl"
    try:
        lays = load_layouts(args.dataset, args.data_root)
        prepare_outputs(json_out, codes_out, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, IsADirectoryError) as exc:
        parser.error(str(exc))
    summ, code_rows = evaluate(args.dataset, lays, limit=args.limit)
    print(json.dumps({k: v for k, v in summ.items() if k != "violation_records"}), flush=True)
    with json_out.open("w", encoding="utf-8") as fp:
        json.dump(summ, fp, indent=1, default=str)
    with codes_out.open("wb") as fp:
        pickle.dump(code_rows, fp)
    print(f"saved {json_out} and {codes_out}")
    if args.limit is not None:
        print("PARTIAL DIAGNOSTIC: --limit results are not the archived full-validation experiment")


if __name__ == "__main__":
    main()

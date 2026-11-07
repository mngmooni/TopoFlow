"""Experiment B paired bootstrap analysis of v2 and chart-corrected P1.

Reads paired per-run ``.npz`` records; reproduces the historical statistics and
writes a new report under ``reproductions/`` by default. Importing this module
does not load data, run the bootstrap, or write scientific results.

Original per-run NPZ records are *not* included in the public candidate. The
curated ``results/final/baseline_paired.json`` is an archived result, not a
substitute for the missing inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .output_guard import ROOT, prepare_outputs
except ImportError:
    from output_guard import ROOT, prepare_outputs

DEFAULT_INPUT = ROOT / "results/final/baseline"
DEFAULT_OUTPUT = ROOT / "reproductions/baseline_paired.json"
METRICS = ["diversity", "length_ratio", "roughness", "min_clearance"]


def load(input_dir: Path, ds: str, arm: str, ms: int, ns: int, K: int):
    """Load one archived run if present, retaining the historic file schema."""
    path = Path(input_dir) / f"{ds}_{arm}_m{ms}_n{ns}_K{K}.npz"
    return dict(np.load(path)) if path.exists() else None


def layout_sums(r, layout_count: int = 300):
    """Per-layout numerator and denominator for each original aggregate."""
    lid = r["lid"]
    valid = r["cat"] == "V"
    cls = r["cls"]
    out = {}
    out["diversity"] = (
        np.bincount(cls[:, 0].astype(int), cls[:, 2], layout_count),
        np.bincount(cls[:, 0].astype(int), None, layout_count),
    )
    ok = valid & ~np.isnan(r["len_ratio"])
    out["length_ratio"] = (
        np.bincount(lid[ok], r["len_ratio"][ok], layout_count),
        np.bincount(lid[ok], None, layout_count),
    )
    out["roughness"] = (
        np.bincount(lid, r["rough"], layout_count),
        np.bincount(lid, None, layout_count),
    )
    out["min_clearance"] = (
        np.bincount(lid, r["minclear"], layout_count),
        np.bincount(lid, None, layout_count),
    )
    return out


def estimate(s, weights):
    return (s[0] * weights).sum() / (s[1] * weights).sum()


def summary(r):
    return dict(
        n=int(len(r["cat"])),
        V=float(np.mean(r["cat"] == "V")),
        cats={c: int((r["cat"] == c).sum()) for c in "UMTNV"},
        indep_euclid_viol=int(r["eu_viol"].sum()),
        indep_square_viol=int(r["sq_viol"].sum()),
        indep_word_bad=int((r["word_ok"] == 0).sum()),
        min_clearance_min=float(r["minclear"].min()),
        chart_infeasible=float(r["chart_infeasible"]),
        sec=float(r["sec"]),
    )


def analyze(input_dir: Path, bootstrap_count: int = 4000, layout_count: int = 300):
    """Compute historic layout-clustered statistics from supplied per-run inputs.

    The iteration and RNG order match the archived computation. Returns a
    dictionary; does *not* write a file. A complete archive is required to
    reproduce all published entries; incomplete archives produce only entries
    for present, matched pairs.
    """
    if bootstrap_count < 1 or layout_count < 1:
        raise ValueError("bootstrap_count and layout_count must be positive")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing per-run Experiment B inputs: {input_dir}")
    rng = np.random.default_rng(12)
    results = {}
    for ds, mss in (("d2", (0, 1, 2)), ("d3", (0,))):
        for K in (5, 20):
            for ms in mss:
                pairs = [
                    (ns, load(input_dir, ds, "v2", ms, ns, K),
                     load(input_dir, ds, "p1chart", ms, ns, K))
                    for ns in (777, 778, 779)
                ]
                pairs = [p for p in pairs if p[1] is not None and p[2] is not None]
                if not pairs:
                    continue
                key = f"{ds}_K{K}_m{ms}"
                sums = [(ns, layout_sums(a, layout_count), layout_sums(b, layout_count))
                        for ns, a, b in pairs]
                entry = dict(
                    noise_seeds=[p[0] for p in pairs],
                    runs={str(ns): dict(v2=summary(a), p1chart=summary(b)) for ns, a, b in pairs},
                    metrics={},
                )
                for met in METRICS:
                    per_seed = {}
                    for ns, sa, sb in sums:
                        one = np.ones(layout_count)
                        pv, pp = estimate(sa[met], one), estimate(sb[met], one)
                        W = rng.multinomial(layout_count, np.full(layout_count, 1 / layout_count), size=bootstrap_count)
                        diff = np.array([estimate(sa[met], w) - estimate(sb[met], w) for w in W])
                        la = sa[met][0] / np.maximum(sa[met][1], 1e-12)
                        lb = sb[met][0] / np.maximum(sb[met][1], 1e-12)
                        both = (sa[met][1] > 0) & (sb[met][1] > 0)
                        per_seed[str(ns)] = dict(
                            v2=pv, p1chart=pp, diff=pv - pp,
                            ci=list(np.percentile(diff, [2.5, 97.5])),
                            layouts_v2_greater=float(np.mean(la[both] > lb[both])),
                            layout_mean_diff=float(np.mean(la[both] - lb[both])),
                            layouts_both=int(both.sum()),
                        )
                    W = rng.multinomial(layout_count, np.full(layout_count, 1 / layout_count), size=bootstrap_count)
                    pick = rng.integers(0, len(sums), bootstrap_count)
                    diff = np.array([
                        estimate(sums[k][1][met], w) - estimate(sums[k][2][met], w)
                        for w, k in zip(W, pick)
                    ])
                    diffs = [v["diff"] for v in per_seed.values()]
                    entry["metrics"][met] = dict(
                        per_seed=per_seed,
                        pooled_diff=float(np.mean(diffs)),
                        pooled_ci=list(np.percentile(diff, [2.5, 97.5])),
                        seed_range=float(np.ptp(diffs)),
                        rel_seed_range=float(np.ptp(diffs) / abs(np.mean(diffs))) if np.mean(diffs) else None,
                    )
                results[key] = entry
                metrics = entry["metrics"]
                print(key, {k: (round(metrics[k]["pooled_diff"], 5),
                                 [round(x, 5) for x in metrics[k]["pooled_ci"]]) for k in METRICS}, flush=True)
    if not results:
        raise FileNotFoundError(
            f"No complete v2/p1chart NPZ pairs found under {input_dir}; "
            "this public candidate does not bundle the original per-run archives."
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Original paired per-run Experiment B .npz directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New JSON output (default: reproductions/baseline_paired.json)")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the chosen output only")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Missing input directory: {args.input_dir}")
    # Validate destination before expensive work. Guard against replacing the
    # published canonical result even when --overwrite was supplied.
    if args.output.expanduser().resolve() == (ROOT / "results/final/baseline_paired.json").resolve():
        parser.error("The curated baseline_paired.json is immutable; choose a reproduction output path")
    prepare_outputs(args.output, overwrite=args.overwrite)
    result = analyze(args.input_dir, bootstrap_count=args.bootstrap)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=1)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

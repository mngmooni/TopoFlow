"""Experiment C class-matched bootstrap and Wasserstein diagnostics.

Read original per-class ``.npz`` archives to reproduce the study's empirical
summaries. Importing this module is side-effect-free. The archived per-class
inputs are *not* bundled in the public repository candidate; therefore the
frozen ``results/final/distribution_fidelity.json`` cannot be regenerated from
this repository alone.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

try:
    from .fidelity_metrics import STATS, class_table, per_class_samples, agg
    from .output_guard import ROOT, prepare_outputs
except ImportError:
    from fidelity_metrics import STATS, class_table, per_class_samples, agg
    from output_guard import ROOT, prepare_outputs

DEFAULT_INPUT = ROOT / "results/final/fidelity"
DEFAULT_OUTPUT = ROOT / "reproductions/distribution_fidelity.json"


def analyze(input_dir: Path, bootstrap_count: int = 4000):
    """Recompute historic class-matched summaries; returns data, writes nothing.

    Maintains the original RNG and dictionary traversal order, including its
    run-resampling and class-level Wasserstein calculations.
    """
    if bootstrap_count < 1:
        raise ValueError("bootstrap_count must be positive")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing Experiment C per-class NPZ inputs: {input_dir}")
    rng = np.random.default_rng(21)
    results = {}
    for ds in ("d2", "d3"):
        ref_file = input_dir / f"{ds}_refs.npz"
        if not ref_file.is_file():
            raise FileNotFoundError(f"Missing per-class reference archive: {ref_file}")
        ref = dict(np.load(ref_file))
        lid = ref["lid"]
        ncls = len(lid)
        layout_count = lid.max() + 1
        nref = ref["nref"]
        matched = nref > 0
        R = class_table(ref, ncls)
        runs = {}
        for file in sorted(glob.glob(str(input_dir / f"{ds}_*_m*_n*.npz"))):
            arm, m, n = os.path.basename(file)[:-4].split("_")[1:]
            runs.setdefault(arm, {})[(int(m[1:]), int(n[1:]))] = dict(np.load(file))
        if not runs:
            raise FileNotFoundError(f"No per-class generated samples found for {ds} under {input_dir}")
        entry = dict(
            classes_chart_feasible=int(ncls),
            matched=int(matched.sum()),
            unmatched_no_reference=int((~matched).sum()),
            matched_frac=float(matched.mean()),
            refs_per_class_median=float(np.median(nref[matched])),
            references_without_feasible_chart=None,
            arms={},
        )
        for arm, rr in runs.items():
            tabs = {k: class_table(v, ncls) for k, v in rr.items()}
            e = {}
            for wname, wcls in (("equal_class", matched.astype(float)), ("ref_weighted", nref.astype(float))):
                pts = {k: agg(T, R, lid, wcls, np.ones(layout_count)) for k, T in tabs.items()}
                keys = list(tabs)
                bs = {s: [] for s in STATS}
                for _ in range(bootstrap_count):
                    wl = rng.multinomial(layout_count, np.full(layout_count, 1 / layout_count)).astype(float)
                    a = agg(tabs[keys[rng.integers(len(keys))]], R, lid, wcls, wl)
                    for s in STATS:
                        bs[s].append(a[s])
                e[wname] = {
                    s: dict(
                        point=float(np.mean([p[s] for p in pts.values()])),
                        ci=[float(x) for x in np.percentile(bs[s], [2.5, 97.5])],
                        per_run={f"m{k[0]}_n{k[1]}": float(p[s]) for k, p in pts.items()},
                    ) for s in STATS
                }
            if arm == "v2" and len({k[0] for k in tabs}) > 1:
                by_m = {}
                for (model_seed, noise_seed), p in {
                    k: agg(T, R, lid, matched.astype(float), np.ones(layout_count)) for k, T in tabs.items()
                }.items():
                    by_m.setdefault(model_seed, []).append(p["rough"])
                e["model_seed_rough_ratio"] = {str(model_seed): float(np.mean(v)) for model_seed, v in by_m.items()}
            k0 = sorted(rr)[0]
            w1 = {}
            for key in ("len_ratio", "rough", "minclear"):
                gs, rs = per_class_samples(rr[k0], ncls, key), per_class_samples(ref, ncls, key)
                v = [wasserstein_distance(gs[j], rs[j]) for j in range(ncls)
                     if matched[j] and len(gs[j]) and len(rs[j])]
                w1[key] = dict(median_gen_ref=float(np.median(v)))
                same_m = [k for k in sorted(rr) if k[0] == k0[0] and k != k0]
                if same_m:
                    g2 = per_class_samples(rr[same_m[0]], ncls, key)
                    v2 = [wasserstein_distance(gs[j], g2[j]) for j in range(ncls)
                          if matched[j] and len(gs[j]) and len(g2[j])]
                    w1[key]["median_gen_gen_noise_floor"] = float(np.median(v2))
            e["w1_secondary"] = w1
            e["min_clearance_min"] = float(min(v["minclear"].min() for v in rr.values()))
            e["runs"] = [f"m{k[0]}_n{k[1]}" for k in sorted(rr)]
            entry["arms"][arm] = e
            print(ds, arm, {s: (round(e["equal_class"][s]["point"], 4),
                                   [round(x, 4) for x in e["equal_class"][s]["ci"]]) for s in STATS}, flush=True)
        results[ds] = entry
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT,
                        help="Directory containing original D2/D3 reference and generated NPZ files")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New JSON output (default: reproductions/distribution_fidelity.json)")
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the chosen output only")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"Missing input directory: {args.input_dir}")
    if args.output.expanduser().resolve() == (ROOT / "results/final/distribution_fidelity.json").resolve():
        parser.error("The curated distribution_fidelity.json is immutable; choose a reproduction output path")
    prepare_outputs(args.output, overwrite=args.overwrite)
    result = analyze(args.input_dir, bootstrap_count=args.bootstrap)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=1)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

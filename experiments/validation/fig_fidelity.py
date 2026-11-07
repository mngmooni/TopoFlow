"""Experiment C: per-class fidelity visualization; explicit paths and protected reproduction output."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__:
    from .fidelity_metrics import class_table
    from .output_guard import ROOT, prepare_outputs
else:
    from fidelity_metrics import class_table
    from output_guard import ROOT, prepare_outputs


def render(input_dir, output_png):
    D = input_dir
    cols = {"v2": "#2a78d6", "p1chart": "#eb6834"}
    names = {"v2": "v2", "p1chart": "P1 + DiRecT-chart"}
    panels = [("len_ratio gap (gen − ref)", 0, "diff"), ("roughness, log(gen / ref)", 1, "log"),
              ("mean min clearance gap (gen − ref)", 2, "diff"), ("diversity, log(gen / ref)", 4, "log")]
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), dpi=200)
    for r, ds in enumerate(("d2", "d3")):
        ref = dict(np.load(f"{D}/{ds}_refs.npz")); ncls = len(ref["lid"]); matched = ref["nref"] > 0
        R = class_table(ref, ncls)
        vals = {}
        for arm in ("v2", "p1chart"):
            G = class_table(dict(np.load(f"{D}/{ds}_{arm}_m0_n11.npz")), ncls)
            vals[arm] = G
        for c, (title, k, kind) in enumerate(panels):
            ax = axes[r, c]
            data = {}
            for arm, G in vals.items():
                ok = matched & ~np.isnan(G[:, k]) & ~np.isnan(R[:, k])
                if kind == "log":
                    ok &= (G[:, k] > 0) & (R[:, k] > 0)
                    d = np.log(G[ok, k] / R[ok, k])
                else:
                    d = G[ok, k] - R[ok, k]
                data[arm] = d
            allv = np.concatenate(list(data.values()))
            lo, hi = np.percentile(allv, [1, 99]); bins = np.linspace(lo, hi, 50)
            for arm, d in data.items():
                ax.hist(np.clip(d, lo, hi), bins=bins, histtype="step", lw=1.5, color=cols[arm],
                        label=f"{names[arm]} (median {np.median(d):+.3g})")
            ax.axvline(0, color="#52514e", lw=0.8, ls="--")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.4)
            ax.tick_params(labelsize=6.5, colors="#52514e")
            for s in ax.spines.values():
                s.set_color("#c9c8c2")
            if r == 0:
                ax.set_title(title, fontsize=8, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"{ds.upper()}: classes", fontsize=8, color="#52514e")
            ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle("Per-class fidelity to the planner reference (matched classes, model seed 0, noise seed 11; dashed = equal to reference; 1–99% range shown, end bins hold clipped tails)",
                 fontsize=9, x=0.01, ha="left", color="#0b0b0b")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_png}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/final/fidelity")
    parser.add_argument("--output", type=Path, default=ROOT / "reproductions/figures/fidelity_distributions.png")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing the chosen output file")
    args = parser.parse_args(argv)
    needed = [args.input_dir / f"{ds}_{suffix}.npz" for ds in ("d2", "d3")
              for suffix in ("refs", "v2_m0_n11", "p1chart_m0_n11")]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        parser.error("Missing fidelity inputs: " + ", ".join(missing))
    prepare_outputs(args.output, overwrite=args.overwrite)
    render(args.input_dir, args.output)


if __name__ == "__main__":
    main()

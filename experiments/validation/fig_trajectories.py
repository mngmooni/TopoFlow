"""Experiment C matched trajectory panel; explicit checkpoints and guarded output path."""
from __future__ import annotations
import argparse
from pathlib import Path
import itertools, os, pickle, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as Pc, Rectangle as Pr

try:
    from .output_guard import ROOT, prepare_outputs
except ImportError:
    from output_guard import ROOT, prepare_outputs


def render(layouts_path, v2_checkpoint, p1_checkpoint, output_png):
    # Historical script is a direct entrypoint: delay all model/data work until called.
    sys.path.insert(0, str(ROOT))
    from topoflow import direct
    from topoflow.e2 import MAX_OBS
    from topoflow.p1 import ObstacleAttnField
    from topoflow.v2 import ChartCache, ChartField, sample
    torch.set_num_threads(1)
    C, PER = 0.01, 8
    with open(layouts_path, "rb") as fp:
        lays = pickle.load(fp)[1][:3]
    cache = ChartCache(lays, C, 60)
    pick = []
    for i, lay in enumerate(lays):
        best = max(sorted(lay["found"]), key=lambda c: len(lay["found"][c]))
        pick.append((i, best))
    req = [k for k in pick for _ in range(PER)]
    x0 = torch.randn(len(req), 60, generator=torch.Generator().manual_seed(11))
    v2 = ChartField(n=60, max_obs=MAX_OBS, d=128, depth=4, chunk=6); v2.load_state_dict(torch.load(v2_checkpoint, map_location="cpu")); v2.eval()
    G = np.asarray(sample(v2, cache, req, 20, x0))
    p1 = ObstacleAttnField(30, MAX_OBS, d=128, depth=4, chunk=3, use_code=True); p1.load_state_dict(torch.load(p1_checkpoint, map_location="cpu")); p1.eval()
    H, _ = direct.sample(p1, lays, req, direct.build_cond(lays, req, True), x0, 20, "chart", C); H = np.asarray(H)

    cols = {"ref": "#52514e", "v2": "#2a78d6", "p1chart": "#eb6834"}
    fig, axes = plt.subplots(3, 3, figsize=(9, 7.2), dpi=200, sharex=True, sharey=True)
    for r, (i, code) in enumerate(pick):
        env = lays[i]["env"]
        sets = [("planner references", lays[i]["found"][code], "ref"), ("v2 (chart-space flow)", G[r * PER:(r + 1) * PER], "v2"),
                ("P1 + DiRecT-chart", H[r * PER:(r + 1) * PER], "p1chart")]
        for c_, (name, P, key) in enumerate(sets):
            ax = axes[r, c_]
            for o in env.obstacles:
                if o.kind == "circle":
                    ax.add_patch(Pc((o.cx, o.cy), o.r, color="#d9d8d3", lw=0))
                else:
                    ax.add_patch(Pr((o.cx - o.hw, o.cy - o.hh), 2 * o.hw, 2 * o.hh, color="#d9d8d3", lw=0))
            for p in P:
                ax.plot(p[:, 0], p[:, 1], color=cols[key], lw=1.0, alpha=0.8)
            ax.plot(*env.start, "o", color="#0b0b0b", ms=3); ax.plot(*env.goal, "o", color="#0b0b0b", ms=3)
            ax.set_aspect("equal"); ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.8, 0.8)
            ax.tick_params(labelsize=7, colors="#52514e")
            for s in ax.spines.values():
                s.set_color("#c9c8c2")
            if r == 0:
                ax.set_title(name, fontsize=9, color="#0b0b0b")
            if c_ == 0:
                ax.set_ylabel(f"layout {i}, code {''.join(map(str, code))}\n({len(lays[i]['found'][code])} refs)", fontsize=8, color="#52514e")
    fig.suptitle("Same environment, same route class: references vs generated (8 samples, NFE 20)", fontsize=10, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output_png}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", type=Path, default=ROOT / "results/d2/layouts.pkl")
    parser.add_argument("--v2-checkpoint", type=Path, default=ROOT / "results/d2/models/V2_s0.pt")
    parser.add_argument("--p1-checkpoint", type=Path, default=ROOT / "results/d2/models/P1_s0.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "reproductions/figures/matched_trajectories.png")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.layouts, args.v2_checkpoint, args.p1_checkpoint):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    prepare_outputs(args.output, overwrite=args.overwrite)
    render(args.layouts, args.v2_checkpoint, args.p1_checkpoint, args.output)


if __name__ == "__main__":
    main()

"""Experiment A coverage figure. Safe CLI: defaults write to reproductions/, not frozen figures."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .output_guard import ROOT, prepare_outputs
except ImportError:
    from output_guard import ROOT, prepare_outputs


def render(input_json, output_png):
    with open(input_json, encoding="utf-8") as fp:
        A = json.load(fp)
    names = {"d2": "D2", "d3": "D3", "d4": "D4 (OOD)", "H1": "H1 stacked", "H2": "H2 tight gaps", "H3": "H3 dense", "H4": "H4 stacked boxes"}
    cats = [("covered_n60", "covered by the n = 60 chart", "#2a78d6"),
            ("resolution_recovered", "covered after local bisection", "#1baf7a"),
            ("certified_infeasible", "certified infeasible (witness)", "#c9c8c2")]
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    keys = list(names)[::-1]
    for y, k in enumerate(keys):
        t = A[k]["taxonomy_all"]; n = A[k]["codes"]; left = 0
        for c, lab, col in cats:
            w = t[c] / n
            ax.barh(y, w, left=left, color=col, height=0.62, edgecolor="white", linewidth=1.5, label=lab if y == 0 else None)
            if w > 0.06:
                ax.text(left + w / 2, y, f"{100 * w:.0f}%", ha="center", va="center", fontsize=7.5,
                        color="white" if col != "#c9c8c2" else "#0b0b0b")
            left += w
        other = n - sum(t[c] for c, _, _ in cats)
        ax.text(1.01, y, f"{n:,} codes" + (f"  (+{other} other)" if other else ""), va="center", fontsize=7.5, color="#52514e")
    ax.set_yticks(range(len(keys)), [names[k] for k in keys])
    ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of all 2^m route codes per layout (300 layouts each)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, frameon=False, fontsize=7.5)
    ax.set_title("Chart coverage (x-monotone family, c = 0.01): every code is covered or certified infeasible", fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/final/chart_coverage_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reproductions/figures/chart_coverage.png")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing the chosen output file")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"Input JSON not found: {args.input}")
    prepare_outputs(args.output, overwrite=args.overwrite)
    render(args.input, args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

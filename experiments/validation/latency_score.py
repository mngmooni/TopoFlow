"""Experiment B: standardized latency scoring and Pareto figure; protected reproduction outputs.

Requires source JSONL, paired analysis JSON and archived direct-score text; the
last text file must be regenerated from the historic direct-score JSONL if absent.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .output_guard import ROOT, prepare_outputs
except ImportError:
    from output_guard import ROOT, prepare_outputs


def analyze(latency_path, paired_path, direct_score_path, json_output, figure_output):
    with open(latency_path, encoding="utf-8") as fp:
        rows = [json.loads(l) for l in fp]
    with open(paired_path, encoding="utf-8") as fp:
        P = json.load(fp)


    def T(ds, arm, K, iters=None):
        for r in rows:
            if r["ds"] == ds and r["arm"] == arm and r["K"] == K and r.get("iters") == iters:
                return r
        return None


    # research phase quality (single seed): arm,K -> (U+M %, diversity); v2 NFE -> diversity
    d12 = {}
    for l in open(direct_score_path, encoding="utf-8"):
        m = re.match(r"(d\d) (\w+)\s+K\s*(\d+)\s+[\d.]+ ms U\+M\s+([\d.]+)% T [\d.]+% div ([\d.]+) TV [\d.]+ \| v2 NFE\s+(\d+) \([\d.]+ ms\) div ([\d.]+)", l)
        if m:
            ds, arm, K, um, dv, vk, vd = m.groups()
            d12[(ds, arm, int(K))] = (float(um), float(dv))
            d12[(ds, "v2", int(vk))] = (0.0, float(vd))
    # v2 NFE 1/10/50 (DIRECT_RESULTS.md line 48) and 2/5/20 are all in direct_score's right column above.


    def expB(ds, arm, K):
        """pooled (3 noise seeds) diversity mean for model seed 0 at NFE 5/20 from Experiment B."""
        key = f"{ds}_K{K}_m0"
        if key not in P:
            return None
        ps = P[key]["metrics"]["diversity"]["per_seed"]
        vals = [ps[s][arm] for s in ps]
        return sum(vals) / len(vals) if vals else None


    out = dict(source="results/final/baseline_latency.jsonl", host=rows[0]["host"], cpu=rows[0]["cpu"],
               threads=rows[0]["threads"], batch=rows[0]["batch"], configs=rows, predictions={}, equal_wall_clock={})
    for ds in ("d2", "d3"):
        v20 = T(ds, "v2", 20)["ms_median"]
        c20 = T(ds, "p1chart", 20)["ms_median"]
        f20 = T(ds, "p1free", 20, 100)["ms_median"]
        out["predictions"].setdefault("B8", {})[ds] = dict(ratio=c20 / v20, hit=1.0 <= c20 / v20 <= 2.0)
        out["predictions"].setdefault("B10", {})[ds] = dict(ratio=f20 / v20, hit=f20 / v20 >= 30)
        v2s = sorted([r for r in rows if r["ds"] == ds and r["arm"] == "v2"], key=lambda r: r["K"])
        tab = []
        for r in rows:
            if r["ds"] != ds or r["arm"] == "v2":
                continue
            ok = [v for v in v2s if v["ms_median"] <= r["ms_median"]]
            match = ok[-1] if ok else None
            q = d12.get((ds, r["arm"], r["K"]))
            tab.append(dict(arm=r["arm"], K=r["K"], iters=r.get("iters"), ms=r["ms_median"],
                            um_pct=q[0] if q else None, div=q[1] if q else None,
                            v2_K=match["K"] if match else None, v2_ms=match["ms_median"] if match else None,
                            v2_div=d12.get((ds, "v2", match["K"]), (None, None))[1] if match else None))
        out["equal_wall_clock"][ds] = tab
        m = [t for t in tab if t["arm"] == "p1chart" and t["K"] == 20][0]
        vB = expB(ds, "v2", m["v2_K"]) if m["v2_K"] in (5, 20) else None
        cB = expB(ds, "p1chart", 20)
        out["predictions"].setdefault("B9", {})[ds] = dict(v2_K=m["v2_K"], v2_div=vB if vB is not None else m["v2_div"],
                                                           p1chart_div=cB if cB is not None else m["div"])
        b9 = out["predictions"]["B9"][ds]; b9["hit"] = b9["v2_div"] >= b9["p1chart_div"]
    for k in ("B8", "B9", "B10"):
        out["predictions"][k]["hit"] = all(v["hit"] for v in out["predictions"][k].values() if isinstance(v, dict))
    with open(json_output, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=1)

    for k in ("B8", "B9", "B10"):
        print(k, json.dumps(out["predictions"][k]))
    for ds in ("d2", "d3"):
        print(f"\nequal wall-clock {ds} (standardized ms/sample; quality: research phase single seed)")
        for t in out["equal_wall_clock"][ds]:
            it = f"it{t['iters']}" if t["iters"] else ""
            print(f"{t['arm']:8s} K{t['K']:>3} {it:6s} {t['ms']:8.2f} ms  U+M {t['um_pct']}  div {t['div']} | v2 NFE {t['v2_K']} ({t['v2_ms']:.2f} ms) div {t['v2_div']}" if t["v2_K"] else
                  f"{t['arm']:8s} K{t['K']:>3} {it:6s} {t['ms']:8.2f} ms  (no v2 config this fast)")

    # Pareto: time vs diversity; filled = 0 independent-check violations, hollow = violations (U+M > 0)
    cols = {"v2": "#2a78d6", "p1chart": "#eb6834", "p1": "#8a8984", "p1final": "#1baf7a", "p1free": "#7f5bd0", "b0free": "#c2185b"}
    names = {"v2": "v2 (chart-space flow)", "p1chart": "P1 + DiRecT-chart", "p1": "P1 raw", "p1final": "P1 + final projection (20 it)",
             "p1free": "P1 + DiRecT free-space (100 it)", "b0free": "B0 (code-blind) + free-space (100 it)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200, sharey=True)
    for ax, ds in zip(axes, ("d2", "d3")):
        for arm, it in (("v2", None), ("p1chart", None), ("p1", None), ("p1final", 20), ("p1free", 100), ("b0free", 100)):
            pts = []
            for r in sorted([r for r in rows if r["ds"] == ds and r["arm"] == arm and r.get("iters") == it], key=lambda r: r["K"]):
                q = d12.get((ds, arm, r["K"]))
                if q is None:
                    continue
                pts.append((r["ms_median"], q[1], q[0], r["K"]))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=cols[arm], lw=1.5, label=names[arm], zorder=2)
            for x, y, um, K in pts:
                ax.scatter([x], [y], s=40, color=cols[arm] if um == 0 else "white", edgecolor=cols[arm], lw=1.5, zorder=3)
                if arm in ("v2", "p1chart"):
                    ax.annotate(f"{K}", (x, y), textcoords="offset points", xytext=(4, -9 if arm == "p1chart" else 4), fontsize=6.5, color="#52514e")
        ax.set_xscale("log"); ax.set_title(ds.upper(), fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("standardized time per sample (ms, 1 CPU thread, log scale)", fontsize=8, color="#52514e")
        ax.grid(True, which="major", color="#e6e5e0", lw=0.6); ax.tick_params(labelsize=7, colors="#52514e")
        for s in ax.spines.values():
            s.set_color("#c9c8c2")
    axes[0].set_ylabel("within-class diversity (mean pairwise RMS)", fontsize=8, color="#52514e")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Safety-quality-time: filled = 0 violations (independent checks), hollow = clearance/class violations > 0; numbers = NFE",
                 fontsize=8.5, x=0.02, ha="left", color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(figure_output, bbox_inches="tight")
    plt.close(fig)
    print("saved")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=Path, default=ROOT / "results/final/baseline_latency.jsonl")
    parser.add_argument("--paired", type=Path, default=ROOT / "results/final/baseline_paired.json")
    parser.add_argument("--direct-score", type=Path, default=ROOT / "reproductions/direct_score.txt")
    parser.add_argument("--json-output", type=Path, default=ROOT / "reproductions/baseline_latency.json")
    parser.add_argument("--figure-output", type=Path, default=ROOT / "reproductions/figures/pareto_time_diversity.png")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing selected output files")
    args = parser.parse_args(argv)
    for path in (args.latency, args.paired, args.direct_score):
        if not path.is_file():
            parser.error(f"Missing input: {path}. Generate direct-score text with python -m experiments.validation.direct_score --output reproductions/direct_score.txt")
    prepare_outputs(args.json_output, args.figure_output, overwrite=args.overwrite)
    analyze(args.latency, args.paired, args.direct_score, args.json_output, args.figure_output)


if __name__ == "__main__":
    main()

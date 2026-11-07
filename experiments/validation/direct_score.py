"""Score the archived direct comparison. Reads JSONL; prints results; never edits frozen data."""
from __future__ import annotations
import argparse
from pathlib import Path
from contextlib import redirect_stdout
import json
import sys

try:
    from .output_guard import ROOT, prepare_outputs
except ImportError:
    from output_guard import ROOT, prepare_outputs


def score(d2_path, d3_path, sensitivity_path):
    rows = {}
    for f, sens in ((d2_path, False), (d3_path, False), (sensitivity_path, True)):
        for l in open(f):
            r = json.loads(l)
            rows[(r["data"][-2:], r["arm"], r["K"], r.get("iters"), sens)] = r


    def g(d, arm, K, it=100, sens=False):
        it = None if arm in ("v2", "p1", "p1chart") else it
        return rows[(d, arm, K, it, sens)]


    res = {}
    um = lambda r: r["U"] + r["M"]  # noqa: E731
    res["X1"] = all(um(g(d, "p1free", K)) <= 0.005 for d in ("d2", "d3") for K in (5, 20))
    res["X2"] = all(g(d, "p1free", K)["T"] <= 0.005 for d in ("d2", "d3") for K in (5, 20))
    res["X3"] = all(g(d, "p1free", 20)["ms_per_sample"] >= 5 * g(d, "v2", 20)["ms_per_sample"] for d in ("d2", "d3"))
    res["X4"] = all(g(d, "p1free", 20)["diversity"] >= 0.9 * max(g(d, "p1", 20)["diversity"], g(d, "v2", 20)["diversity"])
                    for d in ("d2", "d3"))
    res["X5"] = all(g(d, "p1chart", K)[k] == 0 for d in ("d2", "d3") for K in (2, 5, 20) for k in ("U", "M", "T"))
    res["X6"] = all(g(d, "p1chart", 20)["diversity"] >= 0.9 * g(d, "v2", 20)["diversity"] for d in ("d2", "d3"))
    res["X7"] = all(abs(g(d, "p1free", 20)["tv_T1"] - g(d, "v2", 20)["tv_T1"]) <= 0.01 for d in ("d2", "d3"))
    res["X8"] = all(um(g(d, "b0free", 20)) <= 0.005 and g(d, "b0free", 20)["tv_T1"] >= g(d, "v2", 20)["tv_T1"] + 0.05
                    for d in ("d2", "d3"))
    s20, s300 = g("d2", "p1free", 20, 20, True), g("d2", "p1free", 20, 300, True)
    res["X9"] = um(s20) >= um(s300) and um(s300) <= 0.001


    def v2_at(d, ms):
        """largest v2 NFE whose ms/sample <= ms (same dataset)."""
        c = [(K, g(d, "v2", K)) for K in (1, 2, 5, 10, 20, 50) if g(d, "v2", K)["ms_per_sample"] <= ms]
        return max(c, key=lambda x: x[0]) if c else None


    x10 = []
    for d in ("d2", "d3"):
        b = g(d, "p1free", 5); K, v = v2_at(d, b["ms_per_sample"])
        x10.append(v["U"] == v["M"] == v["T"] == 0 and v["diversity"] >= b["diversity"])
    res["X10"] = all(x10)

    print("score", sum(res.values()), "/", len(res))
    for k, v in res.items():
        print(k, "hit" if v else "miss")

    print("\nequal wall-clock (v2 NFE with ms <= baseline ms)")
    for d in ("d2", "d3"):
        for arm in ("p1", "p1final", "p1chart", "p1free", "b0free"):
            for K in (2, 5, 20):
                b = g(d, arm, K); m = v2_at(d, b["ms_per_sample"])
                mv = f"v2 NFE {m[0]:2d} ({m[1]['ms_per_sample']:.2f} ms) div {m[1]['diversity']:.4f} TV {m[1]['tv_T1']:.3f}" if m else "none"
                print(f"{d} {arm:8s} K{K:2d} {b['ms_per_sample']:8.2f} ms U+M {um(b)*100:5.2f}% T {b['T']*100:.2f}% "
                      f"div {b['diversity']:.4f} TV {b['tv_T1']:.3f} | {mv}")

    print("\nsensitivity (D2, first 50 layouts)")
    for arm in ("p1final", "p1free"):
        for K in (5, 20):
            for it in (20, 100, 300, 1000):
                r = rows.get(("d2", arm, K, it, True))
                if r:
                    print(f"{arm:8s} K{K:2d} it{it:4d} {r['ms_per_sample']:8.1f} ms U+M {um(r)*100:.2f}% T {r['T']*100:.2f}% "
                          f"TV {r['tv_T1']:.4f} div {r['diversity']:.5f} len {r['length_ratio']:.4f} res+ {r['residual_pos_frac']:.4f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d2", type=Path, default=ROOT / "results/d2/direct.jsonl")
    parser.add_argument("--d3", type=Path, default=ROOT / "results/d3/direct.jsonl")
    parser.add_argument("--sensitivity", type=Path, default=ROOT / "results/d2/direct_sens.jsonl")
    parser.add_argument("--output", type=Path, help="Optional text log; stdout used if omitted")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.d2, args.d3, args.sensitivity):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    if args.output:
        prepare_outputs(args.output, overwrite=args.overwrite)
        with args.output.open("w", encoding="utf-8") as fp, redirect_stdout(fp):
            score(args.d2, args.d3, args.sensitivity)
        print(f"saved {args.output}")
    else:
        score(args.d2, args.d3, args.sensitivity)


if __name__ == "__main__":
    main()

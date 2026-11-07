"""research phase: layout-clustered bootstrap over results/hard/boot/*.npz (locked: the archived prediction record).
  python experiments/analysis/quality_boot.py  -> results/hard/quality_boot.json
"""
import glob, json, os, re
import numpy as np

D = "results/hard/boot"; B = 4000
ref = np.load(f"{D}/refs.npz"); R, lid, mt = ref["R"], ref["lid"], ref["matched"]
cols = list(ref["cols"]); c = {k: i for i, k in enumerate(cols)}
runs = {}
for f in sorted(glob.glob(f"{D}/d*_m*_n*.npz")):
    m = re.search(r"(d\d)_m(\d)_n(\d+)", f); runs[(m[1], int(m[2]), int(m[3]))] = np.load(f)["M"]
L = int(lid.max()) + 1
divok = mt & ~np.isnan(R[:, c["div"]])


def per_layout(M):
    """per-layout sums for every statistic -> (L, k) arrays, so a bootstrap replicate is a weighted sum."""
    q = {}
    def add(name, v, mask):
        q[name] = np.bincount(lid[mask], v[mask], minlength=L)
    add("n", np.ones(len(lid)), mt); add("n_all", np.ones(len(lid)), np.ones(len(lid), bool)); add("n_div", np.ones(len(lid)), divok)
    add("m_rough", M[:, c["rough"]], mt); add("r_rough", R[:, c["rough"]], mt)
    add("m_rough_all", M[:, c["rough"]], np.ones(len(lid), bool))
    add("len_gap", M[:, c["len"]] / R[:, c["len"]] - 1, mt)
    add("lenratio_gap", M[:, c["len_ratio"]] - R[:, c["len_ratio"]], mt)
    add("clear_gap", M[:, c["clear"]] - R[:, c["clear"]], mt)
    add("m_clear", M[:, c["clear"]], mt); add("r_clear", R[:, c["clear"]], mt)
    add("hug_gap", M[:, c["hug"]] - R[:, c["hug"]], mt)
    add("m_div", M[:, c["div"]], divok); add("r_div", R[:, c["div"]], divok)
    return q


def stat(q, w):
    s = {k: (v * w).sum() for k, v in q.items()}
    rr = s["m_rough"] / s["r_rough"]
    return dict(rough_ratio=rr, len_gap=s["len_gap"] / s["n"], lenratio_gap=s["lenratio_gap"] / s["n"],
                clear_gap=s["clear_gap"] / s["n"], clear_ratio=s["m_clear"] / s["r_clear"], hug_gap=s["hug_gap"] / s["n"],
                div_ratio=s["m_div"] / s["r_div"],
                confound=((s["m_rough_all"] / s["n_all"]) / (s["r_rough"] / s["n"])) / rr, m_rough=s["m_rough"] / s["n"])


Q = {k: per_layout(M) for k, M in runs.items()}
one = np.ones(L)
point = {f"{k[0]}_m{k[1]}_n{k[2]}": {a: float(b) for a, b in stat(q, one).items()} for k, q in Q.items()}
# validation gate vs research phase
old = json.load(open("results/hard/quality_d4_matched.json"))["d2_matched_class"]
M0 = runs[("d2", 0, 11)]
gate = {k: float(abs(np.nanmean(M0[mt, c[k]]) - old[k])) for k in ("rough", "clear", "hug")}
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.integers(0, L, L), minlength=L) for _ in range(B)])
out = dict(gate=gate, point=point, B=B, layouts=L, matched_classes=int(mt.sum()))
for model in ("d2", "d4"):
    ks = [k for k in Q if k[0] == model]
    pick = rng.integers(0, len(ks), B)
    reps = [stat(Q[ks[p]], W[b]) for b, p in enumerate(pick)]
    # paired OOD part: same layouts, d2 replicate / d4 replicate (noise seed matched by index when possible)
    out[f"{model}_hier"] = {s: dict(mean=float(np.mean([r[s] for r in reps])),
                                    ci=[float(np.percentile([r[s] for r in reps], 2.5)), float(np.percentile([r[s] for r in reps], 97.5))])
                            for s in reps[0]}
    out[f"{model}_runs"] = [f"{k[0]}_m{k[1]}_n{k[2]}" for k in ks]
kd2 = [k for k in Q if k[0] == "d2"]; kd4 = [k for k in Q if k[0] == "d4"]
ood = []
for b in range(B):
    a, d = Q[kd2[rng.integers(len(kd2))]], Q[kd4[rng.integers(len(kd4))]]
    ood.append(stat(a, W[b])["m_rough"] / stat(d, W[b])["m_rough"])
out["ood_rough"] = dict(mean=float(np.mean(ood)), ci=[float(np.percentile(ood, 2.5)), float(np.percentile(ood, 97.5))])
# model-seed spread for d2 (noise pooled)
for ms in sorted({k[1] for k in kd2}):
    out.setdefault("d2_seed_rough_ratio", {})[ms] = float(np.mean([point[f"d2_m{ms}_n{n}"]["rough_ratio"] for n in sorted({k[2] for k in kd2 if k[1] == ms})]))
json.dump(out, open("results/hard/quality_boot.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("gate", "d2_hier", "d4_hier", "ood_rough", "d2_seed_rough_ratio") if k in out}, indent=1))

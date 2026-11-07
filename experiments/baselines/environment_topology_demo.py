"""research phase demo: build E0 and a few E2-style layouts, generate labeled routes, plot, save.

Run from the repo root:  python experiments/baselines/environment_topology_demo.py
Outputs go to ./results/baselines/environment_topology/
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle as MplCircle, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.env import Box, Circle, e0_env, random_layout  # noqa: E402
from topoflow.routes import enumerate_codes, make_dataset  # noqa: E402
from topoflow.topology import code_to_str, crossing_word, word_to_str  # noqa: E402

OUT = "results/baselines/environment_topology"
os.makedirs(OUT, exist_ok=True)
T = 32
CMAP = plt.get_cmap("tab10")


def draw_env(ax, env, show_cuts=True):
    for j, o in enumerate(env.obstacles):
        if isinstance(o, Circle):
            ax.add_patch(MplCircle((o.cx, o.cy), o.r, color="0.25", zorder=3))
        else:
            ax.add_patch(Rectangle((o.cx - o.hw, o.cy - o.hh), 2 * o.hw, 2 * o.hh, color="0.25", zorder=3))
        if show_cuts:
            ax.plot([o.cx, o.cx], [o.cut_top(), -1.2], ls="--", lw=1, color="crimson", zorder=2)
            ax.text(o.cx + 0.02, -1.12, "abcdefg"[j], color="crimson", fontsize=9)
    ax.plot(*env.start, "o", color="k", ms=6, zorder=4)
    ax.plot(*env.goal, "*", color="k", ms=10, zorder=4)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_paths(ax, ds, max_per_class=25, alpha=0.35):
    for i, code in enumerate(ds.classes):
        idx = np.where(ds.class_id == i)[0][:max_per_class]
        for k in idx:
            ax.plot(ds.X[k, :, 0], ds.X[k, :, 1], color=CMAP(i % 10), lw=0.9, alpha=alpha, zorder=1)
        if len(idx):
            ax.plot([], [], color=CMAP(i % 10),
                    label=f"{code_to_str(code)}  w={word_to_str(ds.word[idx[0]])}  n={np.sum(ds.class_id == i)}")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.85)


summary = {}

# ------------------------------------------------------------------ E0
rng = np.random.default_rng(0)
env0 = e0_env()
ds0 = make_dataset(env0, 1000, rng, T=T)
ds0.save(f"{OUT}/e0_balanced.npz", env0)
ds0_imb = make_dataset(env0, 1000, rng, T=T, codes=[(0,), (1,)], class_probs={(0,): 0.9, (1,): 0.1})
ds0_imb.save(f"{OUT}/e0_imbalanced_90_10.npz", env0)
summary["E0_balanced"] = ds0.stats["per_class"]
summary["E0_90_10"] = ds0_imb.stats["per_class"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
draw_env(axes[0], env0)
draw_paths(axes[0], ds0, max_per_class=40)
axes[0].set_title("E0 balanced (40/class shown)")
draw_env(axes[1], env0)
draw_paths(axes[1], ds0_imb, max_per_class=90)
axes[1].set_title("E0 imbalanced 90:10")
fig.tight_layout()
fig.savefig(f"{OUT}/e0_routes.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ validator showcase
env = env0
cases = {
    "below (a)": np.array([(-0.9, 0), (0, -0.5), (0.9, 0)]),
    "loop (aa)": np.array([(-0.9, 0), (-0.4, -0.5), (0.4, -0.5), (0.5, 0.5), (-0.5, 0.5),
                           (-0.5, -0.6), (0.6, -0.6), (0.9, 0)]),
    "backtrack (aA -> e)": np.array([(-0.9, 0), (-0.3, -0.5), (0.2, -0.5), (-0.3, -0.6),
                                     (-0.5, 0.0), (0.0, 0.5), (0.9, 0)]),
}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, p) in zip(axes, cases.items()):
    draw_env(ax, env)
    ax.plot(p[:, 0], p[:, 1], "-o", ms=3, color="tab:blue", zorder=5)
    ax.set_title(f"{name}\nreduced word = {word_to_str(crossing_word(env, p))}")
fig.tight_layout()
fig.savefig(f"{OUT}/validator_cases.png", dpi=140)
plt.close(fig)

# ------------------------------------------------------------------ E2-style layouts
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
summary["E2_layouts"] = []
rng = np.random.default_rng(42)
for ax, n_obs, box_prob in zip(axes.ravel(), [1, 2, 2, 3, 3, 3], [0, 0, 0.5, 0, 0.5, 0.3]):
    envk = random_layout(rng, n_obs, box_prob=box_prob)
    status = enumerate_codes(envk, rng, T=T)
    found = [c for c, s in status.items() if s == "found"]
    dsk = make_dataset(envk, 60 * len(found), rng, T=T, codes=found)
    draw_env(ax, envk)
    draw_paths(ax, dsk, max_per_class=15, alpha=0.45)
    nf = [code_to_str(c) for c, s in status.items() if s == "not_found"]
    ax.set_title(f"{n_obs} obstacles: {len(found)}/{2**n_obs} codes found"
                 + (f"\nnot found: {', '.join(nf)}" if nf else ""), fontsize=10)
    summary["E2_layouts"].append(dict(
        n_obs=n_obs,
        obstacles=[o.as_vector().round(3).tolist() for o in envk.obstacles],
        status={code_to_str(c): s for c, s in status.items()},
        per_class=dsk.stats["per_class"],
        collisions_in_dataset=int(envk.collision_mask(dsk.X).sum()),
    ))
fig.tight_layout()
fig.savefig(f"{OUT}/e2_layouts.png", dpi=140)
plt.close(fig)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=1)

# ------------------------------------------------------------------ console report
print("E0 balanced:", ds0.stats["per_class"])
print("E0 90:10   :", ds0_imb.stats["per_class"])
print("E0 interior representation shape:", ds0.interior().shape)
for lay in summary["E2_layouts"]:
    acc = {k: f"{v['got']}/{v['tries']}" for k, v in lay["per_class"].items()}
    print(f"E2 {lay['n_obs']} obs | status={lay['status']} | accepted/tries={acc} "
          f"| collisions={lay['collisions_in_dataset']}")

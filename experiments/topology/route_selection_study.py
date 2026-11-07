"""research phase: cost-aware targets, beta calibration, selector, and P1 with oracle / uniform / learned codes.

B0-attn is trained separately (unconditional_attention_baseline.py) and evaluated with route_selection_study.py on the same
layouts and noise.

Run from the repo root:  python experiments/topology/route_selection_study.py
"""
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from topoflow.e2 import build_layouts  # noqa: E402
from topoflow.p1 import ObstacleAttnField  # noqa: E402
from topoflow.tdfm import (N_CODES, boltzmann, calibrate_beta, code_index, evaluate_samples, found_mask,  # noqa: E402
                           generate, layout_costs, mean_norm_entropy, sample_codes, selector_inputs,
                           target_vector, train_selector)

torch.set_num_threads(max(1, os.cpu_count() or 1))
OUT = "results/topology/route_selection"
os.makedirs(f"{OUT}/models", exist_ok=True)
KS = [2, 5, 20]
N_PER = 32
REGIMES = {"uniform": 1.0, "medium": 0.5, "concentrated": 0.1}

with open("results/baselines/variable_layout/layouts.pkl", "rb") as f:
    train_layouts, unseen = pickle.load(f)

# ---------------------------------------------------------------- costs + beta
cache = f"{OUT}/costs.pkl"
if os.path.exists(cache):
    with open(cache, "rb") as f:
        C = pickle.load(f)
else:
    t0 = time.time()
    calib = build_layouts(np.random.default_rng(300), 300, 6)
    C = {}
    for name, lays in [("train", train_layouts), ("unseen", unseen), ("calib", calib)]:
        cs, fb = zip(*[layout_costs(l) for l in lays])
        C[name] = list(cs)
        print(f"costs {name}: {len(lays)} layouts, fallback codes {sum(fb)}", flush=True)
    with open(cache, "wb") as f:
        pickle.dump(C, f)
    print(f"costs in {time.time() - t0:.0f}s", flush=True)

betas = {r: (0.0 if h >= 1.0 else calibrate_beta(C["calib"], h)) for r, h in REGIMES.items()}
info = {r: dict(beta=b, entropy_calib=mean_norm_entropy(C["calib"], b),
                entropy_unseen=mean_norm_entropy(C["unseen"], b)) for r, b in betas.items()}
print(json.dumps(info, indent=1), flush=True)
with open(f"{OUT}/betas.json", "w") as f:
    json.dump(info, f, indent=1)

# ---------------------------------------------------------------- selector per regime
Xtr, Str = selector_inputs(train_layouts)
Xun, Sun = selector_inputs(unseen)
Fun = found_mask(unseen)
results = {"betas": info}
selectors = {}
for r, b in betas.items():
    Ptr = torch.tensor(np.array([target_vector(boltzmann(c, b), None) for c in C["train"]]), dtype=torch.float32)
    t0 = time.time()
    sel = train_selector(Xtr, Str, Ptr, seed=0)
    torch.save(sel.state_dict(), f"{OUT}/models/selector_{r}.pt")
    selectors[r] = sel
    Pun = np.array([target_vector(boltzmann(c, b), None) for c in C["unseen"]])
    with torch.no_grad():
        q_struct = torch.softmax(sel(Xun, Sun), -1).numpy()
        q_found = torch.softmax(sel(Xun, Fun), -1).numpy()
    tv = lambda q: float(np.mean(0.5 * np.abs(q - Pun).sum(1)))  # noqa: E731
    results[f"selector_{r}"] = dict(train_sec=round(time.time() - t0),
                                    tv_learned_feasibility=tv(q_struct), tv_oracle_mask=tv(q_found),
                                    mass_on_not_found=float((q_struct * (~Fun.numpy())).sum(1).mean()),
                                    tv_uniform_vs_target=tv(Fun.numpy() / Fun.numpy().sum(1, keepdims=True)))
    print(r, json.dumps(results[f"selector_{r}"]), flush=True)

# ---------------------------------------------------------------- P1 end-to-end
x0 = torch.randn(len(unseen) * N_PER, 60, generator=torch.Generator().manual_seed(777))
for seed in [0, 1]:
    p1 = ObstacleAttnField(30, 3, d=128, depth=4, chunk=3)
    p1.load_state_dict(torch.load(f"results/baselines/obstacle_attention/models/P1_s{seed}.pt"))
    p1.eval()
    for r, b in betas.items():
        targets = [boltzmann(c, b) for c in C["unseen"]]
        Pun = np.array([target_vector(t, None) for t in targets])
        with torch.no_grad():
            q_struct = torch.softmax(selectors[r](Xun, Sun), -1).numpy()
            q_found = torch.softmax(selectors[r](Xun, Fun), -1).numpy()
        unif = Fun.numpy() / Fun.numpy().sum(1, keepdims=True)
        modes = {"oracle": Pun, "uniform": unif, "B2_oracle_mask": q_found, "B2_learned_feas": q_struct}
        for mode, probs in modes.items():
            codes = sample_codes(probs, N_PER, seed=900)  # same code-sampling seed for every mode
            for K in KS:
                G = generate(p1, unseen, codes, K, x0)
                m = evaluate_samples(unseen, G, targets, C["unseen"], code_idx=codes)
                rec = dict(model=f"P1+{mode}", p1_seed=seed, regime=r, K=K, **m)
                print(json.dumps(rec), flush=True)
                with open(f"{OUT}/results.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
# reference: expected target cost per regime
for r, b in betas.items():
    tc = np.mean([sum(boltzmann(c, b)[k] * c[k] for k in c) for c in C["unseen"]])
    results[f"target_cost_{r}"] = float(tc)
with open(f"{OUT}/selector_results.json", "w") as f:
    json.dump(results, f, indent=1)
print("done")

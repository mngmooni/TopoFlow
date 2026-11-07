"""Explicit, preview-first replay queue for historical correction experiments.

No jobs run on import or by the default CLI. After restoring the original
layout archives and model checkpoints, inspect the plan first:

    python -m experiments.validation.run_queue
    python -m experiments.validation.run_queue --job-index 3

An explicit --execute plus --job-index or --all is required to launch model
subprocesses. Replays write to ``reproductions/direct_queue`` instead of the
archived results tree. The child script is the historical ``direct_eval.py``;
its inference/solver calculations have not been modified by this wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .output_guard import ROOT

BASE = {"TOPO_MAX_OBS": "5", "SUB": "32"}


def chunks(n, size):
    return [(a, min(a + size, n)) for a in range(0, n, size)]


D2S = dict(DATA="results/d2", N_EVAL="50", OUT="results/d2/direct_sens.jsonl",
           CHUNKDIR="results/d2/direct_chunks_sens")
D3 = dict(DATA="results/d3", SG="1", N_EVAL="300", OUT="results/d3/direct.jsonl",
          CHUNKDIR="results/d3/direct_chunks")
JOBS = [
    dict(D2S, ARMS="p1free", K="20", ITERS="300", chunk=5, n=50),
    dict(D2S, ARMS="p1free", K="5", ITERS="1000", chunk=5, n=50),
    dict(D2S, ARMS="p1free", K="20", ITERS="1000", chunk=2, n=50),
    *[dict(D3, ARMS="v2", K=k, ITERS="100") for k in ("1", "2", "5", "10", "20", "50")],
    *[dict(D3, ARMS=a, K=k, ITERS="100") for a in ("p1", "p1chart", "p1final") for k in ("2", "5", "20")],
    *[dict(D3, ARMS=a, K=k, ITERS="100", chunk=c, n=300)
      for a in ("p1free", "b0free") for k, c in (("2", 50), ("5", 30), ("20", 10))],
]


def output_paths(job: dict, output_root: Path):
    """Equivalent job identifiers, but never canonical historical destinations."""
    family = Path(job["DATA"]).name
    filename = "direct_sens.jsonl" if family == "d2" else "direct.jsonl"
    stem = "direct_chunks_sens" if family == "d2" else "direct_chunks"
    return output_root / family / filename, output_root / family / stem


def chunk_path(job: dict, chunkdir: Path, a: int):
    return chunkdir / f"{job['ARMS']}_K{job['K']}_it{job['ITERS']}_{a}.pkl"


def done(job: dict, out: Path):
    """Read-only resume check against outputs in the reproduction directory."""
    if not out.is_file():
        return False
    it = None if job["ARMS"] in ("v2", "p1", "p1chart") else int(job["ITERS"])
    with out.open(encoding="utf-8") as fp:
        for line in fp:
            row = json.loads(line)
            if (row.get("data") == job["DATA"] and row.get("arm") == job["ARMS"]
                    and row.get("K") == int(job["K"]) and row.get("iters") == it):
                return True
    return False


def required_inputs(job):
    """Static preflight; child dependencies may require additional files."""
    data = ROOT / job["DATA"]
    arm = job["ARMS"]
    checkpoints = ["sel_relabel_T1.pt"]
    if arm == "v2":
        checkpoints += ["V2_s0.pt"]
    elif arm == "b0free":
        checkpoints += ["B0attn_medium_s0.pt"]
    else:
        checkpoints += ["P1_s0.pt"]
    return [data / name for name in ("betas.json", "layouts.pkl", "chart_labels.pkl", "relabel.json")] + [
        data / "models" / name for name in checkpoints]


def execute_one(job: dict, output_root: Path, deadline: float, *, subprocess_run=subprocess.run):
    """Run one selected queue job, retaining the historical child configuration."""
    out, chunkdir = output_paths(job, output_root)
    if done(job, out):
        print("already completed:", job["DATA"], job["ARMS"], job["K"], job["ITERS"])
        return

    def launch(extra):
        if time.monotonic() >= deadline:
            raise TimeoutError("Queue budget exhausted before a new subprocess; rerun to resume")
        out.parent.mkdir(parents=True, exist_ok=True)
        chunkdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, **BASE, **{k: v for k, v in job.items() if k not in ("chunk", "n", "K")},
                   KS=job["K"], SG="1" if Path(job["DATA"]).name == "d3" else "0")
        # Ambient partial-run variables must not leak between unrelated jobs.
        env.pop("CHUNK", None)
        env.pop("MERGE", None)
        env.update(extra)
        env["OUT"] = str(out)
        env["CHUNKDIR"] = str(chunkdir)
        result = subprocess_run([sys.executable, "-m", "experiments.validation.direct_eval"],
                                cwd=ROOT, env=env, capture_output=True, text=True)
        tail = [line for line in (result.stdout + result.stderr).splitlines() if "Warning" not in line][-2:]
        print(job["DATA"], job["ARMS"], job["K"], job["ITERS"], extra,
              "rc", result.returncode, *tail, flush=True)
        if result.returncode:
            raise RuntimeError(f"Evaluation subprocess failed ({result.returncode})")

    if "chunk" in job:
        for a, b in chunks(job["n"], job["chunk"]):
            if not chunk_path(job, chunkdir, a).is_file():
                launch({"CHUNK": f"{a}:{b}"})
        launch({"MERGE": "1"})
    else:
        launch({})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--job-index", type=int, help="Zero-based index of one historical queue job")
    group.add_argument("--all", action="store_true", help="Select all historical queue jobs")
    parser.add_argument("--execute", action="store_true", help="Explicitly permit expensive evaluation subprocesses")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproductions/direct_queue")
    parser.add_argument("--budget-min", type=float, default=95, help="Maximum time *before starting new jobs* (not a process timeout)")
    args = parser.parse_args(argv)
    if args.job_index is not None and not 0 <= args.job_index < len(JOBS):
        parser.error(f"--job-index must be in range 0..{len(JOBS)-1}")
    if args.budget_min <= 0:
        parser.error("--budget-min must be positive")
    if args.execute and args.job_index is None and not args.all:
        parser.error("Execution needs an explicit --job-index N or --all")
    output_root = args.output_root.expanduser().resolve()
    if output_root.is_relative_to((ROOT / "results").resolve()):
        parser.error("The queued replay cannot write inside the archived results/ tree")
    selected = ([(args.job_index, JOBS[args.job_index])] if args.job_index is not None
                else list(enumerate(JOBS)))
    if not args.execute:
        print("PREVIEW ONLY. No subprocesses will run. Use --execute plus --job-index N or --all.")
        for idx, job in selected:
            out, _ = output_paths(job, output_root)
            missing = sum(not p.is_file() for p in required_inputs(job))
            print(f"[{idx:02d}] {job['DATA']} {job['ARMS']} K={job['K']} iters={job['ITERS']} "
                  f"{'chunked' if 'chunk' in job else 'single'} "
                  f"missing_prerequisites={missing} output={out}")
        return
    preflight = sorted({str(p) for _, job in selected for p in required_inputs(job) if not p.is_file()})
    if preflight:
        parser.error("Original data/checkpoints not restored: " + ", ".join(preflight))
    start = time.monotonic()
    for idx, job in selected:
        print(f"execute job [{idx:02d}]", flush=True)
        try:
            execute_one(job, output_root, start + args.budget_min * 60)
        except TimeoutError as exc:
            print(str(exc), flush=True)
            return
    print("queue selection completed", flush=True)


if __name__ == "__main__":
    main()

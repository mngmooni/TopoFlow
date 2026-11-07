"""Regenerate Figure 1 in a temporary location, protecting the curated PNG.

Usage: python -m tools.replay_coverage_figure
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    input_json = root / "results/final/chart_coverage_audit.json"
    frozen_figure = root / "figures/chart_coverage.png"
    for p in (input_json, frozen_figure):
        if not p.is_file():
            print(f"Missing prerequisite: {p}", file=sys.stderr)
            return 1
    with tempfile.TemporaryDirectory(prefix="topoflow_coverage_replay_") as tmp:
        scratch = Path(tmp)
        generated = scratch / "chart_coverage.png"
        env = dict(os.environ, MPLCONFIGDIR=str(scratch / "matplotlib"), PYTHONDONTWRITEBYTECODE="1")
        cmd = [sys.executable, "-m", "experiments.validation.fig_coverage",
               "--input", str(input_json), "--output", str(generated)]
        p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
        if p.returncode or not generated.is_file():
            print("Coverage figure replay failed.", p.stderr, file=sys.stderr)
            return 1
        same = digest(generated) == digest(frozen_figure)
        print("Coverage figure replay: PASS")
        print("Frozen input: results/final/chart_coverage_audit.json")
        print("Output: temporary chart_coverage.png only")
        print("Byte-identical to curated figure:", "YES" if same else "NO")
        if not same:
            print("Matplotlib/font differences can change PNG bytes. Inspect before any replacement.")
        print("Original repository files were not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

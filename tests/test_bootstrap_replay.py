"""Safe Experiment B/C analysis: import guards, deterministic synthetic regression and missing inputs.

These tests do NOT reconstruct the unavailable original per-run research archives.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.validation import baseline_boot, fidelity_boot
from experiments.validation.output_guard import ROOT


def make_baseline_inputs(folder: Path) -> None:
    folder.mkdir(parents=True)
    rng = np.random.default_rng(81)
    for ds, K, ms, ns in (("d2", 5, 0, 777), ("d2", 5, 0, 778),
                           ("d2", 20, 1, 779), ("d3", 20, 0, 777)):
        lid = np.arange(300, dtype=np.int64)
        for arm in ("v2", "p1chart"):
            bias = 0.002 if arm == "v2" else 0
            n = len(lid)
            cls = np.stack((lid, np.zeros(n), .04 + bias + .005 * rng.random(n)), axis=1)
            cat = np.full(n, "V")
            cat[lid % 19 == 0] = "M"
            np.savez(
                folder / f"{ds}_{arm}_m{ms}_n{ns}_K{K}.npz",
                lid=lid, cat=cat, cls=cls,
                len_ratio=1 + .03 * rng.random(n) + bias,
                rough=.001 + .001 * rng.random(n),
                minclear=.01 + .02 * rng.random(n),
                eu_viol=(lid % 19 == 0), sq_viol=np.zeros(n, dtype=bool),
                word_ok=np.ones(n, dtype=int), chart_infeasible=np.array(0.0), sec=np.array(10.0),
            )


def make_fidelity_inputs(folder: Path) -> None:
    folder.mkdir(parents=True)
    rng = np.random.default_rng(93)
    ncls = 12
    lid = np.repeat(np.arange(3), 4)
    refcls = np.repeat(np.arange(ncls), 4)
    for ds in ("d2", "d3"):
        data = dict(
            cls=refcls, lid=lid, nref=np.full(ncls, 4),
            len_ratio=1 + .1 * rng.random(len(refcls)),
            rough=.001 + .004 * rng.random(len(refcls)),
            minclear=.01 + .02 * rng.random(len(refcls)),
            div=.02 + .02 * rng.random(ncls),
        )
        np.savez(folder / f"{ds}_refs.npz", **data)
        for arm, model_seed, noise_seed in (("v2", 0, 11), ("v2", 0, 12),
                                             ("v2", 1, 11), ("p1chart", 0, 11)):
            cls = np.repeat(np.arange(ncls), 8)
            data = dict(
                cls=cls,
                len_ratio=1 + .1 * rng.random(len(cls)),
                rough=.001 + .004 * rng.random(len(cls)),
                minclear=.01 + .02 * rng.random(len(cls)),
                div=.02 + .02 * rng.random(ncls),
            )
            np.savez(folder / f"{ds}_{arm}_m{model_seed}_n{noise_seed}.npz", **data)


def test_bootstrap_modules_are_import_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for module in ("baseline_boot", "fidelity_boot"):
        mod = importlib.reload(importlib.import_module(f"experiments.validation.{module}"))
        assert callable(mod.main)
        assert callable(mod.analyze)
    assert not list(tmp_path.iterdir())


def test_baseline_synthetic_results_and_safe_cli(tmp_path):
    folder = tmp_path / "baseline"
    make_baseline_inputs(folder)
    result = baseline_boot.analyze(folder, bootstrap_count=30)
    assert set(result) == {"d2_K5_m0", "d2_K20_m1", "d3_K20_m0"}
    assert result["d2_K5_m0"]["noise_seeds"] == [777, 778]
    assert set(result["d2_K5_m0"]["metrics"]) == set(baseline_boot.METRICS)
    output = tmp_path / "reproductions" / "paired.json"
    baseline_boot.main(["--input-dir", str(folder), "--output", str(output), "--bootstrap", "30"])
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        baseline_boot.main(["--input-dir", str(folder), "--output", str(output), "--bootstrap", "1"])
    with pytest.raises(SystemExit):
        baseline_boot.main(["--input-dir", str(folder), "--output", str(ROOT / "results/final/baseline_paired.json"), "--overwrite"])


def test_fidelity_synthetic_results_and_safe_cli(tmp_path):
    folder = tmp_path / "fidelity"
    make_fidelity_inputs(folder)
    result = fidelity_boot.analyze(folder, bootstrap_count=30)
    assert set(result) == {"d2", "d3"}
    assert result["d2"]["matched"] == 12
    assert set(result["d2"]["arms"]) == {"v2", "p1chart"}
    assert "model_seed_rough_ratio" in result["d2"]["arms"]["v2"]
    output = tmp_path / "reproductions" / "fidelity.json"
    fidelity_boot.main(["--input-dir", str(folder), "--output", str(output), "--bootstrap", "30"])
    assert json.loads(output.read_text()) == result
    with pytest.raises(FileExistsError):
        fidelity_boot.main(["--input-dir", str(folder), "--output", str(output), "--bootstrap", "1"])
    with pytest.raises(SystemExit):
        fidelity_boot.main(["--input-dir", str(folder), "--output", str(ROOT / "results/final/distribution_fidelity.json"), "--overwrite"])


def test_missing_legacy_inputs_report_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing per-run"):
        baseline_boot.analyze(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="Missing Experiment C"):
        fidelity_boot.analyze(tmp_path / "missing")
    folder = tmp_path / "empty"
    folder.mkdir()
    with pytest.raises(FileNotFoundError, match="No complete"):
        baseline_boot.analyze(folder, bootstrap_count=3)
    with pytest.raises(FileNotFoundError, match="reference archive"):
        fidelity_boot.analyze(folder, bootstrap_count=3)

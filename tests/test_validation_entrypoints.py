"""Public replay guards for the final-validation entrypoints.

Fixtures do not contain the original research data or trained checkpoints;
these tests verify safety, deterministic analysis, and queue orchestration.
"""
from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.validation.output_guard import ROOT


def test_three_replay_modules_are_import_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for stem in ("chart_soundness", "baseline_paired", "run_queue"):
        module = importlib.import_module(f"experiments.validation.{stem}")
        assert callable(module.main)
    assert list(tmp_path.iterdir()) == []


def test_soundness_analysis_is_deterministic_and_is_read_only():
    from experiments.validation.chart_soundness import evaluate
    from topoflow.env import Circle, Env
    env = Env([Circle(0., 0., .13)])
    lay = {"env": env, "found": {}}
    first, details_a = evaluate("d2", [lay], progress=False)
    second, details_b = evaluate("d2", [lay], progress=False)
    for result in (first, second):
        result.pop("seconds")
        assert result["layouts"] == 1
        assert result["codes"] == 2
        assert result["decoded_paths"] == 48
        assert result["violations_euclid"] == 0
        assert result["violations_square"] == 0
    assert first == second
    assert details_a == details_b


def test_soundness_rejects_canonical_frozen_outputs(tmp_path):
    from experiments.validation.chart_soundness import main
    with pytest.raises(SystemExit) as exc:
        main(["--dataset", "d2", "--output-dir", str(ROOT / "results/final/soundness")])
    assert exc.value.code == 2


def test_soundness_missing_sources_do_not_create_outputs(tmp_path):
    from experiments.validation.chart_soundness import main
    folder = tmp_path / "unknown"
    out = tmp_path / "outputs"
    with pytest.raises(SystemExit) as exc:
        main(["--data-root", str(folder), "--output-dir", str(out)])
    assert exc.value.code == 2
    assert not out.exists()


def test_baseline_run_parser_rejects_invalid_specs():
    from experiments.validation.baseline_paired import parse_runs
    assert parse_runs("v2:0:777:20,p1chart:0:777:20,v2:0:777:20") == [
        ("v2", 0, 777, 20), ("p1chart", 0, 777, 20)]
    for bad in ("p1free:0:777:20", "v2:0:777", "v2:0:-1:20", "v2:0:777:0", "v2:xx:1:1"):
        with pytest.raises(ValueError):
            parse_runs(bad)


def test_baseline_rejects_archived_output_even_when_overwrite_requested():
    from experiments.validation.baseline_paired import main
    with pytest.raises(SystemExit) as exc:
        main(["--data", "results/d2", "--runs", "v2:0:777:20",
              "--output-dir", str(ROOT / "results/final/baseline"), "--overwrite"])
    assert exc.value.code == 2


def test_baseline_missing_models_fail_without_writing(tmp_path):
    from experiments.validation.baseline_paired import main
    output = tmp_path / "new_records"
    with pytest.raises(SystemExit) as exc:
        main(["--data", "results/d2", "--runs", "v2:0:777:20", "--output-dir", str(output)])
    assert exc.value.code == 2
    assert not output.exists()


def test_queue_default_is_preview_without_subprocess(tmp_path, monkeypatch, capsys):
    from experiments.validation import run_queue
    monkeypatch.setattr(run_queue.subprocess, "run", lambda *a, **k: pytest.fail("Subprocess launched in preview"))
    run_queue.main(["--job-index", "3", "--output-root", str(tmp_path / "queue")])
    txt = capsys.readouterr().out
    assert "PREVIEW ONLY" in txt and "missing_prerequisites=" in txt
    assert not (tmp_path / "queue").exists()


def test_queue_requires_explicit_job_selection_to_execute():
    from experiments.validation.run_queue import main
    with pytest.raises(SystemExit) as exc:
        main(["--execute"])
    assert exc.value.code == 2


def test_queue_refuses_to_write_archived_results():
    from experiments.validation.run_queue import main
    with pytest.raises(SystemExit) as exc:
        main(["--output-root", str(ROOT / "results/d2")])
    assert exc.value.code == 2


def test_queue_chunk_execution_and_resume_on_mock_subprocess(tmp_path):
    from experiments.validation import run_queue
    job = dict(run_queue.D2S, ARMS="p1free", K="5", ITERS="300", chunk=1, n=2)
    called = []

    def fake_run(args, **kw):
        env = kw["env"]
        called.append((args, env.copy()))
        assert Path(env["OUT"]).is_relative_to(tmp_path)
        assert Path(env["CHUNKDIR"]).is_relative_to(tmp_path)
        if "CHUNK" in env:
            a = int(env["CHUNK"].split(":")[0])
            target = run_queue.chunk_path(job, Path(env["CHUNKDIR"]), a)
            target.write_bytes(b"synthetic chunk marker")
        else:
            with Path(env["OUT"]).open("a", encoding="utf-8") as f:
                f.write(json.dumps({"data": job["DATA"], "arm": job["ARMS"],
                                    "K": int(job["K"]), "iters": int(job["ITERS"])}) + "\n")
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    run_queue.execute_one(job, tmp_path, time.monotonic() + 20, subprocess_run=fake_run)
    assert len(called) == 3  # two chunks and one merge
    assert all(a[0][1:] == ["-m", "experiments.validation.direct_eval"] for a in called)
    assert called[-1][1]["MERGE"] == "1"
    run_queue.execute_one(job, tmp_path, time.monotonic() + 20, subprocess_run=fake_run)
    assert len(called) == 3  # resume from the completed JSONL row


def test_queue_budget_prevents_starting_subprocess(tmp_path):
    from experiments.validation import run_queue
    with pytest.raises(TimeoutError):
        run_queue.execute_one(run_queue.JOBS[3], tmp_path, time.monotonic() - 1,
                              subprocess_run=lambda *a, **kw: pytest.fail("Unexpected subprocess"))

"""Reproduction entrypoints: no import-time execution, protected outputs, fixed numeric summaries."""
from __future__ import annotations
from contextlib import redirect_stdout
import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.validation.output_guard import ROOT, prepare_outputs


def test_guard_rejects_overwrite_and_duplicate_outputs(tmp_path):
    existing = tmp_path / "frozen.json"
    existing.write_text('original', encoding='utf-8')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        prepare_outputs(existing)
    assert existing.read_text(encoding='utf-8') == 'original'
    with pytest.raises(ValueError, match='same output path'):
        prepare_outputs(tmp_path / 'shared', tmp_path / 'shared')
    fresh = tmp_path / 'new' / 'results.json'
    prepare_outputs(fresh)
    assert fresh.parent.is_dir()
    assert not fresh.exists()


def test_selected_modules_are_import_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ('direct_score', 'latency_score', 'fig_coverage', 'fig_fidelity', 'fig_trajectories'):
        module = importlib.import_module(f'experiments.validation.{name}')
        assert callable(module.main)
    assert not any(tmp_path.iterdir()), 'Import of a figure or scoring module wrote to working directory'


def test_direct_score_matches_archived_calculation():
    from experiments.validation.direct_score import score
    target = io.StringIO()
    with redirect_stdout(target):
        score(ROOT / 'results/d2/direct.jsonl', ROOT / 'results/d3/direct.jsonl',
              ROOT / 'results/d2/direct_sens.jsonl')
    rendered = target.getvalue()
    assert rendered.startswith('score 8 / 10\n')
    assert hashlib.sha256(rendered.encode()).hexdigest() == (
        '40664f00b6f6dba38cffee4c973303eb75184448b83210c89187d17d94953299'
    )


def test_latency_score_regenerates_frozen_json_without_writing_frozen_evidence(tmp_path):
    from experiments.validation.direct_score import score
    from experiments.validation.latency_score import analyze
    log = tmp_path / 'direct_score.txt'
    with log.open('w', encoding='utf-8') as fp, redirect_stdout(fp):
        score(ROOT / 'results/d2/direct.jsonl', ROOT / 'results/d3/direct.jsonl',
              ROOT / 'results/d2/direct_sens.jsonl')
    out_json = tmp_path / 'replay.json'
    out_png = tmp_path / 'replay.png'
    prepare_outputs(out_json, out_png)
    with redirect_stdout(io.StringIO()):
        analyze(ROOT / 'results/final/baseline_latency.jsonl',
                ROOT / 'results/final/baseline_paired.json', log, out_json, out_png)
    reproduced = json.loads(out_json.read_text(encoding='utf8'))
    frozen = json.loads((ROOT / 'results/final/baseline_latency.json').read_text(encoding='utf8'))
    # Frozen manifest records the historical pre-migration path under outputs/.
    # The refactored source retains the scoring script's results/-relative source.
    assert reproduced.pop('source') == 'results/final/baseline_latency.jsonl'
    assert frozen.pop('source') == 'outputs/final/baseline_latency.jsonl'
    assert reproduced == frozen
    assert out_png.is_file() and out_png.stat().st_size > 20000


def test_fidelity_render_on_synthetic_npz_and_no_missing_reference_error(tmp_path):
    from experiments.validation.fig_fidelity import render
    folder = tmp_path / 'samples'
    folder.mkdir()
    rng = np.random.default_rng(153)
    for ds in ('d2', 'd3'):
        ncls = 12
        refcls = np.repeat(np.arange(ncls), 4)
        data = {'cls': refcls, 'lid': np.arange(ncls) // 4,
                'nref': np.full(ncls, 4),
                'len_ratio': 1 + rng.random(len(refcls)) / 10,
                'rough': 0.001 + rng.random(len(refcls)) * .003,
                'minclear': .01 + rng.random(len(refcls)) * .02,
                'div': .02 + rng.random(ncls) * .02}
        np.savez(folder / f'{ds}_refs.npz', **data)
        for arm in ('v2', 'p1chart'):
            cls = np.repeat(np.arange(ncls), 8)
            data = {'cls': cls, 'len_ratio': 1 + rng.random(len(cls)) / 10,
                    'rough': .001 + rng.random(len(cls)) * .003,
                    'minclear': .01 + rng.random(len(cls)) * .02,
                    'div': .02 + rng.random(ncls) * .02}
            np.savez(folder / f'{ds}_{arm}_m0_n11.npz', **data)
    png = tmp_path / 'fidelity.png'
    render(folder, png)
    assert png.is_file() and png.stat().st_size > 20000

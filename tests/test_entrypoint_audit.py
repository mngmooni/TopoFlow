"""Ensure source inventories do not execute experiment code."""
from pathlib import Path

from tools.audit_entrypoints import audit_file, audit_repo, to_markdown


def test_audit_never_runs_script(tmp_path):
    (tmp_path / "experiments").mkdir()
    (tmp_path / "results").mkdir()
    script = tmp_path / "experiments" / "bad.py"
    marker = tmp_path / "should_not_exist"
    script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('BAD')\n")
    info = audit_file(script, tmp_path)
    assert info.potential_import_time_work
    assert not marker.exists()
    data = audit_repo(tmp_path)
    assert data["experiment_scripts"] == 1
    assert not marker.exists()
    assert "bad.py" in to_markdown(data)


def test_audit_scans_real_experiments():
    from tools.audit_entrypoints import ROOT
    data = audit_repo(ROOT)
    names = {row["path"]: row for row in data["entries"]}
    assert "experiments/validation/chart_soundness.py" in names
    assert names["experiments/validation/chart_soundness.py"]["reads_pickle"]
    assert "experiments/validation/fig_fidelity.py" in names
    assert names["experiments/validation/fig_fidelity.py"]["main_guard"]

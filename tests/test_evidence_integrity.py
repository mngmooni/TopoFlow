"""Fast, read-only regression checks of shipped frozen study artifacts."""
from pathlib import Path

from tools.verify_evidence import audit


def test_frozen_evidence_cross_checks_and_checksums():
    root = Path(__file__).resolve().parents[1]
    result = audit(root)
    assert result.ok, "\n".join(result.failures)
    assert result.summary["decoded_paths"] == 695488
    assert result.summary["paired_generated_paths"] == 460800

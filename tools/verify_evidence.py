"""Read-only checks of the curated, frozen TopoFlow paper evidence.

Usage (from any working directory):
    python -m tools.verify_evidence
    python -m tools.verify_evidence --root /path/to/TopoFlow

This checks internal consistency of *saved artifacts*. It does not rerun
training, regenerate data, or independently verify every sampled trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

FAMILIES = ("d2", "d3", "d4", "H1", "H2", "H3", "H4")
FIGURES = (
    "chart_coverage.png",
    "pareto_time_diversity.png",
    "matched_trajectories.png",
    "fidelity_distributions.png",
)
REQUIRED_ARTIFACTS = (
    "results/final/chart_soundness.json",
    "results/final/chart_coverage_audit.json",
    "results/final/baseline_paired.json",
    "results/final/baseline_latency.json",
    "results/final/distribution_fidelity.json",
)
MISSING_RUNTIME_INPUT_EXAMPLES = (
    "results/d2/layouts.pkl",
    "results/d2/models/V2_s0.pt",
    "results/final/fidelity/d2_refs.npz",
    "results/final/baseline/d2_v2_m0_n777_K20.npz",
)


@dataclass(frozen=True)
class EvidenceAudit:
    ok: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    summary: dict[str, int]
    absent_runtime_inputs: tuple[str, ...]


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def audit(root: Path) -> EvidenceAudit:
    root = Path(root).resolve()
    checks: list[str] = []
    failures: list[str] = []
    summary: dict[str, int] = {}

    def check(condition: bool, explanation: str) -> None:
        (checks if condition else failures).append(explanation)

    for rel in REQUIRED_ARTIFACTS:
        check((root / rel).is_file(), f"required evidence exists: {rel}")
    if failures:
        return EvidenceAudit(False, tuple(checks), tuple(failures), summary,
                             tuple(p for p in MISSING_RUNTIME_INPUT_EXAMPLES if not (root / p).exists()))

    sound = json.loads((root / "results/final/chart_soundness.json").read_text(encoding="utf8"))
    cover = json.loads((root / "results/final/chart_coverage_audit.json").read_text(encoding="utf8"))
    paired = json.loads((root / "results/final/baseline_paired.json").read_text(encoding="utf8"))
    fidelity = json.loads((root / "results/final/distribution_fidelity.json").read_text(encoding="utf8"))

    check(set(sound) == set(FAMILIES) == set(cover), "all seven soundness/coverage families are present")
    paths = segments = codes = layouts = n60 = refined = infeasible = 0
    for fam in FAMILIES:
        if fam not in sound or fam not in cover:
            continue
        s, c = sound[fam], cover[fam]
        taxonomy = c["taxonomy_all"]
        covered = int(taxonomy["covered_n60"])
        recovered = int(taxonomy["resolution_recovered"])
        invalid = int(taxonomy["certified_infeasible"])
        check(sum(int(v) for v in taxonomy.values()) == c["codes"], f"{fam}: taxonomy partitions codes")
        check(s["codes"] == c["codes"] and s["layouts"] == c["layouts"], f"{fam}: soundness and coverage populations agree")
        check(s["ok60"] == covered and s["bisect_recovered"] == recovered, f"{fam}: chart success counts agree")
        check(all(s[key] == 0 for key in ("violations_euclid", "violations_square", "exact_failed")),
              f"{fam}: no measured geometric violations in saved soundness report")
        check(s["word_mismatch_paths"] == 0, f"{fam}: no reported word mismatches (not a certificate on invalid-cut layouts)")
        codes += int(c["codes"])
        layouts += int(c["layouts"])
        paths += int(s["decoded_paths"])
        segments += int(s["segments"])
        n60 += covered
        refined += recovered
        infeasible += invalid
    summary.update(layouts=layouts, route_code_instances=codes, decoded_paths=paths,
                   checked_segments=segments, covered_n60=n60, refined=refined, certified_infeasible=infeasible)
    check(layouts == 2100 and codes == 82224, "soundness experiment totals match frozen paper report")
    check(paths == 695488 and segments == 42631824, "soundness tested-path and segment totals match frozen paper report")
    check(n60 + refined + infeasible == codes, "coverage categories exhaust tested route-code instances")

    n_compared = 0
    check(len(paired) == 8, "eight primary model-seed/NFE comparison configurations are recorded")
    for key, val in paired.items():
        check(tuple(val["noise_seeds"]) == (777, 778, 779), f"{key}: sampling seeds match evaluation plan")
        check(set(val["runs"]) == {"777", "778", "779"}, f"{key}: all sampling runs present")
        for run_id, run in val["runs"].items():
            for arm in ("v2", "p1chart"):
                info = run[arm]
                n_compared += info["n"]
                check(all(info[field] == 0 for field in ("indep_euclid_viol", "indep_square_viol", "indep_word_bad")),
                      f"{key}/{run_id}/{arm}: no measured violation recorded")
    summary["paired_generated_paths"] = n_compared
    check(n_compared == 460800, "paired generative test count matches frozen paper report")

    for fam in ("d2", "d3"):
        stats = fidelity[fam]
        check(stats["classes_chart_feasible"] == cover[fam]["taxonomy_all"]["covered_n60"],
              f"{fam}: fidelity class count agrees with chart coverage")
        check(stats["matched"] + stats["unmatched_no_reference"] == stats["classes_chart_feasible"],
              f"{fam}: matched/unmatched class count partitions chart-feasible classes")
    summary["fidelity_matched_d2"] = fidelity["d2"]["matched"]
    summary["fidelity_matched_d3"] = fidelity["d3"]["matched"]
    check(summary["fidelity_matched_d2"] == 6240 and summary["fidelity_matched_d3"] == 6147,
          "fidelity matched-class counts match frozen paper report")

    for fig in FIGURES:
        path = root / "figures" / fig
        check(path.is_file() and path.stat().st_size > 1000 and path.open("rb").read(8) == b"\x89PNG\r\n\x1a\n",
              f"curated paper figure exists and has PNG signature: {fig}")

    manifest = root / "results/final/ARTIFACT_SHA256.txt"
    check(manifest.is_file(), "artifact checksum manifest exists")
    if manifest.is_file():
        lines = [l for l in manifest.read_text(encoding="utf8").splitlines() if l.strip() and not l.startswith("#")]
        check(len(lines) >= len(REQUIRED_ARTIFACTS) + len(FIGURES), "manifest covers core results and all paper figures")
        for item in lines:
            digest, rel = item.split("  ", 1)
            path = root / rel
            check(path.is_file() and _sha256(path) == digest, f"SHA-256 intact: {rel}")

    absent = tuple(rel for rel in MISSING_RUNTIME_INPUT_EXAMPLES if not (root / rel).is_file())
    return EvidenceAudit(not failures, tuple(checks), tuple(failures), summary, absent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="the checked-out TopoFlow root")
    args = parser.parse_args()
    result = audit(args.root)
    print("Frozen evidence consistency:", "PASS" if result.ok else "FAIL")
    print("Checks passed:", len(result.checks), "  Failed:", len(result.failures))
    for name, val in result.summary.items():
        print(f"  {name}: {val:,}")
    if result.failures:
        for issue in result.failures:
            print("  ERROR:", issue)
    print("\nExamples of runtime inputs not bundled (expected, not an evidence-check failure):")
    for rel in result.absent_runtime_inputs:
        print("  -", rel)
    print("\nScope: validates internal consistency and checksums of saved reports;"
          " does NOT rerun the original neural evaluations or prove path feasibility.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

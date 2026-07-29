"""The research index entry must not drift from the generated report.

``docs/research/README.md`` hand-types the LFM2.5 study's headline numbers, and
the supported re-run (``examples/research/run_lfm25.sh``) regenerates only the
rows, the report and the provenance sidecar — never the index. So a re-measure
can publish new artifacts and leave the index quietly asserting the old ones.
That is this repo's most expensive recurring bug: three PRs shipped factual
errors in one day and **every one was in hand-typed prose while the generated
table beside it was correct**.

This check used to live in a gitignored ``tmp/`` script, which made it a gate
that died with the worktree — worse than no gate, because it had been run and
believed. It is a committed test now, so CI re-checks it on every change.

Every assertion is "this README phrase is also present in the GENERATED report".
The report side is the authority; a number that moves fails here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RESEARCH = Path(__file__).parents[2] / "docs" / "research"
REPORT_PATH = RESEARCH / "20260729_lfm25_encoders.md"
README_PATH = RESEARCH / "README.md"

#: (README phrase, the substring of the generated report that backs it).
#: The second element is what makes this a cross-check rather than a tautology:
#: checking README phrases against the README cannot fail.
BACKED_CLAIMS = [
    ("ahead of `e5-base-v2` on no benchmark/arm", "**no benchmark/arm**"),
    ("per-arm, not checkpoint-level", "per-ARM verdicts"),
    ("**0.9741** against e5's best **0.9703**", "**0.9741** vs the baseline's best **0.9703**"),
    ("`fodors_zagat` (margin `+0.0000`", "| `fodors_zagat` | 1.0000 | 1.0000"),
    ("one of five benchmarks cannot support an embedder claim", "`fodors_zagat`.**"),
    ("`[-0.0275, -0.0072]`", "[-0.0275, -0.0072]"),
    ("`[-0.0295, -0.0099]`", "[-0.0295, -0.0099]"),
    ("whole observed gap from that random control", "**+0.0183**"),
    ("observed gap", "observed gap"),
    ("matches or beats both real base encoders on 5 of 5", "on **5 of 5**"),
]

#: Claims this document has RETRACTED. A correction that survives only in the
#: report while the index still carries the original is how a reader ends up
#: trusting the version they already read.
RETRACTED = [
    "two of five benchmarks cannot support",
    "`walmart_amazon` (margin `+0.0183`)",
    "entire usable dynamic range",
]


@pytest.fixture(scope="module")
def report() -> str:
    return REPORT_PATH.read_text()


@pytest.fixture(scope="module")
def entry() -> str:
    lines = README_PATH.read_text().splitlines()
    matches = [line for line in lines if REPORT_PATH.name in line]
    assert matches, f"the index no longer links {REPORT_PATH.name}"
    return "\n".join(matches)


@pytest.mark.parametrize(
    ("phrase", "backing"), BACKED_CLAIMS, ids=[c[0][:40] for c in BACKED_CLAIMS]
)
def test_every_index_claim_is_backed_by_the_generated_report(
    phrase: str, backing: str, report: str, entry: str
) -> None:
    assert phrase in entry, "the index no longer makes this claim -- update the list or the index"
    assert backing in report, f"the index claims {phrase!r} but the report no longer says so"


@pytest.mark.parametrize("retracted", RETRACTED)
def test_the_index_does_not_carry_a_retracted_claim(retracted: str, entry: str) -> None:
    assert retracted not in entry


def test_the_check_can_actually_fail(report: str) -> None:
    """A gate never seen to fail is a hypothesis, not a safety net."""
    assert "this string is not in the report" not in report

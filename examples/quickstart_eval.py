"""Quickstart: turn judged pairs into a self-contained eval tearsheet, for free.

The story: judging costs money (LLM calls, embeddings); *analysing* what you
already judged is free. This example blocks a toy dataset, judges every candidate
pair, clusters the matches, then builds an ``EvalReport`` and writes a single
self-contained ``tearsheet.html`` -- pair precision/recall/F1, PR and ROC curves,
a score histogram, a confidence-calibration diagram, and the most-confident
errors -- at **zero** extra cost.

Fully offline on purpose: no API key, no network, no torch, no embeddings. The
judge here is a :class:`~langres.testing.ScriptedJudge` standing in for a real
one -- it scores each pair by string similarity and reports a confidence, exactly
the shape an ``LLMMatcher(confidence="logprob")`` produces, but with no spend. For a
real (paid) judge you would wrap the scoring call in
:func:`~langres.core.benchmark.evaluate` for its default $1 spend cap; the report
below is identical either way.

Run it:
    uv run python examples/quickstart_eval.py
"""

import difflib
from pathlib import Path

from langres import Resolver
from langres.core.benchmark import gold_pairs_from_clusters
from langres.core.clusterer import Clusterer
from langres.core.eval_report import EvalReport
from langres.core.models import CompanySchema
from langres.testing import ScriptedJudge

# A toy dataset and the true clustering (what a small labeled sample gives you).
# The last pair is a deliberately HARD true match -- "IBM" and its full legal
# name share almost no characters, so a string-similarity judge scores it low and
# misses it. That one honest error is what gives the tearsheet something to show:
# a non-trivial ROC/PR curve and a populated "most-confident errors" panel.
records = [
    {"id": "1", "name": "Acme Corporation"},
    {"id": "2", "name": "Acme Corp"},
    {"id": "3", "name": "Globex Inc"},
    {"id": "4", "name": "Globex Incorporated"},
    {"id": "5", "name": "Initech"},
    {"id": "6", "name": "Initech LLC"},
    {"id": "7", "name": "Unrelated Bakery"},
    {"id": "8", "name": "IBM"},
    {"id": "9", "name": "International Business Machines"},
]
gold_clusters = [{"1", "2"}, {"3", "4"}, {"5", "6"}, {"7"}, {"8", "9"}]

# 1. BLOCK: generate candidate pairs to judge (no judge, no spend yet).
resolver = Resolver.from_schema(CompanySchema, matcher="string")
candidates = resolver.candidates(records)


# 2. JUDGE: a mocked judge that scores each pair AND reports a confidence,
#    standing in for LLMMatcher(confidence="logprob") -- deterministic, offline.
def _similarity(candidate: object) -> float:
    left, right = candidate.left.name.lower(), candidate.right.name.lower()  # type: ignore[attr-defined]
    return difflib.SequenceMatcher(None, left, right).ratio()


def _credence(candidate: object) -> float:
    # More confident the further the score is from the 0.5 fence -- a plausible
    # stand-in for a real judge's self-reported credence.
    return round(0.5 + abs(_similarity(candidate) - 0.5), 3)


judge = ScriptedJudge(
    _similarity,
    confidence=_credence,
    confidence_source="logprob",
    provenance={"cost_usd": 0.0},  # free judge -> $0 cost track
)
judgements = list(judge.forward(iter(candidates)))

# 3. CLUSTER: transitive-closure the predicted matches into entities.
clusters = Clusterer(threshold=0.6).cluster(judgements)
print(f"{len(clusters)} cluster(s): {[sorted(c) for c in clusters]}")

# 4. REPORT: grade the judgements against gold and render a $0 tearsheet.
gold_pairs = gold_pairs_from_clusters(gold_clusters)
report = EvalReport.from_judgements(judgements, gold_pairs, threshold=0.6)
print(report.summary)

out_path = Path("tearsheet.html")
out_path.write_text(report.to_html(title="langres eval tearsheet"), encoding="utf-8")
print(f"\nWrote {out_path.resolve()} -- open it in a browser (no server needed).")

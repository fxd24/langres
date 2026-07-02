"""Quickstart: dedupe records with zero labels in a handful of lines.

Offline by default (D4): no API key is required, no network call is made, and
no embedding model is downloaded -- this toy dataset (N <= 100) resolves
through langres.dedupe's default zero-spend "string" judge. If
OPENROUTER_API_KEY or OPENAI_API_KEY is set, this prints an "upgrade" note
(it does NOT actually spend money -- see docs/EXPERIMENTS.md for a real
LLM-judge walkthrough).

Run it:
    uv run python examples/quickstart_verbs.py
"""

import os

from langres import dedupe

# --- the ~10 lines that matter: dedupe a batch of records, zero labels ---
records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
    {"id": "4", "name": "Totally Different Company", "city": "Chicago"},
    {"id": "5", "name": "Unrelated Bakery", "city": "Miami"},
]

result = dedupe(records, judge="string", threshold=0.6)

for cluster in result:
    print(sorted(cluster))
# --- end of the dedupe walkthrough ---

print(
    f"\n{len(result)} cluster(s) found, judge_used={result.judge_used!r}, "
    f"score_type={result.score_type!r}"
)

if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"):
    print(
        "\nAn OPENROUTER_API_KEY/OPENAI_API_KEY is set: dedupe(records) with the "
        'default judge="auto" would pick a real LLM judge instead of the '
        "zero-spend 'string' one. This example pins judge=\"string\" deliberately "
        'to stay free and fully offline -- drop that kwarg (or pass judge="auto") '
        "to try it, spend-capped at $1 by default (budget_usd=)."
    )
else:
    print(
        "\nNo OPENROUTER_API_KEY/OPENAI_API_KEY set: dedupe(records) with the "
        "default judge=\"auto\" would fall back to this same zero-spend 'string' "
        "judge (with one warning). Set one of those env vars to try a real LLM "
        "judge -- it runs under a $1 default spend cap (budget_usd=)."
    )

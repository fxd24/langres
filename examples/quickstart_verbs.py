"""Quickstart: dedupe records with zero labels in a handful of lines.

Offline on purpose: this example pins judge="string", so no API key is
required, no network call is made, and no embedding model is downloaded --
this toy dataset (N <= 100) resolves through the zero-spend "string" judge.
The default judge="auto" is different: it requires an LLM API key
(OPENROUTER_API_KEY/OPENAI_API_KEY, plus the [llm] extra) and raises
NoJudgeAvailableError without one -- langres never silently falls back to
fuzzy matching. If a key IS set, this prints an "upgrade" note (it does NOT
actually spend money -- see docs/EXPERIMENTS.md for a real LLM-judge
walkthrough).

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
        'default judge="auto" would raise NoJudgeAvailableError -- langres '
        "never silently falls back to fuzzy matching. This example opts into "
        "the zero-spend 'string' judge explicitly (judge=\"string\"). To use a "
        "real LLM judge, set one of those env vars and install the [llm] extra "
        "(uv sync --extra llm) -- it runs under a $1 default spend cap "
        "(budget_usd=)."
    )

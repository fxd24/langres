"""The flywheel inlet: log every judge call, then read the log back.

``log=`` is opt-in on both ``.dedupe()`` and ``.compare()`` (omit it -- the
default -- for zero overhead). Pass a path (or a ``JudgementLog`` instance
for more control, e.g. ``features=True``) and every judge call is appended
as one JSON line: pair ids, score, verdict, model, cost, decision_step,
timestamp, and a schema-version field ``"v": 1``.

This is the harvest source a future milestone (W2.4) turns into labeled
training pairs for ``derive_threshold``/``fit()`` -- see the "Signal log"
section of docs/EXPERIMENTS.md.

Offline and free: the zero-spend "string" judge, no API key needed.

Run it:
    uv run python examples/judgement_log_demo.py
"""

import json

from langres import JudgementLog
from langres.architectures import FuzzyString

records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
]

log = JudgementLog("tmp/judgement_log_demo.jsonl")
result = FuzzyString(threshold=0.6).dedupe(records, log=log)

print(f"{len(result)} cluster(s): {list(result)}")

# --- read the log back (the round-trip) ---
rows = log.read()
print(f"\n{len(rows)} judgement(s) logged to {log.path}:")
for row in rows:
    print(json.dumps(row))

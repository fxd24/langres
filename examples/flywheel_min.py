"""The langres flywheel in one script -- offline, $0, zero labels to start.

dedupe -> log every judge call -> review the uncertain margin -> label it via
the CLI's CSV round-trip -> harvest -> data-driven threshold -> re-run -> grade
both passes against a small gold sample. Run: uv run python examples/flywheel_min.py
"""

import csv
import subprocess
import sys
from pathlib import Path

from langres import (
    CorrectionLog,
    EvalReport,
    JudgementLog,
    ReviewQueue,
    dedupe,
    derive_threshold_from_pairs,
    gold_pairs_from_clusters,
    harvest_labeled_pairs,
    select_for_review,
)

WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp/flywheel_min")
WORK.mkdir(parents=True, exist_ok=True)

# Three true duplicate pairs -- plus same-city DISTINCT companies (Stark/Wayne,
# Pied Piper/Hooli) that the string judge's default 0.5 cut over-merges.
records = [
    {"id": "a1", "name": "Acme Corporation", "city": "New York"},
    {"id": "a2", "name": "Acme Corp", "city": "New York"},
    {"id": "g1", "name": "Globex Inc", "city": "Boston"},
    {"id": "g2", "name": "Globex Incorporated", "city": "Boston"},
    {"id": "i1", "name": "Initech", "city": "Austin"},
    {"id": "i2", "name": "Initech LLC", "city": "Austin"},
    {"id": "s1", "name": "Stark Industries", "city": "Chicago"},
    {"id": "w1", "name": "Wayne Enterprises", "city": "Chicago"},
    {"id": "p1", "name": "Pied Piper", "city": "Palo Alto"},
    {"id": "h1", "name": "Hooli", "city": "Palo Alto"},
]
# A small labeled gold sample: set[frozenset] of true pairs. Grades the
# tearsheet only -- the tuned threshold comes from the review loop below.
gold = gold_pairs_from_clusters([{"a1", "a2"}, {"g1", "g2"}, {"i1", "i2"}])

# [1] First pass: offline string judge, every call logged (the flywheel inlet).
log = JudgementLog(WORK / "judgements.jsonl")
first = dedupe(records, judge="string", log=log)  # threshold=None -> 0.5 default
rows = log.read()
print(f"[1] clusters @ {first.threshold}: {sorted(sorted(c) for c in first)}")

# [2] Queue the pairs the judge was least sure about -- the cut comes off the result.
items = select_for_review(
    rows,
    strategy="uncertainty",
    threshold=first.threshold,
    margin=0.25,
    records=records,
    limit=20,
    audit_fraction=0.0,
)
queue = WORK / "review_queue.jsonl"
ReviewQueue(queue).write(items)

# [3] Label via the CLI CSV round-trip (the real UX: a spreadsheet, not JSONL).
labeled, corrections = WORK / "to_label.csv", WORK / "corrections.jsonl"
subprocess.run(["langres", "export-csv", str(queue), str(labeled)], check=True)
with labeled.open(newline="", encoding="utf-8") as fh:
    table = list(csv.DictReader(fh))
for row in table:  # stand-in for the human: fill 'label' (y/n) from gold
    row["label"] = "y" if frozenset({row["left_id"], row["right_id"]}) in gold else "n"
with labeled.open("w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(table[0]))
    writer.writeheader()
    writer.writerows(table)
subprocess.run(
    ["langres", "import-csv", str(labeled), str(queue), "--out", str(corrections)], check=True
)

# [4] Harvest logged verdicts + human corrections into a data-driven cut.
pairs = harvest_labeled_pairs(rows, CorrectionLog(corrections).read())
tuned = derive_threshold_from_pairs(pairs)
print(f"[4] harvested {len(pairs)} labeled pairs -> tuned threshold {tuned:.3f}")

# [5] Re-run at the tuned cut: the over-merged clusters split.
second = dedupe(records, judge="string", threshold=tuned, log=JudgementLog(WORK / "v2.jsonl"))
print(f"[5] clusters @ {second.threshold:.3f}: {sorted(sorted(c) for c in second)}")

# [6] Grade both passes against gold at $0 and write the HTML tearsheet.
before = EvalReport.from_log(rows, gold, threshold=first.threshold)
print(f"[6] BEFORE {before.summary}")
after = EvalReport.from_log(rows, gold, threshold=tuned)
print(f"[6] AFTER  {after.summary}")
(WORK / "tearsheet.html").write_text(after.to_html(title="flywheel: tuned"), encoding="utf-8")
print(f"[6] tearsheet -> {WORK / 'tearsheet.html'}")

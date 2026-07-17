"""Quickstart: dedupe records with zero labels, by naming the model you want.

Offline and free, and — this is the point — *obviously* so. `FuzzyString` is a
whole ER pipeline: all-pairs blocking, per-field string similarity, a weighted
average, transitive-closure clustering. It cannot make a paid API call, cannot
read your API keys, and cannot touch the network, because none of its parts do.
You can see that from its name and its one file (src/langres/architectures/).

That last paragraph used to be much longer. `dedupe(records)` defaulted to
matcher="auto", which sniffed OPENROUTER_API_KEY/OPENAI_API_KEY out of your
environment and spent real money on whatever it found — so this example had to
pin matcher="string" and then explain, at length, what it was opting out of and
why. There is nothing to opt out of now: a model that spends money is a model
you constructed on purpose (see the bottom of this file).

Run it:
    uv run python examples/quickstart_models.py
"""

from langres.architectures import FuzzyString

# --- the ~5 lines that matter: name a model, dedupe a batch, zero labels ---
records = [
    {"id": "1", "name": "Acme Corporation", "city": "New York"},
    {"id": "2", "name": "Acme Corp", "city": "New York"},
    {"id": "3", "name": "Totally Different Co", "city": "Chicago"},
    {"id": "4", "name": "Totally Different Company", "city": "Chicago"},
    {"id": "5", "name": "Unrelated Bakery", "city": "Miami"},
]

result = FuzzyString(threshold=0.6).dedupe(records)

for cluster in result:
    print(sorted(cluster))
# --- end of the dedupe walkthrough ---

# The result is a plain list[set[str]] that also says what produced it, so you
# never have to remember which model ran or what its scores meant.
print(
    f"\n{len(result)} cluster(s) found, architecture={result.architecture!r}, "
    f"backbone={result.backbone!r}, score_type={result.score_type!r}, "
    f"threshold={result.threshold!r}"
)
# backbone=None is not a gap in the reporting — it is the honest answer. Nothing
# with weights ran here. A model that ran one would name it.

# Same model, one pair, a verdict you can branch on:
verdict = FuzzyString(threshold=0.6).compare(records[0], records[1])
print(f"\ncompare(1, 2) -> {verdict!r}")
print(f"  match={verdict.match} score={verdict.score:.3f} (truthy: {bool(verdict)})")

print(
    "\nWant a better answer than fuzzy string matching? Name a model that can "
    "give you one — it is one line, and it is explicit:\n"
    "\n    from langres.architectures import VectorLLMCascade\n"
    '    model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini")\n'
    "    clusters = model.dedupe(records)\n"
    "\nThat one makes PAID API calls (needs OPENROUTER_API_KEY and the [llm] +\n"
    "[semantic] extras), under a $1 default spend cap you can set with budget_usd=.\n"
    "It spends because you named it, not because it found a key lying around."
)

# A closing note on quality, since this example is free and therefore tempting:
# unsupervised fuzzy matching over-merges on unlabeled data. That is exactly why
# it was never a silent fallback for a missing API key. Calibrate the threshold
# against real labels (langres.training.calibration.derive_threshold, or
# fit(method=Platt())) before trusting FuzzyString on anything that matters.

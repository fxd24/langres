"""Quickstart: mine a training-ready pair set from labeled candidates -- for free.

Before you fine-tune or fit a matcher, the *labeled pairs* need preparation: the
classes are usually lopsided (many more non-matches than matches), some positives
are boundary-hard, and a few labels are simply wrong. This example runs the
:mod:`langres.data.mining` substrate over a tiny in-file company dataset and then
summarises the result with a :class:`MiningReadinessSection`.

The miners work on the pair currency ``(ERCandidate, is_match)``. The two
*featurizing* miners (hard-positive mining, denoise) need a ``comparison`` vector
on each candidate, so we attach one with a single ``StringComparator`` pass -- the
same thing ``Resolver.candidates(records)`` does inside a pipeline.

Fully offline: no API key, no network, no LLM. It does use scikit-learn (the
``[trained]`` extra) for the out-of-fold RandomForest behind hard-positive mining
and denoising.

Run it:
    uv run python examples/quickstart_mining.py
"""

from langres.core.comparator import StringComparator
from langres.core.models import CompanySchema, ERCandidate
from langres.data import (
    augment_by_attribute,
    denoise_pairs,
    mine_misclassified_pairs,
    sample_negative_pairs,
)
from langres.data.data_profile import profile_mining_readiness

# A tiny labeled dataset: matching companies share their strings; non-matches do
# not. One positive (id "hardL") is deliberately hard -- a true match whose
# strings look nothing alike. One label is deliberately WRONG (id "noisyL") -- a
# non-match mislabeled as a match -- for the denoiser to catch.
comparator = StringComparator.from_schema(CompanySchema)


def candidate(left: CompanySchema, right: CompanySchema) -> ERCandidate[CompanySchema]:
    """Block-then-compare: attach a comparison vector (one Comparator pass)."""
    pair = ERCandidate(left=left, right=right, blocker_name="quickstart")
    return pair.model_copy(update={"comparison": comparator.compare(left, right)})


def company(id: str, name: str, address: str) -> CompanySchema:
    return CompanySchema(id=id, name=name, address=address)


labeled: list[tuple[ERCandidate[CompanySchema], bool]] = []

# 15 easy positives.
for i in range(15):
    labeled.append(
        (
            candidate(
                company(f"m{i}L", f"Acme Corporation {i}", f"{i} Main Street"),
                company(f"m{i}R", f"Acme Corporation {i}", f"{i} Main Street"),
            ),
            True,
        )
    )
# 15 easy negatives.
for i in range(15):
    labeled.append(
        (
            candidate(
                company(f"n{i}L", f"Zephyr Holdings {i}", f"{i} Ocean Avenue"),
                company(f"n{i}R", f"Quasar Industries {i}", f"{i} Mountain Road"),
            ),
            False,
        )
    )
# 1 hard positive: a true match whose strings look like a non-match.
labeled.append(
    (
        candidate(
            company("hardL", "Aardvark Systems", "1 Alpha Way"),
            company("hardR", "Zenith Partners", "9 Omega Blvd"),
        ),
        True,
    )
)
# 1 label error: a genuine non-match mislabeled as a match.
labeled.append(
    (
        candidate(
            company("noisyL", "Willow Foods", "3 Cedar Lane"),
            company("noisyR", "Ironclad Metals", "7 Steel Court"),
        ),
        True,
    )
)

print(
    f"Input: {len(labeled)} labeled pairs "
    f"({sum(1 for _, m in labeled if m)} positive, "
    f"{sum(1 for _, m in labeled if not m)} negative)\n"
)

# 1) Mine AnyMatch-style hard positives (out-of-fold misclassified positives).
hard_positives = mine_misclassified_pairs(labeled, cap=5)
print(
    f"Hard positives mined: {len(hard_positives)} (ids: {[c.left.id for c, _ in hard_positives]})"
)

# 2) Balance the negatives to 2:1 (the AnyMatch lever).
negatives = sample_negative_pairs(labeled, ratio=2.0)
print(f"Negatives sampled at 2:1: {len(negatives)}")

# 3) Attribute augmentation (blank one field at a time; comparison reset to None,
#    so a real training run re-compares the augmented set in one pass).
augmented = augment_by_attribute(labeled, cap=10)
print(f"Attribute-augmented positives: {len(augmented)} (comparison=None -> re-compare before use)")

# 4) Confident-learning denoise: split the set into clean vs likely-mislabeled.
clean, flagged = denoise_pairs(labeled)
print(
    f"Denoise: {len(clean)} clean, {len(flagged)} flagged "
    f"(ids: {[c.left.id for c, _ in flagged]})\n"
)

# Summarise the readiness of the labeled set (a pure consumer of the counts above).
section = profile_mining_readiness(
    n_positive=sum(1 for _, m in labeled if m),
    n_negative=sum(1 for _, m in labeled if not m),
    n_hard_positive=len(hard_positives),
    n_flagged_noise=len(flagged),
)
print(section.to_markdown())

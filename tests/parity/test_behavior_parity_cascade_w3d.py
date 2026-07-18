"""Mixed-family cascade byte-parity net for epic #193 -- the W3-d match-cut guard.

The companion to :mod:`tests.parity.test_behavior_parity_w0` for the *cascade*
path W0 deferred. Where the W0 net covers the ``$0`` string architecture, this
one covers the fragile bit W3-d actually rewires: a **mixed-family** cascade that
emits some rows with a numeric ``sim_cos`` *score* and escalates others to a
``prob_llm`` *decision*, run end-to-end through ``resolve()`` / ``dedupe()`` /
``predict()`` and frozen to a byte-identical golden.

Why this and not a full ``VectorLLMCascade``: the risk W3-d must not regress is
narrow. It moves the match cut out of the :class:`~langres.core.clusterer.Clusterer`
into a ``Select(THRESHOLD)`` and runs the clusterer as pure transitive closure,
on the claim ``ThresholdSelect(t) -> Clusterer(0.0) == Clusterer(t)``. That
equivalence turns entirely on :func:`~langres.core.models.predicted_match` -- and
``predicted_match`` gives a judge's boolean ``decision`` precedence over its
numeric ``score``. The path most likely to break is a mixed stream where the two
disagree, which needs no real embedder or LLM to reproduce -- only a scripted
``sim_cos`` student and a scripted ``prob_llm`` decider (see
:mod:`tests.parity._cascade_fixture`). A faithful offline embedder was the reason
W0 deferred cascade parity; this reframing sidesteps it while guarding the same
seam.

What the golden pins (all offline, deterministic, ``$0`` -- no network, no key):

- **``resolve()`` clusters** -- the connected components the match cut produces.
- **``dedupe()`` ``DedupeResult`` metadata** -- ``score_type`` (the FIRST scored
  row's own family: ``sim_cos``, honestly reporting a mixed stream) + effective
  ``threshold`` + ``architecture`` + ``backbone``.
- **every blocked pair's ``predict()`` judgement** -- ``(score, score_type,
  decision, decision_step)`` sorted canonically, so a change to how *any* pair is
  scored (a lost decision, a mislabelled family, a dropped escalation) is caught
  at the pair level, not only if it survives to alter the final clusters.

Regenerate deliberately (a re-baseline, never the default -- CI always asserts)::

    LANGRES_PARITY_UPDATE=1 uv run pytest tests/parity --no-cov
"""

from __future__ import annotations

import warnings

from tests.parity._cascade_fixture import (
    CASCADE_BAND,
    CASCADE_THRESHOLD,
    EXPECTED_CLUSTERS,
    EXPECTED_UNMERGED_IDS,
    INTERESTING_PAIRS,
    RECORDS,
    build_cascade_model,
)
from tests.parity._golden import canonical_clusters, check_golden


def _run_all() -> tuple[list[dict[str, object]], object, object, object]:
    """Run predict/resolve/dedupe once on one model; return (pairs, judgements, clusters, result).

    The cascade emits a one-time ``UserWarning`` for its ``sim_cos`` student
    (outside the shared probability scale) -- that contract is already covered in
    ``tests/core/modules/test_cascade_judge.py``; here it is expected noise, so we
    silence it rather than let it fail a ``-W error`` run.
    """
    model, _escalation = build_cascade_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        judgements = model.predict(RECORDS)
        clusters = model.resolve(RECORDS)
        result = model.dedupe(RECORDS)
    pairs = [
        {
            "left_id": judgement.left_id,
            "right_id": judgement.right_id,
            "score": judgement.score,
            "score_type": judgement.score_type,
            "decision": judgement.decision,
            "decision_step": judgement.decision_step,
        }
        for judgement in sorted(judgements, key=lambda j: (j.left_id, j.right_id))
    ]
    return pairs, judgements, clusters, result


def test_mixed_family_cascade_parity() -> None:
    """Byte-parity snapshot of the whole mixed-family cascade run.

    One frozen golden over ``predict()`` + ``resolve()`` + ``dedupe()`` on the
    scripted ``sim_cos``/``prob_llm`` cascade -- the tightest net for the W3-d
    match-cut rewire.
    """
    pairs, _judgements, clusters, result = _run_all()
    payload = {
        "threshold": CASCADE_THRESHOLD,
        "band": list(CASCADE_BAND),
        "n_pairs": len(pairs),
        "pairs": pairs,
        "resolve_clusters": canonical_clusters(clusters),
        "dedupe": {
            "architecture": result.architecture,  # type: ignore[attr-defined]
            "backbone": result.backbone,  # type: ignore[attr-defined]
            "score_type": result.score_type,  # type: ignore[attr-defined]
            "threshold": result.threshold,  # type: ignore[attr-defined]
            "clusters": canonical_clusters(result),
        },
    }
    check_golden("mixed_family_cascade_w3d", payload)


def test_decision_precedence_semantics_are_pinned() -> None:
    """Semantic backstop: the decision-vs-score outcomes hold, byte-parity aside.

    A byte golden re-baselines silently under ``LANGRES_PARITY_UPDATE=1``; these
    assertions do not. They pin the exact behavior the W3-d equivalence rests on,
    so a rewrite that swaps ``predicted_match`` for a bare ``score >= t`` cut
    fails here even if someone regenerates the golden.
    """
    _pairs, judgements, clusters, _result = _run_all()
    cluster_sets = {frozenset(c) for c in clusters}

    # The three multi-record clusters, and nothing merged that must not be.
    assert cluster_sets == {frozenset(c) for c in EXPECTED_CLUSTERS}
    merged_ids = {rid for cluster in clusters for rid in cluster}
    assert merged_ids.isdisjoint(EXPECTED_UNMERGED_IDS)

    by_pair = {frozenset({j.left_id, j.right_id}): j for j in judgements}

    # decision wins over a LOW score: score 0.20 < 0.5 threshold, decision=True
    # -> MATCH. A ``score >= t`` cut would DROP this cluster.
    low = by_pair[frozenset(INTERESTING_PAIRS["decision_over_low_score"])]
    assert low.decision is True and low.score is not None and low.score < CASCADE_THRESHOLD
    assert set(INTERESTING_PAIRS["decision_over_low_score"]) in [set(c) for c in clusters]

    # decision suppresses a HIGH score: score 0.90 >= 0.5 threshold, decision=False
    # -> NON-match. A ``score >= t`` cut would WRONGLY merge this pair.
    high = by_pair[frozenset(INTERESTING_PAIRS["decision_over_high_score"])]
    assert high.decision is False and high.score is not None and high.score >= CASCADE_THRESHOLD
    assert not (set(INTERESTING_PAIRS["decision_over_high_score"]) <= merged_ids)

    # abstain (no decision, no score) is EXCLUDED -- never graded a confident "no".
    abstain = by_pair[frozenset(INTERESTING_PAIRS["escalated_abstain"])]
    assert abstain.decision is None and abstain.score is None
    assert abstain.is_abstain


def test_stream_is_genuinely_mixed_family() -> None:
    """The stream really carries both families (not a degenerate single-type run).

    If the whole stream collapsed to one ``score_type`` the golden would still
    pass but stop testing the *mixed*-family path this net exists for -- so pin
    that both ``sim_cos`` (student) and ``prob_llm`` (escalated) rows are present.
    """
    _pairs, judgements, _clusters, result = _run_all()
    score_types = {j.score_type for j in judgements}
    assert {"sim_cos", "prob_llm"} <= score_types
    # dedupe reports the FIRST scored row's own family, honestly (the r01/r02
    # student pair leads the AllPairsBlocker order): a mixed stream is not
    # mislabelled as a single global type.
    assert result.score_type == "sim_cos"  # type: ignore[attr-defined]


def test_escalation_is_lazy_only_in_band_pairs_escalate() -> None:
    """Only the five in-band pairs reach the (expensive) escalation tier.

    Proves the mixed stream is produced the real way -- student everywhere,
    escalation at the margin -- rather than by every pair hitting the decider,
    which would erase the ``sim_cos`` half of the stream.
    """
    model, escalation = build_cascade_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.predict(RECORDS)
    in_band = {
        frozenset(INTERESTING_PAIRS[name])
        for name in (
            "escalated_decision_match",
            "escalated_decision_nonmatch",
            "escalated_abstain",
            "decision_over_low_score",
            "decision_over_high_score",
        )
    }
    assert set(escalation.seen) == in_band
    assert len(escalation.seen) == 5

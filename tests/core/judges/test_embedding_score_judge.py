"""Tests for EmbeddingScoreMatcher (the embedding-cosine scorer Matcher).

The judge is the zero-spend scorer that turns a ``VectorBlocker``'s attached
``similarity_score`` into a ``PairwiseJudgement`` (no Comparator, no LLM). It
must be registry-serializable exactly like ``WeightedAverageMatcher``: register
under ``embedding_score_judge``, round-trip through a Resolver artifact, and load
in a fresh process via the public package alone.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from langres.core.matchers.embedding_score import EmbeddingScoreMatcher
from langres.core.models import CompanySchema, ERCandidate
from langres.core.registry import get_component


def _candidate(similarity: float | None) -> ERCandidate[CompanySchema]:
    """A candidate pair carrying ``similarity`` as its ``similarity_score``."""
    return ERCandidate(
        left=CompanySchema(id="a", name="Acme"),
        right=CompanySchema(id="b", name="Acme Inc"),
        blocker_name="vector",
        similarity_score=similarity,
    )


# ---------------------------------------------------------------------------
# Scoring behaviour
# ---------------------------------------------------------------------------


def test_score_equals_similarity_score() -> None:
    """The emitted judgement's score is exactly the candidate's similarity_score."""
    judge: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher(threshold=0.5)
    [judgement] = list(judge.forward(iter([_candidate(0.83)])))

    assert judgement.left_id == "a"
    assert judgement.right_id == "b"
    assert judgement.score == pytest.approx(0.83)
    assert judgement.score_type == "sim_cos"
    assert judgement.provenance["similarity_score"] == pytest.approx(0.83)
    assert judgement.provenance["threshold"] == pytest.approx(0.5)


def test_decision_step_reflects_threshold() -> None:
    """decision_step records match/no-match against the judge threshold (score unchanged)."""
    judge: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher(threshold=0.7)

    [above] = list(judge.forward(iter([_candidate(0.71)])))
    [at] = list(judge.forward(iter([_candidate(0.70)])))
    [below] = list(judge.forward(iter([_candidate(0.69)])))

    assert above.decision_step == "embedding_match"
    assert at.decision_step == "embedding_match"  # >= threshold is a match
    assert below.decision_step == "embedding_no_match"
    # The score itself is always the raw similarity, never thresholded.
    assert below.score == pytest.approx(0.69)


def test_missing_similarity_raises_with_actionable_message() -> None:
    """A None similarity_score (non-VectorBlocker upstream) raises a clear ValueError."""
    judge: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher()
    with pytest.raises(ValueError, match="VectorBlocker"):
        list(judge.forward(iter([_candidate(None)])))


def test_rejects_out_of_range_threshold() -> None:
    """The threshold is validated to ``[0, 1]`` at construction."""
    with pytest.raises(ValueError, match="threshold"):
        EmbeddingScoreMatcher(threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        EmbeddingScoreMatcher(threshold=-0.1)


def test_inspect_scores_delegates_to_shared_util() -> None:
    """inspect_scores returns a report over the judged pairs (shared Matcher utility)."""
    judge: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher()
    judgements = list(judge.forward(iter([_candidate(0.4), _candidate(0.9)])))
    report = judge.inspect_scores(judgements, sample_size=2)
    assert report.total_judgements == 2


# ---------------------------------------------------------------------------
# Registry serialization (langres DoD #1)
# ---------------------------------------------------------------------------


def test_is_registered_with_type_name() -> None:
    """EmbeddingScoreMatcher is discoverable in the registry under its type_name."""
    assert get_component("embedding_score_judge") is EmbeddingScoreMatcher
    assert EmbeddingScoreMatcher.type_name == "embedding_score_judge"


def test_config_round_trips() -> None:
    """config -> from_config rebuilds an equivalent judge (JSON-serializable)."""
    original: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher(threshold=0.62)
    config = original.config

    assert config == {"threshold": 0.62}
    json.dumps(config)  # must be JSON-serializable

    rebuilt = EmbeddingScoreMatcher.from_config(config)
    assert rebuilt.threshold == pytest.approx(0.62)


def test_resolver_with_embedding_judge_saves_and_loads(tmp_path: Path) -> None:
    """A Resolver with an EmbeddingScoreMatcher in the module slot round-trips offline."""
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker

    judge: EmbeddingScoreMatcher[CompanySchema] = EmbeddingScoreMatcher(threshold=0.55)
    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=judge,
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    manifest = json.loads((tmp_path / "resolver.json").read_text())
    module_spec = next(c for c in manifest["components"] if c["slot"] == "module")
    assert module_spec["type_name"] == "embedding_score_judge"
    assert module_spec["config"]["threshold"] == pytest.approx(0.55)

    reloaded = Resolver.load(tmp_path)
    assert isinstance(reloaded.module, EmbeddingScoreMatcher)
    assert reloaded.module.threshold == pytest.approx(0.55)


@pytest.mark.slow
def test_resolver_load_registers_judge_in_a_fresh_process(tmp_path: Path) -> None:
    """A clean process can ``Resolver.load`` the judge via ``langres.core`` alone.

    Regression for the load-path registration contract: ``@register`` only fires
    when the judge module is imported, so ``langres.core.__init__`` must import it.
    The subprocess imports ONLY ``langres.core`` (the public package), never the
    judge module directly — that is the whole point of the check.
    """
    from langres.core import Clusterer, Resolver
    from langres.core.blockers import AllPairsBlocker

    resolver = Resolver(
        blocker=AllPairsBlocker(schema=CompanySchema),
        comparator=None,
        matcher=EmbeddingScoreMatcher(threshold=0.55),
        clusterer=Clusterer(threshold=0.7),
    )
    resolver.save(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"from langres.core import Resolver; Resolver.load(r'{tmp_path}')",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "fresh-process Resolver.load failed (EmbeddingScoreMatcher not registered on "
        f"the import-langres.core path).\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "UnknownComponentType" not in result.stderr

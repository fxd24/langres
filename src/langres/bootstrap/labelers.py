"""Labelers for cold-start gold-set bootstrapping (M1).

Two concrete labelers, both emitting :class:`~langres.bootstrap.models.GoldPair`:

- :class:`GroundTruthLabeler`: deterministic, zero-spend labeling from a known
  positive-pair set. Used to prove the bootstrap loop end-to-end with no cost.
- :class:`TeacherLabeler`: the real LLM teacher. Wraps an injected
  :class:`~langres.core.modules.llm_judge.LLMJudge` and enforces a hard spend
  cap so a runaway loop can never burn the budget (design-review B1 + B4).
"""

import logging
import math
from typing import Any

from langres.bootstrap.base import Labeler
from langres.bootstrap.models import GoldPair
from langres.core.models import ERCandidate
from langres.core.modules.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


def _pair_key(left_id: str, right_id: str) -> tuple[str, str]:
    """Order-independent identity of a pair (handles (a,b) == (b,a))."""
    return (left_id, right_id) if left_id <= right_id else (right_id, left_id)


class GroundTruthLabeler(Labeler):
    """Deterministic, zero-spend labeler driven by a known positive-pair set.

    Given the gold positive pairs (the matching record pairs), it labels each
    candidate ``True`` iff its ``(left.id, right.id)`` pair is in that set, and
    ``False`` otherwise. No model is called and no budget is spent — this proves
    the bootstrap loop without cost.
    """

    def __init__(self, positive_pairs: set[tuple[str, str]]) -> None:
        """Initialize with the set of order-independent positive (match) pairs.

        Args:
            positive_pairs: Matching pairs as order-independent ``(id, id)``
                tuples. Use :meth:`from_clusters` to derive these from the
                benchmark ``gold_clusters`` contract.
        """
        self._positive_pairs = {_pair_key(a, b) for a, b in positive_pairs}

    @classmethod
    def from_clusters(cls, gold_clusters: list[set[str]]) -> "GroundTruthLabeler":
        """Build from the ``gold_clusters`` contract (cross-source match sets).

        Each cluster contributes the unordered pairs of all its member ids as
        positives. For the 2-element clusters of the Fodors-Zagat adapter this
        is simply the one matching pair per cluster.

        Args:
            gold_clusters: Match sets, e.g. from
                :func:`langres.data.er_benchmarks.load_fodors_zagat`.

        Returns:
            A labeler whose positive set is every intra-cluster pair.
        """
        positives: set[tuple[str, str]] = set()
        for cluster in gold_clusters:
            members = sorted(cluster)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    positives.add(_pair_key(members[i], members[j]))
        return cls(positives)

    def label(self, candidates: list[ERCandidate[Any]]) -> list[GoldPair]:
        """Label each candidate by membership in the positive-pair set.

        Args:
            candidates: Candidate pairs to label.

        Returns:
            One :class:`GoldPair` per candidate, ``source="ground_truth"`` and
            ``confidence=1.0``.
        """
        labels: list[GoldPair] = []
        for cand in candidates:
            left_id: str = cand.left.id
            right_id: str = cand.right.id
            is_match = _pair_key(left_id, right_id) in self._positive_pairs
            labels.append(
                GoldPair(
                    left_id=left_id,
                    right_id=right_id,
                    label=is_match,
                    source="ground_truth",
                    confidence=1.0,
                )
            )
        return labels


class BlindCostError(RuntimeError):
    """Raised when the teacher cannot observe its own spend.

    If a judgement reports neither token counts nor a cost, the running tally
    can no longer be trusted, so the cap is blind. Continuing would risk
    unbounded spend, so :class:`TeacherLabeler` aborts instead.
    """


class TeacherLabeler(Labeler):
    """Budget-capped LLM-teacher labeler (design-review B1 + B4).

    Wraps an injected :class:`LLMJudge` and turns its judgements into
    :class:`GoldPair` labels, while guaranteeing the labeling run can never
    exceed ``budget_usd``. Three layers enforce this:

    1. **Pre-flight hard cap.** Before any call, the input is truncated to
       ``floor(budget_soft_usd / worst_case_per_pair_cost)`` pairs, where the
       per-pair worst case is ``worst_case_tokens_per_pair`` priced at the more
       expensive of the two pinned prices. This bounds spend even if every call
       is maximally expensive, with ``budget_soft_usd`` headroom below the hard
       budget.
    2. **Running tally + budget stop.** Spend is tallied from each judgement's
       actual token counts (priced with the pinned rates, cross-checked against
       the judge's reported ``cost_usd``, taking the max). Before each batch, if
       the projected spend would exceed ``budget_usd`` the run stops and returns
       what was labeled so far — a partial gold set is valid.
    3. **Per-call resilience.** Each pair is judged in its own ``forward`` call
       wrapped in ``try/except``; one failed call skips that pair and the loop
       continues, so a single error never discards an already-paid batch.

    The teacher uses the **synchronous** :meth:`LLMJudge.forward` deliberately:
    an async ``gather`` over a whole batch cannot be stopped mid-flight once
    dispatched, so it cannot honor the budget cap (design-review B1).

    Spend / count attributes are exposed for the run report and are updated live
    (so they remain meaningful even if :class:`BlindCostError` aborts the run):
    :attr:`total_spent_usd`, :attr:`labeled_count`, :attr:`skipped_count`,
    :attr:`dropped_by_cap_count`.
    """

    def __init__(
        self,
        judge: LLMJudge[Any],
        *,
        price_per_1m_prompt_tokens: float,
        price_per_1m_completion_tokens: float,
        worst_case_tokens_per_pair: int,
        budget_usd: float = 20.0,
        budget_soft_usd: float = 15.0,
        batch_size: int = 50,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the teacher.

        Args:
            judge: The wrapped LLM judge (inject for tests; use :meth:`from_env`
                for the happy path).
            price_per_1m_prompt_tokens: Pinned price per 1M prompt tokens (USD).
            price_per_1m_completion_tokens: Pinned price per 1M completion
                tokens (USD).
            worst_case_tokens_per_pair: Worst-case total tokens for one pair,
                used to size the pre-flight cap.
            budget_usd: Hard spend ceiling — the run stops before crossing it.
            budget_soft_usd: Soft ceiling used to size the pre-flight cap, giving
                headroom below ``budget_usd``.
            batch_size: Pairs between budget checks (the budget-stop granularity).
            threshold: ``score >= threshold`` is labeled a match.

        Raises:
            ValueError: If any price/token/budget/batch value is non-positive, if
                ``budget_soft_usd > budget_usd``, or if ``threshold`` is outside
                ``[0, 1]``.
        """
        if price_per_1m_prompt_tokens <= 0.0 or price_per_1m_completion_tokens <= 0.0:
            raise ValueError("pinned token prices must be positive")
        if worst_case_tokens_per_pair <= 0:
            raise ValueError("worst_case_tokens_per_pair must be positive")
        if budget_usd <= 0.0 or budget_soft_usd <= 0.0:
            raise ValueError("budgets must be positive")
        if budget_soft_usd > budget_usd:
            raise ValueError("budget_soft_usd must not exceed budget_usd")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        self._judge = judge
        self.price_per_1m_prompt_tokens = price_per_1m_prompt_tokens
        self.price_per_1m_completion_tokens = price_per_1m_completion_tokens
        self.worst_case_tokens_per_pair = worst_case_tokens_per_pair
        self.budget_usd = budget_usd
        self.budget_soft_usd = budget_soft_usd
        self.batch_size = batch_size
        self.threshold = threshold

        # Worst-case per-pair cost: all tokens at the more expensive rate.
        self._worst_case_per_pair_cost = (
            worst_case_tokens_per_pair
            / 1_000_000.0
            * max(price_per_1m_prompt_tokens, price_per_1m_completion_tokens)
        )

        # Live run statistics (reset at the start of each label() call).
        self.total_spent_usd: float = 0.0
        self.labeled_count: int = 0
        self.skipped_count: int = 0
        self.dropped_by_cap_count: int = 0

    @classmethod
    def from_env(
        cls,
        *,
        price_per_1m_prompt_tokens: float,
        price_per_1m_completion_tokens: float,
        worst_case_tokens_per_pair: int,
        model: str = "gpt-5-mini",
        entity_noun: str = "entity",
        budget_usd: float = 20.0,
        budget_soft_usd: float = 15.0,
        batch_size: int = 50,
        threshold: float = 0.5,
    ) -> "TeacherLabeler":
        """Build a teacher with an LLM judge constructed from the environment.

        The judge's client is built with ``enable_langfuse=False`` so the
        teacher never depends on Langfuse credentials (design-review B4): the
        usual ``LLMJudge.from_env`` enables Langfuse by default, which raises
        without ``LANGFUSE_*`` env vars.

        Args:
            price_per_1m_prompt_tokens: See :meth:`__init__`.
            price_per_1m_completion_tokens: See :meth:`__init__`.
            worst_case_tokens_per_pair: See :meth:`__init__`.
            model: Judge model name.
            entity_noun: Domain noun woven into the judge's prompt.
            budget_usd: See :meth:`__init__`.
            budget_soft_usd: See :meth:`__init__`.
            batch_size: See :meth:`__init__`.
            threshold: See :meth:`__init__`.

        Returns:
            A configured :class:`TeacherLabeler`.
        """
        from langres.clients import Settings, create_llm_client

        client = create_llm_client(Settings(), enable_langfuse=False)
        judge: LLMJudge[Any] = LLMJudge(client=client, model=model, entity_noun=entity_noun)
        return cls(
            judge,
            price_per_1m_prompt_tokens=price_per_1m_prompt_tokens,
            price_per_1m_completion_tokens=price_per_1m_completion_tokens,
            worst_case_tokens_per_pair=worst_case_tokens_per_pair,
            budget_usd=budget_usd,
            budget_soft_usd=budget_soft_usd,
            batch_size=batch_size,
            threshold=threshold,
        )

    def _pair_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost of one judged pair from its token counts and the pinned prices."""
        return (
            prompt_tokens / 1_000_000.0 * self.price_per_1m_prompt_tokens
            + completion_tokens / 1_000_000.0 * self.price_per_1m_completion_tokens
        )

    def label(self, candidates: list[ERCandidate[Any]]) -> list[GoldPair]:
        """Label candidates with the teacher, never exceeding the budget.

        Args:
            candidates: Candidate pairs to label.

        Returns:
            The labeled pairs (``source="teacher"``). May be fewer than the
            input: dropped by the pre-flight cap, skipped on a failed call, or
            truncated when the budget stop fires.

        Raises:
            BlindCostError: If a judgement reports neither token counts nor a
                cost, so spend can no longer be tracked.
        """
        self.total_spent_usd = 0.0
        self.labeled_count = 0
        self.skipped_count = 0
        self.dropped_by_cap_count = 0

        capped = self._apply_preflight_cap(candidates)
        labeled: list[GoldPair] = []

        for start in range(0, len(capped), self.batch_size):
            batch = capped[start : start + self.batch_size]
            projected = self.total_spent_usd + len(batch) * self._worst_case_per_pair_cost
            if projected > self.budget_usd:
                logger.info(
                    "Budget stop: spent=$%.4f + est=$%.4f would exceed budget $%.2f; "
                    "returning %d labeled pairs",
                    self.total_spent_usd,
                    len(batch) * self._worst_case_per_pair_cost,
                    self.budget_usd,
                    len(labeled),
                )
                break
            for candidate in batch:
                gold = self._label_one(candidate)
                if gold is not None:
                    labeled.append(gold)

        return labeled

    def _apply_preflight_cap(
        self, candidates: list[ERCandidate[Any]]
    ) -> list[ERCandidate[Any]]:
        """Truncate the input so even worst-case spend stays under the soft budget."""
        max_pairs = math.floor(self.budget_soft_usd / self._worst_case_per_pair_cost)
        if len(candidates) > max_pairs:
            self.dropped_by_cap_count = len(candidates) - max_pairs
            logger.info(
                "Pre-flight cap: keeping %d of %d pairs (soft budget $%.2f, "
                "worst-case $%.6f/pair)",
                max_pairs,
                len(candidates),
                self.budget_soft_usd,
                self._worst_case_per_pair_cost,
            )
            return candidates[:max_pairs]
        return list(candidates)

    def _label_one(self, candidate: ERCandidate[Any]) -> GoldPair | None:
        """Judge one pair, tally its spend, and map it to a :class:`GoldPair`.

        Returns ``None`` (and increments :attr:`skipped_count`) if the judge
        call fails or yields nothing, so the caller's loop continues.

        Raises:
            BlindCostError: If the judgement reports neither tokens nor cost.
        """
        try:
            judgements = list(self._judge.forward(iter([candidate])))
        except Exception as exc:  # noqa: BLE001 — one bad call must not abort the run
            self.skipped_count += 1
            logger.warning(
                "Teacher call failed for pair %s/%s: %s; skipping",
                candidate.left.id,
                candidate.right.id,
                exc,
            )
            return None

        if not judgements:
            self.skipped_count += 1
            logger.warning(
                "Teacher yielded no judgement for pair %s/%s; skipping",
                candidate.left.id,
                candidate.right.id,
            )
            return None

        judgement = judgements[0]
        prov = judgement.provenance
        prompt_tokens = int(prov.get("prompt_tokens", 0) or 0)
        completion_tokens = int(prov.get("completion_tokens", 0) or 0)
        token_cost = self._pair_cost(prompt_tokens, completion_tokens)
        reported_cost = float(prov.get("cost_usd", 0.0) or 0.0)

        if token_cost == 0.0 and reported_cost == 0.0:
            raise BlindCostError(
                f"Judgement for pair {judgement.left_id}/{judgement.right_id} reported "
                "neither token counts nor cost; cannot track spend"
            )

        spent = max(token_cost, reported_cost)
        self.total_spent_usd += spent
        self.labeled_count += 1

        return GoldPair(
            left_id=judgement.left_id,
            right_id=judgement.right_id,
            label=judgement.score >= self.threshold,
            source="teacher",
            confidence=judgement.score,
            reasoning=judgement.reasoning,
            provenance={
                "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
                "cost_usd": spent,
                "model": prov.get("model"),
            },
        )

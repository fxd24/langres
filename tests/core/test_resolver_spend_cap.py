"""Tests for the Resolver's per-instance spend cap (B1).

Zero-spend throughout: the "paid" matcher is a fake stamping a made-up
``provenance["cost_usd"]``. No network, no LLM, no API key.

Before B1 the Resolver -- the whole low-level public API -- had NO budget guard:
the cap lived in ``core.presets``, which ``Resolver`` is architecturally
forbidden to import, so only ``link``/``dedupe`` were ever capped.
"""

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.core.matcher import Matcher
from langres.core.matchers.weighted_average import WeightedAverageMatcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.resolver import Resolver
from langres.core.spend import BudgetExceeded
from langres.core.spend_cap import DEFAULT_BUDGET_USD, UNCAPPED_BUDGET_USD


class CapCo(BaseModel):
    id: str
    name: str | None = None


#: Two records -> AllPairsBlocker yields exactly ONE candidate pair, so one
#: resolve() == one paid judgement. That makes the spend arithmetic below
#: readable: N resolves == N x cost_each.
RECORDS = [{"id": "1", "name": "Acme Corp"}, {"id": "2", "name": "Acme Corporation"}]


class _CountingCostlyMatcher(Matcher[object]):
    """Scores every candidate at a fixed cost, counting the paid calls it made."""

    def __init__(self, cost_each: float) -> None:
        self._cost_each = cost_each
        self.produced = 0

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        for candidate in candidates:
            self.produced += 1
            yield PairwiseJudgement(
                left_id=str(candidate.left.id),
                right_id=str(candidate.right.id),
                score=0.9,
                score_type="prob_llm",
                decision_step="fake",
                provenance={"cost_usd": self._cost_each},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


def _resolver(matcher: Matcher[object], *, budget_usd: float | None = None) -> Resolver:
    return Resolver(
        blocker=AllPairsBlocker(schema=CapCo),
        comparator=None,
        matcher=matcher,
        clusterer=Clusterer(threshold=0.5),
        budget_usd=budget_usd,
    )


class TestResolverCapIsPerInstance:
    """THE B1 proof. A verb call is one-shot; a Resolver instance is not."""

    def test_two_resolve_calls_share_one_budget(self) -> None:
        """Pre-B1 this spent 2 x budget: each resolve() got a fresh monitor."""
        matcher = _CountingCostlyMatcher(cost_each=0.5)
        resolver = _resolver(matcher, budget_usd=0.9)

        resolver.resolve(RECORDS)  # spent 0.50 <= 0.90 -> fine
        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)  # spent 1.00 > 0.90 -> refused

    def test_a_tripped_resolver_costs_nothing_on_the_next_resolve(self) -> None:
        """Once over budget, further resolve() calls must pay for NOTHING."""
        matcher = _CountingCostlyMatcher(cost_each=2.0)
        resolver = _resolver(matcher, budget_usd=0.9)

        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)
        assert matcher.produced == 1

        for _ in range(3):
            with pytest.raises(BudgetExceeded):
                resolver.resolve(RECORDS)
        assert matcher.produced == 1, "a tripped Resolver kept paying for calls"

    def test_predict_shares_the_same_ledger_as_resolve(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=0.5)
        resolver = _resolver(matcher, budget_usd=0.9)

        resolver.predict(RECORDS)
        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)

    def test_separate_resolvers_do_not_share_a_budget(self) -> None:
        """The ledger is per instance -- not global, not per class."""
        first = _resolver(_CountingCostlyMatcher(cost_each=0.5), budget_usd=0.9)
        second = _resolver(_CountingCostlyMatcher(cost_each=0.5), budget_usd=0.9)

        first.resolve(RECORDS)
        second.resolve(RECORDS)  # must NOT inherit first's spend

    def test_budget_survives_module_reassignment(self) -> None:
        """dedupe() reassigns resolver.module (LoggingMatcher); distil() replaces
        it outright. The cap must meter whatever is in the slot NOW, off the
        same ledger -- the monitor, not the wrapper, is the durable thing."""
        resolver = _resolver(_CountingCostlyMatcher(cost_each=0.5), budget_usd=0.9)
        resolver.resolve(RECORDS)  # spent 0.50

        resolver.module = _CountingCostlyMatcher(cost_each=0.5)  # swap the scorer
        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)  # spent 1.00 > 0.90 -- ledger carried over


class TestResolverBudgetEdges:
    def test_none_budget_means_the_default_not_uncapped(self) -> None:
        """The whole point of B1: forgetting budget_usd must not mean 'unlimited'."""
        resolver = _resolver(_CountingCostlyMatcher(cost_each=0.1))
        assert resolver._spend_monitor.budget_usd == DEFAULT_BUDGET_USD

    def test_default_budget_actually_binds(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=DEFAULT_BUDGET_USD + 0.01)
        resolver = _resolver(matcher)
        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)

    def test_exactly_at_budget_does_not_trip(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=0.5)
        resolver = _resolver(matcher, budget_usd=1.0)
        resolver.resolve(RECORDS)
        resolver.resolve(RECORDS)  # spent == 1.00 exactly -> still fine
        assert resolver._spend_monitor.spent == 1.0

    def test_zero_budget_refuses_the_first_paid_call(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=0.01)
        resolver = _resolver(matcher, budget_usd=0.0)
        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)

    def test_zero_budget_still_runs_a_free_matcher(self) -> None:
        """A $0 budget bans SPEND, not work: the string matcher meters nothing."""
        resolver = Resolver.from_schema(CapCo, matcher="string", budget_usd=0.0)
        assert resolver.resolve(RECORDS) == [{"1", "2"}]

    def test_uncapped_budget_never_trips(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=1_000.0)
        resolver = _resolver(matcher, budget_usd=UNCAPPED_BUDGET_USD)
        for _ in range(3):
            resolver.resolve(RECORDS)
        assert resolver._spend_monitor.spent == 3_000.0

    def test_budget_exceeded_carries_the_paid_judgements(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=2.0)
        resolver = _resolver(matcher, budget_usd=0.9)
        with pytest.raises(BudgetExceeded) as excinfo:
            resolver.resolve(RECORDS)
        assert len(excinfo.value.partial_judgements) == 1


class TestFromSchemaThreadsBudget:
    def test_from_schema_threads_budget_usd(self) -> None:
        resolver = Resolver.from_schema(CapCo, matcher="string", budget_usd=4.2)
        assert resolver._spend_monitor.budget_usd == 4.2

    def test_from_schema_defaults_to_the_capped_default(self) -> None:
        resolver = Resolver.from_schema(CapCo, matcher="string")
        assert resolver._spend_monitor.budget_usd == DEFAULT_BUDGET_USD


class TestCapDoesNotDisturbTheMatcherSlot:
    """The cap wraps at SCORING time, never in the slot: `self.module` stays the
    raw matcher so fit()'s isinstance checks, save()'s registry lookup and
    `.model`/`.config` access all keep seeing the real component."""

    def test_module_slot_holds_the_raw_matcher_not_a_wrapper(self) -> None:
        matcher = _CountingCostlyMatcher(cost_each=0.0)
        assert _resolver(matcher).module is matcher

    def test_from_schema_module_slot_is_the_real_component(self) -> None:
        resolver = Resolver.from_schema(CapCo, matcher="string")
        assert isinstance(resolver.module, WeightedAverageMatcher)

    def test_capped_resolver_still_round_trips_through_save_load(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """A spend-cap wrapper in the slot would have no `type_name` and break
        serialization -- regression guard for keeping it out of the slot."""
        resolver = Resolver.from_schema(CapCo, matcher="string", budget_usd=2.0)
        resolver.save(tmp_path / "artifact")
        loaded = Resolver.load(tmp_path / "artifact")
        assert loaded.resolve(RECORDS) == [{"1", "2"}]

    def test_loaded_resolver_is_capped_at_the_default(self) -> None:
        """budget_usd is a runtime knob, not pipeline topology: it is not
        serialized, and a loaded Resolver comes back capped at the DEFAULT
        rather than uncapped."""
        resolver = Resolver.from_schema(CapCo, matcher="string", budget_usd=2.0)
        assert resolver._spend_monitor.budget_usd == 2.0

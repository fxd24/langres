"""Tests for the Resolver's per-instance spend cap (B1).

Zero-spend throughout: the "paid" matcher is a fake stamping a made-up
``provenance["cost_usd"]``. No network, no LLM, no API key.

Before B1 the Resolver -- the whole low-level public API -- had NO budget guard:
the cap lived in ``core.presets``, which ``Resolver`` is architecturally
forbidden to import, so only ``link``/``dedupe`` were ever capped.
"""

import ast
import pathlib
from collections.abc import Iterator

import pytest
from pydantic import BaseModel

import langres
from langres.curation.anchor_store import AnchorStore
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.clusterer import Clusterer
from langres.tracking.judgement_log import JudgementLog, LoggingMatcher
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


class TestCapComposesWithTheLoggingMatcher:
    """dedupe() reassigns `resolver.module = LoggingMatcher(judge)`, so B1 flips
    that path's composition from LoggingMatcher(Cap(judge)) to
    Cap(LoggingMatcher(judge)). The flywheel needs EVERY paid judgement logged --
    especially the one that trips the cap, which pre-B1 was never yielded and so
    had to be recovered from BudgetExceeded.partial_judgements by hand."""

    def test_the_tripping_judgement_is_still_logged_exactly_once(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        log = JudgementLog(tmp_path / "log.jsonl")
        # 4 records -> 6 pairs at $0.6 each: $0.60 ok, $1.20 trips the $1.00 cap.
        records = [{"id": str(i), "name": f"Acme {i}"} for i in range(4)]
        resolver = _resolver(_CountingCostlyMatcher(cost_each=0.6), budget_usd=1.0)
        resolver.module = LoggingMatcher(
            resolver.module, log=log, threshold=0.5, model="fake/model"
        )

        with pytest.raises(BudgetExceeded) as excinfo:
            resolver.predict(records)

        rows = log.read()
        logged = [(r["left_id"], r["right_id"]) for r in rows]
        partial = [(j.left_id, j.right_id) for j in excinfo.value.partial_judgements]
        assert logged == partial, "a paid judgement was logged but not reported (or vice versa)"
        assert len(logged) == 2, f"expected the 2 paid judgements, got {logged}"
        assert len(set(logged)) == len(logged), "a judgement was logged twice"

    def test_spend_overshoots_by_at_most_the_tripping_call(self) -> None:
        """The honest guarantee: budget + at most ONE further call. Not less --
        the tripping call's cost is only knowable once it has been paid for."""
        resolver = _resolver(_CountingCostlyMatcher(cost_each=0.6), budget_usd=1.0)
        records = [{"id": str(i), "name": f"Acme {i}"} for i in range(4)]
        with pytest.raises(BudgetExceeded):
            resolver.predict(records)
        assert resolver._spend_monitor.spent == pytest.approx(1.2)
        assert resolver._spend_monitor.spent <= 1.0 + 0.6


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


class TestEverySeamSharesTheOneLedger:
    """The hole this class exists to keep shut.

    ``.module`` is a PUBLIC attribute holding the RAW matcher (the slot must stay
    unwrapped for ``save()``/``fit()`` -- see the class above). So any collaborator
    can reach through it and score with no cap and no ledger, which is exactly
    what ``AnchorStore._judge`` did: ``self._resolver.module.forward(...)``. A
    long-lived store calling ``assign`` per arriving record is the *worst* case
    for an unbounded paid matcher.
    """

    def test_assign_bills_against_the_resolvers_ledger(self) -> None:
        """THE proof: an AnchorStore pass moves the SAME meter resolve() moves.

        Mutation test -- restore ``self._resolver.module.forward(...)`` in
        ``AnchorStore._judge`` and this fails: ``spent`` never moves past what
        build()'s resolve() paid, because the raw matcher reports to nobody.
        """
        matcher = _CountingCostlyMatcher(cost_each=0.10)
        resolver = _resolver(matcher, budget_usd=10.0)
        store = AnchorStore.build(resolver, RECORDS)

        spent_after_build = resolver._spend_monitor.spent
        assert spent_after_build > 0.0, "build()'s resolve() should already have billed"

        store.assign({"id": "3", "name": "Acme Corp"})

        assert resolver._spend_monitor.spent > spent_after_build
        # The ledger agrees with the matcher's own call count: nothing was
        # scored off-ledger.
        assert resolver._spend_monitor.spent == pytest.approx(0.10 * matcher.produced)

    def test_a_tripped_ledger_makes_assign_cost_nothing(self) -> None:
        """Money safety, end to end: once the budget is gone, assign() stops
        paying. Pre-cap, every assign() on a spent Resolver kept billing."""
        matcher = _CountingCostlyMatcher(cost_each=0.60)
        resolver = _resolver(matcher, budget_usd=1.0)
        store = AnchorStore.build(resolver, RECORDS)  # resolve(): 1 pair, $0.60

        with pytest.raises(BudgetExceeded):
            store.assign({"id": "3", "name": "Acme Corp"})

        paid_so_far = matcher.produced
        with pytest.raises(BudgetExceeded):
            store.assign({"id": "4", "name": "Acme Corp"})
        assert matcher.produced == paid_so_far, "a tripped cap must cost $0, not one more call"

    def test_assign_and_resolve_cannot_each_spend_a_full_budget(self) -> None:
        """One Resolver == ONE budget, whichever seam does the spending."""
        matcher = _CountingCostlyMatcher(cost_each=0.60)
        resolver = _resolver(matcher, budget_usd=1.0)
        store = AnchorStore.build(resolver, RECORDS)

        with pytest.raises(BudgetExceeded):
            store.assign({"id": "3", "name": "Acme Corp"})
        # The store drained the shared ledger, so the Resolver's OWN scoring
        # path is now refused too -- one meter, not two.
        with pytest.raises(BudgetExceeded):
            resolver.resolve(RECORDS)

    def test_no_src_site_scores_through_a_raw_module_slot(self) -> None:
        """The structural guard: ban ``<anything>.module.forward(...)`` in src/.

        The bug was never "AnchorStore is wrong" -- it was that the capped scorer
        was built inline at ONE call site, so every other caller silently got the
        raw matcher. ``Resolver._scorer()`` is now the single seam; this sweep is
        what stops the next caller from re-opening the hole. Parsed, not grepped:
        a docstring saying "module.forward" is not a call.
        """
        src_root = pathlib.Path(langres.__file__).parent
        offenders = []
        for path in sorted(src_root.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "forward"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "module"
                ):
                    offenders.append(f"{path}:{node.lineno}: {ast.unparse(node.func)}(...)")

        assert offenders == [], (
            "these sites score through the RAW matcher slot -- no spend cap, no "
            "ledger, unbounded bill. Route them through Resolver._scorer():\n  "
            + "\n  ".join(offenders)
        )

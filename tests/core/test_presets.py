"""Tests for langres.core.presets (judge="auto" resolution + spend-capped Resolver).

Zero-spend throughout: real network/LLM calls are never made. The
"zero_shot_llm" branch is only ever exercised up to *construction* (verifying
the DSPyJudge instance's model/price), never ``.forward()`` -- any pairwise
scoring test injects a DummyLM-backed ``DSPyJudge`` or a tiny fake Module
instead. "embedding" tests that need a real ``.encode()`` call load the local
MiniLM model (no API key, no paid call) and are marked ``@pytest.mark.slow``.
"""

import re
import sys
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from dspy.utils.dummies import DummyLM
from pydantic import BaseModel

from langres.clients.openrouter import PRICES_PER_1M, BudgetExceeded
from langres.clients.settings import Settings
from langres.core.blockers.all_pairs import AllPairsBlocker
from langres.core.blockers.vector import VectorBlocker
from langres.core.judges.embedding_score import EmbeddingScoreJudge
from langres.core.judges.weighted_average import WeightedAverageJudge
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.module import Module, stamp_group_cost
from langres.core.modules.dspy_judge import DSPyJudge
from langres.core.presets import (
    DEFAULT_BUDGET_USD,
    _ALL_PAIRS_MAX_N,
    NoJudgeAvailableError,
    _build_vector_blocker,
    _estimate_n_pairs,
    _OPENAI_MODEL,
    _OPENROUTER_MODEL,
    _SpendCappedModule,
    _text_field_extractor,
    build_embedding_candidate,
    build_judge,
    build_resolver,
    choose_auto_judge,
    notice_pre_scoring_cost,
    resolve_judge,
)


class PresetCompany(BaseModel):
    id: str
    name: str | None = None
    address: str | None = None


class PresetProduct(BaseModel):
    id: str
    title: str | None = None
    brand: str | None = None


def _settings(*, openrouter: str | None = None, openai: str | None = None) -> Settings:
    return Settings(openrouter_api_key=openrouter, openai_api_key=openai)


class _FakeCostlyModule(Module[object]):
    """Yields N judgements each costing a fixed amount -- for cap-breach tests."""

    def __init__(self, n: int, cost_each: float) -> None:
        self._n = n
        self._cost_each = cost_each

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)  # drain (content unused)
        for i in range(self._n):
            yield PairwiseJudgement(
                left_id=str(i),
                right_id=str(i + 1),
                score=0.9,
                score_type="prob_llm",
                decision_step="fake",
                provenance={"cost_usd": self._cost_each},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


def _raw_group_judgements(n: int, group_id: str) -> list[PairwiseJudgement]:
    """Unstamped judgements for one group -- ``stamp_group_cost`` applies the
    E5 cost/group_id/group_end convention on top (real production path)."""
    return [
        PairwiseJudgement(
            left_id="anchor",
            right_id=f"{group_id}-{i}",
            score=0.9,
            score_type="prob_llm",
            decision_step="fake_group",
            provenance={},
        )
        for i in range(n)
    ]


class _FakeGroupModule(Module[object]):
    """Yields one group's judgements per the E5 group-cost convention.

    Full ``call_cost_usd`` on the first judgement, ``$0`` (plus
    ``group_end=True`` on the last) on the ``n_siblings`` remaining ones, all
    sharing ``provenance["group_id"]`` -- built via the real
    :func:`~langres.core.module.stamp_group_cost`, the same helper
    ``SelectJudge`` uses for one LLM call spanning a whole group.
    """

    def __init__(self, first_cost: float, n_siblings: int, group_id: str = "g1") -> None:
        self._first_cost = first_cost
        self._n_siblings = n_siblings
        self._group_id = group_id

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)  # drain (content unused)
        raw = _raw_group_judgements(self._n_siblings + 1, self._group_id)
        yield from stamp_group_cost(raw, call_cost_usd=self._first_cost, group_id=self._group_id)

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class _LazyGroupsModule(Module[object]):
    """Concatenates several groups' judgement streams, tracking when each
    group's (paid) computation actually STARTS -- mirroring
    ``GroupwiseModule.forward_groups``'s real contract: one paid LLM call per
    group, fired lazily only when the generator is advanced into that group,
    before it can yield anything to compare against (the exact shape of the
    #68-review bug: pulling one item past an already-drained group's last
    judgement to check its group_id resumes computation of -- and pays for --
    the NEXT group before the boundary can be detected).
    """

    def __init__(self, groups: list[tuple[float, int, str]]) -> None:
        # each item: (first_cost, n_siblings, group_id)
        self._groups = groups
        self.groups_computed = 0

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)  # drain (content unused)
        for first_cost, n_siblings, group_id in self._groups:
            self.groups_computed += 1  # the "paid call" fires here, lazily
            raw = _raw_group_judgements(n_siblings + 1, group_id)
            yield from stamp_group_cost(raw, call_cost_usd=first_cost, group_id=group_id)

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class _FakeMalformedGroupModule(Module[object]):
    """A group-wise module that violates the E5 convention: never stamps
    ``group_end`` at all. Exercises the drain loop's defensive fallback --
    if the boundary marker never appears, draining runs to the end of the
    (in this fake, finite) stream instead of assuming a sibling exists
    forever or crashing.
    """

    def __init__(self, first_cost: float, n_siblings: int, group_id: str = "g1") -> None:
        self._first_cost = first_cost
        self._n_siblings = n_siblings
        self._group_id = group_id

    def forward(self, candidates: Iterator[ERCandidate[object]]) -> Iterator[PairwiseJudgement]:
        list(candidates)  # drain (content unused)
        for i in range(self._n_siblings + 1):
            cost = self._first_cost if i == 0 else 0.0
            yield PairwiseJudgement(
                left_id="anchor",
                right_id=str(i),
                score=0.9,
                score_type="prob_llm",
                decision_step="fake_group_malformed",
                provenance={"cost_usd": cost, "group_id": self._group_id},
            )

    def inspect_scores(self, judgements: list[PairwiseJudgement], sample_size: int = 10) -> object:
        raise NotImplementedError


class _FakeVectorIndex:
    """No-op stand-in for a real FAISS index -- create_index does nothing."""

    def create_index(self, texts: list[str]) -> None:
        pass


class _FakeEmptyStreamBlocker:
    """A fake VectorBlocker whose stream() yields no candidates (L2 test seam).

    Exercises build_embedding_candidate's defensive "no candidate" guard
    without needing a real, genuinely-broken VectorBlocker -- that case
    "should never happen" for a real 2-record stream, so it's only reachable
    by substituting a blocker built this way.
    """

    vector_index = _FakeVectorIndex()

    @staticmethod
    def schema_factory(record: dict[str, Any]) -> dict[str, Any]:
        return record

    @staticmethod
    def text_field_extractor(entity: dict[str, Any]) -> str:
        return str(entity)

    @staticmethod
    def stream(records: list[dict[str, Any]]) -> Iterator[ERCandidate[Any]]:
        return iter(())


# ---------------------------------------------------------------------------
# choose_auto_judge
# ---------------------------------------------------------------------------


class TestChooseAutoJudge:
    def test_no_keys_raises_with_actionable_copy(self) -> None:
        """The error IS the keyless persona's landing page: what / why the
        library refuses / fix A (key + install line) / fix B (explicit
        offline opt-in + caveat) / default-cap reassurance / guide URL."""
        with pytest.raises(NoJudgeAvailableError) as excinfo:
            choose_auto_judge(_settings())
        message = str(excinfo.value)
        assert "no API key" in message
        assert "OPENROUTER_API_KEY" in message and "OPENAI_API_KEY" in message
        assert "over-merges" in message  # why the library refuses to fall back
        assert "uv sync --extra llm" in message
        assert "pip install 'langres[llm]'" in message
        assert 'judge="string"' in message and "derive_threshold" in message
        assert "$1.00" in message  # default spend-cap reassurance
        assert "docs/GETTING_STARTED.md" in message
        assert len(message.splitlines()) <= 5

    def test_openrouter_key_selects_llm_with_selection_notice(self) -> None:
        with pytest.warns(UserWarning, match="PAID") as record:
            judge, model = choose_auto_judge(_settings(openrouter="or-key"))
        assert judge == "zero_shot_llm"
        assert model == _OPENROUTER_MODEL
        assert PRICES_PER_1M[model][0] > 0.0 and PRICES_PER_1M[model][1] > 0.0
        message = str(record[0].message)
        assert model in message  # which model was picked
        assert f"${DEFAULT_BUDGET_USD:.2f}" in message  # the cap

    def test_openai_key_used_only_when_no_openrouter_key(self) -> None:
        with pytest.warns(UserWarning, match="PAID"):
            judge, model = choose_auto_judge(_settings(openai="oai-key"))
        assert judge == "zero_shot_llm"
        assert model == _OPENAI_MODEL
        assert PRICES_PER_1M[model][0] > 0.0 and PRICES_PER_1M[model][1] > 0.0

    def test_openrouter_key_preferred_over_openai_key(self) -> None:
        with pytest.warns(UserWarning):
            judge, model = choose_auto_judge(_settings(openrouter="or-key", openai="oai-key"))
        assert (judge, model) == ("zero_shot_llm", _OPENROUTER_MODEL)

    def test_refuses_paid_judge_with_unpinned_price(self) -> None:
        """E1/TD1: a model with no pinned price raises -- a blind $0 cap is no
        cap at all; explicit judge="zero_shot_llm" stays the escape hatch."""
        unpriced = {k: v for k, v in PRICES_PER_1M.items() if k != _OPENROUTER_MODEL}
        with (
            patch.dict("langres.clients.openrouter.PRICES_PER_1M", unpriced, clear=True),
            pytest.raises(NoJudgeAvailableError, match="no pinned price") as excinfo,
        ):
            choose_auto_judge(_settings(openrouter="or-key"))
        message = str(excinfo.value)
        assert _OPENROUTER_MODEL in message
        assert 'judge="zero_shot_llm"' in message  # the blind-cap escape hatch

    def test_caller_model_override_is_honored_and_named_in_notice(self) -> None:
        with pytest.warns(UserWarning, match=re.escape(_OPENAI_MODEL)):
            judge, model = choose_auto_judge(_settings(openrouter="or-key"), model=_OPENAI_MODEL)
        assert (judge, model) == ("zero_shot_llm", _OPENAI_MODEL)

    def test_caller_model_override_runs_the_pinned_price_check(self) -> None:
        with pytest.raises(NoJudgeAvailableError, match="no pinned price"):
            choose_auto_judge(_settings(openrouter="or-key"), model="unknown/model-not-in-table")

    def test_no_keys_raises_even_with_model_override(self) -> None:
        with pytest.raises(NoJudgeAvailableError, match="no API key"):
            choose_auto_judge(_settings(), model=_OPENROUTER_MODEL)

    def test_budget_usd_is_named_in_the_selection_notice(self) -> None:
        with pytest.warns(UserWarning, match=r"\$2\.50"):
            choose_auto_judge(_settings(openrouter="or-key"), budget_usd=2.5)

    def test_offline_flag_raises_even_with_keys_present(self) -> None:
        """LANGRES_OFFLINE beats every discoverable key: the deterministic
        keyless switch must win even when both keys are set, and its error
        copy must name the flag and the offline opt-in fix."""
        settings = Settings(
            langres_offline=True, openrouter_api_key="or-key", openai_api_key="oai-key"
        )
        with pytest.raises(NoJudgeAvailableError, match="LANGRES_OFFLINE") as excinfo:
            choose_auto_judge(settings)
        message = str(excinfo.value)
        assert 'judge="string"' in message  # the offline opt-in fix
        assert "docs/GETTING_STARTED.md" in message


def warnings_none() -> "_NoWarnings":
    """Assert the wrapped block emits no warnings at all."""
    return _NoWarnings()


class _NoWarnings:
    def __enter__(self) -> "_NoWarnings":
        import warnings as _w

        self._catch = _w.catch_warnings(record=True)
        self._records = self._catch.__enter__()
        _w.simplefilter("always")
        return self

    def __exit__(self, *exc: object) -> None:
        assert not self._records, f"unexpected warnings: {[str(r.message) for r in self._records]}"
        self._catch.__exit__(*exc)


# ---------------------------------------------------------------------------
# build_judge
# ---------------------------------------------------------------------------


class TestBuildJudge:
    def test_string_returns_weighted_average_judge_with_schema_features(self) -> None:
        module = build_judge("string", PresetCompany)
        assert isinstance(module, WeightedAverageJudge)
        names = {spec.name for spec in module.feature_specs}
        assert names == {"name", "address"}

    def test_string_is_schema_agnostic_across_two_different_schemas(self) -> None:
        company_module = build_judge("string", PresetCompany)
        product_module = build_judge("string", PresetProduct)
        assert {s.name for s in company_module.feature_specs} == {"name", "address"}
        assert {s.name for s in product_module.feature_specs} == {"title", "brand"}

    def test_embedding_returns_embedding_score_judge(self) -> None:
        assert isinstance(build_judge("embedding", PresetCompany), EmbeddingScoreJudge)

    def test_zero_shot_llm_returns_dspy_judge_with_default_model_and_pinned_price(self) -> None:
        module = build_judge("zero_shot_llm", PresetCompany, entity_noun="company")
        assert isinstance(module, DSPyJudge)
        assert module.model == _OPENROUTER_MODEL
        assert module.entity_noun == "company"
        assert module.price_per_1k_tokens > 0.0

    def test_zero_shot_llm_respects_model_override(self) -> None:
        module = build_judge("zero_shot_llm", PresetCompany, model=_OPENAI_MODEL)
        assert isinstance(module, DSPyJudge)
        assert module.model == _OPENAI_MODEL
        assert module.price_per_1k_tokens > 0.0

    def test_module_instance_passed_through_verbatim(self) -> None:
        injected: DSPyJudge[PresetCompany] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        assert build_judge(injected, PresetCompany) is injected

    def test_unknown_judge_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown judge"):
            build_judge("not-a-real-judge", PresetCompany)  # type: ignore[arg-type]

    def test_auto_is_not_resolved_here(self) -> None:
        with pytest.raises(ValueError, match="unknown judge"):
            build_judge("auto", PresetCompany)

    def test_zero_shot_llm_without_llm_extra_raises_with_install_line(self) -> None:
        """The keyed path must not dead-end in a raw ModuleNotFoundError: dspy
        lives in the [llm] extra but is imported at dspy_judge.py module
        level, so build_judge re-raises with the exact install guidance."""
        with (
            patch.dict(sys.modules, {"langres.core.modules.dspy_judge": None}),
            pytest.raises(ImportError, match=r"uv sync --extra llm") as excinfo,
        ):
            build_judge("zero_shot_llm", PresetCompany)
        assert "pip install 'langres[llm]'" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _SpendCappedModule / resolve_judge
# ---------------------------------------------------------------------------


class TestSpendCappedModule:
    def test_zero_cost_judgements_never_trip_the_cap(self) -> None:
        module = _SpendCappedModule(_FakeCostlyModule(5, 0.0), budget_usd=0.01)
        candidates = iter([_candidate(str(i)) for i in range(5)])
        judgements = list(module.forward(candidates))
        assert len(judgements) == 5

    def test_cap_breach_raises_budget_exceeded_with_partial_judgements(self) -> None:
        module = _SpendCappedModule(_FakeCostlyModule(5, 0.5), budget_usd=0.9)
        candidates = iter([_candidate(str(i)) for i in range(5)])
        with pytest.raises(BudgetExceeded) as excinfo:
            list(module.forward(candidates))
        partial = excinfo.value.partial_judgements
        # 0.5 -> ok, 1.0 -> breaches $0.9 cap: exactly 2 judgements were paid for.
        assert len(partial) == 2
        assert all(isinstance(j, PairwiseJudgement) for j in partial)

    def test_inspect_scores_delegates_to_wrapped_module(self) -> None:
        inner = build_judge("string", PresetCompany)
        module = _SpendCappedModule(inner, budget_usd=1.0)
        report = module.inspect_scores([])
        assert report is not None

    def test_cap_breach_drains_group_siblings_into_partial_judgements(self) -> None:
        """A group-wise module's tripping judgement must not split its group (#68 review).

        ``SelectJudge``-style modules stamp the FULL call cost onto the first
        judgement of a group and $0 onto its K-1 siblings, all sharing
        ``provenance["group_id"]`` (E5). If the cap trips on that first
        judgement, the already-paid-for siblings must still be drained into
        ``partial_judgements`` -- a group must never be split across the cap
        boundary.
        """
        module = _SpendCappedModule(
            _FakeGroupModule(first_cost=1.0, n_siblings=2, group_id="g1"), budget_usd=0.9
        )
        candidates = iter([_candidate(str(i)) for i in range(3)])
        with pytest.raises(BudgetExceeded) as excinfo:
            list(module.forward(candidates))
        partial = excinfo.value.partial_judgements
        assert len(partial) == 3
        assert all(j.provenance.get("group_id") == "g1" for j in partial)
        # No cost was double-counted for the $0 siblings.
        assert sum(float(j.provenance["cost_usd"]) for j in partial) == 1.0

    def test_cap_breach_drain_never_computes_the_next_group(self) -> None:
        """The drain must stop at group_end, never peek into (and pay for) the NEXT group.

        Regression for the claude-review finding on this PR: detecting the
        group boundary by peeking at the next judgement's group_id (instead
        of the group_end marker) resumes a lazy GroupwiseModule's generator
        one group too far -- for a real module (SelectJudge) that fires the
        next group's paid LLM call before there's anything to compare
        against, silently discarding it. ``_LazyGroupsModule.groups_computed``
        makes that "was the next group's paid call fired at all" question
        observable: it must stay at 1 (only "g1" fires) after the drain,
        never 2 (which would mean g2's call fired and its result was thrown
        away).
        """
        fake = _LazyGroupsModule([(1.0, 1, "g1"), (0.0, 1, "g2")])
        module = _SpendCappedModule(fake, budget_usd=0.9)
        candidates = iter([_candidate(str(i)) for i in range(4)])
        with pytest.raises(BudgetExceeded) as excinfo:
            list(module.forward(candidates))
        partial = excinfo.value.partial_judgements
        assert len(partial) == 2
        assert all(j.provenance.get("group_id") == "g1" for j in partial)
        assert fake.groups_computed == 1, (
            "g2's paid call fired even though the cap tripped on g1 -- "
            f"got {fake.groups_computed} groups computed, want 1"
        )

    def test_cap_breach_drain_runs_to_stream_end_if_group_end_never_set(self) -> None:
        """Defensive fallback: a module that never stamps group_end (a convention
        violation) still drains to the end of its stream rather than looping
        forever or crashing -- exercises the drain loop's non-break exit path.
        """
        module = _SpendCappedModule(
            _FakeMalformedGroupModule(first_cost=1.0, n_siblings=2, group_id="g1"), budget_usd=0.9
        )
        candidates = iter([_candidate(str(i)) for i in range(3)])
        with pytest.raises(BudgetExceeded) as excinfo:
            list(module.forward(candidates))
        partial = excinfo.value.partial_judgements
        assert len(partial) == 3

    def test_cap_breach_without_group_id_is_unchanged(self) -> None:
        """Pairwise (no group_id) modules keep the pre-existing single-judgement cap behavior."""
        module = _SpendCappedModule(_FakeCostlyModule(5, 0.5), budget_usd=0.9)
        candidates = iter([_candidate(str(i)) for i in range(5)])
        with pytest.raises(BudgetExceeded) as excinfo:
            list(module.forward(candidates))
        partial = excinfo.value.partial_judgements
        # 0.5 -> ok, 1.0 -> breaches $0.9 cap: exactly 2 judgements were paid for,
        # and none carry a group_id to drain siblings for.
        assert len(partial) == 2
        assert all(j.provenance.get("group_id") is None for j in partial)


def _candidate(suffix: str) -> ERCandidate[PresetCompany]:
    return ERCandidate(
        left=PresetCompany(id=f"l{suffix}", name="A"),
        right=PresetCompany(id=f"r{suffix}", name="B"),
        blocker_name="test",
    )


class TestResolveJudge:
    def test_string_judge_used_and_default_budget(self) -> None:
        resolved = resolve_judge("string", PresetCompany)
        assert resolved.judge_used == "string"
        assert resolved.model is None
        assert isinstance(resolved.module, _SpendCappedModule)
        assert resolved.module._budget_usd == DEFAULT_BUDGET_USD

    def test_custom_budget_usd_override(self) -> None:
        resolved = resolve_judge("string", PresetCompany, budget_usd=3.5)
        assert resolved.module._budget_usd == 3.5

    def test_zero_shot_llm_explicit_defaults_model_when_none(self) -> None:
        resolved = resolve_judge("zero_shot_llm", PresetCompany, model=None)
        assert resolved.judge_used == "zero_shot_llm"
        assert resolved.model == _OPENROUTER_MODEL
        assert resolved.module._module.model == _OPENROUTER_MODEL  # type: ignore[attr-defined]

    def test_injected_module_reports_judge_used_custom(self) -> None:
        injected: DSPyJudge[PresetCompany] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        resolved = resolve_judge(injected, PresetCompany)
        assert resolved.judge_used == "custom"
        assert resolved.module._module is injected

    def test_auto_resolution_is_delegated_to_choose_auto_judge(self) -> None:
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-not-a-real-key"}, clear=True),
            pytest.warns(UserWarning, match="selected the LLM judge"),
        ):
            resolved = resolve_judge("auto", PresetCompany)
        assert resolved.judge_used == "zero_shot_llm"
        assert resolved.model == _OPENROUTER_MODEL
        assert isinstance(resolved.module._module, DSPyJudge)

    def test_auto_resolution_raises_without_keys(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pydantic_settings.sources.DotEnvSettingsSource.__call__", return_value={}),
            pytest.raises(NoJudgeAvailableError),
        ):
            resolve_judge("auto", PresetCompany)

    def test_auto_resolution_raises_with_langres_offline_env(self) -> None:
        """LANGRES_OFFLINE=1 forces the keyless fail-fast path WITHOUT any
        dotenv patching -- hermetic even inside a repo whose .env carries a
        real key (the process env beats the .env file). This is the
        documented, deterministic way to test NoJudgeAvailableError; before
        it existed, no environment manipulation could produce a keyless run
        in-repo (popping the vars just let .env refill them)."""
        with (
            patch.dict(
                "os.environ",
                {"LANGRES_OFFLINE": "1", "OPENROUTER_API_KEY": "fake-not-a-real-key"},
                clear=True,
            ),
            pytest.raises(NoJudgeAvailableError, match="LANGRES_OFFLINE"),
        ):
            resolve_judge("auto", PresetCompany)

    def test_auto_resolution_raises_with_empty_key_env_vars(self) -> None:
        """The per-key keyless mechanism: an env var set to the EMPTY string
        wins over the .env file (no dotenv patching here) and counts as
        absent, so ``OPENROUTER_API_KEY="" OPENAI_API_KEY=""`` fails fast."""
        with (
            patch.dict(
                "os.environ", {"OPENROUTER_API_KEY": "", "OPENAI_API_KEY": ""}, clear=True
            ),
            pytest.raises(NoJudgeAvailableError, match="no API key"),
        ):
            resolve_judge("auto", PresetCompany)

    def test_auto_path_honors_caller_model_override(self) -> None:
        """Regression: choose_auto_judge's key-derived pick used to clobber a
        caller-supplied model= on the auto path."""
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "fake-not-a-real-key"}, clear=True),
            pytest.warns(UserWarning, match=re.escape(_OPENAI_MODEL)),
        ):
            resolved = resolve_judge("auto", PresetCompany, model=_OPENAI_MODEL)
        assert resolved.model == _OPENAI_MODEL
        assert resolved.module._module.model == _OPENAI_MODEL  # type: ignore[attr-defined]

    def test_explicit_zero_shot_llm_emits_no_selection_notice(self) -> None:
        """The selection notice is auto-path-only: an explicit judge name is
        the caller's own decision and gets no extra chatter here."""
        with warnings_none():
            resolved = resolve_judge("zero_shot_llm", PresetCompany)
        assert resolved.judge_used == "zero_shot_llm"


# ---------------------------------------------------------------------------
# notice_pre_scoring_cost / _estimate_n_pairs
# ---------------------------------------------------------------------------


class TestNoticeAndEstimate:
    def test_notice_message_format(self) -> None:
        with pytest.warns(UserWarning, match=r"scoring ~10 pairs with '.*', est\. cost \$"):
            notice_pre_scoring_cost(_OPENROUTER_MODEL, 10)

    def test_unpinned_model_warns_blind_cap_not_reassuring_zero(self) -> None:
        """M1 regression: an unpinned paid model must never print the
        reassuring (and false) "est. cost $0.0000" -- the spend cap tallies
        that same $0 and can never trip, so it's silently blind while
        OpenRouter still bills. The notice must say so honestly."""
        with pytest.warns(UserWarning, match=r"CANNOT enforce a limit") as record:
            notice_pre_scoring_cost("unknown/model-not-in-table", 10, budget_usd=2.5)
        message = str(record[0].message)
        assert "est. cost $0.0000" not in message
        assert "unknown/model-not-in-table" in message
        assert "$2.50" in message

    def test_unpinned_model_defaults_budget_in_message_when_omitted(self) -> None:
        with pytest.warns(UserWarning, match=rf"\${DEFAULT_BUDGET_USD:.2f}"):
            notice_pre_scoring_cost("unknown/model-not-in-table", 10)

    def test_estimate_all_pairs(self) -> None:
        assert _estimate_n_pairs(5, use_vector=False) == 10  # C(5,2)

    def test_estimate_vector(self) -> None:
        assert _estimate_n_pairs(5, use_vector=True) == 50  # 5 * k=10


# ---------------------------------------------------------------------------
# build_resolver
# ---------------------------------------------------------------------------


class TestBuildResolver:
    def test_string_judge_small_n_uses_all_pairs_blocker(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="string",
            model=None,
            entity_noun="entity",
            threshold=None,
            n_records=5,
        )
        assert resolved.judge_used == "string"
        assert resolved.score_type == "heuristic"
        assert isinstance(resolved.resolver.blocker, AllPairsBlocker)
        assert resolved.resolver.comparator is not None

    def test_string_judge_large_n_uses_vector_blocker(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="string",
            model=None,
            entity_noun="entity",
            threshold=None,
            n_records=_ALL_PAIRS_MAX_N + 1,
        )
        assert isinstance(resolved.resolver.blocker, VectorBlocker)

    def test_embedding_judge_always_uses_vector_blocker_even_for_small_n(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="embedding",
            model=None,
            entity_noun="entity",
            threshold=None,
            n_records=2,
        )
        assert isinstance(resolved.resolver.blocker, VectorBlocker)
        assert resolved.resolver.comparator is None
        assert resolved.score_type == "sim_cos"

    def test_threshold_defaults_per_judge_when_none(self) -> None:
        resolved = build_resolver(
            PresetCompany,
            judge="embedding",
            model=None,
            entity_noun="e",
            threshold=None,
            n_records=2,
        )
        assert resolved.resolver.clusterer.threshold == 0.5

    def test_explicit_threshold_overrides_default(self) -> None:
        resolved = build_resolver(
            PresetCompany, judge="string", model=None, entity_noun="e", threshold=0.9, n_records=2
        )
        assert resolved.resolver.clusterer.threshold == 0.9

    def test_zero_shot_llm_emits_pre_scoring_notice(self) -> None:
        with pytest.warns(UserWarning, match="scoring ~"):
            build_resolver(
                PresetCompany,
                judge="zero_shot_llm",
                model=_OPENROUTER_MODEL,
                entity_noun="e",
                threshold=None,
                n_records=4,
            )

    def test_zero_shot_llm_explicit_unpinned_model_warns_blind_cap(self) -> None:
        """M1 regression: an explicit judge="zero_shot_llm" with an unpinned
        model= must warn that the spend cap is blind (construction only --
        DummyLM-equivalent zero-spend, DSPyJudge is never .forward()ed here),
        not silently proceed under a reassuring but false $0.0000 estimate."""
        with pytest.warns(UserWarning, match="CANNOT enforce a limit") as record:
            build_resolver(
                PresetCompany,
                judge="zero_shot_llm",
                model="unknown/model-not-in-table",
                entity_noun="e",
                threshold=None,
                n_records=4,
                budget_usd=3.0,
            )
        message = str(record[0].message)
        assert "est. cost $0.0000" not in message
        assert "$3.00" in message

    def test_string_judge_emits_no_notice(self) -> None:
        with warnings_none():
            build_resolver(
                PresetCompany,
                judge="string",
                model=None,
                entity_noun="e",
                threshold=None,
                n_records=4,
            )

    def test_custom_module_uses_n_based_blocker_rule_and_no_notice(self) -> None:
        injected: DSPyJudge[PresetCompany] = DSPyJudge(lm=DummyLM([]), entity_noun="thing")
        with warnings_none():
            resolved = build_resolver(
                PresetCompany,
                judge=injected,
                model=None,
                entity_noun="e",
                threshold=None,
                n_records=4,
            )
        assert resolved.judge_used == "custom"
        assert isinstance(resolved.resolver.blocker, AllPairsBlocker)
        assert resolved.score_type == "unknown"


# ---------------------------------------------------------------------------
# _text_field_extractor (schema-agnostic)
# ---------------------------------------------------------------------------


class TestTextFieldExtractor:
    def test_concatenates_non_empty_string_fields(self) -> None:
        extractor = _text_field_extractor(PresetCompany)
        entity = PresetCompany(id="1", name="Acme", address=None)
        assert extractor(entity) == "Acme"

    def test_schema_agnostic_second_schema(self) -> None:
        extractor = _text_field_extractor(PresetProduct)
        entity = PresetProduct(id="1", title="Widget", brand="Acme")
        text = extractor(entity)
        assert "Widget" in text and "Acme" in text


# ---------------------------------------------------------------------------
# Vector-blocker / embedding construction (real MiniLM load -- slow, local, $0)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestVectorBlockerAndEmbedding:
    def test_build_vector_blocker_shape(self) -> None:
        blocker = _build_vector_blocker(PresetCompany)
        assert isinstance(blocker, VectorBlocker)
        assert blocker.k_neighbors == 10

    def test_build_embedding_candidate_identical_texts_score_near_one(self) -> None:
        record = {"id": "a", "name": "Acme Corporation"}
        candidate = build_embedding_candidate(PresetCompany, record, dict(record, id="b"))
        assert candidate.similarity_score is not None
        assert candidate.similarity_score > 0.99

    def test_build_embedding_candidate_different_texts_score_lower(self) -> None:
        left = {"id": "a", "name": "Acme Corporation"}
        right = {"id": "b", "name": "Totally Unrelated Restaurant Chain"}
        candidate = build_embedding_candidate(PresetCompany, left, right)
        assert candidate.similarity_score is not None
        assert candidate.similarity_score < 0.99


class TestBuildEmbeddingCandidateNoCandidateGuard:
    """L2 regression: a bare StopIteration must never leak from next()."""

    def test_empty_blocker_stream_raises_clear_runtime_error(self) -> None:
        fake_blocker = _FakeEmptyStreamBlocker()
        with patch("langres.core.presets._build_vector_blocker", return_value=fake_blocker):
            with pytest.raises(RuntimeError, match="produced no candidate"):
                build_embedding_candidate(
                    PresetCompany, {"id": "a", "name": "X"}, {"id": "b", "name": "Y"}
                )

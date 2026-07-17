"""The W4 proofs: the claims the ERModel/architectures wave stands or falls on.

Each test here exists to be **falsifiable**. That distinction is the point of the
file, and it is worth being blunt about:

    FuzzyString().dedupe(records)   # $0

is a **smoke test, not a proof**. It cannot fail in the interesting way: it would
pass just as happily if the spend cap were broken, if `auto` still sniffed keys,
and if a paid call were one refactor away — because nothing in it *could* spend.
A test that cannot fail gates nothing. So the proofs below each name a specific
way the wave could be wrong, and then try to make it wrong:

- **#2a** -- a POPULATED ``.env`` on disk, no keys exported, and a paid
  architecture. Proves the paid path spends only when the user names the class,
  and that the cap actually binds. Fails if `auto` comes back, if construction
  spends, or if the cap leaks.
- **#2b** -- a tripwire on ``Settings.__init__``. Fails if any import-time or
  default-construction path calls it.
- **#3** -- same architecture, two backbones. Fails if swapping a backbone mints
  a new architecture identity, or if the swap is cosmetic (the embedder unchanged).
- **#4** -- weightless save/load round-trip. Fails if config drifts, if weight
  bytes land in the artifact, or if a loaded architecture decays into a base
  ERModel. This is the HF-readiness gate, and T1 (`load` calling `__init__`) is
  what stood in its way.

**Money safety.** Nothing here makes a live paid call. The paid architecture's
matcher slot is replaced with a fake that *reports* a cost, so the ledger and cap
are exercised for real while the network is not. `env -u` and `os.environ.pop()`
are theatre in this repo -- litellm's import-time `load_dotenv()` walks up from
any worktree and pydantic-settings reads the repo `.env` FILE directly -- so this
file never relies on either to stay free. It relies on never constructing a real
client.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Any

import pytest
from pydantic import BaseModel

from langres.architectures import FuzzyString, VectorLLMCascade
from langres.core.matcher import Matcher
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.registry import model_type_name
from langres.core.resolver import ERModel
from langres.core.spend import BudgetExceeded


class Company(BaseModel):
    id: str
    name: str | None = None
    city: str | None = None


RECORDS: list[dict[str, Any]] = [
    {"id": "1", "name": "Acme Corporation", "city": "NY"},
    {"id": "2", "name": "Acme Corp", "city": "NY"},
    {"id": "3", "name": "Zebra Bakery", "city": "LA"},
]


class _PricedFakeMatcher(Matcher[Any]):
    """A matcher that reports a real dollar cost without making a real call.

    The spend cap meters ``provenance["cost_usd"]``; it neither knows nor cares
    whether a socket was opened. So this exercises the ledger, the cap and the
    ``BudgetExceeded`` path *exactly* as a live LLM would, at $0 and offline.
    """

    def __init__(self, cost_per_pair: float = 0.40) -> None:
        self.cost_per_pair = cost_per_pair
        self.calls = 0
        self.model = "fake/priced-model"

    def forward(self, candidates: Any) -> Any:
        for candidate in candidates:
            self.calls += 1
            yield PairwiseJudgement(
                left_id=candidate.left.id,
                right_id=candidate.right.id,
                score=0.9,
                score_type="prob_llm",
                decision_step="fake_priced",
                provenance={"cost_usd": self.cost_per_pair, "model": self.model},
            )


# ---------------------------------------------------------------------------
# Proof #2a -- the honest spend gate
# ---------------------------------------------------------------------------


class TestProof2aHonestSpendGate:
    """A paid model spends only because you named it -- and the cap binds."""

    def test_a_reachable_populated_env_file_is_the_hazard_being_disarmed(self) -> None:
        """Guard the guard: #2a is vacuous if no reachable `.env` carries a key.

        The whole point of #2a is that a populated ``.env`` on disk does NOT cause
        spend. If no such file were reachable, every proof below would pass for
        the wrong reason and nobody would notice. So assert the hazard is REAL
        before asserting we are safe from it.

        **It walks UP the tree, deliberately** -- that is the actual threat model,
        and checking only the local directory is how you fool yourself here. This
        very test file usually runs from a git worktree nested under the main
        checkout, which has no ``.env`` of its own; the key lives two or three
        levels up. litellm's import-time ``load_dotenv()`` walks up exactly like
        this and finds it. (Measured while writing this test: the worktree root
        has no ``.env``; ``/Users/davidgraf/work/langres/.env`` has a live
        ``OPENROUTER_API_KEY``. A local-only check skipped, and reported safety it
        had not established.)
        """
        here = pathlib.Path(__file__).resolve()
        found = [
            parent / ".env"
            for parent in here.parents
            if (parent / ".env").is_file()
            and any(
                key in (parent / ".env").read_text()
                for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY")
            )
        ]
        if not found:
            pytest.skip(
                "no reachable .env carries an LLM API key (e.g. a clean CI checkout) -- "
                "#2a's hazard is genuinely absent here, so there is nothing to disarm"
            )
        # The hazard is live: a key is sitting on disk, reachable by the exact
        # walk-up litellm performs. Every proof below runs with it there.

    def test_no_module_level_name_can_dedupe_without_naming_a_model(self) -> None:
        """`langres.dedupe(records)` must not exist -- at all, in any form.

        The deleted verb defaulted to matcher="auto", which read the .env above
        and spent. The fix is not a safer default; it is that there is no
        module-level verb to give a default TO.
        """
        import langres

        for gone in ("dedupe", "link", "NoMatcherAvailableError", "DEFAULT_AUTO_MODEL"):
            assert not hasattr(langres, gone), (
                f"langres.{gone} still exists. The verbs and the auto key-sniffing "
                "path were deleted, not shimmed."
            )
            assert gone not in langres.__all__

        with pytest.raises(ModuleNotFoundError):
            __import__("langres.verbs")
        with pytest.raises(ModuleNotFoundError):
            __import__("langres.core.presets")

    def test_constructing_the_paid_model_spends_nothing(self) -> None:
        """Naming the class is free. Only running it can cost."""
        model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", budget_usd=1.0)
        assert model._spend_monitor.spent == 0.0
        # Not even the components exist yet -- no client, no embedder, no index.
        assert not model.is_bound

    def test_the_paid_model_spends_only_when_run_and_the_cap_binds(self) -> None:
        """The cap truncates a runaway bill mid-batch, and hands back what was paid for.

        The batch is sized so the cap MUST bite: 10 records -> 45 all-pairs
        candidates at $0.40 = **$18.00** wanted, against a **$1.00** budget. A
        broken cap spends 18x the budget here, which is the failure this test
        exists to catch. (An earlier draft used 3 records/$1.20 -- the cap passed,
        but so would a cap that merely stopped one call late. Too small a batch
        cannot tell a binding cap from a lucky one.)
        """
        records = [{"id": str(i), "name": f"Acme {i}", "city": "NY"} for i in range(10)]
        fake = _PricedFakeMatcher(cost_per_pair=0.40)
        model = ERModel.from_schema(Company, matcher=fake, threshold=0.5, budget_usd=1.00)
        assert model._spend_monitor.spent == 0.0  # constructed != run
        assert fake.calls == 0

        with pytest.raises(BudgetExceeded) as exc:
            model.dedupe(records)

        # Bounded at budget + AT MOST ONE further call -- an LLM call's cost is
        # only knowable once it has been made, so overshooting by one is the
        # honest floor, not a bug. $1.00 + $0.40 = $1.40 is the ceiling.
        assert model._spend_monitor.spent <= 1.40 + 1e-9
        assert fake.calls <= 4, f"the cap let {fake.calls} of 45 pairs through -- it is not binding"
        assert exc.value.partial_judgements, "paid-for judgements were dropped on the floor"

    def test_one_budget_spans_the_whole_model_lifetime_not_per_call(self) -> None:
        """Repeated dedupes cannot each spend a full budget (B1).

        The ledger is per-instance and built once; only the cap WRAPPER is rebuilt
        per call. Get that backwards and a long-lived model multiplies its budget
        by its call count -- an unbounded spend with a cap that reports success.

        Sized so the bug it names MUST fail it: 10 calls x $0.30 = $3.00 against a
        $1.00 budget, so a per-instance ledger is *obliged* to raise partway
        through. A per-call ledger restarts at $0.00 every time and would sail
        through all ten, so the ``pytest.fail`` below is reachable -- an earlier
        version of this test spent only $0.90 against $1.00, never tripped the cap
        at all, and passed just as happily under the bug as without it.
        """
        fake = _PricedFakeMatcher(cost_per_pair=0.30)
        model = ERModel.from_schema(Company, matcher=fake, threshold=0.5, budget_usd=1.00)
        pair = RECORDS[:2]

        for _ in range(10):
            try:
                model.dedupe(pair)
            except BudgetExceeded:
                break
        else:
            pytest.fail(
                f"10 calls x $0.30 = $3.00 against a $1.00 budget never raised: the "
                f"ledger is being rebuilt per call, so each call gets a fresh budget "
                f"(spent reads ${model._spend_monitor.spent:.2f})"
            )
        assert model._spend_monitor.spent <= 1.30 + 1e-9, (
            "spend exceeded one budget plus at most one further call across the "
            "model's lifetime -- the monitor is not accumulating"
        )

    def test_fuzzystring_at_zero_is_a_smoke_test_not_a_proof(self) -> None:
        """$0 offline, deterministic, no key. **Labelled as a smoke test on purpose.**

        This is the headline DX and it must work -- but it proves nothing about
        spend safety, because FuzzyString has no way to spend whether or not the
        cap works. The real gate is the sibling tests above. Kept, and named,
        so nobody mistakes it for the proof.
        """
        first = FuzzyString(threshold=0.6).dedupe(RECORDS)
        second = FuzzyString(threshold=0.6).dedupe(RECORDS)

        assert [sorted(c) for c in first] == [["1", "2"]]
        assert list(first) == list(second)  # deterministic
        assert first.architecture == "FuzzyString"
        assert first.backbone is None  # nothing with weights ran; honest, not a gap
        assert first.threshold == 0.6


# ---------------------------------------------------------------------------
# Proof #2b -- nothing reaches Settings()/litellm by default
# ---------------------------------------------------------------------------


class TestProof2bNoDefaultPathReachesSettingsOrLitellm:
    def test_no_import_or_construction_path_calls_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tripwire on ``Settings.__init__``, not a grep.

        ``Settings()`` is what reads the repo's `.env` FILE (pydantic-settings
        does this directly -- which is why `env -u`/`os.environ.pop()` are
        theatre here). If nothing calls it, nothing can discover a key.
        """
        import langres.clients.settings as settings_mod

        calls: list[str] = []
        real_init = settings_mod.Settings.__init__

        def tripwire(self: Any, *args: Any, **kwargs: Any) -> None:
            calls.append("Settings()")
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(settings_mod.Settings, "__init__", tripwire)

        import langres  # noqa: F401
        from langres.architectures import FuzzyString as _FS
        from langres.architectures import VectorLLMCascade as _VLC

        assert calls == [], f"import reached Settings(): {calls}"

        _VLC(llm="openrouter/openai/gpt-4o-mini")
        _FS()
        assert calls == [], f"constructing an architecture reached Settings(): {calls}"

        _FS(threshold=0.6).dedupe(RECORDS)
        assert calls == [], f"the $0 architecture's dedupe() reached Settings(): {calls}"

    def test_bare_import_langres_loads_no_heavy_dep(self) -> None:
        """litellm/torch/faiss must not ride in on `import langres`."""
        import subprocess
        import sys

        heavy = subprocess.run(
            [
                sys.executable,
                "-c",
                "import langres, langres.architectures, sys;"
                "langres.architectures.VectorLLMCascade(llm='openrouter/openai/gpt-4o-mini');"
                "print([m for m in ('litellm','torch','faiss','sentence_transformers','dspy')"
                " if m in sys.modules])",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert heavy.stdout.strip() == "[]", (
            f"a heavy dep entered sys.modules on import+construct: {heavy.stdout}"
        )

    def test_architectures_keep_heavy_imports_out_of_module_scope(self) -> None:
        """AST gate: no ``[semantic]``/``[llm]`` import at an architecture's module scope.

        The lazy seam is only lazy if it stays inside a function body. A module-
        scope `from langres.core.matchers.llm_judge import LLMMatcher` would put
        litellm in every `import langres` -- and the sibling test above would
        catch it only while `_exports/_models.py` imports this package eagerly.
        This one states the rule directly.
        """
        heavy_modules = (
            "langres.core.matchers.llm_judge",
            "langres.core.blockers.vector",
            "langres.core.embeddings",
            "langres.core.indexes.vector_index",
            "langres.core.matchers.cascade_judge",
        )
        pkg = pathlib.Path(__import__("langres.architectures", fromlist=["x"]).__file__).parent
        offenders: list[str] = []
        for path in sorted(pkg.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in tree.body:  # module scope ONLY -- function bodies are the point
                if isinstance(node, ast.ImportFrom) and node.module in heavy_modules:
                    offenders.append(f"{path.name}:{node.lineno} imports {node.module}")
        assert not offenders, (
            f"heavy imports at architecture module scope (must move inside _topology): {offenders}"
        )


# ---------------------------------------------------------------------------
# Proof #3 -- a backbone swap never mints a new architecture
# ---------------------------------------------------------------------------


class TestProof3BackboneSwapKeepsArchitectureIdentity:
    @pytest.mark.slow  # constructs a real SentenceTransformerEmbedder ([semantic])
    def test_same_architecture_different_embedder_same_identity(self) -> None:
        """Two embedders, one architecture. The identity must not move.

        This is the invariant the whole architecture/backbone split exists to
        protect: swapping what fills a model slot is a *configuration* change,
        not a new topology. If this fails, "backbone" has quietly become
        "architecture" and the vocabulary is a lie.
        """
        pytest.importorskip("sentence_transformers")
        pytest.importorskip("faiss")

        a = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", embedder="all-MiniLM-L6-v2")
        b = VectorLLMCascade(
            llm="openrouter/openai/gpt-4o-mini", embedder="paraphrase-MiniLM-L3-v2"
        )

        # Same class, same registered identity, same manifest name.
        assert type(a) is type(b) is VectorLLMCascade
        assert model_type_name(type(a)) == model_type_name(type(b)) == "vector_llm_cascade"

        # ...and the swap is REAL, not cosmetic: bind both and read the embedder
        # that the blocker will actually run.
        a._bind(Company)
        b._bind(Company)
        embedder_a = a.blocker.vector_index.embedder.model_name  # type: ignore[attr-defined]
        embedder_b = b.blocker.vector_index.embedder.model_name  # type: ignore[attr-defined]
        assert embedder_a == "all-MiniLM-L6-v2"
        assert embedder_b == "paraphrase-MiniLM-L3-v2"
        assert embedder_a != embedder_b, "the backbone swap did not change the embedder in use"

    def test_llm_backbone_swap_keeps_identity_and_is_visible(self) -> None:
        """The same, for the paid slot -- and free of the [semantic] extra."""
        a = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini")
        b = VectorLLMCascade(llm="openrouter/deepseek/deepseek-chat")

        assert type(a) is type(b)
        assert model_type_name(type(a)) == model_type_name(type(b)) == "vector_llm_cascade"
        # Different backbone, and the model says so rather than hiding it.
        assert a.backbone == "openrouter/openai/gpt-4o-mini"
        assert b.backbone == "openrouter/deepseek/deepseek-chat"
        assert a.backbone != b.backbone

    def test_backbones_are_weightless_reference_strings(self) -> None:
        """A backbone is a ModelRef -- strings, never weight bytes."""
        model = VectorLLMCascade(llm={"base": "org/ft-matcher", "kind": "hf", "revision": "abc123"})
        assert model.llm.base == "org/ft-matcher"
        assert model.llm.kind == "hf"
        assert model.llm.revision == "abc123"


# ---------------------------------------------------------------------------
# Proof #4 -- the weightless save/load round-trip (the HF-readiness gate)
# ---------------------------------------------------------------------------


class TestProof4WeightlessRoundTrip:
    def test_a_named_architecture_reloads_as_itself_with_identical_config(
        self, tmp_path: pathlib.Path
    ) -> None:
        """save -> load -> identical versioned config, and still a FuzzyString.

        **This is the test T1 blocked.** Before ``from_components``, ``load``
        called ``FuzzyString(blocker=..., comparator=...)`` and died with
        ``TypeError: unexpected keyword argument 'blocker'`` -- the artifact
        recorded its architecture faithfully and then could not rebuild it.
        """
        original = FuzzyString(threshold=0.62, schema=Company)
        original.save(tmp_path / "m")

        loaded = ERModel.load(tmp_path / "m")

        # It came back as ITSELF, not as a base ERModel.
        assert type(loaded) is FuzzyString
        assert isinstance(loaded, ERModel)
        # The config is identical, not merely equivalent.
        assert loaded.config_dict() == original.config_dict()
        # And the round-trip is idempotent: re-saving a loaded model reproduces
        # the artifact byte-for-byte. This is what pins the from_components
        # invariant -- an architecture that hid state outside its slots would
        # come back missing it and drift here on the second lap.
        loaded.save(tmp_path / "m2")
        assert (tmp_path / "m2" / "resolver.json").read_text() == (
            tmp_path / "m" / "resolver.json"
        ).read_text()

    def test_the_artifact_carries_config_not_weights(self, tmp_path: pathlib.Path) -> None:
        """Weightless by construction: a config file, and nothing that looks like weights."""
        FuzzyString(threshold=0.62, schema=Company).save(tmp_path / "m")

        files = sorted(p.name for p in (tmp_path / "m").iterdir())
        assert files == ["resolver.json"]

        manifest = (tmp_path / "m" / "resolver.json").read_text()
        assert '"model_class": "fuzzy_string"' in manifest
        assert "0.62" in manifest
        # Small enough to be self-evidently config. Real weights are megabytes.
        assert (tmp_path / "m" / "resolver.json").stat().st_size < 4096
        for weighty in (".safetensors", ".bin", ".pt", ".faiss", ".pkl"):
            assert not list((tmp_path / "m").rglob(f"*{weighty}"))

    def test_a_loaded_model_actually_runs_and_agrees_with_the_original(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A round-trip that reconstructs a museum piece is not a round-trip."""
        original = FuzzyString(threshold=0.62, schema=Company)
        original.save(tmp_path / "m")
        loaded = ERModel.load(tmp_path / "m")

        assert [sorted(c) for c in loaded.dedupe(RECORDS)] == [
            sorted(c) for c in original.dedupe(RECORDS)
        ]
        assert loaded.dedupe(RECORDS).architecture == "FuzzyString"

    def test_load_does_not_replay_the_ergonomic_constructor(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """T1, stated directly: ``load`` must never call the class's ``__init__``.

        Pinning the *mechanism*, not just its happy-path result -- because the
        result would keep passing if someone "fixed" a future architecture by
        widening its ``__init__`` to swallow ``**kwargs``, which is exactly the
        design this wave rejected.
        """
        FuzzyString(threshold=0.62, schema=Company).save(tmp_path / "m")

        def exploding_init(self: Any, **kwargs: Any) -> None:
            raise AssertionError(f"load() called the ergonomic __init__ with {sorted(kwargs)}")

        monkeypatch.setattr(FuzzyString, "__init__", exploding_init)
        loaded = ERModel.load(tmp_path / "m")
        assert type(loaded) is FuzzyString
        assert loaded.clusterer.threshold == 0.62


# ---------------------------------------------------------------------------
# T5 -- an architecture defends its identity at the fit boundary
# ---------------------------------------------------------------------------


class TestArchitecturesDefendIdentityAtFit:
    def test_fuzzystring_refuses_a_topology_changing_fit(self) -> None:
        """``fit(method=QLoRA())`` would repoint the matcher slot at an LLM.

        The result would be an LLM pipeline still calling itself "FuzzyString" --
        a name that lies. ``accepted_method_kinds`` is how a class refuses.
        """
        from langres.core.methods_api import UnsupportedMethodKind

        assert FuzzyString.accepted_method_kinds == frozenset({"calibrate"})
        assert VectorLLMCascade.accepted_method_kinds == frozenset()
        # The base makes no identity claim, so it constrains nothing.
        assert ERModel.accepted_method_kinds is None

        pytest.importorskip("sklearn")
        from langres.training.methods_prompt import MIPRO

        with pytest.raises(UnsupportedMethodKind):
            FuzzyString(schema=Company).fit(RECORDS, method=MIPRO())

    def test_fuzzystring_accepts_the_calibrate_fit_it_declares(self) -> None:
        """The gate must not be a blanket "no" -- that would prove nothing."""
        pytest.importorskip("sklearn")
        from langres.training.methods_calibrate import Platt

        records: list[dict[str, Any]] = []
        pairs = []
        from langres.core.harvest import LabeledPair

        for g, (x, y) in enumerate([("Acme", "Beta"), ("Gamma", "Delta"), ("Eps", "Zeta")]):
            x0, x1, y0 = f"g{g}x0", f"g{g}x1", f"g{g}y0"
            records += [
                {"id": x0, "name": f"{x} Corp"},
                {"id": x1, "name": f"{x} Corporation"},
                {"id": y0, "name": f"{y} Inc"},
            ]
            pairs += [
                LabeledPair(left_id=x0, right_id=x1, score=None, label=True, source="correction"),
                LabeledPair(left_id=x0, right_id=y0, score=None, label=False, source="correction"),
            ]

        model = FuzzyString(schema=Company)
        model.fit(records, pairs=pairs, method=Platt())
        assert model.calibrator is not None


class TestReviewFoundRegressions:
    """Three defects a code review caught that every proof above sailed past.

    Recorded as tests rather than quiet fixes, because each one marks a blind
    spot in the proofs themselves:

    * The proofs round-trip only ``FuzzyString`` -- the one architecture with no
      backbones -- so ``VectorLLMCascade``'s ``save`` and its reconstruction path
      were never executed at all. That gap is exactly where two of the three
      lived.
    * ``ERModel.schema`` was asserted nowhere, so a property that returned
      ``None`` for every model in the library looked fine.
    """

    def test_an_explicit_schema_is_actually_used(self) -> None:
        """``FuzzyString(schema=S).schema`` must be ``S`` (it was ``None``).

        ``ERModel.schema`` reads the blocker slot -- correct by design, since the
        blocker is what survives save/load -- but NO blocker exposed ``.schema``,
        so it silently returned ``None`` for every architecture. ``_prepare`` then
        passed ``None`` to ``normalize_records`` and took the *inferred* path even
        when the caller had handed over a schema.
        """
        assert FuzzyString(schema=Company).schema is Company
        assert FuzzyString().schema is None, "an unbound model is bound to nothing"

    def test_an_explicit_schema_defeats_the_nested_value_guard(self) -> None:
        """The sharpest face of the bug: the error told you to do what you did.

        Inference rejects a nested value and says "Pass schema=<YourModel>
        explicitly". Passing one changed nothing, so the advice was unfollowable.
        """

        class Nested(BaseModel):
            id: str
            name: str
            meta: dict[str, Any] | None = None

        model = FuzzyString(schema=Nested)
        clusters = model.dedupe(
            [
                {"id": "1", "name": "Acme Corporation", "meta": {"src": "a"}},
                {"id": "2", "name": "Acme Corp", "meta": {"src": "b"}},
            ]
        )
        assert clusters == [{"1", "2"}]

    def test_schemaless_inference_still_works(self) -> None:
        """The fix must not cost the schema-optional path that makes the DX."""
        model = FuzzyString()
        assert model.dedupe([{"name": "Acme Corporation"}, {"name": "Acme Corp"}]) == [{"0", "1"}]

    def test_from_components_does_not_need_the_constructor_arguments(self) -> None:
        """``VectorLLMCascade.backbone`` read ``self.llm`` -- absent after ``load``.

        ``from_components`` builds via ``cls.__new__``, so ``__init__`` never runs
        and no constructor argument exists on the instance. Reading one raised
        ``AttributeError`` on ``.backbone`` and on any ``dedupe``. Identity must
        come from the slots; this is that invariant, executed.
        """
        from langres.core.blockers import AllPairsBlocker
        from langres.core.clusterer import Clusterer
        from langres.core.comparators import StringComparator
        from langres.core.matchers import WeightedAverageMatcher

        comparator: StringComparator[Any] = StringComparator.from_schema(Company)
        rebuilt = VectorLLMCascade.from_components(
            blocker=AllPairsBlocker(schema=Company),
            comparator=comparator,
            matcher=WeightedAverageMatcher(feature_specs=comparator.feature_specs),
            clusterer=Clusterer(threshold=0.5),
        )
        # No AttributeError, and honest: nothing with an LLM is in the slot.
        assert rebuilt.backbone is None
        assert rebuilt.dedupe(RECORDS[:2]) == [{"1", "2"}]

    def test_a_constructed_cascade_still_reports_the_llm_it_was_given(self) -> None:
        """Sourcing from the slot must not cost the answer we already have.

        An unbound model has no slots yet (topology builds lazily) but was handed
        an ``llm=``; reporting ``None`` there would hide a known answer.
        """
        assert VectorLLMCascade(llm="openrouter/deepseek/deepseek-chat").backbone == (
            "openrouter/deepseek/deepseek-chat"
        )

    def test_vector_llm_cascade_save_fails_with_a_followable_reason(self) -> None:
        """The gap is real; the message must be about the user's model, not ours.

        `VectorLLMCascade` cannot persist: its `VectorBlocker` holds a
        `text_field_extractor` closure. Unfixed, the user got VectorBlocker's own
        error telling them to "construct with schema= and text_field=" -- but they
        never constructed the VectorBlocker, `_topology` did. This pins the honest
        failure until the named-extractor seam lands.
        """
        model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", schema=Company)
        with pytest.raises(NotImplementedError, match="cannot be saved yet"):
            model.save("unreachable")

    def test_save_error_names_every_required_arg_of_the_escape_hatch(self) -> None:
        """The advice must RUN. It told users to do something that TypeErrors.

        The message's escape hatch is a hand-built `VectorBlocker` with a named
        `text_field=` instead of the closure. As shipped it read
        ``VectorBlocker(schema=..., text_field=...)`` -- which raises
        ``TypeError: missing 1 required positional argument: 'vector_index'``.
        Advice a user cannot copy is worse than none: it costs them the round-trip
        to discover our error message is wrong.

        This asserts the message names every REQUIRED parameter, derived from the
        live signature rather than hard-coded, so adding a required arg to
        `VectorBlocker` fails here instead of silently rotting the message again.
        """
        import inspect

        from langres.core.blockers.vector import VectorBlocker

        model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini", schema=Company)
        try:
            model.save("unreachable")
        except NotImplementedError as exc:
            message = str(exc)

        required = [
            name
            for name, p in inspect.signature(VectorBlocker.__init__).parameters.items()
            if name != "self" and p.default is inspect.Parameter.empty
        ]
        assert required, "VectorBlocker grew no required args -- rewrite this proof"
        for name in required:
            assert f"{name}=" in message, (
                f"save()'s advice omits required VectorBlocker arg {name!r}; a user "
                f"copying it hits a TypeError. Message: {message}"
            )

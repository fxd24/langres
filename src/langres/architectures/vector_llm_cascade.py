"""``VectorLLMCascade``: vector blocking + a cheap student + an LLM at the margin.

One architecture, one file, deliberately self-contained -- see
:mod:`langres.architectures` for why this file rebuilds a vector blocker that
other code also knows how to build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from langres.core.clusterer import Clusterer
from langres.core.comparators import StringComparator
from langres.core.matchers.embedding_score import EmbeddingScoreMatcher
from langres.core.model_ref import DEFAULT_EMBEDDING_MODEL, ModelRef, normalize_model_ref, to_config
from langres.core.registry import register_model
from langres.core.resolver import ERModel

__all__ = ["VectorLLMCascade"]

#: Neighbors per record the vector blocker retrieves. The recall/cost dial: it
#: makes blocking O(N*k) instead of O(N^2), and any true pair outside a record's
#: top-k is unrecoverable downstream -- no matcher can judge a pair it never sees.
_K_NEIGHBORS = 10

#: The student-score band that escalates to the LLM (both edges inclusive). Below
#: it the pair is confidently different, above it confidently the same; inside it
#: the cheap embedding score is not trustworthy and the paid model earns its fee.
#: **This band is the cost lever**: widen it to spend more and (hopefully) score
#: better, narrow it to spend less.
_ESCALATION_BAND = (0.35, 0.85)


@register_model("vector_llm_cascade")
class VectorLLMCascade(ERModel):
    """Vector blocking, an embedding student everywhere, an LLM only at the margin.

    **The paid architecture.** It makes real API calls, and it does so because
    you constructed it -- there is no key-sniffing, no ``"auto"``, and no way to
    reach this class by accident::

        from langres.architectures import VectorLLMCascade

        model = VectorLLMCascade(llm="openrouter/openai/gpt-4o-mini")
        clusters = model.dedupe(records)

    Topology:

    ===============  ===========================================================
    blocker          ``VectorBlocker`` (FAISS, cosine) over ``embedder`` -- O(N*k)
    comparator       ``StringComparator`` -- attached for provenance/inspection
    matcher          ``CascadeMatcher``: ``EmbeddingScoreMatcher`` (free) with
                     escalation to ``LLMMatcher`` (paid) inside the band
    clusterer        ``Clusterer`` -- transitive closure above ``threshold``
    ===============  ===========================================================

    **Architecture vs backbone, which this class exists to demonstrate.** The
    topology above is the architecture. ``embedder=`` and ``llm=`` are
    *backbones* -- what fills the two model slots. Swapping either gives you a
    different-performing, different-priced model that is still a
    ``VectorLLMCascade``: same class, same identity. That is the invariant, and
    ``TestProof3BackboneSwapKeepsArchitectureIdentity`` in
    ``tests/architectures/test_w4_proofs.py`` proves it rather than asserting it.
    Both are :class:`~langres.core.model_ref.ModelRef`\\ s, so both are
    **weightless**: neither carries weight bytes, only a reference string.

    .. warning::
       **This architecture cannot** ``save()`` **yet** -- it raises
       :class:`NotImplementedError` with the reason. Its ``VectorBlocker`` is
       built with a ``text_field_extractor`` closure (blocking text = every
       comparable field concatenated), and a callable cannot round-trip through
       JSON config. This is a **pre-existing limitation inherited from the
       deleted ``presets``/``_build_embedding_blocker`` path, not a W4
       regression** -- that path never called ``save()``, so it never surfaced.
       What W4 changed is that this is now a named class that *looks* like it
       persists, so the gap is stated rather than discovered at runtime.

       The fix is a named-extractor seam mirroring the one
       :class:`~langres.core.matchers.llm_judge.LLMMatcher` already has for
       ``response_parser`` (accept ``Callable | str``, resolve a registered name,
       serialize the name). Deliberately deferred: it changes what
       ``VectorBlocker`` accepts, in the paid path. ``FuzzyString`` round-trips
       today and proves the mechanism.

    **Where the money goes.** The student scores every blocked pair for free; only
    pairs in ``escalation_band`` reach the LLM. Cost therefore tracks the *band
    width* and the *blocked pair count* (``~N*k``), not the record count squared.
    Every paid call is metered by this model's one spend ledger and stops at
    ``budget_usd`` -- see :meth:`~langres.core.resolver.ERModel._scorer`.

    Requires the ``[semantic]`` extra (the embedder + FAISS) and ``[llm]`` (the
    LLM matcher). Both are imported lazily, inside ``_topology``, so merely
    importing this module -- or constructing the class -- pulls in neither, and
    reaches no ``Settings`` and no litellm. Construction is free and offline;
    only :meth:`~langres.core.resolver.ERModel.dedupe`/``compare`` can spend.

    Args:
        llm: The escalation backbone -- a litellm model id
            (``"openrouter/openai/gpt-4o-mini"``), a dict, or a
            :class:`~langres.core.model_ref.ModelRef`. This is the one that costs
            money.
        embedder: The student/blocking backbone -- a sentence-transformers model
            (``"BAAI/bge-base-en-v1.5"``). Defaults to
            :data:`~langres.core.model_ref.DEFAULT_EMBEDDING_MODEL`. Runs
            in-process; costs nothing but time.
        threshold: The match cut on the cascade's output.
        escalation_band: The student-score interval that escalates to the LLM.
            Widen to spend more, narrow to spend less.
        k_neighbors: Neighbors per record the blocker retrieves.
        entity_noun: The domain noun woven into the LLM's prompt ("company",
            "product", ...). Free, and it measurably helps.
        schema: The entity schema; omit to infer it from the records on first
            use. Pass it explicitly for anything you intend to ``save``.
        budget_usd: Spend cap for this model's whole lifetime, in USD. ``None``
            resolves to :data:`~langres.core.spend_cap.DEFAULT_BUDGET_USD` -- it
            does **not** mean uncapped.
    """

    #: Refuses every fit kind. ``prompt`` (DSPy) and ``finetune`` (QLoRA) both
    #: repoint the matcher slot at a different model -- ``_fit_finetune``
    #: literally replaces it -- which would leave something else calling itself
    #: VectorLLMCascade. ``calibrate`` is refused too, and that one is a genuine
    #: judgement call rather than an obvious "no": a Platt map is post-hoc and
    #: leaves the topology alone, so it is *safe* here in the way it is safe on
    #: FuzzyString. It is excluded because this architecture's scores come from a
    #: CascadeMatcher -- a mix of the student's cosine and the LLM's probability,
    #: two different scales stitched at the band edges -- and fitting one
    #: monotone map across that seam produces a number that looks like a
    #: calibrated probability and is not. Calibrate the LLM matcher directly if
    #: you need one. Flip this to ``frozenset({"calibrate"})`` if that reasoning
    #: is ever shown wrong; nothing else depends on it.
    accepted_method_kinds: ClassVar[frozenset[str] | None] = frozenset()

    def __init__(
        self,
        *,
        llm: str | dict[str, str] | ModelRef,
        embedder: str | dict[str, str] | ModelRef = DEFAULT_EMBEDDING_MODEL,
        threshold: float = 0.5,
        escalation_band: tuple[float, float] = _ESCALATION_BAND,
        k_neighbors: int = _K_NEIGHBORS,
        entity_noun: str = "entity",
        schema: type[BaseModel] | None = None,
        budget_usd: float | None = None,
    ) -> None:
        # Normalize both backbones HERE, at construction, so a malformed ref
        # raises now -- with the traceback pointing at the caller's own line --
        # rather than deep inside a blocker build on the first dedupe().
        self.llm = normalize_model_ref(llm)
        self.embedder = normalize_model_ref(embedder)
        self.threshold = threshold
        self.escalation_band = escalation_band
        self.k_neighbors = k_neighbors
        self.entity_noun = entity_noun
        self._init_state(budget_usd=budget_usd)
        if schema is not None:
            self._bind(schema)

    @property
    def backbone(self) -> str | None:
        """The **paid** backbone -- the LLM id.

        Overridden because the base implementation reads ``self.module.model``,
        and this architecture's matcher is a ``CascadeMatcher``, which has no
        ``model`` of its own: it is a composition of two matchers that do. The
        base would honestly report ``None``, which would be true and useless --
        this model plainly runs a model. Of the two backbones it carries, the LLM
        is the one that costs money and decides the hard pairs, so it is the one
        a result stamps. The embedder is reachable at ``self.embedder``.

        **The slot wins; the constructor argument is only a fallback.** The two
        agree for a constructed-and-bound model, but they come apart at both ends:

        * :meth:`~langres.core.resolver.ERModel.from_components` (which ``load``
          uses) never runs ``__init__``, so ``self.llm`` does not exist on a
          reloaded model -- reading it would raise ``AttributeError``. Identity
          lives in the slots; that is the invariant ``from_components``
          documents.
        * An *unbound* model has no slots yet (topology is built lazily, on first
          use) but the caller did name an ``llm=``. Reporting ``None`` there
          would be hiding an answer we have.

        Hence: read the slot when there is one, else fall back to the argument,
        else ``None``. ``getattr(self, "llm", None)`` rather than ``self.llm``
        precisely because the reloaded model has no such attribute.
        """
        escalation = getattr(self._module, "escalation", None)
        model = getattr(escalation, "model", None)
        if isinstance(model, str):
            return model
        llm: ModelRef | None = getattr(self, "llm", None)
        return None if llm is None else llm.base

    def save(self, path: str | Path) -> None:
        """Not supported yet -- fails with the reason instead of a confusing one.

        Without this override the user gets ``VectorBlocker``'s own error, which
        tells them to "construct with schema= and text_field= to persist" -- an
        instruction they cannot act on, because they never constructed the
        ``VectorBlocker``; :meth:`_topology` did. Raising here names the real
        situation at the layer the user is actually holding.

        Raises:
            NotImplementedError: Always. See the class docstring's warning for
                why, and for the named-extractor seam that would fix it.
        """
        raise NotImplementedError(
            "VectorLLMCascade cannot be saved yet: its VectorBlocker uses a "
            "text_field_extractor closure (blocking text = every comparable field "
            "concatenated) and a callable cannot round-trip through JSON config. "
            "This is a known gap inherited from the pre-W4 embedding path, not a "
            "property of your model. FuzzyString saves and loads today. To persist "
            "an embedding pipeline now, build a Resolver/ERModel directly via "
            "ERModel.from_components(...) with a VectorBlocker constructed as "
            "VectorBlocker(vector_index=..., schema=..., text_field='<field>') -- a "
            "single named field, which is serializable, unlike a closure."
        )

    def _topology(self, schema: type[BaseModel]) -> dict[str, Any]:
        """Build the four slots for ``schema``. Called once, on binding.

        Every heavy import is inside this function on purpose: ``[semantic]``
        (faiss/sentence-transformers/torch) and ``[llm]`` (litellm) must stay out
        of ``sys.modules`` for anyone who merely imports or constructs this class
        -- ``tests/test_import_budget.py`` is the gate, and proof #2b pins that
        construction alone reaches neither ``Settings`` nor litellm.
        """
        from langres.core.blockers.vector import VectorBlocker
        from langres.core.embeddings import SentenceTransformerEmbedder
        from langres.core.indexes.vector_index import FAISSIndex
        from langres.core.matchers.cascade_judge import CascadeMatcher
        from langres.core.matchers.llm_judge import LLMMatcher

        comparator: StringComparator[Any] = StringComparator.from_schema(schema)
        field_names = [spec.name for spec in comparator.feature_specs]

        def extract(entity: Any) -> str:
            """Concatenate every comparable string field into one blocking text."""
            parts = [
                str(getattr(entity, name)) for name in field_names if getattr(entity, name, None)
            ]
            return " ".join(parts)

        index = FAISSIndex(
            embedder=SentenceTransformerEmbedder(self.embedder.base), metric="cosine"
        )
        blocker = VectorBlocker(
            vector_index=index,
            schema=schema,
            text_field_extractor=extract,
            k_neighbors=self.k_neighbors,
        )
        student: EmbeddingScoreMatcher[Any] = EmbeddingScoreMatcher(threshold=self.threshold)
        escalation: LLMMatcher[Any] = LLMMatcher(
            model=to_config(self.llm), entity_noun=self.entity_noun
        )
        return {
            "blocker": blocker,
            "comparator": comparator,
            "matcher": CascadeMatcher(
                student=student, escalation=escalation, band=self.escalation_band
            ),
            "clusterer": Clusterer(threshold=self.threshold),
        }

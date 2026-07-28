"""VectorBlocker implementation for embedding-based candidate generation.

This blocker uses injected embedding and vector index providers for
efficient candidate pair generation without N² complexity. It is
schema-agnostic, accepting a schema_factory and text_field_extractor to work
with any Pydantic schema type.

The separation of embedding and indexing concerns enables:
- Swapping embedding models during optimization
- Caching embeddings between train and inference
- Testing with fake implementations (no model loading)
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, ClassVar

from langres.core.blocker import Blocker, SchemaT
from langres.core.blockers.all_pairs import (
    register_schema_idempotent,
    schema_to_factory,
)
from langres.core.groups import ERCandidateGroup
from langres.core.indexes.vector_index import VectorIndex
from langres.core.models import ERCandidate
from langres.core.named_callable import resolve_named
from langres.core.registry import (
    UnknownComponentType,
    get_component,
    get_schema,
    register,
)
from langres.core.reports import CandidateInspectionReport
from langres.core.serialization import ComponentSpec, SerializableState

logger = logging.getLogger(__name__)


#: Bound prompt names that mean "this prefix belongs on the QUERY side".
#:
#: Binding one of these index-wide and leaving ``query_prompt`` unset is a
#: coherent **symmetric** recipe, not half an asymmetric one: every record is
#: encoded with the query prefix and ``search_all`` compares those vectors to
#: each other. ``intfloat/e5-base-v2``'s model card says so in as many words --
#: *"Use 'query: ' prefix for symmetric tasks such as semantic similarity,
#: paraphrase retrieval"* (verified against the card, 2026-07-28) -- and
#: e5-base-v2 is one of the models in this repo's own embedder ladder. Warning
#: on it would fire on a documented, correct configuration.
#:
#: An allow-list, and a one-element one: ``"query"`` is what
#: ``SentenceTransformer.encode_query`` binds, so it is the name the library
#: itself reserves for the query side. Any other name is unknowable from here.
_QUERY_SIDE_PROMPT_NAMES = frozenset({"query"})


def _document_prompt_name(vector_index: Any) -> str | None:
    """The document-side prompt bound to whatever embedder ``vector_index`` owns.

    Duck-typed on purpose. :class:`~langres.core.indexes.vector_index.VectorIndex`
    is a **structural** protocol with test doubles inside and outside this repo,
    so this must never require an attribute: every lookup falls back to ``None``,
    which reads as "cannot tell" and disables the check rather than raising in a
    constructor.

    Returns:
        The embedder's ``prompt_name`` (the deliberate document-side opt-in), or
        ``None`` when there is no embedder, no ``prompt_name``, or the index does
        not expose one.
    """
    embedder = getattr(vector_index, "embedder", None)
    if embedder is None:
        # QdrantHybridIndex/QdrantHybridRerankingIndex name theirs differently:
        # the dense side is the one create_index() encodes documents with.
        embedder = getattr(vector_index, "dense_embedder", None)

    # Unwrap ONE decorator level. `DiskCachedEmbedder` holds the real embedder on
    # `.embedder` and carries no `prompt_name` of its own, so without this the
    # check would go quiet for every cached embedder -- a silent hole in a check
    # whose whole job is not to be silent. One level, not a loop: an unbounded
    # walk would follow any `.embedder` chain a caller invented.
    if getattr(embedder, "prompt_name", None) is None:
        embedder = getattr(embedder, "embedder", embedder)

    name = getattr(embedder, "prompt_name", None)
    return name if isinstance(name, str) else None


def _neighbor_columns(neighbor_row: Any, anchor: int, limit: int) -> list[int]:
    """Columns of one ``search_all`` row that are real neighbours of ``anchor``.

    Drops the anchor's self-match **by identity** (``neighbour == anchor``)
    rather than by position. Position is not safe:

    - Under an asymmetric ``query_prompt`` the query vector is re-encoded with
      the prompt while the stored document vector is not, so the anchor is not
      guaranteed to rank first — slicing ``row[1:]`` would then keep the anchor
      (yielding a degenerate self-pair) *and* drop a genuine neighbour.
    - Even symmetrically, an exact-duplicate text ties with the anchor at
      distance 0 and can take column 0, with the same consequence.

    Negative positions are dropped too. An index that finds fewer than ``k``
    neighbours pads the row with ``-1`` (``QdrantHybridIndex.search`` does this
    explicitly), and ``-1`` is a perfectly valid Python index: passing it through
    would silently fabricate a candidate pairing the anchor with the **last
    record in the corpus** at whatever similarity the padding carried.

    Args:
        neighbor_row: One row of the index's ``indices`` matrix (corpus
            positions of the anchor's nearest neighbours, nearest first).
        anchor: The querying record's own corpus position.
        limit: Maximum number of neighbour columns to return (``k_neighbors``),
            so the candidate budget stays the same whether or not the anchor
            appeared in its own result row.

    Returns:
        Column positions into ``neighbor_row``, nearest first, at most ``limit``
        of them, never including the anchor itself and never a padding slot.
    """
    columns: list[int] = []
    for column, neighbour in enumerate(neighbor_row):
        position = int(neighbour)
        if position == anchor or position < 0:
            continue
        columns.append(column)
        if len(columns) == limit:
            break
    return columns


def _index_type_name(index: object) -> str:
    """Resolve the registry type name for a vector index instance.

    The registry is keyed name->class with no reverse map, so this round-trips
    each registered name through :func:`get_component` to find the one whose
    class matches ``type(index)``.

    Args:
        index: A vector index instance whose class is registered.

    Returns:
        The registry key under which the index's class was registered.

    Raises:
        ValueError: If the index's class is not registered (Wave 2d/users must
            register concrete indexes before a VectorBlocker can serialize).
    """
    from langres.core.registry import _COMPONENT_REGISTRY

    for name in _COMPONENT_REGISTRY:
        try:
            if get_component(name) is type(index):
                return name
        except UnknownComponentType:  # pragma: no cover - name came from the map
            continue
    raise ValueError(
        f"Vector index type {type(index).__name__!r} is not registered; "
        "register it with @register(...) so the VectorBlocker can serialize it."
    )


def _index_config_dict(index: object) -> dict[str, object]:
    """Return a vector index's construction config as a plain JSON-able dict.

    Bridges the two component-config conventions: a ``config`` **property**
    returning a ``dict`` (returned as-is) and a ``config()`` **method**
    returning a Pydantic model (dumped).
    """
    from pydantic import BaseModel

    # Inspect ``config`` on the class so a property reads as non-callable and a
    # real method reads as callable (see _artifacts.component_config_dict).
    raw = index.config() if callable(getattr(type(index), "config", None)) else index.config  # type: ignore[attr-defined]
    if isinstance(raw, BaseModel):
        return raw.model_dump()
    return dict(raw)


def concat_comparable_fields(entity: Any) -> str:
    """Blocking text = every comparable string field, space-joined.

    The serializable, *named* replacement for the multi-field extractor closure
    the embedding architectures (e.g.
    :class:`~langres.architectures.vector_llm_cascade.VectorLLMCascade`) used to
    build inline. A closure cannot round-trip through JSON config -- this can,
    because it is referenced by name (see :data:`TEXT_FIELD_EXTRACTORS`).

    It reproduces that closure **byte-for-byte**: the field set is
    ``StringComparator.from_schema(...).feature_specs`` (the ``str | None`` fields
    except ``id``, in schema-declaration order), each present (truthy) field's
    value is ``str()``-cast, falsy values are skipped, and the parts are joined
    with a single space. Because a named module-level function cannot close over a
    per-schema field list, it derives the field set from ``type(entity)`` at call
    time -- identical to the closure whenever the blocker's ``schema_factory``
    builds ``schema`` instances, which it does (see
    :func:`~langres.core.blockers.all_pairs.schema_to_factory`).
    """
    from langres.core.comparators import StringComparator

    field_names = [spec.name for spec in StringComparator.from_schema(type(entity)).feature_specs]
    parts = [str(getattr(entity, name)) for name in field_names if getattr(entity, name, None)]
    return " ".join(parts)


#: Named ``text_field_extractor`` registry -- the same serialization contract as
#: ``llm_judge.RESPONSE_PARSERS`` / ``RECORD_SERIALIZERS`` (both resolved by the
#: shared :func:`~langres.core.named_callable.resolve_named`): a name here is
#: accepted as ``VectorBlocker(text_field_extractor="concat_comparable_fields")``
#: and, unlike a bare closure, round-trips in :attr:`VectorBlocker.config`. Add an
#: entry to make a multi-field extractor name-selectable and serializable.
TEXT_FIELD_EXTRACTORS: dict[str, Callable[[Any], str]] = {
    "concat_comparable_fields": concat_comparable_fields,
}


@register("vector_blocker")
class VectorBlocker(Blocker[SchemaT]):
    """Schema-agnostic blocker using embeddings and ANN search with dependency injection.

    This blocker separates embedding computation from vector indexing by
    accepting injected EmbeddingProvider and VectorIndex implementations.
    This enables:
    - Swapping embedding models during optimization
    - Caching embeddings between train and inference phases
    - Testing with fake implementations (no expensive model loading)
    - Using different vector backends (FAISS, Annoy, cloud services)

    The blocker is schema-agnostic: it works with ANY Pydantic schema by
    accepting a schema_factory (to transform raw dicts) and a
    text_field_extractor (to extract text for embedding).

    IMPORTANT: You must call vector_index.create_index(texts) BEFORE
    calling stream(data).

    Example (production with FAISS):
        from langres.core.embeddings import SentenceTransformerEmbedder
        from langres.core.indexes.vector_index import FAISSIndex

        def company_factory(record: dict) -> CompanySchema:
            return CompanySchema(
                id=record["id"],
                name=record["name"],
                address=record.get("address")
            )

        # 1. Setup embedder and index
        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        index = FAISSIndex(embedder=embedder, metric="cosine")

        # 2. Build index (one-time preprocessing)
        entities = [{"id": 1, "name": "Apple"}, ...]
        texts = [e["name"] for e in entities]
        index.create_index(texts)  # <- REQUIRED

        # 3. Create blocker with pre-built index
        blocker = VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            vector_index=index,  # Pre-built!
            k_neighbors=10
        )

        # 4. Generate candidates (fast - reuses index)
        candidates = list(blocker.stream(entities))

        # 5. Optimize k (fast - no re-indexing)
        for k in [10, 20, 30]:
            blocker.k_neighbors = k
            candidates = list(blocker.stream(entities))

    Example (production with Qdrant hybrid search):
        from qdrant_client import QdrantClient
        from langres.core.embeddings import SentenceTransformerEmbedder, FastEmbedSparseEmbedder
        from langres.core.indexes.hybrid_vector_index import QdrantHybridIndex

        client = QdrantClient(url="http://localhost:6333")
        dense_embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        sparse_embedder = FastEmbedSparseEmbedder("Qdrant/bm25")

        index = QdrantHybridIndex(
            client=client,
            collection_name="companies",
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
        )

        # Build index first
        texts = [e["name"] for e in entities]
        index.create_index(texts)

        blocker = VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            vector_index=index,
            k_neighbors=10
        )

        candidates = list(blocker.stream(entities))

    Example (asymmetric instruction recipe — EmbeddingGemma, E5, BGE, Qwen3):
        An instruction-trained checkpoint documents **two** prefixes: one for
        documents and a different one for queries. Both are expressible today,
        but they live on two different objects, so it is easy to set one and not
        the other:

        - the **document** side is bound to the embedder as ``prompt_name`` (with
          ``prompts`` when the mapping is not already in the checkpoint's
          ``config_sentence_transformers.json``). ``create_index`` encodes every
          corpus text through that embedder, and sentence-transformers resolves
          an ``encode(prompt=None)`` call back to the bound default — so the
          documents carry the prefix without ``create_index`` taking one.
        - the **query** side is ``VectorBlocker(query_prompt=...)``. It is passed
          explicitly at search time, which takes precedence over the embedder's
          default — so queries carry the query prefix, not the document one.

        ```
        embedder = SentenceTransformerEmbedder(
            "google/embeddinggemma-300m",
            prompts={
                "document": "title: none | text: ",
                "query": "task: search result | query: ",
            },
            prompt_name="document",
        )
        blocker = VectorBlocker(
            vector_index=FAISSIndex(embedder),
            query_prompt="task: search result | query: ",
            schema=Company,
            text_field="name",
        )
        ```

        Setting ``prompt_name`` and forgetting ``query_prompt`` is **worse than
        prompting neither side**: ``search_all`` then reuses the
        document-prompted corpus vectors as the queries, so the queries get the
        *document* prefix. This constructor logs a warning for exactly that
        combination. Measured evidence for whether the recipe helps at all is in
        ``docs/research/20260727_embedder_ladder.md``.

    Example (testing with fakes):
        from langres.core.indexes.vector_index import FakeVectorIndex

        index = FakeVectorIndex()

        # Build index first
        texts = [d["name"] for d in test_data]
        index.create_index(texts)

        blocker = VectorBlocker(
            schema_factory=company_factory,
            text_field_extractor=lambda x: x.name,
            vector_index=index,
            k_neighbors=10
        )

        # Instant, deterministic testing!
        candidates = list(blocker.stream(test_data))

    Why separate index creation?
        - Performance: Build once, search many times
        - Optimization: Tune k_neighbors without re-indexing
        - Clarity: Explicit preprocessing vs runtime phases

    Note:
        This blocker has O(N log N) complexity for building the index and
        O(k log N) for searching, where k is the number of neighbors. This
        is much more scalable than all-pairs O(N²) blocking.

    Note:
        Blocking recall is critical - we must not miss true matches. The
        k_neighbors parameter should be tuned to achieve >= 95% recall.
        Higher k = better recall but more candidates (and cost).
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "vector_blocker"

    def __init__(
        self,
        vector_index: VectorIndex,
        schema_factory: Callable[[dict[str, Any]], SchemaT] | None = None,
        schema: type[SchemaT] | None = None,
        text_field_extractor: Callable[[SchemaT], str] | str | None = None,
        text_field: str | None = None,
        k_neighbors: int = 10,
        query_prompt: str | None = None,
    ):
        """Initialize VectorBlocker with injected dependencies.

        Provide exactly one of ``schema``/``schema_factory`` and exactly one of
        ``text_field``/``text_field_extractor``:

        - ``schema=`` + ``text_field=`` (declarative): entities are rebuilt as
          ``schema(**{f: record.get(f) for f in schema.model_fields})`` and text
          is ``getattr(entity, text_field)``. This form is **config-serializable**.
        - ``schema=`` + ``text_field_extractor="<name>"`` (a registered name, see
          :data:`TEXT_FIELD_EXTRACTORS` -- e.g. ``"concat_comparable_fields"``):
          full multi-field extraction that is **still config-serializable**,
          because the name (not the callable) is what round-trips.
        - ``schema_factory=`` + ``text_field_extractor=<callable>`` (an
          unregistered callable): full control, but **not serializable**
          (a bare callable can't round-trip).

        The two styles can be mixed per-axis, but a blocker only serializes if
        **both** axes are declarative (a schema type + a ``text_field`` name or a
        registered ``text_field_extractor`` name).

        Args:
            vector_index: Index for ANN search on embeddings. The index owns the
                embedder. Use FAISSIndex/QdrantHybridIndex for production,
                FakeVectorIndex for testing.
            schema_factory: Callable mapping a raw dict to a SchemaT. Mutually
                exclusive with ``schema``.
            schema: Pydantic schema class for declarative reconstruction.
                Mutually exclusive with ``schema_factory``.
            text_field_extractor: Either a registered extractor *name* (see
                :data:`TEXT_FIELD_EXTRACTORS`, e.g. ``"concat_comparable_fields"``
                -- serializable) or a callable extracting embed-text from a
                SchemaT (e.g. ``lambda x: x.name`` -- opaque, not serializable).
                A registered name (or one of the registered callables) round-trips
                in :attr:`config`; an unregistered callable does not. Mutually
                exclusive with ``text_field``.
            text_field: Attribute name to read embed-text from each entity.
                Mutually exclusive with ``text_field_extractor``.
            k_neighbors: Number of nearest neighbors per entity. Higher = better
                recall, more candidates. Default: 10.
            query_prompt: Optional instruction prepended to each query at search
                time (for instructional embeddings such as EmbeddingGemma / E5 /
                BGE / Qwen3-Embedding). When set, the index re-encodes the query
                side with the prompt while the indexed documents stay generic —
                so this costs one extra encode pass over the corpus per search.
                Default: None (symmetric; no re-encode).

                **This is only half of an asymmetric recipe** — see the class
                docstring for the other half and the full worked example.

                **How far it reaches depends on the index.**
                :class:`~langres.core.indexes.vector_index.FAISSIndex` honours it
                fully — ``search_all`` re-encodes the query side.
                :class:`~langres.core.indexes.reranking_vector_index.QdrantHybridRerankingIndex`
                honours it at the stage that decides the final order: its dense
                *prefetch* is short-circuited on cached vectors, but the
                late-interaction reranking pass encodes queries with the prompt
                (``reranking_vector_index.py:294``).
                :class:`~langres.core.indexes.hybrid_vector_index.QdrantHybridIndex`
                **raises** ``NotImplementedError`` from ``search_all`` rather than
                serving a query-prompted search it cannot perform: it answers from
                the dense vectors cached at index-build time, so the prompt can
                never reach the encoder. It previously accepted and discarded the
                argument, which made a sweep over this axis return identical
                numbers at every setting — a flat result that reads as "the prompt
                does not help".

        Raises:
            ValueError: If k_neighbors is not positive, or if the schema /
                text-field arguments are not provided exactly once each.
        """
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")

        if (schema is None) == (schema_factory is None):
            raise ValueError(
                "VectorBlocker requires exactly one of 'schema' or "
                "'schema_factory' (got both or neither)."
            )
        if (text_field is None) == (text_field_extractor is None):
            raise ValueError(
                "VectorBlocker requires exactly one of 'text_field' or "
                "'text_field_extractor' (got both or neither)."
            )

        # Schema axis: declarative form records a serializable type name.
        self._schema_type_name: str | None = None
        self.schema_factory: Callable[[dict[str, Any]], SchemaT]
        if schema is not None:
            self._schema_type_name = register_schema_idempotent(schema)
            self.schema_factory = schema_to_factory(schema)
        else:
            assert schema_factory is not None  # narrowed by the guard above
            self.schema_factory = schema_factory

        # Text axis: a declarative form records what serializes -- the field NAME
        # (``text_field``) or the registered extractor NAME
        # (``text_field_extractor="..."``). A bare callable extractor stays opaque
        # (it runs, but ``_extractor_name`` is None so ``config`` refuses it).
        self._text_field: str | None = text_field
        self._extractor_name: str | None = None
        self.text_field_extractor: Callable[[SchemaT], str]
        if text_field is not None:
            field = text_field
            self.text_field_extractor = lambda entity: str(getattr(entity, field))
        else:
            assert text_field_extractor is not None  # narrowed by the guard
            self.text_field_extractor, self._extractor_name = resolve_named(
                text_field_extractor, TEXT_FIELD_EXTRACTORS, kind="text_field_extractor"
            )

        self.vector_index = vector_index
        self.k_neighbors = k_neighbors
        self.query_prompt = query_prompt

        # Coherence between the two halves of an asymmetric recipe. The two
        # prompts live on two different objects and nothing else connects them,
        # so a half-configured recipe is silent -- and one direction of it is
        # actively wrong, not merely incomplete. See the class docstring.
        document_prompt = _document_prompt_name(vector_index)
        if (
            document_prompt is not None
            and query_prompt is None
            and document_prompt not in _QUERY_SIDE_PROMPT_NAMES
        ):
            # The two remedies are NOT interchangeable, and the order matters.
            # Dropping prompt_name works on every index. Adding query_prompt only
            # works on an index that re-encodes the query side: VectorBlocker
            # forwards it to search_all(), and QdrantHybridIndex.search_all()
            # raises NotImplementedError on any non-None prompt. Leading with
            # query_prompt would hand a hybrid user a remedy that converts a
            # running configuration into a crash. (Caught by cross-model review.)
            logger.warning(
                "VectorBlocker: the index's embedder binds a document-side prompt "
                "(prompt_name=%r) but this blocker sets no query_prompt. search_all() "
                "then reuses the DOCUMENT-prompted corpus vectors as queries, so the "
                "queries carry the document prefix -- which is not what an asymmetric "
                "checkpoint documents, and is worse than prompting neither side. Drop "
                "prompt_name from the embedder to run both sides bare (works with any "
                "index); or, if your index re-encodes queries, pass the checkpoint's "
                "query prefix as query_prompt=... -- FAISSIndex does, "
                "QdrantHybridIndex.search_all() does not and will raise.",
                document_prompt,
            )

    @property
    def config(self) -> dict[str, object]:
        """Serializable construction config for the registry.

        Returns a mapping with the schema type name, the text axis (either
        ``text_field`` -- a field name -- or ``text_field_extractor`` -- a
        registered extractor name; the unused one is ``None``), the
        ``k_neighbors`` / ``query_prompt`` knobs, and the vector index nested as
        a :class:`~langres.core.serialization.ComponentSpec` (the index's own
        ``type_name`` + ``config``). The index's out-of-band state (e.g. a built
        FAISS index) is persisted separately via
        :class:`~langres.core.serialization.SerializableState`, not here.

        Raises:
            ValueError: If the blocker was built with a ``schema_factory`` or an
                *unregistered* ``text_field_extractor`` callable (opaque callables
                can't round-trip). Construct with ``schema=`` and either
                ``text_field=`` or a registered ``text_field_extractor`` name to
                persist.
        """
        if self._schema_type_name is None or (
            self._text_field is None and self._extractor_name is None
        ):
            raise ValueError(
                "VectorBlocker built with 'schema_factory' or an unregistered "
                "'text_field_extractor' callable is not serializable (callables "
                "cannot round-trip through config); construct with schema= and "
                "either text_field='<field>' or a registered text_field_extractor "
                "name (see TEXT_FIELD_EXTRACTORS, e.g. 'concat_comparable_fields') "
                "to persist."
            )

        index_type = _index_type_name(self.vector_index)
        index_spec = ComponentSpec(
            type_name=index_type,
            config=_index_config_dict(self.vector_index),
        )
        return {
            "schema_type_name": self._schema_type_name,
            "text_field": self._text_field,
            "text_field_extractor": self._extractor_name,
            "k_neighbors": self.k_neighbors,
            "query_prompt": self.query_prompt,
            "vector_index": index_spec,
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, object],
        state_dir: Path | None = None,
    ) -> "VectorBlocker[SchemaT]":
        """Rebuild a VectorBlocker from its serialized config.

        Reconstructs the nested vector index via the registry
        (``get_component(type_name).from_config(...)``) and, when the index
        implements :class:`~langres.core.serialization.SerializableState` and a
        ``state_dir`` is given, restores its out-of-band state.

        Args:
            config: A mapping as produced by :attr:`config` (the
                ``"vector_index"`` value may be a ``ComponentSpec`` or its
                ``model_dump()`` dict). Exactly one of ``"text_field"`` /
                ``"text_field_extractor"`` is non-``None``; a pre-named-extractor
                artifact carries only ``"text_field"``.
            state_dir: Directory holding the index's persisted state. Required
                only when the reconstructed index is a ``SerializableState``.

        Returns:
            A VectorBlocker equivalent to the one that produced ``config``.
        """
        schema = get_schema(str(config["schema_type_name"]))
        index_spec = config["vector_index"]
        if not isinstance(index_spec, ComponentSpec):
            index_spec = ComponentSpec.model_validate(index_spec)

        index_cls: Any = get_component(index_spec.type_name)
        index_config_model = getattr(index_cls, "config_model", None)
        if index_config_model is not None:
            index = index_cls.from_config(index_config_model.model_validate(index_spec.config))
        else:
            index = index_cls.from_config(index_spec.config)
        if isinstance(index, SerializableState) and state_dir is not None:
            index.load_state(state_dir)

        # Rebuild whichever text axis was declarative. ``.get`` (not ``[]``) for
        # ``text_field_extractor`` so a pre-named-extractor artifact -- which has
        # no such key -- loads via ``text_field`` unchanged.
        text_field = config["text_field"]
        extractor_name = config.get("text_field_extractor")
        return cls(
            vector_index=index,
            schema=schema,  # type: ignore[arg-type]
            text_field=str(text_field) if text_field is not None else None,
            text_field_extractor=str(extractor_name) if extractor_name is not None else None,
            k_neighbors=int(config["k_neighbors"]),  # type: ignore[call-overload]
            query_prompt=(
                str(config["query_prompt"]) if config["query_prompt"] is not None else None
            ),
        )

    def _index_is_built(self) -> bool:
        """Check if vector index has been built.

        Returns:
            True if index is ready for search, False otherwise.
        """
        # Check for FAISSIndex
        if hasattr(self.vector_index, "_index"):
            return self.vector_index._index is not None

        # Check for QdrantHybridIndex / QdrantHybridRerankingIndex
        if hasattr(self.vector_index, "_corpus_texts"):
            return self.vector_index._corpus_texts is not None

        # Check for fake indexes (test doubles)
        if hasattr(self.vector_index, "_n_samples"):
            return self.vector_index._n_samples is not None

        return False

    def stream(self, data: list[Any]) -> Iterator[ERCandidate[SchemaT]]:
        """Generate candidate pairs using embedding similarity and ANN search.

        Args:
            data: List of raw data items (typically dicts). The schema_factory
                transforms these into SchemaT objects.

        Yields:
            ERCandidate[SchemaT] objects containing:
            - left: Normalized entity (SchemaT)
            - right: Normalized entity (SchemaT)
            - blocker_name: "vector_blocker"

        Raises:
            RuntimeError: If index has not been built via create_index().

        Note:
            You must call vector_index.create_index(texts) before calling
            this method. Extract texts in the same order as data records.

        Note:
            This implementation:
            1. Normalizes raw data to SchemaT using schema_factory
            2. Searches pre-built index for k nearest neighbors
            3. Yields deduplicated pairs (no both (a,b) and (b,a))

        Note:
            Empty datasets or single-entity datasets produce no candidates.

        Example:
            texts = [record['name'] for record in data]
            blocker.vector_index.create_index(texts)
            candidates = list(blocker.stream(data))
        """
        # Validate index is built
        if not self._index_is_built():
            raise RuntimeError(
                "Index not built. Call vector_index.create_index(texts) "
                "before blocker.stream(data).\n\n"
                "Example:\n"
                "    texts = [record['name'] for record in data]\n"
                "    blocker.vector_index.create_index(texts)\n"
                "    candidates = list(blocker.stream(data))"
            )

        # Handle empty dataset
        if len(data) == 0:
            return

        # 1. Normalize schema: transform raw dicts to SchemaT
        entities = [self.schema_factory(record) for record in data]

        # Handle single entity (no pairs possible)
        if len(entities) <= 1:
            return

        # 4. Search for k nearest neighbors for each entity (deduplication pattern)
        # k+1 leaves room for the entity's own self-match, which _neighbor_columns
        # then drops by identity (it is usually, but not always, at column 0).
        k = min(self.k_neighbors + 1, len(entities))
        distances, indices = self.vector_index.search_all(k, query_prompt=self.query_prompt)

        # 5. Convert distances to similarity scores in [0, 1] (1.0 = most similar).
        # The vector index owns this conversion because only it knows its metric
        # (e.g. FAISS L2 squared distances vs. cosine inner products vs. Qdrant
        # fusion scores). Delegating here avoids guessing the metric blocker-side.
        similarities = self.vector_index.to_similarities(distances)

        # 6. Generate pairs from nearest neighbors
        # Use a set to track seen pairs and avoid duplicates
        seen_pairs: set[frozenset[str]] = set()

        for i in range(
            len(entities)
        ):  # TODO: instead of loop is there a vector operation that can speed things up?
            # Drop entity i's own self-match by identity, not by position (see
            # _neighbor_columns: a query_prompt or a duplicate text can move it).
            columns = _neighbor_columns(indices[i], i, self.k_neighbors)

            for j, similarity in ((int(indices[i][c]), similarities[i][c]) for c in columns):
                # Create a canonical pair representation (order-independent)
                pair_key = frozenset([entities[i].id, entities[j].id])  # type: ignore[attr-defined]

                # Skip if we've already seen this pair
                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)

                # Yield the candidate pair with consistent ordering (i < j)
                if i < j:
                    yield ERCandidate(
                        left=entities[i],
                        right=entities[j],
                        blocker_name="vector_blocker",
                        similarity_score=float(similarity),
                    )
                else:
                    yield ERCandidate(
                        left=entities[j],
                        right=entities[i],
                        blocker_name="vector_blocker",
                        similarity_score=float(similarity),
                    )

    def stream_groups(self, data: list[Any]) -> Iterator[ERCandidateGroup[SchemaT]]:
        """Generate anchor + k-nearest-neighbor groups natively (W1.0, E3).

        Unlike the base ``Blocker.stream_groups()`` default (which derives
        groups from a pairwise stream and is anchor-skewed), this is a NATIVE
        implementation: VectorBlocker's kNN search is already per-anchor, so
        each entity's own search result IS its group -- no derivation, no
        skew. One group is yielded per entity, with its (deduplicated) k
        nearest neighbors as members.

        Args:
            data: List of raw data items (typically dicts). The schema_factory
                transforms these into SchemaT objects.

        Yields:
            ERCandidateGroup[SchemaT] objects containing:
            - anchor: One entity from ``data``
            - members: Its nearest neighbors not already claimed by an
                earlier anchor's group (see the dedup note below)
            - group_id: The anchor's ``id``

        Raises:
            RuntimeError: If index has not been built via create_index().

        Note:
            You must call vector_index.create_index(texts) before calling
            this method, exactly as for stream().

        Note:
            Empty datasets or single-entity datasets produce no groups.

        Note:
            Cross-anchor dedup, matching stream()'s semantics exactly: a
            single ``seen_pairs`` set is threaded across all entities (same
            iteration order, same first-seen-wins rule stream() uses), so
            each undirected pair is assigned to exactly ONE group -- whichever
            anchor is processed first. Without this, two mutual nearest
            neighbors (A's nearest neighbor is B AND B's nearest neighbor is
            A -- common with real ANN indexes on near-duplicate records) would
            otherwise appear as a member edge in BOTH groups, and a consumer
            issuing one call per group (e.g. a future SelectMatcher) would
            emit and charge for the same undirected pair twice.

            The pairs recoverable by flattening these groups back to
            (anchor, member) edges are therefore exactly the pairs stream()
            would yield, with NO duplicates and NO losses (CEO #14 + E5;
            verified by property test, see tests/core/blockers/test_vector.py).
        """
        if not self._index_is_built():
            raise RuntimeError(
                "Index not built. Call vector_index.create_index(texts) "
                "before blocker.stream_groups(data).\n\n"
                "Example:\n"
                "    texts = [record['name'] for record in data]\n"
                "    blocker.vector_index.create_index(texts)\n"
                "    groups = list(blocker.stream_groups(data))"
            )

        if len(data) == 0:
            return

        entities = [self.schema_factory(record) for record in data]

        if len(entities) <= 1:
            return

        k = min(self.k_neighbors + 1, len(entities))
        _distances, indices = self.vector_index.search_all(k, query_prompt=self.query_prompt)

        seen_pairs: set[frozenset[str]] = set()
        for i in range(len(entities)):
            # Same identity-based self-match drop as stream() (see _neighbor_columns).
            members = []
            for j in (
                int(indices[i][c]) for c in _neighbor_columns(indices[i], i, self.k_neighbors)
            ):
                pair_key = frozenset([entities[i].id, entities[j].id])  # type: ignore[attr-defined]
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                members.append(entities[j])
            yield ERCandidateGroup(
                anchor=entities[i],
                members=members,
                group_id=entities[i].id,  # type: ignore[attr-defined]
            )

    def inspect_candidates(
        self,
        candidates: list[ERCandidate[SchemaT]],
        entities: list[SchemaT],
        sample_size: int = 10,
    ) -> CandidateInspectionReport:
        """Explore candidates without ground truth labels.

        Use this method to understand VectorBlocker output before labeling.
        Provides statistics, distribution, examples, and k_neighbors tuning
        recommendations.

        Args:
            candidates: List of candidate pairs generated by blocker
            entities: List of normalized entities (for readable text extraction)
            sample_size: Number of examples to include in report (default: 10)

        Returns:
            CandidateInspectionReport with VectorBlocker-specific recommendations.
        """
        # Handle empty cases
        if len(candidates) == 0:
            return CandidateInspectionReport(
                total_candidates=0,
                avg_candidates_per_entity=0.0,
                candidate_distribution={},
                examples=[],
                recommendations=[
                    "No candidates generated - check data and k_neighbors parameter",
                    "Consider increasing k_neighbors if you have enough entities",
                ],
            )

        # Compute total candidates and average per entity
        total_candidates = len(candidates)
        num_entities = len(entities)
        avg_candidates_per_entity = (
            total_candidates * 2 / num_entities if num_entities > 0 else 0.0
        )  # *2 because each candidate involves 2 entities

        # Build distribution histogram
        # Count how many candidates each entity appears in
        entity_candidate_count: dict[str, int] = {}
        for candidate in candidates:
            left_id = candidate.left.id  # type: ignore[attr-defined]
            right_id = candidate.right.id  # type: ignore[attr-defined]
            entity_candidate_count[left_id] = entity_candidate_count.get(left_id, 0) + 1
            entity_candidate_count[right_id] = entity_candidate_count.get(right_id, 0) + 1

        # Create histogram buckets
        distribution: dict[str, int] = {
            "1-3": 0,
            "4-6": 0,
            "7-9": 0,
            "10+": 0,
        }

        for count in entity_candidate_count.values():
            if count <= 3:
                distribution["1-3"] += 1
            elif count <= 6:
                distribution["4-6"] += 1
            elif count <= 9:
                distribution["7-9"] += 1
            else:
                distribution["10+"] += 1

        # Sample examples with readable text
        examples = []
        for candidate in candidates[:sample_size]:
            left_text = self.text_field_extractor(candidate.left)
            right_text = self.text_field_extractor(candidate.right)
            examples.append(
                {
                    "left_id": candidate.left.id,  # type: ignore[attr-defined]
                    "right_id": candidate.right.id,  # type: ignore[attr-defined]
                    "left_text": left_text,
                    "right_text": right_text,
                }
            )

        # Generate recommendations based on statistics
        recommendations = []
        if avg_candidates_per_entity < 3:
            recommendations.append(
                f"Low candidate count (avg {avg_candidates_per_entity:.1f} per entity) - "
                f"increase k_neighbors (current: {self.k_neighbors}) for better recall"
            )
        elif avg_candidates_per_entity > 8:
            recommendations.append(
                f"High candidate count (avg {avg_candidates_per_entity:.1f} per entity) - "
                f"decrease k_neighbors (current: {self.k_neighbors}) to reduce false positives"
            )
        else:
            recommendations.append(
                f"Candidate count looks reasonable (avg {avg_candidates_per_entity:.1f} per entity)"
            )

        # Add threshold suggestion based on distribution
        if avg_candidates_per_entity > 0:
            suggested_k = int(avg_candidates_per_entity / 2)
            if suggested_k != self.k_neighbors:
                recommendations.append(
                    f"Consider trying k_neighbors={suggested_k} based on current distribution"
                )

        return CandidateInspectionReport(
            total_candidates=total_candidates,
            avg_candidates_per_entity=avg_candidates_per_entity,
            candidate_distribution=distribution,
            examples=examples,
            recommendations=recommendations,
        )

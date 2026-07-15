"""Convenience builders: compose the default section set with sensible defaults.

The Wave 2 seam :class:`~langres.data.data_profile.base.DataProfileReport`'s two
convenience constructors (:meth:`~DataProfileReport.from_benchmark` /
:meth:`~DataProfileReport.from_records`) delegate to. Everything here is pure
composition over the Wave-1 profiler functions -- it computes nothing new. Both
entry points converge on one internal assembler (:func:`_assemble`) so the two
paths return the *same* pinned section layout:

    hero -> label-structure -> separability -> [mining-readiness] ->
    corpus-fields -> embeddings -> embedding-comparison

Each section is included only when its input is present; omitting an input drops
that section silently (never a raise). Mining readiness is the one section this
package does not *compute* -- it is a precomputed
:class:`~langres.data.data_profile.mining_readiness.MiningReadinessSection` the
caller assembles from the miners (which need scikit-learn) and passes in, keeping
this package import-light. ``include=`` narrows the set further -- it
is a **selector of section kinds** (it validates its keys and raises on an unknown
one, because a typo there means "select nothing you meant to"), never a data
input.

**Import-light.** Module scope is stdlib + numpy + the leaf profiler modules --
no ``[semantic]`` stack. :func:`from_benchmark` imports ``get_benchmark`` locally
(only when handed a name), and :func:`from_embedder` -- the one place vectors are
*produced* rather than consumed -- imports ``sentence-transformers`` inside its
body, so a bare ``import langres.data.data_profile`` never pulls a heavy dep.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Collection, Hashable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from langres.data.data_profile.base import DataProfileReport, ProfileSection
from langres.data.data_profile.corpus_field import profile_corpus_fields
from langres.data.data_profile.embedding_section import (
    profile_embedding,
    profile_embedding_comparison,
)
from langres.data.data_profile.embedding_source import (
    EmbeddingSource,
    NpySource,
    cosine_signal,
)
from langres.data.data_profile.hero import build_hero
from langres.data.data_profile.label_structure import profile_label_structure
from langres.data.data_profile.mining_readiness import MiningReadinessSection
from langres.data.data_profile.separability import (
    SeparabilitySection,
    profile_separability,
    string_signal,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from langres.core.benchmark import Benchmark

logger = logging.getLogger(__name__)

#: The section kinds ``include=`` may select. A superset of what any single call
#: emits (a call only ever produces the kinds its inputs support); ``include=``
#: filters that produced set. Validated so an unknown key fails loudly.
_KNOWN_KINDS: frozenset[str] = frozenset(
    {
        "hero",
        "label_structure",
        "mining_readiness",
        "separability",
        "corpus_field",
        "embedding",
        "embedding_comparison",
    }
)

#: Default cap on how many negative (non-matching) pairs are sampled per report
#: for the separability chart. Bounds the scan on a huge corpus; logged when it
#: bites (never silent). Positives come from the gold clusters directly.
_DEFAULT_NEGATIVES_CAP = 50_000

#: Seed for the deterministic negative-pair sample (reproducible profiles).
_NEGATIVES_SEED = 0


# ---------------------------------------------------------------------------
# Public convenience constructors
# ---------------------------------------------------------------------------


def from_benchmark(
    benchmark_or_name: "str | Benchmark[Any]",
    *,
    include: Collection[str] | None = None,
    embeddings: Sequence[EmbeddingSource] | None = None,
    mining_readiness: MiningReadinessSection | None = None,
    negatives_cap: int = _DEFAULT_NEGATIVES_CAP,
    top_n_fields: int = 50,
    seed: int = _NEGATIVES_SEED,
) -> DataProfileReport:
    """Profile a registered benchmark (or a benchmark object) with sensible defaults.

    Resolves ``benchmark_or_name`` (via :func:`~langres.data.registry.get_benchmark`
    when given a name), loads its full corpus + gold clustering, and composes the
    default section set over them: a KPI hero, label structure, separability
    (rapidfuzz ``string_signal`` by default, plus a cosine signal per embedding
    source when ``embeddings=`` is given), corpus fields, one embedding section per
    source, and an embedding comparison when two or more sources are passed.

    Args:
        benchmark_or_name: A registered benchmark name (e.g. ``"abt_buy"``) or an
            already-built benchmark object exposing ``load()`` (and ``schema`` for
            the string separability signal).
        include: Optional selector of section *kinds* to keep (see
            :data:`_KNOWN_KINDS`). ``None`` keeps every section the inputs support.
            An unknown kind raises :class:`ValueError`.
        embeddings: Optional precomputed
            :class:`~langres.data.data_profile.embedding_source.EmbeddingSource` s
            (aligned to the corpus ids). Omitting them drops the embedding sections
            -- never a raise.
        mining_readiness: Optional precomputed
            :class:`~langres.data.data_profile.mining_readiness.MiningReadinessSection`
            (built by the caller from the miners -- this package runs no matcher).
            Omitting it drops the mining-readiness section.
        negatives_cap: Cap on sampled non-matching pairs for the separability
            chart (logged when it truncates).
        top_n_fields: Row cap for the corpus-field table.
        seed: Seed for the deterministic negative-pair sample.

    Returns:
        A :class:`DataProfileReport` over the pinned default section layout.

    Raises:
        ValueError: If ``include`` names a kind not in :data:`_KNOWN_KINDS`.
    """
    if isinstance(benchmark_or_name, str):
        # Local import: only a name lookup needs the registry, and keeping it out
        # of module scope preserves the import-light budget of this package.
        from langres.data.registry import get_benchmark

        bench: Any = get_benchmark(benchmark_or_name)
    else:
        bench = benchmark_or_name

    corpus, gold_clusters, _gold_pairs = bench.load()
    records = [record.model_dump() for record in corpus]
    schema = getattr(bench, "schema", None)

    return _assemble(
        records=records,
        clusters=gold_clusters,
        schema=schema,
        embeddings=embeddings,
        include=include,
        negatives_cap=negatives_cap,
        top_n_fields=top_n_fields,
        seed=seed,
        id_key="id",
        mining_readiness=mining_readiness,
    )


def from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    gold: Sequence[Collection[Hashable]] | None = None,
    schema: "type[BaseModel] | None" = None,
    embeddings: Sequence[EmbeddingSource] | None = None,
    mining_readiness: MiningReadinessSection | None = None,
    include: Collection[str] | None = None,
    id_key: str = "id",
    negatives_cap: int = _DEFAULT_NEGATIVES_CAP,
    top_n_fields: int = 50,
    seed: int = _NEGATIVES_SEED,
) -> DataProfileReport:
    """Profile raw records (+ optional gold / embeddings) -- the BYO-data counterpart.

    The bring-your-own-data twin of :func:`from_benchmark`, composing the same
    pinned section set over records you hand in directly. Every input beyond
    ``records`` is optional and drops its section(s) when omitted (never a raise):
    no ``gold`` -> no label-structure or separability; no ``schema`` -> no string
    separability; no ``embeddings`` -> no embedding sections.

    Args:
        records: The corpus as a sequence of field mappings (dicts). Pydantic
            records should be dumped first (``[r.model_dump() for r in ...]``).
        gold: Optional gold clustering -- a sequence of clusters, each a collection
            of record ids. Drives label structure and separability positives.
        schema: Optional Pydantic entity schema; enables the rapidfuzz
            ``string_signal`` separability.
        embeddings: Optional precomputed
            :class:`~langres.data.data_profile.embedding_source.EmbeddingSource` s.
        mining_readiness: Optional precomputed
            :class:`~langres.data.data_profile.mining_readiness.MiningReadinessSection`
            (see :func:`from_benchmark`).
        include: Optional selector of section *kinds* (see :func:`from_benchmark`).
        id_key: The record field holding the id used for pairs / embedding
            alignment (default ``"id"``).
        negatives_cap: Cap on sampled non-matching pairs for separability.
        top_n_fields: Row cap for the corpus-field table.
        seed: Seed for the deterministic negative-pair sample.

    Returns:
        A :class:`DataProfileReport` over the pinned default section layout.

    Raises:
        ValueError: If ``include`` names a kind not in :data:`_KNOWN_KINDS`.
    """
    return _assemble(
        records=[dict(record) for record in records],
        clusters=gold,
        schema=schema,
        embeddings=embeddings,
        include=include,
        negatives_cap=negatives_cap,
        top_n_fields=top_n_fields,
        seed=seed,
        id_key=id_key,
        mining_readiness=mining_readiness,
    )


def from_embedder(
    records: Sequence[Mapping[str, Any]],
    model: str,
    *,
    out_path: str | Path,
    id_key: str = "id",
    text_key: str | None = None,
    batch_size: int = 64,
    normalize: bool = False,
) -> NpySource:
    """Embed ``records`` once with a sentence-transformer and persist an ``NpySource``.

    The ``[semantic]``-gated on-ramp for users who have *no* precomputed vectors
    yet. It is a **separate** step from the report itself: the report still never
    generates embeddings -- this helper produces a ``.npy`` (+ an ids sidecar) once,
    then hands back a memory-efficient
    :class:`~langres.data.data_profile.embedding_source.NpySource` you pass into
    ``from_benchmark``/``from_records`` like any other source. Mirrors
    ``langres.eval.candidates_for``: a paid/heavy step kept off the report's
    import-light, ``$0`` path.

    Args:
        records: The corpus as field mappings; one embedding per record.
        model: A sentence-transformers model id (e.g. ``"all-MiniLM-L6-v2"``);
            also the resulting source's ``name``.
        out_path: Where to write the ``.npy`` (a ``.npy`` suffix is enforced). An
            ``<stem>.ids.json`` sidecar is written beside it so the source
            self-aligns on a later bare ``NpySource(model, out_path)``.
        id_key: Record field holding the id (default ``"id"``).
        text_key: Record field to embed. When ``None`` (default), every non-empty
            string field except ``id_key`` is joined into the embedding text.
        batch_size: Encoder batch size.
        normalize: L2-normalize the vectors (a cosine store); marks the source
            ``pre_normalized`` with ``metric="cosine"``.

    Returns:
        An :class:`~langres.data.data_profile.embedding_source.NpySource` over the
        written vectors, aligned to the records' ids.

    Raises:
        ImportError: If the ``[semantic]`` extra (sentence-transformers) is not
            installed -- with an actionable ``pip install`` hint.
        KeyError: If a record is missing ``id_key``.
    """
    try:
        # Local import: producing vectors is the one heavy step here; keeping it
        # inside the body preserves the module's import-light budget so a bare
        # ``import langres.data.data_profile`` never pulls sentence-transformers.
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised only without [semantic]
        raise ImportError(
            "from_embedder needs the [semantic] extra (sentence-transformers) to "
            "generate embeddings. Install it: pip install langres[semantic]. "
            "(The report itself never generates embeddings -- pass precomputed "
            "vectors via ArraySource/NpySource to skip this step.)"
        ) from exc

    dict_records = [dict(record) for record in records]
    ids: list[Hashable] = [record[id_key] for record in dict_records]
    texts = [_embed_text(record, id_key=id_key, text_key=text_key) for record in dict_records]

    encoder = SentenceTransformer(model)
    matrix = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
    )

    out = Path(out_path)
    if out.suffix != ".npy":
        out = out.with_suffix(".npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, matrix)
    # Sidecar so a later ``NpySource(model, out)`` (no id_order) self-aligns.
    out.with_name(f"{out.stem}.ids.json").write_text(
        json.dumps([str(identifier) for identifier in ids])
    )

    return NpySource(
        model,
        out,
        ids,
        pre_normalized=normalize,
        metric="cosine" if normalize else None,
    )


# ---------------------------------------------------------------------------
# Internal assembly (shared by both public constructors)
# ---------------------------------------------------------------------------


def _assemble(
    *,
    records: list[dict[str, Any]],
    clusters: Sequence[Collection[Hashable]] | None,
    schema: "type[BaseModel] | None",
    embeddings: Sequence[EmbeddingSource] | None,
    include: Collection[str] | None,
    negatives_cap: int,
    top_n_fields: int,
    seed: int,
    id_key: str,
    mining_readiness: MiningReadinessSection | None = None,
) -> DataProfileReport:
    """Compose the pinned default section set; ``include=`` narrows it.

    Builds each section only when its input is present, in the pinned order, then
    prepends the derived hero and applies the ``include=`` kind filter. A
    ``mining_readiness`` section, when supplied, is a **precomputed** input the
    caller assembled from the miners (this package never runs a matcher), inserted
    after the label-structure / separability blocks.
    """
    sources = list(embeddings or [])
    id_map = _index_by_id(records, id_key)
    corpus_ids = list(id_map)

    body: list[ProfileSection] = []

    label = profile_label_structure(clusters, n_records=len(records))
    if label is not None:
        body.append(label)

    body.extend(
        _build_separability(
            id_map=id_map,
            schema=schema,
            clusters=clusters,
            sources=sources,
            negatives_cap=negatives_cap,
            seed=seed,
        )
    )

    if mining_readiness is not None:
        body.append(mining_readiness)

    fields = profile_corpus_fields(records, top_n=top_n_fields)
    if fields is not None:
        body.append(fields)

    for source in sources:
        body.append(profile_embedding(source, corpus_ids))
    if len(sources) >= 2:
        body.append(profile_embedding_comparison(sources, corpus_ids))

    hero = build_hero(body)
    sections: list[ProfileSection] = ([hero] if hero is not None else []) + body

    if include is not None:
        _validate_include(include)
        sections = [section for section in sections if section.kind in include]

    return DataProfileReport(sections)


def _index_by_id(
    records: Sequence[Mapping[str, Any]],
    id_key: str,
) -> dict[Hashable, Mapping[str, Any]]:
    """Index ``records`` by their ``id_key``, **first-wins** on a duplicate id.

    A duplicate id cannot be addressed twice, so -- matching the embedding side's
    :class:`~langres.data.data_profile.embedding_source._RowGatherSource`
    convention -- the first occurrence wins and later duplicates are dropped
    (a naive ``{r[id_key]: r ...}`` comprehension would silently keep the *last*).
    Records missing ``id_key`` are skipped. Any dropped duplicate is logged with
    its count -- never silent, since a wrong id map corrupts every id-keyed lookup
    (e.g. the separability signal) downstream.
    """
    id_map: dict[Hashable, Mapping[str, Any]] = {}
    n_duplicate_ids = 0
    for record in records:
        if id_key not in record:
            continue
        identifier = record[id_key]
        if identifier in id_map:
            n_duplicate_ids += 1  # first-wins: keep the existing, drop this one
            continue
        id_map[identifier] = record
    if n_duplicate_ids:
        logger.warning(
            "data-profile: dropped %d duplicate %r id(s) (kept the first occurrence "
            "of each); a last-wins map would silently corrupt id-keyed lookups",
            n_duplicate_ids,
            id_key,
        )
    return id_map


def _build_separability(
    *,
    id_map: Mapping[Hashable, Mapping[str, Any]],
    schema: "type[BaseModel] | None",
    clusters: Sequence[Collection[Hashable]] | None,
    sources: Sequence[EmbeddingSource],
    negatives_cap: int,
    seed: int,
) -> list[SeparabilitySection]:
    """Build the separability section(s): string by default, cosine per source.

    Needs a gold clustering (for the positive pairs) -- returns ``[]`` without one.
    Positives are within-cluster pairs (reservoir-sampled to ``negatives_cap`` so a
    pathological gold cluster cannot blow up memory); negatives are a deterministic
    numpy sample of non-matching pairs (capped, logged), excluded via an
    ``id -> cluster`` map rather than a materialised positive-pair set. A
    string-similarity section is added when a ``schema`` is available, and one
    cosine-similarity section per embedding source, each named distinctly so their
    titles never collide.
    """
    if clusters is None:
        return []

    positives = _positive_pairs(clusters, cap=negatives_cap, seed=seed)
    member_cluster = _member_cluster_index(clusters)
    negatives = _sample_negative_pairs(list(id_map), member_cluster, negatives_cap, seed)
    if not positives and not negatives:
        return []

    out: list[SeparabilitySection] = []
    if schema is not None:
        section = profile_separability(
            positives, negatives, string_signal(id_map, schema), name="string", cap=negatives_cap
        )
        if section is not None:
            out.append(section)
    for source in sources:
        section = profile_separability(
            positives,
            negatives,
            cosine_signal(source),
            name=f"cosine · {source.name}",
            cap=negatives_cap,
        )
        if section is not None:
            out.append(section)
    return out


def _positive_pairs(
    clusters: Sequence[Collection[Hashable]],
    *,
    cap: int = _DEFAULT_NEGATIVES_CAP,
    seed: int = _NEGATIVES_SEED,
) -> list[tuple[Hashable, Hashable]]:
    """Up to ``cap`` within-cluster id pairs (the ER positives); singletons contribute none.

    Every within-cluster pair is a positive, but one pathological gold cluster (a
    bad-merge "default entity" of thousands of members) yields ``O(size**2)`` pairs
    -- millions of tuples -- which would defeat the package's memory-efficiency
    contract. So the pairs are **reservoir-sampled to** ``cap`` *during* generation
    (``O(cap)`` memory, seeded for reproducibility), and any truncation is logged --
    never silent, mirroring the negatives. A corpus small enough to stay under
    ``cap`` yields the exact same full pair set an un-capped enumeration would;
    ``cap`` bites only on the pathological case. ``cap`` matches the per-class
    scoring cap
    :func:`~langres.data.data_profile.separability.profile_separability` applies, so
    no positive information is lost (it subsamples to ``cap`` for scoring anyway).
    """
    reservoir: list[tuple[Hashable, Hashable]] = []
    n_seen = 0
    rng = np.random.default_rng(seed)
    for cluster in clusters:
        members = list(cluster)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                n_seen += 1
                if n_seen <= cap:
                    reservoir.append((members[i], members[j]))
                else:
                    # Reservoir sampling (Algorithm R): the n-th pair replaces a
                    # uniformly-random slot with probability cap/n, so the kept set
                    # stays a uniform sample of every pair seen -- without ever
                    # holding more than `cap` of them in memory.
                    replace = int(rng.integers(0, n_seen))
                    if replace < cap:
                        reservoir[replace] = (members[i], members[j])
    if n_seen > cap:
        logger.warning(
            "data-profile separability: sampled %d of %d positive pairs (cap=%d)",
            cap,
            n_seen,
            cap,
        )
    return reservoir


def _member_cluster_index(
    clusters: Sequence[Collection[Hashable]],
) -> dict[Hashable, int]:
    """Map each record id to its gold-cluster index (for same-cluster exclusion).

    A pair is a positive iff both ids fall in the same gold cluster. Sampling
    negatives against this ``O(n_records)`` map -- rather than a materialised set of
    every positive *pair* -- keeps the exclusion memory-linear even when one cluster
    contributes ``O(size**2)`` positives (see :func:`_positive_pairs`). First-wins
    on the (malformed) case of an id in two clusters.
    """
    index_of: dict[Hashable, int] = {}
    for index, cluster in enumerate(clusters):
        for member in cluster:
            index_of.setdefault(member, index)
    return index_of


def _sample_negative_pairs(
    ids: Sequence[Hashable],
    member_cluster: Mapping[Hashable, int],
    cap: int,
    seed: int,
) -> list[tuple[Hashable, Hashable]]:
    """A deterministic numpy sample of up to ``cap`` non-matching id pairs.

    Draws random id pairs (seeded), rejecting self-pairs, same-cluster pairs (the
    positives -- tested against ``member_cluster``, an ``id -> cluster index`` map,
    so the full positive-pair set never has to be materialised), and repeats.
    Bounded by an attempt ceiling so a corpus with few possible negatives cannot
    spin. Logs the cap it targets (never silent).
    """
    n = len(ids)
    if n < 2 or cap <= 0:
        return []
    logger.info("data-profile separability: sampling up to %d negative pairs (cap)", cap)
    rng = np.random.default_rng(seed)
    negatives: list[tuple[Hashable, Hashable]] = []
    seen: set[frozenset[Hashable]] = set()
    # A generous attempt ceiling: enough to fill the cap on a normal corpus, yet
    # a hard stop so a tiny/degenerate corpus (few possible negatives) terminates.
    max_attempts = max(cap * 20, 40)
    for _ in range(max_attempts):
        if len(negatives) >= cap:
            break
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        left, right = ids[i], ids[j]
        left_cluster = member_cluster.get(left)
        same_cluster = left_cluster is not None and left_cluster == member_cluster.get(right)
        key = frozenset((left, right))
        if len(key) != 2 or same_cluster or key in seen:
            continue
        seen.add(key)
        negatives.append((left, right))
    return negatives


def _validate_include(include: Collection[str]) -> None:
    """Reject any ``include=`` key that is not a known section kind.

    ``include`` selects kinds; a typo means "silently drop a section you wanted",
    so an unknown kind is a loud :class:`ValueError`, not a no-op.
    """
    unknown = sorted(set(include) - _KNOWN_KINDS)
    if unknown:
        raise ValueError(
            f"include= got unknown section kind(s) {unknown}; "
            f"valid kinds are {sorted(_KNOWN_KINDS)}"
        )


def _embed_text(record: Mapping[str, Any], *, id_key: str, text_key: str | None) -> str:
    """The text to embed for one record.

    Uses ``record[text_key]`` when a key is given; otherwise joins every non-empty
    string field except ``id_key`` -- a sensible generic default for a schema-free
    record.
    """
    if text_key is not None:
        value = record.get(text_key)
        return str(value) if value is not None else ""
    parts = [
        value
        for key, value in record.items()
        if key != id_key and isinstance(value, str) and value.strip()
    ]
    return " ".join(parts)

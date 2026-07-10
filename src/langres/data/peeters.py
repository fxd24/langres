"""Peeters, Steiner & Bizer LLM-for-entity-matching replication seam.

Reproduces the **$0, offline** half of *Entity Matching using Large Language
Models* (Peeters, Steiner & Bizer, arXiv 2310.11244 v4, EDBT 2025; repo
``wbsg-uni-mannheim/MatchGPT``, subdir ``LLMForEM/``): regenerate their
deterministic evaluation subset from our already-vendored DeepMatcher CSVs,
serialize each record with their exact per-dataset recipe, render their prompt,
and parse their binary Yes/No answers — so their published F1 can be reproduced
by *replaying their archived model answers*, with no API calls.

Why this is a **slice**, not a new benchmark
--------------------------------------------
Their evaluation set is a deterministic subset of the DeepMatcher ``test.csv``
we already ship (see :func:`regenerate_sample_rows`): the same two source tables,
the same labels, a fixed ``sample(random_state=42)`` downsample of the *existing*
``test`` split. It is therefore a **slice** of ``abt_buy`` / ``amazon_google``,
not a new dataset — so it is deliberately kept out of
:mod:`langres.data.registry` (whose entries are ``VectorBlocker``-based
*clustering* benchmarks that feed ``portfolio_race``). Their protocol is a fixed
**binary pair-classification** task — no blocking, no clustering, no threshold
sweep — a fundamentally different eval shape. Instead this module carries its own
small manifest + loader-factory (:data:`PEETERS_REPLICATIONS`,
:func:`get_peeters_replication`, :func:`list_peeters_replications`) mirroring
``registry.py``'s pattern, scoped to this replication.

Licensing
---------
MatchGPT ships **no LICENSE** (GitHub reports ``license: null``); langres is
Apache-2.0. No MatchGPT file is vendored or redistributed. The committed pair
lists (``datasets/<ds>/peeters_sampled_test.csv``) are *regenerated from our own
vendored CSVs* and verified once against their published sample; their archived
prompts/answers are downloaded transiently by the offline replay harness
(``examples/research/peeters_llm_em_replication.py``).

Currently registered: ``abt-buy`` and ``amazon-google`` (both verified: exact
pair-set equality against their published sample, and 100.00% / 99.51% prompt
round-trip — the amazon-google residual is a float-repr artifact in *their* gold
standard's ``price`` column, not a serializer bug). The four remaining datasets
(walmart-amazon, dblp-scholar, dblp-acm, wdc) are a single :func:`register` call
away — supply their ``serialization_fields`` recipe, id prefixes, and (for
bibliographic sets) the ``Publication`` noun + bib task prefix.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.modules.llm_judge import parse_binary_yes_no
from langres.data import _benchmark_utils as _bu

__all__ = [
    "DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX",
    "PEETERS_REPLICATIONS",
    "PeetersPairPrompt",
    "PeetersRecord",
    "PeetersReplicationSpec",
    "build_candidates",
    "build_llm_prompt_template",
    "get_peeters_replication",
    "gold_match_pairs",
    "judgements_from_answers",
    "list_peeters_replications",
    "load_peeters_records",
    "load_peeters_sample",
    "make_record_serializer",
    "parse_binary_answer",
    "regenerate_sample_rows",
    "register",
    "render_prompt",
    "render_sample_prompts",
    "serialize_record",
]

#: Task prefix for the target ``domain-complex-force`` *product* prompt design
#: (arXiv v4 Table 2). Built by MatchGPT as ``f"{DOMAIN_COMPLEX} {FORCE}\n"``.
DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX = (
    "Do the two product descriptions refer to the same real-world product? "
    "Answer with 'Yes' if they do and 'No' if they do not.\n"
)


@dataclass(frozen=True)
class PeetersReplicationSpec:
    """Everything needed to replicate one Peeters LLM-EM dataset, offline.

    Attributes:
        name: Their dataset key (e.g. ``"abt-buy"``) — also this replication's
            registry key.
        dataset_package: Importable package holding our vendored CSVs (e.g.
            ``"langres.data.datasets.abt_buy"``).
        left_prefix: Id prefix our corpus loader gives ``tableA`` records
            (``"a"`` for abt/amazon). Maps their ``abt_N``/``amazon_N`` → ``aN``.
        right_prefix: Id prefix for ``tableB`` records (``"b"`` for buy,
            ``"g"`` for google). Maps their ``buy_N``/``google_N`` → ``bN``/``gN``.
        entity_noun: Prompt noun — ``"Product"`` (product sets) or
            ``"Publication"`` (bibliographic sets).
        task_prefix: The rendered prompt prefix for the chosen prompt design.
        serialization_fields: Ordered ``(column, max_tokens)`` recipe; each field
            is truncated to its first ``max`` single-space tokens (``split(' ')``,
            **not** ``split()``) and the fields are joined by single spaces
            (reproducing their f-string).
        sample_file: Committed regenerated pair-list filename (in
            ``dataset_package``), columns ``ltable_id,rtable_id,label`` in
            regeneration order.
        source_test_file / table_left_file / table_right_file: Our vendored CSVs.
        max_positives / num_negatives / random_state: Their downsample knobs
            (keep all positives if ``<= max_positives`` else sample; sample
            ``num_negatives`` negatives; ``sample(random_state=...)``).
    """

    name: str
    dataset_package: str
    left_prefix: str
    right_prefix: str
    entity_noun: str
    task_prefix: str
    serialization_fields: tuple[tuple[str, int], ...]
    sample_file: str = "peeters_sampled_test.csv"
    source_test_file: str = "test.csv"
    table_left_file: str = "tableA.csv"
    table_right_file: str = "tableB.csv"
    max_positives: int = 250
    num_negatives: int = 1000
    random_state: int = 42


@dataclass(frozen=True)
class PeetersPairPrompt:
    """One candidate pair rendered to its exact Peeters prompt, with its label.

    Attributes:
        left_id / right_id: Source-prefixed record ids (align with
            :func:`load_peeters_records`).
        label: Gold label (``1`` = match, ``0`` = non-match).
        prompt: The full rendered prompt (prefix + serialized pair).
    """

    left_id: str
    right_id: str
    label: int
    prompt: str


# ---------------------------------------------------------------------------
# Manifest + loader-factory (the discoverability seam, mirroring registry.py).
# ---------------------------------------------------------------------------

PEETERS_REPLICATIONS: dict[str, PeetersReplicationSpec] = {}


def register(spec: PeetersReplicationSpec) -> None:
    """Add ``spec`` to :data:`PEETERS_REPLICATIONS`.

    Raises:
        ValueError: If a spec with the same ``name`` is already registered.
    """
    if spec.name in PEETERS_REPLICATIONS:
        raise ValueError(f"Peeters replication {spec.name!r} is already registered")
    PEETERS_REPLICATIONS[spec.name] = spec


def list_peeters_replications() -> list[str]:
    """Return the registered replication names, sorted."""
    return sorted(PEETERS_REPLICATIONS)


def get_peeters_replication(name: str) -> PeetersReplicationSpec:
    """Return the registered :class:`PeetersReplicationSpec` named ``name``.

    Raises:
        KeyError: If ``name`` is unknown (with a did-you-mean suggestion and the
            available names, mirroring ``registry.UnknownBenchmark``).
    """
    try:
        return PEETERS_REPLICATIONS[name]
    except KeyError:
        available = sorted(PEETERS_REPLICATIONS)
        suggestions = difflib.get_close_matches(name, available, n=3)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(
            f"Unknown Peeters replication {name!r}.{hint} "
            f"Available: {', '.join(available) or '(none registered)'}"
        ) from None


# ---------------------------------------------------------------------------
# Pure functions: serialization, prompt rendering, answer parsing.
# ---------------------------------------------------------------------------


def serialize_record(record: Mapping[str, str], fields: Sequence[tuple[str, int]]) -> str:
    """Serialize a record with Peeters' per-field whitespace-token truncation.

    For each ``(column, max_tokens)`` the field value is truncated to its first
    ``max_tokens`` **single-space** tokens (``value.split(" ")`` — deliberately
    NOT ``value.split()``, so internal runs of spaces/tabs are kept as tokens,
    matching MatchGPT's ``prep_em_tasks.ipynb``) and stripped; the results are
    joined by single spaces. This reproduces their f-string exactly — including
    the leading/trailing/collapsed spaces a missing field leaves behind (e.g. an
    empty ``price`` yields a trailing space), which is load-bearing for the
    byte-exact prompt round-trip. Values are already lowercased in the
    DeepMatcher source.

    Args:
        record: A CSV-row-shaped mapping ``column -> value``.
        fields: Ordered ``(column, max_whitespace_tokens)`` recipe.

    Returns:
        The serialized record string.
    """
    parts = [
        " ".join((record.get(column) or "").split(" ")[:max_tokens]).strip()
        for column, max_tokens in fields
    ]
    return " ".join(parts)


def render_prompt(
    left_serialized: str,
    right_serialized: str,
    *,
    task_prefix: str,
    entity_noun: str,
) -> str:
    """Render the full Peeters single-pair prompt.

    Reproduces ``prepare_examples`` in ``prep_em_tasks.ipynb``: the task prefix
    (which already ends in a newline) followed by the two quoted, serialized
    records on their own lines.

    Args:
        left_serialized: Serialized left record (see :func:`serialize_record`).
        right_serialized: Serialized right record.
        task_prefix: The prompt-design prefix (e.g.
            :data:`DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX`).
        entity_noun: ``"Product"`` or ``"Publication"``.

    Returns:
        The full rendered prompt string.
    """
    return (
        f"{task_prefix}{entity_noun} 1: '{left_serialized}'\n{entity_noun} 2: '{right_serialized}'"
    )


def parse_binary_answer(answer: str) -> int:
    """Parse a raw model answer into a binary match decision (``int`` adapter).

    Thin ``int`` wrapper over the canonical
    :func:`langres.core.modules.llm_judge.parse_binary_yes_no`, which owns the
    exact ``check_for_prediction`` semantics (strip → delete ``string.punctuation``
    → lowercase → ``"yes" in text``). There is a single code path so the offline
    ``$0`` replay validates the same parser the paid ``LLMJudge`` run uses.

    Args:
        answer: The raw model response.

    Returns:
        ``1`` (match) or ``0`` (non-match). Anything that is not a clear "yes" —
        ``"No"``, an empty string, or a refusal — parses to ``0``.
    """
    # parse_binary_yes_no is total: decision is always True/False, never None.
    decision = parse_binary_yes_no(answer).decision
    assert decision is not None  # totality invariant (documented above)
    return int(decision)


# ---------------------------------------------------------------------------
# The pair-set slice: regenerate from our CSVs / load the committed artifact.
# ---------------------------------------------------------------------------


def _read_test_rows(spec: PeetersReplicationSpec) -> list[tuple[int, int, int]]:
    """Read our vendored ``test.csv`` as ``(ltable_id, rtable_id, label)`` in file order."""
    return [
        (int(r["ltable_id"]), int(r["rtable_id"]), int(r["label"]))
        for r in _bu.read_csv_rows(spec.dataset_package, spec.source_test_file)
    ]


def regenerate_sample_rows(spec: PeetersReplicationSpec) -> list[tuple[int, int, int]]:
    """Deterministically regenerate their sampled pair set from our ``test.csv``.

    Reproduces ``downsample_deepmatcher_tasks.ipynb``: split the test gold
    standard (in file order) into positives/negatives; keep all positives when
    ``<= max_positives`` (else ``sample(max_positives, random_state=...)``); take
    ``sample(num_negatives, random_state=...)`` negatives; return
    positives-then-sampled-negatives. ``pandas.DataFrame.sample(n,
    random_state=s)`` selects positions ``RandomState(s).choice(N, n,
    replace=False)`` and returns them **in that order**, so this ``numpy``-only
    reimplementation matches their published sample *set and order* (verified)
    without a pandas dependency.

    The returned **order** matches their published sample's row order (hence the
    archived answers' JSONL line order), so a replay can align line ``i`` to
    pair ``i``.

    Args:
        spec: The replication spec.

    Returns:
        Ordered ``(ltable_id, rtable_id, label)`` rows (raw integer ids).
    """
    rows = _read_test_rows(spec)
    positives = [(left, right) for left, right, label in rows if label == 1]
    negatives = [(left, right) for left, right, label in rows if label == 0]

    kept_positives = positives
    if len(positives) > spec.max_positives:
        # Fresh RandomState per sample() call (they pass the int 42 each time, so
        # positives and negatives each get an independent RandomState(42) — not
        # one advancing RNG threaded across both). Neither shipped slice trips
        # this branch (abt-buy 206, amazon-google 234 are under the 250 cap), but
        # it is verified to match Peeters' published sample exactly on a set that
        # DOES exceed the cap: dblp-scholar (250 positives after the cap).
        pos_idx = np.random.RandomState(spec.random_state).choice(
            len(positives), size=spec.max_positives, replace=False
        )
        kept_positives = [positives[int(i)] for i in pos_idx]

    neg_idx = np.random.RandomState(spec.random_state).choice(
        len(negatives), size=spec.num_negatives, replace=False
    )
    kept_negatives = [negatives[int(i)] for i in neg_idx]

    return [(left, right, 1) for left, right in kept_positives] + [
        (left, right, 0) for left, right in kept_negatives
    ]


def load_peeters_sample(spec: PeetersReplicationSpec) -> list[tuple[str, str, int]]:
    """Load the committed regenerated pair list with source-prefixed ids.

    Reads ``spec.sample_file`` (the tracked artifact produced by
    :func:`regenerate_sample_rows`) and maps the raw ``ltable_id``/``rtable_id``
    to the ``spec.left_prefix``/``spec.right_prefix`` ids that
    :func:`load_peeters_records` keys on. Preserves the file's row order.

    Args:
        spec: The replication spec.

    Returns:
        Ordered ``(left_id, right_id, label)`` tuples (prefixed ids).
    """
    return [
        (
            f"{spec.left_prefix}{row['ltable_id'].strip()}",
            f"{spec.right_prefix}{row['rtable_id'].strip()}",
            int(row["label"]),
        )
        for row in _bu.read_csv_rows(spec.dataset_package, spec.sample_file)
    ]


def load_peeters_records(
    spec: PeetersReplicationSpec,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Load the two source tables as ``{prefixed_id -> raw CSV row}`` dicts.

    Args:
        spec: The replication spec.

    Returns:
        ``(left_records, right_records)`` — each a mapping from source-prefixed
        id to its raw CSV row dict (``column -> value``).
    """
    left = {
        f"{spec.left_prefix}{row['id'].strip()}": row
        for row in _bu.read_csv_rows(spec.dataset_package, spec.table_left_file)
    }
    right = {
        f"{spec.right_prefix}{row['id'].strip()}": row
        for row in _bu.read_csv_rows(spec.dataset_package, spec.table_right_file)
    }
    return left, right


# ---------------------------------------------------------------------------
# Bridges: sample -> rendered prompts, and replayed answers -> metric inputs.
# ---------------------------------------------------------------------------


def render_sample_prompts(spec: PeetersReplicationSpec) -> list[PeetersPairPrompt]:
    """Render every sampled pair to its exact Peeters prompt, from our records.

    Ties the committed pair-set slice, the source records, the serializer and the
    prompt template together — the reusable bridge a replay/round-trip uses. The
    output order matches the committed sample (and the archived answers' line
    order).

    Args:
        spec: The replication spec.

    Returns:
        One :class:`PeetersPairPrompt` per sampled pair, in sample order.
    """
    left_records, right_records = load_peeters_records(spec)
    prompts: list[PeetersPairPrompt] = []
    for left_id, right_id, label in load_peeters_sample(spec):
        left = serialize_record(left_records[left_id], spec.serialization_fields)
        right = serialize_record(right_records[right_id], spec.serialization_fields)
        prompt = render_prompt(
            left, right, task_prefix=spec.task_prefix, entity_noun=spec.entity_noun
        )
        prompts.append(PeetersPairPrompt(left_id, right_id, label, prompt))
    return prompts


# ---------------------------------------------------------------------------
# Live (paid) LLM-judge seam: template + record_serializer + candidates.
#
# The offline replay above validates the parser + metric wiring at $0. To run a
# *live* judge over the same slice, an ``LLMJudge`` renders each pair itself from
# ``prompt_template`` + ``record_serializer``; these three functions build those
# from a spec so the live prompt is byte-identical to the archived one (proven in
# tests). Reusable across any Peeters slice/prompt-design, not just abt-buy.
# ---------------------------------------------------------------------------


class PeetersRecord(BaseModel):
    """A single record for the live LLM-judge path: an id + its raw CSV row.

    The minimal :class:`~langres.core.models.ERCandidate` entity an ``LLMJudge``
    needs — it carries ``id`` (for the judgement) and the raw ``fields`` mapping
    the :func:`make_record_serializer` closure truncates into the prompt. Kept
    dataset-agnostic (a raw-row bag, not a typed product/publication schema) so
    one type serves every Peeters slice.
    """

    id: str
    fields: dict[str, str]


def build_llm_prompt_template(spec: PeetersReplicationSpec) -> str:
    """The ``LLMJudge.prompt_template`` (with ``{left}``/``{right}``) for ``spec``.

    Renders the spec's prompt design with the two records left as the literal
    ``{left}`` / ``{right}`` placeholders ``LLMJudge`` substitutes at judgement
    time (literal replacement, so the surrounding quotes/newlines are preserved).
    Substituting the serialized records back in reproduces :func:`render_prompt`
    exactly.
    """
    return render_prompt(
        "{left}", "{right}", task_prefix=spec.task_prefix, entity_noun=spec.entity_noun
    )


def make_record_serializer(spec: PeetersReplicationSpec) -> Callable[[PeetersRecord], str]:
    """An ``LLMJudge.record_serializer`` applying ``spec``'s per-field truncation.

    Binds ``spec.serialization_fields`` into a closure that runs
    :func:`serialize_record` over a :class:`PeetersRecord`'s raw ``fields`` — so
    the live judge serializes each record byte-identically to the offline replay.
    """
    fields = spec.serialization_fields

    def serializer(record: PeetersRecord) -> str:
        return serialize_record(record.fields, fields)

    return serializer


def build_candidates(spec: PeetersReplicationSpec) -> list[ERCandidate[PeetersRecord]]:
    """The sampled pairs as :class:`ERCandidate`\\ s, in sample order, for a live run.

    The live-path sibling of :func:`render_sample_prompts`: instead of
    pre-rendering the prompt, it wraps each record as a :class:`PeetersRecord` so
    an ``LLMJudge`` (with :func:`build_llm_prompt_template` +
    :func:`make_record_serializer`) renders it. Order matches the committed sample
    (and hence :func:`gold_match_pairs` from the same slice).
    """
    left_records, right_records = load_peeters_records(spec)
    return [
        ERCandidate(
            left=PeetersRecord(id=left_id, fields=left_records[left_id]),
            right=PeetersRecord(id=right_id, fields=right_records[right_id]),
            blocker_name="peeters_sample",
        )
        for left_id, right_id, _label in load_peeters_sample(spec)
    ]


def judgements_from_answers(
    prompts: Sequence[PeetersPairPrompt],
    answers: Sequence[str],
    *,
    decision_step: str = "peeters_replay",
) -> list[PairwiseJudgement]:
    """Turn replayed raw answers into :class:`PairwiseJudgement`\\ s for metrics.

    Each answer is parsed with :func:`parse_binary_answer` into a binary
    ``score`` (``1.0`` = match, ``0.0`` = non-match), so a threshold of ``0.5``
    over these judgements recovers the binary decision and
    ``langres.core.metrics.classify_pairs`` computes the exact pairwise
    precision/recall/F1.

    Args:
        prompts: The rendered pairs (from :func:`render_sample_prompts`), giving
            each judgement its ids.
        answers: The raw model answers, aligned 1:1 with ``prompts``.
        decision_step: Provenance tag on each judgement.

    Returns:
        One :class:`PairwiseJudgement` per pair.

    Raises:
        ValueError: If ``prompts`` and ``answers`` differ in length.
    """
    if len(prompts) != len(answers):
        raise ValueError(f"prompts ({len(prompts)}) and answers ({len(answers)}) must align 1:1")
    return [
        PairwiseJudgement(
            left_id=prompt.left_id,
            right_id=prompt.right_id,
            score=float(parse_binary_answer(answer)),
            score_type="prob_llm",
            decision_step=decision_step,
            provenance={"raw_answer": answer},
        )
        # Lengths are guaranteed equal by the explicit check above.
        for prompt, answer in zip(prompts, answers)
    ]


def gold_match_pairs(prompts: Sequence[PeetersPairPrompt]) -> set[frozenset[str]]:
    """Return the gold *match* pairs as order-independent ``frozenset`` pairs.

    The gold set for ``classify_pairs``: every sampled pair whose label is ``1``.
    Because the candidate set includes all positives, ``classify_pairs``'s ``fn``
    counts exactly the positives the model answered "no" to.

    Args:
        prompts: The rendered pairs.

    Returns:
        The set of gold match pairs.
    """
    return {frozenset({p.left_id, p.right_id}) for p in prompts if p.label == 1}


# ---------------------------------------------------------------------------
# The registered replications (both verified against the published sample).
# Recipes verified against LLMForEM/notebooks/prep_em_tasks.ipynb + the archived
# prompts (abt-buy 100.00% / amazon-google 99.51% byte-exact prompt round-trip).
# ---------------------------------------------------------------------------

register(
    PeetersReplicationSpec(
        name="abt-buy",
        dataset_package="langres.data.datasets.abt_buy",
        left_prefix="a",  # abt   (tableA) -> their abt_N
        right_prefix="b",  # buy   (tableB) -> their buy_N
        entity_noun="Product",
        task_prefix=DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX,
        # abt-buy serializes name (<=50 tokens) + price (<=5); description DROPPED.
        serialization_fields=(("name", 50), ("price", 5)),
    )
)
register(
    PeetersReplicationSpec(
        name="amazon-google",
        dataset_package="langres.data.datasets.amazon_google",
        left_prefix="a",  # amazon (tableA) -> their amazon_N
        right_prefix="g",  # google (tableB) -> their google_N
        entity_noun="Product",
        task_prefix=DOMAIN_COMPLEX_FORCE_PRODUCT_PREFIX,
        # amazon-google serializes manufacturer (<=5) + title (<=50) + price (<=5).
        serialization_fields=(("manufacturer", 5), ("title", 50), ("price", 5)),
    )
)

"""The asymmetric instruction recipe: document prompt + query prompt, together.

An instruction-trained checkpoint (EmbeddingGemma, E5, BGE, Qwen3) documents two
different prefixes — one for documents, one for queries. langres expresses that
across **two objects**: ``SentenceTransformerEmbedder(prompt_name=..., prompts=...)``
for the document side and ``VectorBlocker(query_prompt=...)`` for the query side.

That combination was long believed to be inexpressible — a merged research report
stated the recipe was "not expressible through the blocking API" because
``create_index(texts)`` takes no prompt argument. It is expressible: the document
prefix travels on the **embedder**, not through that argument. This file is the
proof, so the claim cannot quietly flip back.

Every assertion here is a *result* comparison, never a plumbing one. A
``query_prompt`` that silently did nothing already shipped once
(``FAISSIndex.search_all`` handed cached corpus vectors to ``search()``), and it
made an entire published prompt table read exactly ``0.0000``. A test that only
checks "the argument was forwarded" would have passed throughout.
"""

from __future__ import annotations

import logging
import zlib

import numpy as np
import pytest

from langres.core.blockers.vector import _QUERY_SIDE_PROMPT_NAMES, VectorBlocker
from langres.core.indexes.vector_index import FAISSIndex, FakeVectorIndex

DOCUMENT_PROMPT = "title: none | text: "
QUERY_PROMPT = "task: search result | query: "

TEXTS = [
    "apple iphone 12 128gb black",
    "samsung galaxy s21 ultra",
    "sony wh-1000xm4 wireless headphones",
    "apple iphone 13 pro 256gb",
    "dell xps 15 laptop",
]


class _PrefixEmbedder:
    """A deterministic stand-in that honours sentence-transformers' prompt rules.

    Real ``SentenceTransformer`` semantics, reproduced in eight lines so this
    file does not have to download a checkpoint to make its point:

    - an explicit ``prompt=`` wins;
    - ``prompt=None`` falls back to ``prompts[prompt_name]``
      (``default_prompt_name``, ``base/model.py::_resolve_prompt``);
    - the prompt is applied as ``prompt + text``.

    The vector is a function of the *rendered* string, so any difference in what
    was prefixed shows up as a different vector — which is what every assertion
    below actually reads.
    """

    embedding_dim = 16

    def __init__(self, prompt_name: str | None = None, prompts: dict[str, str] | None = None):
        self.prompt_name = prompt_name
        self.prompts = prompts

    def _render(self, text: str, prompt: str | None) -> str:
        if prompt is None and self.prompt_name is not None:
            prompt = (self.prompts or {}).get(self.prompt_name)
        return (prompt or "") + text

    def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
        rendered = [self._render(text, prompt) for text in texts]
        # crc32, not the builtin hash(): str hashing is salted per process
        # (PYTHONHASHSEED), so a builtin-hash embedder would make these vectors
        # -- and any future cross-process comparison of them -- irreproducible.
        return np.array(
            [
                [float((zlib.crc32(f"{shift}|{text}".encode()) % 97)) for shift in range(16)]
                for text in rendered
            ],
            dtype=np.float32,
        )


def _asymmetric_index() -> FAISSIndex:
    embedder = _PrefixEmbedder(
        prompt_name="document",
        prompts={"document": DOCUMENT_PROMPT, "query": QUERY_PROMPT},
    )
    index = FAISSIndex(embedder=embedder, metric="cosine")
    index.create_index(TEXTS)
    return index


def _bare_index(texts: list[str]) -> FAISSIndex:
    index = FAISSIndex(embedder=_PrefixEmbedder(), metric="cosine")
    index.create_index(texts)
    return index


BLOCKER_LOGGER = "langres.core.blockers.vector"


def _blocker_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Warnings from the blocker only.

    ``caplog.text`` is the *root* capture, so an unrelated WARNING from faiss or
    sentence-transformers would make a "stays silent" assertion fail for a reason
    that has nothing to do with this check -- and, worse, would make it pass for
    the wrong reason if the polarity were ever flipped.
    """
    return [
        # getMessage() already interpolates record.args -- applying `% args`
        # again would raise on the interpolated string, not just be redundant.
        record.getMessage()
        for record in caplog.records
        if record.name == BLOCKER_LOGGER and record.levelno >= logging.WARNING
    ]


class TestTheRecipeIsExpressible:
    """The shipped API reaches the documented asymmetric configuration."""

    def test_matches_hand_prefixing_the_corpus(self) -> None:
        """``prompt_name`` on the embedder == prefixing every corpus text by hand.

        The research harness prefixes the corpus itself
        (``examples/research/embedder_ladder.py::build_prompted_index``). If the
        API route did not reach the same configuration, one of the two would be
        measuring something nobody chose.
        """
        via_api_distances, via_api_indices = _asymmetric_index().search_all(
            k=3, query_prompt=QUERY_PROMPT
        )

        by_hand = _bare_index([DOCUMENT_PROMPT + text for text in TEXTS])
        by_hand_distances, by_hand_indices = by_hand.search(
            [QUERY_PROMPT + text for text in TEXTS], k=3
        )

        np.testing.assert_array_equal(via_api_indices, by_hand_indices)
        np.testing.assert_allclose(via_api_distances, by_hand_distances, atol=1e-5)

    def test_the_document_prompt_changes_the_result(self) -> None:
        """Control: without it the numbers must differ, or the test above is vacuous."""
        prompted, _ = _asymmetric_index().search_all(k=3, query_prompt=QUERY_PROMPT)
        bare, _ = _bare_index(list(TEXTS)).search_all(k=3, query_prompt=QUERY_PROMPT)

        assert not np.allclose(prompted, bare, atol=1e-6)

    def test_the_query_prompt_changes_the_result(self) -> None:
        """Control: the same, for the other half of the recipe.

        This is the assertion the shipped ``query_prompt`` no-op would have
        failed — and did fail, silently, for an entire published table.
        """
        index = _asymmetric_index()
        with_query_prompt, _ = index.search_all(k=3, query_prompt=QUERY_PROMPT)
        without, _ = index.search_all(k=3)

        assert not np.allclose(with_query_prompt, without, atol=1e-6)

    def test_queries_do_not_get_the_document_prefix(self) -> None:
        """The two sides stay separate: an explicit prompt beats the bound default.

        If ``default_prompt_name`` leaked onto the query encode, queries would
        come out as ``DOCUMENT_PROMPT + text`` and match a bare-document index —
        the double-prefix trap the ladder harness works around.
        """
        via_api, _ = _asymmetric_index().search_all(k=3, query_prompt=QUERY_PROMPT)

        doubly_prefixed = _bare_index([DOCUMENT_PROMPT + text for text in TEXTS])
        wrong, _ = doubly_prefixed.search(
            [QUERY_PROMPT + DOCUMENT_PROMPT + text for text in TEXTS], k=3
        )

        assert not np.allclose(via_api, wrong, atol=1e-6)


class TestCoherenceWarning:
    """Half a recipe is silent today; one half of it is actively wrong."""

    def test_warns_when_the_document_side_is_driven_and_the_query_side_is_not(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        index = _asymmetric_index()

        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=index,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        warnings = _blocker_warnings(caplog)
        assert len(warnings) == 1
        assert "prompt_name='document'" in warnings[0]
        # States what is observably true (both sides carry the prefix) without
        # asserting the recipe is asymmetric -- which the blocker cannot know.
        assert "BOTH sides carry that prefix" in warnings[0]
        # The remedies are not interchangeable and the warning must not imply they
        # are: `VectorBlocker.stream()` forwards `query_prompt` to `search_all()`,
        # and `QdrantHybridIndex.search_all()` raises on any non-None prompt. A
        # message that led with `query_prompt=` would hand a hybrid user a fix
        # that turns a running configuration into a crash. (Cross-model review.)
        assert warnings[0].index("Drop prompt_name") < warnings[0].index("query_prompt=...")
        assert "QdrantHybridIndex.search_all() does not and will raise" in warnings[0]

    def test_warns_through_a_caching_decorator(self, caplog: pytest.LogCaptureFixture) -> None:
        """``DiskCachedEmbedder`` holds the real embedder on ``.embedder``.

        It carries no ``prompt_name`` of its own, so a check that only looked at
        the outermost object would go quiet for every cached embedder — a silent
        hole in a check whose entire purpose is not to be silent.
        """

        class _Cache:
            def __init__(self, embedder: _PrefixEmbedder) -> None:
                self.embedder = embedder

            def encode(self, texts: list[str], prompt: str | None = None) -> np.ndarray:
                return self.embedder.encode(texts, prompt=prompt)

        inner = _PrefixEmbedder(
            prompt_name="document",
            prompts={"document": DOCUMENT_PROMPT, "query": QUERY_PROMPT},
        )
        index = FAISSIndex(embedder=_Cache(inner), metric="cosine")
        index.create_index(list(TEXTS))

        with caplog.at_level(logging.WARNING, logger=BLOCKER_LOGGER):
            VectorBlocker(
                vector_index=index,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert len(_blocker_warnings(caplog)) == 1

    def test_silent_when_both_sides_are_driven(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=_asymmetric_index(),
                query_prompt=QUERY_PROMPT,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert _blocker_warnings(caplog) == []

    def test_silent_when_neither_side_is_driven(self, caplog: pytest.LogCaptureFixture) -> None:
        """langres's default configuration must not warn."""
        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=_bare_index(list(TEXTS)),
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert _blocker_warnings(caplog) == []

    def test_silent_when_only_the_query_side_is_driven(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A query-only prompt is a legitimate, measured configuration.

        It is the ladder's ``instruct`` arm. Warning here would be noise on a
        deliberate choice — and the *other* direction is the one that silently
        encodes queries with the wrong prefix.
        """
        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=_bare_index(list(TEXTS)),
                query_prompt=QUERY_PROMPT,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert _blocker_warnings(caplog) == []

    def test_silent_for_the_documented_symmetric_query_prefix_recipe(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``prompt_name="query"`` with no ``query_prompt`` is symmetric, not broken.

        ``intfloat/e5-base-v2``'s model card: *"Use 'query: ' prefix for symmetric
        tasks such as semantic similarity, paraphrase retrieval"*. Everything gets
        the query prefix and ``search_all`` compares those vectors to each other —
        which is exactly what happens here. e5-base-v2 is in this repo's own
        embedder ladder, so the warning would fire on a correct, documented setup.
        """
        embedder = _PrefixEmbedder(prompt_name="query", prompts={"query": QUERY_PROMPT})
        index = FAISSIndex(embedder=embedder, metric="cosine")
        index.create_index(list(TEXTS))

        with caplog.at_level(logging.WARNING, logger=BLOCKER_LOGGER):
            VectorBlocker(
                vector_index=index,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert _blocker_warnings(caplog) == []

    def test_a_custom_prompt_name_resolving_to_the_query_prefix_is_silent(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The NAME cannot settle it; the resolved VALUE can.

        An embedder may register a custom name whose prefix is identical to the
        query one. Reusing the corpus vectors as queries is then exactly the
        intended symmetric recipe, and warning would be an unmeasured quality
        claim drawn from a string. (Cross-model review.)
        """
        embedder = _PrefixEmbedder(
            prompt_name="symmetric",
            prompts={"symmetric": QUERY_PROMPT, "query": QUERY_PROMPT},
        )
        index = FAISSIndex(embedder=embedder, metric="cosine")
        index.create_index(list(TEXTS))

        with caplog.at_level(logging.WARNING, logger=BLOCKER_LOGGER):
            VectorBlocker(
                vector_index=index,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert _blocker_warnings(caplog) == []

    def test_a_custom_name_resolving_to_a_DIFFERENT_prefix_still_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Control: comparing values must not become "never warn".

        If the resolved prefixes differ, the queries really do carry a prefix the
        query side was not given, and that is the whole point of the check.
        """
        embedder = _PrefixEmbedder(
            prompt_name="passage",
            prompts={"passage": DOCUMENT_PROMPT, "query": QUERY_PROMPT},
        )
        index = FAISSIndex(embedder=embedder, metric="cosine")
        index.create_index(list(TEXTS))

        with caplog.at_level(logging.WARNING, logger=BLOCKER_LOGGER):
            VectorBlocker(
                vector_index=index,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert len(_blocker_warnings(caplog)) == 1

    def test_the_symmetric_exemption_is_exactly_one_name(self) -> None:
        """Guard the premise: an exemption that widened would mute the warning.

        The check above passes for *any* prompt name the exemption covers, so on
        its own it cannot tell "we exempt the query side" from "we exempt
        everything". This pins the set — the same reason
        ``NOT_VECTOR_INDEX_CLAIMANTS`` is written out rather than derived.
        """
        assert _QUERY_SIDE_PROMPT_NAMES == frozenset({"query"})

    def test_an_index_without_an_embedder_is_not_an_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``VectorIndex`` is a structural protocol with test doubles everywhere.

        The check must degrade to "cannot tell" rather than raise on an index
        that exposes no embedder at all.
        """
        index = FakeVectorIndex()
        index.create_index(list(TEXTS))

        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=index,
                query_prompt=QUERY_PROMPT,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert _blocker_warnings(caplog) == []

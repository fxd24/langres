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

from langres.core.blockers.vector import VectorBlocker
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

        assert "document-side prompt" in caplog.text
        assert "prompt_name='document'" in caplog.text

    def test_silent_when_both_sides_are_driven(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=_asymmetric_index(),
                query_prompt=QUERY_PROMPT,
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert caplog.text == ""

    def test_silent_when_neither_side_is_driven(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """langres's default configuration must not warn."""
        with caplog.at_level(logging.WARNING, logger="langres.core.blockers.vector"):
            VectorBlocker(
                vector_index=_bare_index(list(TEXTS)),
                schema_factory=lambda record: record,
                text_field_extractor=lambda entity: str(entity["name"]),
                k_neighbors=3,
            )

        assert caplog.text == ""

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

        assert caplog.text == ""

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

        assert caplog.text == ""

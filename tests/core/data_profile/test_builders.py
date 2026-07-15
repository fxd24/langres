"""Tests for the Wave 2 convenience builders (``from_benchmark`` / ``from_records``).

Covers the happy paths (default pinned section set), the ``include=`` kind
selector (unknown key -> ``ValueError``), graceful degradation (no gold / no
schema / no embeddings / empty -> the section is simply absent, never a raise),
embedding wiring (single source + a >=2-source comparison), and the internal
sampling / helper functions.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Hashable

import numpy as np
import pytest

import langres.data.registry as registry
from langres.core.benchmark import gold_pairs_from_clusters
from langres.core.data_profile import ArraySource, NpySource
from langres.core.data_profile.builders import (
    _build_separability,
    _embed_text,
    _index_by_id,
    _member_cluster_index,
    _positive_pairs,
    _sample_negative_pairs,
    _validate_include,
    from_benchmark,
    from_embedder,
    from_records,
)
from langres.core.models import CompanySchema

# Shared toy corpus + gold used across the builder tests.
_RECORDS = [
    {"id": "1", "name": "Acme Corporation"},
    {"id": "2", "name": "Acme Corp"},
    {"id": "3", "name": "Globex Inc"},
    {"id": "4", "name": "Globex Incorporated"},
    {"id": "5", "name": "Initech"},
]
_CLUSTERS = [{"1", "2"}, {"3", "4"}, {"5"}]
_IDS = [record["id"] for record in _RECORDS]


def _matrix(dim: int, seed: int) -> np.ndarray:
    """A row-per-id matrix with deliberately varying norms (non-degenerate)."""
    rng = np.random.default_rng(seed)
    return np.asarray([rng.normal(size=dim) * (1.0 + i) for i in range(len(_IDS))])


class _FakeBenchmark:
    """A minimal core-only ``Benchmark`` stand-in (no [semantic], no registry)."""

    name = "fake"
    schema = CompanySchema

    def load(
        self,
    ) -> tuple[list[CompanySchema], list[set[str]], set[frozenset[str]]]:
        corpus = [CompanySchema(**record) for record in _RECORDS]
        return (
            corpus,
            [set(cluster) for cluster in _CLUSTERS],
            gold_pairs_from_clusters([set(cluster) for cluster in _CLUSTERS]),
        )


class TestFromBenchmark:
    def test_object_builds_default_section_set(self) -> None:
        report = from_benchmark(_FakeBenchmark())
        assert [section.kind for section in report.sections] == [
            "hero",
            "label_structure",
            "separability",
            "corpus_field",
        ]
        # The hero distilled the label-structure headline numbers.
        hero = report.sections[0]
        assert hero.summary["Overview.n_records"] == 5  # type: ignore[index]

    def test_name_resolves_via_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        def _fake_get(name: str) -> _FakeBenchmark:
            calls.append(name)
            return _FakeBenchmark()

        # from_benchmark does a LOCAL `from langres.data.registry import get_benchmark`,
        # so patching the attribute on the module is picked up at call time.
        monkeypatch.setattr(registry, "get_benchmark", _fake_get)
        report = from_benchmark("abt_buy")
        assert calls == ["abt_buy"]
        assert report.sections[0].kind == "hero"

    def test_include_narrows_sections(self) -> None:
        report = from_benchmark(_FakeBenchmark(), include={"hero", "corpus_field"})
        assert [section.kind for section in report.sections] == ["hero", "corpus_field"]


class TestFromRecords:
    def test_happy_path_default_sections(self) -> None:
        report = from_records(_RECORDS, gold=_CLUSTERS, schema=CompanySchema)
        assert [section.kind for section in report.sections] == [
            "hero",
            "label_structure",
            "separability",
            "corpus_field",
        ]

    def test_no_gold_drops_label_and_separability_and_hero(self) -> None:
        report = from_records(_RECORDS)  # no gold, no schema, no embeddings
        assert [section.kind for section in report.sections] == ["corpus_field"]

    def test_gold_without_schema_keeps_label_no_separability(self) -> None:
        report = from_records(_RECORDS, gold=_CLUSTERS)  # gold but no signal
        assert [section.kind for section in report.sections] == [
            "hero",
            "label_structure",
            "corpus_field",
        ]

    def test_empty_records_yields_empty_report_no_raise(self) -> None:
        report = from_records([])
        assert report.sections == []

    def test_include_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown section kind"):
            from_records(_RECORDS, gold=_CLUSTERS, schema=CompanySchema, include={"labels"})

    def test_include_selects_by_kind(self) -> None:
        report = from_records(
            _RECORDS, gold=_CLUSTERS, schema=CompanySchema, include={"corpus_field"}
        )
        assert [section.kind for section in report.sections] == ["corpus_field"]


class TestEmbeddingWiring:
    def test_single_source_adds_embedding_and_cosine_separability(self) -> None:
        src = ArraySource("m1", _IDS, _matrix(dim=8, seed=1))
        report = from_records(_RECORDS, gold=_CLUSTERS, schema=CompanySchema, embeddings=[src])
        assert [section.kind for section in report.sections] == [
            "hero",
            "label_structure",
            "separability",  # string
            "separability",  # cosine · m1
            "corpus_field",
            "embedding",
        ]
        titles = [section.title for section in report.sections]
        assert "Separability (string)" in titles
        assert "Separability (cosine · m1)" in titles

    def test_two_sources_add_comparison(self) -> None:
        src_a = ArraySource("m1", _IDS, _matrix(dim=8, seed=1))
        src_b = ArraySource("m2", _IDS, _matrix(dim=16, seed=2))
        report = from_records(
            _RECORDS, gold=_CLUSTERS, schema=CompanySchema, embeddings=[src_a, src_b]
        )
        assert [section.kind for section in report.sections] == [
            "hero",
            "label_structure",
            "separability",  # string
            "separability",  # cosine · m1
            "separability",  # cosine · m2
            "corpus_field",
            "embedding",
            "embedding",
            "embedding_comparison",
        ]

    def test_embeddings_without_schema_still_add_cosine_separability(self) -> None:
        src = ArraySource("m1", _IDS, _matrix(dim=8, seed=1))
        report = from_records(_RECORDS, gold=_CLUSTERS, embeddings=[src])
        kinds = [section.kind for section in report.sections]
        # No schema -> no string separability, but the cosine one still lands.
        assert kinds.count("separability") == 1
        assert "Separability (cosine · m1)" in [s.title for s in report.sections]


class TestInternalHelpers:
    def test_positive_pairs_are_within_cluster_only(self) -> None:
        pairs = {frozenset(pair) for pair in _positive_pairs([{"1", "2", "3"}, {"4"}])}
        assert pairs == {frozenset({"1", "2"}), frozenset({"1", "3"}), frozenset({"2", "3"})}

    def test_positive_pairs_bounded_and_logs_on_huge_cluster(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A 500-member gold cluster already yields 124,750 within-cluster pairs; a
        # 5000-member bad-merge "default entity" would materialise ~12.5M. The
        # generator must reservoir-sample that down to `cap` (O(cap) memory), not
        # enumerate the whole O(size**2) set into a list.
        big = {str(i) for i in range(500)}
        with caplog.at_level(logging.WARNING):
            pairs = _positive_pairs([big], cap=200, seed=0)
        assert len(pairs) == 200  # bounded to cap, not 124,750
        assert all(a != b and a in big and b in big for a, b in pairs)
        assert "positive pairs" in caplog.text  # truncation logged (never silent)

    def test_positive_pairs_small_cluster_is_full_set_no_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            pairs = _positive_pairs([{"1", "2", "3"}, {"4"}], cap=1000, seed=0)
        # Under the cap: the exact same full within-cluster pair set as an un-capped
        # enumeration -- the cap only bites on pathologically large clusters.
        assert {frozenset(p) for p in pairs} == {
            frozenset({"1", "2"}),
            frozenset({"1", "3"}),
            frozenset({"2", "3"}),
        }
        assert "positive pairs" not in caplog.text  # no truncation, no log

    def test_member_cluster_index_maps_ids_to_their_cluster(self) -> None:
        index = _member_cluster_index([{"a", "b"}, {"c"}])
        assert index["a"] == index["b"]  # same cluster
        assert index["c"] != index["a"]  # different cluster

    def test_sample_negatives_excludes_same_cluster_and_repeats(self) -> None:
        member_cluster = {"1": 0, "2": 0}  # 1 and 2 share a gold cluster (a positive)
        negatives = _sample_negative_pairs(["1", "2", "3", "4"], member_cluster, cap=100, seed=0)
        keys = [frozenset(pair) for pair in negatives]
        assert frozenset({"1", "2"}) not in keys  # same-cluster positive excluded
        assert len(keys) == len(set(keys))  # no repeats
        assert all(len(k) == 2 for k in keys)  # no self-pairs

    def test_sample_negatives_respects_cap(self) -> None:
        negatives = _sample_negative_pairs(list("abcdefgh"), {}, cap=3, seed=0)
        assert len(negatives) <= 3

    def test_sample_negatives_degenerate_inputs(self) -> None:
        assert _sample_negative_pairs(["only"], {}, cap=10, seed=0) == []
        assert _sample_negative_pairs(["a", "b"], {}, cap=0, seed=0) == []

    def test_sample_negatives_is_deterministic(self) -> None:
        ids: list[Hashable] = list("abcdef")
        first = _sample_negative_pairs(ids, {}, cap=4, seed=7)
        second = _sample_negative_pairs(ids, {}, cap=4, seed=7)
        assert first == second

    def test_embed_text_joins_string_fields_except_id(self) -> None:
        text = _embed_text(
            {"id": "1", "name": "Acme", "n": 3, "note": "  "}, id_key="id", text_key=None
        )
        assert text == "Acme"  # id skipped, non-str skipped, blank skipped

    def test_embed_text_uses_explicit_key(self) -> None:
        text = _embed_text({"id": "1", "blob": "hello"}, id_key="id", text_key="blob")
        assert text == "hello"
        assert _embed_text({"id": "1"}, id_key="id", text_key="blob") == ""

    def test_validate_include_accepts_known_and_rejects_unknown(self) -> None:
        _validate_include({"hero", "embedding"})  # no raise
        with pytest.raises(ValueError, match=r"unknown section kind\(s\) \['nope'\]"):
            _validate_include({"nope"})


class TestIndexById:
    def test_first_wins_and_logs_duplicate_count(self, caplog: pytest.LogCaptureFixture) -> None:
        records = [
            {"id": "1", "name": "first"},
            {"id": "1", "name": "second"},  # duplicate id -> dropped (first-wins)
            {"id": "2", "name": "other"},
        ]
        with caplog.at_level(logging.WARNING):
            id_map = _index_by_id(records, "id")
        assert id_map["1"]["name"] == "first"  # first-wins, not last ("second")
        assert list(id_map) == ["1", "2"]
        assert "dropped 1 duplicate" in caplog.text  # the count is surfaced

    def test_no_duplicates_is_silent(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            id_map = _index_by_id([{"id": "1"}, {"id": "2"}], "id")
        assert list(id_map) == ["1", "2"]
        assert caplog.text == ""  # no duplicates -> no warning

    def test_skips_records_missing_id_key(self) -> None:
        id_map = _index_by_id([{"id": "1"}, {"name": "no-id"}], "id")
        assert list(id_map) == ["1"]  # the id-less record is skipped, not raised


class TestBuildSeparabilityBranches:
    """Direct tests for the separability assembler's degenerate branches."""

    def test_singleton_gold_yields_no_separability(self) -> None:
        # No within-cluster pairs and (with one id) no negatives -> [] (no raise).
        sections = _build_separability(
            id_map={"1": {"id": "1", "name": "Acme"}},
            schema=None,
            clusters=[{"1"}],
            sources=[],
            negatives_cap=10,
            seed=0,
        )
        assert sections == []

    def test_string_signal_unscorable_drops_the_section(self) -> None:
        # Records carry no comparable schema field -> string_signal returns None
        # for every pair -> profile_separability returns None -> section skipped.
        sections = _build_separability(
            id_map={"1": {"id": "1"}, "2": {"id": "2"}},
            schema=CompanySchema,
            clusters=[{"1", "2"}],
            sources=[],
            negatives_cap=10,
            seed=0,
        )
        assert sections == []

    def test_cosine_signal_total_miss_drops_the_section(self) -> None:
        # Source ids don't overlap the corpus -> cosine returns None for all pairs
        # -> the cosine separability section is dropped (not raised).
        mismatched = ArraySource("m", ["x", "y"], np.array([[1.0, 0.0], [0.0, 1.0]]))
        sections = _build_separability(
            id_map={"1": {"id": "1"}, "2": {"id": "2"}},
            schema=None,
            clusters=[{"1", "2"}],
            sources=[mismatched],
            negatives_cap=10,
            seed=0,
        )
        assert sections == []


class _FakeSentenceTransformer:
    """A stand-in encoder: deterministic vectors, no torch/sentence-transformers."""

    def __init__(self, model: str) -> None:
        self.model = model

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        base = np.arange(len(texts) * 3, dtype=float).reshape(len(texts), 3) + 1.0
        if normalize_embeddings:
            base /= np.linalg.norm(base, axis=1, keepdims=True)
        return base


class TestFromEmbedder:
    def _install_fake_st(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mod = types.ModuleType("sentence_transformers")
        fake_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    def test_embeds_and_persists_npy_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._install_fake_st(monkeypatch)
        out = tmp_path / "vecs"  # no .npy suffix -> exercises the suffix-enforce branch
        src = from_embedder(
            [{"id": "1", "name": "Acme"}, {"id": "2", "name": "Globex"}],
            "fake-model",
            out_path=out,
        )
        assert isinstance(src, NpySource)
        assert src.name == "fake-model"
        assert src.dim == 3
        assert (tmp_path / "vecs.npy").exists()
        assert (tmp_path / "vecs.ids.json").exists()
        # Rows are aligned to the record ids.
        assert src.vectors_for(["2"]).shape == (1, 3)

    def test_normalize_marks_cosine_and_keeps_npy_suffix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._install_fake_st(monkeypatch)
        out = tmp_path / "vecs.npy"  # already .npy -> the suffix branch is skipped
        src = from_embedder(
            [{"id": "1", "name": "Acme"}],
            "fake-model",
            out_path=out,
            normalize=True,
        )
        assert src.pre_normalized is True
        assert src.metric == "cosine"
        assert (tmp_path / "vecs.npy").exists()

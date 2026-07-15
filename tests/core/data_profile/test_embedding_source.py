"""Tests for the memory-efficient ``EmbeddingSource`` family.

Consumes *precomputed* vectors only -- fixtures are synthesised with
``numpy.save`` / in-memory arrays; nothing here imports sentence-transformers,
torch, or faiss. Covers the id<->row alignment guards (length desync, unknown
partial vs total miss, same-length permutation via fingerprint), the sidecar id
auto-load, the ``from_anchor_store`` reuse path (metric-driven pre-normalized
caveat), and the ``cosine_signal`` A<->B bridge (degenerate/missing -> ``None``).
The ``O(batch)`` memory behaviour of ``NpySource`` lives in
``test_embedding_memory.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from langres.core.data_profile.embedding_source import (
    ArraySource,
    EmbeddingSource,
    NpySource,
    _fingerprint,
    cosine_signal,
)


def _matrix(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------- ArraySource
class TestArraySource:
    def test_basic_lookup_returns_requested_rows_in_order(self) -> None:
        src = ArraySource("m", ["a", "b", "c"], _matrix([[1, 0], [0, 1], [1, 1]]))
        out = src.vectors_for(["c", "a"])
        assert out.shape == (2, 2)
        np.testing.assert_allclose(out, [[1, 1], [1, 0]])

    def test_is_instance_of_protocol(self) -> None:
        src = ArraySource("m", ["a"], _matrix([[1, 0]]))
        assert isinstance(src, EmbeddingSource)

    def test_name_dim_and_id_order(self) -> None:
        src = ArraySource("minilm", ["a", "b"], _matrix([[1, 0, 0], [0, 1, 0]]))
        assert src.name == "minilm"
        assert src.dim == 3
        assert src.id_order == ["a", "b"]

    def test_id_order_is_a_copy(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        src.id_order.append("mutate")
        assert src.id_order == ["a", "b"]

    def test_row_id_desync_raises(self) -> None:
        with pytest.raises(ValueError, match="desync"):
            ArraySource("m", ["a", "b"], _matrix([[1, 0]]))

    def test_non_2d_matrix_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            ArraySource("m", ["a"], np.asarray([1.0, 2.0], dtype=np.float32))

    def test_empty_source_is_valid(self) -> None:
        src = ArraySource("m", [], np.zeros((0, 4), dtype=np.float32))
        assert src.dim == 4
        assert src.vectors_for([]).shape == (0, 4)

    def test_duplicate_ids_first_wins(self) -> None:
        src = ArraySource("m", ["a", "a"], _matrix([[1, 0], [0, 9]]))
        np.testing.assert_allclose(src.vectors_for(["a"]), [[1, 0]])

    def test_returned_array_is_independent_copy(self) -> None:
        matrix = _matrix([[1, 0], [0, 1]])
        src = ArraySource("m", ["a", "b"], matrix)
        out = src.vectors_for(["a"])
        out[0, 0] = 999.0
        np.testing.assert_allclose(matrix[0], [1, 0])  # backing untouched


# ------------------------------------------------------------------- id guards
class TestIdGuards:
    def test_unknown_ids_partial_miss_dropped_and_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        with caplog.at_level(logging.WARNING):
            out = src.vectors_for(["a", "ghost", "b"])
        assert out.shape == (2, 2)  # ghost dropped
        assert "dropped 1 unknown id" in caplog.text

    def test_total_miss_raises(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        with pytest.raises(KeyError, match="wrong id namespace"):
            src.vectors_for(["ghost1", "ghost2"])

    def test_empty_request_is_not_a_miss(self) -> None:
        src = ArraySource("m", ["a"], _matrix([[1, 0]]))
        assert src.vectors_for([]).shape == (0, 2)


# ------------------------------------------------------------------ fingerprint
class TestFingerprint:
    def test_permutation_changes_fingerprint(self) -> None:
        assert _fingerprint(["a", "b", "c"]) != _fingerprint(["a", "c", "b"])

    def test_same_order_same_fingerprint(self) -> None:
        assert _fingerprint(["a", "b"]) == _fingerprint(["a", "b"])

    def test_boundary_collision_avoided(self) -> None:
        assert _fingerprint(["a", "bc"]) != _fingerprint(["ab", "c"])

    def test_source_fingerprint_matches_helper(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        assert src.id_fingerprint == _fingerprint(["a", "b"])

    def test_verify_alignment_passes_on_match(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        src.verify_alignment(_fingerprint(["a", "b"]))  # no raise

    def test_verify_alignment_raises_on_permutation(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            src.verify_alignment(_fingerprint(["b", "a"]))


# --------------------------------------------------------------------- NpySource
class TestNpySource:
    def _write_npy(self, tmp_path: Path, rows: list[list[float]], name: str = "vecs.npy") -> Path:
        path = tmp_path / name
        np.save(path, _matrix(rows))
        return path

    def test_memmap_lookup(self, tmp_path: Path) -> None:
        path = self._write_npy(tmp_path, [[1, 0], [0, 1], [2, 2]])
        src = NpySource("m", path, ["a", "b", "c"])
        np.testing.assert_allclose(src.vectors_for(["b", "c"]), [[0, 1], [2, 2]])

    def test_backing_is_memmap(self, tmp_path: Path) -> None:
        path = self._write_npy(tmp_path, [[1, 0], [0, 1]])
        src = NpySource("m", path, ["a", "b"])
        assert isinstance(src._matrix, np.memmap)

    def test_returned_array_is_not_a_memmap(self, tmp_path: Path) -> None:
        path = self._write_npy(tmp_path, [[1, 0], [0, 1]])
        src = NpySource("m", path, ["a", "b"])
        out = src.vectors_for(["a"])
        assert not isinstance(out, np.memmap)
        assert out.base is None  # independent, owns its data

    def test_sidecar_stem_ids_json_auto_load(self, tmp_path: Path) -> None:
        path = self._write_npy(tmp_path, [[1, 0], [0, 1]], name="vecs.npy")
        (tmp_path / "vecs.ids.json").write_text(json.dumps(["a", "b"]))
        src = NpySource("m", path)  # no id_order -> sidecar
        assert src.id_order == ["a", "b"]

    def test_sidecar_plain_ids_json_auto_load(self, tmp_path: Path) -> None:
        path = self._write_npy(tmp_path, [[1, 0], [0, 1]], name="vecs.npy")
        (tmp_path / "ids.json").write_text(json.dumps(["x", "y"]))
        src = NpySource("m", path)
        assert src.id_order == ["x", "y"]

    def test_missing_sidecar_raises(self, tmp_path: Path) -> None:
        path = self._write_npy(tmp_path, [[1, 0]])
        with pytest.raises(ValueError, match="needs an id_order"):
            NpySource("m", path)


# ------------------------------------------------------------- from_anchor_store
class TestFromAnchorStore:
    def _make_artifact(
        self,
        root: Path,
        anchor_ids: list[str],
        vectors: list[list[float]],
        metric: str,
        *,
        nested: bool = True,
    ) -> None:
        """Synthesise an AnchorStore.save-shaped artifact (no faiss / resolver)."""
        root.mkdir(parents=True, exist_ok=True)
        # Manifest at the top (AnchorStoreManifest-shaped).
        manifest = {
            "store_version": "1",
            "entity_prefix": "e",
            "next_ordinal": len(anchor_ids),
            "anchor_ids": anchor_ids,
            "records": {rid: {"id": rid} for rid in anchor_ids},
            "assignments": {rid: f"e{i}" for i, rid in enumerate(anchor_ids)},
        }
        (root / "anchor_store.json").write_text(json.dumps(manifest))
        # Vectors + corpus.json nested under resolver/ (as AnchorStore.save does).
        index_dir = root / "resolver" / "index" if nested else root
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "corpus_embeddings.npy", _matrix(vectors))
        (index_dir / "corpus.json").write_text(
            json.dumps({"corpus_texts": anchor_ids, "metric": metric})
        )

    def test_cosine_sets_pre_normalized(self, tmp_path: Path) -> None:
        self._make_artifact(tmp_path, ["a", "b"], [[1, 0], [0, 1]], "cosine")
        src = NpySource.from_anchor_store(tmp_path, "idx")
        assert src.pre_normalized is True
        assert src.metric == "cosine"
        assert src.id_order == ["a", "b"]
        np.testing.assert_allclose(src.vectors_for(["b"]), [[0, 1]])

    def test_l2_not_pre_normalized(self, tmp_path: Path) -> None:
        self._make_artifact(tmp_path, ["a", "b"], [[3, 4], [0, 2]], "L2")
        src = NpySource.from_anchor_store(tmp_path, "idx")
        assert src.pre_normalized is False
        assert src.metric == "L2"

    def test_fingerprint_persisted_then_permutation_raises(self, tmp_path: Path) -> None:
        self._make_artifact(tmp_path, ["a", "b", "c"], [[1, 0], [0, 1], [1, 1]], "L2")
        NpySource.from_anchor_store(tmp_path, "idx")  # persists fingerprint
        assert (tmp_path / "embedding_ids.fingerprint").exists()
        # Same vectors, permuted anchor ids -> a length check would miss it.
        manifest = json.loads((tmp_path / "anchor_store.json").read_text())
        manifest["anchor_ids"] = ["a", "c", "b"]
        (tmp_path / "anchor_store.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="changed under the same vectors"):
            NpySource.from_anchor_store(tmp_path, "idx")

    def test_reload_same_order_is_stable(self, tmp_path: Path) -> None:
        self._make_artifact(tmp_path, ["a", "b"], [[1, 0], [0, 1]], "L2")
        NpySource.from_anchor_store(tmp_path, "idx")
        src = NpySource.from_anchor_store(tmp_path, "idx")  # fingerprint matches
        assert src.id_order == ["a", "b"]

    def test_missing_metric_json_defaults_none(self, tmp_path: Path) -> None:
        self._make_artifact(tmp_path, ["a"], [[1, 0]], "L2")
        # Remove corpus.json to exercise the "no metric" branch.
        next(tmp_path.rglob("corpus.json")).unlink()
        src = NpySource.from_anchor_store(tmp_path, "idx")
        assert src.metric is None
        assert src.pre_normalized is False

    def test_missing_vectors_raises(self, tmp_path: Path) -> None:
        (tmp_path / "anchor_store.json").write_text(
            json.dumps(
                {
                    "store_version": "1",
                    "entity_prefix": "e",
                    "next_ordinal": 0,
                    "anchor_ids": [],
                    "records": {},
                    "assignments": {},
                }
            )
        )
        with pytest.raises(FileNotFoundError, match="corpus_embeddings.npy"):
            NpySource.from_anchor_store(tmp_path, "idx")

    def test_flat_layout_supported(self, tmp_path: Path) -> None:
        self._make_artifact(tmp_path, ["a", "b"], [[1, 0], [0, 1]], "cosine", nested=False)
        src = NpySource.from_anchor_store(tmp_path, "idx")
        assert src.id_order == ["a", "b"]
        assert src.pre_normalized is True


# ------------------------------------------------------------------ cosine_signal
class TestCosineSignal:
    def test_orthogonal_is_zero(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        signal = cosine_signal(src)
        assert signal("a", "b") == pytest.approx(0.0)

    def test_identical_direction_is_one(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [3, 0]]))
        signal = cosine_signal(src)
        assert signal("a", "b") == pytest.approx(1.0)

    def test_opposite_is_minus_one(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [-1, 0]]))
        assert cosine_signal(src)("a", "b") == pytest.approx(-1.0)

    def test_missing_id_returns_none(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        signal = cosine_signal(src)
        assert signal("a", "ghost") is None
        assert signal("ghost", "b") is None

    def test_both_missing_returns_none(self) -> None:
        src = ArraySource("m", ["a", "b"], _matrix([[1, 0], [0, 1]]))
        assert cosine_signal(src)("g1", "g2") is None

    def test_zero_norm_vector_returns_none(self) -> None:
        src = ArraySource("m", ["a", "z"], _matrix([[1, 0], [0, 0]]))
        assert cosine_signal(src)("a", "z") is None

    def test_non_finite_vector_returns_none(self) -> None:
        src = ArraySource("m", ["a", "n"], _matrix([[1, 0], [np.nan, 0]]))
        assert cosine_signal(src)("a", "n") is None

    def test_same_id_twice_is_one(self) -> None:
        src = ArraySource("m", ["a"], _matrix([[2, 1]]))
        assert cosine_signal(src)("a", "a") == pytest.approx(1.0)


# --------------------------------------------------------------- bare-ndarray guard
class TestBareNdarrayGuard:
    def test_cosine_signal_rejects_ndarray(self) -> None:
        with pytest.raises(TypeError, match="bare numpy ndarray"):
            cosine_signal(np.zeros((2, 3)))  # type: ignore[arg-type]

    def test_cosine_signal_rejects_non_source(self) -> None:
        with pytest.raises(TypeError, match="vectors_for"):
            cosine_signal(object())  # type: ignore[arg-type]

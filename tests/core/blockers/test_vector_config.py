"""Config-registry plumbing tests for VectorBlocker (Wave 2b).

Covers:
- the declarative ``schema=`` constructor (coexists with ``schema_factory=``),
- the declarative ``text_field=`` alternative to ``text_field_extractor=``,
- registry serialization (``config`` / ``from_config``) with the vector index
  nested as a ``ComponentSpec``.

The concrete FAISS index serialization is Wave 2d's responsibility. To stay
independent, these tests use a tiny in-test serializable index stub that
implements the index protocol + ``SerializableState`` and is registered under a
test-only ``type_name``.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langres.core.blockers.vector import VectorBlocker
from langres.core.indexes.vector_index import inverse_distances_to_similarities
from langres.core.models import CompanySchema
from langres.core.registry import get_component, register
from langres.core.serialization import ComponentSpec, SerializableState


@register("fake_serializable_index_v2b")
class _FakeSerializableIndex:
    """Minimal VectorIndex + SerializableState stub for round-trip testing.

    Mirrors FakeVectorIndex's deterministic search_all output so the blocker
    produces stable candidates, and persists its only state (``n_samples`` and
    a ``label``) through save_state/load_state to exercise the
    SerializableState branch of VectorBlocker.from_config.
    """

    def __init__(self, label: str = "default") -> None:
        self.label = label
        self._n_samples: int | None = None

    # --- config-registry protocol -------------------------------------
    @property
    def config(self) -> dict[str, object]:
        return {"label": self.label}

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "_FakeSerializableIndex":
        return cls(label=str(config["label"]))

    # --- VectorIndex protocol -----------------------------------------
    def create_index(self, texts: list[str]) -> None:
        self._n_samples = len(texts)

    def search(
        self,
        query_texts: str | list[str] | np.ndarray,
        k: int,
        query_prompt: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover - unused by stream()
        raise NotImplementedError

    def search_all(self, k: int, query_prompt: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        assert self._n_samples is not None
        indices = np.zeros((self._n_samples, k), dtype=np.int64)
        distances = np.zeros((self._n_samples, k), dtype=np.float32)
        for i in range(self._n_samples):
            for j in range(k):
                indices[i, j] = (i + j) % self._n_samples
                distances[i, j] = j * 0.1
        return distances, indices

    def to_similarities(self, distances: np.ndarray) -> np.ndarray:
        # Synthetic distances are rank-ordered (lower = closer), like FakeVectorIndex.
        return inverse_distances_to_similarities(distances)

    # --- SerializableState --------------------------------------------
    def save_state(self, state_dir: Path) -> None:
        (state_dir / "state.json").write_text(json.dumps({"n_samples": self._n_samples}))

    def load_state(self, state_dir: Path) -> None:
        data = json.loads((state_dir / "state.json").read_text())
        self._n_samples = data["n_samples"]


def _company_factory(record: dict) -> CompanySchema:
    return CompanySchema(id=record["id"], name=record["name"])


COMPANY_DATA = [
    {"id": "a", "name": "Acme"},
    {"id": "b", "name": "Beta"},
    {"id": "c", "name": "Gamma"},
]


def _build_index() -> _FakeSerializableIndex:
    index = _FakeSerializableIndex(label="companies")
    index.create_index([r["name"] for r in COMPANY_DATA])
    return index


def test_schema_and_text_field_constructors_work() -> None:
    """schema= + text_field= produce a working blocker."""
    blocker = VectorBlocker(
        schema=CompanySchema,
        text_field="name",
        vector_index=_build_index(),
        k_neighbors=2,
    )

    candidates = list(blocker.stream(COMPANY_DATA))

    assert len(candidates) > 0
    assert all(isinstance(c.left, CompanySchema) for c in candidates)
    # text_field extraction matches the entity attribute
    assert blocker.text_field_extractor(candidates[0].left) == candidates[0].left.name


def test_schema_factory_and_text_field_extractor_still_work() -> None:
    """Existing callable constructors are unchanged (coexistence)."""
    blocker = VectorBlocker(
        schema_factory=_company_factory,
        text_field_extractor=lambda e: e.name,
        vector_index=_build_index(),
        k_neighbors=2,
    )

    candidates = list(blocker.stream(COMPANY_DATA))

    assert len(candidates) > 0
    assert all(isinstance(c.left, CompanySchema) for c in candidates)


def test_both_schema_and_factory_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VectorBlocker(
            schema=CompanySchema,
            schema_factory=_company_factory,
            text_field="name",
            vector_index=_build_index(),
        )


def test_neither_schema_nor_factory_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VectorBlocker(text_field="name", vector_index=_build_index())


def test_both_text_field_and_extractor_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VectorBlocker(
            schema=CompanySchema,
            text_field="name",
            text_field_extractor=lambda e: e.name,
            vector_index=_build_index(),
        )


def test_neither_text_field_nor_extractor_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        VectorBlocker(schema=CompanySchema, vector_index=_build_index())


def test_registered_under_type_name() -> None:
    assert get_component("vector_blocker") is VectorBlocker


def test_config_shape() -> None:
    """config exposes schema name, text_field, knobs, and nested index spec."""
    blocker = VectorBlocker(
        schema=CompanySchema,
        text_field="name",
        vector_index=_build_index(),
        k_neighbors=5,
        query_prompt="query: ",
    )

    config = blocker.config

    assert config["schema_type_name"] == "CompanySchema"
    assert config["text_field"] == "name"
    assert config["k_neighbors"] == 5
    assert config["query_prompt"] == "query: "
    index_spec = config["vector_index"]
    assert isinstance(index_spec, ComponentSpec)
    assert index_spec.type_name == "fake_serializable_index_v2b"
    assert index_spec.config == {"label": "companies"}


def test_factory_blocker_config_raises_not_serializable() -> None:
    """A schema_factory-constructed blocker cannot serialize its config."""
    blocker = VectorBlocker(
        schema_factory=_company_factory,
        text_field="name",
        vector_index=_build_index(),
    )

    with pytest.raises(ValueError, match="not serializable"):
        _ = blocker.config


def test_extractor_blocker_config_raises_not_serializable() -> None:
    """A text_field_extractor-constructed blocker cannot serialize its config."""
    blocker = VectorBlocker(
        schema=CompanySchema,
        text_field_extractor=lambda e: e.name,
        vector_index=_build_index(),
    )

    with pytest.raises(ValueError, match="not serializable"):
        _ = blocker.config


def test_config_from_config_roundtrip(tmp_path: Path) -> None:
    """config -> save_state -> from_config -> load_state reproduces candidates."""
    blocker = VectorBlocker(
        schema=CompanySchema,
        text_field="name",
        vector_index=_build_index(),
        k_neighbors=2,
    )
    config = blocker.config

    # Simulate the Resolver save step: persist the index's out-of-band state.
    state_dir = tmp_path / "index"
    state_dir.mkdir()
    assert isinstance(blocker.vector_index, SerializableState)
    blocker.vector_index.save_state(state_dir)

    # Serialize config to JSON-able dict (ComponentSpec -> dict) and back, to
    # prove the nested spec survives a real persistence boundary.
    json_config: dict[str, Any] = dict(config)
    json_config["vector_index"] = blocker.config["vector_index"].model_dump()

    rebuilt = VectorBlocker.from_config(json_config, state_dir=state_dir)

    before = [(c.left.id, c.right.id, c.similarity_score) for c in blocker.stream(COMPANY_DATA)]
    after = [(c.left.id, c.right.id, c.similarity_score) for c in rebuilt.stream(COMPANY_DATA)]

    assert before == after
    assert len(after) > 0


def test_from_config_without_state_dir_when_not_serializable() -> None:
    """from_config works when the index is not SerializableState (no state_dir)."""
    # Register a non-serializable index inline for this test.

    @register("fake_plain_index_v2b")
    class _PlainIndex:
        def __init__(self, label: str = "x") -> None:
            self.label = label
            self._n_samples: int | None = None

        @property
        def config(self) -> dict[str, object]:
            return {"label": self.label}

        @classmethod
        def from_config(cls, config: dict[str, object]) -> "_PlainIndex":
            return cls(label=str(config["label"]))

        def create_index(self, texts: list[str]) -> None:
            self._n_samples = len(texts)

        def search(self, query_texts: Any, k: int, query_prompt: str | None = None) -> Any:
            raise NotImplementedError  # pragma: no cover

        def search_all(
            self, k: int, query_prompt: str | None = None
        ) -> tuple[np.ndarray, np.ndarray]:
            assert self._n_samples is not None
            indices = np.zeros((self._n_samples, k), dtype=np.int64)
            distances = np.zeros((self._n_samples, k), dtype=np.float32)
            for i in range(self._n_samples):
                for j in range(k):
                    indices[i, j] = (i + j) % self._n_samples
                    distances[i, j] = j * 0.1
            return distances, indices

        def to_similarities(self, distances: np.ndarray) -> np.ndarray:
            return inverse_distances_to_similarities(distances)

    index = _PlainIndex(label="companies")
    index.create_index([r["name"] for r in COMPANY_DATA])
    blocker = VectorBlocker(
        schema=CompanySchema,
        text_field="name",
        vector_index=index,
        k_neighbors=2,
    )

    config = blocker.config
    json_config: dict[str, Any] = dict(config)
    json_config["vector_index"] = config["vector_index"].model_dump()

    # No state_dir needed: a freshly-from_config'd plain index must be rebuilt
    # by the caller. Here we just confirm reconstruction succeeds and the
    # index is recreated by the caller before streaming.
    rebuilt = VectorBlocker.from_config(json_config)
    rebuilt.vector_index.create_index([r["name"] for r in COMPANY_DATA])

    after = list(rebuilt.stream(COMPANY_DATA))
    assert len(after) > 0


def test_config_raises_for_unregistered_index() -> None:
    """config fails clearly when the vector index class is not registered."""

    class _UnregisteredIndex:
        def __init__(self) -> None:
            self._n_samples: int | None = None

        @property
        def config(self) -> dict[str, object]:
            return {}

        def create_index(self, texts: list[str]) -> None:
            self._n_samples = len(texts)

        def search(self, query_texts: Any, k: int, query_prompt: str | None = None) -> Any:
            raise NotImplementedError  # pragma: no cover

        def search_all(
            self, k: int, query_prompt: str | None = None
        ) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
            raise NotImplementedError

    index = _UnregisteredIndex()
    index.create_index([r["name"] for r in COMPANY_DATA])
    blocker = VectorBlocker(
        schema=CompanySchema,
        text_field="name",
        vector_index=index,
        k_neighbors=2,
    )

    with pytest.raises(ValueError, match="is not registered"):
        _ = blocker.config

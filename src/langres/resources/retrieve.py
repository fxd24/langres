"""Embedding-backed retrieval as the first Source in an explicit topology."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from langres.core.blockers.all_pairs import register_schema_idempotent, schema_to_factory
from langres.core.comparators import StringComparator
from langres.core.model_ref import to_config
from langres.core.op import Records, Source
from langres.core.pairs import PairRow, Pairs
from langres.core.registry import (
    OpSerializer,
    get_schema,
    register_op_serializer,
)
from langres.resources.base import Embedder, require_unique_ids

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class Retrieve(Source[SchemaT], Generic[SchemaT]):
    """Retrieve each record's nearest neighbours with one Embedder resource.

    Retrieval owns candidate generation and its heuristic cosine score. It does
    not decide whether that score is used for a top-k prune, a final threshold,
    or input to a downstream Rerank/LLM stage; following Select operations own
    those roles.
    """

    def __init__(
        self,
        resource: Embedder,
        *,
        schema: type[SchemaT],
        k: int = 20,
        text_field: str | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("Retrieve.k must be positive")
        if text_field is not None and text_field not in schema.model_fields:
            raise ValueError(
                f"Retrieve text_field {text_field!r} is not declared by {schema.__name__}"
            )
        self.resource = resource
        self._schema = schema
        self._schema_name = register_schema_idempotent(schema)
        self.k = k
        self.text_field = text_field
        self._factory = schema_to_factory(schema)
        self._default_fields = (
            ()
            if text_field is not None
            else tuple(spec.name for spec in StringComparator.from_schema(schema).feature_specs)
        )

    @property
    def schema(self) -> type[BaseModel]:
        """The declarative entity schema used to normalize retrieval records."""
        return self._schema

    @property
    def config(self) -> dict[str, object]:
        """Safe scalar topology parameters; the Embedder is a nested resource."""
        return {
            "schema_name": self._schema_name,
            "k": self.k,
            "text_field": self.text_field,
        }

    @classmethod
    def from_config(
        cls,
        resource: Embedder,
        config: dict[str, object],
    ) -> "Retrieve[Any]":
        """Rebuild around a validated Embedder resource."""
        schema = get_schema(str(config["schema_name"]))
        text_field = config["text_field"]
        return Retrieve(
            resource,
            schema=schema,
            k=int(config["k"]),  # type: ignore[call-overload]
            text_field=str(text_field) if text_field is not None else None,
        )

    def _text(self, entity: SchemaT) -> str:
        if self.text_field is not None:
            return str(getattr(entity, self.text_field))
        return " ".join(
            str(getattr(entity, field))
            for field in self._default_fields
            if getattr(entity, field, None)
        )

    def forward(self, records: Records) -> Pairs[SchemaT]:
        """Embed records once and emit the union of their top-k cosine neighbours."""
        entities = [self._factory(record) for record in records]
        ids = [str(entity.id) for entity in entities]  # type: ignore[attr-defined]
        require_unique_ids(ids, field="record ids", operation="Retrieve")
        store = dict(zip(ids, entities, strict=True))
        if len(entities) < 2:
            return Pairs(store=store, rows=[])

        batch = self.resource.embed([self._text(entity) for entity in entities])
        if batch.model_ref != self.resource.model_ref:
            raise ValueError(
                "Embedder returned a batch with a different model_ref from the resource"
            )
        vectors = np.asarray(batch.vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(entities):
            raise ValueError(
                "Embedder must return one 2D vector row per retrieval record; "
                f"got shape {vectors.shape} for {len(entities)} records"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = np.divide(
            vectors,
            norms,
            out=np.zeros_like(vectors),
            where=norms != 0,
        )
        similarities = normalized @ normalized.T
        np.fill_diagonal(similarities, -np.inf)

        positions = {entity_id: index for index, entity_id in enumerate(ids)}
        selected: dict[tuple[str, str], tuple[str, str, float]] = {}
        limit = min(self.k, len(entities) - 1)
        for index, row in enumerate(similarities):
            neighbours = np.argsort(-row, kind="stable")[:limit]
            for neighbour in neighbours:
                first_id, second_id = ids[index], ids[int(neighbour)]
                pair = (
                    (first_id, second_id)
                    if first_id <= second_id
                    else (second_id, first_id)
                )
                left_id, right_id = sorted(
                    (first_id, second_id),
                    key=positions.__getitem__,
                )
                score = float(np.clip(row[int(neighbour)], 0.0, 1.0))
                prior = selected.get(pair)
                selected[pair] = (
                    left_id,
                    right_id,
                    max(score, prior[2] if prior is not None else 0.0),
                )

        provenance = {
            "retrieve": {
                "model_ref": to_config(batch.model_ref),
                "runtime": self.resource.runtime_config.model_dump(mode="json"),
                "embedding": (
                    batch.facts.model_dump(mode="json") if batch.facts is not None else None
                ),
            }
        }
        rows: list[PairRow[SchemaT]] = [
            PairRow(
                left_id=left_id,
                right_id=right_id,
                blocker_name="retrieve",
                score=score,
                score_type="heuristic",
                decision_step="retrieve",
                provenance=provenance,
            )
            for _, (left_id, right_id, score) in sorted(selected.items())
        ]
        return Pairs(store=store, rows=rows)


class _RetrieveParams(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_name: str = Field(min_length=1)
    k: int = Field(gt=0)
    text_field: str | None = None


def _dump_retrieve(stage: object) -> tuple[dict[str, object], object | None]:
    if not isinstance(stage, Retrieve):
        raise TypeError("retrieve serializer requires an exact Retrieve operation")
    return stage.config, stage.resource


def _load_retrieve(
    params: dict[str, object],
    component: object | None,
    _state_dir: Path,
) -> Retrieve[Any]:
    if not isinstance(component, Embedder):
        raise TypeError("OpSpec role 'retrieve' requires an Embedder resource")
    return Retrieve.from_config(component, params)


def _validate_retrieve(params: dict[str, object]) -> dict[str, object]:
    return _RetrieveParams.model_validate(params).model_dump()


register_op_serializer(
    OpSerializer(
        role="retrieve",
        op_type=Retrieve,
        dump=_dump_retrieve,
        load=_load_retrieve,
        component_slot="resource",
        validate_params=_validate_retrieve,
    )
)

__all__ = ["Retrieve"]

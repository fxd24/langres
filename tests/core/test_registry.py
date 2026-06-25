"""Unit tests for the component/schema registry (M0 Wave 1)."""

import pytest
from pydantic import BaseModel

from langres.core.registry import (
    SchemaNotRegistered,
    UnknownComponentType,
    get_component,
    get_schema,
    register,
    register_schema,
)


class TestRegisterComponent:
    def test_register_and_lookup(self) -> None:
        @register("test_component_unique_a")
        class _Comp:
            pass

        assert get_component("test_component_unique_a") is _Comp

    def test_decorator_preserves_type(self) -> None:
        # mypy --strict relies on the decorator being identity-typed; at runtime
        # the decorated symbol must be the same class object.
        @register("test_component_unique_b")
        class _Comp:
            value: int = 5

        assert _Comp().value == 5
        assert _Comp.__name__ == "_Comp"

    def test_duplicate_raises(self) -> None:
        @register("test_component_dup")
        class _First:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register("test_component_dup")
            class _Second:
                pass

    def test_unknown_component_raises_with_suggestions(self) -> None:
        @register("rapidfuzz_judge")
        class _Comp:
            pass

        with pytest.raises(UnknownComponentType) as exc:
            get_component("rapidfuzz_judg")  # typo
        msg = str(exc.value)
        assert "rapidfuzz_judg" in msg
        # did-you-mean from difflib.get_close_matches
        assert "rapidfuzz_judge" in msg

    def test_unknown_component_lists_available(self) -> None:
        with pytest.raises(UnknownComponentType) as exc:
            get_component("zzz_definitely_not_registered_xyz")
        assert "Available" in str(exc.value) or "available" in str(exc.value)


class TestRegisterSchema:
    def test_register_and_lookup_schema(self) -> None:
        @register_schema("TestSchemaUnique")
        class _Schema(BaseModel):
            id: str

        assert get_schema("TestSchemaUnique") is _Schema

    def test_duplicate_schema_raises(self) -> None:
        @register_schema("TestSchemaDup")
        class _SchemaA(BaseModel):
            id: str

        with pytest.raises(ValueError, match="already registered"):

            @register_schema("TestSchemaDup")
            class _SchemaB(BaseModel):
                id: str

    def test_unknown_schema_raises(self) -> None:
        with pytest.raises(SchemaNotRegistered) as exc:
            get_schema("NoSuchSchema123")
        assert "NoSuchSchema123" in str(exc.value)

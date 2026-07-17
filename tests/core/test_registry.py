"""Unit tests for the component/schema registry (M0 Wave 1)."""

import pytest
from pydantic import BaseModel

from langres.core.registry import (
    SchemaNotRegistered,
    UnknownComponentType,
    UnknownModelType,
    get_component,
    get_model,
    get_schema,
    model_type_name,
    register,
    register_model,
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


class TestRegisterModel:
    """The third namespace: Resolver subclasses (architectures), for save/load identity."""

    def test_register_and_lookup(self) -> None:
        @register_model("test_model_unique_a")
        class _Arch:
            pass

        assert get_model("test_model_unique_a") is _Arch

    def test_duplicate_registration_raises(self) -> None:
        @register_model("test_model_dup")
        class _ArchA:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @register_model("test_model_dup")
            class _ArchB:
                pass

    def test_unknown_model_raises_actionably(self) -> None:
        with pytest.raises(UnknownModelType) as exc:
            get_model("no_such_model_123")
        message = str(exc.value)
        assert "no_such_model_123" in message
        assert "never imported" in message  # the usual cause, named

    def test_unknown_model_suggests_a_near_miss(self) -> None:
        @register_model("test_model_suggestible")
        class _Arch:
            pass

        with pytest.raises(UnknownModelType, match="Did you mean: test_model_suggestible"):
            get_model("test_model_suggestable")

    def test_model_type_name_reverses_the_lookup(self) -> None:
        @register_model("test_model_reverse")
        class _Arch:
            pass

        assert model_type_name(_Arch) == "test_model_reverse"

    def test_model_type_name_is_none_for_unregistered(self) -> None:
        class _Unregistered:
            pass

        assert model_type_name(_Unregistered) is None

    def test_model_type_name_does_not_walk_the_mro(self) -> None:
        """A subclass of a registered model is its own thing, not its parent.

        Claiming the parent's name would make ``load`` hand back the wrong class.
        """

        @register_model("test_model_parent")
        class _Parent:
            pass

        class _Child(_Parent):
            pass

        assert model_type_name(_Child) is None

    def test_models_are_a_separate_namespace_from_components(self) -> None:
        """A model name must not resolve as a component (it cannot fill a slot)."""

        @register_model("test_namespace_isolation")
        class _Arch:
            pass

        with pytest.raises(UnknownComponentType):
            get_component("test_namespace_isolation")

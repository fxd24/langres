"""Unit tests for the component/schema registry (M0 Wave 1)."""

from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from langres.core import registry as registry_module
from langres.core.registry import (
    OpSerializer,
    SchemaNotRegistered,
    UnknownComponentType,
    UnknownModelType,
    UnknownOpType,
    get_component,
    get_model,
    get_op_serializer,
    get_schema,
    model_type_name,
    register,
    register_model,
    register_op,
    register_op_serializer,
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


class _StrictOpConfig(BaseModel):
    """The parameter envelope ``@register_op`` demands: closed and strictly typed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    factor: float = 1.0


class TestRegisterOpValidation:
    """``@register_op`` is a fail-closed contract; each rule rejects at decoration time.

    These guardrails are what stop a saved chain from naming a class whose
    parameter envelope is open or loosely typed — without them an artifact could
    smuggle unvalidated keys straight into ``from_config``.
    """

    def test_requires_from_config(self) -> None:
        with pytest.raises(TypeError, match="from_config"):

            @register_op("test_op_no_from_config")
            class _NoFromConfig:
                config_model = _StrictOpConfig
                config: dict[str, object] = {"factor": 1.0}

    def test_requires_config(self) -> None:
        with pytest.raises(TypeError, match=r"requires _NoConfig\.config"):

            @register_op("test_op_no_config")
            class _NoConfig:
                config_model = _StrictOpConfig

                @classmethod
                def from_config(cls, config: dict[str, object]) -> "_NoConfig":
                    return cls()

    def test_requires_a_pydantic_config_model(self) -> None:
        with pytest.raises(TypeError, match="config_model to be a "):

            @register_op("test_op_no_config_model")
            class _NoConfigModel:
                config: dict[str, object] = {"factor": 1.0}
                config_model = dict  # not a BaseModel subclass

                @classmethod
                def from_config(cls, config: dict[str, object]) -> "_NoConfigModel":
                    return cls()

    def test_requires_extra_forbid(self) -> None:
        class _OpenConfig(BaseModel):
            model_config = ConfigDict(strict=True)

        with pytest.raises(TypeError, match="extra='forbid'"):

            @register_op("test_op_extra_allowed")
            class _OpenOp:
                config: dict[str, object] = {}
                config_model = _OpenConfig

                @classmethod
                def from_config(cls, config: dict[str, object]) -> "_OpenOp":
                    return cls()

    def test_requires_strict(self) -> None:
        class _LooseConfig(BaseModel):
            model_config = ConfigDict(extra="forbid")

        with pytest.raises(TypeError, match="strict=True"):

            @register_op("test_op_not_strict")
            class _LooseOp:
                config: dict[str, object] = {}
                config_model = _LooseConfig

                @classmethod
                def from_config(cls, config: dict[str, object]) -> "_LooseOp":
                    return cls()


class TestRegisteredOpSerializer:
    """What the serializer built by ``@register_op`` actually does."""

    def test_dump_reads_a_mapping_config_property(self) -> None:
        @register_op("test_op_mapping_config")
        class _MappingOp:
            config_model = _StrictOpConfig

            def __init__(self, factor: float = 2.0) -> None:
                self.factor = factor

            @property
            def config(self) -> dict[str, object]:
                return {"factor": self.factor}

            @classmethod
            def from_config(cls, config: dict[str, object]) -> "_MappingOp":
                return cls(float(config["factor"]))  # type: ignore[arg-type]

        serializer = get_op_serializer("test_op_mapping_config")
        params, component = serializer.dump(_MappingOp(3.0))
        assert params == {"factor": 3.0}
        assert component is None

    def test_dump_reads_a_callable_config_returning_a_model(self) -> None:
        """``config`` may be a method returning a BaseModel; both dump as plain data."""

        @register_op("test_op_model_config")
        class _ModelConfigOp:
            config_model = _StrictOpConfig

            def __init__(self, factor: float = 1.0) -> None:
                self.factor = factor

            def config(self) -> _StrictOpConfig:
                return _StrictOpConfig(factor=self.factor)

            @classmethod
            def from_config(cls, config: dict[str, object]) -> "_ModelConfigOp":
                return cls(float(config["factor"]))  # type: ignore[arg-type]

        serializer = get_op_serializer("test_op_model_config")
        params, _ = serializer.dump(_ModelConfigOp(4.0))
        assert params == {"factor": 4.0}

    def test_validate_params_rejects_unknown_keys(self) -> None:
        @register_op("test_op_validate")
        class _ValidatedOp:
            config_model = _StrictOpConfig

            def __init__(self, factor: float = 1.0) -> None:
                self.factor = factor

            @property
            def config(self) -> dict[str, object]:
                return {"factor": self.factor}

            @classmethod
            def from_config(cls, config: dict[str, object]) -> "_ValidatedOp":
                return cls(float(config["factor"]))  # type: ignore[arg-type]

        serializer = get_op_serializer("test_op_validate")
        assert serializer.validate_params is not None
        assert serializer.validate_params({"factor": 2.0}) == {"factor": 2.0}
        with pytest.raises(ValueError, match="smuggled"):
            serializer.validate_params({"factor": 2.0, "smuggled": True})

    def test_load_rebuilds_from_validated_params(self) -> None:
        @register_op("test_op_load")
        class _LoadableOp:
            config_model = _StrictOpConfig

            def __init__(self, factor: float = 1.0) -> None:
                self.factor = factor

            @property
            def config(self) -> dict[str, object]:
                return {"factor": self.factor}

            @classmethod
            def from_config(cls, config: dict[str, object]) -> "_LoadableOp":
                return cls(float(config["factor"]))  # type: ignore[arg-type]

        serializer = get_op_serializer("test_op_load")
        restored = serializer.load({"factor": 5.0}, None, Path("."))
        assert isinstance(restored, _LoadableOp)
        assert restored.factor == 5.0

    def test_load_rejects_a_component_it_cannot_own(self) -> None:
        """A ``@register_op`` Op is component-free; a spec carrying one is corrupt."""

        @register_op("test_op_rejects_component")
        class _ComponentFreeOp:
            config_model = _StrictOpConfig

            @property
            def config(self) -> dict[str, object]:
                return {}

            @classmethod
            def from_config(cls, config: dict[str, object]) -> "_ComponentFreeOp":
                return cls()

        serializer = get_op_serializer("test_op_rejects_component")
        with pytest.raises(ValueError, match="component-free"):
            serializer.load({}, object(), Path("."))


class TestOpSerializerRegistration:
    @staticmethod
    def _dump(op: object) -> tuple[dict[str, object], object | None]:
        return {}, None

    @staticmethod
    def _load(params: dict[str, object], component: object | None, state_dir: Path) -> object:
        return object()

    def test_duplicate_role_is_rejected(self) -> None:
        class _RoleA:
            pass

        class _RoleB:
            pass

        register_op_serializer(
            OpSerializer(role="test_dup_role", op_type=_RoleA, dump=self._dump, load=self._load)
        )
        with pytest.raises(ValueError, match="already registered"):
            register_op_serializer(
                OpSerializer(role="test_dup_role", op_type=_RoleB, dump=self._dump, load=self._load)
            )

    def test_duplicate_op_type_is_rejected(self) -> None:
        """One class, one serializer — a second would silently shadow the first."""

        class _SharedOp:
            pass

        register_op_serializer(
            OpSerializer(
                role="test_dup_type_a", op_type=_SharedOp, dump=self._dump, load=self._load
            )
        )
        with pytest.raises(ValueError, match="already has a registered serializer"):
            register_op_serializer(
                OpSerializer(
                    role="test_dup_type_b", op_type=_SharedOp, dump=self._dump, load=self._load
                )
            )


class TestTrustedLazyOpRoles:
    """The closed trusted-role map may import one module; it never trusts the result."""

    def test_a_trusted_module_that_does_not_register_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point a role at a real, importable module that registers no serializer.
        # Importing it must not leave the role resolvable: the lookup fails closed
        # rather than handing back something the module never claimed.
        monkeypatch.setitem(
            registry_module._LAZY_OP_SERIALIZER_MODULES,
            "test_trusted_but_silent",
            "langres.core.score_type",
        )
        with pytest.raises(UnknownOpType, match="did not register role"):
            get_op_serializer("test_trusted_but_silent")

    def test_an_unknown_role_imports_nothing_and_fails_closed(self) -> None:
        with pytest.raises(UnknownOpType) as exc:
            get_op_serializer("no_such_op_role_123")
        assert "no_such_op_role_123" in str(exc.value)

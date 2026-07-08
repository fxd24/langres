"""Canonical resolution-method NAMES — import-light single source of truth (no heavy deps), so name-listing (`registry.list_methods`) and dispatch (`methods`) agree."""

#: Methods whose scorer makes no API call — fully deterministic and zero-spend.
ZERO_SPEND_METHODS: tuple[str, ...] = ("rapidfuzz", "weighted_average", "embedding_cosine")

#: Methods whose scorer calls an LLM — they take an injected client (mock/real).
#: ``dspy_judge`` and ``select_judge`` are LLM-backed too, but their injected
#: client is a **DSPy LM** (``dspy.LM`` / ``DummyLM``), not the LiteLLM/OpenAI
#: client the others take — see :func:`_make_module_builder`. ``select_judge``
#: (W1.1, ComEM-style set-wise) additionally makes ONE LLM call per anchor
#: GROUP instead of one call per pair — see
#: :class:`~langres.core.modules.select_judge.SelectJudge`.
LLM_METHODS: tuple[str, ...] = ("llm_judge", "cascade", "dspy_judge", "select_judge")

#: Every method the registry can build, in race order.
ALL_METHODS: tuple[str, ...] = ZERO_SPEND_METHODS + LLM_METHODS

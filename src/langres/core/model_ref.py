""":class:`ModelRef`: the ONE backbone concept — what fills a model slot.

An *architecture* is a topology (which components, in what order). A **backbone**
is what fills one of its model slots, and a :class:`ModelRef` is how langres
names one. Swapping a backbone must never mint a new architecture, so a
``ModelRef`` is **weightless by construction**: a few reference *strings*, never
weight bytes, so it round-trips through ``Resolver.save`` / ``load`` as plain
JSON config (:func:`to_config`). The weights are loaded lazily by the backend on
first use.

This module is deliberately import-light (stdlib only — no torch / transformers
/ litellm) so a bare ``import langres`` and the matchers' own imports stay
heavy-dependency free. It imports nothing from ``langres`` and must stay that
way: it is the leaf every model-bearing component depends on.

The ``kind`` discriminator
--------------------------
``kind`` names the *form* of the reference, and it is the **sole input to
routing**. Four forms:

===========  ==========================================  ==================
``kind``     what ``base`` names                         runs
===========  ==========================================  ==================
``api``      a litellm model id (``"openai/gpt-4o"``,    served (litellm)
             ``"gpt-5-mini"``)
``endpoint`` a model served at :attr:`ModelRef.api_base` served (litellm)
             (vLLM / Ollama / any OpenAI-compatible)
``hf``       a Hugging Face Hub id (``"org/name"``)      in-process
``local``    a local directory path                      in-process
===========  ==========================================  ==================

**Routing is a pure function of ``kind`` — never of the filesystem or the
current working directory** (B17). The predecessor of this module probed
``os.path.isdir(ref.base)`` on a *relative* path to decide, which meant a local
``./openai`` directory silently flipped routing litellm -> transformers: the
same saved config resolved to a different backend in a different working
directory. ``kind`` is carried in the config, so a saved artifact resolves
identically anywhere. :func:`backend_for` is the whole routing rule.

Why there is no "did you mean?" on an ``org/name`` typo
------------------------------------------------------
It is tempting to catch ``"opeani/gpt-4o"`` (a typo for the ``openai/``
provider) at construction by fuzzy-matching the org segment against the known
provider list. **This is not safely implementable, and the numbers say so.**
Measured with ``difflib.SequenceMatcher`` against :data:`_API_MODEL_PREFIXES`:

=========================  =========  =======
org segment                closest    ratio
=========================  =========  =======
``opeani``   (a typo)      openai     0.833
``openia``   (a typo)      openai     0.833
``mistralai`` (**REAL**)   mistral    **0.875**
``deepseek-ai`` (**REAL**) deepseek   **0.842**
=========================  =========  =======

The real HF orgs score *higher* than the typos. Any cutoff that catches
``opeani`` also rejects ``mistralai/Mistral-7B-v0.1`` — a wrong answer that
breaks working code, traded for a right answer on a typo that merely 404s. The
ranges overlap, so no threshold exists. ``org/name`` is genuinely ambiguous
between "HF Hub id" and "typo'd provider" and cannot be disambiguated by syntax.

So the escape hatch is the discriminator itself, not a guess: a caller who means
an API model says so (``{"base": "opeani/gpt-4o", "kind": "api"}``) and gets a
litellm error naming the provider. A did-you-mean belongs where a false positive
is *free* — in the failure message of a backend that already failed — never in
the routing decision that precedes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

#: The default embedding backbone. Owned here — beside the other backbone
#: vocabulary and reachable from both ``core.embeddings`` (which builds the
#: embedder) and ``core.method_registry`` (which declares the ``embedding``
#: method's identity) without either importing the other. It previously existed
#: as three copies of the same literal (``method_registry`` + two in
#: ``embeddings``), which is exactly how a default drifts.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: The form of a backbone reference, and the sole input to routing.
BackboneKind = Literal["api", "endpoint", "hf", "local"]

#: Kinds that run **in-process** (transformers): the weights are loaded here.
IN_PROCESS_KINDS: frozenset[BackboneKind] = frozenset({"hf", "local"})

#: Kinds that run **served** (litellm): the weights live behind an API.
SERVED_KINDS: frozenset[BackboneKind] = frozenset({"api", "endpoint"})

#: Model-id prefixes that name a litellm provider, used by
#: :func:`infer_kind` to recognize an ``api`` reference. Not exhaustive and not
#: meant to be: it disambiguates the *common* provider-prefixed ids from HF Hub
#: ``org/name`` ids. An id whose provider is missing here is still reachable —
#: name the ``kind`` explicitly rather than growing this list on every release.
_API_MODEL_PREFIXES: tuple[str, ...] = (
    "openrouter/",
    "openai/",
    "azure/",
    "azure_ai/",
    "anthropic/",
    "gemini/",
    "vertex_ai/",
    "bedrock/",
    "together_ai/",
    "fireworks_ai/",
    "groq/",
    "mistral/",
    "codestral/",
    "cohere/",
    "deepseek/",
    "deepinfra/",
    "perplexity/",
    "xai/",
    "cerebras/",
    "sambanova/",
    "huggingface/",
    "hosted_vllm/",
    "ollama/",
    "ollama_chat/",
    "replicate/",
    "watsonx/",
)

#: Prefixes that make a string *unambiguously* a filesystem path, by **syntax
#: alone**. Deliberately syntactic: ``os.path.isdir`` would reintroduce the
#: CWD-dependent routing this module exists to kill. ``"my-model-dir"`` (a bare
#: relative name) is therefore an ``api`` id, not a path — write ``"./my-model-dir"``
#: or pass ``kind="local"``.
_PATH_PREFIXES: tuple[str, ...] = ("./", "../", "/", "~")


class InvalidModelRefError(ValueError):
    """Raised when a model reference is malformed or its form is unrecognizable.

    Carries an actionable message naming the accepted forms. Raised at
    *construction* — a ``ModelRef`` that exists is always well-formed, so no
    downstream backend has to re-check it.
    """


class UnsupportedBackboneError(ValueError):
    """Raised when a slot cannot run a well-formed backbone.

    Distinct from :class:`InvalidModelRefError`: the reference is *fine*, the
    slot just cannot honor it — e.g. a DSPy-backed matcher handed a ``local``
    directory (DSPy has no in-process route; see
    :mod:`langres.core.matchers.dspy_judge`), or a string-similarity method
    handed any backbone at all.
    """


@dataclass(frozen=True)
class ModelRef:
    """A normalized, weightless reference to a model that fills one slot.

    Args:
        base: The model id or path — an HF Hub id, a local directory, or an API
            model name, per ``kind``.
        kind: The discriminator (see the module docstring). **Required**: a
            discriminator you can forget to set is not a discriminator, and
            routing reads nothing else. Use :func:`normalize_model_ref` to infer
            it from a user-supplied string.
        adapter: An optional PEFT-adapter HF id / local dir applied on top of
            ``base`` at load time (QLoRA served unmerged). In-process kinds only.
        api_base: The served endpoint's URL. Required by — and exclusive to —
            ``kind="endpoint"``.
        revision: The HF Hub git revision (a commit sha, tag, or branch) pinning
            ``base``. ``hf`` only. **Without it an ``org/name`` reference
            drifts**: the same config resolves to different weights as the Hub
            moves, so an "identical versioned config" round-trip is not
            identical across time (B16).

    Raises:
        InvalidModelRefError: Any field combination the table above forbids.
    """

    base: str
    kind: BackboneKind
    adapter: str | None = None
    api_base: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        """Validate the ref itself.

        Validation lives *here*, not only in :func:`normalize_model_ref`: this is
        a public dataclass, so a caller constructing one directly bypasses the
        normalizer entirely. A frozen dataclass that validates nothing is a
        contract in name only — every field combination below is one a backend
        would otherwise discover at load time, or worse, silently ignore.
        """
        if not isinstance(self.base, str) or not self.base:
            raise InvalidModelRefError(
                f"ModelRef.base must be a non-empty string; got {self.base!r}"
            )
        valid_kinds = get_args(BackboneKind)
        if self.kind not in valid_kinds:
            raise InvalidModelRefError(
                f"ModelRef.kind must be one of {', '.join(map(repr, valid_kinds))}; "
                f"got {self.kind!r}"
            )
        if self.adapter is not None:
            if not isinstance(self.adapter, str) or not self.adapter:
                raise InvalidModelRefError(
                    f"ModelRef.adapter must be a non-empty string or None; got {self.adapter!r}"
                )
            if self.kind in SERVED_KINDS:
                raise InvalidModelRefError(
                    f"ModelRef(kind={self.kind!r}) cannot carry an adapter: an unmerged PEFT "
                    "adapter can only be assembled in-process, but this kind runs behind an "
                    "API. Use kind='hf'/'local' to serve base+adapter in-process, or merge the "
                    "adapter into the base weights and reference the merged model."
                )
        if self.kind == "endpoint":
            if not self.api_base:
                raise InvalidModelRefError(
                    "ModelRef(kind='endpoint') requires api_base — the URL the model is served "
                    "at (e.g. 'http://localhost:8000/v1'). Use kind='api' for a hosted provider "
                    "model that needs no endpoint."
                )
        elif self.api_base is not None:
            raise InvalidModelRefError(
                f"ModelRef(kind={self.kind!r}) cannot carry api_base: only kind='endpoint' names "
                f"a served URL. Did you mean ModelRef(base={self.base!r}, kind='endpoint', "
                f"api_base={self.api_base!r})?"
            )
        if self.revision is not None and self.kind != "hf":
            raise InvalidModelRefError(
                f"ModelRef(kind={self.kind!r}) cannot carry a revision: only kind='hf' names a "
                "Hugging Face Hub git revision. A local directory is pinned by its contents, and "
                "an API model is versioned by its id."
            )


def backend_for(kind: BackboneKind) -> Literal["litellm", "transformers"]:
    """Route a backbone ``kind`` to a completion backend. **The whole routing rule.**

    A pure function of the discriminator: no filesystem, no environment, no CWD.
    That is the point — see the module docstring's B17 note.
    """
    return "litellm" if kind in SERVED_KINDS else "transformers"


def require_served(ref: ModelRef, *, slot: str) -> ModelRef:
    """Assert ``ref`` can run behind an API, or raise :class:`UnsupportedBackboneError`.

    The guard for **DSPy-backed slots**, which have no in-process route at all.
    Verified against the installed ``dspy`` (3.2.1), not assumed:
    ``dspy.clients.lm.LM.forward`` routes *every* completion through
    ``litellm_completion`` -> ``litellm.completion``, and ``LM.__init__``
    documents ``model`` as ``"llm_provider/llm_name"``. ``dspy.clients.lm_local``
    looks like an in-process escape hatch but is not one: ``LocalProvider.launch``
    requires ``sglang``, shells out with ``subprocess.Popen``, and then points
    litellm at ``http://localhost:{port}/v1`` — i.e. it *serves* the model and
    goes back through litellm.

    So handing such a slot a ``local`` dir or a base+adapter ref cannot work; it
    dies deep inside litellm with a provider error that names nothing useful.
    Raising here — at construction — is that failure, hoisted to where the caller
    can act on it.

    Args:
        ref: The backbone to check.
        slot: The slot's user-facing name, woven into the message.

    Raises:
        UnsupportedBackboneError: ``ref.kind`` is an in-process kind.
    """
    if ref.kind in SERVED_KINDS:
        return ref
    detail = (
        f"an unmerged base+adapter ref (base={ref.base!r}, adapter={ref.adapter!r})"
        if ref.adapter is not None
        else f"a {ref.kind!r} backbone ({ref.base!r})"
    )
    raise UnsupportedBackboneError(
        f"{slot} cannot run {detail}: it is DSPy-backed, and DSPy routes every "
        "completion through litellm — it has no in-process route, so local weights and "
        "PEFT adapters are unreachable from this slot.\n"
        "Fix A: serve the model (e.g. `vllm serve <model>`) and pass the endpoint — "
        '{"base": "<served-id>", "kind": "endpoint", "api_base": "http://localhost:8000/v1"}.\n'
        "Fix B: use LLMMatcher instead, which has a transformers backend and runs "
        "hf/local/base+adapter refs in-process.\n"
        "Fix C: name an API model (e.g. 'openrouter/openai/gpt-4o-mini')."
    )


def infer_kind(base: str, *, api_base: str | None = None) -> BackboneKind:
    """Infer a :data:`BackboneKind` from a bare model string, by **syntax alone**.

    The rules, unambiguous cases first:

    1. ``api_base`` given -> ``endpoint``.
    2. an explicit path prefix (``./``, ``../``, ``/``, ``~``) -> ``local``.
    3. a known provider prefix (:data:`_API_MODEL_PREFIXES`) -> ``api``.
    4. no ``/`` -> ``api`` (a bare litellm id like ``"gpt-5-mini"``).
    5. exactly one ``/`` -> ``hf`` (an ``org/name`` Hub id).
    6. otherwise -> :class:`InvalidModelRefError`.

    Rule 6 is the "unknown forms raise" case, and it is deliberately the *only*
    one: a multi-slash id with no known provider (``"foo/bar/baz"``) is neither a
    valid Hub id (those carry exactly one slash) nor a provider id we can route,
    so guessing would only defer the error to a 404. By contrast rule 5's
    ``org/name`` genuinely *is* a well-formed Hub id even when the caller meant a
    provider — see the module docstring on why that cannot be second-guessed.

    Raises:
        InvalidModelRefError: An empty string, or an unrecognizable form.
    """
    if not isinstance(base, str) or not base:
        raise InvalidModelRefError(f"model string must be non-empty; got {base!r}")
    if api_base is not None:
        return "endpoint"
    if base.startswith(_PATH_PREFIXES):
        return "local"
    if base.startswith(_API_MODEL_PREFIXES):
        return "api"
    slashes = base.count("/")
    if slashes == 0:
        return "api"
    if slashes == 1:
        return "hf"
    raise InvalidModelRefError(
        f"cannot infer a backbone kind for {base!r}: it has {slashes} '/' separators, so it is "
        "neither a Hugging Face Hub id ('org/name') nor an id starting with a known provider "
        f"prefix ({', '.join(_API_MODEL_PREFIXES[:4])}, ...). Name the kind explicitly, e.g. "
        f'{{"base": {base!r}, "kind": "api"}} to route it to litellm, or "kind": "local" for a '
        "directory path."
    )


def normalize_model_ref(
    model: str | dict[str, str] | ModelRef, *, api_base: str | None = None
) -> ModelRef:
    """Coerce a user-supplied model reference into a validated :class:`ModelRef`.

    Accepts the three surface forms:

    - ``str`` — the kind is inferred by :func:`infer_kind`.
    - ``dict`` — must carry a non-empty ``"base"``; ``"kind"`` is honored when
      present and inferred otherwise; ``"adapter"``, ``"api_base"`` and
      ``"revision"`` are optional.
    - :class:`ModelRef` — returned unchanged (idempotent), since it is already
      validated by construction.

    Args:
        model: The reference in any of the three forms above.
        api_base: A served endpoint URL supplied *alongside* the model (the
            legacy parallel kwarg — see :class:`~langres.core.matchers.llm_judge.LLMMatcher`).
            It is absorbed into the ref as the ``endpoint`` form, which is why
            ``api_base`` is not a second model concept any more.

    Raises:
        InvalidModelRefError: A malformed ref, an unrecognizable form, or an
            ``api_base`` that contradicts the one already on the ref.
        TypeError: Any other type.
    """
    if isinstance(model, ModelRef):
        if api_base is not None and model.api_base != api_base:
            raise InvalidModelRefError(
                f"conflicting api_base: the ModelRef names {model.api_base!r} but api_base="
                f"{api_base!r} was passed alongside it. Pass one or the other, not both."
            )
        return model
    if isinstance(model, str):
        return ModelRef(base=model, kind=infer_kind(model, api_base=api_base), api_base=api_base)
    if isinstance(model, dict):
        base = model.get("base")
        if not isinstance(base, str) or not base:
            raise InvalidModelRefError(
                f"model dict must carry a non-empty string 'base'; got {model!r}"
            )
        adapter = model.get("adapter")
        if adapter is not None and not isinstance(adapter, str):
            raise InvalidModelRefError(
                f"model dict 'adapter' must be a string or absent; got {adapter!r}"
            )
        revision = model.get("revision")
        if revision is not None and not isinstance(revision, str):
            raise InvalidModelRefError(
                f"model dict 'revision' must be a string or absent; got {revision!r}"
            )
        ref_api_base = model.get("api_base")
        if ref_api_base is not None and api_base is not None and ref_api_base != api_base:
            raise InvalidModelRefError(
                f"conflicting api_base: the model dict names {ref_api_base!r} but api_base="
                f"{api_base!r} was passed alongside it. Pass one or the other, not both."
            )
        resolved_api_base = ref_api_base or api_base
        kind = model.get("kind")
        if kind is None:
            # An unmerged adapter can only be assembled in-process, so a dict
            # naming one is an in-process ref even when `base` looks like a bare
            # litellm id. Inference sees only `base`, so it cannot know that.
            inferred = infer_kind(base, api_base=resolved_api_base)
            kind = "hf" if adapter is not None and inferred == "api" else inferred
        return ModelRef(
            base=base,
            kind=kind,  # type: ignore[arg-type]  # validated in __post_init__
            adapter=adapter,
            api_base=resolved_api_base,
            revision=revision,
        )
    raise TypeError(f"model must be a str, dict, or ModelRef; got {type(model).__name__}")


def to_config(ref: ModelRef) -> str | dict[str, str]:
    """Serialize a :class:`ModelRef` for ``config`` (the inverse of :func:`normalize_model_ref`).

    Emits the **compact string form** (a bare ``base``) exactly when that string
    round-trips back to an equal ref — i.e. when the kind is what
    :func:`infer_kind` would have guessed and no other field is set. That keeps
    the common case **byte-identical to the pre-``kind`` string config**, so
    existing saved artifacts and their round-trips are unchanged, while any ref
    that inference could not reproduce widens to an explicit dict.

    The invariant, pinned by test: ``normalize_model_ref(to_config(ref)) == ref``
    for every ref.
    """
    try:
        inferable = infer_kind(ref.base) == ref.kind
    except InvalidModelRefError:
        # `base` is a form inference cannot name (e.g. a multi-slash id routed by
        # an explicit kind). It must carry its kind, or it would not survive load.
        inferable = False
    if inferable and ref.adapter is None and ref.api_base is None and ref.revision is None:
        return ref.base
    config: dict[str, str] = {"base": ref.base, "kind": ref.kind}
    if ref.adapter is not None:
        config["adapter"] = ref.adapter
    if ref.api_base is not None:
        config["api_base"] = ref.api_base
    if ref.revision is not None:
        config["revision"] = ref.revision
    return config

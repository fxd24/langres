"""DSPyMatcher: a serializable, compilable DSPy ChainOfThought entity-matching Matcher.

This is the M4 "learnable scorer seam": a :class:`~langres.core.matcher.Matcher`
whose match decision comes from a DSPy ``ChainOfThought`` program over a typed
:class:`PairwiseMatchSignature`. Because the program is a DSPy artifact it can be
**compiled** against a gold set (``BootstrapFewShot`` / ``MIPROv2`` / ``GEPA``)
to tune the prompt from data — the direct answer to M3's finding that a cheap
judge's precision collapses under a generic, hand-written prompt.

It mirrors :class:`~langres.core.matchers.llm_judge.LLMMatcher`'s serializable shape
so a Resolver with a DSPyMatcher in the ``module`` slot can ``save`` / ``load``:

- pure :attr:`config` (``model`` / ``temperature`` / ``entity_noun`` — never the
  ``dspy.LM`` client or the program bytes);
- :class:`~langres.core.serialization.SerializableState` for the compiled program
  (``program.json`` via ``program.save`` / ``program.load``);
- lazy LM construction (an injected ``lm`` — e.g. ``DummyLM`` in tests — wins;
  otherwise a ``dspy.LM`` is built on first use), so ``load`` never needs a key.

**Import-safety:** this module is deliberately *not* eager-imported by
``langres.core`` — importing ``dspy`` opens a disk cache and is undesirable on
plain ``import langres.core``. It is registered lazily via
``registry._LAZY_COMPONENT_MODULES`` so ``Resolver.load`` on a ``dspy_judge``
artifact imports it on demand (firing ``@register``).
"""

import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self

import dspy
from dspy.utils.exceptions import AdapterParseError

from langres.core.model_ref import (
    ModelRef,
    normalize_model_ref,
    require_litellm_routable,
    to_config,
)
from langres.core.models import ERCandidate, PairwiseJudgement
from langres.core.matcher import Matcher, SchemaT
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.tracking.runs import RunContext, RunStore, capture_run
from langres.tracking.trackers import ExperimentTracker, resolve_tracker
from langres.core.usage import LLMUsage

logger = logging.getLogger(__name__)


class PairwiseMatchSignature(dspy.Signature):  # type: ignore[misc]  # dspy is untyped (Any)
    """Decide whether two {entity_noun} records refer to the same real-world {entity_noun}.

    Weigh the concrete evidence in both records, not surface overlap. A different
    model, size, edition, or variant means a *different* {entity_noun}, even when
    titles look similar; conversely, differently-worded records can be the same
    {entity_noun}. Return the boolean decision, a calibrated probability that the
    two records are the same {entity_noun} (1.0 = certainly same, 0.0 = certainly
    different), and a brief justification.
    """

    left: str = dspy.InputField(desc="First record (rendered fields)")
    right: str = dspy.InputField(desc="Second record (rendered fields)")
    match: bool = dspy.OutputField(desc="True if the two records are the same entity")
    match_probability: float = dspy.OutputField(desc="Probability in [0, 1] they are the same")
    reasoning: str = dspy.OutputField(desc="Brief justification for the decision")


def _signature_for(entity_noun: str) -> Any:
    """Return :class:`PairwiseMatchSignature` with ``entity_noun`` woven into its instructions.

    The signature's docstring is the optimizable instruction (with ``{entity_noun}``
    placeholders); this substitutes the domain noun at construction so the same
    signature serves products, companies, people, etc. Compilation later rewrites
    these instructions from data.
    """
    instructions = PairwiseMatchSignature.instructions.replace("{entity_noun}", entity_noun)
    return PairwiseMatchSignature.with_instructions(instructions)


def _clamp01(value: float) -> float:
    """Clamp ``value`` into the ``[0.0, 1.0]`` range required by ``PairwiseJudgement``."""
    return max(0.0, min(1.0, value))


def _salvage_usage(lm: Any) -> tuple[int, int]:
    """Best-effort ``(prompt_tokens, completion_tokens)`` for a call that WAS billed.

    On an :class:`AdapterParseError` there is no ``Prediction`` to call
    ``get_lm_usage`` on, yet the underlying LM completion already ran (and was
    billed). DSPy records every call in ``lm.history``, so read the last entry's
    usage to cost the failed call instead of silently treating it as free.
    Returns ``(0, 0)`` when no usage is recoverable (e.g. a stub LM with no
    history), which keeps the honest-cost math a no-op rather than crashing.
    """
    history = getattr(lm, "history", None)
    if not history:
        return 0, 0
    usage = history[-1].get("usage") or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def _pair_metric(example: dspy.Example, prediction: Any, trace: Any = None) -> bool:
    """Compilation metric: the predicted ``match`` bool equals the gold ``match`` bool.

    Used by both ``BootstrapFewShot`` and ``MIPROv2`` — a demonstration/instruction
    is kept only when it reproduces the labeled match decision.
    """
    return bool(prediction.match) == bool(example.match)


def _gepa_metric(
    example: dspy.Example,
    prediction: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
) -> float:
    """GEPA-shaped compilation metric: 1.0 when the predicted ``match`` is correct.

    ``dspy.GEPA`` validates (in its constructor) that the metric accepts *five*
    arguments — ``(gold, pred, trace, pred_name, pred_trace)`` — and rejects the
    three-argument :func:`_pair_metric` that ``BootstrapFewShot`` / ``MIPROv2``
    take. This is the same match-decision metric, adapted to that signature and
    returning a scalar score (``1.0``/``0.0``) GEPA can rank on its Pareto
    frontier.

    A *feedback-returning* metric — ``dspy.Prediction(score=..., feedback=...)``,
    letting GEPA reflect on *why* a pair was misjudged rather than only on the
    score — would sharpen the reflection, but it is a deliberate future
    enhancement, not built now (simplicity-first): the scalar path already
    exercises the full reflective-evolution loop.
    """
    return 1.0 if _pair_metric(example, prediction) else 0.0


def _trainset_fingerprint(trainset: Sequence[dspy.Example]) -> str:
    """Content-address a ``trainset`` so compiles on DIFFERENT labels get DIFFERENT ids.

    Each ``dspy.Example`` is reduced to its field dict (``toDict``) and dumped to
    canonical JSON (sorted keys, no whitespace); the per-example strings are then
    sorted before hashing, so an identical labeled set fingerprints identically
    regardless of row order, while any content change — a flipped label, an edited
    field, an added/removed example — changes the digest. Fed to the compile
    :class:`~langres.tracking.runs.RunContext`'s ``dataset_fingerprint`` so two
    ``compile`` runs on different trainsets no longer collapse to the same
    ``recipe_id``. Mirrors :func:`langres.tracking.runs.dataset_fingerprint`'s style.
    """
    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(example.toDict(), sort_keys=True, separators=(",", ":")) for example in trainset
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@register("dspy_judge")
class DSPyMatcher(Matcher[SchemaT]):
    """DSPy ``ChainOfThought`` entity-matching scorer — compilable and serializable.

    Example:
        # Zero-spend: inject a DummyLM, compile against a gold set, then score.
        from dspy.utils.dummies import DummyLM

        judge = DSPyMatcher(lm=DummyLM([...]), entity_noun="product")
        judge.compile(trainset, optimizer="bootstrap")
        for j in judge.forward(candidates):
            print(j.score, j.reasoning, j.provenance["cost_usd"])
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "dspy_judge"

    def __init__(
        self,
        lm: Any = None,
        model: str | dict[str, str] | ModelRef = "openrouter/openai/gpt-4o-mini",
        temperature: float = 0.0,
        entity_noun: str = "entity",
        program: Any = None,
    ) -> None:
        """Initialize a DSPyMatcher.

        Args:
            lm: Optional pre-built DSPy LM (``dspy.LM`` or ``DummyLM``). When
                ``None`` a ``dspy.LM(model, cache=False)`` is built lazily on first
                use — so ``from_config`` / ``load`` never need a key. Inject an LM
                as an escape hatch (tests inject ``DummyLM`` for zero-spend runs).
            model: The backbone that fills this slot — an API model id, or any
                :class:`~langres.core.model_ref.ModelRef` surface form. **litellm-routable
                refs only**: this slot is DSPy-backed, so a ``local`` directory or
                a base+adapter ref raises
                :class:`~langres.core.model_ref.UnsupportedBackboneError` here —
                see :func:`~langres.core.model_ref.require_litellm_routable`.
            temperature: Sampling temperature for the lazily-built LM.
            entity_noun: Domain noun woven into the signature instructions.
            program: Optional pre-built DSPy program (e.g. a compiled one). When
                ``None`` a fresh ``ChainOfThought`` over the woven signature is
                built (uncompiled — call :meth:`compile` to tune it).

        Raises:
            UnsupportedBackboneError: ``model`` names an in-process backbone.
        """
        self.model_ref = require_litellm_routable(normalize_model_ref(model), slot="DSPyMatcher")
        # ``self.model`` stays the base id *string* for litellm/provenance/pricing;
        # the full ref (incl. any endpoint URL) lives on ``self.model_ref``.
        self.model = self.model_ref.base
        self.temperature = temperature
        self.entity_noun = entity_noun
        self._lm = lm
        self._program: Any = (
            program if program is not None else dspy.ChainOfThought(_signature_for(entity_noun))
        )
        # A program is "compiled" once :meth:`compile` has tuned it; a bare
        # ``ChainOfThought`` (or a ``from_config`` rebuild) starts uncompiled and
        # :meth:`forward` warns so a user never silently benchmarks an untuned judge.
        self._compiled = False
        # Cost seam: honest per-pair cost = tokens * price. OpenRouter is priced at
        # $0 by litellm, so we never trust ``completion_cost``; instead we multiply
        # real token counts by a pinned price. Default 0.0 keeps zero-spend runs
        # (e.g. a ``DummyLM`` in tests) at $0; the real pinned price is set after
        # construction — the ``dspy_judge`` builder in ``langres.methods`` wires it
        # from the ``langres.clients.openrouter`` price table (unknown models stay
        # $0), or a caller may set ``judge.price_per_1k_tokens`` directly.
        self.price_per_1k_tokens = 0.0
        # Lineage carrier: :meth:`compile` records the compilation as a tracked
        # optimization run and stamps its ``attempt_id`` here, so a later
        # ``capture_run`` (the eval run using this compiled program) can read it
        # into ``parent_run_id`` — otherwise the compile→eval lineage stays None.
        self._compile_run_id: str | None = None

    def _get_lm(self) -> Any:
        """Return the DSPy LM, lazily building a ``dspy.LM`` from ``model`` on first use.

        ``api_base`` is forwarded for the ``endpoint`` kind: ``dspy.LM`` passes
        unknown kwargs straight to litellm (it lists ``api_base`` among the args it
        excludes from its cache key), so a served backbone needs no new judge.
        """
        if self._lm is None:
            extra = {"api_base": self.model_ref.api_base} if self.model_ref.api_base else {}
            self._lm = dspy.LM(self.model, cache=False, temperature=self.temperature, **extra)
        return self._lm

    @property
    def compiled(self) -> bool:
        """Whether the program has been tuned by :meth:`compile` (public read-only)."""
        return self._compiled

    @property
    def n_demos(self) -> int:
        """Total bootstrapped few-shot demos across the program's predictors.

        ``0`` before :meth:`compile` (a bare ``ChainOfThought`` carries none);
        the count the ``FitReport`` surfaces after a prompt-optimize fit as "what
        the compile learned". Mirrors the demo-count probe :meth:`load_state`
        uses to infer compilation on markerless artifacts.
        """
        return sum(len(predictor.demos) for _, predictor in self._program.named_predictors())

    @property
    def config(self) -> dict[str, object]:
        """Pure, serializable construction config (never the LM, program, or secrets)."""
        return {
            # A plain API id serializes byte-identically to the pre-``kind`` config;
            # only an endpoint ref widens to a dict (see ``to_config``).
            "model": to_config(self.model_ref),
            "temperature": self.temperature,
            "entity_noun": self.entity_noun,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "DSPyMatcher[SchemaT]":
        """Rebuild a fresh (uncompiled) judge from :attr:`config`.

        The program is uncompiled; :meth:`load_state` overwrites it with the saved
        (possibly compiled) program during ``Resolver.load``.
        """
        return cls(
            lm=None,
            # Passed through unchanged: a ``str()`` coercion here would stringify an
            # endpoint ref's dict into ``"{'base': ...}"`` and silently misroute it.
            model=config["model"],  # type: ignore[arg-type]
            temperature=float(config["temperature"]),  # type: ignore[arg-type]
            entity_noun=str(config["entity_noun"]),
        )

    def _cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Honest per-pair cost from token counts and the pinned per-1k price."""
        return (prompt_tokens + completion_tokens) / 1000.0 * self.price_per_1k_tokens

    def _render_entity(self, entity: SchemaT) -> str:
        """Render an entity for the prompt (LLMMatcher's JSON convention)."""
        return entity.model_dump_json(indent=2)

    def examples_from_candidates(
        self, candidates: Sequence[ERCandidate[SchemaT]], labels: Sequence[bool]
    ) -> list[dspy.Example]:
        """Build a :meth:`compile` trainset from labeled candidates.

        Each pair becomes a ``dspy.Example`` whose ``left`` / ``right`` inputs are
        rendered *exactly* as :meth:`forward` renders them (via
        :meth:`_render_entity`), so the demos the optimizer learns reflect what the
        program sees at inference; ``match`` carries the gold label. This is the
        candidate->``dspy.Example`` bridge ``Resolver.fit(method=<prompt>)`` uses
        to feed :meth:`compile`, keeping the rendering convention owned by the
        matcher rather than duplicated at the call site.

        Args:
            candidates: Blocked candidate pairs to train on, positionally aligned
                with ``labels`` (e.g. ``align_pairs(...).train.candidates``).
            labels: Gold match/non-match labels for each candidate.

        Returns:
            One ``dspy.Example`` per candidate, inputs marked ``left`` / ``right``.
        """
        return [
            dspy.Example(
                left=self._render_entity(candidate.left),
                right=self._render_entity(candidate.right),
                match=bool(label),
            ).with_inputs("left", "right")
            for candidate, label in zip(candidates, labels, strict=True)
        ]

    def forward(self, candidates: Iterator[ERCandidate[SchemaT]]) -> Iterator[PairwiseJudgement]:
        """Score each candidate pair with the DSPy program, yielding PairwiseJudgements.

        Each call runs the program under ``dspy.context(lm=..., track_usage=True)``
        — never ``dspy.settings.configure`` — so the global LM is left untouched and
        real token usage is captured for honest cost. A DSPy parse/validation error
        does not skip the pair: it yields a contract-correct abstention
        (``decision=None, score=None`` — so ``is_abstain`` is ``True`` — with
        ``provenance["parse_error"] = True``) tagged ``dspy_parse_error`` and the
        error recorded in provenance. ``predicted_match`` therefore *excludes* the
        pair from the predicted set (never a silent match), and the evaluator's
        abstention accounting (``core.benchmark``, ``n_abstained``) counts it — an
        honest "I don't know", not a fabricated verdict.
        """
        if not self._compiled:
            logger.warning(
                "DSPyMatcher.forward is running on an UNCOMPILED program — the prompt "
                "is untuned. Call compile(trainset) before benchmarking to avoid "
                "silently scoring with an untuned judge."
            )
        lm = self._get_lm()
        for candidate in candidates:
            left = self._render_entity(candidate.left)
            right = self._render_entity(candidate.right)
            left_id = candidate.left.id  # type: ignore[attr-defined]
            right_id = candidate.right.id  # type: ignore[attr-defined]
            try:
                with dspy.context(lm=lm, track_usage=True):
                    prediction = self._program(left=left, right=right)
            except AdapterParseError as error:
                logger.warning(
                    "DSPyMatcher parse failure for %s vs %s: %s", left_id, right_id, error
                )
                # The LM completion WAS billed even though parsing failed. Salvage
                # whatever token counts DSPy recorded and flag the cost as
                # untrackable (``cost_untracked``) so downstream accounting does not
                # silently undercount once a real price is wired to the token seam.
                err_prompt_tokens, err_completion_tokens = _salvage_usage(lm)
                err_usage = LLMUsage(
                    input_tokens=err_prompt_tokens,
                    output_tokens=err_completion_tokens,
                    model=self.model,
                )
                yield PairwiseJudgement(
                    left_id=left_id,
                    right_id=right_id,
                    # A parse failure is a genuine abstention: null the verdict
                    # (decision=None default, score=None) so is_abstain is True and
                    # predicted_match excludes it -- never a silent 0.0/0.5 match.
                    score=None,
                    score_type="prob_llm",
                    decision_step="dspy_parse_error",
                    reasoning=None,
                    provenance={
                        "model": self.model,
                        "cost_usd": self._cost_usd(err_prompt_tokens, err_completion_tokens),
                        "prompt_tokens": err_prompt_tokens,
                        "completion_tokens": err_completion_tokens,
                        "cost_untracked": True,
                        "parse_error": True,
                        "error": str(error),
                        "usage": err_usage.model_dump(),
                    },
                )
                continue

            # {model: {prompt_tokens, completion_tokens, *_details, ...}}
            usage = LLMUsage.from_lm_usage(prediction.get_lm_usage(), model=self.model)
            yield PairwiseJudgement(
                left_id=left_id,
                right_id=right_id,
                score=_clamp01(float(prediction.match_probability)),
                score_type="prob_llm",
                decision_step="dspy_judgment",
                reasoning=prediction.reasoning,
                provenance={
                    "model": self.model,
                    "cost_usd": self._cost_usd(usage.input_tokens, usage.output_tokens),
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "usage": usage.model_dump(),
                },
            )

    def compile(
        self,
        trainset: Sequence[dspy.Example],
        valset: Sequence[dspy.Example] | None = None,
        *,
        optimizer: str = "bootstrap",
        auto: str = "light",
        reflection_model: str | None = None,
        reflection_minibatch_size: int = 3,
        max_metric_calls: int | None = None,
        tracker: ExperimentTracker | None = None,
        store: str | Path | RunStore | None = None,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Compile (tune) the DSPy program against a gold set, in place.

        The compilation is recorded as a first-class **optimization run** via
        :func:`~langres.tracking.runs.capture_run`, and its ``attempt_id`` is stamped
        onto :attr:`_compile_run_id` so a later ``capture_run`` (the eval run that
        uses the compiled program) can thread it into ``parent_run_id`` for the
        compile→eval lineage. Persistence is opt-in: with the default
        ``store=None`` / ``tracker=None`` (resolved to a no-op) nothing is written
        and the compiled program is byte-identical to the un-tracked path — only
        the in-memory ``_compile_run_id`` carrier is stamped.

        Args:
            trainset: Labeled ``dspy.Example`` s (``left`` / ``right`` inputs +
                gold ``match``) the optimizer tunes against.
            valset: Optional validation set (used by ``mipro``).
            optimizer: ``"bootstrap"`` (``BootstrapFewShot`` — deterministic under
                ``DummyLM``, the zero-spend path), ``"mipro"`` (``MIPROv2`` — the
                paid path, exercised only by the example), or ``"gepa"``
                (``dspy.GEPA`` — reflective Genetic-Pareto instruction evolution;
                runs zero-spend under ``DummyLM`` for both the student and the
                reflection LM).
            auto: Search-budget preset (``"light"`` / ``"medium"`` / ``"heavy"``)
                for ``"mipro"`` and ``"gepa"``; ignored by ``"bootstrap"``, and
                by ``"gepa"`` when ``max_metric_calls`` is given. Threaded from
                the method's ``auto`` field by ``Resolver.fit``.
            reflection_model: ``"gepa"`` only — LM id for GEPA's reflection step.
                ``None`` reuses this matcher's own LM (:meth:`_get_lm`), which is
                what keeps the ``DummyLM`` path zero-spend while still satisfying
                GEPA's required-reflection-LM contract.
            reflection_minibatch_size: ``"gepa"`` only — examples reflected over
                per step (``dspy.GEPA`` default 3).
            max_metric_calls: ``"gepa"`` only — precise metric-call budget; when
                set it supersedes ``auto`` (``dspy.GEPA`` takes exactly one budget
                knob). ``None`` falls back to the ``auto`` preset.
            tracker: Experiment tracker for the compile run (``None`` — the
                default — resolves to a no-op via ``resolve_tracker``).
            store: Where to persist the compile :class:`RunRecord` (default: none).
            parent_run_id: Optional parent run this compilation belongs to (e.g. a
                sweep) — recorded on the compile run's context.
            **kwargs: Forwarded to the optimizer's ``compile`` (the tracking
                params above are bound explicitly, so they never leak into it).

        Returns:
            ``self`` with ``_program`` replaced by the compiled program,
            ``_compiled`` set, and ``_compile_run_id`` stamped — so callers can
            chain ``judge.compile(...).forward(...)``.
        """
        if optimizer not in ("bootstrap", "mipro", "gepa"):
            raise ValueError(
                f"unknown optimizer {optimizer!r}; choose 'bootstrap', 'mipro', or 'gepa'"
            )
        tracker = resolve_tracker(tracker)
        context = RunContext(
            experiment="dspy_compile",
            method="dspy_compile",
            dataset_name="dspy_trainset",
            # Content-address the labels so two compiles on DIFFERENT trainsets
            # get DIFFERENT recipe_ids (dataset_fingerprint feeds compute_recipe_id).
            # Without it, a constant dataset_name collapsed distinct compiles to
            # one id, letting a store-based replay guard treat different labels as
            # the same run.
            dataset_fingerprint=_trainset_fingerprint(trainset),
            llm_model=self.model,
            parent_run_id=parent_run_id,
            resolver_config={
                "type_name": self.type_name,
                "optimizer": optimizer,
                **self.config,
            },
        )
        # NOTE: this compile run records $0 spend; capturing paid DSPy-compile
        # spend (the ``spend_usd`` seam) is deferred to issue #100 (cost-tracking).
        with capture_run(context, store=store, tracker=tracker) as run:
            with dspy.context(lm=self._get_lm()):
                if optimizer == "bootstrap":
                    self._program = dspy.BootstrapFewShot(metric=_pair_metric).compile(
                        self._program, trainset=list(trainset), **kwargs
                    )
                elif optimizer == "gepa":
                    # Reflective Genetic-Pareto evolution: GEPA runs the program on
                    # the trainset, reflects (in natural language, via reflection_lm)
                    # on the traces, and evolves the *instruction* on a Pareto
                    # frontier. Two hard requirements from dspy.GEPA's constructor:
                    #  (1) the metric must be 5-arg (_gepa_metric, not _pair_metric);
                    #  (2) reflection_lm must be non-None -- reuse this matcher's own
                    #      LM when unset (so a DummyLM student also reflects at $0).
                    # GEPA also takes EXACTLY ONE budget knob, so pass max_metric_calls
                    # when given, else the auto preset.
                    reflection_lm = (
                        dspy.LM(reflection_model) if reflection_model else self._get_lm()
                    )
                    gepa_budget: dict[str, int | str] = (
                        {"max_metric_calls": max_metric_calls}
                        if max_metric_calls is not None
                        else {"auto": auto}
                    )
                    self._program = dspy.GEPA(
                        metric=_gepa_metric,
                        reflection_lm=reflection_lm,
                        reflection_minibatch_size=reflection_minibatch_size,
                        **gepa_budget,
                    ).compile(
                        self._program,
                        trainset=list(trainset),
                        valset=list(valset) if valset is not None else None,
                    )
                else:  # "mipro"  # pragma: no cover - paid, non-deterministic path
                    # Exercised only by the paid example (MIPROv2 proposes+evaluates
                    # instructions via real LM calls; it is not deterministic under
                    # DummyLM, so it is kept out of the zero-spend unit suite).
                    self._program = dspy.MIPROv2(metric=_pair_metric, auto=auto).compile(
                        self._program,
                        trainset=list(trainset),
                        valset=list(valset) if valset is not None else None,
                        **kwargs,
                    )
        self._compiled = True
        # Stamp the lineage carrier AFTER a successful compile (a failed compile
        # propagates out of ``capture_run`` before this line, so it never stamps).
        self._compile_run_id = run.attempt_id
        return self

    #: Sidecar file recording whether the saved program was compiled, so a reload
    #: restores the real compiled flag instead of assuming a saved judge is tuned.
    _COMPILED_MARKER: ClassVar[str] = "compiled"

    def save_state(self, state_dir: Path) -> None:
        """Persist the DSPy program (``program.json``) plus the compiled marker."""
        self._program.save(state_dir / "program.json", save_program=False)
        (state_dir / self._COMPILED_MARKER).write_text("true" if self._compiled else "false")

    def load_state(self, state_dir: Path) -> None:
        """Restore the DSPy program from ``program.json`` written by :meth:`save_state`.

        The compiled flag is restored from the sidecar marker so it reflects
        reality: a judge saved BEFORE :meth:`compile` reloads as *uncompiled* (and
        ``forward`` still warns), not silently marked tuned. Older artifacts
        without the marker fall back to inferring compilation from whether the
        reloaded program carries bootstrapped demos.
        """
        self._program.load(state_dir / "program.json")
        marker = state_dir / self._COMPILED_MARKER
        if marker.exists():
            self._compiled = marker.read_text().strip() == "true"
        else:
            self._compiled = any(len(p.demos) > 0 for _, p in self._program.named_predictors())

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground-truth labels (shared report helper)."""
        return _inspect_scores_impl(judgements, sample_size)

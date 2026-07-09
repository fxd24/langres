"""SelectJudge: a ComEM-style set-wise judge (W1.1, S1 — the set-wise keystone).

Ordinary judges (``LLMJudge``, ``DSPyJudge``, ...) make ONE LLM call per pair:
K candidates scored against an anchor cost K calls. ``SelectJudge`` is a
:class:`~langres.core.module.GroupwiseModule`: it makes ONE LLM call per
*group* -- "which one of these K candidates (if any) matches the anchor?" --
and decomposes the single answer into K :class:`PairwiseJudgement`\\ s. This
mirrors the "selecting" strategy from Wang et al., "Match, Compare, or
Select? An Investigation of Large Language Models for Entity Matching"
(COLING 2025, the ComEM paper): selecting identifies **the single** best
match from a candidate list, not an arbitrary subset -- reusing the DSPy
signature's ``list[str]`` output field (to score by id membership like the
illustrative skeleton in ``docs/TECHNICAL_OVERVIEW.md``) but enforcing a
0-or-1 selection contract in code.

Set-wise IN, pairwise OUT: :meth:`forward_groups` is the only method a caller
needs; :meth:`~langres.core.module.GroupwiseModule.forward` (inherited,
concrete) makes this a drop-in ``Module`` for the Resolver spine, benchmark
dispatch, and ``inspect_scores`` -- zero downstream changes.

**select_error handling (CEO #12):** the LLM's single-call answer can go
wrong in three ways a per-pair judge never has to consider, because they are
properties of the WHOLE selection, not one pair: the response fails to parse
into the signature's typed fields at all (``AdapterParseError``), the
selection names a candidate id that is not a member of the group, or the
selection names more than one candidate when the contract is "at most one".
None of these raise mid-stream -- each maps the WHOLE group to "no match"
judgements carrying ``provenance["select_error"]`` describing what went
wrong, so a bad LLM turn degrades one group's judgements rather than killing
the run.

**Group-call cost convention (E5):** the one LLM call's ``cost_usd`` is
stamped on the FIRST judgement of the group via
:func:`~langres.core.module.stamp_group_cost`; every sibling (including
select_error siblings) carries ``$0``, and ``provenance["group_id"]`` is set
on all of them, so cost aggregation downstream sums a group to exactly one
call's cost.

**Import-safety:** not eager-imported by ``langres.core`` (mirrors
``DSPyJudge`` -- importing ``dspy`` opens a disk cache on import).

**Serialization:** registered under ``"select_judge"`` (lazily -- see
``langres.core.registry._LAZY_COMPONENT_MODULES``) with a pure ``config``
(``model``/``temperature``/``entity_noun``, never the DSPy LM), mirroring
``DSPyJudge``'s no-pickle config-registry contract exactly. Unlike
``DSPyJudge``, there is no ``compile()``/out-of-band program state here (no
``save_state``/``load_state``): ``from_config`` alone rebuilds an equivalent,
uncompiled judge, the same as any other stateless judge (``LLMJudge``,
``WeightedAverageJudge``).
"""

import json
import logging
from collections.abc import Iterator
from typing import Any, ClassVar

import dspy
from dspy.utils.exceptions import AdapterParseError

from langres.core.groups import ERCandidateGroup
from langres.core.models import PairwiseJudgement
from langres.core.module import GroupwiseModule, SchemaT, stamp_group_cost
from langres.core.modules.dspy_judge import _salvage_usage
from langres.core.registry import register
from langres.core.reports import ScoreInspectionReport, _inspect_scores_impl
from langres.core.usage import LLMUsage

logger = logging.getLogger(__name__)


class SelectSignature(dspy.Signature):  # type: ignore[misc]  # dspy is untyped (Any)
    """From K candidate {entity_noun} records, select the id of the single record
    that refers to the same real-world {entity_noun} as the anchor record, if any
    of them does. Weigh the concrete evidence in each candidate, not surface
    overlap -- a different model, size, edition, or variant is a *different*
    {entity_noun} even when descriptions look similar. Return the id of the ONE
    matching candidate as a single-element list, or an empty list if none of the
    candidates match the anchor. Never select more than one candidate.
    """

    anchor: str = dspy.InputField(desc="The reference record (rendered fields)")
    candidates: str = dspy.InputField(
        desc="K candidate records to select from, each rendered with its 'id' field"
    )
    selected_ids: list[str] = dspy.OutputField(
        desc="The id of the single matching candidate, as a one-element list, "
        "or an empty list if none of the candidates match the anchor"
    )
    reasoning: str = dspy.OutputField(desc="Brief justification for the selection")


def _signature_for(entity_noun: str) -> Any:
    """Return :class:`SelectSignature` with ``entity_noun`` woven into its instructions."""
    instructions = SelectSignature.instructions.replace("{entity_noun}", entity_noun)
    return SelectSignature.with_instructions(instructions)


@register("select_judge")
class SelectJudge(GroupwiseModule[SchemaT]):
    """ComEM-style set-wise judge: one LLM call per group, not per pair.

    Example:
        # Zero-spend: inject a DummyLM, score groups from a blocker's stream_groups().
        from dspy.utils.dummies import DummyLM

        judge = SelectJudge(lm=DummyLM([...]), entity_noun="product")
        for j in judge.forward_groups(blocker.stream_groups(records)):
            print(j.score, j.reasoning, j.provenance["cost_usd"])
    """

    # Registry key, mirrored as a class attribute so the Resolver's uniform
    # serialization helper can discover the type name (see resolver.py).
    type_name: ClassVar[str] = "select_judge"

    def __init__(
        self,
        lm: Any = None,
        model: str = "openrouter/openai/gpt-4o-mini",
        temperature: float = 0.0,
        entity_noun: str = "entity",
    ) -> None:
        """Initialize a SelectJudge.

        Args:
            lm: Optional pre-built DSPy LM (``dspy.LM`` or ``DummyLM``). When
                ``None`` a ``dspy.LM(model, cache=False)`` is built lazily on
                first use, mirroring ``DSPyJudge``.
            model: OpenRouter/LiteLLM model id used when ``lm`` is built lazily.
            temperature: Sampling temperature for the lazily-built LM.
            entity_noun: Domain noun woven into the signature instructions.
        """
        self.model = model
        self.temperature = temperature
        self.entity_noun = entity_noun
        self._lm = lm
        self._program: Any = dspy.ChainOfThought(_signature_for(entity_noun))
        # Honest per-call cost = tokens * price, same seam as DSPyJudge: $0 by
        # default (zero-spend tests), pinned by the ``select_judge`` builder in
        # ``langres.methods`` from the OpenRouter price table for real runs.
        self.price_per_1k_tokens = 0.0

    @property
    def config(self) -> dict[str, object]:
        """Pure, serializable construction config (never the LM or secrets)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "entity_noun": self.entity_noun,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "SelectJudge[SchemaT]":
        """Rebuild a fresh judge from :attr:`config` (no injected LM; built lazily)."""
        return cls(
            lm=None,
            model=str(config["model"]),
            temperature=float(config["temperature"]),  # type: ignore[arg-type]
            entity_noun=str(config["entity_noun"]),
        )

    def _get_lm(self) -> Any:
        """Return the DSPy LM, lazily building a ``dspy.LM`` from ``model`` on first use."""
        if self._lm is None:
            self._lm = dspy.LM(self.model, cache=False, temperature=self.temperature)
        return self._lm

    def _cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Honest per-call cost from token counts and the pinned per-1k price."""
        return (prompt_tokens + completion_tokens) / 1000.0 * self.price_per_1k_tokens

    def _render_entity(self, entity: SchemaT) -> str:
        """Render a single entity for the prompt (LLMJudge/DSPyJudge's JSON convention)."""
        return entity.model_dump_json(indent=2)

    def _render_members(self, members: list[SchemaT]) -> str:
        """Render K candidate members as a JSON array, each carrying its ``id`` field."""
        return json.dumps([member.model_dump(mode="json") for member in members], indent=2)

    def _select_error_judgements(
        self,
        group: ERCandidateGroup[SchemaT],
        error: str,
        call_cost_usd: float,
    ) -> list[PairwiseJudgement]:
        """Map a whole group to 'no match' judgements carrying ``select_error`` (CEO #12).

        Never raises mid-stream: a malformed/out-of-group/over-selected LLM
        answer degrades this ONE group's judgements to score 0.0 rather than
        killing the run. The group-call cost convention still applies -- the
        call was billed regardless of whether its answer was usable.
        """
        judgements = [
            PairwiseJudgement(
                left_id=group.anchor.id,  # type: ignore[attr-defined]
                right_id=member.id,  # type: ignore[attr-defined]
                score=0.0,
                score_type="prob_group_llm",
                decision_step="select_judge_error",
                reasoning=None,
                provenance={"model": self.model, "select_error": error},
            )
            for member in group.members
        ]
        return stamp_group_cost(judgements, call_cost_usd=call_cost_usd, group_id=group.group_id)

    def forward_groups(
        self, groups: Iterator[ERCandidateGroup[SchemaT]]
    ) -> Iterator[PairwiseJudgement]:
        """Score each group with ONE LLM call, decomposed into per-member judgements.

        A group with no members is skipped without an LLM call (there is no
        pair to judge). Otherwise the program is asked to select at most one
        matching candidate id; a malformed response, an id outside the group,
        or more than one selected id all route to
        :meth:`_select_error_judgements` instead of raising.
        """
        lm = self._get_lm()
        for group in groups:
            if not group.members:
                continue

            member_ids = {member.id for member in group.members}  # type: ignore[attr-defined]
            anchor_text = self._render_entity(group.anchor)
            members_text = self._render_members(group.members)

            try:
                with dspy.context(lm=lm, track_usage=True):
                    prediction = self._program(anchor=anchor_text, candidates=members_text)
            except AdapterParseError as error:
                logger.warning("SelectJudge parse failure for group %s: %s", group.group_id, error)
                prompt_tokens, completion_tokens = _salvage_usage(lm)
                cost = self._cost_usd(prompt_tokens, completion_tokens)
                yield from self._select_error_judgements(
                    group, f"malformed LLM response: {error}", cost
                )
                continue

            # {model: {prompt_tokens, completion_tokens, *_details, ...}}
            usage = LLMUsage.from_lm_usage(prediction.get_lm_usage(), model=self.model)
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens
            cost = self._cost_usd(prompt_tokens, completion_tokens)

            selected_ids = list(prediction.selected_ids)
            # Unknown-id is checked before the multi-select count: a response with
            # BOTH an out-of-group id and >1 selection is still exactly one error,
            # and "not in group" is the more specific, more actionable diagnosis of
            # the two (it points at which id is wrong, not just how many).
            unknown_ids = [sid for sid in selected_ids if sid not in member_ids]
            if unknown_ids:
                yield from self._select_error_judgements(
                    group,
                    f"selected id(s) not in group: {unknown_ids}",
                    cost,
                )
                continue
            if len(selected_ids) > 1:
                yield from self._select_error_judgements(
                    group,
                    f"selected {len(selected_ids)} candidates; expected at most one",
                    cost,
                )
                continue

            selected = set(selected_ids)
            judgements = [
                PairwiseJudgement(
                    left_id=group.anchor.id,  # type: ignore[attr-defined]
                    right_id=member.id,  # type: ignore[attr-defined]
                    score=1.0 if member.id in selected else 0.0,  # type: ignore[attr-defined]
                    score_type="prob_group_llm",
                    decision_step="select_judgment",
                    reasoning=prediction.reasoning,
                    provenance={
                        "model": self.model,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "usage": usage.model_dump(),
                    },
                )
                for member in group.members
            ]
            yield from stamp_group_cost(judgements, call_cost_usd=cost, group_id=group.group_id)

    def inspect_scores(
        self, judgements: list[PairwiseJudgement], sample_size: int = 10
    ) -> ScoreInspectionReport:
        """Explore scores without ground-truth labels (shared report helper)."""
        return _inspect_scores_impl(judgements, sample_size)

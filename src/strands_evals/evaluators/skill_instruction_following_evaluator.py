from enum import Enum
from typing import Literal, cast

from pydantic import BaseModel, Field
from strands import Agent
from strands.models.model import Model

from ..extractors.skills import InvokedSkill, extract_selected_skills
from ..types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput, InputT, OutputT
from .evaluator import Evaluator
from .prompt_templates.skill_instruction_following import get_template
from .prompt_templates.trajectory_prompt_template import serialize_trajectory


class SkillFollowingScore(str, Enum):
    """Five-point ordinal rating for how fully the agent followed a skill's steps.

    Mirrors the five-point scale used by the coherence, faithfulness, and response
    relevance evaluators (the framework's convention for graded quality judgments).
    """

    FULLY_FOLLOWED = "Fully Followed"
    MOSTLY_FOLLOWED = "Mostly Followed"
    PARTIALLY_FOLLOWED = "Partially Followed"
    MINIMALLY_FOLLOWED = "Minimally Followed"
    NOT_FOLLOWED = "Not Followed"


# The fields the AgentSkills plugin generates after the skill's own text, used to tell its runtime
# block apart from a Markdown rule inside real instructions.
_HARNESS_METADATA_FIELDS = ("Location:", "Allowed tools:", "Compatibility:", "Available resources:")


def _strip_frontmatter(body: str) -> str:
    """Drop a leading YAML frontmatter block (`---\\n ... \\n---`) so the judge sees only steps.

    Some harnesses return the raw SKILL.md (frontmatter included); others return the
    body alone. Stripping is a no-op when there is no frontmatter.
    """
    if not body.startswith("---"):
        return body
    lines = body.splitlines()
    # find the closing '---' after the opening one
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")
    return body


def _strip_harness_metadata(body: str) -> str:
    """Drop the runtime block the harness appends after the skill's own instructions.

    The Strands AgentSkills plugin ends a filesystem-skill result with a `---` rule followed by
    lines it generated rather than the skill author: `Location:`, `Allowed tools:`,
    `Compatibility:`, and an `Available resources:` list. The prompt labels this whole string
    "SKILL.md instructions", so a judge can read `Available resources: scripts/extract.py` as a
    prescribed step nobody wrote, or `Allowed tools:` as a constraint from the skill.

    Keyed on those field names rather than on the `---` alone, because a rule is legal Markdown
    inside real instructions and splitting on it would truncate them.
    """
    marker = "\n---\n"
    index = body.rfind(marker)
    if index == -1:
        return body
    tail = body[index + len(marker) :]
    if not tail.strip():
        return body
    first = tail.lstrip().split("\n", 1)[0]
    if any(first.startswith(field) for field in _HARNESS_METADATA_FIELDS):
        return body[:index].rstrip("\n")
    return body


class SkillFollowingRating(BaseModel):
    """Structured output for skill instruction following evaluation."""

    reasoning: str = Field(description="Brief overall reasoning about adherence to the skill")
    # Still required, but deliberately without `min_length=1`. This model is the
    # structured-output schema, and a skill body with nothing prescriptive in it (reference
    # material, frontmatter only) makes an empty list the correct answer. Rejecting it sends the
    # judge back to re-emit the same answer until the retry loop dies of recursion depth. The
    # empty case is handled in `_rating_to_output` instead.
    steps: list["SkillStepRating"] = Field(
        description="One status and evidence record per prescribed step, in instruction order",
    )
    score: SkillFollowingScore = Field(
        description=(
            "Overall five-point rating of how fully the skill's steps were followed, "
            "consistent with the per-step statuses"
        )
    )

    @property
    def coverage(self) -> float:
        """Derive coverage from structured statuses instead of trusting model arithmetic.

        Zero when there are no steps, which is the vacuous case rather than a failure; callers
        distinguish the two by checking `steps` themselves.
        """
        if not self.steps:
            return 0.0
        weights = {"covered": 1.0, "partial": 0.5, "skipped": 0.0}
        return sum(weights[step.status] for step in self.steps) / len(self.steps)


class SkillStepRating(BaseModel):
    """Judge result for one prescribed skill step."""

    step: str = Field(description="The prescribed step being evaluated")
    status: Literal["covered", "partial", "skipped"]
    evidence: str = Field(description="Concrete trajectory evidence for the status")


class SkillInstructionFollowingEvaluator(Evaluator[InputT, OutputT]):
    """Evaluates whether the agent followed the steps of each skill it invoked.

    Returns one `EvaluationOutput` per invoked skill, scored on a five-point rating. When no
    skill was invoked there is nothing to follow, so a single not-applicable row is returned
    and dropped from the aggregated mean.
    """

    _score_mapping = {
        SkillFollowingScore.FULLY_FOLLOWED: 1.0,
        SkillFollowingScore.MOSTLY_FOLLOWED: 0.75,
        SkillFollowingScore.PARTIALLY_FOLLOWED: 0.5,
        SkillFollowingScore.MINIMALLY_FOLLOWED: 0.25,
        SkillFollowingScore.NOT_FOLLOWED: 0.0,
    }

    def __init__(
        self,
        version: str = "v0",
        model: Model | str | None = None,
        system_prompt: str | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.system_prompt = system_prompt if system_prompt is not None else get_template(version).SYSTEM_PROMPT
        self.version = version
        self.model = model
        # Drop not-applicable rows from the aggregate so no-skill runs don't deflate the mean.
        self.aggregator = self._aggregate_dropping_na

    def _not_applicable_row(self, reason: str, test_pass: bool = True) -> EvaluationOutput:
        return EvaluationOutput(score=0.0, test_pass=test_pass, reason=reason, label=NOT_APPLICABLE)

    def _missing_trajectory_row(self) -> EvaluationOutput:
        """A missing trajectory is absent data, not a run that had nothing to follow."""
        return self._not_applicable_row("no trajectory provided", test_pass=False)

    @staticmethod
    def _unscorable_reason(skill: InvokedSkill) -> str | None:
        """Why this skill cannot be scored for adherence, or None when it can be.

        A refused load is reported separately from a missing body: the agent never received any
        instructions, so there was nothing it could have followed. Both are not-applicable, but
        conflating them hides a broken harness behind what looks like a capture gap.
        """
        if skill.status == "failed":
            # The harness's own message says which refusal it was, and so what to fix: a
            # misspelled skill name in the agent's call, or a harness that mounted none.
            refusal = f" ({skill.error})" if skill.error else ""
            return f"{skill.name}: the harness refused the load{refusal}, so no instructions were received"
        if not skill.body:
            return f"{skill.name}: skill body unavailable"
        return None

    def _build_prompt(self, skill: InvokedSkill, evaluation_case: EvaluationData[InputT, OutputT]) -> str:
        body = _strip_harness_metadata(_strip_frontmatter(skill.body or ""))
        return (
            f"## Skill: {skill.name}\n\n"
            f"## SKILL.md instructions\n{body}\n\n"
            f"## Agent trajectory\n{serialize_trajectory(evaluation_case.actual_trajectory)}\n\n"
            f"## Agent's final response\n{evaluation_case.actual_output}"
        )

    def _rating_to_output(self, skill: InvokedSkill, rating: SkillFollowingRating) -> EvaluationOutput:
        # A skill that prescribes nothing has nothing to follow, so scoring it either way would be
        # arbitrary: it is the same vacuous case as "no skill invoked", not a failure to adhere.
        if not rating.steps:
            return self._not_applicable_row(f"{skill.name}: no prescribed steps found in the skill body")
        # Score off the five-point ordinal rating via `_score_mapping`, following the
        # framework's graded-quality judges. The per-step statuses ground that rating and
        # are preserved in `reason` (a plain field), so the base output schema is untouched.
        # `label` carries the ordinal rating (its enum value), consistent with the other judges.
        normalized_score = self._score_mapping[rating.score]
        step_evidence = "\n".join(f"- {step.step}: {step.status}; evidence: {step.evidence}" for step in rating.steps)
        return EvaluationOutput(
            score=normalized_score,
            # A skill's steps are prescriptive, so the bar is "Mostly Followed" rather than the
            # mid-scale 0.5 the open-ended quality judges use.
            test_pass=normalized_score >= 0.75,
            reason=(f"{skill.name}: {rating.reasoning}\nCoverage: {rating.coverage:.2f}\nSteps:\n{step_evidence}"),
            label=rating.score.value,
        )

    def evaluate(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if evaluation_case.actual_trajectory is None:
            return [self._missing_trajectory_row()]
        invoked = extract_selected_skills(evaluation_case.actual_trajectory)
        if not invoked:
            return [self._not_applicable_row("no skill invoked")]
        results = []
        for skill in invoked:
            if reason := self._unscorable_reason(skill):
                results.append(self._not_applicable_row(reason))
                continue
            prompt = self._build_prompt(skill, evaluation_case)
            evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            result = evaluator_agent(prompt, structured_output_model=SkillFollowingRating)
            rating = cast(SkillFollowingRating, result.structured_output)
            results.append(self._rating_to_output(skill, rating))
        return results

    async def evaluate_async(self, evaluation_case: EvaluationData[InputT, OutputT]) -> list[EvaluationOutput]:
        if evaluation_case.actual_trajectory is None:
            return [self._missing_trajectory_row()]
        invoked = extract_selected_skills(evaluation_case.actual_trajectory)
        if not invoked:
            return [self._not_applicable_row("no skill invoked")]
        results = []
        for skill in invoked:
            if reason := self._unscorable_reason(skill):
                results.append(self._not_applicable_row(reason))
                continue
            prompt = self._build_prompt(skill, evaluation_case)
            evaluator_agent = Agent(model=self.model, system_prompt=self.system_prompt, callback_handler=None)
            result = await evaluator_agent.invoke_async(prompt, structured_output_model=SkillFollowingRating)
            rating = cast(SkillFollowingRating, result.structured_output)
            results.append(self._rating_to_output(skill, rating))
        return results

from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.skill_instruction_following_evaluator import (
    SkillFollowingRating,
    SkillFollowingScore,
    SkillInstructionFollowingEvaluator,
    SkillStepRating,
    _strip_frontmatter,
)
from strands_evals.extractors import extract_selected_skills
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput

_MODULE = "strands_evals.evaluators.skill_instruction_following_evaluator.Agent"

SKILL_BODY = "# PDF Processing Skill\n1. Identify the PDF path.\n2. Extract text.\n3. Summarize."


def _ordinal_for(coverage: float) -> SkillFollowingScore:
    """A reasonable ordinal for a coverage fraction, for building test ratings."""
    if coverage >= 0.95:
        return SkillFollowingScore.FULLY_FOLLOWED
    if coverage >= 0.75:
        return SkillFollowingScore.MOSTLY_FOLLOWED
    if coverage >= 0.5:
        return SkillFollowingScore.PARTIALLY_FOLLOWED
    if coverage >= 0.25:
        return SkillFollowingScore.MINIMALLY_FOLLOWED
    return SkillFollowingScore.NOT_FOLLOWED


def _rating(
    *statuses: str,
    reasoning: str = "step evidence",
    score: SkillFollowingScore | None = None,
) -> SkillFollowingRating:
    steps = [
        SkillStepRating(step=f"step {i}", status=status, evidence=f"evidence {i}")
        for i, status in enumerate(statuses, start=1)
    ]
    coverage = sum({"covered": 1.0, "partial": 0.5, "skipped": 0.0}[s] for s in statuses) / len(statuses)
    return SkillFollowingRating(
        reasoning=reasoning,
        steps=steps,
        score=score if score is not None else _ordinal_for(coverage),
    )


def _case(invoked: list[tuple[str, str]]):
    """invoked: list of (skill_name, body) -> Strands message list with toolUse/toolResult pairs."""
    messages = []
    for i, (name, body) in enumerate(invoked):
        tid = f"t{i}"
        messages.append(
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": tid, "name": "skills", "input": {"skill_name": name}}}],
            }
        )
        messages.append({"role": "user", "content": [{"toolResult": {"toolUseId": tid, "content": [{"text": body}]}}]})
    return EvaluationData(input="do pdf", actual_output="done", actual_trajectory=messages)


def test_strip_frontmatter():
    raw = "---\nname: pdf\ndescription: x\n---\n\n# Body\n1. step"
    assert _strip_frontmatter(raw) == "# Body\n1. step"
    # no frontmatter -> unchanged
    assert _strip_frontmatter("# Body\n1. step") == "# Body\n1. step"


def test_init_defaults():
    ev = SkillInstructionFollowingEvaluator()
    assert ev.version == "v0"
    # aggregator overridden to drop N/A rows
    assert ev.aggregator == ev._aggregate_dropping_na


def test_rating_derives_coverage_from_step_statuses():
    assert _rating("covered", "partial", "skipped").coverage == 0.5


def test_rating_accepts_no_steps():
    """A skill body that prescribes nothing is a valid judgment, not a schema violation.

    `steps` is the structured-output schema, so rejecting an empty list sends the judge back to
    re-emit the same answer until the retry loop exhausts the recursion limit.
    """
    rating = SkillFollowingRating(reasoning="reference material only", steps=[], score=_ordinal_for(0.0))

    assert rating.steps == []
    assert rating.coverage == 0.0  # vacuous, not a failure; callers check `steps` to tell them apart


def test_score_mapping_is_five_point_ordinal():
    ev = SkillInstructionFollowingEvaluator()
    assert ev._score_mapping == {
        SkillFollowingScore.FULLY_FOLLOWED: 1.0,
        SkillFollowingScore.MOSTLY_FOLLOWED: 0.75,
        SkillFollowingScore.PARTIALLY_FOLLOWED: 0.5,
        SkillFollowingScore.MINIMALLY_FOLLOWED: 0.25,
        SkillFollowingScore.NOT_FOLLOWED: 0.0,
    }


@pytest.mark.parametrize("field", ["step", "evidence"])
def test_step_rating_accepts_empty_text(field):
    """No length floor on the judge's own prose.

    `SkillStepRating` is part of the structured-output schema, and a rejected value is not a
    caught error: the judge is sent back to produce the same answer again, so a plausible output
    (empty `evidence` for a step it found no evidence of) becomes an unbounded retry loop.
    """
    values = {"step": "step", "status": "covered", "evidence": "evidence"}
    values[field] = ""

    assert getattr(SkillStepRating(**values), field) == ""


def test_prompt_includes_trajectory_evidence():
    case = _case([("pdf-processing", SKILL_BODY)])
    case.actual_trajectory.append({"role": "assistant", "content": [{"text": "Extracted report.pdf"}]})

    prompt = SkillInstructionFollowingEvaluator()._build_prompt(
        extract_selected_skills(case.actual_trajectory)[0],
        case,
    )

    assert "Agent trajectory" in prompt
    assert "Extracted report.pdf" in prompt
    assert "Agent's final response" in prompt


@patch(_MODULE)
def test_evaluate_one_per_invoked_skill(mock_agent_class):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = _rating("covered", reasoning="all steps covered")
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    result = SkillInstructionFollowingEvaluator().evaluate(
        _case([("pdf-processing", SKILL_BODY), ("spreadsheet", "# S\n1. inspect")])
    )
    # One EvaluationOutput per invoked skill; both fully followed here.
    assert len(result) == 2
    assert all(r.label == "Fully Followed" for r in result)
    assert all(r.score == 1.0 and r.test_pass for r in result)


@pytest.mark.parametrize(
    "ordinal,score,expected_pass",
    [
        (SkillFollowingScore.FULLY_FOLLOWED, 1.0, True),
        (SkillFollowingScore.MOSTLY_FOLLOWED, 0.75, True),
        (SkillFollowingScore.PARTIALLY_FOLLOWED, 0.5, False),
        (SkillFollowingScore.MINIMALLY_FOLLOWED, 0.25, False),
        (SkillFollowingScore.NOT_FOLLOWED, 0.0, False),
    ],
)
@patch(_MODULE)
def test_ordinal_score_and_threshold(mock_agent_class, ordinal, score, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    # A single per-step status is enough; the shipped score comes from the ordinal rating.
    mock_result.structured_output = _rating("covered", score=ordinal)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    result = SkillInstructionFollowingEvaluator().evaluate(_case([("pdf-processing", SKILL_BODY)]))
    # Score is the five-point ordinal mapped to [0, 1]; test_pass requires Mostly Followed.
    assert result[0].score == score
    assert result[0].test_pass is expected_pass
    assert result[0].label == ordinal.value
    # Per-step statuses/evidence and the derived coverage are preserved in the reason string,
    # not as an EvaluationOutput subclass field, so the base output schema stays unchanged.
    assert "Steps:" in result[0].reason
    assert "Coverage:" in result[0].reason
    assert "evidence 1" in result[0].reason


def test_label_carries_ordinal_not_skill_name():
    """The shipped label is the ordinal rating value (like the other judges), not the skill name."""
    with patch(_MODULE) as mock_agent_class:
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.structured_output = _rating("covered", score=SkillFollowingScore.MOSTLY_FOLLOWED)
        mock_agent.return_value = mock_result
        mock_agent_class.return_value = mock_agent
        result = SkillInstructionFollowingEvaluator().evaluate(_case([("pdf-processing", SKILL_BODY)]))
    assert result[0].label == "Mostly Followed"


def test_no_skill_returns_not_applicable_row():
    ev = SkillInstructionFollowingEvaluator()
    case = EvaluationData(input="x", actual_output="y", actual_trajectory=[])
    result = ev.evaluate(case)
    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].test_pass is True  # no violation


@patch(_MODULE)
def test_skill_prescribing_nothing_is_not_applicable(mock_agent_class):
    """A skill body with no prescribed steps has nothing to follow, so it is not scored.

    Scoring it either way would be arbitrary: 0.0 reads as a failure to adhere, 1.0 as vacuous
    adherence, and both distort the mean.
    """
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = SkillFollowingRating(
        reasoning="the body is reference material, not instructions",
        steps=[],
        score=SkillFollowingScore.NOT_FOLLOWED,
    )
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    result = SkillInstructionFollowingEvaluator().evaluate(_case([("pdf-processing", "# Reference\nField notes.")]))

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].reason == "pdf-processing: no prescribed steps found in the skill body"
    assert result[0].test_pass is True


def test_missing_trajectory_does_not_pass():
    """Absent data is not a run that had nothing to follow, so it must not report a pass."""
    ev = SkillInstructionFollowingEvaluator()
    case = EvaluationData(input="x", actual_output="y", actual_trajectory=None)

    result = ev.evaluate(case)

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].test_pass is False
    assert result[0].reason == "no trajectory provided"
    # The aggregator must carry the failure rather than defaulting an all-N/A row to pass.
    assert ev.aggregator(result)[1] is False


@pytest.mark.asyncio
async def test_missing_trajectory_does_not_pass_async():
    ev = SkillInstructionFollowingEvaluator()
    case = EvaluationData(input="x", actual_output="y", actual_trajectory=None)

    result = await ev.evaluate_async(case)

    assert len(result) == 1
    assert result[0].test_pass is False
    assert result[0].reason == "no trajectory provided"


@patch(_MODULE)
def test_invoked_skill_without_body_is_not_mislabeled_as_no_invocation(mock_agent_class):
    messages = [
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"toolUseId": "t", "name": "load_skill", "input": {"skill_name": "pdf-processing"}}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t",
                        "content": [{"text": '{"status":"loaded","path":".agents/pdf-processing"}'}],
                    }
                }
            ],
        },
    ]
    case = EvaluationData(input="x", actual_output="y", actual_trajectory=messages)

    result = SkillInstructionFollowingEvaluator().evaluate(case)

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].reason == "pdf-processing: skill body unavailable"
    mock_agent_class.assert_not_called()


@patch(_MODULE)
def test_refused_load_reports_why_nothing_could_be_followed(mock_agent_class):
    """The harness refused the load, so the agent never received any instructions.

    Reported separately from a missing body: both are not-applicable, but conflating them hides
    a broken harness behind what reads as a trajectory-capture gap.
    """
    messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "t", "name": "skills", "input": {"skill_name": "pdf-processing"}}}],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "t", "status": "error", "content": [{"text": "skill not found"}]}}
            ],
        },
    ]
    case = EvaluationData(input="do pdf", actual_output="done", actual_trajectory=messages)

    result = SkillInstructionFollowingEvaluator().evaluate(case)

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].reason == (
        "pdf-processing: the harness refused the load (skill not found), so no instructions were received"
    )
    mock_agent_class.assert_not_called()


@patch(_MODULE)
def test_duplicate_loads_trigger_one_judge_call(mock_agent_class):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = _rating("covered", reasoning="covered")
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    case = _case([("pdf-processing", SKILL_BODY), ("pdf-processing", SKILL_BODY)])

    result = SkillInstructionFollowingEvaluator().evaluate(case)

    assert len(result) == 1
    mock_agent.assert_called_once()


@patch(_MODULE)
def test_each_row_names_the_skill_it_scored(mock_agent_class):
    """With several skills invoked, `label` holds the rating, so the reason must attribute the row."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = _rating("covered", reasoning="covered")
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    case = _case([("pdf-processing", SKILL_BODY), ("redaction", SKILL_BODY)])

    result = SkillInstructionFollowingEvaluator().evaluate(case)

    assert len(result) == 2
    assert result[0].reason.startswith("pdf-processing: ")
    assert result[1].reason.startswith("redaction: ")


def test_aggregator_drops_not_applicable():
    ev = SkillInstructionFollowingEvaluator()
    rows = [
        EvaluationOutput(score=1.0, test_pass=True, reason="covered", label="pdf-processing"),
        EvaluationOutput(score=0.0, test_pass=True, reason="no skill invoked", label=NOT_APPLICABLE),
    ]
    avg, all_pass, _ = ev.aggregator(rows)
    # the N/A row must not deflate the mean
    assert avg == 1.0
    assert all_pass is True

    # all-N/A aggregates to a clean pass, not 0-deflated failure
    only_na = [EvaluationOutput(score=0.0, test_pass=True, reason="no skill invoked", label=NOT_APPLICABLE)]
    assert ev.aggregator(only_na) == (0.0, True, "no skill invoked")

    missing_body = [
        EvaluationOutput(
            score=0.0,
            test_pass=True,
            reason="pdf-processing: skill body unavailable",
            label=NOT_APPLICABLE,
        )
    ]
    assert ev.aggregator(missing_body) == (0.0, True, "pdf-processing: skill body unavailable")


@pytest.mark.asyncio
@patch(_MODULE)
async def test_evaluate_async(mock_agent_class):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        # coverage 0.95 -> the judge rates it MOSTLY_FOLLOWED (score 0.75)
        mock_result.structured_output = _rating(
            "covered",
            "covered",
            "covered",
            "covered",
            "covered",
            "covered",
            "covered",
            "covered",
            "covered",
            "partial",
            score=SkillFollowingScore.MOSTLY_FOLLOWED,
        )
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent

    result = await SkillInstructionFollowingEvaluator().evaluate_async(_case([("pdf-processing", SKILL_BODY)]))
    assert len(result) == 1
    # Shipped score is the ordinal mapping (0.75), not the raw coverage (0.95).
    assert result[0].score == 0.75
    assert result[0].test_pass is True

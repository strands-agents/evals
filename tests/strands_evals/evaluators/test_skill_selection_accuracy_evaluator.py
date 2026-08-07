from unittest.mock import Mock, patch

import pytest

from strands_evals.evaluators.skill_selection_accuracy_evaluator import (
    SkillSelectionAccuracyEvaluator,
    SkillSelectionRating,
    SkillSelectionScore,
)
from strands_evals.extractors import InvokedSkill
from strands_evals.types.evaluation import NOT_APPLICABLE, EvaluationData, EvaluationOutput

_MODULE = "strands_evals.evaluators.skill_selection_accuracy_evaluator.Agent"

AVAILABLE_BLOCK = """<available_skills>
<skill><name>pdf-processing</name><description>Extract text from PDFs.</description></skill>
<skill><name>spreadsheet</name><description>Analyze spreadsheets.</description></skill>
</available_skills>"""


def _case(invoked: list[str] | str | None):
    names = [invoked] if isinstance(invoked, str) else (invoked or [])
    messages = [{"role": "system", "content": AVAILABLE_BLOCK}]
    for i, name in enumerate(names):
        tid = f"t{i}"
        messages.append(
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": tid, "name": "skills", "input": {"skill_name": name}}}],
            }
        )
        messages.append(
            {"role": "user", "content": [{"toolResult": {"toolUseId": tid, "content": [{"text": "# Skill body"}]}}]}
        )
    return EvaluationData(input="Extract text from report.pdf", actual_output="done", actual_trajectory=messages)


def test_init_defaults():
    ev = SkillSelectionAccuracyEvaluator()
    assert ev.version == "v0"
    assert ev.model is None
    assert ev.system_prompt is not None
    # unlike the tool evaluator, it does NOT slice via TraceExtractor
    assert ev.evaluation_level is None


def test_aggregator_drops_not_applicable_rows():
    """A case with one judged skill and one unjudgeable row is not half right.

    The per-case aggregate has to drop the same rows `calculate_overall_score` drops, or the
    case score reported in the table disagrees with the overall score computed from it.
    """
    ev = SkillSelectionAccuracyEvaluator()
    assert ev.aggregator == ev._aggregate_dropping_na

    rows = [
        EvaluationOutput(score=1.0, test_pass=True, reason="pdf-processing: fits", label="Yes"),
        EvaluationOutput(score=0.0, test_pass=True, reason="no skills were available", label=NOT_APPLICABLE),
    ]
    avg, all_pass, _ = ev.aggregator(rows)
    assert avg == 1.0
    assert all_pass is True


def test_prompt_focuses_on_one_invoked_skill():
    ev = SkillSelectionAccuracyEvaluator()
    prompt = ev._build_prompt(_case("pdf-processing"), focus_skill=InvokedSkill("pdf-processing", "# body"))
    assert "pdf-processing: Extract text from PDFs." in prompt  # available list
    assert "invoked the skill: pdf-processing" in prompt  # focal decision
    assert "Agent trajectory" in prompt
    assert "report.pdf" in prompt


def test_prompt_tells_the_judge_a_refused_load_is_not_a_wrong_choice():
    """Selection is about the choice, not the outcome.

    Left unsaid, the judge sees the harness error in the trajectory and scores a correct
    selection as wrong for failing to load.
    """
    ev = SkillSelectionAccuracyEvaluator()

    prompt = ev._build_prompt(
        _case("pdf-processing"),
        focus_skill=InvokedSkill("pdf-processing", None, status="failed"),
    )

    assert "invoked the skill: pdf-processing" in prompt
    assert "harness refused the load" in prompt
    assert "not whether it worked" in prompt


def test_prompt_carries_the_harness_refusal_message():
    """Which refusal it was still bears on the choice.

    A skill name the harness did not recognize is a worse pick than a right one the harness
    could not mount, and only the refusal text says which happened.
    """
    ev = SkillSelectionAccuracyEvaluator()

    prompt = ev._build_prompt(
        _case("pdf-processing"),
        focus_skill=InvokedSkill(
            "pdf-procesing", None, status="failed", error="Skill 'pdf-procesing' not found. Available: pdf-processing"
        ),
    )

    assert "The harness said: Skill 'pdf-procesing' not found. Available: pdf-processing" in prompt


def test_prompt_has_no_abstention_branch():
    """Selection judges invoked skills only, so no prompt path offers an abstention verdict."""
    ev = SkillSelectionAccuracyEvaluator()
    prompt = ev._build_prompt(_case("pdf-processing"), focus_skill=InvokedSkill("pdf-processing", "# body"))
    assert "abstained" not in prompt
    assert "abstention" not in ev.system_prompt.casefold()


@pytest.mark.parametrize(
    "score,expected_value,expected_pass",
    [
        (SkillSelectionScore.YES, 1.0, True),
        (SkillSelectionScore.NO, 0.0, False),
    ],
)
@patch(_MODULE)
def test_evaluate_score_mapping_labels_by_rating(mock_agent_class, score, expected_value, expected_pass):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = SkillSelectionRating(reasoning="because", score=score)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    result = SkillSelectionAccuracyEvaluator().evaluate(_case("pdf-processing"))

    # One output for the single invoked skill. `label` is the judge's rating, as in every other
    # judge in the framework; which decision the row is about is named in `reason`.
    assert len(result) == 1
    assert result[0].score == expected_value
    assert result[0].test_pass is expected_pass
    assert result[0].label == score.value
    assert result[0].reason == "pdf-processing: because"


@patch(_MODULE)
def test_evaluate_loops_per_invoked_skill(mock_agent_class):
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = SkillSelectionRating(reasoning="fits", score=SkillSelectionScore.YES)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent

    result = SkillSelectionAccuracyEvaluator().evaluate(_case(["pdf-processing", "spreadsheet"]))

    # one EvaluationOutput per invoked skill, each naming its skill in `reason`
    assert len(result) == 2
    assert {r.reason.split(":")[0] for r in result} == {"pdf-processing", "spreadsheet"}
    assert mock_agent.call_count == 2
    # A fresh judge per skill: a reused Agent would carry the first verdict into the second
    # prompt as conversation history and resend the trajectory on top of it.
    assert mock_agent_class.call_count == 2


@patch(_MODULE)
def test_missing_trajectory_is_not_scored(mock_agent_class):
    """A None trajectory is absent data, so it fails rather than passing as not-applicable."""
    case = EvaluationData(input="do pdf", actual_output="done", actual_trajectory=None)

    result = SkillSelectionAccuracyEvaluator().evaluate(case)

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].score == 0.0
    assert result[0].test_pass is False
    mock_agent_class.assert_not_called()  # no judge call on missing data


@patch(_MODULE)
def test_no_invocation_without_a_catalog_is_not_scored(mock_agent_class):
    """With nothing on offer and nothing invoked there was no selection decision to judge.

    A "Yes" would credit the agent for declining an offer it never received, and a "No" would
    penalize it for the same. Not every trajectory carries a catalog, so this is common.
    """
    case = EvaluationData(
        input="Extract text from report.pdf",
        actual_output="done",
        actual_trajectory=[{"role": "user", "content": [{"text": "Extract text from report.pdf"}]}],
    )

    result = SkillSelectionAccuracyEvaluator().evaluate(case)

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].reason == "no skills were available to select from"
    assert result[0].test_pass is True
    mock_agent_class.assert_not_called()  # nothing to judge, so no judge call


@pytest.mark.asyncio
@patch(_MODULE)
async def test_no_invocation_without_a_catalog_is_not_scored_async(mock_agent_class):
    case = EvaluationData(input="do pdf", actual_output="done", actual_trajectory=[])

    result = await SkillSelectionAccuracyEvaluator().evaluate_async(case)

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    mock_agent_class.assert_not_called()


@patch(_MODULE)
def test_invocation_with_no_catalog_is_still_judged(mock_agent_class):
    """A skill that was actually invoked is a real decision, catalog or not."""
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = SkillSelectionRating(reasoning="fits", score=SkillSelectionScore.YES)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    case = _case("pdf-processing")
    case.actual_trajectory = case.actual_trajectory[1:]  # drop the <available_skills> system message

    result = SkillSelectionAccuracyEvaluator().evaluate(case)

    assert [r.reason for r in result] == ["pdf-processing: fits"]
    assert mock_agent.call_count == 1


@patch(_MODULE)
def test_refused_load_is_judged_as_a_selection_not_an_abstention(mock_agent_class):
    """A skill the harness refused is still a selection the agent made.

    Dropping the row would leave the run looking like an abstention, so an agent that picked the
    right skill and was refused would be scored on a decision it never took.
    """
    mock_agent = Mock()
    mock_result = Mock()
    mock_result.structured_output = SkillSelectionRating(reasoning="right skill", score=SkillSelectionScore.YES)
    mock_agent.return_value = mock_result
    mock_agent_class.return_value = mock_agent
    case = _case("pdf-processing")
    case.actual_trajectory[-1]["content"][0]["toolResult"]["status"] = "error"

    result = SkillSelectionAccuracyEvaluator().evaluate(case)

    assert [r.reason for r in result] == ["pdf-processing: right skill"]
    assert result[0].label == "Yes"
    assert "harness refused the load" in mock_agent.call_args.args[0]


@patch(_MODULE)
def test_no_invocation_with_a_catalog_is_not_scored(mock_agent_class):
    """Skills were on offer and none was taken: still not this evaluator's decision to judge.

    Whether declining was correct depends on the whole offered set, so it is a session-level
    question. The reason distinguishes this from the nothing-on-offer case.
    """
    result = SkillSelectionAccuracyEvaluator().evaluate(_case(None))

    assert len(result) == 1
    assert result[0].label == "not_applicable"
    assert result[0].score == 0.0
    assert result[0].test_pass is True
    assert result[0].reason == "no skill invoked; whether declining was correct is not judged here"
    mock_agent_class.assert_not_called()


@pytest.mark.asyncio
@patch(_MODULE)
async def test_evaluate_async_loops_per_skill(mock_agent_class):
    mock_agent = Mock()

    async def mock_invoke_async(*args, **kwargs):
        mock_result = Mock()
        mock_result.structured_output = SkillSelectionRating(reasoning="ok", score=SkillSelectionScore.YES)
        return mock_result

    mock_agent.invoke_async = mock_invoke_async
    mock_agent_class.return_value = mock_agent

    result = await SkillSelectionAccuracyEvaluator().evaluate_async(_case(["pdf-processing", "spreadsheet"]))
    assert len(result) == 2
    assert {r.reason.split(":")[0] for r in result} == {"pdf-processing", "spreadsheet"}
    assert all(r.score == 1.0 for r in result)
    assert mock_agent_class.call_count == 2  # a fresh judge per skill, same as the sync path


@patch(_MODULE)
def test_each_judge_sees_exactly_one_prompt(mock_agent_class):
    """Contract: no judge is asked twice, so no verdict leaks into the next skill's prompt."""
    agents = []

    def new_agent(*_args, **_kwargs):
        agent = Mock()
        result = Mock()
        result.structured_output = SkillSelectionRating(reasoning="fits", score=SkillSelectionScore.YES)
        agent.return_value = result
        agents.append(agent)
        return agent

    mock_agent_class.side_effect = new_agent

    SkillSelectionAccuracyEvaluator().evaluate(_case(["pdf-processing", "spreadsheet"]))

    assert len(agents) == 2
    assert [a.call_count for a in agents] == [1, 1]
    prompts = [a.call_args.args[0] for a in agents]
    assert "invoked the skill: pdf-processing" in prompts[0]
    assert "invoked the skill: spreadsheet" in prompts[1]

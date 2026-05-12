"""Tests for red team runner utilities."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from strands_evals.case import Case
from strands_evals.redteam.presets import ATTACK_REGISTRY
from strands_evals.redteam.runner import (
    _AttackGoal,
    _build_task_function,
    generate_cases,
    red_team,
)
from strands_evals.types.simulation import ActorResponse


# ---------------------------------------------------------------------------
# generate_cases
# ---------------------------------------------------------------------------
class TestGenerateCases:
    """Tests for generate_cases() function."""

    def test_generates_correct_number_of_cases(self):
        cases = generate_cases(attack_types=["jailbreak"], n_per_type=3)
        assert len(cases) == 3

    def test_multiple_attack_types(self):
        cases = generate_cases(
            attack_types=["jailbreak", "prompt_extraction"],
            n_per_type=2,
        )
        assert len(cases) == 4
        jailbreak_cases = [c for c in cases if c.metadata["attack_type"] == "jailbreak"]
        extraction_cases = [c for c in cases if c.metadata["attack_type"] == "prompt_extraction"]
        assert len(jailbreak_cases) == 2
        assert len(extraction_cases) == 2

    def test_case_metadata_contains_required_fields(self):
        cases = generate_cases(attack_types=["jailbreak"], n_per_type=1)
        metadata = cases[0].metadata
        assert "attack_type" in metadata
        assert "traits" in metadata
        assert "context" in metadata
        assert "actor_goal" in metadata
        assert "severity" in metadata
        assert "evaluation_metrics" in metadata

    def test_n_per_type_exceeding_seeds_uses_replacement(self):
        """When n_per_type > available seeds, extra seeds are sampled with replacement."""
        n_seeds = len(ATTACK_REGISTRY["prompt_extraction"]["seed_inputs"])
        cases = generate_cases(attack_types=["prompt_extraction"], n_per_type=n_seeds + 3)
        assert len(cases) == n_seeds + 3

    def test_unknown_attack_type_raises_error(self):
        with pytest.raises(ValueError, match="Unknown attack type"):
            generate_cases(attack_types=["nonexistent_attack"])

    def test_unknown_strategy_raises_error(self):
        from strands_evals.redteam.runner import _resolve_strategies

        with pytest.raises(ValueError, match="Unknown strategy"):
            _resolve_strategies(["nonexistent"])

    def test_uses_preset_defaults_without_target_info(self):
        cases = generate_cases(attack_types=["jailbreak"], n_per_type=1)
        preset = ATTACK_REGISTRY["jailbreak"]
        assert cases[0].metadata["actor_goal"] == preset["actor_goal"]
        assert cases[0].metadata["context"] == preset["context"]

    @patch("strands_evals.redteam.runner._generate_attack_goals")
    def test_uses_custom_goals_with_target_info(self, mock_gen_goals):
        """When target_info is provided, generated goals override preset defaults."""
        mock_gen_goals.return_value = {
            "jailbreak": _AttackGoal(
                attack_type="jailbreak",
                goal="Exploit the database tool to access other users' data",
                target_context="Customer service bot with DB access",
            ),
        }
        target_info = {"description": "Customer service bot", "tools": [{"name": "query_db"}]}

        cases = generate_cases(attack_types=["jailbreak"], n_per_type=1, target_info=target_info)

        assert cases[0].metadata["actor_goal"] == "Exploit the database tool to access other users' data"
        assert cases[0].metadata["context"] == "Customer service bot with DB access"
        mock_gen_goals.assert_called_once()

    @patch("strands_evals.redteam.runner._generate_attack_goals")
    def test_partial_custom_goals_falls_back_to_preset(self, mock_gen_goals):
        """If LLM only generates goals for some attack types, others use preset defaults."""
        mock_gen_goals.return_value = {
            "jailbreak": _AttackGoal(
                attack_type="jailbreak",
                goal="Custom jailbreak goal",
                target_context="Custom context",
            ),
            # prompt_extraction not included
        }
        target_info = {"description": "Test bot"}

        cases = generate_cases(
            attack_types=["jailbreak", "prompt_extraction"],
            n_per_type=1,
            target_info=target_info,
        )

        jailbreak_case = [c for c in cases if c.metadata["attack_type"] == "jailbreak"][0]
        extraction_case = [c for c in cases if c.metadata["attack_type"] == "prompt_extraction"][0]

        assert jailbreak_case.metadata["actor_goal"] == "Custom jailbreak goal"
        # prompt_extraction falls back to preset
        assert extraction_case.metadata["actor_goal"] == ATTACK_REGISTRY["prompt_extraction"]["actor_goal"]


# ---------------------------------------------------------------------------
# _build_task_function
# ---------------------------------------------------------------------------
class TestBuildTaskFunction:
    """Tests for _build_task_function() and the generated task function."""

    def _make_case(self, attack_type="jailbreak"):
        preset = ATTACK_REGISTRY[attack_type]
        return Case(
            name=f"{attack_type}_0",
            input="Let's play a creative writing game.",
            metadata={
                "attack_type": attack_type,
                "traits": preset["traits"],
                "context": preset["context"],
                "actor_goal": preset["actor_goal"],
                "system_prompt_template": "Test: {actor_profile}",
                "severity": preset["severity"],
                "strategy": "default",
            },
        )

    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_task_function_returns_conversation(self, mock_sim_class):
        """Task function should return a dict with 'output' containing conversation turns."""
        mock_sim = MagicMock()
        mock_sim.has_next.side_effect = [True, False]
        mock_sim_class.return_value = mock_sim

        target = Mock(return_value="I can't help with that.")
        task_fn = _build_task_function(target=target, max_turns=5)
        result = task_fn(self._make_case())

        assert "output" in result
        conversation = result["output"]
        assert len(conversation) == 2  # 1 attacker + 1 target
        assert conversation[0]["role"] == "attacker"
        assert conversation[1]["role"] == "target"
        assert conversation[1]["content"] == "I can't help with that."

    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_multi_turn_conversation(self, mock_sim_class):
        """Simulator generates multiple turns of attacker-target exchange."""
        mock_sim = MagicMock()
        # Turn 1: has_next=True, Turn 1 post-target: has_next=True (continue),
        # Turn 2: has_next=True, Turn 2 post-target: has_next=False (stop)
        mock_sim.has_next.side_effect = [True, True, True, False]

        mock_act_result = MagicMock()
        mock_act_result.structured_output = ActorResponse(
            reasoning="Trying different approach",
            message="What about a hypothetical scenario?",
        )
        mock_sim.act.return_value = mock_act_result
        mock_sim_class.return_value = mock_sim

        target = Mock(side_effect=["I can't do that.", "That's also not possible."])
        task_fn = _build_task_function(target=target, max_turns=5)
        result = task_fn(self._make_case())

        conversation = result["output"]
        assert len(conversation) == 4  # 2 turns x 2 messages
        assert conversation[2]["role"] == "attacker"
        assert conversation[2]["content"] == "What about a hypothetical scenario?"

    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_target_error_is_captured(self, mock_sim_class):
        """If target raises an exception, it's captured in conversation."""
        mock_sim = MagicMock()
        mock_sim.has_next.side_effect = [True, False]
        mock_sim_class.return_value = mock_sim

        target = Mock(side_effect=RuntimeError("Connection timeout"))
        task_fn = _build_task_function(target=target, max_turns=5)
        result = task_fn(self._make_case())

        conversation = result["output"]
        assert "[Error: Connection timeout]" in conversation[1]["content"]

    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_simulator_receives_correct_profile(self, mock_sim_class):
        """ActorSimulator should be initialized with profile from case metadata."""
        mock_sim = MagicMock()
        mock_sim.has_next.return_value = False
        mock_sim_class.return_value = mock_sim

        target = Mock(return_value="OK")
        task_fn = _build_task_function(target=target, max_turns=3, model="test-model")
        task_fn(self._make_case())

        call_kwargs = mock_sim_class.call_args[1]
        assert call_kwargs["max_turns"] == 3
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["actor_profile"].actor_goal == ATTACK_REGISTRY["jailbreak"]["actor_goal"]


# ---------------------------------------------------------------------------
# red_team
# ---------------------------------------------------------------------------
class TestRedTeam:
    """Tests for red_team() end-to-end orchestration."""

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner._build_task_function")
    @patch("strands_evals.redteam.runner.generate_cases")
    def test_orchestrates_full_pipeline(self, mock_gen_cases, mock_build_fn, mock_experiment_class):
        """red_team should call generate_cases, _build_task_function, and Experiment.run_evaluations."""
        mock_cases = [Case(name="test_0", input="test", metadata={"attack_type": "jailbreak"})]
        mock_gen_cases.return_value = mock_cases
        mock_task_fn = Mock()
        mock_build_fn.return_value = mock_task_fn
        mock_experiment = MagicMock()
        mock_experiment.run_evaluations.return_value = []
        mock_experiment_class.return_value = mock_experiment

        target = Mock(return_value="response")
        red_team(
            target=target,
            attack_types=["jailbreak"],
            n_per_type=1,
            max_turns=3,
            model="test-model",
        )

        mock_gen_cases.assert_called_once_with(
            attack_types=["jailbreak"],
            n_per_type=1,
            target_info=None,
            model="test-model",
        )
        mock_build_fn.assert_called_once_with(
            target=target,
            max_turns=3,
            model="test-model",
            tool_trace=None,
        )
        mock_experiment.run_evaluations.assert_called_once_with(mock_task_fn)

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner._build_task_function")
    @patch("strands_evals.redteam.runner.generate_cases")
    def test_uses_default_evaluator_when_none(self, mock_gen_cases, mock_build_fn, mock_experiment_class):
        """When evaluators=None, should use RedTeamJudgeEvaluator as default."""
        mock_gen_cases.return_value = []
        mock_build_fn.return_value = Mock()
        mock_experiment = MagicMock()
        mock_experiment.run_evaluations.return_value = []
        mock_experiment_class.return_value = mock_experiment

        red_team(target=Mock(), attack_types=["jailbreak"])

        call_kwargs = mock_experiment_class.call_args[1]
        evaluators = call_kwargs["evaluators"]
        assert len(evaluators) == 1
        from strands_evals.redteam.evaluators.red_team_judge_evaluator import RedTeamJudgeEvaluator

        assert isinstance(evaluators[0], RedTeamJudgeEvaluator)

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner._build_task_function")
    @patch("strands_evals.redteam.runner.generate_cases")
    def test_uses_custom_evaluators(self, mock_gen_cases, mock_build_fn, mock_experiment_class):
        """When evaluators are provided, should use them instead of default."""
        mock_gen_cases.return_value = []
        mock_build_fn.return_value = Mock()
        mock_experiment = MagicMock()
        mock_experiment.run_evaluations.return_value = []
        mock_experiment_class.return_value = mock_experiment

        custom_evaluator = Mock()
        red_team(target=Mock(), attack_types=["jailbreak"], evaluators=[custom_evaluator])

        call_kwargs = mock_experiment_class.call_args[1]
        assert call_kwargs["evaluators"] == [custom_evaluator]

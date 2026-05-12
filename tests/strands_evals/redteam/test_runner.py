"""Tests for red team runner utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from strands_evals.redteam.presets import ATTACK_REGISTRY
from strands_evals.redteam.runner import generate_cases, red_team
from strands_evals.types.simulation import ActorResponse


def _goal(goal: str, target_context: str) -> SimpleNamespace:
    """Build a duck-typed stand-in for the runner's internal _AttackGoal."""
    return SimpleNamespace(goal=goal, target_context=target_context)


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
        n_seeds = len(ATTACK_REGISTRY["prompt_extraction"].seed_inputs)
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
        assert cases[0].metadata["actor_goal"] == preset.actor_goal
        assert cases[0].metadata["context"] == preset.context

    @patch("strands_evals.redteam.runner._generate_attack_goals")
    def test_uses_custom_goals_with_target_info(self, mock_gen_goals):
        """When target_info is provided, generated goals override preset defaults."""
        mock_gen_goals.return_value = {
            "jailbreak": _goal(
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
            "jailbreak": _goal(goal="Custom jailbreak goal", target_context="Custom context"),
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
        assert extraction_case.metadata["actor_goal"] == ATTACK_REGISTRY["prompt_extraction"].actor_goal

    def test_seed_makes_sampling_reproducible(self):
        """Same seed yields the same selected seed inputs across runs."""
        a = generate_cases(attack_types=["jailbreak"], n_per_type=3, seed=42)
        b = generate_cases(attack_types=["jailbreak"], n_per_type=3, seed=42)
        assert [c.input for c in a] == [c.input for c in b]


# ---------------------------------------------------------------------------
# red_team — exercises the multi-turn task function end-to-end via mocks
# ---------------------------------------------------------------------------
class TestRedTeam:
    """Tests for red_team() orchestration and the task function it builds."""

    def _stub_experiment(self, experiment_cls_mock):
        """Wire Experiment so run_evaluations simply forwards cases through task_fn."""
        experiment = MagicMock()

        def _run(task_fn):
            # Experiment would normally run evaluators; we only need task_fn invoked
            # per case so the multi-turn conversation logic executes.
            for case in experiment_cls_mock.call_args[1]["cases"]:
                task_fn(case)
            return []

        experiment.run_evaluations.side_effect = _run
        experiment_cls_mock.return_value = experiment
        return experiment

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_task_function_returns_conversation(self, mock_sim_class, mock_experiment_class):
        """Task function should produce an output dict with attacker/target turns."""
        mock_sim = MagicMock()
        mock_sim.has_next.side_effect = [True, False]
        mock_sim_class.return_value = mock_sim
        self._stub_experiment(mock_experiment_class)

        target = Mock(return_value="I can't help with that.")
        captured: list[dict] = []

        def capture_task_fn(task_fn):
            # Replace Experiment stub: capture task_fn outputs per case.
            for case in mock_experiment_class.call_args[1]["cases"]:
                captured.append(task_fn(case))
            return []

        mock_experiment_class.return_value.run_evaluations.side_effect = capture_task_fn

        red_team(
            target=target,
            attack_types=["jailbreak"],
            n_per_type=1,
            max_turns=5,
            seed=0,
        )

        assert captured, "task_fn was never invoked"
        conversation = captured[0]["output"]
        assert len(conversation) == 2
        assert conversation[0]["role"] == "attacker"
        assert conversation[1]["role"] == "target"
        assert conversation[1]["content"] == "I can't help with that."

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_multi_turn_conversation(self, mock_sim_class, mock_experiment_class):
        """Simulator drives multiple attacker-target exchanges."""
        mock_sim = MagicMock()
        mock_sim.has_next.side_effect = [True, True, True, False]
        mock_act_result = MagicMock()
        mock_act_result.structured_output = ActorResponse(
            reasoning="Trying different approach",
            message="What about a hypothetical scenario?",
        )
        mock_sim.act.return_value = mock_act_result
        mock_sim_class.return_value = mock_sim

        target = Mock(side_effect=["I can't do that.", "That's also not possible."])
        captured: list[dict] = []

        def run(task_fn):
            for case in mock_experiment_class.call_args[1]["cases"]:
                captured.append(task_fn(case))
            return []

        mock_experiment_class.return_value.run_evaluations.side_effect = run

        red_team(target=target, attack_types=["jailbreak"], n_per_type=1, max_turns=5, seed=0)

        conversation = captured[0]["output"]
        assert len(conversation) == 4
        assert conversation[2]["role"] == "attacker"
        assert conversation[2]["content"] == "What about a hypothetical scenario?"

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_target_error_is_captured(self, mock_sim_class, mock_experiment_class):
        """If the target raises, the conversation records the error instead of crashing."""
        mock_sim = MagicMock()
        mock_sim.has_next.side_effect = [True, False]
        mock_sim_class.return_value = mock_sim

        target = Mock(side_effect=RuntimeError("Connection timeout"))
        captured: list[dict] = []

        def run(task_fn):
            for case in mock_experiment_class.call_args[1]["cases"]:
                captured.append(task_fn(case))
            return []

        mock_experiment_class.return_value.run_evaluations.side_effect = run

        red_team(target=target, attack_types=["jailbreak"], n_per_type=1, max_turns=5, seed=0)

        conversation = captured[0]["output"]
        assert "[Error: Connection timeout]" in conversation[1]["content"]

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_simulator_receives_correct_profile(self, mock_sim_class, mock_experiment_class):
        """ActorSimulator is initialized with profile fields pulled from case metadata."""
        mock_sim = MagicMock()
        mock_sim.has_next.return_value = False
        mock_sim_class.return_value = mock_sim
        self._stub_experiment(mock_experiment_class)

        red_team(
            target=Mock(return_value="OK"),
            attack_types=["jailbreak"],
            n_per_type=1,
            max_turns=3,
            model="test-model",
            seed=0,
        )

        call_kwargs = mock_sim_class.call_args[1]
        assert call_kwargs["max_turns"] == 3
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["actor_profile"].actor_goal == ATTACK_REGISTRY["jailbreak"].actor_goal

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_uses_default_evaluator_when_none(self, mock_sim_class, mock_experiment_class):
        """When evaluators=None, RedTeamJudgeEvaluator is used as the default."""
        mock_sim = MagicMock()
        mock_sim.has_next.return_value = False
        mock_sim_class.return_value = mock_sim
        self._stub_experiment(mock_experiment_class)

        red_team(target=Mock(return_value="OK"), attack_types=["jailbreak"], n_per_type=1, seed=0)

        evaluators = mock_experiment_class.call_args[1]["evaluators"]
        assert len(evaluators) == 1
        from strands_evals.redteam.evaluators.red_team_judge_evaluator import RedTeamJudgeEvaluator

        assert isinstance(evaluators[0], RedTeamJudgeEvaluator)

    @patch("strands_evals.redteam.runner.Experiment")
    @patch("strands_evals.redteam.runner.ActorSimulator")
    def test_uses_custom_evaluators(self, mock_sim_class, mock_experiment_class):
        """Explicit evaluators are forwarded unchanged."""
        mock_sim = MagicMock()
        mock_sim.has_next.return_value = False
        mock_sim_class.return_value = mock_sim
        self._stub_experiment(mock_experiment_class)

        custom_evaluator = Mock()
        red_team(
            target=Mock(return_value="OK"),
            attack_types=["jailbreak"],
            n_per_type=1,
            evaluators=[custom_evaluator],
            seed=0,
        )

        assert mock_experiment_class.call_args[1]["evaluators"] == [custom_evaluator]

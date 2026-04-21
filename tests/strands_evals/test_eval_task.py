"""Tests for @eval_task decorator and handlers."""

from unittest.mock import MagicMock, patch

from strands_evals.case import Case
from strands_evals.eval_task_handler import EvalTaskHandler, TracedHandler, eval_task


class TestEvalTaskDecorator:
    """Tests for the @eval_task decorator."""

    def test_decorated_function_is_callable(self):
        @eval_task()
        def my_task(case):
            return "output"

        assert callable(my_task)

    def test_passes_case_to_function_with_case_param(self):
        received_case = None

        @eval_task()
        def my_task(case):
            nonlocal received_case
            received_case = case
            return "output"

        case = Case(name="test", input="hello")
        my_task(case)
        assert received_case is case

    def test_no_case_param_function_works(self):
        """Functions with no parameters are called without case."""

        @eval_task()
        def my_task():
            return "output"

        result = my_task(Case(name="test", input="hi"))
        assert result == {"output": "output"}

    def test_agent_return_auto_invoked(self):
        """When function returns an Agent, decorator invokes it with case.input."""
        from strands import Agent

        mock_agent = MagicMock(spec=Agent)
        mock_agent.return_value.__str__ = MagicMock(return_value="42")

        @eval_task()
        def my_task():
            return mock_agent

        result = my_task(Case(name="test", input="What is 2+2?"))
        mock_agent.assert_called_once_with("What is 2+2?")
        assert result["output"] == "42"

    def test_agent_return_with_case_param(self):
        """Agent returned from function with case param is also auto-invoked."""
        from strands import Agent

        mock_agent = MagicMock(spec=Agent)
        mock_agent.return_value.__str__ = MagicMock(return_value="answer")

        @eval_task()
        def my_task(case):
            return mock_agent

        result = my_task(Case(name="test", input="question"))
        mock_agent.assert_called_once_with("question")
        assert result["output"] == "answer"

    def test_string_return_wrapped_as_dict(self):
        @eval_task()
        def my_task(case):
            return "the answer"

        result = my_task(Case(name="test", input="hi"))
        assert result == {"output": "the answer"}

    def test_dict_return_passed_through(self):
        @eval_task()
        def my_task(case):
            return {"output": "answer", "custom_key": "value"}

        result = my_task(Case(name="test", input="hi"))
        assert result == {"output": "answer", "custom_key": "value"}

    def test_works_with_experiment(self):
        from strands_evals.evaluators.evaluator import Evaluator
        from strands_evals.experiment import Experiment
        from strands_evals.types.evaluation import EvaluationOutput

        class PassingEvaluator(Evaluator):
            def evaluate(self, evaluation_case):
                return [EvaluationOutput(score=1.0, test_pass=True, reason="pass")]

        from strands import Agent

        mock_agent = MagicMock(spec=Agent)
        mock_agent.return_value.__str__ = MagicMock(return_value="output")

        @eval_task()
        def my_task():
            return mock_agent

        experiment = Experiment(
            cases=[Case(name="test", input="hi")],
            evaluators=[PassingEvaluator()],
        )
        reports = experiment.run_evaluations(my_task)
        assert reports[0].scores[0] == 1.0


class TestEvalTaskHandler:
    """Tests for the base EvalTaskHandler."""

    def test_before_is_noop(self):
        handler = EvalTaskHandler()
        handler.before(Case(name="test", input="hi"))  # should not raise

    def test_after_wraps_string(self):
        handler = EvalTaskHandler()
        result = handler.after(Case(name="test", input="hi"), "output")
        assert result == {"output": "output"}

    def test_after_passes_dict_through(self):
        handler = EvalTaskHandler()
        result = handler.after(Case(name="test", input="hi"), {"output": "x", "extra": "y"})
        assert result == {"output": "x", "extra": "y"}


class TestTracedHandler:
    """Tests for TracedHandler that collects telemetry spans."""

    @patch("strands_evals.eval_task_handler.StrandsEvalsTelemetry")
    @patch("strands_evals.eval_task_handler.StrandsInMemorySessionMapper")
    def test_clears_spans_on_before(self, mock_mapper_cls, mock_telemetry_cls):
        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry_cls.return_value = mock_telemetry

        handler = TracedHandler()
        handler.before(Case(name="test", input="hi"))

        mock_telemetry.in_memory_exporter.clear.assert_called_once()

    @patch("strands_evals.eval_task_handler.StrandsEvalsTelemetry")
    @patch("strands_evals.eval_task_handler.StrandsInMemorySessionMapper")
    def test_after_adds_trajectory(self, mock_mapper_cls, mock_telemetry_cls):
        mock_spans = [MagicMock()]
        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = mock_spans
        mock_telemetry_cls.return_value = mock_telemetry

        mock_session = MagicMock()
        mock_mapper_cls.return_value.map_to_session.return_value = mock_session

        handler = TracedHandler()
        case = Case(name="test", input="hi", session_id="sess-1")
        result = handler.after(case, "output")

        assert result["output"] == "output"
        assert result["trajectory"] is mock_session
        mock_mapper_cls.return_value.map_to_session.assert_called_once_with(list(mock_spans), "sess-1")

    @patch("strands_evals.eval_task_handler.StrandsEvalsTelemetry")
    @patch("strands_evals.eval_task_handler.StrandsInMemorySessionMapper")
    def test_does_not_override_user_trajectory(self, mock_mapper_cls, mock_telemetry_cls):
        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry
        mock_mapper_cls.return_value.map_to_session.return_value = MagicMock()

        handler = TracedHandler()
        user_trajectory = ["tool1", "tool2"]
        result = handler.after(
            Case(name="test", input="hi"),
            {"output": "x", "trajectory": user_trajectory},
        )

        assert result["trajectory"] is user_trajectory

    @patch("strands_evals.eval_task_handler.StrandsEvalsTelemetry")
    @patch("strands_evals.eval_task_handler.StrandsInMemorySessionMapper")
    def test_accepts_custom_mapper(self, mock_mapper_cls, mock_telemetry_cls):
        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        custom_mapper = MagicMock()
        custom_session = MagicMock()
        custom_mapper.map_to_session.return_value = custom_session

        handler = TracedHandler(mapper=custom_mapper)
        result = handler.after(Case(name="test", input="hi", session_id="s1"), "out")

        custom_mapper.map_to_session.assert_called_once()
        assert result["trajectory"] is custom_session

    @patch("strands_evals.eval_task_handler.StrandsEvalsTelemetry")
    @patch("strands_evals.eval_task_handler.StrandsInMemorySessionMapper")
    def test_full_decorator_flow(self, mock_mapper_cls, mock_telemetry_cls):
        mock_telemetry = MagicMock()
        mock_telemetry.setup_in_memory_exporter.return_value = mock_telemetry
        mock_telemetry.in_memory_exporter.get_finished_spans.return_value = []
        mock_telemetry_cls.return_value = mock_telemetry

        mock_session = MagicMock()
        mock_mapper_cls.return_value.map_to_session.return_value = mock_session

        @eval_task(TracedHandler())
        def my_task(case):
            return f"answer to {case.input}"

        result = my_task(Case(name="test", input="question"))
        assert result["output"] == "answer to question"
        assert result["trajectory"] is mock_session

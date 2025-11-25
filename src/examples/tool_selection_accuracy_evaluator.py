from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent
from strands_tools import calculator

from strands_evals import Case, Experiment
from strands_evals.evaluators import ToolSelectionAccuracyEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# Setup telemetry
telemetry = StrandsEvalsTelemetry()
memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
telemetry.tracer_provider.add_span_processor(span_processor)


def user_task_function(case: Case) -> dict:
    memory_exporter.clear()

    agent = Agent(
        trace_attributes={"gen_ai.conversation.id": case.session_id, "session.id": case.session_id},
        tools=[calculator],
        callback_handler=None,
    )
    agent_response = agent(case.input)
    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)
    return {"output": str(agent_response), "trajectory": session}


test_cases = [
    Case[str, str](name="math-1", input="Calculate the square root of 144", metadata={"category": "math"}),
    Case[str, str](
        name="math-2",
        input="What is 25 * 4? can you use that output and then divide it by 4, then the final output should be squared. Give me the final value.",
        metadata={"category": "math"},
    ),
]

evaluator = ToolSelectionAccuracyEvaluator()
experiment = Experiment[str, str](cases=test_cases, evaluator=evaluator)
report = experiment.run_evaluations(user_task_function)
report.run_display()

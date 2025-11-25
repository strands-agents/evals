from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent
from strands_tools import calculator

from strands_evals import Case, Experiment
from strands_evals.evaluators import ToolParameterAccuracyEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# Set up the tracer provider
telemetry = StrandsEvalsTelemetry()
memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
telemetry.tracer_provider.add_span_processor(span_processor)


# ======================================
# SETUP AND RUN STRANDS EVAL
# ======================================
def user_task_function(case: Case) -> dict:
    """Execute agent with tools and capture trajectory."""
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


# Create test cases
test_cases = [
    Case[str, str](
        name="simple-calculation",
        input="Calculate the square root of 144",
        metadata={"category": "math", "difficulty": "easy"},
    ),
    Case[str, str](
        name="percentage-calculation",
        input="What's 20 percent of 250?",
        metadata={"category": "math", "difficulty": "easy"},
    ),
    Case[str, str](
        name="complex-calculation",
        input="I need to calculate 15 + 27, then multiply the result by 3, and finally subtract 10.",
        metadata={"category": "math", "difficulty": "medium"},
    ),
]

# Create evaluators
# The evaluator will check if tool parameters are faithful to the context
evaluators = [ToolParameterAccuracyEvaluator()]
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

# Run evaluations
reports = experiment.run_evaluations(user_task_function)
reports[0].run_display()

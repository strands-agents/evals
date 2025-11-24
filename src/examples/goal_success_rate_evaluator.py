from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent

from strands_evals import Case, Dataset
from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# ======================================
# SETUP TELEMETRY
# ======================================
# 1. Set up the tracer provider
telemetry = StrandsEvalsTelemetry()

# 2. Set up the in-memory exporter
memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
telemetry.tracer_provider.add_span_processor(span_processor)


# ======================================
# SETUP AND RUN STRANDS EVAL
# ======================================


# 1. Define a task function
def user_task_function(case: Case) -> dict:
    agent = Agent(
        trace_attributes={"gen_ai.conversation.id": case.session_id, "session.id": case.session_id},
        callback_handler=None,
    )
    agent_response = agent(case.input)
    finished_spans = memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id=case.session_id)

    return {"output": str(agent_response), "trajectory": session}


# 2. Create test cases
test_cases = [
    Case[str, str](name="math-1", input="What is 25 * 4?", metadata={"category": "math"}),
    Case[str, str](name="math-2", input="Calculate the square root of 144", metadata={"category": "math"}),
]

# 3. Create an evaluator
evaluator = GoalSuccessRateEvaluator()

# 4. Create a dataset
dataset = Dataset[str, str](cases=test_cases, evaluator=evaluator)

# 5. Run evaluations
report = dataset.run_evaluations(user_task_function)
report.run_display()

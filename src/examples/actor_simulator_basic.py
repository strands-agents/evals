from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent

from strands_evals import ActorSimulator, Case, Dataset
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# ======================================
# SETUP TELEMETRY
# ======================================
telemetry = StrandsEvalsTelemetry()
memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
telemetry.tracer_provider.add_span_processor(span_processor)


# ======================================
# SETUP AND RUN STRANDS EVAL
# ======================================


def task_function(case: Case) -> dict:
    # Create simulator
    user_sim = ActorSimulator.from_case_for_user_simulator(case=case, max_turns=3)

    # Create target agent
    agent = Agent(
        trace_attributes={"gen_ai.conversation.id": case.session_id, "session.id": case.session_id},
        system_prompt="You are a helpful travel assistant.",
        callback_handler=None,
    )

    # Accumulate target spans across all turns
    all_target_spans = []

    user_message = case.input
    while user_sim.has_next():
        # Clear before each target agent call to ensure we don't capture simulator traces.
        memory_exporter.clear()
        agent_response = agent(user_message)
        agent_message = str(agent_response)
        turn_spans = list(memory_exporter.get_finished_spans())
        all_target_spans.extend(turn_spans)
        user_result = user_sim.act(agent_message)
        user_message = str(user_result.structured_output.message)

    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(all_target_spans, session_id=case.session_id)

    return {"output": agent_message, "trajectory": session}


test_cases = [
    Case[str, str](
        name="booking-simple",
        input="I need to book a flight to Paris next week",
        metadata={"category": "booking", "task_description": "Flight booking confirmed"},
    )
]

evaluator = HelpfulnessEvaluator()
dataset = Dataset[str, str](cases=test_cases, evaluator=evaluator)

report = dataset.run_evaluations(task_function)
report.run_display()

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent, tool
from strands_evals import Case, Dataset
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper

# ======================================
# SETUP TELEMETRY
# ======================================
provider = TracerProvider()
trace.set_tracer_provider(provider)

memory_exporter = InMemorySpanExporter()
span_processor = BatchSpanProcessor(memory_exporter)
provider.add_span_processor(span_processor)


# ======================================
# DEFINE TOOLS
# ======================================
@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")

    Returns:
        The result of the calculation
    """
    return eval(expression)


# ======================================
# SETUP AND RUN STRANDS EVAL
# ======================================
def user_task_function(query: str) -> dict:
    agent = Agent(tools=[calculator], callback_handler=None)
    agent_response = agent(query)
    finished_spans = memory_exporter.get_finished_spans()
    import json
    
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id="test-session")
    print('=== agent.response.metrics.traces ===')
    print([json.dumps(trace.to_dict()) for trace in agent_response.metrics.traces])
    print('\n')
    print('=== finished_spans ===')
    print([span.to_json() for span in finished_spans])
    print(finished_spans[0])
    print(dir(finished_spans[0]))
    print(finished_spans[0].attributes) 
    print(finished_spans[0].events)
    print(finished_spans[0].context)
    print('\n')
    print('=== session ===')
    print(session)
    return {"output": str(agent_response), "trajectory": session}


# Create test cases with tool usage
test_cases = [
    Case[str, str](
        name="math-1",
        input="What is 15 multiplied by 23?",
        metadata={"category": "math"}
    ),
    Case[str, str](
        name="math-2",
        input="Calculate the square root of 144",
        metadata={"category": "math"}
    ),
]

# Create evaluator
evaluator = HelpfulnessEvaluator()

# Create dataset
dataset = Dataset[str, str](cases=test_cases, evaluator=evaluator)

# Run evaluations
report = dataset.run_evaluations(user_task_function)
report.run_display()

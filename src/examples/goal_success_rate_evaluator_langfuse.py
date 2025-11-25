from langfuse_utils.utils import get_langfuse_trace

from strands_evals import Case, Dataset
from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.mappers.langfuse_mapper import LangfuseSessionMapper


# 1. Define a task function
def user_task_function(case: Case) -> dict:
    mapper = LangfuseSessionMapper()
    trace_id = case.metadata.get("trace_id", "") if case.metadata else ""
    langfuse_traces = get_langfuse_trace(trace_id=trace_id)
    session = mapper.map_to_session(langfuse_traces, trace_id)
    agent_response = mapper.get_agent_final_response(session)

    return {"output": str(agent_response), "trajectory": session}


test_cases = [
    Case[str, str](
        name="knowledge-1",
        input="What is the status of my package",
        metadata={"category": "knowledge", "trace_id": "13cc6db7f2e290b54bab3061a6fc5db0"},
    ),
]


# 3. Create an evaluator
evaluator = GoalSuccessRateEvaluator()

# 4. Create a dataset
dataset = Dataset[str, str](cases=test_cases, evaluator=evaluator)

# 5. Run evaluations
report = dataset.run_evaluations(user_task_function)
report.run_display()

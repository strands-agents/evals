# Strands Evaluation

## Setup

### Installation
```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with dependencies
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with both test and dev dependencies
pip install -e ".[test,dev]"
```

## Basic Usage

### OutuptEvaluator (LLM judge)

```python
from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator

# 1. Create test cases
test_cases = [
    Case[str, str](
        name="knowledge-1",
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        metadata={"category": "knowledge"}
    ),
    Case[str, str](
        name="knowledge-2",
        input="What color is the ocean?",
        metadata={"category": "knowledge"}
    )
]

# 2. Create evaluators
evaluators = [
    OutputEvaluator(
        rubric="The output should represent a reasonable answer to the input."
    )
]

# 3. Create an experiment
experiment = Experiment[str, str](
    cases=test_cases,
    evaluators=evaluators
)

# 4. Define a task function
def get_response(case: Case) -> str:
    agent = Agent(callback_handler=None)
    return str(agent(case.input))

# 5. Run evaluations
reports = experiment.run_evaluations(get_response)
reports[0].run_display()
```

### Trace-based Evaluator

```python
from strands import Agent

from strands_evals import Case, Experiment
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.telemetry import StrandsEvalsTelemetry


# 1. Set up the tracer provider with in_memory_exporter
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()


# 2. Create test cases
test_cases = [
    Case[str, str](name="knowledge-1", input="What is the capital of France?", metadata={"category": "knowledge"}),
    Case[str, str](name="knowledge-2", input="What color is the ocean?", metadata={"category": "knowledge"}),
]

# 3. Create evaluators
evaluators = [HelpfulnessEvaluator()]

# 4. Create an experiment
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

# 5. Define a task function
def user_task_function(case: Case) -> dict:
    agent = Agent(callback_handler=None)
    agent_response = agent(case.input)
    finished_spans = telemetry.memory_exporter.get_finished_spans()
    
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(finished_spans, session_id="test-session")

    return {"output": str(agent_response), "trajectory": session}

# 6. Run evaluations
reports = experiment.run_evaluations(user_task_function)
reports[0].run_display()
```

## Saving and Loading Experiments

```python
# Save experiment to JSON
experiment.to_file("my_experiment", "json")

# Load experiment from JSON
loaded_experiment = Experiment.from_file("./experiment_files/my_experiment.json", "json")
```

## Custom Evaluators

```python
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class CustomEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Custom evaluation logic
        if evaluation_case.expected_output in evaluation_case.actual_output:
            score = 1.0
            test_pass = True
        else:
            score = 0.0
            test_pass = False
            
        return EvaluationOutput(
            score=score,
            test_pass=test_pass,
            reason="Custom evaluation reason"
        )

# Use custom evaluator
experiment = Experiment[str, str](
    cases=test_cases,
    evaluators=[CustomEvaluator()]
)
```

## Evaluating Tool Usage

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import TrajectoryEvaluator
from strands_tools import calculator

# 1. Define task that returns tool usage
def get_response_with_tools(case: Case) -> dict:
    agent = Agent(tools=[calculator])
    response = agent(case.input)
    
    return {
        "output": str(response),
        "trajectory": list(response.metrics.tool_metrics.keys())
    }

# 2. Create test cases with expected tool trajectories
test_case = Case[str, str](
    name="calculator-1",
    input="What is the square root of 9?",
    expected_output="The square root of 9 is 3.",
    expected_trajectory=["calculator"],
    metadata={"category": "math"}
)

# 3. Create trajectory evaluator
trajectory_evaluator = TrajectoryEvaluator(
    rubric="Scoring should measure how well the agent uses appropriate tools for the given task.",
    include_inputs=True
)

# 4. Create experiment and run evaluations
experiment = Experiment[str, str](
    cases=[test_case],
    evaluators=[trajectory_evaluator]
)

reports = experiment.run_evaluations(get_response_with_tools)
```

## Experiment Generation

```python
from strands_evals.generators import ExperimentGenerator
from strands_evals.evaluators import TrajectoryEvaluator

# 1. Define tool context
tool_context = """
Available tools:
- calculator(expression: str) -> float: Evaluate mathematical expressions
- web_search(query: str) -> str: Search the web for information
"""

# 2. Generate experiment from context
generator = ExperimentGenerator[str, str](str, str)

experiment = await generator.from_context_async(
    context=tool_context,
    num_cases=10,
    evaluator=TrajectoryEvaluator,
    task_description="Math and research assistant with tool usage",
    # num_topics=3  # Optional: Distribute cases across multiple topics
)

# 3. Save generated experiment
experiment.to_file("generated_math_research_experiment")
```

## Available Evaluators

- **OutputEvaluator**: Evaluates the quality and correctness of agent outputs
- **TrajectoryEvaluator**: Evaluates the sequence of tools/actions used by agents
- **InteractionsEvaluator**: Evaluates multi-agent interactions and handoffs
- **Custom Evaluators**: Create your own evaluation logic by extending the base Evaluator class

## More Examples

See the `examples` directory for more detailed examples.

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
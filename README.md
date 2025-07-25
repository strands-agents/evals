# Strands Evaluation

## Setup

### Installation

The package uses `pyproject.toml` for dependency management. To install:

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

### Running Tests

```bash
# Run all tests
cd tests
pytest .
```

## Basic Usage

```python
from strands import Agent
from strands_evaluation.dataset import Dataset
from strands_evaluation.case import Case
from strands_evaluation.evaluators.output_evaluator import OutputEvaluator

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

# 2. Create an evaluator
evaluator = OutputEvaluator(
    rubric="The output should represent a reasonable answer to the input."
)

# 3. Create a dataset
dataset = Dataset[str, str](
    cases=test_cases,
    evaluator=evaluator
)

# 4. Define a task function
def get_response(query: str) -> str:
    agent = Agent(callback_handler=None)
    return str(agent(query))

# 5. Run evaluations
report = dataset.run_evaluations(get_response)
report.display()
```

## Saving and Loading Datasets

```python
# Save dataset to JSON
dataset.to_file("my_dataset", "json")

# Load dataset from JSON
loaded_dataset = Dataset.from_file("./dataset_files/my_dataset.json", "json")
```

## Custom Evaluators

```python
from strands_evaluation.evaluators.evaluator import Evaluator
from strands_evaluation.types.evaluation import EvaluationData, EvaluationOutput

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
dataset = Dataset[str, str](
    cases=test_cases,
    evaluator=CustomEvaluator()
)
```

## Evaluating Tool Usage

```python
from strands_tools import calculator
from strands_evaluation.evaluators.trajectory_evaluator import TrajectoryEvaluator

# Create test cases with expected tool trajectories
test_case = Case[str, str](
    name="calculator-1",
    input="What is the square root of 9?",
    expected_output="The square root of 9 is 3.",
    expected_trajectory=["calculator"],
    metadata={"category": "math"}
)

# Create trajectory evaluator
trajectory_evaluator = TrajectoryEvaluator(
    rubric="The trajectory should represent a reasonable use of tools based on the input.",
    include_inputs=True
)

# Define task that returns tool usage
def get_response_with_tools(query: str) -> dict:
    agent = Agent(tools=[calculator])
    response = agent(query)
    
    return {
        "output": str(response),
        "trajectory": list(response.metrics.tool_metrics.keys())
    }

# Create dataset and run evaluations
dataset = Dataset[str, str](
    cases=[test_case],
    evaluator=trajectory_evaluator
)

report = dataset.run_evaluations(get_response_with_tools)
```

## Async Evaluation

For improved performance with many test cases, use async evaluation:

```python
import asyncio
from strands_evaluation.dataset import Dataset
from strands_evaluation.evaluators.output_evaluator import OutputEvaluator

# Create dataset with cases and evaluator
dataset = Dataset(cases=test_cases, evaluator=OutputEvaluator(rubric="Test rubric"))

# Define async task function (optional)
async def async_task(query):
    agent = Agent(callback_handler=None)
    response = await agent.invoke_async(query)
    return str(response)

# Run evaluations asynchronously (works with both sync and async task functions)
async def main():
    report = await dataset.run_evaluations_async(async_task, max_workers=5)
    report.display()
    return report

# Run the async function
report = asyncio.run(main())
```

## More Examples

See the `examples` directory for more detailed examples.

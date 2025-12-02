<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Evals SDK
  </h1>
  <h2>
    A comprehensive evaluation framework for AI agents and LLM applications.
  </h2>

  <div align="center">
    <a href="https://github.com/strands-agents/evals/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/evals"/></a>
    <a href="https://github.com/strands-agents/evals/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/evals"/></a>
    <a href="https://github.com/strands-agents/evals/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/evals"/></a>
    <a href="https://github.com/strands-agents/evals/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/evals"/></a>
    <a href="https://pypi.org/project/strands-agents-evals/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-agents-evals"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/strands-agents-evals"/></a>
  </div>
  
  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/sdk-typescript">Typescript SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/evals">Evaluations</a>
  </p>
</div>

Strands Evaluation is a powerful framework for evaluating AI agents and LLM applications. From simple output validation to complex multi-agent interaction analysis, trajectory evaluation, and automated experiment generation, Strands Evaluation provides comprehensive tools to measure and improve your AI systems.

## Feature Overview

- **Multiple Evaluation Types**: Output evaluation, trajectory analysis, tool usage assessment, and interaction evaluation
- **LLM-as-a-Judge**: Built-in evaluators using language models for sophisticated assessment with structured scoring
- **Trace-based Evaluation**: Analyze agent behavior through OpenTelemetry execution traces
- **Automated Experiment Generation**: Generate comprehensive test suites from context descriptions
- **Custom Evaluators**: Extensible framework for domain-specific evaluation logic
- **Experiment Management**: Save, load, and version your evaluation experiments with JSON serialization
- **Built-in Scoring Tools**: Helper functions for exact, in-order, and any-order trajectory matching

## Quick Start

```bash
# Install Strands Evals SDK
pip install strands-agents-evals
```

```python
from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import OutputEvaluator

# Create test cases
test_cases = [
    Case[str, str](
        name="knowledge-1",
        input="What is the capital of France?",
        expected_output="The capital of France is Paris.",
        metadata={"category": "knowledge"}
    )
]

# Create evaluators with custom rubric
evaluators = [
    OutputEvaluator(
        rubric="""
        Evaluate based on:
        1. Accuracy - Is the information correct?
        2. Completeness - Does it fully answer the question?
        3. Clarity - Is it easy to understand?
        
        Score 1.0 if all criteria are met excellently.
        Score 0.5 if some criteria are partially met.
        Score 0.0 if the response is inadequate.
        """
    )
]

# Create experiment and run evaluation
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

def get_response(case: Case) -> str:
    agent = Agent(callback_handler=None)
    return str(agent(case.input))

# Run evaluations
reports = experiment.run_evaluations(get_response)
reports[0].run_display()
```

## Installation

Ensure you have Python 3.10+ installed, then:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e ".[test]"

# Install with both test and dev dependencies
pip install -e ".[test,dev]"
```

## Features at a Glance

### Output Evaluation with Custom Rubrics

Evaluate agent responses using LLM-as-a-judge with flexible scoring criteria:

```python
from strands_evals.evaluators import OutputEvaluator

evaluator = OutputEvaluator(
    rubric="Score 1.0 for accurate, complete responses. Score 0.5 for partial answers. Score 0.0 for incorrect or unhelpful responses.",
    include_inputs=True,  # Include context in evaluation
    model="us.anthropic.claude-sonnet-4-20250514-v1:0"  # Custom judge model
)
```

### Trajectory Evaluation with Built-in Scoring

Analyze agent tool usage and action sequences with helper scoring functions:

```python
from strands_evals.evaluators import TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from strands_tools import calculator

def get_response_with_tools(case: Case) -> dict:
    agent = Agent(tools=[calculator])
    response = agent(case.input)
    
    # Extract trajectory efficiently to prevent context overflow
    trajectory = tools_use_extractor.extract_agent_tools_used_from_messages(agent.messages)
    
    # Update evaluator with tool descriptions
    evaluator.update_trajectory_description(
        tools_use_extractor.extract_tools_description(agent, is_short=True)
    )
    
    return {"output": str(response), "trajectory": trajectory}

# Evaluator includes built-in scoring tools: exact_match_scorer, in_order_match_scorer, any_order_match_scorer
evaluator = TrajectoryEvaluator(
    rubric="Score 1.0 if correct tools used in proper sequence. Use scoring tools to verify trajectory matches."
)
```

### Trace-based Helpfulness Evaluation

Evaluate agent helpfulness using OpenTelemetry traces with seven-level scoring:

```python
from strands_evals.evaluators import HelpfulnessEvaluator
from strands_evals.telemetry import StrandsEvalsTelemetry
from strands_evals.mappers import StrandsInMemorySessionMapper

# Setup telemetry for trace capture
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()

def user_task_function(case: Case) -> dict:
    telemetry.memory_exporter.clear()
    
    agent = Agent(
        trace_attributes={"session.id": case.session_id},
        callback_handler=None
    )
    response = agent(case.input)
    
    # Map spans to session for evaluation
    spans = telemetry.memory_exporter.get_finished_spans()
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(spans, session_id=case.session_id)
    
    return {"output": str(response), "trajectory": session}

# Seven-level scoring: Not helpful (0.0) to Above and beyond (1.0)
evaluators = [HelpfulnessEvaluator()]
experiment = Experiment[str, str](cases=test_cases, evaluators=evaluators)

# Run evaluations
reports = experiment.run_evaluations(user_task_function)
reports[0].run_display()
```

### Automated Experiment Generation

Generate comprehensive test suites automatically from context descriptions:

```python
from strands_evals.generators import ExperimentGenerator
from strands_evals.evaluators import TrajectoryEvaluator

# Define available tools and context
tool_context = """
Available tools:
- calculator(expression: str) -> float: Evaluate mathematical expressions
- web_search(query: str) -> str: Search the web for information
- file_read(path: str) -> str: Read file contents
"""

# Generate experiment with multiple test cases
generator = ExperimentGenerator[str, str](str, str)
experiment = await generator.from_context_async(
    context=tool_context,
    num_cases=10,
    evaluator=TrajectoryEvaluator,
    task_description="Math and research assistant with tool usage",
    num_topics=3  # Distribute cases across multiple topics
)

# Save generated experiment
experiment.to_file("generated_experiment", "json")
```

### Custom Evaluators with Structured Output

Create domain-specific evaluation logic with standardized output format:

```python
from strands_evals.evaluators import Evaluator
from strands_evals.types import EvaluationData, EvaluationOutput

class PolicyComplianceEvaluator(Evaluator[str, str]):
    def evaluate(self, evaluation_case: EvaluationData[str, str]) -> EvaluationOutput:
        # Custom evaluation logic
        response = evaluation_case.actual_output
        
        # Check for policy violations
        violations = self._check_policy_violations(response)
        
        if not violations:
            return EvaluationOutput(
                score=1.0,
                test_pass=True,
                reason="Response complies with all policies",
                label="compliant"
            )
        else:
            return EvaluationOutput(
                score=0.0,
                test_pass=False,
                reason=f"Policy violations: {', '.join(violations)}",
                label="non_compliant"
            )
    
    def _check_policy_violations(self, response: str) -> list[str]:
        # Implementation details...
        return []
```

### Tool Usage and Parameter Evaluation

Evaluate specific aspects of tool usage with specialized evaluators:

```python
from strands_evals.evaluators import ToolSelectionAccuracyEvaluator, ToolParameterAccuracyEvaluator

# Evaluate if correct tools were selected
tool_selection_evaluator = ToolSelectionAccuracyEvaluator(
    rubric="Score 1.0 if optimal tools selected, 0.5 if suboptimal but functional, 0.0 if wrong tools"
)

# Evaluate if tool parameters were correct
tool_parameter_evaluator = ToolParameterAccuracyEvaluator(
    rubric="Score based on parameter accuracy and appropriateness for the task"
)
```

## Available Evaluators

### Core Evaluators
- **OutputEvaluator**: Flexible LLM-based evaluation with custom rubrics
- **TrajectoryEvaluator**: Action sequence evaluation with built-in scoring tools
- **HelpfulnessEvaluator**: Seven-level helpfulness assessment from user perspective
- **FaithfulnessEvaluator**: Evaluates if responses are grounded in conversation history
- **GoalSuccessRateEvaluator**: Measures if user goals were achieved

### Specialized Evaluators
- **ToolSelectionAccuracyEvaluator**: Evaluates appropriateness of tool choices
- **ToolParameterAccuracyEvaluator**: Evaluates correctness of tool parameters
- **InteractionsEvaluator**: Multi-agent interaction and handoff evaluation
- **Custom Evaluators**: Extensible base class for domain-specific logic

## Experiment Management and Serialization

Save, load, and version experiments for reproducibility:

```python
# Save experiment with metadata
experiment.to_file("customer_service_eval", "json")

# Load experiment from file
loaded_experiment = Experiment.from_file("./experiment_files/customer_service_eval.json", "json")

# Experiment files include:
# - Test cases with metadata
# - Evaluator configuration
# - Expected outputs and trajectories
# - Versioning information
```

## Evaluation Metrics and Analysis

Track comprehensive metrics across multiple dimensions:

```python
# Built-in metrics to consider:
metrics = {
    "accuracy": "Factual correctness of responses",
    "task_completion": "Whether agent completed the task",
    "tool_selection": "Appropriateness of tool choices", 
    "response_time": "Agent response latency",
    "hallucination_rate": "Frequency of fabricated information",
    "token_usage": "Efficiency of token consumption",
    "user_satisfaction": "Subjective helpfulness ratings"
}

# Generate analysis reports
reports = experiment.run_evaluations(task_function)
reports[0].run_display()  # Interactive display with metrics breakdown
```

## Best Practices

### Evaluation Strategy
1. **Diversify Test Cases**: Cover knowledge, reasoning, tool usage, conversation, edge cases, and safety scenarios
2. **Use Statistical Baselines**: Run multiple evaluations to account for LLM non-determinism
3. **Combine Multiple Evaluators**: Use output, trajectory, and helpfulness evaluators together
4. **Regular Evaluation Cadence**: Implement consistent evaluation schedules for continuous improvement

### Performance Optimization
1. **Use Extractors**: Always use `tools_use_extractor` functions to prevent context overflow
2. **Update Descriptions Dynamically**: Call `update_trajectory_description()` with tool descriptions
3. **Choose Appropriate Judge Models**: Use stronger models for complex evaluations
4. **Batch Evaluations**: Process multiple test cases efficiently

### Experiment Design
1. **Write Clear Rubrics**: Include explicit scoring criteria and examples
2. **Include Expected Trajectories**: Define exact sequences for trajectory evaluation
3. **Use Appropriate Matching**: Choose between exact, in-order, or any-order matching
4. **Version Control**: Track agent configurations alongside evaluation results

## Documentation

For detailed guidance & examples, explore our documentation:

- [User Guide](https://strandsagents.com/latest//user-guide/evals-sdk/quickstart.md)
- [Evaluator Reference](https://strandsagents.com/latest/user-guide/evals-sdk/evaluators/)

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

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
- **Multimodal Evaluation**: MLLM-as-a-Judge evaluators for image-to-text tasks with built-in rubrics
- **Dynamic Simulators**: Multi-turn conversation simulation with realistic user behavior, goal-oriented interactions, and LLM-powered tool simulation with shared state
- **LLM-as-a-Judge**: Built-in evaluators using language models for sophisticated assessment with structured scoring
- **Trace-based Evaluation**: Analyze agent behavior through OpenTelemetry execution traces
- **Automated Experiment Generation**: Generate comprehensive test suites from context descriptions
- **Custom Evaluators**: Extensible framework for domain-specific evaluation logic
- **Experiment Management**: Save, load, and version your evaluation experiments with JSON serialization
- **Built-in Scoring Tools**: Helper functions for exact, in-order, and any-order trajectory matching
- **Failure Detection & Root Cause Analysis**: Automatically detect failures in agent sessions and diagnose root causes with actionable fix recommendations

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

### Multi-turn Conversation Simulation

Simulate realistic user interactions with dynamic, goal-oriented conversations using ActorSimulator:

```python
from strands import Agent
from strands_evals import Case, Experiment, ActorSimulator
from strands_evals.evaluators import HelpfulnessEvaluator, GoalSuccessRateEvaluator
from strands_evals.mappers import StrandsInMemorySessionMapper
from strands_evals.telemetry import StrandsEvalsTelemetry

# Setup telemetry
telemetry = StrandsEvalsTelemetry().setup_in_memory_exporter()
memory_exporter = telemetry.in_memory_exporter

def task_function(case: Case) -> dict:
    # Create simulator to drive conversation
    simulator = ActorSimulator.from_case_for_user_simulator(
        case=case,
        max_turns=10
    )

    # Create agent to evaluate
    agent = Agent(
        trace_attributes={
            "gen_ai.conversation.id": case.session_id,
            "session.id": case.session_id
        },
        callback_handler=None
    )

    # Run multi-turn conversation
    all_spans = []
    user_message = case.input

    while simulator.has_next():
        memory_exporter.clear()
        agent_response = agent(user_message)
        turn_spans = list(memory_exporter.get_finished_spans())
        all_spans.extend(turn_spans)

        user_result = simulator.act(str(agent_response))
        user_message = str(user_result.structured_output.message)

    # Map to session for evaluation
    mapper = StrandsInMemorySessionMapper()
    session = mapper.map_to_session(all_spans, session_id=case.session_id)

    return {"output": str(agent_response), "trajectory": session}

# Use evaluators to assess simulated conversations
evaluators = [
    HelpfulnessEvaluator(),
    GoalSuccessRateEvaluator()
]

experiment = Experiment(cases=test_cases, evaluators=evaluators)
reports = experiment.run_evaluations(task_function)
```

**Key Benefits:**
- **Dynamic Interactions**: Simulator adapts responses based on agent behavior
- **Goal-Oriented Testing**: Verify agents can complete user objectives through dialogue
- **Realistic Conversations**: Generate authentic multi-turn interaction patterns
- **No Predefined Scripts**: Test agents without hardcoded conversation paths
- **Comprehensive Evaluation**: Combine with trace-based evaluators for full assessment

### Tool Simulation

Simulate tool behavior with LLM-powered responses for controlled agent evaluation using ToolSimulator. Register tools with a decorator, define output schemas, and optionally share state across related tools — the simulator replaces real execution with realistic, schema-validated responses:

```python
from typing import Any
from enum import Enum
from pydantic import BaseModel, Field
from strands import Agent
from strands_evals import Case, Experiment
from strands_evals.evaluators import GoalSuccessRateEvaluator
from strands_evals.simulation.tool_simulator import ToolSimulator

tool_simulator = ToolSimulator()

# Define output schema
class HVACMode(str, Enum):
    HEAT = "heat"
    COOL = "cool"
    AUTO = "auto"
    OFF = "off"

class HVACResponse(BaseModel):
    temperature: float = Field(..., description="Target temperature in Fahrenheit")
    mode: HVACMode = Field(..., description="HVAC mode")
    status: str = Field(default="success", description="Operation status")

# Register tool — the function body is never called; the LLM generates responses
@tool_simulator.tool(
    share_state_id="room_environment",
    initial_state_description="Room: 68°F, humidity 45%, HVAC off",
    output_schema=HVACResponse,
)
def hvac_controller(temperature: float, mode: str) -> dict[str, Any]:
    """Control heating/cooling system that affects room temperature and humidity."""
    pass

def task_function(case: Case) -> dict:
    hvac_tool = tool_simulator.get_tool("hvac_controller")
    agent = Agent(tools=[hvac_tool], callback_handler=None)
    response = agent(case.input)
    return {"output": str(response)}

cases = [Case(name="heat_control", input="Turn on the heat to 72 degrees")]
experiment = Experiment(cases=cases, evaluators=[GoalSuccessRateEvaluator()])
reports = experiment.run_evaluations(task_function)
```

**Key Benefits:**
- **No Real Infrastructure**: Test tool-using agents without live APIs, databases, or services
- **Schema-Validated Responses**: Pydantic output schemas ensure structured, consistent tool responses
- **Shared State**: Related tools (e.g., sensor + controller) share state via `share_state_id` for coherent behavior
- **Stateful Context**: Call history and initial state are passed to the LLM for consistent multi-call sequences
- **Drop-in Replacement**: Simulated tools plug directly into Strands `Agent` via `get_tool()`

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

### Failure Detection & Diagnosis

Detect failures in agent sessions and analyze root causes to get actionable fix recommendations:

```python
from strands_evals import Case, DiagnosisConfig, Experiment
from strands_evals.evaluators import HelpfulnessEvaluator

# Enable diagnosis on failing cases — runs failure detection then root cause analysis
experiment = Experiment(
    cases=test_cases,
    evaluators=[HelpfulnessEvaluator()],
    diagnosis_config=DiagnosisConfig(
        trigger="on_failure",           # "on_failure" or "always"
        confidence_threshold="medium",  # "low", "medium", or "high"
    ),
)

reports = experiment.run_evaluations(task_function)

# Display results with recommendations
reports[0].display(include_recommendations=True)
```

You can also use the detectors standalone on any `Session` object (`strands_evals.types.trace.Session`):

```python
from strands_evals.detectors import detect_failures, analyze_root_cause, diagnose_session

# Full pipeline: detect failures → root cause analysis
result = diagnose_session(session, confidence_threshold="medium")

for rc in result.root_causes:
    print(f"{rc.fix_type}: {rc.fix_recommendation}")

# Or run each step independently
failures = detect_failures(session, confidence_threshold="medium")
if failures.failures:
    rca = analyze_root_cause(session, failures=failures.failures)

# analyze_root_cause calls detect_failures automatically if failures is not provided
rca = analyze_root_cause(session)
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

### Multimodal Evaluation (MLLM-as-a-Judge)

Evaluate multimodal agent responses involving images and text using MLLM-as-a-Judge:

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import (
    MultimodalCorrectnessEvaluator,
    MultimodalFaithfulnessEvaluator,
    MultimodalInstructionFollowingEvaluator,
    MultimodalOverallQualityEvaluator,
)
from strands_evals.types import ImageData, MultimodalInput

# Create image data from various sources
image = ImageData(source="path/to/image.png")
# Also supports: base64 strings, data URLs, HTTP URLs, PIL Images, raw bytes

# Define test cases with multimodal input
test_cases = [
    Case[MultimodalInput, str](
        name="image-description-1",
        input=MultimodalInput(
            media=image,
            instruction="Describe the contents of this image in detail.",
        ),
    )
]

# Use built-in evaluators with default rubrics (reference-free by default; reference-based rubric auto-selected when expected_output is provided)
evaluators = [
    MultimodalCorrectnessEvaluator(),       # Factual accuracy and completeness
    MultimodalFaithfulnessEvaluator(),      # Grounded in media without hallucinations
    MultimodalInstructionFollowingEvaluator(),  # Addresses query requirements
    MultimodalOverallQualityEvaluator(),    # Visual accuracy, adherence, completeness, coherence
]

# Run evaluation
experiment = Experiment[MultimodalInput, str](cases=test_cases, evaluators=evaluators)

def get_response(case: Case) -> str:
    agent = Agent(callback_handler=None)
    image = case.input.media
    return str(agent([
        {"image": {"format": image.format or "png", "source": {"bytes": image.to_bytes()}}},
        {"text": case.input.instruction}
    ]))

reports = experiment.run_evaluations(get_response)
reports[0].run_display()
```

## Available Evaluators

### Output-Based Evaluators
These evaluators work directly with inputs and outputs without requiring OpenTelemetry traces:

- **OutputEvaluator**: Flexible LLM-based evaluation with custom rubrics
- **TrajectoryEvaluator**: Action sequence evaluation with built-in scoring tools (supports both list-based trajectories and Session traces via extractors)
- **InteractionsEvaluator**: Multi-agent interaction and handoff evaluation
- **Custom Evaluators**: Extensible base class for domain-specific logic

### Trace-Based Evaluators
These evaluators require OpenTelemetry traces (Session objects) to analyze agent behavior:

#### Tool-Level Evaluators
Evaluate individual tool calls within a conversation:
- **ToolSelectionAccuracyEvaluator**: Evaluates appropriateness of tool choices at specific points
- **ToolParameterAccuracyEvaluator**: Evaluates correctness of tool parameters based on context

#### Trace-Level Evaluators
Evaluate the most recent turn in a conversation:
- **HelpfulnessEvaluator**: Seven-level helpfulness assessment from user perspective
- **FaithfulnessEvaluator**: Evaluates if responses are grounded in conversation history
- **CoherenceEvaluator**: Assesses logical cohesion and reasoning quality with five-level scoring
- **ConcisenessEvaluator**: Evaluates response brevity with three-level scoring
- **ResponseRelevanceEvaluator**: Evaluates relevance of responses to user questions
- **HarmfulnessEvaluator**: Binary evaluation for harmful content detection

#### Session-Level Evaluators
Evaluate entire conversation sessions:
- **GoalSuccessRateEvaluator**: Measures if user goals were achieved across the full conversation

### Multimodal Evaluators
These evaluators assess multimodal (image-to-text) responses using MLLM-as-a-Judge, with built-in rubrics that support both reference-based and reference-free evaluation:

- **MultimodalCorrectnessEvaluator**: Evaluates factual accuracy and completeness against media content
- **MultimodalFaithfulnessEvaluator**: Evaluates whether responses are grounded in media without hallucinations
- **MultimodalInstructionFollowingEvaluator**: Evaluates whether responses address query requirements and constraints
- **MultimodalOverallQualityEvaluator**: Holistic evaluation across visual accuracy, instruction adherence, completeness, and coherence
- **MultimodalOutputEvaluator**: Base class for custom multimodal evaluators with configurable rubrics

### Detectors
Analyze agent sessions to identify failures and their root causes:

- **detect_failures**: Scans a Session for failures with configurable confidence thresholds
- **analyze_root_cause**: Identifies root causes for detected failures with fix recommendations
- **diagnose_session**: End-to-end pipeline combining failure detection and root cause analysis

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

# Flatten multiple evaluator reports into a single combined view
from strands_evals.types.evaluation_report import EvaluationReport
combined = EvaluationReport.flatten(reports)
combined.display(include_recommendations=True)
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

- [User Guide](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/quickstart/)
- [Evaluator Reference](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/evaluators/)
- [Simulators Guide](https://strandsagents.com/latest/documentation/docs/user-guide/evals-sdk/simulators/)

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

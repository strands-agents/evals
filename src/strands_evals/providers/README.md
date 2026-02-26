# Trace Providers

Trace providers fetch agent execution data from observability backends and convert it into the format the evaluation pipeline expects. This lets you run evaluators against traces from production or staging agents without re-running them.

## Available Providers

| Provider | Backend | Auth |
|----------|---------|------|
| `CloudWatchProvider` | AWS CloudWatch Logs (Bedrock AgentCore runtime logs) | AWS credentials (boto3) |
| `LangfuseProvider` | Langfuse | API keys |

## Quick Start

### CloudWatch

```python
from strands_evals.providers import CloudWatchProvider

# Option 1: Provide the log group directly
provider = CloudWatchProvider(
    log_group="/aws/bedrock-agentcore/runtimes/my-agent-abc123-DEFAULT",
    region="us-east-1",
)

# Option 2: Discover the log group from the agent name
provider = CloudWatchProvider(agent_name="my-agent", region="us-east-1")
```

### Langfuse

```python
from strands_evals.providers import LangfuseProvider

# Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env by default
provider = LangfuseProvider()

# Or pass credentials explicitly
provider = LangfuseProvider(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://us.cloud.langfuse.com",
)
```

## Core API

All providers implement the `TraceProvider` interface:

```python
# Fetch traces for a session, ready for evaluation
data = provider.get_evaluation_data(session_id="my-session-id")
# data["output"]     -> str (final agent response)
# data["trajectory"] -> Session (traces and spans)
```

## Running Evaluators on Remote Traces

Pass the provider's data into the standard `Experiment` pipeline:

```python
from strands_evals import Case, Experiment
from strands_evals.evaluators import CoherenceEvaluator, OutputEvaluator
from strands_evals.providers import CloudWatchProvider

provider = CloudWatchProvider(log_group="/aws/...", region="us-east-1")

def task(case: Case) -> dict:
    return provider.get_evaluation_data(case.input)

cases = [Case(name="session_1", input="my-session-id", expected_output="any")]
evaluators = [
    OutputEvaluator(rubric="Score 1.0 if the output is coherent. Score 0.0 otherwise."),
    CoherenceEvaluator(),
]

experiment = Experiment(cases=cases, evaluators=evaluators)
reports = experiment.run_evaluations(task)

for report in reports:
    print(f"{report.overall_score:.2f} - {report.reasons}")
```

## Error Handling

```python
from strands_evals.providers import SessionNotFoundError, ProviderError

try:
    data = provider.get_evaluation_data("unknown-session")
except SessionNotFoundError:
    print("No traces found for that session")
except ProviderError:
    print("Provider unreachable or query failed")
```

## Implementing a Custom Provider

Subclass `TraceProvider` and implement `get_evaluation_data`:

```python
from strands_evals.providers import TraceProvider

class MyProvider(TraceProvider):
    def get_evaluation_data(self, session_id: str) -> dict:
        # Fetch traces from your backend, return:
        # {"output": "final response text", "trajectory": Session(...)}
        ...
```

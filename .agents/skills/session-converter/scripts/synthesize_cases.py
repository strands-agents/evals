"""Append synthetic cases, written in the voice of an experiment's existing cases, to the experiment file."""

import argparse
import asyncio
import json
import random
import sys
import uuid

from strands_evals import Case, Experiment
from strands_evals.generators import ExperimentGenerator

# Replaces the default case template to match the original cases' voice.
VOICE_SYSTEM_PROMPT = """You generate test case inputs for evaluating an AI agent.

The reference test cases are real messages a user sent to the agent. Each new input should read like another \
message from that same user: the same register, casing, punctuation, shorthand, length, and kinds of requests. \
Borrow the user's wording, not their requests: ask about something the references don't, preferring the \
constructions the references use. Don't add explanations, qualifiers, or follow-up questions the \
references don't have. Ignore any requested difficulty level; match the difficulty of the reference cases."""

# Sample a subset of cases to avoid generalizing over many cases.
NUM_REFERENCES_PER_CALL = 10
# Cap concurrent calls to avoid model throttling.
NUM_CONCURRENT_CALLS = 10


async def _generate(references: list[Case], task_description: str, num_cases: int) -> list[Case]:
    # Don't generate expected output so that they don't dilute the voice.
    generator: ExperimentGenerator[str, str] = ExperimentGenerator(
        input_type=str, output_type=str, include_expected_output=False, case_system_prompt=VOICE_SYSTEM_PROMPT
    )
    generated: list[Case] = []
    for start in range(0, num_cases, NUM_CONCURRENT_CALLS):
        # Generate one case per call, so each call gets its own references and don't converge to the same exact output.
        results = await asyncio.gather(
            *(
                generator.from_experiment_async(
                    Experiment(cases=random.sample(references, min(NUM_REFERENCES_PER_CALL, len(references)))),
                    task_description=task_description,
                    num_cases=1,
                )
                for _ in range(min(NUM_CONCURRENT_CALLS, num_cases - start))
            )
        )
        generated += [case for result in results for case in result.cases]
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_file")
    parser.add_argument("--task-description", required=True)
    parser.add_argument("--num-cases", type=int, default=5)
    args = parser.parse_args()

    with open(args.experiment_file) as f:
        experiment = json.load(f)
    cases = experiment["cases"]
    # Exclude synthetic cases from references.
    references = [Case(input=c["input"]) for c in cases if not (c.get("metadata") or {}).get("synthetic")]
    if not references:
        sys.exit(f"error: {args.experiment_file} has no non-synthetic cases to use as references")
    generated = asyncio.run(_generate(references, args.task_description, args.num_cases))
    if not generated:
        sys.exit(f"error: generated 0 of {args.num_cases} cases, see the errors above")
    if len(generated) < args.num_cases:
        print(f"warning: generated {len(generated)} of {args.num_cases} cases, see the errors above", file=sys.stderr)

    # Placeholder names; the skill renames them after appending.
    for case in generated:
        cases.append({"name": str(uuid.uuid4()), "input": case.input, "metadata": {"synthetic": True}})

    with open(args.experiment_file, "w") as f:
        json.dump(experiment, f, indent=2, ensure_ascii=False)
    print(f"appended {len(generated)} cases")


if __name__ == "__main__":
    main()

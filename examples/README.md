# Examples

Sample data for trying the UI without running anything first:

```bash
strands-evals ui examples/experiments --runs-dir examples/optimization/runs
```

## experiments/

Four experiments generated with the strands-evals CLI, plus two reports:

- `refund_desk.json` — 12 cases for a refund support agent. `refund_desk.report.json` is a real report (5 of 12 pass, overall 0.47) from running a harness agent with deliberate policy gaps against it.
- `refund_desk_adversarial.json` — 8 adversarial cases with seeded metadata (`status: draft`, `synthetic: true`, `source`) and three evaluators, for trying the metadata editor and the draft review mode.
- `telecom.json` — 12 cases for a telecom troubleshooting agent. `telecom.report.json` is a hand-built sample report, not from a real run.
- `code_migration.json` — 10 cases for a code migration agent.

## optimization/

Two sample optimization run records (run.json schema_version 1) for the Optimization tab, written to cover every event and outcome shape: a single-objective run and a Pareto multi-objective run, including a kill plus resume, a rescore, and a stop on proposer error. `optimization/README.md` describes them.

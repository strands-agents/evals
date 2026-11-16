# Examples

Sample experiments for trying the UI without writing one first:

```bash
strands-evals ui examples
```

They are also input to the rest of the CLI, which is the point of keeping them as files rather
than as fixtures — `strands-evals run experiments/refund_desk.json --task your_module:your_task`
writes the report the UI then reads.

## experiments/

Four experiments generated with the strands-evals CLI, plus two reports:

- `refund_desk.json` — 12 cases for a refund support agent. `refund_desk.report.json` is a real report (5 of 12 pass, overall 0.47) from running a harness agent with deliberate policy gaps against it.
- `refund_desk_adversarial.json` — 8 adversarial cases with seeded metadata (`status: draft`, `synthetic: true`, `source`) and three evaluators, for trying the metadata editor and the draft review mode.
- `telecom.json` — 12 cases for a telecom troubleshooting agent. `telecom.report.json` is a hand-built sample report, not from a real run.
- `code_migration.json` — 10 cases for a code migration agent.

## No sample optimizer run

There is none here on purpose. A run directory is a few hundred files — a recorded conversation
per trial, a copy of the agent's files per candidate — and checked in, the sample is most of what
anyone reviewing this directory reads. The Optimization section wants a real one instead:

```bash
python tests/ui/fixtures/make_optimizer_run.py   # writes tests/ui/fixtures/optimizer-run
strands-evals ui tests/ui/fixtures
```

That generator is the test fixture, deterministic and also not checked in. Point `ui` at any
directory holding runs the optimizer wrote and it will find them the same way.

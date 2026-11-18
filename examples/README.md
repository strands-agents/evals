# Examples

Sample experiments for trying the UI without writing one first:

```bash
strands-evals ui examples
```

They are also input to the rest of the CLI.

## experiments/

Four experiments generated with the strands-evals CLI:

- `refund_desk.json`: 12 cases for a refund support agent.
- `refund_desk_adversarial.json`: 8 adversarial cases with seeded metadata (`status: draft`, `synthetic: true`, `source`) and three evaluators, for trying the metadata editor and the draft review mode.
- `telecom.json`: 12 cases for a telecom troubleshooting agent.
- `code_migration.json`: 10 cases for a code migration agent.

## Reports

No sample report is checked in. Run an experiment against an agent of your own to produce one:

```bash
strands-evals run experiments/refund_desk.json --task your_module:your_task \
  -o experiments/refund_desk.report.json
```

`ui` pairs it with the experiment by filename, so the Report panel picks it up on the next scan.

## Optimizer runs

No sample optimizer run is checked in either, since a run directory is a few hundred files. The
test fixture generator writes a synthetic one:

```bash
python tests/ui/fixtures/make_optimizer_run.py   # writes tests/ui/fixtures/optimizer-run
strands-evals ui tests/ui/fixtures
```

Point `ui` at any directory holding runs the optimizer wrote and it finds them the same way.

## A folder with something in every view

`make_ui_demo.py` writes a synthetic folder that fills every view of the page: a run with a report and
last week's report to compare it against, recorded sessions, cases still in draft, a report that no
longer matches its experiment, a report with no experiment, these example experiments unrun, two files
the scan sets aside, a finished optimizer run and one still in progress.

```bash
python examples/make_ui_demo.py /tmp/evals-demo
strands-evals ui /tmp/evals-demo
```

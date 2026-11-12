# Browser tests for `strands-evals ui`

The page served by `strands-evals ui` is a single HTML file at
`src/strands_evals/cli/assets/ui.html`. Its discovery, pairing and write logic lives in Python and
is covered by `tests/strands_evals/cli/test_ui.py`; what is left in the page is rendering, and
rendering is where it can be *confidently wrong* — claiming a trajectory is absent when it is
merely an unexpected shape, or counting a failure to judge as a skipped row. These suites run the
real file in jsdom and assert on what it puts on screen.

Not wired into CI, because CI has no Node toolchain. Run them by hand when the asset changes:

```bash
cd tests/ui
npm install          # jsdom only
npm test             # every suite
node standalone.mjs  # or one at a time
```

`server.mjs` starts the real `_ui_server` as a subprocess and forwards the page's `fetch` to it, so
it exercises the HTTP contract rather than a mock of it. It needs a Python that can import
`strands_evals`; override with `PYTHON=/path/to/venv/bin/python npm test`. Point the suites at a
different build of the page with `UI_HTML=path/to/ui.html`.

| Suite | Covers |
| --- | --- |
| `standalone.mjs` | the file opened with no server: bundled sample, folder picker, download-save, add/delete case, evaluator config editing, trajectories that are dicts or a `Session` |
| `fileorigin.mjs` | the file opened from disk over `file://`, which `standalone.mjs` does not cover — it boots at an http URL, so it never reaches the APIs an opaque origin restricts. `history.replaceState` refuses a URL it cannot verify against one, and that rejection used to abort rendering entirely |
| `extras.mjs` | the row cap, deep links, reports missing their optional arrays, a mismatched experiment↔report pair, runs with only one half |
| `optimization.mjs` | the Optimization tab, over `fixtures/optimizer-run`. The **result** first, since that is what the page opens on: the champion against the baseline on the held-out split, where every figure has a wrong neighbour in the log it could have come from instead (the training best, the validation objective, a mean over the cases the search chose on). The verdict is the *interval*, not the gap, so there are logs differing only in that — one clearing zero, one straddling it, one negative, one with no interval at all, and one whose interval is narrow enough that rounding to two decimals would flip the verdict. A `picked`-only run is labelled validation rather than held out, and reads `measured` where a test report says `means`. Then the **winner**: its lineage back to the baseline, only the settings that actually differ, a multi-line prompt diffed as a block, and what the copy button really hands over (the whole configuration, inherited settings included) — with the fallback saying so when the clipboard is unreachable. Then the derived series, which are the numbers that can be plausibly wrong — best-so-far counts only *feasible* candidates and steps rather than slopes, an unscored candidate is absent from the chart rather than plotted at zero, the budget curve is cumulative while the per-iteration cost columns are not, each metric is charted in its own units, the line joining a metric breaks at a candidate it has no value for, and the trials recorded after the last iteration are named rather than folded into it. Also the page's shape: result and winner above the fold with every chart folded into drawers that are closed but already drawn, and the metric and unit pickers redrawing what they promise. Plus a log still being appended to, one with a corrupt line, a candidate the optimizer never labelled, a champion whose values the log never recorded, and a directory missing either half of the pair |
| `server.mjs` | served mode end to end: discovery over HTTP, saving back to disk, the mtime conflict guard, evaluator edits round-tripping, token enforcement, an optimizer log fetched over `/api/events` |

`fixtures/optimizer-run` is synthetic, written by `fixtures/make_optimizer_run.py`. Nothing in the
library writes an `events.jsonl` yet — the optimizer's record layer still raises
`NotImplementedError` — so the page's reader is coded against the schema documented in
`optimization/record/events.py` and this fixture stands in for a real run until that lands.

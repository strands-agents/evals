# Browser tests for `strands-evals ui`

The page served by `strands-evals ui` lives in `src/strands_evals/cli/assets/`: `ui.html` for the
markup, `ui.css` for the theme, and one `ui-*.js` script per concern. There is no build step. The
server inlines the stylesheet and scripts into one document when it serves the page, and
`harness.mjs` does the same for these suites.

Discovery, pairing and write logic lives in Python and is covered by
`tests/strands_evals/cli/test_ui.py`. `ui-folder.js` mirrors the discovery for the no-server mode,
where the page reads a folder the user picks. These suites cover the page: they run the real files
in jsdom and assert on what they put on screen.

Not wired into CI, because CI has no Node toolchain. Run them by hand when the assets change:

```bash
cd tests/ui
npm install          # jsdom only
npm test             # writes the fixture, then every suite
node standalone.mjs  # or one at a time
npm run fixture      # what `npm test` does first, for running a suite on its own
```

`server.mjs` and `parity.mjs` start the real `_ui_server` as a subprocess. `server.mjs` forwards the
page's `fetch` to it, so it exercises the HTTP contract rather than a mock of it. Both need a Python
that can import `strands_evals`; override with `PYTHON=/path/to/venv/bin/python npm test`. Point the
suites at a different build of the page with `UI_HTML=path/to/ui.html`.

`harness.mjs` holds what the suites share: `page()` builds the page in jsdom and returns helpers to
drive it, `fakeServer()` is a `fetch` that answers as the server would, and `runner()` counts
assertions.

| Suite | Covers |
| --- | --- |
| `standalone.mjs` | The page opened with no server: the bundled sample and its report summary, evaluator chips and editing a setting, adding, saving (as a download) and deleting a case, the saves that are refused (no input, a name used twice), opening a folder, and trajectories that are recorded calls or a `Session`. |
| `fileorigin.mjs` | The page opened over `file://`, where the origin is opaque and `history.replaceState` rejects URLs: the sample still renders, and the link is still written and read back through the hash. |
| `extras.mjs` | The Evals section beyond the sample. Report table: the 300-row cap, a report with only its required arrays. Run list: pairing, half-present runs, the cells that jump to a tab, reselecting the open run. Addresses: the path that names a run, a view and a case, and what an unknown or malformed one falls back to. History: each move between views or runs is a step, Back and Forward retrace them across runs, a case pick or a link opened adds none, a typed address is followed, and Back never drops unsaved edits. Case editor: what deleting a case writes, evaluator settings as text. Metadata: value typing, renaming, adding and removing keys, untouched rows written back verbatim. The draft review pass and its shortcuts. Served mode against `fakeServer()`: a failed scan, a save in flight, a save answered 404, the token from the document. Keyboard activation, `aria-current`, the theme toggle, and muted-ink contrast in both themes. |
| `optimization.mjs` | The Optimization section over `fixtures/optimizer-run`: section visibility, the Runs list, Setup (both spellings of `run.json`), Overview (result card, interval verdict, interval picture, champion configuration, case by case, cost drawer), Progress (best-so-far, budget, axes and labels), Candidates (diff against baseline, why one cannot ship, trials, transcripts, opening a candidate as a report and its place in the address), Tunables (space table, effects drawer), Trade-off (axis direction, front), chart hover (tooltip, legend focus, click-through to a candidate), the experiment match, and malformed or partial logs. |
| `server.mjs` | Served mode end to end against the real server: discovery over HTTP, the skipped list (a run's own files kept out of it), the optimizer run summarised by the server, its log, transcripts and effects model fetched over the API, evaluator checks, saving to disk and the document written, the mtime conflict guard and its reload offer, evaluator edits round-tripping, what the server refuses (an invalid case, a missing token), and the link that names the open optimizer run, written and read. |
| `guide.mjs` | What the page explains and where it sends people: every documentation link tagged with its campaign and placement, the guide on each view and the switch that hides it (closed to begin with on a phone-sized window), the ? help on case fields, evaluators and their settings, and the tooltips on terms, column heads and chart marks. |
| `charts.mjs` | The Report's charts: the verdict grid and the row a cell opens, the score distribution per evaluator, this report against another (what flipped, what moved, what only one holds), cost per case read off recorded sessions, and a recorded session in a report row drawn as a trace. |
| `live.mjs` | Keeping up with the folder, against a `fakeServer()` whose tree the test changes: an unchanged folder costs one request, a new run appears, a re-run report is re-read with the filter and open rows kept, a clean editor follows the file, unsaved edits are never replaced, and a hidden tab does not poll. |
| `parity.mjs` | The server's discovery and the page's folder reader run over one tree and compared field by field: which files are experiments and reports, how they pair, the failing and not-applicable counts, what is skipped and why, which directories are left out, and the optimizer run summary. |

`fixtures/optimizer-run` is synthetic and not committed. `fixtures/make_optimizer_run.py` writes
it deterministically (84 files, 80 of them transcripts named by content hash), and `npm test` runs
that first.

The fixture follows the schema documented in `optimization/record/events.py`. The page has also
been run over a directory written by the optimizer on `workstream/optimizer-workflow`, which is
where the nested `split.cases.train` shape and the run's own `result.json` come from.

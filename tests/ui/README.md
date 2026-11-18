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

`server.mjs`, `parity.mjs` and `markdown.mjs` start the real `_ui_server` as a subprocess. `server.mjs`
and `markdown.mjs` forward the page's `fetch` to it, so they exercise the HTTP contract rather than a mock
of it. All three need a Python that can import `strands_evals`; override with `PYTHON=/path/to/venv/bin/python npm test`. Point the
suites at a different build of the page with `UI_HTML=path/to/ui.html`.

`harness.mjs` holds what the suites share: `page()` builds the page in jsdom and returns helpers to
drive it (`openReport()`, `openExperiment()` and `openSession()` pick a row from the lists), `fakeServer()` is a `fetch` that answers as the server would, and `runner()` counts
assertions.

| Suite | Covers |
| --- | --- |
| `standalone.mjs` | The page opened with no server: the bundled sample and its report summary, evaluator chips and editing a setting, adding, saving (as a download) and deleting a case, the saves that are refused (no input, a name used twice), opening a folder, and trajectories that are recorded calls or a `Session`. |
| `fileorigin.mjs` | The page opened over `file://`, where the origin is opaque and `history.replaceState` rejects URLs: the sample still renders, and the link is still written and read back through the hash. |
| `extras.mjs` | The Evals section beyond the sample. Report table: the 300-row cap, a report with only its required arrays. The two lists: what each report was made from and what each experiment has made, a report paired by its cases when no experiment shares its name, a tie left unpaired, a report made in code opening on its own, and the page header's links between a report and its experiment. Addresses: the paths that name a list, a report, an experiment and a case, and what an unknown or malformed one falls back to. No native dialogs: delete is undone rather than confirmed, a tool is typed in place, and opening another experiment over unsaved edits asks in the page (the harness fails a suite that opens `alert`, `confirm` or `prompt`). History: each move between pages is a step, Back and Forward retrace them across reports and experiments, a case pick or a link opened adds none, a typed address is followed, and Back never drops unsaved edits. Case editor: what deleting a case writes, evaluator settings as text. Metadata: value typing, renaming, adding and removing keys, untouched rows written back verbatim. The draft review pass and its shortcuts. Served mode against `fakeServer()`: a failed scan, a save in flight, a save answered 404, the token from the document. Keyboard activation, `aria-current`, the theme toggle, and muted-ink contrast in both themes. |
| `optimization.mjs` | The Optimization section over `fixtures/optimizer-run`: section visibility, the Runs list, the page header that names a run and holds its six views, Setup (both spellings of `run.json`), Overview (result card, interval verdict, interval picture, champion configuration, case by case, cost drawer), Progress (best-so-far, budget, axes and labels), Candidates (diff against baseline, why one cannot ship, trials, transcripts, opening a candidate as a report and its place in the address), Tunables (space table, effects drawer), Trade-off (axis direction, front), chart hover (tooltip, legend focus, click-through to a candidate), the experiment match, and malformed or partial logs. |
| `server.mjs` | Served mode end to end against the real server: discovery over HTTP, the skipped list (a run's own files kept out of it), the optimizer run summarised by the server, its log, transcripts and effects model fetched over the API, evaluator checks, saving to disk and the document written, the mtime conflict guard and its reload offer, evaluator edits round-tripping, what the server refuses (an invalid case, a missing token), the paths that name a report and a case of an experiment, and the link that names the open optimizer run, written and read. |
| `guide.mjs` | What the page explains and where it sends people: every documentation link tagged with its campaign and placement, the guide panel beside the content (closed until asked for, following the open view, remembered across visits), the ? help on case fields, evaluators and their settings, and the tooltips on terms, column heads and chart marks. |
| `charts.mjs` | The Report's charts: the verdict grid and the row a cell opens, the score distribution per evaluator, cost per case read off recorded sessions, and a recorded session in a report row drawn as a trace. |
| `live.mjs` | Keeping up with the folder, against a `fakeServer()` whose tree the test changes: an unchanged folder costs one request, a new experiment appears, a re-run report is re-read on its own with the filter and open rows kept, a clean editor follows the file, unsaved edits are never replaced, and a hidden tab does not poll. |
| `fidelity.mjs` | Nothing the page does not understand is misprinted or lost. A case's input and outputs of any shape in a report row and the editor: a dict or list as indented JSON, a list of role and content messages as a conversation, never `[object Object]`. A structured field edited as JSON, with JSON that does not parse marked and refused. Keys the editor has no field for, on the file, a case and an evaluator, written back by Save in the download and to the server. |
| `compare.mjs` | Reports side by side: ticking reports in the list and Compare with on a report's header, the order (when each run started, else the file's time), the summary and each evaluator's mean, the verdict grid with gaps where a report lacks a row, the oldest against the newest (what flipped, what moved, what only one holds), a mark opening its row, adding and removing reports, the address served and after `#`, Back, a re-run or deleted report under an open comparison, a report made in code, and reports sharing no rows. |
| `lenses.mjs` | Two lenses on a report. Group by: the metadata keys on offer, the summary per value (cases, rows, passing, failing, mean score), the table grouped and filtered to one group, list and dict values, the condition read from chaos case names, a report with no metadata, the choice kept across a live refresh. Diagnosis: what failed, why, where and what to do, spans linked to the trace of the row, the Diagnosed filter and marker, a row with none, and a malformed diagnosis shown as written. |
| `parity.mjs` | The server's discovery and the page's folder reader run over one tree and compared field by field: which files are experiments and reports, how they pair, the failing and not-applicable counts, what is skipped and why, which directories are left out, the optimizer run summary, sessions and their durations, which session a diagnosis goes to, and what a task result store holds. |
| `sessions.mjs` | Sessions as a third kind of thing in the Evals section: a session told by its shape and a look-alike that is not one, the sidebar item only when there is one, the list's figures, a session's page with each trace under its own heading and cost, a trace of several agent invocations read as turns, a diagnosis on its session's page or on its own, a task result store named and its files neither listed nor skipped, the addresses that reopen a session, and a new or rewritten session picked up while the page is open. |
| `markdown.mjs` | What the page hands a coding agent, against the real server. Copy as Markdown for a report, a report row, an experiment, a case, a candidate, the optimizer result and a trace, each pinned to its text; the failing rows as one prompt; Copy path (absolute when served, relative with no server) and Copy command (the real `run` flags, names quoted for a shell); that a copy is confirmed on its button and a failed one says so. The addresses `/reports/<file>/rows/<case>/<evaluator>` and `/optimization/<dir>/candidates/<id>`: written as a row or a candidate opens, and followed to that row or candidate. The Markdown the server serves at `/reports/<file>.md` and `/experiments/<file>.md` compared with the page's own, over a report chosen for what is awkward to write (numbers and key order as JavaScript prints them, backticks, pipes, several outputs). |

`fixtures/optimizer-run` is synthetic and not committed. `fixtures/make_optimizer_run.py` writes
it deterministically (84 files, 80 of them transcripts named by content hash), and `npm test` runs
that first.

The fixture follows the schema documented in `optimization/record/events.py`. The page has also
been run over a directory written by the optimizer on `workstream/optimizer-workflow`, which is
where the nested `split.cases.train` shape and the run's own `result.json` come from.

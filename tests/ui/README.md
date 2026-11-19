# Browser tests for `strands-evals ui`

The page served by `strands-evals ui` lives in `src/strands_evals/cli/assets/`: `ui.html` for the
markup, `ui.css` for the theme, and one `ui-*.js` script per concern. There is no build step. The
server inlines the stylesheet and scripts into one document when it serves the page, and
`harness.mjs` does the same for these suites.

Discovery, pairing, the Markdown and the write logic live in Python and are covered by
`tests/strands_evals/cli/test_ui.py` and `test_ui_markdown.py`. These suites cover the page: they run
the real files in jsdom, over the real server, and assert on what they put on screen.

CI runs them as the `UI Tests` job of `.github/workflows/test-lint.yml`. To run them by hand:

```bash
cd tests/ui
npm ci               # jsdom only, at the versions in package-lock.json
npm test             # writes the fixture, then every suite
node sample.mjs      # or one at a time
npm run fixture      # what `npm test` does first, for running a suite on its own
```

Most pages are built with `served()`, which writes a folder of files, starts the real `_ui_server` over it
and forwards the page's `fetch` there, so a suite exercises the real discovery and the HTTP contract rather
than a mock of them. One Python process (`serve.py`) serves every folder a suite asks for. It needs a Python
that can import `strands_evals`; override with `PYTHON=/path/to/venv/bin/python npm test`. Point the suites
at a different build of the page with `UI_HTML=path/to/ui.html`.

`harness.mjs` holds what the suites share. `served(files)` returns a page over a real server, with `saved()`
for the file its last Save wrote and `write()` to change the folder under it; `SAMPLE` is a folder of one
experiment and its report (`fixtures/sample/`). `page()` builds the page with whatever `fetch` it is given,
and `fakeServer()` is one that answers as the server would, for what the real one cannot be made to do on
demand: an answer held back, a scan that fails, a folder that changes at a chosen moment. Every page has
helpers to drive it (`openReport()`, `openExperiment()` and `openSession()` pick a row from the lists,
`view()` says which view is showing, `travel()` goes Back or Forward, `visit()` follows an address,
`poll()` asks whether the folder changed), and `runner()` counts assertions. `done()` fails a suite when
any page it built raised an error, stops the server and removes the folders.

| Suite | Covers |
| --- | --- |
| `sample.mjs` | The page from end to end over one experiment and its report: the two lists and the report's summary, evaluator chips with what the server made of each and editing a setting, adding, saving and deleting a case, the saves that are refused (no input, a name used twice), a folder with no report opening on its experiments, trajectories that are recorded calls or a `Session`, a server started with `--read-only`, and the page opened with no server behind it or from disk, which says how to start one. |
| `extras.mjs` | The Evals section beyond the sample. Report table: the 300-row cap, a report with only its required arrays. The two lists: what each report was made from and what each experiment has made, a report paired by its cases when no experiment shares its name, a tie left unpaired, a report made in code opening on its own, and the page header's links between a report and its experiment. Addresses: the paths that name a list, a report, an experiment and a case, and the case an experiment opens on when its address names none. No native dialogs: delete is undone rather than confirmed, a tool is typed in place, and opening another experiment over unsaved edits asks in the page (the harness fails a suite that opens `alert`, `confirm` or `prompt`). History: each move between pages is a step, Back and Forward retrace them across reports and experiments, a case pick or a link opened adds none, a typed address is followed, Back never drops unsaved edits, and Back pressed again before the step before it has loaded ends where it went. Case editor: what deleting a case writes, evaluator settings as text. Metadata: value typing, renaming, adding and removing keys, untouched rows written back verbatim. The draft review pass and its shortcuts. Against `fakeServer()`: a failed scan, a save in flight, a save answered 404 and the copy it offers to download, the token from the document. Keyboard activation, where focus goes when the in-page question opens and closes, `aria-current`, the theme toggle, and muted-ink contrast in both themes. |
| `optimization.mjs` | The Optimization section over `fixtures/optimizer-run`: section visibility, the Runs list, the page header that names a run and holds its six views, Setup (both spellings of `run.json`), Overview (result card, interval verdict, interval picture, champion configuration, case by case, cost drawer), Progress (best-so-far, budget, axes and labels), Candidates (diff against baseline, why one cannot ship, trials, transcripts, opening a candidate as a report and its place in the address), Tunables (space table, effects drawer), Trade-off (axis direction, front), chart hover (tooltip, legend focus, click-through to a candidate), the experiment match, and malformed or partial logs. |
| `server.mjs` | One folder holding everything, end to end against the real server: discovery over HTTP, the skipped list (a run's own files kept out of it), the optimizer run summarised by the server, its log, transcripts and effects model fetched over the API, evaluator checks, saving to disk and the document written, the mtime conflict guard and its reload offer, evaluator edits round-tripping, what the server refuses (an invalid case, a missing token), the paths that name a report and a case of an experiment, and the link that names the open optimizer run, written and read. |
| `guide.mjs` | What the page explains and where it sends people: every documentation link tagged with its campaign and placement, the guide panel beside the content (closed until asked for, following the open view, remembered across visits), the ? help on case fields, evaluators and their settings, and the tooltips on terms, column heads and chart marks. |
| `charts.mjs` | The Report's charts: the verdict grid and the row a cell opens, the score distribution per evaluator, cost per case read off recorded sessions, and a recorded session in a report row drawn as a trace. |
| `live.mjs` | Keeping up with the folder, against a `fakeServer()` whose tree the test changes: an unchanged folder costs one request, a new experiment appears with keyboard focus left on its row, a re-run report is re-read on its own with the filter and open rows kept, a clean editor follows the file, unsaved edits are never replaced, Rescan re-reads what is open as well as the lists, a file that arrives after the reader went elsewhere does not take the page, and a hidden tab does not poll. |
| `fidelity.mjs` | Nothing the page does not understand is misprinted or lost. A case's input and outputs of any shape in a report row and the editor: a dict or list as indented JSON, a list of role and content messages as a conversation, never `[object Object]`. A structured field edited as JSON, with JSON that does not parse marked and refused. Keys the editor has no field for, on the file, a case and an evaluator, written back by Save to the file. A case with no name kept as one case across its evaluators, and a row with no verdict counted as failing. |
| `compare.mjs` | Reports side by side: ticking reports in the list and Compare with on a report's header, the order (when each run started, else the file's time), the summary and each evaluator's mean, the verdict grid with gaps where a report lacks a row, the oldest against the newest (what flipped, what moved, what only one holds), a mark opening its row, adding and removing reports, the address, Back, a re-run or deleted report under an open comparison, a report made in code, and reports sharing no rows. |
| `lenses.mjs` | Two lenses on a report. Group by: the metadata keys on offer, the summary per value (cases, rows, passing, failing, mean score), the table grouped and filtered to one group, list and dict values, the condition read from chaos case names, a report with no metadata, the choice kept across a live refresh. Diagnosis: what failed, why, where and what to do, spans linked to the trace of the row, the Diagnosed filter and marker, a row with none, and a malformed diagnosis shown as written. |
| `sessions.mjs` | Sessions as a third kind of thing in the Evals section: a session told by its shape and a look-alike that is not one, the sidebar item only when there is one, the list's figures, a session's page with each trace under its own heading and cost, a trace of several agent invocations read as turns, a diagnosis on its session's page or on its own, a task result store named and its files neither listed nor skipped, the addresses that reopen a session, and a new or rewritten session picked up while the page is open. |
| `markdown.mjs` | What the page hands a coding agent, against the real server. Copy as Markdown for a report, a report row, a comparison, an experiment, a case, a session, a lone diagnosis, a candidate, the optimizer result and a trace, each pinned to its text; the failing rows as one prompt; Copy path (absolute) and Copy command (the real `run` flags, names quoted for a shell); that a copy is confirmed on its button, that one the server refuses or the clipboard does not take says so, and that one answered after a later copy is dropped. The addresses `/reports/<file>/rows/<case>/<evaluator>` and `/optimization/<dir>/candidates/<id>`: written as a row or a candidate opens, and followed to that row or candidate, a row's also while a lens groups the table and shows one group of it. What is copied for a file is what `/reports/<file>.md`, `/experiments/<file>.md` and `/sessions/<file>.md` serve, over a report and a session chosen for what is awkward to write (numbers and key order a browser would rewrite, backticks, pipes, several outputs, tool output in each shape the page unwraps); a candidate's cases and an experiment with unsaved edits are sent to the server to be written. |
| `editor.mjs` | The cases editor: an edit is written as it was made, or Save says why not. A field typed into and emptied again, text that starts with a line break, an evaluator setting that is not a number, the review pass kept to the experiment it was started in, a review key pressed while a save is on its way, the address following the selected case through a rename, a delete, Undo, Add and Revert, and a second save checked against the time the first left the file with. |
| `empty.mjs` | What the page says when there is nothing to show, each under the frog. Not found: an address the page does not have, a report, experiment, session, optimizer run, candidate, case or row that is not there, a file of another kind than the address says, and a comparison naming a missing report; the address stays as typed, the list it would be in stays marked, and the ways out are steps Back can undo. A file that will not load says the server's error at the file's own address, and is opened once a later scan can read it. Empty: the four lists, a report with no rows, an experiment with no cases, and an optimizer run with no result, candidates, search space, second goal or setup, or whose log cannot be read. |

`fixtures/optimizer-run` is synthetic and not committed. `fixtures/make_optimizer_run.py` writes
it deterministically (84 files, 80 of them transcripts named by content hash), and `npm test` runs
that first.

The fixture follows the schema documented in `optimization/record/events.py`. The page has also
been run over a directory written by the optimizer on `workstream/optimizer-workflow`, which is
where the nested `split.cases.train` shape and the run's own `result.json` come from.

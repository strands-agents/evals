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

`server.mjs` starts the real `_ui_server` as a subprocess and forwards the page's `fetch` to it, so
it exercises the HTTP contract rather than a mock of it. It needs a Python that can import
`strands_evals`; override with `PYTHON=/path/to/venv/bin/python npm test`. Point the suites at a
different build of the page with `UI_HTML=path/to/ui.html`.

| Suite | Covers |
| --- | --- |
| `standalone.mjs` | The file opened with no server: bundled sample, folder picker, download-save, adding and deleting cases, evaluator config editing, trajectories that are dicts or a `Session`. |
| `fileorigin.mjs` | The file opened over `file://`, where the origin is opaque and `history.replaceState` rejects URLs. Rendering must survive that rejection. |
| `extras.mjs` | Evals edge cases: the row cap, deep links (including legacy tab names and links the page cannot follow), reports missing optional arrays, mismatched or half-present experiment/report pairs, the run list's jump cells, label contrast, the metadata editor (value typing, key rename, blank rows), what delete and metadata edits write to the file, evaluator setting names as text, a failed scan in served mode, and the draft review pass with its keyboard shortcuts. |
| `optimization.mjs` | The Optimization section over `fixtures/optimizer-run`: section visibility, the Runs list, Setup (both spellings of `run.json`), Overview (result card, interval verdict, interval picture, champion configuration, cost drawer), Progress (best-so-far, budget, axes and labels), Candidates (diff against baseline, trials, transcripts, opening a candidate as a report), Tunables (space table, effects drawer), Trade-off (axis direction, front), chart hover (tooltip, legend focus, click-through to a candidate), the experiment match, and malformed or partial logs. |
| `server.mjs` | Served mode end to end: discovery over HTTP, saving to disk, the mtime conflict guard, evaluator edits round-tripping, token enforcement, a link that names an optimizer run, the optimizer log, transcripts and effects model fetched over the API, a run's own files kept out of the skipped list, and the runs-list result computed by the server. |

`fixtures/optimizer-run` is synthetic and not committed. `fixtures/make_optimizer_run.py` writes
it deterministically (84 files, 80 of them transcripts named by content hash), and `npm test` runs
that first.

The fixture follows the schema documented in `optimization/record/events.py`. The page has also
been run over a directory written by the optimizer on `workstream/optimizer-workflow`, which is
where the nested `split.cases.train` shape and the run's own `result.json` come from.

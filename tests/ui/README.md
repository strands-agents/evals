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
npm test             # all four suites
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
| `server.mjs` | served mode end to end: discovery over HTTP, saving back to disk, the mtime conflict guard, evaluator edits round-tripping, token enforcement |

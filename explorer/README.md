# Run explorer

A browser app for the run records written by `strands_evals.optimization`: a run's `events.jsonl` (schema 1), its
`run.json`, its `transcripts/`, and its `effects.json` when one has been fitted. It shows the overview, progress,
trade-off, search space, every candidate and each of its trials, and the setup.

This is the deep view. `strands-evals ui` has an Optimization tab of its own over the same records — method, result,
winner, candidates — which is the one to reach for when the question is "what came out of this run". The explorer is
where the whole search is, candidate by candidate and trial by trial.

Records are read in the browser; nothing is uploaded.

## Open it

Served beside the main page, which is how it is normally reached:

    strands-evals ui RUNS_PARENT --runs-dir RUNS_DIR

`--runs-dir` is a directory of run folders, each holding an `events.jsonl`. The CLI prints a `/optimization/?t=<token>`
link next to the usual one, and the top bar of the main page gains an **Explorer** link to the same place. The app reads
the records over `/api/optimization/runs` and `/api/optimization/file`, under the same token and the same
path-traversal checks as everything else the server reads.

That link serves the bundle committed at `src/strands_evals/cli/assets/optimization/`. Rebuild it from this source
after changing anything here, or the served app is still the old one:

    cd explorer
    npm install
    npm run build:cli   # typecheck, then build into src/strands_evals/cli/assets/optimization/

The output is deterministic, so an unchanged source rebuilds to the same bytes and `git status` stays clean.

## Develop

    npm run dev         # http://localhost:5173
    npm run build       # typecheck, then a static build in dist/ (npm run preview serves it)
    npm run typecheck   # tsc --noEmit on its own

Without a server to read from, **Open folder…** picks a directory of runs (runs up to three levels down). Chrome and
Edge remember the folder and re-read runs that are still being written; other browsers select or drop the folder,
which loads a snapshot.

The dev server can also read a folder inside the repository directly, which is what the headless checks use:
`http://localhost:5173/?devroot=<absolute path of the folder>&devstudies=<run>,<run>`.

The Overview's and Search space's effects need an `effects.json` next to the record:

    python -m strands_evals.optimization.effects RUNS_DIR/<run>

`examples/optimization/runs/` holds two runs to develop against.

## Embed runs in a page

For a page that can load nothing (an Artifactory page: one HTML file, no network), the embed build makes one script
that the page inlines, `dist-embed/explorer-embed.js` (React and the explorer's CSS included):

    npm run build:embed

It defines `StrandsExplorer`. `StrandsExplorer.mount(element, run, { initialPage })` shows one run in `element`
(give it a height: the explorer scrolls inside it), with the usual header and tabs, in a shadow root: the page's CSS
does not reach it and its CSS does not leave it. Its route is kept in memory, so any number of runs on a page each
navigate on their own and the page's URL never changes. `mount` returns `{ navigate(page), unmount() }`, where a page
is a route without the run: `{ page: "candidate", n: 3 }`, `{ page: "trial", n: 3, case: "airline-1", rep: 1 }`,
`{ page: "progress" }`. `initialPage` is one of the tabs: `overview` (the default), `progress`, `tradeoff`, `space`,
`candidates`, `setup`.

A run is `{ id, record | recordGz, effects?, transcripts? | transcriptsGz? }`: the `events.jsonl`, the parsed
`effects.json` (or null), and the transcripts by hash, as text or gzipped and base64-encoded (a transcript is inflated
when its trial is opened). `StrandsExplorer.inflate(base64Gzip)` inflates one such text. To write runs in this form,
as JSON or as a script that assigns them to a global:

    node scripts/embed-data.mjs [-o FILE] [--var NAME] RUNS_DIR/<run> RUNS_DIR/<run>...

## Not yet migrated

There are no automated tests here. `npm run typecheck` is the only gate, and the app is otherwise checked by hand
against `examples/optimization/runs/`. The Python side of the serving — the routes, the token, the traversal checks —
is covered by `tests/strands_evals/cli/test_ui.py`.

Two helpers still speak the pre-migration `study.jsonl` record and its `{type: "candidate", trials: [...]}` events, so
they do not run against an `events.jsonl` run:

- `fixtures/generate.py`, which wrote fixture records through the optimization framework.
- `scripts/embed-test-page.mjs`, which builds `dist-embed/test.html` from three named studies.

`scripts/embed-data.mjs` reads `events.jsonl` and does work.

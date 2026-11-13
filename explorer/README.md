# Study explorer

A browser app for study records written by `strands_evals.optimization`: a study's `study.jsonl` (format
`strands_evals.optimization`, schema 1), its `transcripts/`, and its `effects.json` when one has been fitted. It shows
the overview, progress, trade-off, search space, every candidate and each of its trials, and the setup. The files are
read in the browser; nothing is uploaded.

## Run

    cd explorer
    npm install
    npm run dev      # http://localhost:5173
    npm run build    # typecheck, then a static build in dist/ (npm run preview serves it)

## Open a folder of studies

A study is a directory holding a `study.jsonl`; a folder of studies holds studies up to three levels down (for example
`studies/`). Click **Open folder…** and pick it. Chrome and Edge remember the folder and re-read studies that are still
running; other browsers select or drop the folder, which loads a snapshot.

The Overview's and Search space's effects need an `effects.json` next to the record:

    python -m strands_evals.optimization.effects studies/<study>

The dev server can also read a folder inside the repository directly (used for headless checks):
`http://localhost:5173/?devroot=<absolute path of the folder>&devstudies=<study>,<study>`.

Fixture records for testing the explorer: `.venv/bin/python explorer/fixtures/generate.py studies`.

## Embed studies in a page

For a page that can load nothing (an Artifactory page: one HTML file, no network), the embed build makes one script
that the page inlines, `dist-embed/explorer-embed.js` (React and the explorer's CSS included):

    npm run build:embed

It defines `StrandsExplorer`. `StrandsExplorer.mount(element, study, { initialPage })` shows one study in `element`
(give it a height: the explorer scrolls inside it), with the usual header and tabs, in a shadow root: the page's CSS does
not reach it and its CSS does not leave it. Its route is kept in memory, so any number of studies on a page each
navigate on their own and the page's URL never changes. `mount` returns `{ navigate(page), unmount() }`, where a page is
a route without the study: `{ page: "candidate", n: 3 }`, `{ page: "trial", n: 3, case: "airline-1", rep: 1 }`,
`{ page: "progress" }`. `initialPage` is one of the tabs: `overview` (the default), `progress`, `tradeoff`, `space`,
`candidates`, `setup`.

A study is `{ id, record | recordGz, effects?, transcripts? | transcriptsGz? }`: `study.jsonl`, the parsed
`effects.json` (or null), and the transcripts by hash, as text or gzipped and base64-encoded (a transcript is inflated
when its trial is opened). `StrandsExplorer.inflate(base64Gzip)` inflates one such text. To write studies in this
form, as JSON or as a script that assigns them to a global:

    node scripts/embed-data.mjs [-o FILE] [--var NAME] studies/<study> studies/<study>...

`node scripts/embed-test-page.mjs` writes `dist-embed/test.html`: three example studies in five panels of one page.

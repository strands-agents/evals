#!/usr/bin/env node
// dist-embed/test.html: the embed bundle and three example studies inlined in one page that loads nothing, mounted in
// five fixed-height panels down a long page whose own CSS would restyle anything it reached. The page's script takes
// each panel somewhere (an initial page, navigate(), clicks on the explorer's own links, a hover) and writes what it
// checked at the top of the page.
//
//   npm run build:embed && node scripts/embed-test-page.mjs [STUDIES_DIR]

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { embedStudies } from "./embed-data.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const studies = process.argv[2] ?? join(here, "../../studies");
const SIM = "sim-6-full-20261014-235442";
const READER = "document-reader-4-prompt_window-20261014-235439";
const TAU = "tau2-airline-full-20261014-234841";

const bundle = readFileSync(join(here, "../dist-embed/explorer-embed.js"), "utf8");
const data = embedStudies([SIM, READER, TAU].map((s) => join(studies, s)), "STUDIES");

// a candidate of the real study with a transcript, for the candidate and trial pages
const candidates = readFileSync(join(studies, TAU, "study.jsonl"), "utf8")
  .split("\n")
  .filter(Boolean)
  .map((l) => JSON.parse(l))
  .filter((e) => e.type === "candidate" && e.role !== "baseline");
const t = candidates.find((e) => e.trials.some((a) => a.transcript));
const a = t.trials.find((x) => x.transcript);
const trial = { page: "trial", n: t.number, case: a.case, rep: a.repetition };

const test = `
const checks = [];
const errors = [];
window.addEventListener("error", (e) => errors.push(String(e.message)));
window.addEventListener("unhandledrejection", (e) => errors.push(String(e.reason)));
const check = (ok, what) => checks.push((ok ? "ok    " : "FAIL  ") + what);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(f) {
  for (let i = 0; i < 200; i++) {
    const v = f();
    if (v) return v;
    await sleep(50);
  }
  throw new Error("timed out: " + f);
}
const $ = (id) => document.getElementById(id);
const scroller = (id) => $(id).shadowRoot.querySelector(".embed");
const q = (id, sel) => $(id).shadowRoot.querySelector(sel);
const shown = (id) => {
  const tab = q(id, ".tabs a.on")?.textContent;
  const head = q(id, "main h2")?.textContent ?? "";
  return tab + (head ? " · " + head.replace(/\\s+/g, " ").trim().slice(0, 60) : "");
};
const sheets = document.styleSheets.length;
const head = document.head.children.length;

const one = StrandsExplorer.mount($("p1"), STUDIES[${JSON.stringify(SIM)}], { initialPage: "progress" });
const two = StrandsExplorer.mount($("p2"), STUDIES[${JSON.stringify(READER)}], { initialPage: "candidates" });
const three = StrandsExplorer.mount($("p3"), STUDIES[${JSON.stringify(TAU)}]);
const four = StrandsExplorer.mount($("p4"), STUDIES[${JSON.stringify(TAU)}], { initialPage: "candidates" });
const five = StrandsExplorer.mount($("p5"), STUDIES[${JSON.stringify(TAU)}], { initialPage: "setup" });

async function p1() {
  const hit = await until(() => q("p1", ".panel svg rect[fill=transparent]:nth-of-type(4)"));
  const b = hit.getBoundingClientRect();
  hit.dispatchEvent(new MouseEvent("mousemove", { bubbles: true, clientX: b.left + b.width / 2, clientY: b.top + 40 }));
  const tip = await until(() => q("p1", ".tooltip"));
  const r = tip.getBoundingClientRect();
  check(Math.abs(r.left - (b.left + b.width / 2 + 16)) < 2 && Math.abs(r.top - (b.top + 56)) < 2, "p1 tooltip next to the pointer: " + tip.querySelector(".tip-head")?.textContent);
}
async function p3() {
  await until(() => q("p3", ".tabs a.on"));
  three.navigate({ page: "candidate", n: ${t.number} });
  await until(() => q("p3", ".candidate-head"));
}
async function p4() {
  const link = await until(() => q("p4", ".tr-table tbody tr:nth-child(3) a"));
  link.click();
  await until(() => q("p4", ".candidate-head"));
  scroller("p4").scrollTop = 400;
  const before = scroller("p4").scrollTop;
  (await until(() => q("p4", "a[href*='/trial/']"))).click();
  await until(() => q("p4", ".trial"));
  check(before > 0 && scroller("p4").scrollTop === 0, "p4 a new page scrolls the panel to its top (from " + before + ")");
  const title = q("p4", "main h2").textContent;
  q("p4", ".candidate-nav a:last-child").click();
  await until(() => q("p4", "main h2").textContent !== title);
}
async function p5() {
  await until(() => q("p5", ".tabs a.on"));
  five.navigate(${JSON.stringify(trial)});
  const msgs = await until(() => q("p5", ".at-transcript .msg") && $("p5").shadowRoot.querySelectorAll(".at-transcript .msg"));
  check(msgs.length > 0, "p5 transcript inflated from the page: " + msgs.length + " messages");
  const card = q("p5", ".at-transcript");
  scroller("p5").scrollTop += card.getBoundingClientRect().top - scroller("p5").getBoundingClientRect().top;
}

Promise.all([p1(), until(() => q("p2", ".tr-table tbody tr")), p3(), p4(), p5()])
  .catch((e) => errors.push(String(e)))
  .then(() => {
    for (const id of ["p1", "p2", "p3", "p4", "p5"]) {
      checks.push("      " + id + " shows " + shown(id));
      const failed = q(id, "#render-error");
      if (failed) errors.push(id + " view failed: " + failed.textContent.split("\\n").slice(0, 4).join(" / "));
    }
    check(location.hash === "", "the window's hash is untouched: " + JSON.stringify(location.hash));
    check(window.scrollY === 0, "the page itself did not scroll");
    check(document.styleSheets.length === sheets && document.head.children.length === head, "no styles added to the page");
    check(getComputedStyle(q("p2", "h1")).fontFamily.startsWith("system-ui"), "the page's CSS does not reach the explorer");
    check(errors.length === 0, "no errors" + (errors.length ? ": " + errors.join(" | ") : ""));
    $("checks").textContent = checks.join("\\n");
  });
`;

const panel = (id, what) => `<h2>${what}</h2>\n<div class="panel" id="${id}"></div>`;
const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Embedded study explorer: test page</title>
<style>
  /* a host page whose styles would restyle the explorer if they reached it */
  body { margin: 0 auto; max-width: 1340px; padding: 0 24px 80px; font: 17px/2 Georgia, serif; color: #3b1d0e; background: #f1ead8; text-align: center; letter-spacing: 0.04em; }
  a { color: crimson; text-decoration: underline wavy; }
  h1, h2 { font-weight: 400; } h2 { font-size: 18px; margin: 28px 0 6px; }
  div { word-spacing: 6px; } table, td, th { border: 2px solid crimson !important; }
  svg text { fill: crimson; } button, select { font: 20px Georgia, serif; }
  pre { text-align: left; font: 12.5px/1.5 Menlo, monospace; background: #fff8; padding: 10px 14px; letter-spacing: 0; }
  .panel { height: min(80dvh, 900px); border: 1px solid #b9ae94; border-radius: 6px; overflow: hidden; text-align: left; }
</style>
</head>
<body>
<h1>Embedded study explorer: five independent panels</h1>
<pre id="checks">running…</pre>
${panel("p1", `${SIM} · initialPage progress · a hover`)}
${panel("p2", `${READER} · initialPage candidates`)}
${panel("p3", `${TAU} · navigate() to candidate ${t.number}`)}
${panel("p4", `${TAU} · candidates, then clicks: a candidate's link in the table, a trial, the next trial`)}
${panel("p5", `${TAU} · navigate() to trial ${a.case} · ${a.repetition} of candidate ${t.number}, scrolled to the transcript`)}
<script>${bundle}</script>
<script>${data}</script>
<script>${test}</script>
</body>
</html>
`;
writeFileSync(join(here, "../dist-embed/test.html"), html);
console.error(`dist-embed/test.html: ${(Buffer.byteLength(html) / 1024).toFixed(0)} KB`);

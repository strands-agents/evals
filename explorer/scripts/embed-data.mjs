#!/usr/bin/env node
// Study directories as the embedded explorer takes them (EmbeddedStudy, src/source.ts), for a page that inlines its
// studies: study.jsonl and each transcript gzipped and base64-encoded, effects.json parsed (null without one).
//
//   node scripts/embed-data.mjs [-o FILE] [--var NAME] STUDY_DIR...
//
// Writes one JSON object, the studies by id (a study's id is its directory's name), to FILE or stdout; with --var, a
// script that assigns it to the global NAME instead. Either can go inside a <script> element as it is. The sizes go to
// stderr.

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { gzipSync } from "node:zlib";

const gz = (bytes) => gzipSync(bytes, { level: 9 }).toString("base64");

/** One study directory as an EmbeddedStudy, and its sizes (bytes on disk, and as embedded). */
export function embedStudy(dir) {
  const record = readFileSync(join(dir, "study.jsonl"));
  const effects = join(dir, "effects.json");
  const transcripts = join(dir, "transcripts");
  const study = {
    id: basename(resolve(dir)),
    recordGz: gz(record),
    effects: existsSync(effects) ? JSON.parse(readFileSync(effects, "utf8")) : null,
    transcriptsGz: {},
  };
  const size = { record: record.length, recordGz: study.recordGz.length, transcripts: 0, transcriptsGz: 0, count: 0 };
  if (existsSync(transcripts))
    for (const file of readdirSync(transcripts).sort()) {
      if (!file.endsWith(".json")) continue;
      const text = readFileSync(join(transcripts, file));
      const packed = gz(text);
      study.transcriptsGz[file.slice(0, -".json".length)] = packed;
      size.transcripts += text.length;
      size.transcriptsGz += packed.length;
      size.count++;
    }
  return { study, size };
}

/** JSON that is also safe inside a <script> element. */
export function scriptJson(value) {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

/** The studies by id, as JSON or (with ``name``) a script that assigns them to a global; the sizes go to stderr. */
export function embedStudies(dirs, name) {
  const studies = {};
  for (const dir of dirs) {
    const { study, size } = embedStudy(dir);
    if (study.id in studies) throw new Error(`two studies named ${study.id}`);
    studies[study.id] = study;
    const kb = (n) => `${(n / 1024).toFixed(0)} KB`;
    console.error(`${study.id}: record ${kb(size.record)} -> ${kb(size.recordGz)}, ${size.count} transcripts ${kb(size.transcripts)} -> ${kb(size.transcriptsGz)}`);
  }
  const json = scriptJson(studies);
  return name ? `var ${name} = ${json};\n` : `${json}\n`;
}

function main(args) {
  let out = null;
  let name = null;
  const dirs = [];
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "-o" || args[i] === "--out") out = args[++i];
    else if (args[i] === "--var") name = args[++i];
    else dirs.push(args[i]);
  }
  const bad = dirs.filter((d) => !existsSync(join(d, "study.jsonl")));
  if (!dirs.length || bad.length || out === undefined || name === undefined || (name && !/^[A-Za-z_$][\w$]*$/.test(name))) {
    console.error("usage: node scripts/embed-data.mjs [-o FILE] [--var NAME] STUDY_DIR...");
    for (const d of bad) console.error(`not a study (no study.jsonl): ${d}`);
    process.exit(2);
  }
  const text = embedStudies(dirs, name);
  if (out) writeFileSync(out, text);
  else process.stdout.write(text);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) main(process.argv.slice(2));

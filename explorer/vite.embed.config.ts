// The embed build (npm run build:embed): one self-contained script, dist-embed/explorer-embed.js, that defines the
// global StrandsExplorer (src/embed/main.tsx) and carries the explorer's CSS, made for a shadow root. No other files and
// no URLs, so a page that can load nothing (an Artifactory page) can inline it.

import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// the document's root and body, as type selectors (not classes such as .html-view)
const DOCUMENT = /(^|[\s,>+~])(?::root|html|body)(?![\w-])/g;

/** PostCSS: in a shadow root the document's root and body are the embed's root element, which is also the viewport the
 * views are sized to (100vh is its height, a width media query asks for its width). */
const shadowRoot = {
  postcssPlugin: "shadow-root",
  Rule(rule: { selector: string }) {
    rule.selector = rule.selector.replace(DOCUMENT, "$1.embed");
  },
  AtRule: {
    media(at: { name: string; params: string }) {
      if (/^\((min|max)-width:[^)]+\)$/.test(at.params)) at.name = "container";
    },
  },
  Declaration(decl: { value: string }) {
    decl.value = decl.value.replace(/(\d)[dls]?v([hw])\b/g, "$1cq$2");
  },
};

/** The bundle's CSS goes into the script, in place of src/embed/main.tsx's placeholder, instead of into a file. */
function cssInScript(): Plugin {
  return {
    name: "css-in-script",
    apply: "build",
    enforce: "post",
    generateBundle(_, bundle) {
      const files = Object.values(bundle);
      const css = files.flatMap((f) => (f.type === "asset" && f.fileName.endsWith(".css") ? [f] : []));
      const [js, ...more] = files.flatMap((f) => (f.type === "chunk" ? [f] : []));
      const placeholder = /(["'`])__EXPLORER_CSS__\1/g;
      if (!js || more.length || js.code.match(placeholder)?.length !== 1) throw new Error("css-in-script: expected one chunk with one CSS placeholder");
      js.code = js.code.replace(placeholder, () => JSON.stringify(css.map((f) => String(f.source)).join("")));
      if (/<\/script|<!--/i.test(js.code)) throw new Error("css-in-script: the script cannot be inlined in a <script> element");
      for (const f of css) delete bundle[f.fileName];
    },
  };
}

export default defineConfig({
  plugins: [react(), cssInScript()],
  css: { postcss: { plugins: [shadowRoot] } },
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  build: {
    outDir: "dist-embed",
    lib: { entry: "src/embed/main.tsx", name: "StrandsExplorer", formats: ["iife"], fileName: () => "explorer-embed.js" },
  },
});

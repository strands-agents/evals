// The explorer embedded in a page that loads nothing (an Artifactory page: one HTML file, no network). The embed build
// (npm run build:embed) makes this one script, dist-embed/explorer-embed.js, which defines StrandsExplorer:
// StrandsExplorer.mount(element, study) shows one study in the element, in a shadow root with the explorer's styles and
// its route in memory, so any number of studies on a page each navigate on their own.

import { createRoot } from "react-dom/client";
import { MemoryRouter, memoryRoute, type Page } from "../router";
import { fromEmbedded, type EmbeddedStudy } from "../source";
import "./embed.css";
import "../styles.css";
import { Boundary, StudyView } from "../StudyView";

export { inflate } from "../source";

// the bundle's CSS, made for a shadow root (vite.embed.config.ts puts it here)
const CSS = "__EXPLORER_CSS__";
let sheet: CSSStyleSheet | null = null;

export type Options = { initialPage?: Exclude<Page["page"], "candidate" | "trial"> };
export type Mounted = { navigate: (page: Page) => void; unmount: () => void };

/** Shows the study in ``element``, which the page gives a height (the explorer scrolls inside it). */
export function mount(element: HTMLElement, data: EmbeddedStudy, options: Options = {}): Mounted {
  if (data.record === undefined && data.recordGz === undefined) throw new TypeError(`study ${data.id}: no record or recordGz`);
  if (!sheet) {
    sheet = new CSSStyleSheet();
    sheet.replaceSync(CSS);
  }
  const shadow = element.shadowRoot ?? element.attachShadow({ mode: "open" });
  shadow.adoptedStyleSheets = [sheet];
  const root = document.createElement("div");
  root.className = "embed";
  shadow.replaceChildren(root);
  const memory = memoryRoute({ study: data.id, page: options.initialPage ?? "overview" });
  const studyRef = fromEmbedded(data);
  const react = createRoot(root);
  react.render(
    <Boundary route="app">
      <MemoryRouter memory={memory} root={root}>
        <StudyView studyRef={studyRef} />
      </MemoryRouter>
    </Boundary>,
  );
  return {
    navigate: (page) => memory.go(data.id, page),
    unmount: () => {
      react.unmount();
      root.remove();
    },
  };
}

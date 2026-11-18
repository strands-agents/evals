'use strict';

/* One tooltip for the whole page, delegated so re-rendered markup needs no rebinding. An element is
   explained by `data-term` (a glossary entry), `data-tip`, a `title`, or an SVG <title> child. */

const SVG_NS = 'http://www.w3.org/2000/svg';
const LEGEND_KINDS = ['feasible', 'infeasible', 'ineligible', 'champion'];

const tip = document.createElement('div');
tip.className = 'tip';
tip.setAttribute('role', 'tooltip');
tip.hidden = true;
app.appendChild(tip);

let tipFrom = null;
const isMark = el => el.namespaceURI === SVG_NS;

function tipTarget(node) {
  for (let el = node; el && el.nodeType === 1 && el !== app; el = el.parentNode) {
    if (isMark(el) ? el.localName !== 'svg' && (el.dataset.tip != null || el.querySelector(':scope > title'))
      : el.dataset.term || el.dataset.tip || el.title) return el;
  }
  return null;
}

/* `[heading, ...lines]`. A native title moves into `data-tip`, so the browser's own tooltip does not
   also show; it stays the accessible name of an element that has no text of its own. */
function tipLines(el) {
  if (el.dataset.term && TERMS[el.dataset.term]) return TERMS[el.dataset.term].slice(0, 2);
  const native = isMark(el) ? el.querySelector(':scope > title') : null;
  if (native || el.title) {
    el.dataset.tip = native ? native.textContent : el.title;
    if (native || !el.textContent.trim()) el.setAttribute('aria-label', el.dataset.tip);
    if (native) native.remove(); else el.removeAttribute('title');
  }
  return isMark(el) ? el.dataset.tip.split(/ · |: | — /) : ['', el.dataset.tip];
}

function showTip(el) {
  const [heading, ...lines] = tipLines(el);
  const parts = [...(heading ? [['b', heading]] : []), ...lines.filter(Boolean).map(line => ['div', line])];
  if (el.dataset.cand) parts.push(['div', 'Click to open in Candidates', 'hint']);
  if (!parts.length) return;
  tip.replaceChildren(...parts.map(([tag, text, cls]) => {
    const part = document.createElement(tag);
    part.textContent = text;
    if (cls) part.className = cls;
    return part;
  }));
  tip.hidden = false;
  tipFrom = el;
  if (isMark(el)) {
    el.classList.add('hot');
    el.ownerSVGElement.classList.add('hovering');
  }
}

function hideTip() {
  if (!tipFrom) return;
  tip.hidden = true;
  tipFrom.classList.remove('hot');
  if (tipFrom.ownerSVGElement) tipFrom.ownerSVGElement.classList.remove('hovering');
  tipFrom = null;
}

/* Beside the point given, kept inside the window. */
function placeTip(x, y) {
  const pad = 14, box = tip.getBoundingClientRect();
  const left = x + pad + box.width > innerWidth - 8 ? x - box.width - pad : x + pad;
  const top = y + pad + box.height > innerHeight - 8 ? y - box.height - pad : y + pad;
  tip.style.left = `${Math.max(8, left)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}
/* A chart mark's tooltip follows the pointer; anything else is explained just under itself. */
function anchorTip(el) {
  const box = el.getBoundingClientRect();
  placeTip(box.left - 14, box.bottom - 8);
}

app.addEventListener('mouseover', e => {
  const el = tipTarget(e.target);
  if (el !== tipFrom) {
    hideTip();
    if (el) {
      showTip(el);
      if (isMark(el)) placeTip(e.clientX, e.clientY); else anchorTip(el);
    }
  }

  /* Hovering a legend entry keeps its marks and fades the rest of that chart. */
  const entry = e.target.closest('.legend span');
  const kind = entry && LEGEND_KINDS.find(k => entry.querySelector(`i.${k}`));
  app.querySelectorAll('.chart[data-focus]').forEach(c => c.removeAttribute('data-focus'));
  if (kind) entry.closest('.chart').dataset.focus = kind;
});
app.addEventListener('mousemove', e => { if (tipFrom && isMark(tipFrom)) placeTip(e.clientX, e.clientY); });
app.addEventListener('mouseleave', hideTip);

/* What is reached by keyboard shows the same tooltip. */
app.addEventListener('focusin', e => {
  const el = tipTarget(e.target);
  hideTip();
  if (!el) return;
  showTip(el);
  anchorTip(el);
});
app.addEventListener('focusout', hideTip);
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideTip(); });
window.addEventListener('scroll', hideTip, { passive: true });

/* ── chart marks ── */

app.addEventListener('click', e => {
  const mark = e.target.closest('.chart [data-cand]');
  if (!mark) return;
  hideTip();
  openCandidate(mark.dataset.cand);
});

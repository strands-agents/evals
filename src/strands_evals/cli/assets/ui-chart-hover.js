'use strict';

/* Hover and click behaviour for every chart, delegated from the page so re-rendered SVG needs no
   rebinding. A mark is an SVG element with a <title> child; its text becomes the tooltip. */

const SVG_NS = 'http://www.w3.org/2000/svg';
const LEGEND_KINDS = ['feasible', 'infeasible', 'ineligible', 'champion'];

const chartTip = document.createElement('div');
chartTip.className = 'chart-tip';
chartTip.setAttribute('role', 'tooltip');
chartTip.hidden = true;
app.appendChild(chartTip);

let hotMark = null;

function markOf(target) {
  for (let el = target; el && el.namespaceURI === SVG_NS && el.localName !== 'svg'; el = el.parentNode) {
    if (el.dataset.tip != null || el.querySelector(':scope > title')) return el;
  }
  return null;
}

/* The <title> moves into `data-tip` on first hover, so the browser's own tooltip does not also show. */
function tipText(mark) {
  const title = mark.querySelector(':scope > title');
  if (title) {
    mark.dataset.tip = title.textContent;
    mark.setAttribute('aria-label', title.textContent);
    title.remove();
  }
  return mark.dataset.tip;
}

function showTip(mark) {
  const [head, ...rest] = tipText(mark).split(/ · |: | — /);
  const lines = [['b', head], ...rest.map(line => ['div', line])];
  if (mark.dataset.cand) lines.push(['div', 'Click to open in Candidates', 'hint']);
  chartTip.replaceChildren(...lines.map(([tag, text, cls]) => {
    const el = document.createElement(tag);
    el.textContent = text;
    if (cls) el.className = cls;
    return el;
  }));
  chartTip.hidden = false;
  mark.classList.add('hot');
  mark.ownerSVGElement.classList.add('hovering');
  hotMark = mark;
}

function hideTip() {
  if (!hotMark) return;
  chartTip.hidden = true;
  hotMark.classList.remove('hot');
  if (hotMark.ownerSVGElement) hotMark.ownerSVGElement.classList.remove('hovering');
  hotMark = null;
}

function moveTip(e) {
  const pad = 14, box = chartTip.getBoundingClientRect();
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + box.width > innerWidth - 8) x = e.clientX - box.width - pad;
  if (y + box.height > innerHeight - 8) y = e.clientY - box.height - pad;
  chartTip.style.left = `${Math.max(8, x)}px`;
  chartTip.style.top = `${Math.max(8, y)}px`;
}

app.addEventListener('mouseover', e => {
  const mark = markOf(e.target);
  if (mark !== hotMark) hideTip();
  if (mark && mark !== hotMark) { showTip(mark); moveTip(e); }

  /* Hovering a legend entry keeps its marks and fades the rest of that chart. */
  const entry = e.target.closest('.legend span');
  const kind = entry && LEGEND_KINDS.find(k => entry.querySelector(`i.${k}`));
  app.querySelectorAll('.chart[data-focus]').forEach(c => c.removeAttribute('data-focus'));
  if (kind) entry.closest('.chart').dataset.focus = kind;
});
app.addEventListener('mousemove', e => { if (hotMark) moveTip(e); });
app.addEventListener('mouseleave', hideTip);

/* A mark reached by keyboard shows the same tooltip, beside the mark. */
app.addEventListener('focusin', e => {
  const mark = markOf(e.target);
  if (!mark) return;
  hideTip();
  showTip(mark);
  const box = mark.getBoundingClientRect();
  moveTip({ clientX: box.right, clientY: box.bottom });
});
app.addEventListener('focusout', e => { if (markOf(e.target)) hideTip(); });

app.addEventListener('click', e => {
  const mark = e.target.closest('.chart [data-cand]');
  if (!mark) return;
  hideTip();
  openCandidate(mark.dataset.cand);
});

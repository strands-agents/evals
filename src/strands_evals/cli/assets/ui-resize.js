'use strict';

/* ── the sidebar's width ──
   The sidebar's edge drags, so long file names in its lists can be given room. Arrow keys move it
   too, and double-click puts the stylesheet back in charge. The chosen width is kept the way the
   theme is: in a cookie, since each run serves from a new port (see ui-theme.js). */

const NAV_KEY = 'strands-evals-nav';
const NAV_DEFAULT = 248, NAV_MIN = 180, NAV_MAX = 560;

{
  const edge = $('nav-resize');
  const kept = document.cookie.split('; ').find(pair => pair.startsWith(NAV_KEY + '='));
  let width = NAV_DEFAULT;
  const size = to => {
    width = Math.round(Math.min(NAV_MAX, Math.max(NAV_MIN, to)));
    app.style.setProperty('--nav-w', `${width}px`);
    edge.setAttribute('aria-valuenow', width);
  };
  if (kept && Number(kept.slice(NAV_KEY.length + 1))) size(Number(kept.slice(NAV_KEY.length + 1)));
  else edge.setAttribute('aria-valuenow', width);

  let from = null; /* [where the pointer went down, the width then] */
  edge.addEventListener('pointerdown', e => {
    e.preventDefault();
    from = [e.clientX, width];
    if (e.pointerId !== undefined && edge.setPointerCapture) edge.setPointerCapture(e.pointerId);
    edge.classList.add('held');
  });
  edge.addEventListener('pointermove', e => { if (from) size(from[1] + e.clientX - from[0]); });
  /* The content column is as wide as the sidebar leaves it, and charts are drawn to the width they have. */
  const drop = () => {
    if (!from) return;
    from = null;
    edge.classList.remove('held');
    remember(NAV_KEY, width);
    redrawCharts();
  };
  edge.addEventListener('pointerup', drop);
  edge.addEventListener('pointercancel', drop);
  edge.addEventListener('dblclick', () => {
    app.style.removeProperty('--nav-w');
    width = NAV_DEFAULT;
    edge.setAttribute('aria-valuenow', width);
    remember(NAV_KEY, '');
    redrawCharts();
  });
  edge.addEventListener('keydown', e => {
    const step = { ArrowLeft: -16, ArrowRight: 16 }[e.key];
    if (!step) return;
    e.preventDefault();
    size(width + step);
    remember(NAV_KEY, width);
    redrawCharts();
  });
}

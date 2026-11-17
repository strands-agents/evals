'use strict';

/* Loaded in <head> so the theme is set before first paint. A stored choice wins over the OS setting.
   The choice is kept in a cookie as well as in storage: every run serves from a new port, which is a
   new origin for storage but not for cookies. Opened from disk there are no cookies, only storage. */
const THEME_KEY = 'strands-evals-theme';

function storedTheme() {
  const cookie = document.cookie.split('; ').find(pair => pair.startsWith(THEME_KEY + '='));
  if (cookie) return cookie.slice(THEME_KEY.length + 1);
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

function storeTheme(theme) {
  document.cookie = `${THEME_KEY}=${theme}; path=/; max-age=31536000; SameSite=Lax`;
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* opaque origin: not persisted */ }
}

const preferred = storedTheme();
document.documentElement.dataset.theme = preferred === 'dark' || preferred === 'light' ? preferred
  : (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

document.addEventListener('click', e => {
  if (!e.target.closest('[data-theme-toggle]')) return;
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  storeTheme(next);
});

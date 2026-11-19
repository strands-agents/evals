'use strict';

/* Loaded in <head> so the theme is set before first paint. A stored choice wins over the OS setting.
   The choice is kept in a cookie: every run serves from a new port, which is a new origin for
   storage but not for cookies. */
const THEME_KEY = 'strands-evals-theme';

function storedTheme() {
  const cookie = document.cookie.split('; ').find(pair => pair.startsWith(THEME_KEY + '='));
  return cookie ? cookie.slice(THEME_KEY.length + 1) : null;
}

/* A choice kept for a year, on whichever port the page is served from next. */
function remember(key, value) {
  document.cookie = `${key}=${value}; path=/; max-age=31536000; SameSite=Lax`;
}

const preferred = storedTheme();
document.documentElement.dataset.theme = preferred === 'dark' || preferred === 'light' ? preferred
  : (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

document.addEventListener('click', e => {
  if (!e.target.closest('[data-theme-toggle]')) return;
  const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  remember(THEME_KEY, next);
});

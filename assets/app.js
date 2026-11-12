import {createFlipbook} from './flipbook.js';
import {BookRasters, openPdf} from './pdf-pages.js';
import {createReadMode} from './read-mode.js';

// Black background and white text; buttons and links are purple (Penny's palette).
const LAVENDER = '#B3A1D7';
const PLUM = '#2E1756';
const ACCENTS = {avatars: LAVENDER, harassment: LAVENDER, bias: LAVENDER};
// Ink for the label and link button on the typeset title page, a white sheet.
const INKS = {avatars: PLUM, harassment: PLUM, bias: PLUM};
const root = document.documentElement;
const params = new URLSearchParams(location.search);
const paperId = params.get('p');
// QA only: ?data=fixture swaps in tests/fixture-papers.js.
const museum = params.get('data') === 'fixture' ? await loadScript('tests/fixture-papers.js').then(() => window.MUSEUM, () => null) : window.MUSEUM;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = resolve;
    script.onerror = reject;
    document.head.append(script);
  });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function ratioOf(paper) {
  const [width, height] = paper.pageSize || [];
  return width > 0 && height > 0 ? width / height : 3 / 4;
}

// Views stay hidden until DM Sans is ready (at most 500 ms by default), so their
// text paints in it from the first frame instead of swapping fonts. Resolves
// false when the time runs out first.
function fontsReady(view, limit = 500) {
  const link = document.getElementById('font-css');
  const sheet = new Promise(resolve => {
    // The link is a preload until its onload turns it into a stylesheet.
    const check = () => { if (!link || link.sheet) resolve(); };
    check();
    link?.addEventListener('load', check);
    link?.addEventListener('error', resolve);
  });
  const faces = sheet.then(() => Promise.all(['400', '500'].map(weight => document.fonts.load(`${weight} 16px "DM Sans"`, view.textContent))));
  return Promise.race([faces.then(() => true, () => true), new Promise(resolve => setTimeout(resolve, limit, false))]);
}

// Typographic quotes and dashes, as on the FRAME panels. papers.js keeps the plain text.
function typo(text) {
  return String(text ?? '')
    .replace(/(^|[\s([{\u2013\u2014-])"/g, '$1\u201C')
    .replace(/"/g, '\u201D')
    .replace(/(^|[\s([{\u201C\u2013\u2014-])'/g, '$1\u2018')
    .replace(/'/g, '\u2019')
    .replace(/ - /g, ' \u2013 ');
}

if (!museum?.papers) {
  void showNotice('The library is unavailable', 'The list of papers could not be loaded. Please try again later.', false);
} else if (!paperId) {
  // The collection's homepage lives on the XR Women Research Library site.
  location.replace('https://abbyabbyzzz.github.io/xr-women-library/');
} else {
  const paper = museum.papers.find(item => item.id === paperId);
  if (paper) void showReader(paper);
  else void showNotice('Paper not found', 'This link doesn’t match a paper in the library.', true, 'Paper not found · Research Library');
}

async function showNotice(heading, text, withLink, title = 'Research Library') {
  document.title = title;
  const notice = document.getElementById('notice');
  notice.querySelector('.notice-title').textContent = heading;
  notice.querySelector('.notice-text').textContent = text;
  notice.querySelector('.notice-link').hidden = !withLink;
  root.classList.add('is-notice');
  await fontsReady(notice);
  notice.hidden = false;
}

// A PDF on another host gets its connection opened while PDF.js loads.
function preconnect(url) {
  let origin;
  try {
    origin = new URL(url, location.href).origin;
  } catch {
    return;
  }
  if (origin === location.origin) return;
  const link = document.createElement('link');
  link.rel = 'preconnect';
  link.href = origin;
  link.crossOrigin = 'anonymous';
  document.head.append(link);
}

async function showReader(paper) {
  if (!paper.pdf && paper.readAt?.url) return showTitlePage(paper);
  const reader = document.getElementById('reader');
  const $ = selector => reader.querySelector(selector);
  const stage = $('.stage');
  const book = $('.book');
  const readButton = $('[data-read]');

  // The paper starts downloading before anything waits.
  preconnect(paper.pdf);
  const pdf = openPdf(paper.pdf);
  pdf.catch(() => {});

  root.classList.add('is-reader');
  root.style.setProperty('--accent', ACCENTS[paper.theme] || ACCENTS.harassment);
  document.title = `${paper.shortTitle || paper.title} · Research Library`;
  $('.paper-title').textContent = paper.shortTitle || paper.title;
  $('.paper-meta').textContent = [paper.authorsShort, paper.year].filter(Boolean).join(' · ');
  fillLinks(paper, reader);
  wireBack(reader);

  const ratio = ratioOf(paper);
  const startPage = Number.parseInt(params.get('page'), 10);
  const rasters = new BookRasters();
  const flip = createFlipbook({
    reader,
    stage,
    book,
    previous: $('[data-previous]'),
    next: $('[data-next]'),
    counter: $('.counter'),
    rasters,
    ratio,
    pageCount: paper.pages,
    startIndex: startPage > 0 ? startPage - 1 : 0,
    coverSrc: `covers/${paper.id}.webp`,
    onChange: syncUrl,
  });
  const loadingTimer = setTimeout(() => flip.setStatus('Loading…'), 700);
  wireBookZoom(reader, flip);

  await fontsReady(reader);
  reader.hidden = false;
  flip.layout();
  new ResizeObserver(() => flip.layout()).observe(stage);

  // The click that closes read mode can be followed by a second one (a
  // double-click) on the link underneath, such as View all papers: ignore it.
  let guardUntil = 0;
  const guardActions = () => { guardUntil = performance.now() + 500; };
  $('.actions').addEventListener('click', event => {
    if (event.detail > 0 && performance.now() < guardUntil) {
      event.preventDefault();
      event.stopImmediatePropagation();
    }
  }, true);

  let readMode = null;
  pdf.then(source => {
    clearTimeout(loadingTimer);
    rasters.attach(source);
    readMode = createReadMode({
      dialog: $('.read-mode'),
      source,
      pageCount: source.numPages,
      fallbackRatio: ratio,
      standIn: index => rasters.standIn(index),
      onClose: index => {
        guardActions();
        flip.goTo(index);
      },
      onOpenChange: open => {
        rasters.setDeferred(open);
        if (!open) guardActions();
      },
    });
    rasters.setFallback(index => readMode.canvasFor(index));
    flip.enable(source.numPages);
    readButton.disabled = false;
    // Page sizes for read mode, fetched while the reader is idle.
    setTimeout(() => source.pageSizes().catch(() => {}), 1200);
  }, () => {
    clearTimeout(loadingTimer);
    flip.setStatus('');
    book.hidden = true;
    $('.pager').style.visibility = 'hidden';
    const error = $('.stage-error');
    const link = error.querySelector('a');
    if (paper.doi) link.href = 'https://doi.org/' + paper.doi;
    else link.hidden = true;
    error.hidden = false;
  });

  readButton.addEventListener('click', () => {
    if (readMode) void readMode.open(flip.visiblePages()[0]);
  });
}

// A paper that isn't openly licensed is read at its publisher. The stage shows
// its title page instead of the book; nothing of the paper itself is loaded.
async function showTitlePage(paper) {
  const reader = document.getElementById('reader');
  const $ = selector => reader.querySelector(selector);
  const stage = $('.stage');
  const frame = $('.tp-frame');
  const page = $('.title-page');

  root.classList.add('is-reader', 'is-publisher');
  root.style.setProperty('--accent', ACCENTS[paper.theme] || ACCENTS.harassment);
  document.title = `${typo(paper.shortTitle || paper.title)} · Research Library`;
  $('.paper-title').textContent = typo(paper.shortTitle || paper.title);
  $('.paper-meta').textContent = [paper.authorsShort, paper.year].filter(Boolean).join(' · ');
  // The page itself carries the details, so the book, its pager and page size go.
  for (const node of [$('.book'), $('.pager'), $('.zoom')]) node.hidden = true;
  fillLinks(paper, reader);
  wireBack(reader);
  fillTitlePage(paper, page);
  frame.hidden = false;

  // The stage scrolls, so the keyboard can reach it, and scroll keys work
  // before anything on the page has focus.
  stage.tabIndex = 0;
  const scrollKeys = new Set(['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End', ' ']);
  document.addEventListener('keydown', event => {
    if (document.activeElement === document.body && scrollKeys.has(event.key) && !event.altKey && !event.ctrlKey && !event.metaKey) stage.focus({preventScroll: true});
  });

  const fontsLoaded = await fontsReady(reader);
  reader.hidden = false;
  const fit = () => fitTitlePage(stage, frame, page, ratioOf(paper));
  fit();
  // The border box ignores the stage's own scrollbar coming and going.
  new ResizeObserver(fit).observe(stage, {box: 'border-box'});
  document.fonts.addEventListener('loadingdone', fit);
  if (!fontsLoaded && document.getElementById('font-css')?.sheet) {
    // DM Sans is still loading: the page waits for it (2 s at most), so it
    // appears once, already fitted. Without the font stylesheet it shows at once.
    frame.style.visibility = 'hidden';
    await fontsReady(page, 2000);
    fit();
    frame.style.visibility = '';
  }
}

function fillTitlePage(paper, page) {
  const $ = selector => page.querySelector(selector);
  page.style.setProperty('--theme-ink', INKS[paper.theme] || 'currentColor');
  $('.tp-theme').textContent = (museum.themes || []).find(theme => theme.id === paper.theme)?.name || '';
  $('.tp-title').textContent = typo(paper.title);
  $('.tp-authors').textContent = (paper.authors || []).join(', ');
  $('.tp-venue').textContent = venueLine(paper);
  $('.tp-abstract').replaceChildren(...(paper.abstract || []).map(text => element('p', '', typo(text))));
  $('.tp-link').href = paper.readAt.url;
  $('.tp-link-label').textContent = paper.readAt.label;
  $('.tp-license').textContent = paper.license?.name || '';
  for (const node of page.querySelectorAll(':scope > :not(a)')) node.hidden = !node.textContent.trim();
}

// The venue and the year, unless the venue already names it. The year and a
// page range ("pp. 326–329") never break apart.
function venueLine(paper) {
  const venue = typo(paper.venue || '').replace(/\bpp\. (\d+)\u2013(\d+)/g, 'pp.\u00a0$1\u2013\u2060$2');
  const year = paper.year ? String(paper.year) : '';
  if (!year || venue.split(/\D+/).includes(year)) return venue;
  return [venue, year].filter(Boolean).join('\u00a0·\u00a0');
}

// The page stands where the flipbook's cover would, in the paper's own
// proportions, and its type shrinks to fit, down to a 13px abstract. Past that
// the page grows and the stage scrolls. A stage too small for a page (a phone,
// FRAME's overlay) gets a plain column instead.
function fitTitlePage(stage, frame, page, ratio) {
  const minScale = 13 / 15;
  const maxScale = 1.5;
  // Page width at which the type has its base size; wider pages scale it up.
  const baseWidth = 560;
  // Whole pixels, rounded down, so a page that fits never adds a scrollbar.
  const box = stage.getBoundingClientRect();
  const width = Math.floor(box.width), height = Math.floor(box.height);
  if (!width || !height) return;
  frame.classList.remove('is-column');
  page.style.height = '';
  const fits = (pageWidth, scale) => {
    page.style.width = pageWidth + 'px';
    page.style.setProperty('--s', String(scale));
    return page.offsetHeight <= Math.round(pageWidth / ratio);
  };
  let pageWidth = 0, scale = minScale;
  if (width >= 600 && height >= 480) {
    // The cover's size in the flipbook's layout().
    const columns = width / height < ratio * 1.2 ? 1 : 2;
    // The page keeps clear of the stage's scrollbar gutters (site.css).
    const widest = stage.clientWidth - 64;
    const cover = Math.floor(Math.min((width - 64) / columns, (height - 56) * ratio, widest));
    if (fits(cover, minScale)) {
      pageWidth = cover;
      let low = minScale, high = Math.min(maxScale, Math.max(minScale, cover / baseWidth));
      if (fits(cover, high)) low = high;
      else for (let step = 0; step < 8; step++) {
        const middle = (low + high) / 2;
        if (fits(cover, middle)) low = middle;
        else high = middle;
      }
      scale = low;
    } else if (widest > cover && fits(widest, minScale)) {
      let low = cover, high = widest;
      while (high - low > 1) {
        const middle = Math.round((low + high) / 2);
        if (fits(middle, minScale)) high = middle;
        else low = middle;
      }
      pageWidth = high;
    }
  }
  if (pageWidth) {
    page.style.width = pageWidth + 'px';
    page.style.height = Math.round(pageWidth / ratio) + 'px';
    page.style.setProperty('--s', String(scale));
  } else {
    frame.classList.add('is-column');
    page.style.width = '';
    page.style.removeProperty('--s');
  }
}

function syncUrl(index) {
  const url = new URL(location.href);
  if (index > 0) url.searchParams.set('page', String(index + 1));
  else url.searchParams.delete('page');
  if (url.href !== location.href) history.replaceState(history.state, '', url);
}

// Page size: Fit shows the whole spread; bigger sizes scroll, and dragging moves the page.
function wireBookZoom(reader, flip) {
  const BOOK_ZOOMS = [1, 1.5, 2, 2.5];
  const out = reader.querySelector('[data-book-zoom-out]');
  const into = reader.querySelector('[data-book-zoom-in]');
  const level = reader.querySelector('.zoom-level');
  const show = () => {
    const step = BOOK_ZOOMS.indexOf(flip.zoom);
    out.disabled = step <= 0;
    into.disabled = step >= BOOK_ZOOMS.length - 1;
    level.textContent = flip.zoom === 1 ? 'Fit' : Math.round(flip.zoom * 100) + '%';
  };
  const go = delta => {
    const step = Math.max(0, Math.min(BOOK_ZOOMS.length - 1, BOOK_ZOOMS.indexOf(flip.zoom) + delta));
    flip.setZoom(BOOK_ZOOMS[step]);
    show();
  };
  out.addEventListener('click', () => go(-1));
  into.addEventListener('click', () => go(1));
  show();
}

// The toolbar's links: download (hosted papers only), the DOI page, and the whole collection.
function fillLinks(paper, reader) {
  const download = reader.querySelector('[data-download]');
  if (paper.allowDownload && paper.pdf) {
    download.href = paper.pdf;
    download.download = `${paper.id}.pdf`;
    // Browsers ignore `download` across origins; a new tab then keeps the reader in place.
    download.target = '_blank';
    download.rel = 'noopener';
    download.hidden = false;
  }
  const doi = reader.querySelector('[data-doi]');
  if (paper.doi) {
    doi.href = 'https://doi.org/' + paper.doi;
    doi.hidden = false;
  }
}

// FRAME opens the reader in a new tab, so going back means closing it. Inside
// FRAME's embedded overlay, or a tab the browser won't close, a note says how.
function wireBack(reader) {
  const inFrame = (() => {
    try { return window.self !== window.top; } catch { return true; }
  })();
  reader.querySelector('[data-back]').addEventListener('click', () => {
    if (inFrame) {
      showToast(reader.querySelector('.toast'), 'Close this panel to return to FRAME.');
      return;
    }
    window.close();
    setTimeout(() => {
      if (!window.closed) showToast(reader.querySelector('.toast'), 'Close this tab to return to FRAME.');
    }, 300);
  });
}

let toastTimer = 0;
function showToast(toast, message) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  requestAnimationFrame(() => requestAnimationFrame(() => toast.classList.add('is-visible')));
  toastTimer = setTimeout(() => {
    toast.classList.remove('is-visible');
    toastTimer = setTimeout(() => { toast.textContent = ''; }, 250);
  }, 4000);
}

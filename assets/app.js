import {createFlipbook} from './flipbook.js';
import {BookRasters, openPdf} from './pdf-pages.js';

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

// Owner's image reveal: blurred from the first paint, sharpening once decoded.
function reveal(image) {
  const done = () => image.classList.add('is-revealed');
  image.decode().then(done, done);
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
  void showLibrary();
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

async function showLibrary() {
  document.title = 'Research Library';
  root.classList.add('is-library');
  const library = document.getElementById('library');
  const halls = library.querySelector('.halls');
  const themes = museum.themes || [];
  library.querySelector('.library-count').textContent = `${museum.papers.length} papers · ${themes.length} halls`;
  for (const theme of themes) {
    const papers = museum.papers.filter(paper => paper.theme === theme.id);
    if (!papers.length) continue;
    const section = element('section', 'hall');
    section.style.setProperty('--accent', ACCENTS[theme.id] || LAVENDER);
    section.setAttribute('aria-labelledby', 'hall-' + theme.id);
    const head = element('div', 'hall-head');
    const name = element('h2', 'hall-name', theme.name);
    name.id = 'hall-' + theme.id;
    head.append(name, element('p', 'hall-blurb', theme.blurb));
    const list = element('ul', 'cards');
    // Covers stand on a shelf as tall as the hall's tallest page, so the titles line up.
    list.style.setProperty('--shelf', String(Math.min(...papers.map(ratioOf))));
    for (const paper of papers) list.append(card(paper, themes));
    section.append(head, list);
    halls.append(section);
  }
  await fontsReady(library);
  library.hidden = false;
}

function card(paper, themes) {
  const item = element('li');
  const link = element('a', 'card');
  link.href = '?p=' + encodeURIComponent(paper.id);
  const shelf = element('span', 'card-shelf');
  const frame = element('span', 'card-cover');
  // The frame takes the page's own aspect ratio, so the cover fills it exactly.
  frame.style.aspectRatio = String(ratioOf(paper));
  if (paper.pdf) {
    const image = new Image();
    image.className = 'reveal';
    image.alt = '';
    image.loading = 'lazy';
    image.decoding = 'async';
    image.src = `covers/${paper.id}.webp`;
    reveal(image);
    frame.append(image);
  } else {
    // A paper that isn't hosted here has no page image: a typeset first page stands in.
    const page = element('span', 'card-page');
    page.setAttribute('aria-hidden', 'true');
    page.append(
      element('span', 'card-page-title', typo(paper.title)),
      element('span', 'card-page-authors', (paper.authors || []).join(', ')),
      element('span', 'card-page-venue', typo(paper.venue || '')),
      element('span', 'card-page-abstract', typo((paper.abstract || []).join(' '))),
    );
    frame.append(page);
  }
  shelf.append(frame);
  link.append(shelf, element('span', 'card-title', typo(paper.shortTitle || paper.title)), element('span', 'card-meta', [paper.authorsShort, paper.year].filter(Boolean).join(' · ')));
  // Other halls this paper also belongs to.
  const others = (paper.tags || []).filter(tag => tag !== paper.theme).map(tag => themes.find(theme => theme.id === tag)).filter(Boolean);
  if (others.length) {
    const chips = element('span', 'card-chips');
    for (const theme of others) {
      const chip = element('span', 'chip', theme.name);
      chip.style.setProperty('--chip', '#ffffff');
      chips.append(chip);
    }
    link.append(chips);
  }
  item.append(link);
  return item;
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

  await fontsReady(reader);
  reader.hidden = false;
  flip.layout();
  new ResizeObserver(() => flip.layout()).observe(stage);

  pdf.then(source => {
    clearTimeout(loadingTimer);
    rasters.attach(source);
    flip.enable(source.numPages);
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
  // The page itself carries the details, so the book and its pager go.
  for (const node of [$('.book'), $('.pager')]) node.hidden = true;
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

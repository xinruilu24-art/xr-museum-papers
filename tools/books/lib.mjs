/* Shared helpers for the library book tools (Node 24, no packages).
   - loadMuseum(): reads data/papers.js in a sandbox (never edits it)
   - bookSpecs(): one spec per paper: size, colours, texts, texture sizes
   - chrome(): runs headless Chrome with a throwaway profile
   Build files (texture PNGs, generated pages) go to BOOKS_BUILD_DIR,
   default <os temp>/xr-museum-books-build. */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const TOOLS = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(TOOLS, '..', '..');
export const BOOKS = path.join(ROOT, 'books');
export const PREVIEWS = path.join(BOOKS, 'previews');
export const BUILD = process.env.BOOKS_BUILD_DIR || path.join(os.tmpdir(), 'xr-museum-books-build');
export const CHROME = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
// A throwaway profile. Never point this at the owner's real Chrome profile.
export const PROFILE = process.env.BOOKS_CHROME_PROFILE || path.join(os.tmpdir(), 'chrome-books-profile');

export function loadMuseum() {
  const file = path.join(ROOT, 'data', 'papers.js');
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: file });
  const m = sandbox.window.MUSEUM;
  if (!m || !Array.isArray(m.papers) || !Array.isArray(m.themes)) {
    throw new Error('data/papers.js did not set window.MUSEUM = { themes, papers }');
  }
  return JSON.parse(JSON.stringify(m));
}

/* Theme colours. cloth = deep book cloth, accent = pastel rule colour,
   foil = cream stamping colour. The three cloths share one OKLCH lightness
   (about 0.34-0.37) and a narrow chroma band (0.064-0.085), so they read as
   one family of book cloths. */
export const THEME_STYLE = {
  avatars: { cloth: '#382f5e', accent: '#b9a7ff' },
  harassment: { cloth: '#612734', accent: '#ef8f9b' },
  bias: { cloth: '#044a43', accent: '#7fd1c7' },
};
export const FOIL = '#f3ead8';
const FALLBACK_STYLE = { cloth: '#3b3a40', accent: '#d8d2c4' };

/* Cover title line breaks, one list of alternatives per paper, most
   preferred first. Kept here so data/papers.js stays untouched; an entry is
   used only if its words still match the paper's shortTitle (a line ending
   in "-" joins the next line without a space). */
export const COVER_LINES = {
  'zhang-2024': [['Gender', 'expression', 'in VRChat']],
  'schulenberg-2023': [['“Creepy', 'towards my', 'avatar body”']],
  'chen-2026': [['Harassment', 'in social', 'virtual reality'], ['Harassment', 'in social', 'virtual', 'reality']],
  'crone-2022': [['Interview', 'with an', 'avatar']],
  'caki-2026': [['VR for gender', 'micro-', 'aggression', 'training']],
  'guthula-2024': [['Navigating', 'gender', 'biases in XR']],
};
/* Two-line spine titles for the thick books (used only when one line would
   come out much smaller). */
export const SPINE_LINES = {
  'schulenberg-2023': ['“Creepy towards', 'my avatar body”'],
  'chen-2026': ['Harassment in social', 'virtual reality'],
};

const norm = (s) => s.toLowerCase().replace(/\s+/g, ' ').trim();
export function joinLines(lines) {
  return lines.reduce((acc, l) => (acc === '' ? l : acc.endsWith('-') ? acc.slice(0, -1) + l : `${acc} ${l}`), '');
}
function checkedLines(table, id, title, nested) {
  const alts = nested ? (table[id] || []) : (table[id] ? [table[id]] : []);
  const ok = alts.filter((a) => norm(joinLines(a)) === norm(title));
  if (alts.length && ok.length < alts.length) {
    console.warn(`warning: line breaks for ${id} no longer match its title, using automatic breaks`);
  }
  return ok;
}

/* OKLCH, for small lightness / chroma shifts of the cloth colours. */
const s2l = (c) => (c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
const l2s = (c) => (c <= 0.0031308 ? 12.92 * c : 1.055 * c ** (1 / 2.4) - 0.055);
function toOklch(hex) {
  const [r, g, b] = [1, 3, 5].map((i) => s2l(parseInt(hex.slice(i, i + 2), 16) / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  const A = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s;
  const B = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s;
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, Math.hypot(A, B), Math.atan2(B, A)];
}
function fromOklch([L, C, h]) {
  const A = C * Math.cos(h);
  const B = C * Math.sin(h);
  const l = (L + 0.3963377774 * A + 0.2158037573 * B) ** 3;
  const m = (L - 0.1055613458 * A - 0.0638541728 * B) ** 3;
  const s = (L - 0.0894841775 * A - 1.2914855480 * B) ** 3;
  const rgb = [
    4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
  ];
  return '#' + rgb.map((c) => Math.round(255 * l2s(clamp(c, 0, 1))).toString(16).padStart(2, '0')).join('');
}
/* kL / kC scale OKLCH lightness and chroma. */
export function shade(hex, { kL = 1, kC = 1 } = {}) {
  const [L, C, h] = toOklch(hex);
  return fromOklch([L * kL, C * kC, h]);
}

/* Sizes in metres. Book lying flat: X = width (spine at -X), Y = thickness,
   Z = height (head at -Z, tail at +Z). */
export const BOOK = {
  height: 0.28,        // board height (spine length)
  board: 0.003,        // board thickness
  square: 0.003,       // board overhang past the page block (head, tail, fore-edge)
  spineShell: 0.002,   // cloth + spine stiffener
};

/* Texture sizes stored in the GLBs (all powers of two). Each page is laid
   out at the part's true proportions and squeezed to fit. The front and
   spine both get 3.66 px/mm along the book's height. The back is rendered
   at twice its size with the front's cloth and downsampled, so its weave
   matches the front's. */
export const TEX = {
  front: [1024, 1024],
  back: [512, 512],
  spine: [256, 1024],
  edges: [512, 256],
};
export const RENDER_SCALE = { front: 1, back: 2, spine: 1, edges: 1 };

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const r5 = (v) => Math.round(v * 1e5) / 1e5;

export function bookDims(paper) {
  const H = BOOK.height;
  const [wPt, hPt] = paper.pageSize;
  const aspect = wPt / hPt;
  const blockH = H - 2 * BOOK.square;
  const W = 1.04 * blockH * aspect;                     // footprint width incl. spine
  const T = clamp(0.014 + paper.pages * 0.001, 0.016, 0.05);
  const bulge = clamp(0.14 * T, 0.0025, 0.006);          // spine rounding (sagitta)
  // Outer spine arc: chord T, sagitta bulge.
  const R = (T * T / 4 + bulge * bulge) / (2 * bulge);
  const half = Math.asin(T / 2 / R);
  const spineArc = 2 * R * half;
  return {
    H: r5(H), W: r5(W), T: r5(T), bulge: r5(bulge), aspect: r5(aspect),
    board: BOOK.board, square: BOOK.square, spineShell: BOOK.spineShell,
    coverW: r5(W - bulge), spineArc: r5(spineArc),
  };
}

const curly = (s) => s
  .replace(/(^|[\s(\[{-])"/g, '$1\u201c').replace(/"/g, '\u201d')
  .replace(/(^|[\s(\[{-])'/g, '$1\u2018').replace(/'/g, '\u2019');

/* Books of one theme get slightly different cloth lightness (within +-3%),
   so neighbouring lecterns don't hold identical-looking copies. */
function lightnessSteps(n) {
  if (n <= 1) return [1];
  const span = n === 2 ? 0.02 : 0.03;
  return Array.from({ length: n }, (_, i) => 1 - span + (2 * span * i) / (n - 1));
}

export function bookSpecs(museum = loadMuseum()) {
  const themes = Object.fromEntries(museum.themes.map((t) => [t.id, t]));
  const perTheme = {};
  for (const p of museum.papers) (perTheme[p.theme] ||= []).push(p.id);
  return museum.papers.map((p) => {
    const theme = themes[p.theme] || { id: p.theme, name: p.theme };
    const style = THEME_STYLE[p.theme] || FALLBACK_STYLE;
    const siblings = perTheme[p.theme];
    const kL = lightnessSteps(siblings.length)[siblings.indexOf(p.id)];
    const cloth = shade(style.cloth, { kL });
    const title = curly(p.shortTitle || p.title);
    return {
      id: p.id,
      theme: theme.id,
      themeName: theme.name,
      title,
      fullTitle: curly(p.title),
      coverLines: checkedLines(COVER_LINES, p.id, title, true),
      spineLines: checkedLines(SPINE_LINES, p.id, title, false)[0] || null,
      authors: p.authorsShort || (p.authors || []).join(', '),
      year: String(p.year),
      pages: p.pages,
      pageSize: p.pageSize,
      themeCloth: style.cloth,
      cloth,
      // the spine is a little sun-faded: lighter and less saturated
      spineCloth: shade(cloth, { kL: 1.04, kC: 0.9 }),
      accent: style.accent,
      foil: FOIL,
      dims: bookDims(p),
      tex: TEX,
    };
  });
}

export function chrome(args, timeoutMs = 120_000) {
  return new Promise((resolve, reject) => {
    const child = spawn(CHROME, args, { windowsHide: true });
    let out = '';
    let err = '';
    child.stdout.setEncoding('utf8').on('data', (d) => { out += d; });
    child.stderr.setEncoding('utf8').on('data', (d) => { err += d; });
    const timer = setTimeout(() => { child.kill(); reject(new Error('Chrome timed out')); }, timeoutMs);
    child.on('error', (e) => { clearTimeout(timer); reject(e); });
    child.on('close', (code) => { clearTimeout(timer); resolve({ code, out, err }); });
  });
}

/* Common flags: headless, throwaway profile, fixed pixel size. */
export function chromeArgs({ w, h, shot, webgl = false, dumpDom = true, budget = 10000 }) {
  return [
    '--headless=new',
    ...(webgl ? ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] : ['--disable-gpu']),
    '--hide-scrollbars',
    '--force-device-scale-factor=1',
    '--disable-lcd-text',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    `--window-size=${w},${h}`,
    `--virtual-time-budget=${budget}`,
    `--user-data-dir=${PROFILE}`,
    ...(dumpDom ? ['--dump-dom'] : []),
    ...(shot ? [`--screenshot=${shot}`] : []),
  ];
}

export function bodyAttrs(html) {
  const tag = (html.match(/<body\b[^>]*>/i) || [''])[0];
  const attrs = {};
  for (const [, k, v] of tag.matchAll(/([\w-]+)="([^"]*)"/g)) {
    attrs[k] = v.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  }
  return attrs;
}

export function pngSize(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error('not a PNG');
  return [buf.readUInt32BE(16), buf.readUInt32BE(20)];
}

/* Writes a page into the build folder with the data injected and a <base>
   pointing back at tools/books so relative links keep working. */
export function writePage(template, outName, data) {
  const src = fs.readFileSync(path.join(TOOLS, template), 'utf8');
  const base = `<base href="${pathToFileURL(TOOLS + path.sep).href}">`;
  const inject = `<script>window.BOOK = ${JSON.stringify(data).replace(/</g, '\\u003c')};</script>`;
  const html = src.replace('<!--BASE-->', base).replace('<!--BOOK-->', inject);
  const out = path.join(BUILD, outName);
  fs.mkdirSync(path.dirname(out), { recursive: true });
  fs.writeFileSync(out, html);
  return out;
}

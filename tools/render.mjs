#!/usr/bin/env node
/* Renders the FRAME museum images with headless Chrome (Node 24, no packages).
     node tools/render.mjs           all images
     node tools/render.mjs plaque    only images whose file name contains "plaque"
   Each page is loaded once with --dump-dom and --screenshot: the dumped <body>
   carries data-ready / data-fit / data-font-px, the screenshot becomes the PNG.
   Exits non-zero if any image errored, overflowed, missed DM Sans or has the
   wrong pixel size. Only a clean render replaces panels/<name>.png; an
   overflowing one is saved as panels/<name>.OVERFLOW.png for inspection.
   CHROME and RENDER_CHROME_PROFILE override the browser and its profile
   (default <os temp>/chrome-render-profile). */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import zlib from 'node:zlib';
import { fileURLToPath, pathToFileURL } from 'node:url';

const CHROME = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
// A throwaway profile. Never point this at the owner's real Chrome profile.
const PROFILE = process.env.RENDER_CHROME_PROFILE || path.join(os.tmpdir(), 'chrome-render-profile');
const TOOLS = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(TOOLS, '..');
const OUT = path.join(ROOT, 'panels');
const MAX_BYTES = 10 * 1024 * 1024; // FRAME's upload limit
const PNG_SIG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

/* fs.rmSync silently does nothing for non-ASCII Windows paths (this folder
   contains "文档") on Node 24, so delete with unlinkSync. */
function removeFile(file) {
  try { fs.unlinkSync(file); } catch (e) { if (e.code !== 'ENOENT') throw e; }
}

function loadMuseum() {
  const file = path.join(ROOT, 'data', 'papers.js');
  const sandbox = { window: {} };
  vm.runInNewContext(fs.readFileSync(file, 'utf8'), sandbox, { filename: file });
  const m = sandbox.window.MUSEUM;
  if (!m || !Array.isArray(m.papers) || !Array.isArray(m.themes)) {
    throw new Error('data/papers.js did not set window.MUSEUM = { themes, papers }');
  }
  return m;
}

function jobsFor(m) {
  return [
    ...m.papers.map((p) => ({ file: `${p.id}.png`, page: 'panel.html', query: { p: p.id }, w: 1500, h: 2000 })),
    ...m.papers.map((p) => ({ file: `plaque-${p.id}.png`, page: 'plaque.html', query: { p: p.id }, w: 1600, h: 900 })),
    ...m.themes.map((t) => ({ file: `hall-${t.id}.png`, page: 'hall.html', query: { t: t.id }, w: 2400, h: 1000 })),
    { file: 'library-sign.png', page: 'library-sign.html', query: {}, w: 2400, h: 1000 },
  ];
}

function chrome(args, timeoutMs = 90_000) {
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

function bodyAttrs(html) {
  const tag = (html.match(/<body\b[^>]*>/i) || [''])[0];
  const attrs = {};
  for (const [, k, v] of tag.matchAll(/([\w-]+)="([^"]*)"/g)) {
    attrs[k] = v.replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  }
  return attrs;
}

/* ---------- PNG: confirm the size, keep it opaque RGB, tag it sRGB ---------- */

function pngChunks(buf) {
  if (!buf.subarray(0, 8).equals(PNG_SIG)) throw new Error('not a PNG file');
  const chunks = [];
  for (let o = 8; o + 12 <= buf.length;) {
    const len = buf.readUInt32BE(o);
    chunks.push({ type: buf.toString('latin1', o + 4, o + 8), data: buf.subarray(o + 8, o + 8 + len) });
    o += 12 + len;
  }
  return chunks;
}

function chunk(type, data) {
  const head = Buffer.alloc(8);
  head.writeUInt32BE(data.length, 0);
  head.write(type, 4, 'latin1');
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(zlib.crc32(Buffer.concat([head.subarray(4), data])) >>> 0, 0);
  return Buffer.concat([head, data, crc]);
}

function paeth(a, b, c) {
  const p = a + b - c;
  const pa = Math.abs(p - a); const pb = Math.abs(p - b); const pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/* RGBA -> RGB. Only used if Chrome ever hands back an alpha channel. */
function dropAlpha(idat, w, h) {
  const src = zlib.inflateSync(idat);
  const stride = w * 4;
  const prev = Buffer.alloc(stride);
  const cur = Buffer.alloc(stride);
  const out = Buffer.alloc(h * (w * 3 + 1));
  let translucent = 0;
  for (let y = 0; y < h; y++) {
    const ft = src[y * (stride + 1)];
    const row = src.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    for (let i = 0; i < stride; i++) {
      const a = i >= 4 ? cur[i - 4] : 0; const b = prev[i]; const c = i >= 4 ? prev[i - 4] : 0;
      const pred = ft === 1 ? a : ft === 2 ? b : ft === 3 ? (a + b) >> 1 : ft === 4 ? paeth(a, b, c) : 0;
      cur[i] = (row[i] + pred) & 255;
    }
    const o = y * (w * 3 + 1);
    out[o] = 0; // filter: none
    for (let x = 0; x < w; x++) {
      if (cur[x * 4 + 3] !== 255) translucent++;
      out[o + 1 + x * 3] = cur[x * 4];
      out[o + 2 + x * 3] = cur[x * 4 + 1];
      out[o + 3 + x * 3] = cur[x * 4 + 2];
    }
    cur.copy(prev);
  }
  if (translucent) throw new Error(`${translucent} translucent pixels in the screenshot`);
  return zlib.deflateSync(out, { level: 9 });
}

function normalizePng(buf) {
  const chunks = pngChunks(buf);
  const ihdr = Buffer.from(chunks[0].data);
  const width = ihdr.readUInt32BE(0);
  const height = ihdr.readUInt32BE(4);
  const depth = ihdr[8]; const colorType = ihdr[9]; const interlace = ihdr[12];
  if (depth !== 8 || interlace !== 0 || (colorType !== 2 && colorType !== 6)) {
    throw new Error(`unexpected PNG format (depth ${depth}, colour type ${colorType}, interlace ${interlace})`);
  }
  let idat = [Buffer.concat(chunks.filter((c) => c.type === 'IDAT').map((c) => c.data))];
  if (colorType === 6) { idat = [dropAlpha(idat[0], width, height)]; ihdr[9] = 2; }
  const png = Buffer.concat([
    PNG_SIG,
    chunk('IHDR', ihdr),
    chunk('sRGB', Buffer.from([0])), // perceptual intent: the pixels are sRGB
    ...idat.map((d) => chunk('IDAT', d)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
  return { png, width, height };
}

/* ---------- render ---------- */

async function render(job) {
  const url = new URL(pathToFileURL(path.join(TOOLS, job.page)).href);
  for (const [k, v] of Object.entries(job.query)) url.searchParams.set(k, v);
  const target = path.join(OUT, job.file);
  // A clipped render goes here, never to the real file name, so it cannot be uploaded by mistake.
  const clippedFile = path.join(OUT, job.file.replace(/\.png$/, '.OVERFLOW.png'));
  const shot = path.join(OUT, `.rendering-${job.file}`);
  const started = Date.now();
  removeFile(shot);
  try {
    const { code, out, err } = await chrome([
      '--headless=new',
      '--disable-gpu',
      '--hide-scrollbars',
      '--force-device-scale-factor=1',
      '--disable-lcd-text', // grayscale text antialiasing (the CSS also forces it)
      `--window-size=${job.w},${job.h}`,
      '--virtual-time-budget=10000',
      `--user-data-dir=${PROFILE}`,
      '--dump-dom',
      `--screenshot=${shot}`,
      url.href,
    ]);
    const a = bodyAttrs(out);
    const res = {
      file: job.file, want: `${job.w}x${job.h}`, size: '-', bytes: 0, fit: a['data-fit'] || 'none',
      fontPx: a['data-font-px'] || '-', titlePx: a['data-title-px'] || '-', fonts: a['data-fonts'] || '-',
      slack: a['data-slack'], problems: [], ms: 0,
    };
    if (a['data-ready'] !== '1') res.problems.push('page never set data-ready="1"');
    if (a['data-error']) res.problems.push(a['data-error']);
    if (res.fit === 'overflow') res.problems.push('text overflows');
    if (res.fonts !== 'dm-sans') res.problems.push('DM Sans did not load');
    if (!fs.existsSync(shot)) {
      res.problems.push(`no screenshot (Chrome exit ${code}) ${err.trim().split('\n').slice(-2).join(' ')}`);
    } else {
      const { png, width, height } = normalizePng(fs.readFileSync(shot));
      res.size = `${width}x${height}`;
      res.bytes = png.length;
      if (width !== job.w || height !== job.h) res.problems.push(`expected ${res.want}`);
      if (png.length > MAX_BYTES) res.problems.push('larger than 10 MB');
      if (res.problems.length === 0) {
        fs.writeFileSync(target, png);
        removeFile(clippedFile); // a clipped render from an earlier run is now stale
      } else if (res.fit === 'overflow') {
        fs.writeFileSync(clippedFile, png);
        res.problems.push(`${job.file} not updated; clipped render saved as ${path.basename(clippedFile)}`);
      }
    }
    res.ms = Date.now() - started;
    return res;
  } finally {
    removeFile(shot);
  }
}

function table(rows) {
  const cols = [
    ['image', (r) => r.file],
    ['pixels', (r) => r.size],
    ['KB', (r) => (r.bytes ? String(Math.round(r.bytes / 1024)) : '-')],
    ['fit', (r) => r.fit],
    ['font px', (r) => r.fontPx],
    ['title px', (r) => r.titlePx],
    ['slack', (r) => (r.slack == null ? '' : r.slack)],
    ['s', (r) => (r.ms / 1000).toFixed(1)],
    ['problems', (r) => r.problems.join('; ')],
  ];
  const cells = rows.map((r) => cols.map(([, f]) => String(f(r))));
  const widths = cols.map(([h], i) => Math.max(h.length, ...cells.map((c) => c[i].length)));
  const line = (vals) => vals.map((v, i) => v.padEnd(widths[i])).join('  ').trimEnd();
  console.log(line(cols.map(([h]) => h)));
  console.log(line(widths.map((w) => '-'.repeat(w))));
  for (const c of cells) console.log(line(c));
}

async function main() {
  const filters = process.argv.slice(2);
  const jobs = jobsFor(loadMuseum()).filter((j) => !filters.length || filters.some((f) => j.file.includes(f)));
  if (!jobs.length) throw new Error(`no images match: ${filters.join(' ')}`);
  fs.mkdirSync(OUT, { recursive: true });
  fs.mkdirSync(PROFILE, { recursive: true });
  const rows = [];
  for (const job of jobs) {
    process.stdout.write(`rendering ${job.file} ...\r`);
    try {
      rows.push(await render(job));
    } catch (e) {
      rows.push({ file: job.file, size: '-', bytes: 0, fit: 'error', fontPx: '-', titlePx: '-', problems: [e.message], ms: 0 });
    }
  }
  process.stdout.write(' '.repeat(60) + '\r');
  table(rows);
  const bad = rows.filter((r) => r.problems.length);
  console.log(bad.length ? `\n${bad.length} of ${rows.length} images have problems.` : `\nAll ${rows.length} images rendered to ${OUT}`);
  process.exitCode = bad.length ? 1 : 0;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });

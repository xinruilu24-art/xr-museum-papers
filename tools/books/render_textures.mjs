#!/usr/bin/env node
/* Step 1 of 3: renders the book textures with headless Chrome.
     node tools/books/render_textures.mjs            every texture
     node tools/books/render_textures.mjs zhang      only names containing "zhang"
   Writes <build>/tex/<id>-front.png, -spine.png, -back.png, edges.png and
   <build>/manifest.json (sizes, colours, texts) for make_books.py.
   Exits non-zero if a page errored, overflowed, missed DM Sans or came out
   at the wrong pixel size. */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { BUILD, PROFILE, RENDER_SCALE, bookSpecs, chrome, chromeArgs, bodyAttrs, pngSize, shade, writePage } from './lib.mjs';

const TEX_DIR = path.join(BUILD, 'tex');

function seedOf(id) {
  let h = 7;
  for (const c of id) h = (h * 31 + c.charCodeAt(0)) % 997;
  return h + 1;
}

function jobsFor(specs) {
  const series = specs.map((s) => ({ id: s.id, title: s.title, coverW: s.dims.coverW * 1000, lines: s.coverLines }));
  const jobs = [];
  for (const s of specs) {
    const Hmm = s.dims.H * 1000;
    const coverW = s.dims.coverW * 1000;
    const arc = s.dims.spineArc * 1000;
    const common = {
      id: s.id, themeName: s.themeName, title: s.title, authors: s.authors, year: s.year,
      cloth: s.cloth, accent: s.accent, foil: s.foil, coverW, seed: seedOf(s.id),
    };
    // The page is laid out in true proportions (mm px per millimetre) and
    // scaled to the texture: covers keep their width and are squeezed in
    // height to a power-of-two texture, the spine keeps its height. Parts
    // with a render scale above 1 are rendered larger and downsampled by
    // make_books.py.
    const part = (name, page, [outW, outH], designWmm, extra = {}, keepWidth = false) => {
      const k = RENDER_SCALE[name] || 1;
      const texW = outW * k;
      const texH = outH * k;
      const mm = keepWidth ? texW / designWmm : texH / Hmm;
      jobs.push({
        name: `${s.id}-${name}`, page, needFont: page !== 'back.html',
        data: { ...common, part: name, texW, texH, designW: designWmm * mm, designH: Hmm * mm, mm, ...extra },
      });
    };
    // cloth threads and grain in cycles per mm, the same on every part
    const weave = { threads: 1.25, grain: 1.1 };
    part('front', 'cover.html', s.tex.front, coverW, { ...weave, series, lines: s.coverLines, rub: 'trb' }, true);
    part('spine', 'spine.html', s.tex.spine, arc, { ...weave, cloth: s.spineCloth, spineLines: s.spineLines, rub: 'tb' });
    part('back', 'back.html', s.tex.back, coverW, { ...weave, rub: 'tlb' }, true);
  }
  // page edges + headband stripes, one per theme
  const [ew, eh] = specs[0].tex.edges;
  for (const s of specs) {
    if (jobs.some((j) => j.name === `edges-${s.theme}`)) continue;
    jobs.push({
      name: `edges-${s.theme}`, page: 'edges.html', needFont: false,
      data: {
        part: 'edges', texW: ew, texH: eh, designW: ew, designH: eh, mm: 1, noCloth: true,
        cloth: s.themeCloth, band: shade(s.themeCloth, { kL: 1.3, kC: 0.95 }), accent: s.accent, foil: s.foil,
      },
    });
  }
  return jobs;
}

async function render(job) {
  const page = writePage(job.page, path.join('pages', `${job.name}.html`), job.data);
  const shot = path.join(TEX_DIR, `${job.name}.png`);
  fs.rmSync(shot, { force: true });
  const t0 = Date.now();
  const { code, out, err } = await chrome([
    ...chromeArgs({ w: job.data.texW, h: job.data.texH, shot }),
    pathToFileURL(page).href,
  ]);
  const a = bodyAttrs(out);
  const problems = [];
  if (a['data-ready'] !== '1') problems.push('page never set data-ready');
  if (a['data-error']) problems.push(a['data-error']);
  if (a['data-fit'] === 'overflow') problems.push('text overflows');
  if (job.needFont && a['data-fonts'] !== 'dm-sans') problems.push('DM Sans did not load');
  let size = '-';
  if (!fs.existsSync(shot)) {
    problems.push(`no screenshot (exit ${code}) ${err.trim().split('\n').slice(-1)[0] || ''}`);
  } else {
    const [w, h] = pngSize(fs.readFileSync(shot));
    size = `${w}x${h}`;
    if (w !== job.data.texW || h !== job.data.texH) problems.push(`expected ${job.data.texW}x${job.data.texH}`);
  }
  return { name: job.name, size, attrs: a, problems, ms: Date.now() - t0, file: shot };
}

async function main() {
  const filters = process.argv.slice(2);
  const specs = bookSpecs();
  fs.mkdirSync(TEX_DIR, { recursive: true });
  fs.mkdirSync(PROFILE, { recursive: true });
  const jobs = jobsFor(specs).filter((j) => !filters.length || filters.some((f) => j.name.includes(f)));
  const results = [];
  for (const job of jobs) {
    const r = await render(job);
    results.push(r);
    const extra = ['titlePx', 'titleLines', 'capMm', 'seriesPx'].filter((k) => r.attrs['data-' + k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())]).map((k) => `${k}=${r.attrs['data-' + k.replace(/[A-Z]/g, (c) => '-' + c.toLowerCase())]}`).join(' ');
    console.log(`${r.problems.length ? 'FAIL' : 'ok  '} ${r.name.padEnd(24)} ${r.size.padEnd(10)} ${(r.ms / 1000).toFixed(1)}s ${extra} ${r.problems.join('; ')}`);
  }
  const manifest = {
    build: BUILD,
    books: specs.map((s) => ({
      ...s,
      textures: {
        ...Object.fromEntries(['front', 'spine', 'back'].map((p) => [p, path.join(TEX_DIR, `${s.id}-${p}.png`)])),
        edges: path.join(TEX_DIR, `edges-${s.theme}.png`),
      },
      // sizes stored in the GLB (renders larger than this get downsampled)
      textureSizes: s.tex,
    })),
  };
  fs.writeFileSync(path.join(BUILD, 'manifest.json'), JSON.stringify(manifest, null, 2));
  const bad = results.filter((r) => r.problems.length);
  console.log(bad.length ? `\n${bad.length} of ${results.length} textures have problems.` : `\nAll ${results.length} textures rendered to ${TEX_DIR}`);
  process.exitCode = bad.length ? 1 : 0;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });

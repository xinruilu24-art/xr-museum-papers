/* Shared page script for the book textures. window.BOOK is injected by
   render_textures.mjs. Each page calls Book.start(layout), where layout()
   places the content inside #content and returns fit details. */
(function () {
  const B = window.BOOK;
  const root = document.documentElement.style;
  const set = (k, v) => root.setProperty(k, v);
  set('--tex-w', B.texW + 'px');
  set('--tex-h', B.texH + 'px');
  set('--design-w', B.designW + 'px');
  set('--design-h', B.designH + 'px');
  set('--sx', String(B.texW / B.designW));
  set('--sy', String(B.texH / B.designH));
  set('--mm', B.mm + 'px');
  set('--cloth', B.cloth);
  set('--accent', B.accent);
  set('--foil', B.foil);

  const mm = B.mm;
  const W = B.designW;
  const H = B.designH;
  const NS = 'http://www.w3.org/2000/svg';

  function noiseLayer(cls, fx, fy, octaves, seed) {
    const id = 'n' + cls.replace(/\W/g, '') + seed;
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('class', 'layer ' + cls);
    svg.setAttribute('width', W);
    svg.setAttribute('height', H);
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML =
      `<filter id="${id}" x="0" y="0" width="100%" height="100%" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB">` +
      `<feTurbulence type="fractalNoise" baseFrequency="${fx} ${fy}" numOctaves="${octaves}" seed="${seed}"/>` +
      '<feColorMatrix type="matrix" values="0.34 0.33 0.33 0 0  0.34 0.33 0.33 0 0  0.34 0.33 0.33 0 0  0 0 0 0 1"/>' +
      '<feComponentTransfer><feFuncR type="linear" slope="1.9" intercept="-0.45"/><feFuncG type="linear" slope="1.9" intercept="-0.45"/><feFuncB type="linear" slope="1.9" intercept="-0.45"/></feComponentTransfer>' +
      '</filter>' +
      `<rect x="0" y="0" width="${W}" height="${H}" filter="url(#${id})"/>`;
    return svg;
  }

  /* Stamping filters. Foil and accent ink sit in a pressed-in bed: a dark
     inner edge along the top of each shape (the wall that faces away from
     the light) and a faint light rim just below it. #deboss is for type,
     #deboss-fine for thin rules. */
  function stampFilters() {
    const f = (id, off, blur, dark, rim, rimOp, box) =>
      `<filter id="${id}" ${box} color-interpolation-filters="sRGB">` +
      `<feOffset in="SourceAlpha" dx="0" dy="${off}" result="down"/>` +
      '<feComposite in="SourceAlpha" in2="down" operator="out" result="edge"/>' +
      `<feGaussianBlur in="edge" stdDeviation="${blur}" result="soft"/>` +
      '<feComposite in="soft" in2="SourceAlpha" operator="in" result="inner"/>' +
      `<feFlood flood-color="#000" flood-opacity="${dark}"/>` +
      '<feComposite in2="inner" operator="in" result="shadow"/>' +
      `<feOffset in="SourceAlpha" dx="0" dy="${rim}" result="low"/>` +
      '<feComposite in="low" in2="SourceAlpha" operator="out" result="rimA"/>' +
      `<feFlood flood-color="#fff" flood-opacity="${rimOp}"/>` +
      '<feComposite in2="rimA" operator="in" result="rim"/>' +
      '<feMerge><feMergeNode in="rim"/><feMergeNode in="SourceGraphic"/><feMergeNode in="shadow"/></feMerge>' +
      '</filter>';
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('width', '0');
    svg.setAttribute('height', '0');
    svg.setAttribute('aria-hidden', 'true');
    svg.style.position = 'absolute';
    svg.innerHTML = '<defs>' +
      f('deboss', 0.3 * mm, 0.07 * mm, 0.5, 0.2 * mm, 0.12, 'x="-5%" y="-10%" width="110%" height="120%"') +
      f('deboss-fine', 0.12 * mm, 0.03 * mm, 0.4, 0.1 * mm, 0.1, 'x="-20%" y="-400%" width="140%" height="900%"') +
      '</defs>';
    document.body.prepend(svg);
  }

  /* Rubbed edges: worn cloth is lighter along the board edges and at the
     corners. B.rub lists the edges that get it (t, r, b, l). */
  function rubLayer() {
    const edges = B.rub == null ? 'trbl' : B.rub;
    if (!edges) return null;
    const band = 2.6 * mm;
    const dir = { t: '180deg', b: '0deg', l: '90deg', r: '270deg' };
    const grads = [];
    for (const e of edges) {
      grads.push(`linear-gradient(${dir[e]}, rgba(255,255,255,0.055) 0, rgba(255,255,255,0.025) ${band * 0.45}px, rgba(255,255,255,0) ${band}px)`);
    }
    const corner = (x, y) => `radial-gradient(${10 * mm}px ${10 * mm}px at ${x} ${y}, rgba(255,255,255,0.06), rgba(255,255,255,0) 100%)`;
    const has = (e) => edges.includes(e);
    if (has('t') && has('r')) grads.push(corner('100%', '0%'));
    if (has('b') && has('r')) grads.push(corner('100%', '100%'));
    if (has('t') && has('l')) grads.push(corner('0%', '0%'));
    if (has('b') && has('l')) grads.push(corner('0%', '100%'));
    const d = document.createElement('div');
    d.className = 'layer rub';
    d.style.background = grads.join(',');
    return d;
  }

  /* Cloth: mottling under the stamping; threads and grain over it so the
     stamping picks up the weave like real stamping on book cloth. */
  function cloth(art, content) {
    const seed = B.seed || 3;
    const threads = B.threads == null ? 1.6 : B.threads; // thread cycles per mm
    const grain = B.grain == null ? 1.4 : B.grain;       // grain cycles per mm
    art.insertBefore(noiseLayer('mottle', 0.03 / mm, 0.03 / mm, 4, seed), content);
    if (threads > 0) {
      art.appendChild(noiseLayer('threads-h', 0.3 / mm, threads / mm, 2, seed + 11));
      art.appendChild(noiseLayer('threads-v', threads / mm, 0.3 / mm, 2, seed + 23));
    }
    art.appendChild(noiseLayer('grain', grain / mm, grain / mm, 2, seed + 37));
    const wear = document.createElement('div');
    wear.className = 'layer wear';
    art.appendChild(wear);
    const rub = rubLayer();
    if (rub) art.appendChild(rub);
  }

  function fit(el, { maxW, maxH = Infinity, maxLines = 99, min = 6, max = 400 }) {
    const ok = (px) => {
      el.style.fontSize = px + 'px';
      const lh = parseFloat(getComputedStyle(el).lineHeight) || px;
      const lines = Math.round(el.scrollHeight / lh);
      return el.scrollWidth <= Math.ceil(maxW) + 0.5 && el.scrollHeight <= maxH + 0.5 && lines <= maxLines;
    };
    if (!ok(min)) { el.style.fontSize = min + 'px'; return { px: min, ok: false }; }
    let lo = min;
    let hi = max;
    for (let i = 0; i < 22; i++) {
      const mid = (lo + hi) / 2;
      if (ok(mid)) lo = mid; else hi = mid;
    }
    ok(lo);
    return { px: lo, ok: true };
  }

  function lineCount(el) {
    const lh = parseFloat(getComputedStyle(el).lineHeight);
    return Math.round(el.scrollHeight / lh);
  }

  /* Sets explicit lines. A line ending in "-" keeps its hyphen, hung in the
     margin so the word itself stays centred. */
  function setLines(el, lines) {
    el.textContent = '';
    lines.forEach((line, i) => {
      if (i) el.appendChild(document.createElement('br'));
      if (line.endsWith('-')) {
        el.appendChild(document.createTextNode(line.slice(0, -1)));
        const hy = document.createElement('span');
        hy.className = 'hang';
        hy.textContent = '-';
        el.appendChild(hy);
      } else {
        el.appendChild(document.createTextNode(line));
      }
    });
  }

  async function start(layout) {
    const body = document.body;
    try {
      stampFilters();
      try { await document.fonts.load('600 48px "DM Sans"', 'AZ'); } catch (e) { /* reported below */ }
      await document.fonts.ready;
      const dm = [...document.fonts].some((f) => f.family.replace(/["']/g, '') === 'DM Sans' && f.status === 'loaded');
      body.dataset.fonts = dm ? 'dm-sans' : 'fallback';
      const art = document.getElementById('art');
      const content = document.getElementById('content');
      const info = (await layout({ mm, W, H, fit, lineCount, setLines, B, noiseLayer })) || {};
      if (!B.noCloth) cloth(art, content);
      for (const [k, v] of Object.entries(info)) body.dataset[k] = typeof v === 'string' ? v : JSON.stringify(v);
      body.dataset.ready = '1';
    } catch (e) {
      body.dataset.error = String(e && e.stack || e);
    }
  }

  window.Book = { start, fit, lineCount, setLines, mm, W, H, B };
})();

// PDF.js loading, one shared render queue per document, and the book's page rasters.

// Keep in step with the modulepreload in index.html.
const PDFJS_VERSION = '4.10.38';
const PDFJS_BASE = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDFJS_VERSION}`;

export const MAX_CANVAS_SIDE = 4096;
// Jobs at or above this priority wait while a page is turning.
export const BACKGROUND = 10;
// Phones and tablets get smaller canvas budgets (iOS caps a tab's total canvas memory).
export const CONSTRAINED = matchMedia('(pointer: coarse)').matches;

let library = null;

function loadLibrary() {
  library ??= import(`${PDFJS_BASE}/legacy/build/pdf.min.mjs`).then(lib => {
    lib.GlobalWorkerOptions.workerSrc = `${PDFJS_BASE}/legacy/build/pdf.worker.min.mjs`;
    return lib;
  });
  return library;
}

function abortError() {
  return new DOMException('Render cancelled', 'AbortError');
}

// Gives a canvas's pixel memory back now instead of at garbage collection.
export function releaseCanvas(canvas) {
  canvas.width = 0;
  canvas.height = 0;
}

export async function openPdf(url) {
  const lib = await loadLibrary();
  const task = lib.getDocument({
    url,
    cMapUrl: `${PDFJS_BASE}/cmaps/`,
    cMapPacked: true,
    standardFontDataUrl: `${PDFJS_BASE}/standard_fonts/`,
    isEvalSupported: false,
  });
  return new PdfSource(await task.promise);
}

export class PdfSource {
  #doc;
  #pages = new Map();
  #queue = [];
  #running = null;
  #held = false;
  #seq = 0;
  #sizes = null;
  #retained = new Map();
  #clean = new Set();

  constructor(doc) {
    this.#doc = doc;
    this.numPages = doc.numPages;
  }

  page(index) {
    if (!this.#pages.has(index)) this.#pages.set(index, this.#doc.getPage(index + 1));
    return this.#pages.get(index);
  }

  // [width, height] in PDF points for every page, honouring /Rotate.
  pageSizes() {
    this.#sizes ??= Promise.all(Array.from({length: this.numPages}, async (_, index) => {
      const viewport = (await this.page(index)).getViewport({scale: 1});
      return [viewport.width, viewport.height];
    }));
    return this.#sizes;
  }

  // Renders one page laid out at width × height (white paper, page scaled to fit
  // and centred) into a new canvas. With a region, only that part of the layout
  // is drawn, so large renders can be split into tiles at full resolution.
  // Lower priority numbers run first; one job at a time.
  request(index, width, height, priority = 0, region = null) {
    let resolve, reject;
    const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
    const job = {index, width, height, region, priority, seq: this.#seq++, resolve, reject, task: null, aborted: false, preempted: false};
    this.#queue.push(job);
    this.#pump();
    return {
      promise,
      setPriority: value => {
        job.priority = value;
        this.#pump();
      },
      cancel: () => this.#abort(job),
    };
  }

  // An invisible, selectable text layer for the page. Its size follows the
  // container's --scale-factor (CSS px per PDF point), so zooming needs no redraw.
  renderText(index, container) {
    let layer = null;
    let cancelled = false;
    const promise = Promise.all([loadLibrary(), this.page(index)]).then(([lib, page]) => {
      if (cancelled) throw abortError();
      layer = new lib.TextLayer({textContentSource: page.streamTextContent(), container, viewport: page.getViewport({scale: 1})});
      return layer.render();
    });
    return {
      promise,
      cancel: () => {
        cancelled = true;
        layer?.cancel();
      },
    };
  }

  // Each view says which pages it is using. Pages no view uses give back PDF.js's
  // parsed data (operator lists, decoded images); PDF.js defers that while a
  // render of the page is still pending.
  retain(owner, indices) {
    this.#retained.set(owner, new Set(indices));
    const used = index => [...this.#retained.values()].some(set => set.has(index));
    for (const [index, page] of this.#pages) {
      if (this.#clean.has(index) || used(index)) continue;
      if (this.#running?.index === index || this.#queue.some(job => job.index === index)) continue;
      this.#clean.add(index);
      page.then(proxy => proxy.cleanup(), () => {});
    }
  }

  // Keeps the main thread free for an animation: background jobs wait and a
  // running background render is cancelled and queued again.
  hold() {
    this.#held = true;
    const job = this.#running;
    if (job && job.priority >= BACKGROUND && !job.preempted) {
      job.preempted = true;
      job.task?.cancel();
    }
  }

  release() {
    if (!this.#held) return;
    this.#held = false;
    this.#pump();
  }

  #abort(job) {
    if (job.aborted) return;
    job.aborted = true;
    const queued = this.#queue.indexOf(job);
    if (queued >= 0) {
      this.#queue.splice(queued, 1);
      job.reject(abortError());
    } else {
      job.task?.cancel();
    }
  }

  #pump() {
    if (this.#running) return;
    let best = null;
    for (const job of this.#queue) {
      if (this.#held && job.priority >= BACKGROUND) continue;
      if (!best || job.priority < best.priority || (job.priority === best.priority && job.seq < best.seq)) best = job;
    }
    if (!best) return;
    this.#queue.splice(this.#queue.indexOf(best), 1);
    this.#running = best;
    this.#clean.delete(best.index);
    void this.#run(best);
  }

  async #run(job) {
    let canvas = null;
    try {
      const page = await this.page(job.index);
      if (job.aborted || job.preempted) throw abortError();
      const area = job.region ?? {x: 0, y: 0, width: job.width, height: job.height};
      canvas = document.createElement('canvas');
      canvas.width = area.width;
      canvas.height = area.height;
      const base = page.getViewport({scale: 1});
      const viewport = page.getViewport({scale: Math.min(job.width / base.width, job.height / base.height)});
      const dx = (job.width - viewport.width) / 2 - area.x;
      const dy = (job.height - viewport.height) / 2 - area.y;
      job.task = page.render({
        canvasContext: canvas.getContext('2d', {alpha: false}),
        viewport,
        transform: dx || dy ? [1, 0, 0, 1, dx, dy] : null,
        background: '#ffffff',
      });
      await job.task.promise;
      job.resolve(canvas);
    } catch (error) {
      if (canvas) releaseCanvas(canvas);
      if (job.preempted && !job.aborted) {
        job.preempted = false;
        job.task = null;
        this.#queue.push(job);
      } else {
        job.reject(job.aborted ? abortError() : error);
      }
    } finally {
      this.#running = null;
      this.#pump();
    }
  }
}

function errorPage(width, height, index) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d', {alpha: false});
  context.fillStyle = '#ffffff';
  context.fillRect(0, 0, width, height);
  context.fillStyle = '#7a7f7c';
  context.font = `${Math.max(12, Math.round(width / 32))}px "DM Sans", system-ui, sans-serif`;
  context.textAlign = 'center';
  context.fillText(`Page ${index + 1} could not be displayed.`, width / 2, height / 2);
  return canvas;
}

function releaseEntry(entry) {
  for (const canvas of entry.levels ?? [entry.canvas]) releaseCanvas(canvas);
}

// Book page rasters at the current page size, kept in a bounded LRU.
// Every canvas handed out is copied by its user, so dropped entries are released.
export class BookRasters {
  #source = null;
  #width = 0;
  #height = 0;
  #cache = new Map();
  #stale = new Map();
  #jobs = new Map();
  #visible = [];
  #ahead = [];
  #listeners = new Set();
  #fallback = null;
  #offset = 0;

  constructor({maxEntries = 12, maxPixels = CONSTRAINED ? 16e6 : 40e6} = {}) {
    this.maxEntries = maxEntries;
    this.maxPixels = maxPixels;
  }

  attach(source) {
    this.#source = source;
    this.#sync();
  }

  onRaster(listener) {
    this.#listeners.add(listener);
  }

  // Another canvas of the same page (any size) that can stand in until a raster is ready.
  setFallback(provider) {
    this.#fallback = provider;
  }

  // While another view needs the renderer, the book's own jobs queue behind it.
  setDeferred(deferred) {
    this.#offset = deferred ? 2 * BACKGROUND : 0;
    for (const job of this.#jobs.values()) {
      job.priority = job.urgent ? 0 : job.base + this.#offset;
      job.handle.setPriority(job.priority);
    }
  }

  setSize(width, height) {
    if (width === this.#width && height === this.#height) return;
    for (const job of this.#jobs.values()) {
      job.handle.cancel();
      for (const resolve of job.waiters) resolve(null);
    }
    this.#jobs.clear();
    // Old-size rasters of the pages on screen stay as provisional stand-ins.
    const keep = new Set([...this.#visible, ...this.#ahead]);
    for (const [index, entry] of this.#cache) {
      if (!keep.has(index)) {
        releaseEntry(entry);
        continue;
      }
      this.#dropStale(index);
      this.#stale.set(index, entry);
    }
    this.#cache.clear();
    this.#width = width;
    this.#height = height;
  }

  peek(index) {
    const entry = this.#cache.get(index);
    if (!entry) return null;
    this.#cache.delete(index);
    this.#cache.set(index, entry);
    return entry;
  }

  fallback(index) {
    return this.#stale.get(index)?.canvas ?? this.#fallback?.(index) ?? null;
  }

  // The book's own canvas of a page (current or previous size), for another view to copy.
  standIn(index) {
    return this.#cache.get(index)?.canvas ?? this.#stale.get(index)?.canvas ?? null;
  }

  // Resolves with the raster entry, or null when no document is attached or the size changed.
  get(index) {
    const cached = this.peek(index);
    if (cached) return Promise.resolve(cached);
    if (!this.#source || !this.#width) return Promise.resolve(null);
    const job = this.#job(index, 0);
    job.urgent = true;
    if (job.priority !== 0) {
      job.priority = 0;
      job.handle.setPriority(0);
    }
    return new Promise(resolve => job.waiters.push(resolve));
  }

  // Visible pages render first, then neighbours in the given order; other book jobs are cancelled.
  want(visible, ahead) {
    this.#visible = visible;
    this.#ahead = ahead;
    for (const index of [...this.#stale.keys()]) {
      if (!visible.includes(index) && !ahead.includes(index)) this.#dropStale(index);
    }
    this.#sync();
  }

  hold() {
    this.#source?.hold();
  }

  release() {
    this.#source?.release();
  }

  // Horizontal mip levels stop fine text from aliasing as a turning sheet narrows.
  levels(entry) {
    if (!entry.levels) {
      const levels = [entry.canvas];
      while (levels.at(-1).width > 1) {
        const source = levels.at(-1);
        const level = document.createElement('canvas');
        level.width = Math.max(1, Math.ceil(source.width / 2));
        level.height = source.height;
        const painter = level.getContext('2d');
        painter.imageSmoothingEnabled = true;
        painter.imageSmoothingQuality = 'high';
        painter.drawImage(source, 0, 0, level.width, level.height);
        levels.push(level);
      }
      entry.levels = levels;
    }
    return entry.levels;
  }

  #dropStale(index) {
    const entry = this.#stale.get(index);
    if (!entry) return;
    this.#stale.delete(index);
    releaseEntry(entry);
  }

  #sync() {
    if (!this.#source || !this.#width) return;
    const wanted = new Map();
    for (const index of this.#visible) wanted.set(index, 0);
    this.#ahead.forEach((index, rank) => {
      if (!wanted.has(index)) wanted.set(index, BACKGROUND + rank);
    });
    for (const [index, job] of this.#jobs) {
      if (wanted.has(index) || job.urgent) continue;
      job.handle.cancel();
      this.#jobs.delete(index);
    }
    for (const [index, base] of wanted) {
      if (this.#cache.has(index)) continue;
      const job = this.#jobs.get(index);
      if (!job) {
        this.#job(index, base);
      } else if (!job.urgent && job.base !== base) {
        job.base = base;
        job.priority = base + this.#offset;
        job.handle.setPriority(job.priority);
      }
    }
    this.#source.retain('book', [...wanted.keys()]);
  }

  #job(index, base) {
    const existing = this.#jobs.get(index);
    if (existing) return existing;
    const width = this.#width, height = this.#height;
    const priority = base + this.#offset;
    const job = {handle: this.#source.request(index, width, height, priority), base, priority, urgent: false, waiters: []};
    this.#jobs.set(index, job);
    job.handle.promise.then(canvas => this.#finish(index, job, canvas), error => {
      if (error?.name === 'AbortError') return;
      console.warn(`Page ${index + 1} could not be rendered.`, error);
      this.#finish(index, job, errorPage(width, height, index));
    });
    return job;
  }

  #finish(index, job, canvas) {
    if (this.#jobs.get(index) !== job) {
      releaseCanvas(canvas);
      return;
    }
    this.#jobs.delete(index);
    const entry = {index, canvas, levels: null};
    this.#cache.set(index, entry);
    this.#dropStale(index);
    this.#evict();
    for (const resolve of job.waiters) resolve(entry);
    for (const listener of this.#listeners) listener(index, entry);
  }

  // Entries on screen or about to be (including a turning sheet's two faces) are never dropped.
  #evict() {
    const keep = new Set([...this.#visible, ...this.#ahead]);
    const cost = entry => entry.canvas.width * entry.canvas.height * (entry.levels ? 2 : 1);
    let pixels = 0;
    for (const entry of this.#cache.values()) pixels += cost(entry);
    for (const [index, entry] of this.#cache) {
      if (this.#cache.size <= this.maxEntries && pixels <= this.maxPixels) break;
      if (keep.has(index)) continue;
      this.#cache.delete(index);
      pixels -= cost(entry);
      releaseEntry(entry);
    }
  }
}

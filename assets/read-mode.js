// Full-screen reading view: every page in one column, rendered lazily at
// fit-width × zoom × devicePixelRatio so text stays crisp at every step. Renders
// larger than MAX_CANVAS_SIDE are split into tiles rather than scaled down.
import {CONSTRAINED, MAX_CANVAS_SIDE, releaseCanvas} from './pdf-pages.js';

const ZOOMS = [1, 1.25, 1.5, 2];
const MAX_FIT_WIDTH = 1100;
// Canvas pixels: all pages together, one page, and what stays after closing.
const PIXEL_BUDGET = CONSTRAINED ? 16e6 : 50e6;
const PAGE_PIXELS = CONSTRAINED ? 12e6 : Infinity;
const KEEP_PIXELS = CONSTRAINED ? 4e6 : 12e6;
const SNAPSHOT_WIDTH = 2048;

export function createReadMode({dialog, source, pageCount, fallbackRatio, standIn = () => null, onClose, onOpenChange = () => {}}) {
  const scroller = dialog.querySelector('.rm-scroll');
  const column = dialog.querySelector('.rm-column');
  const indicator = dialog.querySelector('.rm-indicator');
  const zoomLabel = dialog.querySelector('.rm-zoom');
  const zoomOut = dialog.querySelector('[data-zoom-out]');
  const zoomIn = dialog.querySelector('[data-zoom-in]');
  const closeButton = dialog.querySelector('[data-read-close]');
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const pages = [];
  let sizes = [];
  let zoom = 0;
  let mostInView = 0;
  let scrollFrame = 0;
  let scheduleFrame = 0;
  let opening = false;
  let closing = false;
  let session = 0;
  let closedSession = 0;
  let fitWidth = 0;

  function build() {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) pages[Number(entry.target.dataset.index)].near = entry.isIntersecting;
      requestSchedule();
    }, {root: scroller, rootMargin: '100% 0px'});
    for (let index = 0; index < pageCount; index++) {
      const element = document.createElement('div');
      element.className = 'rm-page';
      element.dataset.index = index;
      element.setAttribute('role', 'group');
      element.setAttribute('aria-label', `Page ${index + 1}`);
      column.append(element);
      pages.push({
        index, element, width: 0, height: 0, near: false,
        // What is shown: a sheet of tile canvases, its render size and key.
        sheet: null, tiles: [], renderWidth: 0, renderHeight: 0, key: '', standIn: false, snapshot: null,
        job: null, failed: '', text: null,
      });
      observer.observe(element);
    }
    new ResizeObserver(() => {
      if (dialog.open && !closing && scroller.clientWidth !== fitWidth) resize(() => {});
    }).observe(scroller);
  }

  function setDisabled(button, disabled) {
    button.setAttribute('aria-disabled', String(disabled));
  }

  function applySizes() {
    fitWidth = scroller.clientWidth;
    const gutter = fitWidth <= 600 ? 8 : 24;
    column.style.setProperty('--gutter', gutter + 'px');
    const base = Math.max(160, Math.min(MAX_FIT_WIDTH, fitWidth - gutter * 2));
    // Whole device pixels, so a canvas at devicePixelRatio maps 1:1 onto the screen.
    const density = window.devicePixelRatio || 1;
    // Floor, so a Fit page never overflows the column by a fraction of a pixel.
    const deviceWidth = Math.floor(base * ZOOMS[zoom] * density);
    for (const page of pages) {
      const size = sizes[page.index];
      page.width = deviceWidth / density;
      page.height = Math.round(deviceWidth / (size ? size[0] / size[1] : fallbackRatio)) / density;
      page.element.style.width = page.width + 'px';
      page.element.style.height = page.height + 'px';
      // CSS px per PDF point: the text layer scales with it.
      if (size) page.element.style.setProperty('--scale-factor', page.width / size[0]);
    }
    zoomLabel.textContent = zoom === 0 ? 'Fit' : Math.round(ZOOMS[zoom] * 100) + '%';
    setDisabled(zoomOut, zoom === 0);
    setDisabled(zoomIn, zoom === ZOOMS.length - 1);
  }

  // Re-lays out the column while keeping the point at the viewport centre in place.
  function resize(change) {
    const anchor = pages[mostInView];
    const rect = anchor.element.getBoundingClientRect();
    const view = scroller.getBoundingClientRect();
    const fy = (view.top + view.height / 2 - rect.top) / rect.height;
    const fx = (view.left + view.width / 2 - rect.left) / rect.width;
    change();
    applySizes();
    scroller.scrollTop = anchor.element.offsetTop + fy * anchor.height - scroller.clientHeight / 2;
    scroller.scrollLeft = anchor.element.offsetLeft + fx * anchor.width - scroller.clientWidth / 2;
    requestSchedule();
  }

  function setZoom(value) {
    const target = Math.max(0, Math.min(ZOOMS.length - 1, value));
    if (target !== zoom) resize(() => { zoom = target; });
  }

  // Device pixels for the whole page: full density, except that a constrained
  // device draws a very large page with at most PAGE_PIXELS.
  function renderSize(page) {
    const density = window.devicePixelRatio || 1;
    const scale = density * Math.min(1, Math.sqrt(PAGE_PIXELS / (page.width * page.height * density * density)));
    return [Math.max(1, Math.round(page.width * scale)), Math.max(1, Math.round(page.height * scale))];
  }

  // The fewest equal tiles that keep every canvas within MAX_CANVAS_SIDE.
  function regions(width, height) {
    const columns = Math.ceil(width / MAX_CANVAS_SIDE), rows = Math.ceil(height / MAX_CANVAS_SIDE);
    const tileWidth = Math.ceil(width / columns), tileHeight = Math.ceil(height / rows);
    const out = [];
    for (let y = 0; y < height; y += tileHeight) {
      for (let x = 0; x < width; x += tileWidth) out.push({x, y, width: Math.min(tileWidth, width - x), height: Math.min(tileHeight, height - y)});
    }
    return out;
  }

  // One job per page; its tiles arrive together.
  function renderPage(index, width, height, priority) {
    const parts = regions(width, height).map(region => ({region, handle: source.request(index, width, height, priority, region)}));
    const promise = Promise.allSettled(parts.map(part => part.handle.promise)).then(results => {
      const failure = results.find(result => result.status === 'rejected');
      if (failure) {
        for (const result of results) if (result.status === 'fulfilled') releaseCanvas(result.value);
        throw failure.reason;
      }
      return results.map((result, i) => ({canvas: result.value, ...parts[i].region}));
    });
    return {
      promise,
      setPriority: value => parts.forEach(part => part.handle.setPriority(value)),
      cancel: () => parts.forEach(part => part.handle.cancel()),
    };
  }

  function requestSchedule() {
    if (!scheduleFrame) scheduleFrame = requestAnimationFrame(schedule);
  }

  function cancelJob(page) {
    if (!page.job) return;
    page.job.handle.cancel();
    page.job = null;
  }

  // Near pages render, closest to the page in view first; far pages are cancelled.
  function schedule() {
    scheduleFrame = 0;
    if (!dialog.open || closing) return;
    const near = [];
    for (const page of pages) {
      if (!page.near) {
        cancelJob(page);
        removeText(page);
        continue;
      }
      near.push(page.index);
      showStandIn(page);
      const [width, height] = renderSize(page);
      const key = width + 'x' + height;
      // A render for another size is no longer wanted (e.g. zooming in and straight back out).
      if (page.job && page.job.key !== key) cancelJob(page);
      if (page.key === key) {
        ensureText(page);
        continue;
      }
      if (page.failed === key) continue;
      const priority = Math.min(9, Math.abs(page.index - mostInView));
      if (page.job) {
        page.job.handle.setPriority(priority);
        continue;
      }
      const handle = renderPage(page.index, width, height, priority);
      const job = page.job = {key, handle};
      handle.promise.then(tiles => {
        if (page.job !== job) {
          for (const tile of tiles) releaseCanvas(tile.canvas);
          return;
        }
        page.job = null;
        show(page, tiles, width, height, key);
        if (page.near) ensureText(page);
        evict();
      }, error => {
        if (page.job !== job) return;
        page.job = null;
        if (error?.name !== 'AbortError') page.failed = key;
      });
    }
    source.retain('read', near);
  }

  function pixels(page) {
    let total = 0;
    for (const tile of page.tiles) total += tile.canvas.width * tile.canvas.height;
    return total;
  }

  function releaseSheet(page) {
    for (const tile of page.tiles) releaseCanvas(tile.canvas);
    if (page.snapshot) releaseCanvas(page.snapshot);
    page.snapshot = null;
  }

  // Puts a page's tiles on screen in one paint. A stand-in stays blurred; the
  // first sharp render, or one replacing a stand-in, clears the blur gradually.
  function show(page, tiles, width, height, key, isStandIn = false) {
    const sheet = document.createElement('div');
    sheet.className = 'rm-sheet';
    sheet.setAttribute('aria-hidden', 'true');
    for (const tile of tiles) {
      tile.canvas.className = 'rm-canvas';
      Object.assign(tile.canvas.style, {
        left: tile.x / width * 100 + '%',
        top: tile.y / height * 100 + '%',
        width: tile.width / width * 100 + '%',
        height: tile.height / height * 100 + '%',
      });
      sheet.append(tile.canvas);
    }
    const blurred = !reducedMotion.matches && (isStandIn || !page.sheet || page.standIn);
    if (blurred) sheet.classList.add('is-blurred');
    if (page.sheet) {
      // A new zoom level replaces the stretched sheet in place.
      page.sheet.replaceWith(sheet);
      releaseSheet(page);
    } else {
      page.element.prepend(sheet);
    }
    Object.assign(page, {sheet, tiles, renderWidth: width, renderHeight: height, key, standIn: isStandIn});
    if (blurred && !isStandIn) requestAnimationFrame(() => requestAnimationFrame(() => sheet.classList.remove('is-blurred')));
  }

  // Until its own render arrives, a page shows a copy of the book's raster.
  function showStandIn(page) {
    if (!page || page.sheet) return;
    const raster = standIn(page.index);
    if (!raster?.width) return;
    // The book frames every page in the paper's nominal shape; crop to this page's own.
    const ratio = page.width / page.height;
    const width = Math.min(raster.width, raster.height * ratio);
    const height = Math.min(raster.height, raster.width / ratio);
    const canvas = document.createElement('canvas');
    canvas.width = Math.max(1, Math.round(width));
    canvas.height = Math.max(1, Math.round(height));
    canvas.getContext('2d', {alpha: false}).drawImage(raster, (raster.width - width) / 2, (raster.height - height) / 2, width, height, 0, 0, canvas.width, canvas.height);
    show(page, [{canvas, x: 0, y: 0, width: canvas.width, height: canvas.height}], canvas.width, canvas.height, '', true);
  }

  // Invisible text over the page, so it can be selected, searched and read aloud.
  function ensureText(page) {
    if (page.text || !sizes[page.index]) return;
    const layer = document.createElement('div');
    layer.className = 'textLayer';
    page.element.append(layer);
    const task = source.renderText(page.index, layer);
    task.promise.catch(() => {});
    page.text = {layer, task};
  }

  function removeText(page) {
    if (!page.text) return;
    page.text.task.cancel();
    page.text.layer.remove();
    page.text = null;
  }

  function drop(page) {
    removeText(page);
    if (!page.sheet) return;
    page.sheet.remove();
    releaseSheet(page);
    Object.assign(page, {sheet: null, tiles: [], key: '', standIn: false});
  }

  // Keeps memory bounded: far pages give their canvases back first.
  function evict() {
    let total = 0;
    for (const page of pages) total += pixels(page);
    if (total <= PIXEL_BUDGET) return;
    const far = pages.filter(page => page.sheet && !page.near).sort((a, b) => Math.abs(b.index - mostInView) - Math.abs(a.index - mostInView));
    for (const page of far) {
      if (total <= PIXEL_BUDGET) break;
      total -= pixels(page);
      drop(page);
    }
  }

  // One canvas of a sharp page for the book to copy (tiles composited if needed).
  function canvasFor(index) {
    const page = pages[index];
    if (!page?.sheet || page.standIn) return null;
    if (page.tiles.length === 1) return page.tiles[0].canvas;
    if (!page.snapshot) {
      const scale = Math.min(1, SNAPSHOT_WIDTH / page.renderWidth);
      const canvas = page.snapshot = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(page.renderWidth * scale));
      canvas.height = Math.max(1, Math.round(page.renderHeight * scale));
      const context = canvas.getContext('2d', {alpha: false});
      context.imageSmoothingQuality = 'high';
      for (const tile of page.tiles) context.drawImage(tile.canvas, tile.x * scale, tile.y * scale, tile.width * scale, tile.height * scale);
    }
    return page.snapshot;
  }

  function measure() {
    const view = scroller.getBoundingClientRect();
    let best = mostInView, bestVisible = 0;
    for (const page of pages) {
      const rect = page.element.getBoundingClientRect();
      if (rect.bottom <= view.top) continue;
      if (rect.top >= view.bottom) break;
      const visible = Math.min(rect.bottom, view.bottom) - Math.max(rect.top, view.top);
      if (visible > bestVisible) {
        bestVisible = visible;
        best = page.index;
      }
    }
    return best;
  }

  function updateIndicator() {
    indicator.textContent = `${mostInView + 1} / ${pageCount}`;
  }

  scroller.addEventListener('scroll', () => {
    if (scrollFrame) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = 0;
      const index = measure();
      if (index === mostInView) return;
      mostInView = index;
      updateIndicator();
      requestSchedule();
    });
  }, {passive: true});

  async function open(startIndex = 0) {
    if (dialog.open || opening || closing) return;
    opening = true;
    session++;
    if (!pages.length) build();
    try {
      sizes = await source.pageSizes();
    } catch {
      sizes = [];
    }
    opening = false;
    closing = false;
    dialog.showModal();
    onOpenChange(true);
    mostInView = Math.max(0, Math.min(pageCount - 1, startIndex));
    applySizes();
    // Open with the page's top edge just below the bar, one page gap above it.
    const gap = parseFloat(getComputedStyle(column).rowGap) || 0;
    scroller.scrollTop = mostInView === 0 ? 0 : pages[mostInView].element.offsetTop - gap;
    scroller.scrollLeft = (scroller.scrollWidth - scroller.clientWidth) / 2;
    updateIndicator();
    // The pages just read in the book appear at once, blurred, until their sharp renders arrive.
    for (let index = mostInView - 1; index <= mostInView + 2; index++) showStandIn(pages[index]);
    requestSchedule();
    scroller.focus({preventScroll: true});
    if (!reducedMotion.matches) dialog.animate([{opacity: 0}, {opacity: 1}], {duration: 180, easing: 'ease-out'});
  }

  // Runs once per close, however the dialog is closed.
  function beginClose() {
    if (closing) return;
    closing = true;
    if (dialog.open) mostInView = measure();
    for (const page of pages) cancelJob(page);
    // The book moves to this page first, so it is already in place underneath.
    onClose(mostInView);
  }

  function finishClose() {
    if (closedSession === session) return;
    closedSession = session;
    // A little sharp rendering stays around the reading position for a quick return.
    const keep = new Set();
    let kept = 0;
    for (const page of [0, 1, -1, 2].map(offset => pages[mostInView + offset])) {
      if (!page?.sheet || page.standIn || kept + pixels(page) > KEEP_PIXELS) continue;
      kept += pixels(page);
      keep.add(page);
    }
    for (const page of pages) {
      removeText(page);
      if (!keep.has(page)) drop(page);
    }
    source.retain('read', []);
    onOpenChange(false);
    closing = false;
  }

  async function close() {
    if (!dialog.open || closing) return;
    const current = session;
    beginClose();
    if (!reducedMotion.matches) {
      const fade = dialog.animate([{opacity: 1}, {opacity: 0}], {duration: 140, easing: 'ease-in', fill: 'forwards'});
      // A throttled (hidden) page may not run the fade; close regardless.
      await Promise.race([fade.finished.catch(() => {}), new Promise(resolve => setTimeout(resolve, 400))]);
      fade.cancel();
    }
    if (current === session && dialog.open) {
      dialog.close();
      // The close event only arrives with the next frame; finish now.
      finishClose();
    }
  }

  zoomOut.addEventListener('click', () => { if (zoomOut.getAttribute('aria-disabled') !== 'true') setZoom(zoom - 1); });
  zoomIn.addEventListener('click', () => { if (zoomIn.getAttribute('aria-disabled') !== 'true') setZoom(zoom + 1); });
  closeButton.addEventListener('click', () => void close());
  dialog.addEventListener('cancel', event => {
    event.preventDefault();
    void close();
  });
  // Esc without recent user activation closes the dialog without a cancelable event.
  dialog.addEventListener('close', () => {
    if (closedSession === session) return;
    beginClose();
    finishClose();
  });

  return {
    open,
    canvasFor,
  };
}

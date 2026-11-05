// Page-turn engine ported from the portfolio's zine reader (zine-reader.js).
// Pages are PDF.js rasters supplied by BookRasters. The turning sheet always
// uses the composited canvas ("stable raster") path, because dense text
// aliases badly on a narrowing CSS 3D sheet.
import {MAX_CANVAS_SIDE, releaseCanvas} from './pdf-pages.js';

export function createFlipbook(options) {
  const {reader, stage, book, previous, next, counter, rasters, ratio, coverSrc = null, onChange = () => {}} = options;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let count = Math.max(1, Math.trunc(options.pageCount) || 1);
  let enabled = false;
  let current = clampIndex(options.startIndex);
  let orientation = 'landscape';
  let pageWidth = 0;
  let pageHeight = 0;
  let rasterWidth = 0;
  let rasterHeight = 0;
  let rasterFrame = 0;
  let lastSize = '';
  let version = 0;
  let active = null;
  let queuedDirection = 0;
  let dragging = null;
  let dragFrame = 0;
  let cover = null;
  let status = '';
  let bookShown = false;

  function clampIndex(index) {
    return Math.max(0, Math.min(count - 1, Math.trunc(index) || 0));
  }

  // The book appears once its first page has something to show (or after
  // 800 ms), so its first visible frame is the blurred cover, not an empty sheet.
  book.style.visibility = 'hidden';
  const bookTimer = setTimeout(showBook, 800);
  function showBook() {
    if (bookShown) return;
    bookShown = true;
    clearTimeout(bookTimer);
    book.style.visibility = '';
  }

  // The museum's cover thumbnail stands in for page 1 while the PDF loads. It
  // stays blurred until the real render replaces it, which then sharpens.
  if (coverSrc) {
    const image = new Image();
    image.decoding = 'async';
    image.src = coverSrc;
    image.decode().then(() => {
      cover = image;
      for (const page of book.querySelectorAll('.page[data-page-index="0"][data-state="blank"]')) draw(page, cover, 'cover');
    }, () => {});
  }

  rasters.onRaster((index, entry) => {
    for (const page of book.querySelectorAll(`.page[data-page-index="${index}"]`)) {
      if (page.dataset.state !== 'sharp') draw(page, entry.canvas, 'sharp');
    }
  });

  function stopRaster() {
    cancelAnimationFrame(rasterFrame);
    rasterFrame = 0;
  }

  // Node canvases are display copies: dropped nodes give their memory back at
  // once (WebKit otherwise counts it against the tab until garbage collection).
  function discard(nodes, keep = []) {
    for (const node of nodes) {
      if (!node || keep.includes(node)) continue;
      for (const canvas of node.querySelectorAll('canvas')) releaseCanvas(canvas);
    }
  }
  // Drops the nodes a turn built that are not going to be shown.
  function abandon(state, keep = []) {
    if (state) discard([...state.layers, ...state.destinationPages, ...(state.originPages || [])], keep);
  }

  // Draws a raster (or a stand-in) into the page's canvas, filling the frame
  // like object-fit: cover. Rasters already match the frame exactly.
  function draw(page, source, state) {
    let canvas = page.querySelector('canvas');
    if (!canvas) {
      canvas = document.createElement('canvas');
      page.append(canvas);
    }
    if (canvas.width !== rasterWidth || canvas.height !== rasterHeight) {
      canvas.width = rasterWidth;
      canvas.height = rasterHeight;
    }
    const context = canvas.getContext('2d', {alpha: false});
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    const width = source.naturalWidth || source.width;
    const height = source.naturalHeight || source.height;
    const scale = Math.max(rasterWidth / width, rasterHeight / height);
    const cropWidth = rasterWidth / scale, cropHeight = rasterHeight / scale;
    context.drawImage(source, (width - cropWidth) / 2, (height - cropHeight) / 2, cropWidth, cropHeight, 0, 0, rasterWidth, rasterHeight);
    const waiting = page.dataset.state === 'blank' || page.dataset.state === 'cover';
    page.dataset.state = state;
    if (state === 'cover') page.classList.add('is-blurred');
    else if (waiting) reveal(page);
    if (page.isConnected) showBook();
  }

  // The blur is painted before the content arrives; only transition toward sharp.
  // A page that was never on screen has nothing to reveal.
  function reveal(page) {
    if (!page.isConnected) {
      page.classList.remove('is-blurred');
      return;
    }
    requestAnimationFrame(() => requestAnimationFrame(() => page.classList.remove('is-blurred')));
  }

  function paint(page, entry = rasters.peek(Number(page.dataset.pageIndex))) {
    const index = Number(page.dataset.pageIndex);
    if (entry) return draw(page, entry.canvas, 'sharp');
    const fallback = rasters.fallback(index);
    if (fallback) return draw(page, fallback, 'provisional');
    if (index === 0 && cover) return draw(page, cover, 'cover');
    page.dataset.state = 'blank';
    page.classList.add('is-blurred');
  }

  async function pageRaster(index) {
    const entry = await rasters.get(index);
    return entry && {canvas: entry.canvas, levels: rasters.levels(entry)};
  }

  // Every page a turn can show has a current raster before the sheet moves.
  async function preparePages(nodes, extra, token) {
    const indices = [...new Set([...nodes.map(page => Number(page.dataset.pageIndex)), ...extra])];
    const entries = await Promise.all(indices.map(index => rasters.get(index)));
    if (token !== version || entries.includes(null)) return false;
    const byIndex = new Map(indices.map((index, i) => [index, entries[i]]));
    for (const page of nodes) paint(page, byIndex.get(Number(page.dataset.pageIndex)));
    return true;
  }

  function slots(index) {
    if (orientation === 'portrait') return [index];
    return index === 0 ? [null, 0] : [index, index + 1 < count ? index + 1 : null];
  }
  function visiblePages(index = current) { return slots(index).filter(i => i !== null); }
  function offset(index) {
    if (orientation === 'portrait') return 0;
    return index === 0 ? -pageWidth / 2 : index === count - 1 ? pageWidth / 2 : 0;
  }
  function node(index, side) {
    if (index === null || index === undefined) return null;
    const page = document.createElement('div');
    page.className = 'page';
    page.dataset.pageIndex = index;
    page.dataset.side = side;
    page.setAttribute('role', 'img');
    page.setAttribute('aria-label', 'Page ' + (index + 1));
    page.setAttribute('aria-hidden', 'true');
    paint(page);
    return page;
  }
  function spread(index) {
    return slots(index).map((i, side) => node(i, orientation === 'portrait' ? 'single' : side ? 'right' : 'left')).filter(Boolean);
  }
  function canTurn(direction) {
    if (!enabled) return false;
    return direction > 0 ? visiblePages().at(-1) < count - 1 : current > 0;
  }
  function setDisabled(button, disabled) {
    button.setAttribute('aria-disabled', String(disabled));
  }
  function writeCounter() {
    counter.textContent = status || visiblePages().map(i => i + 1).join('–') + ' / ' + count;
    counter.classList.toggle('is-status', Boolean(status));
  }
  function update() {
    const visible = visiblePages();
    writeCounter();
    setDisabled(previous, !canTurn(-1));
    setDisabled(next, !canTurn(1));
    reader.dataset.page = current + 1;
    reader.dataset.orientation = orientation;
    reader.dataset.turnState = 'read';
    // Only the book is busy, and only while the paper is still loading or turning.
    if (enabled) book.removeAttribute('aria-busy');
    else book.setAttribute('aria-busy', 'true');
    for (const page of book.querySelectorAll('.page')) page.setAttribute('aria-hidden', 'false');
    // Render the visible pages first, then current-2 .. current+4, forward first.
    const ahead = [];
    for (let i = visible.at(-1) + 1; i <= Math.min(count - 1, current + 4); i++) ahead.push(i);
    for (let i = current - 1; i >= Math.max(0, current - 2); i--) ahead.push(i);
    rasters.want(visible, ahead);
    onChange(current, visible);
  }
  function commit(index, destination = spread(index)) {
    stopRaster();
    current = index;
    // Replace the already-painted destination and finished leaf in the same paint.
    discard([...book.children], destination);
    abandon(active, destination);
    book.replaceChildren(...destination);
    book.style.transform = 'translate3d(' + offset(index) + 'px,0,0)';
    active?.animations?.forEach(animation => animation.cancel());
    active = null;
    rasters.release();
    if (destination.some(page => page.dataset.state !== 'blank')) showBook();
    update();
  }
  function drainQueue() {
    const direction = queuedDirection;
    queuedDirection = 0;
    if (direction) queueMicrotask(() => void turn(direction));
  }
  function destination(direction) {
    if (orientation === 'portrait') return current + direction;
    return direction > 0 ? current === 0 ? 1 : current + 2 : current <= 1 ? 0 : current - 2;
  }
  function desktopFrames(left, right, direction, from = 0, to = 1) {
    const frames = [];
    // Round the visible-width crossover near the spine, avoiding a sudden
    // change in lateral speed when the turning sheet passes through 90 degrees.
    const shoulder = pageWidth * .1;
    const positiveExtent = value => value <= -shoulder ? 0 : value >= shoulder ? value : (value + shoulder) ** 2 / (4 * shoulder);
    for (let i = 0; i <= 60; i++) {
      const frameOffset = i / 60;
      const progress = from + (to - from) * frameOffset;
      const edge = pageWidth * Math.cos(Math.PI * (direction > 0 ? progress : 1 - progress));
      const min = left === null ? -positiveExtent(-edge) : Math.min(-pageWidth, edge);
      const max = right === null ? positiveExtent(edge) : Math.max(pageWidth, edge);
      // Only a boundary spread moves. Interior pages stay anchored.
      const shift = left !== null && right !== null ? 0 : -(min + max) / 2;
      frames.push({transform: 'translate3d(' + shift + 'px,0,0)', offset: frameOffset});
    }
    return frames;
  }
  function animateLayers(scene, timing, from = 0, to = 1) {
    // The leaf's constant animation is only the clock for the canvas sheet;
    // the book follows the unfolding page edge with the same timing.
    const animations = scene.leaf
      ? [scene.leaf.animate([{opacity: 1}, {opacity: 1}], timing), book.animate(desktopFrames(scene.left, scene.right, scene.direction, from, to), timing)]
      : [scene.layers[1].animate([{opacity: from}, {opacity: to}], timing)];
    const start = document.timeline.currentTime;
    animations.forEach(animation => { animation.startTime = start; });
    if (scene.canvas) {
      stopRaster();
      let lastProgress = -1;
      const paintFrame = () => {
        const sampled = animations[0].effect.getComputedTiming().progress;
        const progress = from + (to - from) * (sampled ?? 0);
        if (progress !== lastProgress) {
          drawRaster(scene, progress);
          lastProgress = progress;
        }
        rasterFrame = requestAnimationFrame(paintFrame);
      };
      paintFrame();
    }
    return animations;
  }
  function drawRaster(scene, progress) {
    const context = scene.canvas.getContext('2d');
    const angle = Math.PI * (scene.direction > 0 ? progress : 1 - progress);
    const edge = Math.cos(angle) * scene.width;
    const source = edge >= 0 ? scene.front : scene.back;
    context.clearRect(0, 0, scene.canvas.width, scene.canvas.height);
    if (Math.abs(edge) > .01) {
      const width = Math.abs(edge);
      let sampled = source.canvas;
      for (const level of source.levels) {
        if (level.width < width) break;
        sampled = level;
      }
      context.drawImage(sampled, 0, 0, sampled.width, scene.height, edge >= 0 ? scene.width : scene.width + edge, 0, width, scene.height);
    }
    scene.leaf.dataset.progress = progress.toFixed(5);
  }
  async function completeAnimation(token, index, destinationPages, animations) {
    await Promise.all(animations.map(animation => animation.finished.catch(() => {})));
    if (token !== version) return;
    commit(index, destinationPages);
    drainQueue();
  }
  async function turn(direction, gesture = null) {
    if (reader.dataset.ready !== 'true' || !enabled) return;
    // One more turn may wait behind the running one, if its button was enabled.
    if (active) { if (!gesture && canTurn(direction)) queuedDirection = direction; return; }
    if (!canTurn(direction)) return;
    const target = destination(direction);
    const token = ++version;
    const destinationPages = spread(target);
    const originPages = gesture ? spread(current) : null;
    active = {target, origin: current, destinationPages, originPages, layers: [], animations: [], phase: 'preparing', gesture, token};
    reader.dataset.turnState = 'preparing';
    book.setAttribute('aria-busy', 'true');

    let layers, leaf = null, frames = null, left = null, right = null, sheet = null;
    if (orientation === 'landscape') {
      const from = slots(current), to = slots(target);
      left = direction > 0 ? from[0] : to[0];
      right = direction > 0 ? to[1] : from[1];
      sheet = direction > 0 ? from[1] : to[1];
      leaf = document.createElement('div');
      leaf.className = 'leaf';
      leaf.setAttribute('aria-hidden', 'true');
      layers = [node(left, 'left'), node(right, 'right'), leaf].filter(Boolean);
      frames = desktopFrames(left, right, direction);
    } else {
      // A small display keeps a single page centered throughout the transition.
      const outgoing = node(current, 'single');
      const incoming = node(target, 'single');
      incoming.style.opacity = '0';
      layers = [outgoing, incoming];
    }
    active.layers = layers;
    const pageNodes = [...layers.filter(layer => layer !== leaf), ...destinationPages, ...(originPages || [])];
    const ready = await preparePages(pageNodes, leaf ? [sheet, sheet + 1] : [], token);
    if (token !== version) return;
    if (!ready) {
      abandon(active); active = null; queuedDirection = 0; update(); return;
    }
    if (gesture?.released && !gesture.complete) {
      abandon(active); active = null; update(); drainQueue(); return;
    }
    if (reducedMotion.matches && !gesture) { commit(target, destinationPages); drainQueue(); return; }
    const scene = {layers, leaf, left, right, direction};
    if (leaf) {
      const [front, back] = await Promise.all([pageRaster(sheet), pageRaster(sheet + 1)]);
      if (token !== version) return;
      if (!front || !back) {
        abandon(active); active = null; queuedDirection = 0; update(); return;
      }
      // One filtered raster plane draws the turning sheet.
      const canvas = document.createElement('canvas');
      canvas.className = 'turn-canvas';
      canvas.width = rasterWidth * 2;
      canvas.height = rasterHeight;
      const context = canvas.getContext('2d');
      context.imageSmoothingEnabled = true;
      context.imageSmoothingQuality = 'high';
      leaf.append(canvas);
      Object.assign(scene, {canvas, front, back, width: rasterWidth, height: rasterHeight});
      drawRaster(scene, 0);
    }
    // Background renders wait so the main thread stays free while the sheet moves.
    rasters.hold();
    discard([...book.children], layers);
    book.replaceChildren(...layers);
    if (frames) book.style.transform = frames[0].transform;
    // Warm these exact layers before the compositor starts moving them.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    if (token !== version) return;
    const duration = orientation === 'landscape' ? 500 : 180;
    const timing = {duration, easing: gesture ? 'linear' : 'cubic-bezier(.4,0,.2,1)', fill: 'forwards'};
    const animations = animateLayers(scene, timing);
    Object.assign(active, {animations, scene, duration, phase: gesture ? 'dragging' : 'flipping'});
    if (gesture) {
      animations.forEach(animation => { animation.pause(); animation.currentTime = gesture.progress * duration; });
      reader.dataset.turnState = 'dragging';
      if (gesture.released) settleDrag(gesture);
      return;
    }
    reader.dataset.turnState = 'flipping';
    await completeAnimation(token, target, destinationPages, animations);
  }

  function scrubDrag() {
    dragFrame = 0;
    if (active?.phase !== 'dragging') return;
    active.animations.forEach(animation => { animation.currentTime = active.gesture.progress * active.duration; });
    if (active.scene.canvas) drawRaster(active.scene, active.gesture.progress);
  }
  function settleDrag(gesture) {
    if (active?.gesture !== gesture || active.phase !== 'dragging') return;
    cancelAnimationFrame(dragFrame);
    dragFrame = 0;
    const state = active;
    const goal = gesture.complete ? 1 : 0;
    const index = gesture.complete ? state.target : state.origin;
    const destinationPages = gesture.complete ? state.destinationPages : state.originPages;
    if (reducedMotion.matches || Math.abs(goal - gesture.progress) < .002) { commit(index, destinationPages); drainQueue(); return; }
    state.animations.forEach(animation => animation.cancel());
    const duration = Math.max(140, Math.min(400, state.duration * Math.abs(goal - gesture.progress)));
    const animations = animateLayers(state.scene, {duration, easing: 'cubic-bezier(.2,.65,.3,1)', fill: 'forwards'}, gesture.progress, goal);
    Object.assign(state, {animations, phase: 'flipping', settleTarget: index});
    reader.dataset.turnState = 'flipping';
    void completeAnimation(state.token, index, destinationPages, animations);
  }
  function releaseDrag(cancelled = false) {
    if (!dragging) return;
    const gesture = dragging;
    dragging = null;
    gesture.released = true;
    gesture.complete = !cancelled && gesture.progress >= .22;
    stage.classList.remove('is-dragging');
    if (stage.hasPointerCapture(gesture.pointerId)) stage.releasePointerCapture(gesture.pointerId);
    if (!gesture.triggered) return;
    if (!gesture.animated) {
      if (gesture.complete) void turn(gesture.direction);
    } else settleDrag(gesture);
  }
  // Anchored mouse dragging: the sheet follows the pointer; the book stays put.
  stage.addEventListener('pointerdown', event => {
    if (event.pointerType !== 'mouse' || event.button !== 0 || active || dragging || reader.dataset.ready !== 'true' || !enabled) return;
    const page = [...book.querySelectorAll('.page')].find(page => {
      const rect = page.getBoundingClientRect();
      return event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    });
    if (!page) return;
    const direction = orientation === 'portrait' ? 0 : page.dataset.side === 'right' ? 1 : -1;
    if (direction && !canTurn(direction)) return;
    dragging = {pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, travel: pageWidth * 1.15, direction, progress: 0, triggered: false, released: false};
    stage.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  stage.addEventListener('pointermove', event => {
    if (!dragging || dragging.pointerId !== event.pointerId) return;
    const dx = event.clientX - dragging.startX, dy = event.clientY - dragging.startY;
    if (!dragging.triggered) {
      if (Math.abs(dy) > 12 && Math.abs(dy) > Math.abs(dx)) { releaseDrag(true); return; }
      if (Math.abs(dx) < 8) return;
      if (!dragging.direction) dragging.direction = dx < 0 ? 1 : -1;
      if (!canTurn(dragging.direction)) { releaseDrag(true); return; }
      if (-dragging.direction * dx <= 0) return;
      dragging.triggered = true;
      stage.classList.add('is-dragging');
      if (!reducedMotion.matches) { dragging.animated = true; void turn(dragging.direction, dragging); }
    }
    dragging.progress = Math.max(0, Math.min(1, -dragging.direction * dx / dragging.travel));
    if (!dragFrame) dragFrame = requestAnimationFrame(scrubDrag);
    event.preventDefault();
  });
  stage.addEventListener('pointerup', event => { if (dragging?.pointerId === event.pointerId) releaseDrag(); });
  stage.addEventListener('pointercancel', () => releaseDrag(true));
  stage.addEventListener('lostpointercapture', () => releaseDrag(true));
  stage.addEventListener('dragstart', event => event.preventDefault());
  window.addEventListener('blur', () => releaseDrag(true));
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && dragging) releaseDrag(true);
  });
  previous.addEventListener('click', () => void turn(-1));
  next.addEventListener('click', () => void turn(1));

  function layout() {
    const width = stage.clientWidth, height = stage.clientHeight;
    const density = window.devicePixelRatio || 1;
    const size = [width, height, density, count].join(':');
    if (!width || !height || size === lastSize) return;
    lastSize = size;
    ++version;
    reader.dataset.ready = 'false';
    stopRaster();
    queuedDirection = 0;
    releaseDrag(true);
    // Size changes settle the current animation first.
    if (active?.phase === 'flipping') current = active.settleTarget ?? active.target;
    active?.animations.forEach(animation => animation.cancel());
    stopRaster();
    abandon(active);
    active = null;
    // Two-page spreads only when they leave pages a useful size.
    orientation = width < 520 || width / height < ratio * 1.2 ? 'portrait' : 'landscape';
    if (orientation === 'landscape' && current > 0 && current % 2 === 0) current--;
    const columns = orientation === 'landscape' ? 2 : 1;
    const gutterX = width < 600 ? 12 : 32;
    const gutterY = height < 480 ? 12 : 28;
    const fit = Math.min((width - gutterX * 2) / columns, (height - gutterY * 2) * ratio);
    // Page sizes and positions land on whole device pixels (an even page width
    // keeps the half-page cover offset there too), so settled text stays crisp.
    const deviceWidth = Math.max(2, Math.floor(fit * density / 2) * 2);
    const deviceHeight = Math.max(1, Math.round(deviceWidth / ratio));
    pageWidth = deviceWidth / density;
    pageHeight = deviceHeight / density;
    // Rasters are pageWidth × min(devicePixelRatio, 2). Every canvas stays within
    // MAX_CANVAS_SIDE, including the turning sheet, which spans both pages.
    const scale = Math.min(density, 2) / density;
    const cap = Math.min(1, MAX_CANVAS_SIDE / (deviceWidth * scale * columns), MAX_CANVAS_SIDE / (deviceHeight * scale));
    rasterWidth = Math.max(1, Math.round(deviceWidth * scale * cap));
    rasterHeight = Math.max(1, Math.round(deviceHeight * scale * cap));
    rasters.setSize(rasterWidth, rasterHeight);
    const bookWidth = pageWidth * columns;
    const origin = stage.getBoundingClientRect();
    const snap = (start, value) => Math.round((start + value) * density) / density - start;
    book.style.width = bookWidth + 'px';
    book.style.height = pageHeight + 'px';
    book.style.left = snap(origin.left, (width - bookWidth) / 2) + 'px';
    book.style.top = snap(origin.top, (height - pageHeight) / 2) + 'px';
    book.style.setProperty('--leaf-width', pageWidth + 'px');
    commit(current, spread(current));
    reader.dataset.ready = 'true';
  }

  // Display density changes (moving between screens) do not resize the stage.
  function watchDensity() {
    const query = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    query.addEventListener('change', () => { layout(); watchDensity(); }, {once: true});
  }
  watchDensity();

  // Jumps without animation, e.g. when read mode closes.
  function goTo(index) {
    let target = clampIndex(index);
    if (orientation === 'landscape' && target > 0 && target % 2 === 0) target--;
    if (!pageWidth) { current = target; return; }
    if (target === current && !active && !dragging) return;
    ++version;
    stopRaster();
    queuedDirection = 0;
    releaseDrag(true);
    active?.animations.forEach(animation => animation.cancel());
    abandon(active);
    active = null;
    commit(target, spread(target));
  }

  // A short message in the counter's place, e.g. while the paper loads.
  function setStatus(text) {
    status = text;
    writeCounter();
  }

  // Called once the PDF is open; its page count is authoritative.
  function enable(total) {
    enabled = true;
    status = '';
    if (total > 0 && total !== count) {
      count = total;
      current = clampIndex(current);
      lastSize = '';
      layout();
    } else update();
  }

  return {
    layout,
    goTo,
    enable,
    setStatus,
    visiblePages: () => visiblePages(),
    get current() { return current; },
  };
}

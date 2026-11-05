#!/usr/bin/env node
/* Step 3 of 3: renders previews of every GLB in books/ with three.js in
   headless Chrome (SwiftShader WebGL), and runs the Khronos glTF Validator
   on each file in the same page.
     node tools/books/preview.mjs            every GLB, plus the scene sheets
     node tools/books/preview.mjs lectern    only names containing "lectern"
   Writes to books/previews/:
     <name>-front.png, <name>-angle.png   studio views of every GLB
     <id>-on-lectern-visitor.png          what a standing visitor sees, 1:1:
                                          eye 1.6 m high, 1.5 m from the book,
                                          45.8 degree vertical FOV at
                                          1920x1080 (about 24 px per degree),
                                          no real-time shadows
     visitor-2x.png, visitor-dim-2x.png   the six books from that eye as 2x
                                          nearest-neighbour crops, in studio
                                          light and in dim warm light
     shelf.png                            the six books standing spine-out at
                                          1.5 m, 1:1 and 2x
   Exits non-zero if a file fails validation or a view does not render. */
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { BOOKS, BUILD, PREVIEWS, PROFILE, chrome, chromeArgs, bodyAttrs, loadMuseum, pngSize } from './lib.mjs';

const THREE = 'https://cdn.jsdelivr.net/npm/three@0.160.0';
const VALIDATOR = 'https://cdn.jsdelivr.net/npm/gltf-validator@2.0.0-dev.3.10/+esm';
const STUDIO_SIZE = [1200, 900];
const VISITOR = { eye: 1.6, dist: 1.5, vfov: 45.8, size: [1920, 1080] };

/* Camera directions (from target toward camera) and framing per kind. */
const VIEWS = {
  book: {
    front: { dir: [0, 1, 0.0001], up: [0, 0, -1], fov: 26, fill: 1.0 },
    angle: { dir: [-0.62, 0.62, 0.48], up: [0, 1, 0], fov: 26, fill: 0.92 },
  },
  lectern: {
    front: { dir: [0, 0.32, 1], up: [0, 1, 0], fov: 30, fill: 1.0 },
    angle: { dir: [-0.62, 0.42, 0.66], up: [0, 1, 0], fov: 30, fill: 0.98 },
  },
  onLectern: {
    // close-up of the cover from above the ledge
    front: { dir: [0, 0.82, 0.57], up: [0, 1, 0], fov: 30, fill: 0.92, target: 'book' },
    angle: { dir: [-0.6, 0.45, 0.66], up: [0, 1, 0], fov: 30, fill: 0.98 },
    visitor: { camera: 'visitor', size: VISITOR.size, shadows: false, floor: '#5b4d41', target: 'book' },
  },
};

function kindOf(name) {
  if (name === 'lectern') return 'lectern';
  if (name.endsWith('-on-lectern')) return 'onLectern';
  return 'book';
}

/* Shared page parts: import map, lights, model loading. */
const HEAD = `<script type="importmap">{"imports":{"three":"${THREE}/build/three.module.js","three/addons/":"${THREE}/examples/jsm/"}}</script>`;
const LIB = `
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
const body = document.body;
const fail = (e) => { body.dataset.error = String(e && e.message || e); const p = document.getElementById('err'); if (p) p.textContent = body.dataset.error; };
window.addEventListener('error', (e) => fail(e.error || e.message));
// Decode textures through <img> loads: headless virtual time waits for
// those, but not for createImageBitmap, which can end the page too early.
window.createImageBitmap = undefined;
const LIGHTS = {
  studio: { bg: '#7f8287', env: 0.3, key: [0xfff4e8, 1.25], fill: [0xe8f0ff, 0.5], hemi: [0xffffff, 0x6b6e73, 0.2] },
  visitor: { bg: '#2b2522', env: 0.3, key: [0xfff4e8, 1.25], fill: [0xe8f0ff, 0.5], hemi: [0xffffff, 0x6b6e73, 0.2] },
  dim: { bg: '#17130f', env: 0.1, key: [0xffc690, 0.6], fill: [0x9a8470, 0.1], hemi: [0xffe0b8, 0x2a2018, 0.14] },
};
function makeRenderer(w, h, shadows) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.shadowMap.enabled = !!shadows;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  return renderer;
}
function lightScene(scene, renderer, mode, centre, span, shadows) {
  const L = LIGHTS[mode];
  scene.background = new THREE.Color(L.bg);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(renderer), 0.04).texture;
  const key = new THREE.DirectionalLight(L.key[0], L.key[1]);
  key.position.copy(centre).add(new THREE.Vector3(-0.5, 1.4, 0.9).multiplyScalar(span * 2.5));
  key.target.position.copy(centre);
  if (shadows) {
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.radius = 6;
    key.shadow.bias = -0.0004;
    key.shadow.normalBias = 0.002;
    const sc = key.shadow.camera;
    sc.left = sc.bottom = -span * 1.4; sc.right = sc.top = span * 1.4; sc.near = 0.01; sc.far = span * 10;
  }
  scene.add(key, key.target);
  const fill = new THREE.DirectionalLight(L.fill[0], L.fill[1]);
  fill.position.copy(centre).add(new THREE.Vector3(1.2, 0.6, 0.6).multiplyScalar(span * 2.5));
  scene.add(fill);
  scene.add(new THREE.HemisphereLight(L.hemi[0], L.hemi[1], L.hemi[2]));
  return L;
}
async function loadModel(b64, envIntensity) {
  const bin = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const gltf = await new Promise((res, rej) => new GLTFLoader().parse(bin.buffer, '', res, rej));
  gltf.scene.traverse((o) => {
    if (o.isMesh) {
      // shadow decals are blended: they must not cast shadows themselves
      o.castShadow = !o.material.transparent;
      o.receiveShadow = !o.material.transparent;
      if ('envMapIntensity' in o.material) o.material.envMapIntensity = envIntensity;
    }
  });
  return { gltf, bin };
}
function visitorCamera(box, w, h) {
  const c = box.getCenter(new THREE.Vector3());
  const dy = ${VISITOR.eye} - c.y;
  const dz = Math.sqrt(Math.max(0.01, ${VISITOR.dist} ** 2 - dy * dy));
  const cam = new THREE.PerspectiveCamera(${VISITOR.vfov}, w / h, 0.01, 50);
  cam.position.set(c.x, ${VISITOR.eye}, c.z + dz);
  cam.lookAt(c);
  cam.updateProjectionMatrix();
  return cam;
}
function screenBox(box, cam, w, h, pad) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) {
    const p = new THREE.Vector3(x, y, z).project(cam);
    const px = (p.x + 1) / 2 * w, py = (1 - p.y) / 2 * h;
    x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
  }
  x0 = Math.max(0, Math.floor(x0 - pad)); y0 = Math.max(0, Math.floor(y0 - pad));
  x1 = Math.min(w, Math.ceil(x1 + pad)); y1 = Math.min(h, Math.ceil(y1 + pad));
  return { x0, y0, w: x1 - x0, h: y1 - y0 };
}
function floorPlane(color, y) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color, roughness: 0.92, metalness: 0 }));
  m.rotation.x = -Math.PI / 2;
  m.position.y = y;
  return m;
}
`;

/* One GLB, one view. */
function page({ b64, view, validate, bookName }) {
  const [W, H] = view.size || STUDIO_SIZE;
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>preview</title>
<style>html,body{margin:0;padding:0;background:#7f8287;overflow:hidden}canvas{display:block}
#err{position:absolute;left:12px;top:12px;font:16px monospace;color:#fff;white-space:pre-wrap}</style>
${HEAD}
</head><body><div id="err"></div>
<script type="module">
${LIB}
try {
  const VIEW = ${JSON.stringify(view)};
  const W = ${W}, H = ${H};
  const { gltf, bin } = await loadModel(${JSON.stringify(b64)}, 0.3);
  if (${validate}) {
    try {
      const { validateBytes, version } = await import('${VALIDATOR}');
      const r = await validateBytes(bin, { maxIssues: 50 });
      const s = r.issues;
      body.dataset.validator = version();
      body.dataset.vErrors = s.numErrors; body.dataset.vWarnings = s.numWarnings;
      body.dataset.vInfos = s.numInfos; body.dataset.vHints = s.numHints;
      body.dataset.vMessages = JSON.stringify(s.messages.map((m) => [m.severity, m.code, m.pointer || '', m.message]).slice(0, 20));
      body.dataset.vTriangles = r.info && r.info.totalTriangleCount;
      body.dataset.vVertices = r.info && r.info.totalVertexCount;
      body.dataset.vDrawCalls = r.info && r.info.drawCallCount;
    } catch (e) { body.dataset.validatorError = String(e && e.message || e); }
  }
  const shadows = VIEW.shadows !== false;
  const renderer = makeRenderer(W, H, shadows);
  document.body.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const model = gltf.scene;
  scene.add(model);
  model.updateMatrixWorld(true);
  // frame the visible model, not the flat shadow decals around it
  const box = new THREE.Box3();
  model.traverse((o) => { if (o.isMesh && !o.material.transparent) box.expandByObject(o); });
  let fitBox = box;
  if (VIEW.target === 'book') {
    const book = model.getObjectByName(${JSON.stringify(bookName)});
    if (book) fitBox = new THREE.Box3().setFromObject(book);
  }
  const size = box.getSize(new THREE.Vector3());
  const fitSize = fitBox.getSize(new THREE.Vector3());
  const centre = fitBox.getCenter(new THREE.Vector3());
  body.dataset.bbox = JSON.stringify({ min: box.min.toArray().map((v) => +v.toFixed(4)), max: box.max.toArray().map((v) => +v.toFixed(4)) });
  const span = Math.max(size.x, size.y, size.z);
  const L = lightScene(scene, renderer, VIEW.camera === 'visitor' ? 'visitor' : 'studio', box.getCenter(new THREE.Vector3()), span, shadows);
  document.body.style.background = L.bg;
  if (VIEW.floor && VIEW.floor !== 'shadow') {
    scene.add(floorPlane(VIEW.floor, box.min.y - 0.0001));
  } else {
    // studio floor that only shows shadows, on a mid-grey backdrop
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.ShadowMaterial({ opacity: 0.32 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = box.min.y - 0.0002;
    floor.receiveShadow = true;
    scene.add(floor);
  }
  let camera;
  if (VIEW.camera === 'visitor') {
    camera = visitorCamera(fitBox, W, H);
  } else {
    camera = new THREE.PerspectiveCamera(VIEW.fov, W / H, 0.005, 100);
    const dir = new THREE.Vector3(...VIEW.dir).normalize();
    camera.up.set(...VIEW.up);
    // fit: bounding sphere of the framed part
    const radius = fitSize.length() / 2;
    const vfov = THREE.MathUtils.degToRad(VIEW.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * W / H);
    const dist = radius / Math.sin(Math.min(vfov, hfov) / 2) * VIEW.fill;
    camera.position.copy(centre).addScaledVector(dir, dist);
    camera.lookAt(centre);
    camera.updateProjectionMatrix();
  }
  renderer.render(scene, camera);
  renderer.render(scene, camera);
  body.dataset.ready = '1';
} catch (e) { fail(e); }
</script></body></html>`;
}

/* Several GLBs composed on one sheet.
   mode 'visitor': each on-lectern model from the visitor's eye, cropped to
   its book and shown at 2x (nearest neighbour), in a 3-column grid.
   mode 'shelf': the standalone books standing spine-out on a shelf at
   1.5 m, shown 1:1 and at 2x. */
function scenePage({ items, mode, light, cell }) {
  return `<!doctype html>
<html><head><meta charset="utf-8"><title>sheet</title>
<style>html,body{margin:0;padding:0;background:#1d1b1a;overflow:hidden}canvas{display:block}</style>
${HEAD}
</head><body>
<script>window.__ITEMS = ${JSON.stringify(items).replace(/</g, '\\u003c')};</script>
<script type="module">
${LIB}
try {
  const ITEMS = window.__ITEMS;
  const [FW, FH] = ${JSON.stringify(VISITOR.size)};
  const renderer = makeRenderer(FW, FH, false);
  const scene = new THREE.Scene();
  const L = lightScene(scene, renderer, ${JSON.stringify(light)}, new THREE.Vector3(0, 0.9, 0), 1.1, false);
  const models = [];
  for (const it of ITEMS) models.push({ id: it.id, root: (await loadModel(it.b64, L.env)).gltf.scene });
  const out = document.createElement('canvas');
  const ctx = out.getContext('2d');
  const label = (t, x, y) => { ctx.fillStyle = '#d8d2c8'; ctx.font = '14px monospace'; ctx.fillText(t, x, y); };
  const [cw, ch] = ${JSON.stringify(cell)};
  const blit = (r, dx, dy, k) => { ctx.imageSmoothingEnabled = false; ctx.drawImage(renderer.domElement, r.x0, r.y0, r.w, r.h, dx, dy, r.w * k, r.h * k); };
  if (${JSON.stringify(mode)} === 'visitor') {
    out.width = cw * 3; out.height = ch * Math.ceil(models.length / 3);
    ctx.fillStyle = L.bg; ctx.fillRect(0, 0, out.width, out.height);
    const floor = floorPlane('#5b4d41', -0.0001);
    scene.add(floor);
    models.forEach((m, i) => {
      scene.add(m.root); m.root.updateMatrixWorld(true);
      const book = m.root.getObjectByName(m.id);
      const box = new THREE.Box3().setFromObject(book);
      const cam = visitorCamera(box, FW, FH);
      renderer.render(scene, cam);
      const r = screenBox(box, cam, FW, FH, 8);
      const gx = (i % 3) * cw, gy = Math.floor(i / 3) * ch;
      blit(r, gx + Math.max(0, (cw - r.w * 2) / 2), gy + 24, 2);
      label(m.id + '  ' + r.w + 'x' + r.h + ' px at 1:1, shown 2x', gx + 8, gy + 16);
      scene.remove(m.root);
    });
  } else {
    // stand each book up: head (-Z) up, spine (-X) toward the camera,
    // front cover (+Y) to the right, as on a shelf
    const shelf = new THREE.Group();
    let x = 0;
    for (const m of models) {
      const b0 = new THREE.Box3().setFromObject(m.root);
      const T = b0.max.y - b0.min.y, Hb = b0.max.z - b0.min.z;
      const g = new THREE.Group();
      g.add(m.root);
      g.matrixAutoUpdate = false;
      g.matrix.makeBasis(new THREE.Vector3(0, 0, -1), new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, -1, 0));
      g.matrix.setPosition(x, 1.0 + Hb / 2, 0);
      x += T + 0.002;
      shelf.add(g);
    }
    shelf.position.x = -x / 2;
    scene.add(shelf);
    const board = new THREE.Mesh(new THREE.BoxGeometry(x + 0.12, 0.022, 0.3), new THREE.MeshStandardMaterial({ color: '#3a2618', roughness: 0.7 }));
    board.position.set(0, 1.0 - 0.011, -0.04);
    const back = new THREE.Mesh(new THREE.BoxGeometry(x + 0.12, 0.4, 0.01), new THREE.MeshStandardMaterial({ color: '#2a1c13', roughness: 0.8 }));
    back.position.set(0, 1.18, -0.2);
    scene.add(board, back);
    scene.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(shelf);
    const c = box.getCenter(new THREE.Vector3());
    const cam = new THREE.PerspectiveCamera(${VISITOR.vfov}, FW / FH, 0.01, 50);
    cam.position.set(c.x, ${VISITOR.eye}, c.z + Math.sqrt(${VISITOR.dist} ** 2 - (${VISITOR.eye} - c.y) ** 2));
    cam.lookAt(c);
    cam.updateProjectionMatrix();
    renderer.render(scene, cam);
    const r = screenBox(box, cam, FW, FH, 14);
    out.width = cw; out.height = ch;
    ctx.fillStyle = L.bg; ctx.fillRect(0, 0, out.width, out.height);
    blit(r, 12, 30, 1);
    blit(r, 12 + r.w + 30, 30, 2);
    label('shelf at 1.5 m, eye 1.6 m: 1:1 (' + r.w + 'x' + r.h + ' px) and 2x', 12, 18);
  }
  document.body.appendChild(out);
  body.dataset.w = out.width; body.dataset.h = out.height;
  body.dataset.ready = '1';
} catch (e) { fail(e); document.body.textContent = 'ERROR ' + document.body.dataset.error; }
</script></body></html>`;
}

async function retry(fn) {
  let r;
  for (let attempt = 0; attempt < 3; attempt++) {
    r = await fn();
    if (!r.problems.length) break;
  }
  return r;
}

async function shootOnce(name, kind, viewName, validate) {
  const glb = fs.readFileSync(path.join(BOOKS, `${name}.glb`));
  const view = VIEWS[kind][viewName];
  const [W, H] = view.size || STUDIO_SIZE;
  const html = page({ b64: glb.toString('base64'), view, validate, bookName: name.replace(/-on-lectern$/, '') });
  const file = path.join(BUILD, 'previews', `${name}-${viewName}.html`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, html);
  const shot = path.join(PREVIEWS, `${name}-${viewName}.png`);
  fs.rmSync(shot, { force: true });
  const { out } = await chrome([...chromeArgs({ w: W, h: H, shot, webgl: true, budget: 20000 }), pathToFileURL(file).href], 180_000);
  const a = bodyAttrs(out);
  const problems = [];
  if (a['data-ready'] !== '1') problems.push(`${viewName} did not render`);
  if (a['data-error']) problems.push(a['data-error']);
  if (!fs.existsSync(shot)) problems.push(`no ${viewName} screenshot`);
  else {
    const [w, h] = pngSize(fs.readFileSync(shot));
    if (w !== W || h !== H) problems.push(`${viewName} screenshot ${w}x${h}`);
  }
  return { a, problems, bytes: glb.length };
}

/* Renders a sheet at its own size: the page reports the size it needs. */
async function sheetOnce(name, opts) {
  const html = scenePage(opts);
  const file = path.join(BUILD, 'previews', `${name}.html`);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, html);
  const shot = path.join(PREVIEWS, `${name}.png`);
  fs.rmSync(shot, { force: true });
  const rows = opts.mode === 'visitor' ? Math.ceil(opts.items.length / 3) : 1;
  const W = opts.mode === 'visitor' ? opts.cell[0] * 3 : opts.cell[0];
  const H = opts.cell[1] * rows;
  const { out } = await chrome([...chromeArgs({ w: W, h: H, shot, webgl: true, budget: 60000 }), pathToFileURL(file).href], 300_000);
  const a = bodyAttrs(out);
  const problems = [];
  if (a['data-ready'] !== '1') problems.push('did not render');
  if (a['data-error']) problems.push(a['data-error']);
  if (!fs.existsSync(shot)) problems.push('no screenshot');
  return { problems };
}

async function main() {
  const filters = process.argv.slice(2);
  fs.mkdirSync(PREVIEWS, { recursive: true });
  fs.mkdirSync(PROFILE, { recursive: true });
  const all = fs.readdirSync(BOOKS).filter((f) => f.endsWith('.glb')).map((f) => f.slice(0, -4)).sort();
  const names = all.filter((n) => !filters.length || filters.some((f) => n.includes(f)));
  const report = [];
  let bad = 0;
  for (const name of names) {
    const kind = kindOf(name);
    const problems = [];
    let a = {};
    let bytes = 0;
    for (const viewName of Object.keys(VIEWS[kind])) {
      const r = await retry(() => shootOnce(name, kind, viewName, viewName === 'front'));
      problems.push(...r.problems);
      if (viewName === 'front') { a = r.a; bytes = r.bytes; }
    }
    if (a['data-validator-error']) problems.push('validator: ' + a['data-validator-error']);
    else if (a['data-v-errors'] == null) problems.push('validator did not run');
    else if (+a['data-v-errors'] > 0) problems.push(`${a['data-v-errors']} validation errors`);
    const row = {
      name, bytes, triangles: +a['data-v-triangles'] || null, vertices: +a['data-v-vertices'] || null,
      drawCalls: +a['data-v-draw-calls'] || null,
      errors: +a['data-v-errors'], warnings: +a['data-v-warnings'], infos: +a['data-v-infos'], hints: +a['data-v-hints'],
      messages: a['data-v-messages'] ? JSON.parse(a['data-v-messages']) : [], bbox: a['data-bbox'] ? JSON.parse(a['data-bbox']) : null,
      validator: a['data-validator'], problems,
    };
    report.push(row);
    if (problems.length) bad++;
    console.log(`${problems.length ? 'FAIL' : 'ok  '} ${name.padEnd(30)} ${String(row.bytes).padStart(8)} B  tris ${String(row.triangles).padStart(5)}  draws ${row.drawCalls}  E${row.errors} W${row.warnings} I${row.infos} H${row.hints}  ${problems.join('; ')}`);
    for (const m of row.messages) console.log('       ', m.join(' | '));
  }

  // scene sheets, when every book is there and no filter (or "sheet") is given
  const books = all.filter((n) => kindOf(n) === 'book');
  const lect = books.filter((n) => all.includes(`${n}-on-lectern`));
  if ((!filters.length || filters.includes('sheet')) && books.length) {
    const b64 = (n) => fs.readFileSync(path.join(BOOKS, `${n}.glb`)).toString('base64');
    const order = orderOf(books);
    const sheets = [
      ['visitor-2x', { mode: 'visitor', light: 'visitor', cell: [460, 440], items: order.filter((n) => lect.includes(n)).map((n) => ({ id: n, b64: b64(`${n}-on-lectern`) })) }],
      ['visitor-dim-2x', { mode: 'visitor', light: 'dim', cell: [460, 440], items: order.filter((n) => lect.includes(n)).map((n) => ({ id: n, b64: b64(`${n}-on-lectern`) })) }],
      ['shelf', { mode: 'shelf', light: 'visitor', cell: [900, 700], items: order.map((n) => ({ id: n, b64: b64(n) })) }],
    ];
    for (const [name, opts] of sheets) {
      const r = await retry(() => sheetOnce(name, opts));
      if (r.problems.length) bad++;
      console.log(`${r.problems.length ? 'FAIL' : 'ok  '} sheet ${name}  ${r.problems.join('; ')}`);
    }
  }
  fs.writeFileSync(path.join(BUILD, 'validation.json'), JSON.stringify(report, null, 2));
  console.log(bad ? `\n${bad} problems.` : `\nAll ${names.length} files valid; previews in ${PREVIEWS}`);
  process.exitCode = bad ? 1 : 0;
}

/* Papers in data/papers.js order (falls back to name order). */
function orderOf(books) {
  try {
    const ids = loadMuseum().papers.map((p) => p.id);
    const pos = (id) => (ids.includes(id) ? ids.indexOf(id) : 1e9);
    return [...books].sort((a, b) => pos(a) - pos(b));
  } catch { return books; }
}

export { VIEWS, page, scenePage };
if (import.meta.main) main().catch((e) => { console.error(e); process.exitCode = 1; });
